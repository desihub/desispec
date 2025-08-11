#!/usr/bin/env python

import os
import argparse
from desispec.io.image import write_image
from desispec.io.raw import process_raw, read_raw_primary_header
from desispec.scripts.zproc import distribute_ranks_to_blocks
from desispec.scripts.compute_dark import compute_dark_parser
from desispec.workflow.desi_proc_funcs import assign_mpi
import numpy as np
from astropy.io import fits

import astropy.io.fits as pyfits
from astropy.table import Table,vstack

from desiutil.log import get_logger

from desispec.io.util import decode_camword, difference_camwords
from desispec.io import findfile
from desispec.workflow.tableio import load_table
from desispec.scripts.compute_dark import get_stacked_dark_exposure_table


def preproc_darks_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Compute a preprocs for dark exposures used for dark creation",
                                     epilog='''
                                     Input is a list of raw dark images, possibly with various exposure times.
                                     Raw images are preprocessed without dark,mask correction.
                                     However gains are applied so the output is in electrons/sec.
                                     ''')

    parser.add_argument('-e','--expids', type = str, default = None, required = False, nargs="*",
                        help = 'exposures to process, can be a list of expids or a single expid')
    parser.add_argument('-n', '--night', type=int, default = None, required=False,
                        help='YEARMMDD night defining the darks to run through preproc')
    parser.add_argument('-c','--camword',type = str, required = True,
                        help = 'header HDU (int or string)')
    parser.add_argument('--bias', type = str, default = None, required=False,
                         help = 'specify a bias image calibration file (standard preprocessing calibration is turned off)')
    parser.add_argument('--nocosmic', action = 'store_true',
                        help = 'do not perform comic ray subtraction (much slower, but more accurate because median can leave traces)')
    parser.add_argument('--specprod', type=str, default=None, required=False,
                        help='Specify specprod to use nightly bias files and the exposure tables. Default is $SPECPROD if it is defined, otherwise will use the bias in DESI_SPECTRO_CALIB.')
    parser.add_argument('--preproc-dark-dir', type=str, default=None, required=False,
                        help='Specify alternate specprod directory where preprocessed dark frame images are saved. Default is same input specprod')
    parser.add_argument('--dry-run', action='store_true', help="Print which images would be used, but don't compute dark")
    parser.add_argument('--mpi', action='store_true', help="Run in MPI mode, distributing work across multiple processes.")

    return parser


def parse(options=None):
    # parse the command line arguments
    parser = preproc_darks_parser()
    
    #- uses sys.argv if options=None
    args = parser.parse_args(options)

    return args


def main(args=None):

    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    log  = get_logger()

    comm, rank, size = assign_mpi(args.mpi, do_batch=False, log=log)

    if args.specprod is not None :
        os.environ["SPECPROD"] = args.specprod

    # check consistency of input options
    if args.night is not None and args.expids is not None :
        log.error("Cannot specify both --night and --expids arguments")
        return 1

    if args.expids is None and args.night is None:
        log.error("Need to specify input using --expids or --night")
        return 1

    ## get the requested cameras from the camword
    requested_cameras = set(decode_camword(args.camword))
    if len(requested_cameras) == 0:
        log.error(f'No cameras found in camword {args.camword}.')
        return 1

    # first find the exposures if they are not given in input
    if rank == 0:
        ## Use the compute_dark_night parser to get the exposure table
        ## so that we have consistent exposure selection via default values
        ## for number of nights nbefore and after the reference night
        if args.night is not None:
            opts = ['--nights', str(args.night)]
        elif args.expids is not None:
            opts = ['--expids', ' '.join(map(str, args.expids))]
        else:
            opts = []
        compdark_parser = compute_dark_parser()
        compdark_args = compdark_parser.parse_args(opts)
        exptable = get_stacked_dark_exposure_table(args, skip_camera_check=True)
        # exptablename = findfile("exposure_table", specprod=args.specprod, night=args.night)
        # exptable = load_table(tablename=exptablename, tabletype="exposure_table")
        # assemble corresponding images
        expids, camlists, files = [], [], []
        if args.expids is not None:
            exptable = exptable[np.isin(exptable["EXPID"], args.expids)]
        for row in exptable:
            filename = findfile("raw",night=row["NIGHT"],expid=row["EXPID"])
            if os.path.exists(filename):
                goodcams = set(decode_camword(difference_camwords(row["CAMWORD"], row["BADCAMWORD"])))
                camlist = list(goodcams.intersection(requested_cameras))
                if len(camlist) > 0:
                    camlists.append(list(goodcams.intersection(requested_cameras)))
                    expids.append(row["EXPID"])
                    files.append(filename)
                else:
                    log.warning(f'No requested cameras found in {filename} for expid {row["EXPID"]}')
            else:
                log.error(f'Skipping missing file {filename}')
        data = (expids, camlists, files)
    else:
        data = None

    # Broadcast data to all ranks
    data = comm.bcast(data, root=0)
    expids, camlists, files = data

    if len(expids) == 0:
        log.error("No valid exposures found for dark frame computation.")
        return 1

    if args.dry_run:
        image_str = ' '.join(files)
        log.info(f'Input images: {image_str}')
        log.info('--dry-run mode, exiting before running preproc_darks')
        return 0

    if args.bias is None:
        thisbias = True
    else:
        thisbias = args.bias

    ## Number of task is the total number of cameras to run preproc on plus one additional 
    ## task for each expid to handle the I/O of the files
    lens = [len(cams)+1 for cams in camlist]
    maxlens = np.max(lens)
    ntasks = np.sum(lens)
    # Files are only ~0.25Gb each, so shouldn't need to limit blocks. 
    optimal_nblocks = int(np.ceil(size / maxlens)) #min(int(np.ceil(size / maxlens)), 200) # 50Gb limit  

    ## Split into subcommunicators if we have enough ranks and enough work to do
    nblocks, block_size, block_rank, block_num, block_comm = \
        distribute_ranks_to_blocks(nblocks=optimal_nblocks, rank=rank, size=size, comm=comm,
                            log=log, split_comm=True)

    ## looping over all the exposures, each communicator gets a subset
    for expid, camlist, filename in zip(expids[block_num::nblocks], camlists[block_num::nblocks], files[block_num::nblocks]):
        ## Each subcommunicator has enough ranks for the largest camera list
        ## give each rank a camera, skipping rank 0 which will handle the I/O and MPI
        ## and pad the remaining ranks with None so that scatter works
        ncam = len(camlist)
        if block_rank == 0:
            try:
                primary_header = read_raw_primary_header(filename)
            except:
                primary_header = None
            fx = fits.open(filename, memmap=False)
            hducams = [str(hdu.name).lower().strip() for hdu in fx]
            final_camlist = sorted(list(requested_cameras.intersection(set(camlist))))

            indices = [ list(range(i, ncam, block_size-1)) for i in range(block_size-1) ]
            all_data_header_cams = []
            for camera in enumerate(final_camlist):
                rawimage = fx[camera.upper()].data
                header = fx[camera.upper()].header
                all_data_header_cams.append((rawimage, header, camera))

            broadcast_bundle = [None]
            for inds in indices:
                broadcast_bundle.append(all_data_header_cams[inds])
            if len(broadcast_bundle) < block_size:
                broadcast_bundle += [None] * (block_size - len(broadcast_bundle))
            assert len(broadcast_bundle) == block_size, f"broadcast_bundle length {len(broadcast_bundle)} does not match block_size {block_size}"
        else:
            data_header_cams = None
            primary_header = None

        ## broadcast the primary header 
        primary_header = block_comm.bcast(primary_header, root=0)
        if primary_header is None:
            log.error(f'No primary header in {filename} for expid {expid}')
            continue

        ## scatter the work to the ranks
        data_header_cams = comm.scatter(broadcast_bundle, root=0)
        if data_header_cams is None:
            log.info(f'No data_header_cams for rank {rank} block_rank {block_rank} block_num {block_num}')
            continue

        ## loop over the work each rank was assigned. If not enough ranks, they will have multiple cameras
        for data_header_cam in data_header_cams:
            if data_header_cam is not None:
                rawimage, header, camera = data_header_cam
            else:
                continue
                
            if args.preproc_dark_dir is not None :
                preproc_filename = findfile("preproc_for_dark",night=args.night,expid=expid,camera=camera,specprod_dir=args.preproc_dark_dir)
            else:
                preproc_filename = findfile("preproc_for_dark",night=args.night,expid=expid,camera=camera)
            
            if os.path.exists(preproc_filename):
                log.info(f"Preprocessed dark file {preproc_filename} already exists, skipping.")
                continue
            else:
                log.info(f'Rank {rank} block_rank {block_rank} block_num {block_num} processing {filename} for camera {camera} (expid={expid})')   
                img = process_raw(primary_header, rawimage, header, camera=camera, bias=thisbias, nocosmic=args.nocosmic,
                        mask=False, dark=False, pixflat=False, fallback_on_dark_not_found=True)

            # is saved in preproc_dark_dir if not None
            write_image(preproc_filename,img)
            log.info(f"Wrote {preproc_filename}")

    comm.barrier()
    if rank == 0:
        log.info("All ranks have completed preproc_darks_mpi.")
        
    return 0



if __name__ == "__main__":
    main()
