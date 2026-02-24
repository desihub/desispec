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

from astropy.table import vstack

from desiutil.log import get_logger

from desispec.io.util import decode_camword, difference_camwords
from desispec.io import findfile
from desispec.workflow.tableio import load_table
from desispec.scripts.compute_dark import get_stacked_dark_exposure_table


def preproc_darks_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Computes preprocs for dark exposures used for dark creation",
                                     epilog='''
                                     Input is a list of dark expids. The raw images are preprocessed
                                     without dark,mask correction. However gains are applied so the output is in electrons/sec.
                                     --expids and --camword are required, --nights are optional but improve I/O efficiency.
                                     ''')

    parser.add_argument('-e','--expids', type=str, default=None, required=True,
                        help = 'exposures to process, can be a comma separated list of expids or a single expid')
    parser.add_argument('-n', '--nights', type=str, default = None, required=False,
                        help='Comma separated list of YEARMMDD nights where we find the darks to run through preproc')
    parser.add_argument('-c','--camword', type=str, required = True,
                        help = 'cameras to process, e.g. a0123456789')
    parser.add_argument('--bias', type = str, default = None, required=False,
                         help = 'specify a bias image calibration file (standard preprocessing calibration is turned off)')
    parser.add_argument('--nocosmic', action = 'store_true',
                        help = 'do not perform cosmic ray subtraction (much slower, but more accurate because median can leave traces)')
    parser.add_argument('--specprod', type=str, default=None, required=False,
                        help='Specify specprod containing the nightly bias files and the exposure tables. Default is $SPECPROD if it is defined, otherwise will use the bias in DESI_SPECTRO_CALIB and identify exposures from DESI_SPECTRO_DATA.')
    parser.add_argument('--preproc-dark-dir', type=str, default=None, required=False,
                        help='Specify alternate specprod where we will save the preprocessed dark frame images are saved. Default is same input specprod. Resulting exposures will be save under <preproc_dark_dir>/dark_preproc/<NIGHT>/<EXPID>')
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
    if args.nights is not None and args.expids is not None and rank == 0:
        log.info(f"Assuming all exposures in {args.expids} can be found in nights={args.nights}.")

    args.expids = np.array(args.expids.split(',')).astype(int)
    if args.nights is not None:
        args.nights = np.array(args.nights.split(',')).astype(int)

    ## get the requested cameras from the camword
    requested_cameras = set(decode_camword(args.camword))
    if len(requested_cameras) == 0:
        if rank == 0:
            log.error(f'No cameras found in camword {args.camword}.')
        return 1

    # first find the exposures if they are not given in input
    if rank == 0:
        ## Use the compute_dark_night parser to get the exposure table
        ## so that we have consistent exposure selection via default values
        ## for number of nights nbefore and after the reference night
        if args.nights is None:
            ## camera and outfile are required, so give dummy values for those
            opts = ['--expids', ' '.join(args.expids.astype(str)), '-o', 'temp', '-c', 'b1',
                    '--skip-camera-check', '--dont-search-filesystem']
            compdark_parser = compute_dark_parser()
            compdark_args = compdark_parser.parse_args(opts)
            exptable = get_stacked_dark_exposure_table(compdark_args)
        else:
            exptables = []
            for night in np.unique(args.nights):
                tabname = findfile('exposure_table', night=night, readonly=True)
                exptables.append(load_table(tablename=tabname, tabletype='exposure_table', suppress_logging=True))
            exptable = vstack(exptables)
            if not np.all(np.isin(args.expids, exptable['EXPID'].data)):
                log.error(f"Not all expids in {args.expids} found on nights {args.nights}")
            exptable = exptable[np.isin(exptable['EXPID'].data, args.expids)]
        # assemble corresponding images
        expids, camlists, files, nights = [], [], [], []
        if args.expids is not None:
            exptable = exptable[np.isin(exptable["EXPID"], args.expids)]
        for row in exptable:
            filename = findfile("raw",night=row["NIGHT"],expid=row["EXPID"], readonly=True)
            if os.path.exists(filename):
                goodcams = set(decode_camword(difference_camwords(row["CAMWORD"], row["BADCAMWORD"])))
                camlist = list(goodcams.intersection(requested_cameras))
                if len(camlist) > 0:
                    camlists.append(camlist)
                    expids.append(row["EXPID"])
                    nights.append(row["NIGHT"])
                    files.append(filename)
                else:
                    log.warning(f'No requested cameras found in {filename} for expid {row["EXPID"]}')
            else:
                log.error(f'Skipping missing file {filename}')
        data = (expids, nights, camlists, files)
    else:
        data = None

    # Broadcast data to all ranks
    if comm is not None:
        data = comm.bcast(data, root=0)

    expids, nights, camlists, files = data

    if len(expids) == 0:
        if rank == 0:
            log.error("No valid exposures found for dark frame computation.")
        return 1

    if args.dry_run:
        image_str = ' '.join(files)
        if rank == 0:
            log.info(f'Input images: {image_str}')
            log.info('--dry-run mode, exiting before running preproc_darks')
        return 0

    if args.bias is None:
        thisbias = True
    else:
        thisbias = args.bias

    ## Number of task is the total number of cameras to run preproc on plus one additional
    ## task for each expid to handle the I/O of the files
    lens = [len(cams)+1 for cams in camlists]
    maxlens = np.max(lens)
    # Files are only ~0.25Gb each, so shouldn't need to limit blocks.
    optimal_nblocks = int(np.ceil(size / maxlens)) #min(int(np.ceil(size / maxlens)), 200) # 50Gb limit

    ## Split into subcommunicators if we have enough ranks and enough work to do
    if comm is not None:
        nblocks, block_size, block_rank, block_num, block_comm = \
            distribute_ranks_to_blocks(nblocks=optimal_nblocks, rank=rank, size=size, comm=comm,
                                log=log, split_comm=True)
    else:
        nblocks, block_size, block_rank, block_num, block_comm = 1, 1, 0, 0, None

    ## looping over all the exposures, each communicator gets a subset
    for expid, night, camlist, filename in zip(expids[block_num::nblocks], nights[block_num::nblocks],
                                               camlists[block_num::nblocks], files[block_num::nblocks]):
        ## Each subcommunicator has enough ranks for the largest camera list
        ## give each rank a camera, skipping rank 0 which will handle the I/O and MPI
        ## and pad the remaining ranks with None so that scatter works
        if block_rank == 0:
            try:
                primary_header = read_raw_primary_header(filename)
                # Convert to a plain Header since CompImageHeader can't be pickled
                primary_header = fits.Header(primary_header)
            except Exception as e:
                log.error(f'Failed to read primary header from {filename}: {e}')
                primary_header = None

            with fits.open(filename, memmap=False) as fx:
                hducams = [str(hdu.name).lower().strip() for hdu in fx]
                final_camlist = sorted(list(requested_cameras.intersection(set(camlist))))
                ncam = len(final_camlist)

                indices = [ list(range(i, ncam, block_size-1)) for i in range(block_size-1) ]
                all_data_header_cams = []
                for camera in final_camlist:
                    rawimage = fx[camera.upper()].data
                    # Convert to a plain Header since CompImageHeader can't be pickled
                    header = fits.Header(fx[camera.upper()].header)
                    all_data_header_cams.append((rawimage, header, camera))

            broadcast_bundle = [None]
            for inds in indices:
                broadcast_bundle.append([all_data_header_cams[i] for i in inds])
            if len(broadcast_bundle) < block_size:
                broadcast_bundle += [None] * (block_size - len(broadcast_bundle))
            assert len(broadcast_bundle) == block_size, f"broadcast_bundle length {len(broadcast_bundle)} does not match block_size {block_size}"
        else:
            data_header_cams = None
            primary_header = None
            ## For non-root this isn't needed but we need to have the same variable defined for the mpi scatter
            broadcast_bundle = None

        ## broadcast the primary header
        if block_comm is not None:
            primary_header = block_comm.bcast(primary_header, root=0)
        if primary_header is None:
            log.error(f'No primary header in {filename} for expid {expid} for rank {rank} block_rank {block_rank} block_num {block_num}')
            continue

        ## scatter the work to the ranks
        if block_comm is not None:
            data_header_cams = block_comm.scatter(broadcast_bundle, root=0)
        else:
            data_header_cams = all_data_header_cams

        if data_header_cams is None:
            log.info(f'No data_header_cams for rank {rank} block_rank {block_rank} block_num {block_num}')
            continue

        ## loop over the work each rank was assigned. If not enough ranks, they will have multiple cameras
        for data_header_cam in data_header_cams:
            if data_header_cam is not None:
                rawimage, header, camera = data_header_cam
            else:
                continue

            if args.preproc_dark_dir is not None:
                preproc_filename = findfile("preproc_for_dark",night=night,expid=expid,camera=camera,specprod_dir=args.preproc_dark_dir)
            else:
                preproc_filename = findfile("preproc_for_dark",night=night,expid=expid,camera=camera)

            if os.path.exists(preproc_filename):
                log.info(f"Rank {rank} block_rank {block_rank} block_num {block_num}: "
                         + f"Preprocessed dark file {preproc_filename} already exists, skipping.")
                continue
            else:
                log.info(f'Rank {rank} block_rank {block_rank} block_num {block_num}: '
                         + f'Processing {filename} for camera {camera} (expid={expid})')
                img = process_raw(primary_header, rawimage, header, camera=camera, bias=thisbias, nocosmic=args.nocosmic,
                        mask=False, dark=False, pixflat=False, fallback_on_dark_not_found=True)

            # is saved in preproc_dark_dir if not None
            write_image(preproc_filename,img)
            log.info(f"Wrote {preproc_filename}")

    if comm is not None:
        comm.barrier()
    if rank == 0:
        log.info("All ranks have completed preproc_darks_mpi.")

    return 0



if __name__ == "__main__":
    main()
