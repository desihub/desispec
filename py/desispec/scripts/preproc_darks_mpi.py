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
from desispec.io import findfile, replace_prefix
from desispec.util import header2night
from desispec.ccdcalib import dark_preproc_bias_matches, expected_dark_preproc_bias
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
    parser.add_argument('--reference-night', type=int, default = None, required=False,
                        help='YEARMMDD reference night defining the hardware state for this dark frame (default is most recent)')
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
    parser.add_argument('--allow-default-bias', action='store_true',
                        help='Do not require that the preprocessed darks use a matching bias, i.e. allow '
                        'preprocessing with the default bias in DESI_SPECTRO_CALIB when the nightly bias of a '
                        'night and camera is missing, and accept pre-existing preprocessed darks whatever bias '
                        'they used. Default is to preprocess nothing and exit with an error instead '
                        '(see desispec issue #2741).')
    parser.add_argument('--dry-run', action='store_true', help="Print which images would be used, but don't compute dark")
    parser.add_argument('--mpi', action='store_true', help="Run in MPI mode, distributing work across multiple processes.")

    return parser


def check_matching_biasnights(expids, nights, camlists, rawfiles, bias=True, preproc_dark_dir=None):
    """Check that every camera can be preprocessed with the bias it should use

    Args:
        expids (list of int): dark exposure ids to be preprocessed
        nights (list of int): YEARMMDD night of each exposure
        camlists (list of list of str): cameras to preprocess for each exposure
        rawfiles (list of str): raw data file of each exposure

    Options:
        bias (str or bool): bias that will be used, or True to require the nightly bias of each night and camera
        preproc_dark_dir (str): alternate specprod directory where the preprocessed darks are saved

    Returns:
        list of str: one message per error found, empty if every exposure agrees
        with its exposure table night and every camera either already has a
        preprocessed dark that used the matching bias or has that bias
        available to preprocess with

    The preprocessed darks feed the nightly darks, so they must strictly use
    the bias they are supposed to, normally the nightly bias of their own night
    and camera; preproc otherwise silently falls back to the default bias in
    $DESI_SPECTRO_CALIB (desispec issue #2741).  Pre-existing files are never
    overwritten, so a mismatched one is reported for a human to purge (e.g.
    with desi_purge_night).
    """
    log = get_logger()
    errors = []
    bad_cameras = set()
    ## the same bias is needed by every exposure of a night, so only stat it once
    bias_exists = dict()

    for expid, night, camlist, rawfile in zip(expids, nights, camlists, rawfiles):
        ## preproc looks up the bias with the night in the raw header while the
        ## preprocessed dark is written under the exposure table night, and
        ## compute_dark_file later looks for it under the raw header night, so a
        ## disagreement would write a file that the nightly dark never finds and
        ## that a rerun of this job would reject as using the wrong night's bias
        try:
            header_night = header2night(read_raw_primary_header(rawfile))
        except Exception as err:
            errors.append(f'Unable to read the night from {rawfile}: {err}')
            bad_cameras.update(camlist)
            continue

        if int(header_night) != int(night):
            errors.append(f'{rawfile} header NIGHT={header_night} disagrees with exposure '
                          + f'table NIGHT={night}, so its bias and its preprocessed dark '
                          + 'would come from different nights')
            bad_cameras.update(camlist)
            continue

        for camera in sorted(camlist):
            expected_bias = expected_dark_preproc_bias(bias, night, camera)

            if preproc_dark_dir is not None:
                preproc_filename = findfile("preproc_for_dark", night=night, expid=expid, camera=camera,
                                            specprod_dir=preproc_dark_dir, readonly=True)
            else:
                preproc_filename = findfile("preproc_for_dark", night=night, expid=expid, camera=camera,
                                            readonly=True)

            ## an existing preprocessed dark is all that matters for this camera,
            ## whether or not the bias it used is still on disk
            if os.path.exists(preproc_filename):
                is_match, biasused = dark_preproc_bias_matches(preproc_filename, expected_bias)
                if not is_match:
                    if biasused is None:
                        biasused = 'an unrecorded bias (unreadable or missing CCD_CALIB_BIAS)'
                    errors.append(f'Existing {preproc_filename} was preprocessed with {biasused} '
                                  + f'instead of {os.path.basename(expected_bias)}; purge it before rerunning')
                    bad_cameras.add(camera)
                continue

            if expected_bias not in bias_exists:
                bias_exists[expected_bias] = os.path.exists(expected_bias)

            if not bias_exists[expected_bias]:
                msg = f'Missing {os.path.basename(expected_bias)} needed to preprocess {night} {expid} {camera}'
                if os.path.basename(expected_bias).startswith('biasnight-'):
                    testbias = replace_prefix(expected_bias, 'biasnight', 'biasnighttest')
                    if os.path.exists(testbias):
                        msg += f'; {os.path.basename(testbias)} exists, so the nightly bias was rejected'
                errors.append(msg)
                bad_cameras.add(camera)

    for msg in errors:
        log.error(msg)

    if len(errors) > 0:
        log.error(f'{len(errors)} problem(s) affecting camera(s) {",".join(sorted(bad_cameras))}')

    return errors


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

    ## setup MPI if requested
    comm, rank, size = assign_mpi(args.mpi, do_batch=False, log=log)

    ## Need to know where to look for the exposures
    if args.nights is None and args.reference_night is None:
        if rank == 0:
            log.error("At least one of --nights or --reference-night must be provided to identify the exposures to process.")
        return 1
    elif args.nights is not None and args.reference_night is not None:
        if rank == 0:
            log.warning("Both --nights and --reference-night set, this will IGNORE --reference-night.")

    if args.specprod is not None :
        os.environ["SPECPROD"] = args.specprod

    # check consistency of input options
    if args.nights is not None and args.expids is not None and rank == 0:
        log.info(f"Assuming all exposures in {args.expids} can be found in nights={args.nights}.")

    ## Make sure the expids and nights are lists of integers
    args.expids = np.array(args.expids.split(',')).astype(int)
    if args.nights is not None:
        args.nights = np.array(args.nights.split(',')).astype(int)

    ## get the requested cameras from the camword
    requested_cameras = set(decode_camword(args.camword))
    if len(requested_cameras) == 0:
        if rank == 0:
            log.error(f'No cameras found in camword {args.camword}.')
        return 1

    ## bias to preprocess with; True lets preproc find each night's nightly bias
    if args.bias is None:
        thisbias = True
    else:
        thisbias = args.bias

    # first find the exposures if they are not given in input
    if rank == 0:
        ## Use the compute_dark_night parser to get the exposure table
        ## so that we have consistent exposure selection via default values
        ## for number of nights nbefore and after the reference night
        if args.nights is None:
            ## camera and outfile are required, so give dummy values for those
            opts = ['--reference-night', str(args.reference_night), '-o', 'temp', '-c', 'b1',
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
        exptable = exptable[np.isin(exptable["EXPID"].data, args.expids)]

        # assemble corresponding images
        expids, camlists, files, nights = [], [], [], []
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

        ## The preprocessed darks are only usable for the nightly darks if they
        ## strictly use the bias they are supposed to, i.e. the nightly bias of
        ## their own night and camera unless --bias was given, so exit without
        ## preprocessing anything if even one camera can't (issue #2741)
        if args.allow_default_bias:
            log.warning("--allow-default-bias set, so NOT requiring that every camera "
                        + "uses a matching bias.")
            errors = []
        else:
            errors = check_matching_biasnights(expids, nights, camlists, files,
                                               bias=thisbias,
                                               preproc_dark_dir=args.preproc_dark_dir)

        if len(errors) > 0:
            log.critical(f"Not preprocessing any darks because {len(errors)} problem(s) "
                         + "would keep a camera from using a matching bias. This job "
                         + "will keep failing until the biases listed above are fixed "
                         + "and any mismatched files are purged.")
            data = None
        else:
            data = (expids, nights, camlists, files)
    else:
        data = None

    # Broadcast data to all ranks if we're in MPI mode
    if comm is not None:
        data = comm.bcast(data, root=0)

    if data is None:
        return 1

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

    ## number of cameras this rank refused to write because they wouldn't have
    ## used a matching nightly bias
    nfail = 0

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
                # hdu_cameras = [hdu.name.lower() for hdu in fx if hdu.name.lower() != 'primary']
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
        ## if there is a communicator and more than one rank per block
        if block_comm is not None and block_size > 1:
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

            ## The bias was checked before any preprocessing started, but check
            ## what preproc actually used before writing anything (#2741)
            if not args.allow_default_bias:
                ## preproc looks up the bias with the night in the header, which
                ## has to agree with the exposure table night this file is
                ## written under, since that is where compute_dark_file will
                ## look for it. The upfront check requires that of the primary
                ## header and process_raw requires the camera header to match
                ## the primary one, so this should be unreachable.
                try:
                    header_night = header2night(img.meta)
                except (KeyError, ValueError, TypeError):
                    header_night = night

                if int(header_night) != int(night):
                    log.error(f"Rank {rank} block_rank {block_rank} block_num {block_num}: "
                              + f"NOT writing {preproc_filename} because {filename} camera "
                              + f"{camera} header NIGHT={header_night} disagrees with exposure "
                              + f"table NIGHT={night}, so the nightly dark would look for this "
                              + "file under a different night")
                    nfail += 1
                    continue

                ## the header night is what preproc resolved the bias with, and
                ## what compute_dark_file will expect of this file later
                expected_bias = expected_dark_preproc_bias(thisbias, header_night, camera)
                is_match, biasused = dark_preproc_bias_matches(img.meta, expected_bias)
                if not is_match:
                    log.error(f"Rank {rank} block_rank {block_rank} block_num {block_num}: "
                              + f"NOT writing {preproc_filename} because preproc used {biasused} "
                              + f"instead of {os.path.basename(expected_bias)}")
                    nfail += 1
                    continue

            # is saved in preproc_dark_dir if not None
            write_image(preproc_filename,img)
            log.info(f"Wrote {preproc_filename}")

    if comm is not None:
        comm.barrier()
        nfails = comm.gather(nfail, root=0)
        if rank == 0:
            nfail = int(np.sum(nfails))
        nfail = comm.bcast(nfail, root=0)

    if nfail > 0:
        if rank == 0:
            log.error(f"{nfail} preprocessed dark(s) were not written because they would "
                      + "not have used the required bias. This job will keep failing until "
                      + "the errors logged above are fixed.")
        return 1

    if rank == 0:
        log.info("All ranks have completed preproc_darks_mpi.")

    return 0



if __name__ == "__main__":
    main()
