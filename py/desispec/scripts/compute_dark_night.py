#!/usr/bin/env python

import os
import argparse
import time
from desispec.scripts import compute_dark
from desispec.workflow.desi_proc_funcs import assign_mpi
import numpy as np

from desiutil.log import get_logger

from desispec.util import runcmd
from desispec.io.util import decode_camword
from desispec.io import findfile


def dark_night_parser():
    """
    Takes the "base" argparser that is also used by desi_compute_dark_night and add specific args
    """
    parser = compute_dark.compute_dark_baseparser()

    parser.description="Compute the dark night for a given --reference-night or list of nights"
    parser.usage='desi_compute_dark_night.py [options] -r YYYYMMDD -c CAMWORD'
    parser.epilog='Input is a night and camword'
    parser.add_argument('-c', '--camword', type=str, default='a0123456789', required=False,
                        help='Camera word defining the cameras to process. Default is all cameras (a0123456789).')
    parser.add_argument('--mpi', action='store_true', default=False,
                        help='Run in MPI mode, using all available ranks. Default is False.')
    return parser


def parse(options=None):
    # parse the command line arguments
    parser = dark_night_parser()
    
    #- uses sys.argv if options=None
    args = parser.parse_args(options)

    return args


def main(args=None):
    start_time = time.time()
    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    log  = get_logger()

    comm, rank, size = assign_mpi(args.mpi, do_batch=False, log=log)

    if args.specprod is not None :
        os.environ["SPECPROD"] = args.specprod

    # check consistency of input options
    if args.reference_night is not None and args.nights is not None :
        log.error("Cannot specify both --reference-night and --nights arguments")
        return 1

    if args.reference_night is None and args.nights is None:
        log.error("Need to specify input using --reference-night or --nights")
        return 1

    if args.nights is not None:
        log.info(f'Processing nights: {args.nights}. Assuming last night, {args.nights[-1]}, is the reference night.')
        night = args.nights[-1]
    else:
        night = args.reference_night

    ## get the requested cameras from the camword
    requested_cameras = decode_camword(args.camword)
    if len(requested_cameras) == 0:
        log.error(f'No cameras found in camword {args.camword}.')
        return 1
    
    del args.camword  # not necessary, but remove camword from args to avoid confusion in compute_dark.main 

    ## Only read in the exposure tables once and broadcast it to all ranks.
    exptable = None
    if rank == 0:
        args.skip_camera_check = True  # we are going to run compute_dark for all cameras, so skip the camera check
        args.dont_search_filesystem = True  # we are going to trust the exposure tables, so don't search the filesystem
        exptable = compute_dark.get_stacked_dark_exposure_table(args)
        args.skip_camera_check = False
    if comm is not None:
        exptable = comm.bcast(exptable, root=0)

    original_bias = args.bias
    error_count = 0
    for camera in requested_cameras[rank::size]:
        ## define the bias explicitly
        if original_bias is None:
            args.bias = findfile("biasnight", night=night, camera=camera, readonly=True)

        ## assign camera to the rest of the arguments and pass them into compute_dark.main
        ## don't explciitly list dark inputs since there are many and aren't yet known
        args.images = None
        args.outfile = findfile("darknight", night=night, camera=camera)
        args.camera = camera
        log.info(f'Rank {rank} Running desi_compute_dark for camera: {camera}, outfile: {args.outfile}')
        ## for now let's not do log redirecting, and we don't need to pass the comm since each
        ## rank is running their own serial command
        # with stdouterr_redirected(darklog, comm=comm):
        #result, success = runcmd(compute_dark.main, comm=comm, args=args,
        #                            inputs=[args.bias], outputs=[args.outfile])
        result, success = runcmd(compute_dark.main, args=[args, exptable], expandargs=True,
                                    inputs=[args.bias], outputs=[args.outfile])
        if not success:
            log.error(f'Rank {rank} failed for camera {camera}, outfile: {args.outfile}')
            error_count += 1
     
    if comm is not None:
        comm.barrier()

        all_error_counts = comm.gather(error_count, root=0)
        if rank == 0:
            final_error_count = int(np.sum(all_error_counts))
        else:
            final_error_count = 0
        error_count = comm.bcast(final_error_count, root=0)

    if rank == 0:
        duration_seconds = time.time() - start_time
        mm = int(duration_seconds) // 60
        ss = int(duration_seconds - mm*60)
        goodbye = f'All done at {time.asctime()}; duration {mm}m{ss}s'

        if error_count > 0:
            log.error(f'{error_count} processing errors; see logs above')
            log.error(goodbye)
        else:
            log.info(goodbye)

    return int(error_count)



if __name__ == "__main__":
    main()
