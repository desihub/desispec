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
from mpi4py import MPI


def dark_night_parser():
    parser = compute_dark.compute_dark_parser()

    parser.description="Compute the dark night for a given --reference-night or list of nights",
    parser.usage='desi_compute_dark_night.py [options] -r YYYYMMDD -c CAMWORD',
    parser.epilog='''Input is a list of raw dark images, possibly with various exposure times.
                    Raw images are preprocessed without dark,mask correction.
                    However gains are applied so the output is in electrons/sec.
                    '''
    parser.remove_argument('--images')  # remove the images argument
    parser.remove_argument('--camera')  # remove the camera argument
    parser.add_argument('-c', '--camword', type=str, default='a0123456789', required=False,
                        help='Camera word defining the cameras to process. Default is all cameras (a0123456789).')
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

    # original_bias = args.bias
    error_count = 0
    for camera in requested_cameras[rank::size]:
        # ## define the bias explicitly
        # if original_bias is None:
        #     args.bias = findfile("biasnight", night=night, camera=camera)

        ## assign camera to the rest of the arguments and pass them into compute_dark.main
        ## don't explciitly list inputs since there are many and aren't yet known
        outfile = findfile("darknight", night=night, camera=camera)
        args.camera = camera
        log.info(f'Rank {rank} Running desi_compute_dark for camera: {camera}, outfile: {outfile}')
        # with stdouterr_redirected(darklog, comm=comm):
        result, success = runcmd(compute_dark.main, comm=comm, args=args,
                                    inputs=[], outputs=[outfile])

        if not success:
            log.error(f'Rank {rank} failed for camera {camera}, outfile: {outfile}')
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
