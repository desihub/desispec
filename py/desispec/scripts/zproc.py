"""
One stop shopping for redshifting  DESI spectra

"""

import time, datetime
start_imports = time.time()

import sys, os, argparse, re
import subprocess
from copy import deepcopy
import json

import numpy as np
import fitsio
from astropy.io import fits

from astropy.table import Table,vstack

import glob
import desiutil.timer
import desispec.io
from desispec.io import findfile, replace_prefix, shorten_filename, get_readonly_filepath
from desispec.io.util import create_camword, decode_camword, parse_cameras
from desispec.io.util import validate_badamps, get_tempfilename
from desispec.util import runcmd

from desiutil.log import get_logger, DEBUG, INFO
import desiutil.iers
from desispec.workflow import batch
from desispec.workflow.desi_proc_funcs import assign_mpi, update_args_with_headers, _log_timer
from desispec.workflow.desi_proc_funcs import determine_resources, create_desi_proc_batch_script

stop_imports = time.time()

#########################################
######## Begin Body of the Code #########
#########################################

def parse(options=None):
    """
    Create an argparser object for use with desi_proc AND desi_proc_joint_fit based on arguments from sys.argv
    """
    parser = argparse.ArgumentParser(usage="{prog} [options]")

    parser.add_argument("-n", "--nights", type=int, nargs='*', help="YEARMMDD night")
    parser.add_argument("-g", "--groupname", type=str, default='cumulative',
                        help="Redshift grouping type: cumulative, perexp, pernight")
    parser.add_argument("-e", "--expids", type=int, nargs='*', help="Exposure IDs")
    parser.add_argument("-t", "--tileid", type=str, default=None, help="Tile ID")
    parser.add_argument("-p", "--healpix", type=str, default=None, help="Healpix")
    parser.add_argument("--cameras", type=str,
                        help="Explicitly define the spectrographs for which you want" +
                             " to reduce the data. Should be a comma separated list." +
                             " Numbers only assumes you want to reduce R, B, and Z " +
                             "for that spectrograph. Otherwise specify separately [BRZ|brz][0-9].")
    parser.add_argument("--mpi", action="store_true",
                        help="Use MPI parallelism")
    parser.add_argument("--run-zmtl", action="store_true",
                        help="Whether to run zmtl or not")
    parser.add_argument("--batch", action="store_true",
                        help="Submit a batch job to process this exposure")
    parser.add_argument("--nosubmit", action="store_true",
                        help="Create batch script but don't submit")
    parser.add_argument("-q", "--queue", type=str, default="realtime",
                        help="batch queue to use")
    parser.add_argument("--batch-opts", type=str, default=None,
                        help="additional batch commands")
    parser.add_argument("--runtime", type=int, default=None,
                        help="batch runtime in minutes")
    parser.add_argument("--starttime", type=str,
                        help='start time; use "--starttime `date +%%s`"')
    parser.add_argument("--timingfile", type=str,
                        help='save runtime info to this json file; augment if pre-existing')
    parser.add_argument("--system-name", type=str, default=batch.default_system(),
                        help='Batch system name (cori-haswell, perlmutter-gpu, ...)')
    parser.add_argument("-d", "--dryrun", action="store_true",
                        help="show commands only, do not run")
    args = parser.parse_args(options)
    return args


def main(args=None, comm=None):
    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    log = get_logger()
    start_time = time.time()
    error_count = 0

    start_mpi_connect = time.time()
    if comm is not None:
        #- Use the provided comm to determine rank and size
        rank = comm.rank
        size = comm.size
    else:
        #- Check MPI flags and determine the comm, rank, and size given the arguments
        comm, rank, size = assign_mpi(do_mpi=args.mpi, do_batch=args.batch, log=log)
    stop_mpi_connect = time.time()

    if rank == 0:
        thisfile=os.path.dirname(os.path.abspath(__file__))
        thistime=datetime.datetime.fromtimestamp(start_imports).isoformat()
        log.info(f'rank 0 started {thisfile} at {thistime}')
    #- Start timer; only print log messages from rank 0 (others are silent)
    timer = desiutil.timer.Timer(silent=(rank>0))

    #- Fill in timing information for steps before we had the timer created
    if args.starttime is not None:
        timer.start('startup', starttime=args.starttime)
        timer.stop('startup', stoptime=start_imports)

    timer.start('imports', starttime=start_imports)
    timer.stop('imports', stoptime=stop_imports)

    timer.start('mpi_connect', starttime=start_mpi_connect)
    timer.stop('mpi_connect', stoptime=stop_mpi_connect)

    #- Freeze IERS after parsing args so that it doesn't bother if only --help
    timer.start('freeze_iers')
    desiutil.iers.freeze_iers()
    timer.stop('freeze_iers')

    timer.start('preflight')
    if comm is not None:
        args = comm.bcast(args, root=0)

    known_groups = ['cumulative', 'pernight', 'perexp']
    if args.groupname not in known_groups:
        raise ValueError('obstype {} not in {}'.format(args.groupname, known_groups))

    if isinstance(args.cameras, str):
        args.cameras = parse_cameras(args.cameras)

    timer.stop('preflight')

    #-------------------------------------------------------------------------
    #- Create and submit a batch job if requested

    if args.batch:
        # scriptfile = create_desi_zproc_batch_script(night=args.night, exp=args.expid,
        #                                            cameras=args.cameras,
        #                                            jobdesc=jobdesc, queue=args.queue,
        #                                            runtime=args.runtime,
        #                                            batch_opts=args.batch_opts,
        #                                            timingfile=args.timingfile,
        #                                            system_name=args.system_name)
        log.info("Generating batch script and exiting.")
        err = 0
        # if not args.nosubmit and not args.dryrun:
        #     err = subprocess.call(['sbatch', scriptfile])
        sys.exit(err)

    #-------------------------------------------------------------------------
    #- Proceeding with running

    #- What are we going to do?
    if rank == 0:
        log.info('----------')
        log.info('Input {}'.format(args.input))
        log.info('Groupname {}'.format(args.groupname))
        if args.healpix is not None:
            log.info(f'Healpix {args.healpix} nights {args.nights} expids {args.expids}')
        else:
            log.info(f'Tileid {args.tile} nights {args.nights} expids {args.expids}')
        log.info('Cameras {}'.format(args.cameras))
        log.info('Output root {}'.format(desispec.io.specprod_root()))
        log.info('----------')


    #-------------------------------------------------------------------------
    #- Create output directories if needed
    val = None
    if rank == 0:
        val = 100.
        log.info(f"Define value on rank 0: val={val}")
        # preprocdir = os.path.dirname(findfile('preproc', args.night, args.expid, 'b0'))
        # expdir = os.path.dirname(findfile('frame', args.night, args.expid, 'b0'))
        # os.makedirs(preprocdir, exist_ok=True)
        # os.makedirs(expdir, exist_ok=True)

    log.info(f"Print value for rank {rank}: val={val}")
    #- Wait for rank 0 to make directories before proceeding
    if comm is not None:
        comm.barrier()


    #- If assemble_fibermap failed and obstype is SCIENCE, exit now
    if comm is not None:
        val = comm.bcast(val, root=0)

    log.info(f"Print value for rank {rank}: val={val}")

    timer.stop('fibermap')

    # outpsf = replace_prefix(psfname,"psf","fit-psf-fixed-listed")
    # if os.path.isfile(inpsf) and not os.path.isfile(outpsf):
    #     cmd = 'desi_interpolate_fiber_psf'
    #     cmd += ' --infile {}'.format(inpsf)
    #     cmd += ' --outfile {}'.format(outpsf)
    #     cmd += ' --fibers {}'.format(fibers_to_ignore_str)
    #     log.info('For camera {} interpolating PSF for fibers: {}'.format(camera,fibers_to_ignore_str))
    #     cmdargs = cmd.split()[1:]
    #
    #     result, success = runcmd(desispec.scripts.interpolate_fiber_psf.main,
    #             args=cmdargs, inputs=[inpsf], outputs=[outpsf])
    #
    #     if not success:
    #         error_count += 1

    # dt = time.time() - t0
    # log.info(f'Rank {rank} {camera} PSF interpolation took {dt:.1f} sec')


    #- Compute flux calibration vectors per camera
    # for camera in args.cameras[rank::size]:
    #     framefile = findfile('frame', night, expid, camera, readonly=True)
    #     skyfile = findfile('sky', night, expid, camera, readonly=True)
    #     spectrograph = int(camera[1])
    #     stdfile = findfile('stdstars', night, expid,spectrograph=spectrograph, readonly=True)
    #     fiberflatfile = findfile('fiberflatexp', night, expid, camera, readonly=True)
    #     calibfile = findfile('fluxcalib', night, expid, camera)
    #     calibstars = findfile('calibstars',night, expid)
    #
    #     cmd = "desi_compute_fluxcalibration"
    #     cmd += " --infile {}".format(framefile)
    #     cmd += " --sky {}".format(skyfile)
    #     cmd += " --fiberflat {}".format(fiberflatfile)
    #     cmd += " --models {}".format(stdfile)
    #     cmd += " --outfile {}".format(calibfile)
    #     cmd += " --selected-calibration-stars {}".format(calibstars)
    #
    #     inputs = [framefile, skyfile, fiberflatfile, stdfile, calibstars]
    #     cmdargs = cmd.split()[1:]
    #
    #     result, success = runcmd(desispec.scripts.fluxcalibration.main,
    #             args=cmdargs, inputs=inputs, outputs=[calibfile,])
    #
    #     if not success:
    #         error_count += 1
    #
    # timer.stop('fluxcalib')
    # if comm is not None:
    #     comm.barrier()

    #-------------------------------------------------------------------------
    #- Collect error count
    if comm is not None:
        all_error_counts = comm.gather(error_count, root=0)
        error_count = int(comm.bcast(np.sum(all_error_counts), root=0))

    if rank == 0 and error_count > 0:
        log.error(f'{error_count} processing errors; see logs above')

    #-------------------------------------------------------------------------
    #- Wrap up

    _log_timer(timer, args.timingfile, comm=comm)
    if rank == 0:
        duration_seconds = time.time() - start_time
        mm = int(duration_seconds) // 60
        ss = int(duration_seconds - mm*60)

        log.info('All done at {}; duration {}m{}s'.format(
            time.asctime(), mm, ss))

    if error_count > 0:
        sys.exit(int(error_count))
    else:
        return 0
