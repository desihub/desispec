"""
desispec.scripts.proc_joint_fit
===============================

Please add module-level documentation.
"""
import time
start_imports = time.time()

#- enforce a batch-friendly matplotlib backend
from desispec.util import set_backend
set_backend()

import sys, os, argparse, re
import traceback
import subprocess
from copy import deepcopy
import json
import glob

import numpy as np
import fitsio
from astropy.io import fits

import desiutil.timer
import desispec.io
from desispec.io import findfile, replace_prefix, get_readonly_filepath
from desispec.io.util import create_camword, get_tempfilename, relsymlink
from desispec.calibfinder import findcalibfile,CalibFinder
from desispec.util import runcmd, mpi_count_failures
import desispec.scripts.specex
import desispec.scripts.stdstars
import desispec.scripts.average_fiberflat
import desispec.scripts.autocalib_fiberflat

from desitarget.targetmask import desi_mask

from desiutil.log import get_logger, DEBUG, INFO
import desiutil.iers

from desispec.workflow.desi_proc_funcs import assign_mpi, get_desi_proc_joint_fit_parser, create_desi_proc_batch_script, \
                                              find_most_recent
from desispec.workflow.desi_proc_funcs import load_raw_data_header, update_args_with_headers

stop_imports = time.time()

def parse(options=None):
    parser = get_desi_proc_joint_fit_parser()
    args = parser.parse_args(options)
    return args

def main(args=None, comm=None):
    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    log = get_logger()

    start_mpi_connect = time.time()
    if comm is not None:
        #- Use the provided comm to determine rank and size
        rank = comm.rank
        size = comm.size
    else:
        #- Check MPI flags and determine the comm, rank, and size given the arguments
        comm, rank, size = assign_mpi(do_mpi=args.mpi, do_batch=args.batch, log=log)
    stop_mpi_connect = time.time()

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

    #- Preflight checks
    timer.start('preflight')

    # - Preflight checks
    if rank > 0:
        # - Let rank 0 fetch these, and then broadcast
        args, hdr, camhdr = None, None, None
    else:
        if args.inputs is None:
            if args.night is None or args.expids is None:
                raise RuntimeError('Must specify --inputs or --night AND --expids')
            else:
                args.expids = np.array(args.expids.strip(' \t').split(',')).astype(int)
                args.inputs = []
                for expid in args.expids:
                    infile = findfile('raw', night=args.night, expid=expid, readonly=True)
                    args.inputs.append(infile)
                    if not os.path.isfile(infile):
                        raise IOError('Missing input file: {}'.format(infile))
        else:
            args.inputs = np.array(args.inputs.strip(' \t').split(','))
            #- args.night will be defined in update_args_with_headers,
            #- but let's define the expids here
            #- NOTE: inputs has priority. Overwriting expids if they existed.
            args.expids = []
            for infile in args.inputs:
                hdr = load_raw_data_header(pathname=infile, return_filehandle=False)
                args.expids.append(int(hdr['EXPID']))

        args.expids = np.sort(args.expids)
        args.inputs = np.sort(args.inputs)
        args.expid = args.expids[0]
        args.input = args.inputs[0]

        #- Use header information to fill in missing information in the arguments object
        args, hdr, camhdr = update_args_with_headers(args)

        #- If not a science observation, we don't need the hdr or camhdr objects,
        #- So let's not broadcast them to all the ranks
        if args.obstype != 'SCIENCE':
            hdr, camhdr = None, None

    if comm is not None:
        args = comm.bcast(args, root=0)
        hdr = comm.bcast(hdr, root=0)
        camhdr = comm.bcast(camhdr, root=0)

    known_obstype = ['SCIENCE', 'ARC', 'FLAT']
    if args.obstype not in known_obstype:
        raise RuntimeError('obstype {} not in {}'.format(args.obstype, known_obstype))


    timer.stop('preflight')


    # -------------------------------------------------------------------------
    # - Create and submit a batch job if requested

    if args.batch:
        #camword = create_camword(args.cameras)
        #exp_str = '-'.join('{:08d}'.format(expid) for expid in args.expids)
        if args.obstype.lower() == 'science':
            jobdesc = 'stdstarfit'
        elif args.obstype.lower() == 'arc':
            jobdesc = 'psfnight'
        elif args.obstype.lower() == 'flat':
            jobdesc = 'nightlyflat'
        else:
            jobdesc = args.obstype.lower()
        scriptfile = create_desi_proc_batch_script(night=args.night, exp=args.expids, cameras=args.cameras,\
                                                jobdesc=jobdesc, queue=args.queue, runtime=args.runtime,\
                                                batch_opts=args.batch_opts, timingfile=args.timingfile,
                                                system_name=args.system_name)
        err = 0
        if not args.nosubmit:
            err = subprocess.call(['sbatch', scriptfile])
        sys.exit(err)

    # -------------------------------------------------------------------------
    # - Proceed with running

    # - What are we going to do?
    if rank == 0:
        log.info('----------')
        log.info('Input {}'.format(args.inputs))
        log.info('Night {} expids {}'.format(args.night, args.expids))
        log.info('Obstype {}'.format(args.obstype))
        log.info('Cameras {}'.format(args.cameras))
        log.info('Output root {}'.format(desispec.io.specprod_root()))
        log.info('----------')

    # - sync ranks before proceeding
    if comm is not None:
        comm.barrier()

    # -------------------------------------------------------------------------
    # - Merge PSF of night if applicable

    if args.obstype in ['ARC']:
        timer.start('psfnight')
        num_cmd = num_err = 0

        # rank 0 create output dir for everyone; other ranks wait at barrier
        if rank == 0:
            psfnightfile = findfile('psfnight', args.night, args.expids[0],
                                                args.cameras[0])
            dirname = os.path.dirname(psfnightfile)
            os.makedirs(dirname, exist_ok=True)

        if comm is not None:
            comm.barrier()

        for camera in args.cameras[rank::size]:
            psfnightfile = findfile('psfnight', args.night, args.expids[0], camera)
            if not os.path.isfile(psfnightfile):  # we still don't have a psf night, see if we can compute it ...
                psfs = list()
                for expid in args.expids:
                    psffile = findfile('fitpsf', args.night, expid, camera, readonly=True)
                    if os.path.exists(psffile):
                        psfs.append( psffile )
                    else:
                        log.warning(f'Missing {psffile}')

                log.info("Number of PSF for night={} camera={} = {}/{}".format(
                        args.night, camera, len(psfs), len(args.expids)))
                if len(psfs) >= 3:  # lets do it!
                    log.info(f"Rank {rank} computing {camera} psfnight ...")
                    num_cmd += 1

                    #- generic try/except so that any failure doesn't leave
                    #- MPI rank 0 hanging while others are waiting for it
                    try:
                        desispec.scripts.specex.mean_psf(psfs, psfnightfile)
                    except:
                        log.error('Rank {} specex.meanpsf failed for {}'.format(rank, os.path.basename(psfnightfile)))
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                        log.error(''.join(lines))
                        sys.stdout.flush()

                    if not os.path.exists(psfnightfile):
                        log.error(f'Rank {rank} failed to create {psfnightfile}')
                        num_err += 1
                else:
                    log.info(f"Fewer than 3 {camera} psfs were provided, can't compute psfnight. Exiting ...")
                    num_cmd += 1
                    num_err += 1

        timer.stop('psfnight')

        num_cmd, num_err = mpi_count_failures(num_cmd, num_err, comm=comm)
        if rank == 0:
            if num_err > 0:
                log.error(f'{num_err}/{num_cmd} psfnight commands failed')

        if num_err>0 and num_err==num_cmd:
            sys.stdout.flush()
            if rank == 0:
                log.critical('All psfnight commands failed')
            sys.exit(1)


    # -------------------------------------------------------------------------
    # - Average and auto-calib fiberflats of night if applicable

    if args.obstype in ['FLAT']:
        timer.start('fiberflatnight')
        #- Track number of commands run and number of errors for exit code
        num_cmd = 0
        num_err = 0

        # Commands to run:
        # - desi_average_fiberflat for each (camera, lampbox) combo (max 30*4 = 120)
        # - desi_autocalib_fiberflat for each (b,r,z)

        #- temp directory for averaged flats
        fiberflatnightfile = findfile('fiberflatnight', args.night, args.expids[0], args.cameras[0])
        fiberflatdirname = os.path.dirname(fiberflatnightfile)
        tmpdir = os.path.join(fiberflatdirname, "tmp")

        #- expected inputs and outputs
        inflats = list()
        inflats_for_camera = dict()
        average_flats_for_camera = dict()
        camera_lampboxes = list()
        flats_for_arm = dict(b=list(), r=list(), z=list())
        for camera in args.cameras:
            inflats_for_camera[camera] = list()
            for expid in args.expids:
                filename = findfile('fiberflat', args.night, expid, camera, readonly=True)
                inflats.append(filename)
                inflats_for_camera[camera].append(filename)

            average_flats_for_camera[camera] = list()
            for lampbox in range(4):
                ofile = os.path.join(tmpdir, f"fiberflatnight-camera-{camera}-lamp-{lampbox}.fits")
                average_flats_for_camera[camera].append(ofile)
                camera_lampboxes.append( (camera, lampbox, ofile) )
                arm = camera[0].lower()
                flats_for_arm[arm].append(ofile)

        #- rank 0 checks if we have enough inputs
        enough_inputs = True
        if rank == 0:
            log.info("Number of fiberflat for night {} = {}".format(args.night, len(inflats)))
            if len(inflats) < 3 * 4 * len(args.cameras):
                log.critical("Fewer than 3 exposures with 4 lamps were available. Can't perform joint fit. Exiting...")
                enough_inputs = False
            elif len(args.cameras) < 6:
                log.critical("Fewer than 6 cameras were available, so couldn't perform joint fit. Exiting ...")
                enough_inputs = False

        if comm is not None:
            enough_inputs = comm.bcast(enough_inputs, root=0)

        if not enough_inputs:
            sys.exit(1)

        #- we have enough inputs, ok to proceed
        if rank == 0:
            log.info("Computing fiberflatnight per lamp and camera ...")

        #- rank 0 create tmpdir; other ranks wait at barrier
        if rank == 0:
            os.makedirs(tmpdir, exist_ok=True)

        if comm is not None:
            comm.barrier()

        #- Averaged fiberflats per camera and lampbox
        for camera, lampbox, ofile in camera_lampboxes[rank::size]:
            if not os.path.isfile(ofile):
                log.info(f"Rank {rank} average flat for camera {camera} and lamp box #{lampbox}")
                cmd = []
                cmd.append(f"desi_average_fiberflat")
                cmd.append(f"--program")
                cmd.append(f"CALIB DESI-CALIB-0{lampbox} LEDs only")
                cmd.append(f"--outfile")
                cmd.append(f"{ofile}")
                cmd.append(f"-i")
                for flat in inflats_for_camera[camera]:
                    cmd.append(f"{flat}")
                num_cmd += 1
                cmdargs = cmd[1:]

                result, success = runcmd(desispec.scripts.average_fiberflat.main,
                        args=cmdargs, inputs=inflats_for_camera[camera], outputs=[ofile, ])

                if not success:
                    num_err += 1
            else:
                log.info(f"Rank {rank} will use existing {ofile}")

        if comm is not None:
            comm.barrier()

        log.info("Auto-calibration across lamps and spectro  per camera arm (b,r,z)")
        for camera_arm in ["b", "r", "z"][rank::size]:
            log.info(f"Rank {rank} autocalibrating across spectro for camera arm {camera_arm}")
            cmd = []
            cmd.append(f"desi_autocalib_fiberflat")
            cmd.append(f"--night")
            cmd.append(f"{args.night}")
            cmd.append(f"--arm")
            cmd.append(f"{camera_arm}")
            cmd.append(f"-i")
            for flat in flats_for_arm[camera_arm]:
                cmd.append(f"{flat}")
            num_cmd += 1
            cmdargs = cmd[1:]

            result, success = runcmd(desispec.scripts.autocalib_fiberflat.main,
                    args=cmdargs, inputs=flats_for_arm[camera_arm], outputs=[])

            if not success:
                num_err += 1

        if comm is not None:
            comm.barrier()

        timer.stop('fiberflatnight')
        num_cmd, num_err = mpi_count_failures(num_cmd, num_err, comm=comm)
        if comm is not None:
            comm.barrier()

        if rank == 0:
            if num_err > 0:
                log.error(f'{num_err}/{num_cmd} fiberflat commands failed')
            else:
                log.info('All commands succeeded; removing lamp-averaged fiberflats')
                for camera, lampbox, ofile in camera_lampboxes:
                    os.remove(ofile)

                os.rmdir(tmpdir)

        if num_err>0 and num_err==num_cmd:
            if rank == 0:
                log.critical('All fiberflat commands failed')
            sys.exit(1)


    ##################### Note #############################
    ### Still for single exposure. Needs to be re-factored #
    ########################################################

    if args.obstype in ['SCIENCE']:
        #inputfile = findfile('raw', night=args.night, expid=args.expids[0])
        #if not os.path.isfile(inputfile):
        #    raise IOError('Missing input file: {}'.format(inputfile))
        ## - Fill in values from raw data header if not overridden by command line
        #fx = fitsio.FITS(inputfile)
        #if 'SPEC' in fx:  # - 20200225 onwards
        #    # hdr = fits.getheader(args.input, 'SPEC')
        #    hdr = fx['SPEC'].read_header()
        #elif 'SPS' in fx:  # - 20200224 and before
        #    # hdr = fits.getheader(args.input, 'SPS')
        #    hdr = fx['SPS'].read_header()
        #else:
        #    # hdr = fits.getheader(args.input, 0)
        #    hdr = fx[0].read_header()
        #
        #camhdr = dict()
        #for cam in args.cameras:
        #    camhdr[cam] = fx[cam].read_header()

        #fx.close()

        timer.start('stdstarfit')
        num_err = num_cmd = 0
        if rank == 0:
            log.info('Starting stdstar fitting at {}'.format(time.asctime()))

        # -------------------------------------------------------------------------
        # - Get input fiberflat
        input_fiberflat = dict()
        if rank == 0:
            for camera in args.cameras:
                if args.fiberflat is not None:
                    input_fiberflat[camera] = args.fiberflat
                elif args.calibnight is not None:
                    # look for a fiberflatnight for this calib night
                    fiberflatnightfile = findfile('fiberflatnight',
                                            args.calibnight, args.expids[0], camera, readonly=True)
                    if not os.path.isfile(fiberflatnightfile):
                        log.error("no {}".format(fiberflatnightfile))
                        raise IOError("no {}".format(fiberflatnightfile))
                    input_fiberflat[camera] = fiberflatnightfile
                else:
                    # look for a fiberflatnight fiberflat
                    fiberflatnightfile = findfile('fiberflatnight',
                                                args.night, args.expids[0], camera, readonly=True)
                if os.path.isfile(fiberflatnightfile):
                        input_fiberflat[camera] = fiberflatnightfile
                elif args.most_recent_calib:
                    # -- NOTE: Finding most recent only with respect to the first night
                    nightfile = find_most_recent(args.night, file_type='fiberflatnight')
                    if nightfile is None:
                        input_fiberflat[camera] = findcalibfile([hdr, camhdr[camera]], 'FIBERFLAT')
                    else:
                        input_fiberflat[camera] = nightfile
                else:
                    input_fiberflat[camera] = findcalibfile([hdr, camhdr[camera]], 'FIBERFLAT')
            log.info("Will use input FIBERFLAT: {}".format(input_fiberflat[camera]))

        if comm is not None:
            input_fiberflat = comm.bcast(input_fiberflat, root=0)

        # - Group inputs by spectrograph
        # - collect inputs on rank 0 so only one rank checks for file existence
        framefiles = dict()
        skyfiles = dict()
        fiberflatfiles = dict()
        num_brz = dict()
        if rank == 0:
            for camera in args.cameras:
                sp = int(camera[1])
                if sp not in framefiles:
                    framefiles[sp] = list()
                    skyfiles[sp] = list()
                    fiberflatfiles[sp] = list()
                    num_brz[sp] = dict(b=0, r=0, z=0)

                fiberflatfiles[sp].append(input_fiberflat[camera])
                for expid in args.expids:
                    tmpframefile = findfile('frame', args.night, expid, camera, readonly=True)
                    tmpskyfile = findfile('sky', args.night, expid, camera, readonly=True)

                    inputsok = True
                    if not os.path.exists(tmpframefile):
                        log.error(f'Missing expected frame {tmpframefile}')
                        inputsok = False

                    if not os.path.exists(tmpskyfile):
                        log.error(f'Missing expected sky {tmpskyfile}')
                        inputsok = False

                    if inputsok:
                        framefiles[sp].append(tmpframefile)
                        skyfiles[sp].append(tmpskyfile)
                        num_brz[sp][camera[0]] += 1

        if comm is not None:
            framefiles = comm.bcast(framefiles, root=0)
            skyfiles = comm.bcast(skyfiles, root=0)
            fiberflatfiles = comm.bcast(fiberflatfiles, root=0)
            num_brz = comm.bcast(num_brz, root=0)

        # - Hardcoded stdstar model version
        starmodels = os.path.join(
            os.getenv('DESI_BASIS_TEMPLATES'), 'stdstar_templates_v2.2.fits')
        starmodels = get_readonly_filepath(starmodels)

        # - Fit stdstars per spectrograph (not per-camera)
        spectro_nums = sorted(framefiles.keys())

        if args.mpistdstars and comm is not None:
            #- If using MPI parallelism in stdstar fit, divide comm into subcommunicators.
            #- (spectro_start, spectro_step) determine stride pattern over spectro_nums.
            #- Split comm by at most len(spectro_nums)
            num_subcomms = min(size, len(spectro_nums))
            subcomm_index = rank % num_subcomms
            if rank == 0:
                log.info(f"Splitting comm of {size=} into {num_subcomms=} for stdstar fitting")
            subcomm = comm.Split(color=subcomm_index)
            spectro_start, spectro_step = subcomm_index, num_subcomms
        else:
            #- Otherwise, use multiprocessing assuming 1 MPI rank per spectrograph
            spectro_start, spectro_step = rank, size
            subcomm = None

        for i in range(spectro_start, len(spectro_nums), spectro_step):
            sp = spectro_nums[i]

            have_all_cameras = True
            for cam in ['b', 'r', 'z']:
                if num_brz[sp][cam] == 0:
                    log.critical(f"Missing {cam}{sp} for all exposures; Can't fit standard stars")
                    have_all_cameras = False

            if not have_all_cameras:
                num_cmd += 1
                num_err += 1
                continue

            # - NOTE: Saving the joint fit file with only the name of the first exposure
            stdfile = findfile('stdstars', args.night, args.expids[0], spectrograph=sp)
            #stdfile.replace('{:08d}'.format(args.expids[0]),'-'.join(['{:08d}'.format(eid) for eid in args.expids]))
            cmd = "desi_fit_stdstars"
            cmd += " --delta-color 0.1"
            cmd += " --frames {}".format(' '.join(framefiles[sp]))
            cmd += " --skymodels {}".format(' '.join(skyfiles[sp]))
            cmd += " --fiberflats {}".format(' '.join(fiberflatfiles[sp]))
            cmd += " --starmodels {}".format(starmodels)
            cmd += " --outfile {}".format(stdfile)
            if args.maxstdstars is not None:
                cmd += " --maxstdstars {}".format(args.maxstdstars)

            inputs = framefiles[sp] + skyfiles[sp] + fiberflatfiles[sp]
            num_cmd +=1
            if subcomm is None:
                #- Using multiprocessing
                result, success = runcmd(cmd, inputs=inputs, outputs=[stdfile])
            else:
                #- Using MPI, disable multiprocessing
                cmd += " --ncpu 1"
                cmdargs = cmd.split()[1:]
                result, success = runcmd(desispec.scripts.stdstars.main,
                    args=cmdargs, inputs=inputs, outputs=[stdfile], comm=subcomm
                )

                if not success:
                    #- Catches sys.exit from stdstars.main
                    log.error('Rank {} stdstars.main failed for {}'.format(rank, os.path.basename(stdfile)))
                    err = True

            if not success:
                num_err += 1

        timer.stop('stdstarfit')
        num_cmd, num_err = mpi_count_failures(num_cmd, num_err, comm=comm)
        if comm is not None:
            comm.barrier()

        #- all ranks exit with error if any failed
        if num_err > 0:
            if rank == 0:
                log.critical(f'{num_err}/{num_cmd} stdstar ranks failed')
                if num_err==num_cmd:
                    log.critical('All stdstar commands failed')

            sys.stdout.flush()
            sys.exit(1)

        link_errors = 0
        num_link_cmds = 0
        if rank==0 and len(args.expids) > 1:
            for sp in spectro_nums:
                saved_stdfile = findfile('stdstars', args.night, args.expids[0], spectrograph=sp)
                for expid in args.expids[1:]:
                    new_stdfile = findfile('stdstars', args.night, expid, spectrograph=sp)
                    new_dirname, new_fname = os.path.split(new_stdfile)
                    log.debug("Path exists: %s, file exists: %s, link exists: %s",
                            os.path.exists(new_stdfile),
                            os.path.isfile(new_stdfile),
                            os.path.islink(new_stdfile))
                    log.debug('Sym Linking jointly fitted stdstar file: %s '+\
                            'to existing file at %s', new_stdfile, saved_stdfile)
                    num_link_cmds += 1
                    result, success = runcmd(relsymlink, args=(saved_stdfile, new_stdfile), expandargs=True,
                        inputs=[saved_stdfile, ], outputs=[new_stdfile, ])
                    log.debug("Path exists: %s, file exists: %s, link exists: %s",
                            os.path.exists(new_stdfile),
                            os.path.isfile(new_stdfile),
                            os.path.islink(new_stdfile))
                    if not success:
                        link_errors += 1

        if comm is not None:
            link_errors = comm.bcast(link_errors, root=0)
            num_link_cmds = comm.bcast(num_link_cmds, root=0)

        #- all ranks exit with error if any failed
        if link_errors>0:
            if rank == 0:
                log.critical(f'{link_errors}/{num_link_cmds} stdstar link commands failed')
                if link_errors==num_link_cmds:
                    log.critical('All stdstar link commands failed')

            sys.exit(1)

    # -------------------------------------------------------------------------
    # - Wrap up

    # if rank == 0:
    #     report = timer.report()
    #     log.info('Rank 0 timing report:\n' + report)

    if comm is not None:
        timers = comm.gather(timer, root=0)
    else:
        timers = [timer,]

    if rank == 0:
        stats = desiutil.timer.compute_stats(timers)
        log.info('Timing summary statistics:\n' + json.dumps(stats, indent=2))

        if args.timingfile:
            if os.path.exists(args.timingfile):
                with open(args.timingfile) as fx:
                    previous_stats = json.load(fx)

                #- augment previous_stats with new entries, but don't overwrite old
                for name in stats:
                    if name not in previous_stats:
                        previous_stats[name] = stats[name]

                stats = previous_stats

            tmpfile = get_tempfilename(args.timingfile)
            with open(tmpfile, 'w') as fx:
                json.dump(stats, fx, indent=2)
            os.rename(tmpfile, args.timingfile)

        log.info('Timing max duration per step [seconds]:')
        for stepname, steptiming in stats.items():
            tmax = steptiming['duration.max']
            log.info(f'  {stepname:16s} {tmax:.2f}')

    if rank == 0:
        log.info('All done at {}'.format(time.asctime()))

    return 0
