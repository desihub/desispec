#!/usr/bin/env python


import time

import sys, os, argparse, re

import numpy as np
import fitsio
import desispec.io
from desispec.io import findfile
from desispec.io.util import create_camword
# from desispec.calibfinder import findcalibfile

def get_desi_proc_parser():
    """
    Create an argparser object for use with desi_proc based on arguments from sys.argv
    """
    parser = get_shared_desi_proc_parser()
    parser = add_desi_proc_singular_terms(parser)
    return parser

def get_desi_proc_joint_fit_parser():
    """
    Create an argparser object for use with desi_proc_joint_fit based on arguments from sys.argv
    """
    parser = get_shared_desi_proc_parser()
    parser = add_desi_proc_joint_fit_terms(parser)
    return parser

def get_shared_desi_proc_parser():
    """
    Create an argparser object for use with desi_proc AND desi_proc_joint_fit based on arguments from sys.argv
    """
    parser = argparse.ArgumentParser(usage="{prog} [options]")

    parser.add_argument("-n", "--night", type=int, help="YEARMMDD night")
    parser.add_argument("--obstype", type=str, help="science, arc, flat, dark, zero, ...")
    parser.add_argument("--cameras", type=str, help="Explicitly define the spectrographs for which you want" +
                                                    " to reduce the data. Should be a comma separated list." +
                                                    " Numbers only assumes you want to reduce R, B, and Z " +
                                                    "for that spectrograph. Otherwise specify separately [BRZ|brz][0-9].")
    parser.add_argument("--mpi", action="store_true", help="Use MPI parallelism")
    parser.add_argument("--traceshift", action="store_true", help="Do shift traces")
    parser.add_argument('--maxstdstars', type=int, default=None, \
                        help='Maximum number of stdstars to include')
    parser.add_argument("--psf", type=str, required=False, default=None,
                        help="use this input psf (trace shifts will still be computed)")
    parser.add_argument("--fiberflat", type=str, required=False, default=None, help="use this fiberflat")
    parser.add_argument("--calibnight", type=int, required=False, default=None,
                        help="use this night to find nightly PSF and fiberflats")
    parser.add_argument("--scattered-light", action='store_true', help="fit and remove scattered light")
    parser.add_argument("--extra-variance", action='store_true',
                        help='increase sky model variance based on chi2 on sky lines')
    parser.add_argument("--batch", action="store_true", help="Submit a batch job to process this exposure")
    parser.add_argument("--nosubmit", action="store_true", help="Create batch script but don't submit")
    parser.add_argument("-q", "--queue", type=str, default="realtime", help="batch queue to use")
    parser.add_argument("--batch-opts", type=str, default=None, help="additional batch commands")
    parser.add_argument("--runtime", type=int, default=None, help="batch runtime in minutes")
    parser.add_argument("--most-recent-calib", action="store_true", help="If no calibrations exist for the night," +
                        " use the most recent calibrations from *past* nights. If not set, uses default calibs instead.")
    parser.add_argument("--model-pixel-variance", action="store_true",
                        help="Use a model of the variance in preprocessing")
    parser.add_argument("--adjust-sky-wavelength", action="store_true", help="Adjust wavelength of sky lines")
    parser.add_argument("--adjust-sky-lsf", action="store_true", help="Adjust width of sky lines")
    parser.add_argument("--starttime", type=str, help='start time; use "--starttime `date +%%s`"')
    parser.add_argument("--timingfile", type=str, help='save runtime info to this json file; augment if pre-existing')

    return parser

def add_desi_proc_singular_terms(parser):
    """
    Add parameters to the argument parser that are only used by desi_proc
    """
    #parser.add_argument("-n", "--night", type=int, help="YEARMMDD night")
    parser.add_argument("-e", "--expid", type=int, help="Exposure ID")
    parser.add_argument("-i", "--input", type=str, help="input raw data file")

    parser.add_argument("--fframe", action="store_true", help="Also write non-sky subtracted fframe file")
    parser.add_argument("--nofiberflat", action="store_true", help="Do not apply fiberflat")
    parser.add_argument("--noskysub", action="store_true",
                        help="Do not subtract the sky. Also skips stdstar fit and flux calib")
    parser.add_argument("--noprestdstarfit", action="store_true",
                        help="Do not do any science reductions prior to stdstar fitting")
    parser.add_argument("--nostdstarfit", action="store_true", help="Do not fit standard stars")
    parser.add_argument("--nofluxcalib", action="store_true", help="Do not flux calibrate")

    return parser

def add_desi_proc_joint_fit_terms(parser):
    """
    Add parameters to the argument parser that are only used by desi_proc_joint_fit
    """
    #parser.add_argument("-n", "--nights", type=str, help="YEARMMDD nights")
    parser.add_argument("-e", "--expids", type=str, help="Exposure IDs")
    parser.add_argument("-i", "--inputs", type=str, help="input raw data files")

    return parser

def assign_mpi(do_mpi, do_batch, log):
    """
    Based on whether the mpi flag is set and whether the batch flag is set, assign the appropriate
    MPI values for the communicator, rank, and number of ranks. Also verify that environment 
    variables are set and provide useful information via the log

    Args:
        do_mpi: bool, whether to use mpi or not
        do_batch: bool, whether the script is meant to write a batch script and exit or not.
        log: desi log object for reporting

    Returns:
        comm: MPI communicator object
        rank: int, the numeric number assigned to the currenct MPI rank
        size: int, the total number of MPI ranks
    """
    if do_mpi and not do_batch:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        comm = None
        rank = 0
        size = 1

    #- Prevent MPI from killing off spawned processes
    if 'PMI_MMAP_SYNC_WAIT_TIME' not in os.environ:
        os.environ['PMI_MMAP_SYNC_WAIT_TIME'] = '3600'

    #- Double check env for MPI+multiprocessing at NERSC
    if 'MPICH_GNI_FORK_MODE' not in os.environ:
        os.environ['MPICH_GNI_FORK_MODE'] = 'FULLCOPY'
        if rank == 0:
            log.info('Setting MPICH_GNI_FORK_MODE=FULLCOPY for MPI+multiprocessing')
    elif os.environ['MPICH_GNI_FORK_MODE'] != 'FULLCOPY':
        gnifork = os.environ['MPICH_GNI_FORK_MODE']
        if rank == 0:
            log.error(f'MPICH_GNI_FORK_MODE={gnifork} is not "FULLCOPY"; this might not work with MPI+multiprocessing, but not overriding')
    elif rank == 0:
        log.debug('MPICH_GNI_FORK_MODE={}'.format(os.environ['MPICH_GNI_FORK_MODE']))
        
    return comm, rank, size

def load_raw_data_header(pathname, return_filehandle=False):
    """
    Open raw desi data file given at pathname and return the spectrograph header,
    which varied in name over time

    Args:
        pathname: str, the full path to the raw data file
        return_filehandle: bool, whether to return the open file handle or to close
                                 it and only return the header. Default is False.
    Returns:
        hdr: fitsio header object from the file at pathname
        fx:  optional, fitsio file object returned only if return_filehandle is True 
    """
    # - Fill in values from raw data header if not overridden by command line
    fx = fitsio.FITS(pathname)
    if 'SPEC' in fx:  # - 20200225 onwards
        # hdr = fits.getheader(args.input, 'SPEC')
        hdr = fx['SPEC'].read_header()
    elif 'SPS' in fx:  # - 20200224 and before
        # hdr = fits.getheader(args.input, 'SPS')
        hdr = fx['SPS'].read_header()
    else:
        # hdr = fits.getheader(args.input, 0)
        hdr = fx[0].read_header()

    if return_filehandle:
        return hdr, fx
    else:
        fx.close()
        return hdr
    
def update_args_with_headers(args):
    """
    Update input argparse object with values from header if the argparse values are uninformative defaults (generally
    python None). This also returns the primary header and each camera's header.

    Args:
        args: argparse arguments object. Parsed from the command line arguments based on the parser defined
                  by the function get_desi_proc_parser().

    Returns:
        args: modified version of the input args where values have been updated if None using information from
                  appropriate headers using either night+expid or an input file.
        hdr: fitsio header object obtained using *.read_header() on input file or file determined from args information.
        camhdr: dict, dictionary of fitsio header objects for each camera in the input files.

    Note:
        The input args is modified and returned here.
    """
    if args.input is None:
        if args.night is None or args.expid is None:
            raise RuntimeError('Must specify --input or --night AND --expid')

        args.input = findfile('raw', night=args.night, expid=args.expid)

    if not os.path.isfile(args.input):
        raise IOError('Missing input file: {}'.format(args.input))

    hdr, fx = load_raw_data_header(pathname=args.input, return_filehandle=True)

    if args.expid is None:
        args.expid = int(hdr['EXPID'])

    if args.night is None:
        args.night = int(hdr['NIGHT'])

    if args.obstype is None:
        if 'OBSTYPE' in hdr:
            args.obstype = hdr['OBSTYPE'].strip()
        elif 'FLAVOR' in hdr:
            args.obstype = hdr['FLAVOR'].strip()
            log.warning('Using OBSTYPE={} from FLAVOR keyword'.format(args.obstype))
        else:
            raise RuntimeError('Need --obstype or OBSTYPE or FLAVOR header keywords')

    if args.cameras is None:
        recam = re.compile('^[brzBRZ][\d]$')
        cameras = list()
        for hdu in fx.hdu_list:
            if recam.match(hdu.get_extname()):
                cameras.append(hdu.get_extname().lower())

        if len(cameras) == 0:
            raise RuntimeError("No [BRZ][0-9] camera HDUs found in {}".format(args.input))

        args.cameras = cameras
        cameras = None
    else:
        cam_str = args.cameras.strip(' \t').lower()
        cameras = cam_str.split(',')
        if cam_str[0] not in ['b', 'r', 'z'] and cameras[0].isnumeric():
            args.cameras = []
            for camnum in cameras:
                for ccd in ['b', 'r', 'z']:
                    args.cameras.append('{}{}'.format(ccd, camnum))
        else:
            args.cameras = cameras

    # - Update args to be in consistent format
    args.cameras = sorted(args.cameras)
    args.obstype = args.obstype.upper()
    args.night = int(args.night)
    if args.batch_opts is not None:
        args.batch_opts = args.batch_opts.strip('"\'')

    camhdr = dict()
    for cam in args.cameras:
        camhdr[cam] = fx[cam].read_header()

    fx.close()
    return args, hdr, camhdr

def determine_resources(ncameras, jobdesc, queue, nexps=1, forced_runtime=None):
    """
    Determine the resources that should be assigned to the batch script given what
    desi_proc needs for the given input information.

    Args:
        ncameras: int, number of cameras to be processed
        jobdesc: str, type of data being processed
        queue: str, the queue at NERSC to be submitted to. 'realtime' will force
                    restrictions on number of nodes.
        force_runtime: int, the amount of runtime in minutes to allow for the script. Should be left
                            to default heuristics unless needed for some reason.

    Returns:
        ncores: int, number of cores (actually 2xphysical cores) that should be submitted via "-n {ncores}"
        nodes:  int, number of nodes to be requested in the script. Typically  (ncores-1) // 32 + 1
        runtime: int, the max time requested for the script in minutes for the processing.
    """
    nspectro = (ncameras - 1) // 3 + 1
    if jobdesc in ('ARC', 'TESTARC'):
        ncores, runtime = 20 * ncameras, 35
    elif jobdesc in ('FLAT', 'TESTFLAT'):
        ncores, runtime = 20 * nspectro, 15
    elif jobdesc in ('SKY', 'TWILIGHT', 'SCIENCE','PRESTDSTAR','POSTSTDSTAR'):
        ncores, runtime = 20 * nspectro, 30
    elif jobdesc in ('ZERO', 'DARK'):
        ncores, runtime = 2, 5
    elif jobdesc == 'PSFNIGHT':
        ncores, runtime = 20 * nspectro, 5 #ncameras, 5
    elif jobdesc == 'NIGHTLYFLAT':
        ncores, runtime = 20 * nspectro, 20 #ncameras, 5
    elif jobdesc in ('STDSTARFIT'):
        ncores, runtime = 20 * ncameras, (6+2*nexps) #ncameras, 10
    else:
        print('ERROR: unknown jobdesc={}'.format(jobdesc))
        sys.exit(1)

    if forced_runtime is not None:
        runtime = forced_runtime

    nodes = (ncores - 1) // 32 + 1

    # - Arcs and flats make good use of full nodes, but throttle science
    # - exposures to 5 nodes to enable two to run together in the 10-node
    # - realtime queue, since their wallclock is dominated by less
    # - efficient sky and fluxcalib steps
    if jobdesc in ('ARC', 'TESTARC', 'FLAT', 'TESTFLAT'):
        max_realtime_nodes = 10
    else:
        max_realtime_nodes = 5

    if (queue == 'realtime') and (nodes > max_realtime_nodes):
        nodes = max_realtime_nodes
        ncores = 32 * nodes

    return ncores, nodes, runtime

def create_desi_proc_batch_script(night, exp, cameras, jobdesc, queue, runtime=None, batch_opts=None,\
                                  timingfile=None, batchdir=None, jobname=None, cmdline=None):
    """
    Generate a SLURM batch script to be submitted to the slurm scheduler to run desi_proc.

    Args:
        night: str or int. The night the data was acquired
        exp: str, int, or list of ints. The exposure id(s) for the data.
        cameras: list of str. List of cameras to include in the processing.
        jobdesc: str. Description of the job to be performed. Used to determine requested resources
                      and whether to operate in a more mpi parallelism (all except poststdstar) or less (only poststdstar).
                      Directly relate to the obstype, with science exposures being split into two (pre, post)-stdstar,
                         and adding joint fit categories stdstarfit, psfnight, and nightlyflat.
                      Options include:
                     'prestdstar', 'poststdstar', 'stdstarfit', 'arc', 'flat', 'psfnight', 'nightlyflat'
        queue: str. Queue to be used.
        runtime: str. Timeout wall clock time.
        batch_opts: str. Other options to give to the slurm batch scheduler (written into the script).
        timingfile: str. Specify the name of the timing file.
        batchdir: str. Specify where the batch file will be written.
        jobname: str. Specify the name of the slurm script written.
        cmdline: str. Complete command as would be given in terminal to run the desi_proc. Can be used instead
                      of reading from argv.
        batchdir: can define an alternative location to write the file. The default is to SPECPROD under run/scripts/night/NIGHT
        jobname: name to save this batch script file as and the name of the eventual log file. Script is save  within
                 the batchdir directory.

    Returns:
        scriptfile: the full path name for the script written.

    Note:
        batchdir and jobname can be used to define an alternative pathname, but may not work with assumptions in desi_proc.
            These optional arguments should be used with caution and primarily for debugging.
    """

    if batchdir is None:
        reduxdir = desispec.io.specprod_root()
        batchdir = os.path.join(reduxdir, 'run', 'scripts', 'night', str(night))

    os.makedirs(batchdir, exist_ok=True)

    if jobname is None:
        camword = create_camword(cameras)
        if type(exp) is not str:
            if np.isscalar(exp):
                expstr = '{:08d}'.format(exp)
            else:
                #expstr = '-'.join(['{:08d}'.format(curexp) for curexp in exp])
                expstr = '{:08d}'.format(exp[0])
        else:
            expstr = exp
        jobname = '{}-{}-{}-{}'.format(jobdesc.lower(), night, expstr, camword)

    if timingfile is None:
        timingfile = f'{jobname}-timing-$SLURM_JOBID.json'
        timingfile = os.path.join(batchdir, timingfile)

    scriptfile = os.path.join(batchdir, jobname + '.slurm')

    ncameras = len(cameras)
    nexps = 1
    if not np.isscalar(exp) and type(exp) is not str:
        nexps = len(exp)
    ncores, nodes, runtime = determine_resources(ncameras, jobdesc.upper(), queue=queue, nexps=nexps, forced_runtime=runtime)

    assert runtime <= 60

    with open(scriptfile, 'w') as fx:
        fx.write('#!/bin/bash -l\n\n')
        fx.write('#SBATCH -C haswell\n')
        fx.write('#SBATCH -N {}\n'.format(nodes))
        fx.write('#SBATCH -n {}\n'.format(ncores))
        fx.write('#SBATCH --qos {}\n'.format(queue))
        if batch_opts is not None:
            fx.write('#SBATCH {}\n'.format(batch_opts))
        fx.write('#SBATCH --account desi\n')
        fx.write('#SBATCH --job-name {}\n'.format(jobname))
        fx.write('#SBATCH --output {}/{}-%j.log\n'.format(batchdir, jobname))
        fx.write('#SBATCH --time=00:{:02d}:00\n'.format(runtime))

        # - If we are asking for more than half the node, ask for all of it
        # - to avoid memory problems with other people's jobs
        if ncores > 16:
            fx.write('#SBATCH --exclusive\n')

        fx.write('\n')

        if cmdline is None:
            inparams = list(sys.argv).copy()
        else:
            inparams = cmdline.split(' ')[1:]
        for parameter in ['-q', '--queue', '--batch-opts']:
            if parameter in inparams:
                loc = np.where(np.array(inparams) == parameter)[0][0]
                # Remove the command
                inparams.pop(loc)
                # Remove the argument of the command (now in the command location after pop)
                inparams.pop(loc)


        cmd = ' '.join(inparams)
        cmd = cmd.replace(' --batch', ' ').replace(' --nosubmit', ' ')
        cmd += f' --timingfile {timingfile}'

        if '--mpi' not in cmd:
            cmd += ' --mpi'

        fx.write(f'# {jobdesc} exposure with {ncameras} cameras\n')
        fx.write(f'# using {ncores} cores on {nodes} nodes\n\n')

        fx.write('echo Starting at $(date)\n')

        if jobdesc.lower() not in ['science', 'prestdstar', 'stdstarfit', 'poststdstar']:
            ################################## Note ############################
            ## Needs to be refactored to write the correct thing given flags ###
            ####################################################################
            fx.write('\n# Do steps at full MPI parallelism\n')
            srun = 'srun -N {} -n {} -c 2 {}'.format(nodes, ncores, cmd)
            fx.write('echo Running {}\n'.format(srun))
            fx.write('{}\n'.format(srun))
        else:
            if jobdesc.lower() in ['science','prestdstar']:
                ################################## Note ############################
                ## Needs to be refactored to write the correct thing given flags ###
                ####################################################################
                fx.write('\n# Do steps through skysub at full MPI parallelism\n')
                srun = 'srun -N {} -n {} -c 2 {} --nofluxcalib'.format(nodes, ncores, cmd)
                fx.write('echo Running {}\n'.format(srun))
                fx.write('{}\n'.format(srun))
            if jobdesc.lower() in ['science', 'stdstarfit', 'poststdstar']:
                fx.write('\n# Use less MPI parallelism for fluxcalib MP parallelism\n')
                fx.write('# This should quickly skip over the steps already done\n')
                srun = 'srun -N {} -n {} -c 32 {} '.format(nodes, nodes * 2, cmd)
                fx.write('if [ $? -eq 0 ]; then\n')
                fx.write('  echo Running {}\n'.format(srun))
                fx.write('  {}\n'.format(srun))
                fx.write('else\n')
                fx.write('  echo FAILED: done at $(date)\n')
                fx.write('  exit 1\n')
                fx.write('fi\n')

        fx.write('\nif [ $? -eq 0 ]; then\n')
        fx.write('  echo SUCCESS: done at $(date)\n')
        fx.write('else\n')
        fx.write('  echo FAILED: done at $(date)\n')
        fx.write('  exit 1\n')
        fx.write('fi\n')

    print('Wrote {}'.format(scriptfile))
    print('logfile will be {}/{}-JOBID.log\n'.format(batchdir, jobname))

    return scriptfile

def find_most_recent(night, file_type='psfnight', n_nights=30):
    '''
       Searches back in time for either psfnight or fiberflatnight (or anything supported by
       desispec.calibfinder.findcalibfile.

       Inputs:
         night : str YYYMMDD   night to look back from
         file_type : str psfnight or fiberflatnight
         n_nights : int  number of nights to step back before giving up

      returns:
         nightfile : str Full pathname to calibration file of interest.
                     If none found, None is returned.

    '''
    # Set the time as Noon on the day in question
    today = time.strptime('{} 12'.format(night),'%Y%m%d %H')
    one_day = 60*60*24 # seconds

    test_night_struct = today

    # Search a month in the past
    for daysback in range(n_nights) :
        test_night_struct = time.strptime(time.ctime(time.mktime(test_night_struct)-one_day))
        test_night_str = time.strftime('%Y%m%d', test_night_struct)
        nightfile = findfile(file_type, test_night_str, '00000000', camera)
        if os.path.isfile(nightfile) :
            return nightfile

    return None
