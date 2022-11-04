#!/usr/bin/env python


import time

import sys, os, argparse, re

import numpy as np
import fitsio
import desispec.io
from desispec.io import findfile
from desispec.io.meta import get_desi_root_readonly
from desispec.io.util import create_camword, decode_camword, parse_cameras
# from desispec.calibfinder import findcalibfile
from desiutil.log import get_logger

from . import batch

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

def get_desi_proc_tilenight_parser():
    """
    Create an argparser object for use with desi_proc_tilenight based on arguments from sys.argv
    """
    parser = get_shared_desi_proc_parser()
    parser = add_desi_proc_tilenight_terms(parser)
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
    parser.add_argument("--traceshift", action="store_true", help="(deprecated)")
    parser.add_argument("--no-traceshift", action="store_true", help="Do not shift traces")
    parser.add_argument('--maxstdstars', type=int, default=None, \
                        help='Maximum number of stdstars to include')
    parser.add_argument("--psf", type=str, required=False, default=None,
                        help="use this input psf (trace shifts will still be computed)")
    parser.add_argument("--fiberflat", type=str, required=False, default=None, help="use this fiberflat")
    parser.add_argument("--calibnight", type=int, required=False, default=None,
                        help="use this night to find nightly PSF and fiberflats")
    parser.add_argument("--scattered-light", action='store_true', help="fit and remove scattered light")
    parser.add_argument("--no-bkgsub", action='store_true', help="disable CCD bkg fit between fiber blocks")
    parser.add_argument("--no-extra-variance", action='store_true',
                        help='disable increase sky model variance based on chi2 on sky lines')
    parser.add_argument("--batch", action="store_true", help="Submit a batch job to process this exposure")
    parser.add_argument("--nosubmit", action="store_true", help="Create batch script but don't submit")
    parser.add_argument("-q", "--queue", type=str, default="realtime", help="batch queue to use")
    parser.add_argument("--batch-opts", type=str, default=None, help="additional batch commands")
    parser.add_argument("--runtime", type=int, default=None, help="batch runtime in minutes")
    parser.add_argument("--most-recent-calib", action="store_true", help="If no calibrations exist for the night," +
                        " use the most recent calibrations from *past* nights. If not set, uses default calibs instead.")
    parser.add_argument("--no-model-pixel-variance", action="store_true",
                        help="Do not use a model of the variance in preprocessing")
    parser.add_argument("--no-sky-wavelength-adjustment", action="store_true", help="Do not adjust wavelength of sky lines")
    parser.add_argument("--no-sky-lsf-adjustment", action="store_true", help="Do not adjust width of sky lines")
    parser.add_argument("--adjust-sky-with-more-fibers", action="store_true", help="Use more fibers than just the sky fibers for the adjustements")
    parser.add_argument("--save-sky-adjustments", action="store_true", help="Save sky adjustment terms (wavelength and LSF)")
    parser.add_argument("--starttime", type=str, help='start time; use "--starttime `date +%%s`"')
    parser.add_argument("--timingfile", type=str, help='save runtime info to this json file; augment if pre-existing')
    parser.add_argument("--no-xtalk", action="store_true", help='disable fiber crosstalk correction')
    parser.add_argument("--system-name", type=str, default=batch.default_system(), help='Batch system name (cori-haswell, perlmutter-gpu, ...)')
    parser.add_argument("--extract-subcomm-size", type=int, default=None, help="Size to use for GPU extract subcomm")
    parser.add_argument("--gpuspecter", action="store_true", help="Use GPU specter")
    parser.add_argument("--gpuextract", action="store_true", help="Use GPU extraction")
    parser.add_argument("--mpistdstars", action="store_true", help="Use MPI parallelism in stdstar fitting instead of multiprocessing")
    parser.add_argument("--no-skygradpca", action="store_true", help="Do not fit sky gradient")
    parser.add_argument("--no-tpcorrparam", action="store_true", help="Do not apply tpcorrparam spatial model or fit tpcorrparam pca terms")
    parser.add_argument("--no-barycentric-correction", action="store_true", help="Do not apply barycentric correction to wavelength")
    parser.add_argument("--apply-sky-throughput-correction", action="store_true", help="Apply throughput correction to sky fibers (default: do not apply!)")
    return parser


def add_desi_proc_singular_terms(parser):
    """
    Add parameters to the argument parser that are only used by desi_proc
    """
    #parser.add_argument("-n", "--night", type=int, help="YEARMMDD night")
    parser.add_argument("-e", "--expid", type=int, default=None, help="Exposure ID")
    parser.add_argument("-i", "--input", type=str, default=None, help="input raw data file")
    parser.add_argument("--badamps", type=str, default=None, help="comma separated list of {camera}{petal}{amp}"+\
                                                                  ", i.e. [brz][0-9][ABCD]. Example: 'b7D,z8A'")
    parser.add_argument("--fframe", action="store_true", help="Also write non-sky subtracted fframe file")
    parser.add_argument("--nofiberflat", action="store_true", help="Do not apply fiberflat")
    parser.add_argument("--noskysub", action="store_true",
                        help="Do not subtract the sky. Also skips stdstar fit and flux calib")
    parser.add_argument("--noprestdstarfit", action="store_true",
                        help="Do not do any science reductions prior to stdstar fitting")
    parser.add_argument("--nostdstarfit", action="store_true", help="Do not fit standard stars")
    parser.add_argument("--nofluxcalib", action="store_true", help="Do not flux calibrate")
    parser.add_argument("--nightlybias", action="store_true", help="Create nightly bias model from ZEROs")
    return parser

def add_desi_proc_joint_fit_terms(parser):
    """
    Add parameters to the argument parser that are only used by desi_proc_joint_fit
    """
    #parser.add_argument("-n", "--nights", type=str, help="YEARMMDD nights")
    parser.add_argument("-e", "--expids", type=str, help="Exposure IDs")
    parser.add_argument("-i", "--inputs", type=str, help="input raw data files")
    return parser

def add_desi_proc_tilenight_terms(parser):
    """
    Add parameters to the argument parser that are only used by desi_proc_tilenight
    """
    parser.add_argument("-t", "--tileid", type=str, help="Tile ID")
    parser.add_argument("-d", "--dryrun", action="store_true", help="show commands only, do not run")

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

def cameras_from_raw_data(rawdata):
    """
    Takes a filepath or fitsio FITS object corresponding to a DESI raw data file
    and returns a list of cameras for which data exists in the file.

    Args:
        rawdata, str or fitsio.FITS object. The input raw desi data file. str must be a full file path. Otherwise
                                            it must be a fitsio.FITS object.
    Returns:
        cameras, str. The list of cameras that have data in the given file.
    """
    ## Be flexible on whether input is filepath or a filehandle
    if type(rawdata) is str:
        if os.path.isfile(rawdata):
            fx = fitsio.FITS(rawdata)
        else:
            raise IOError(f"File {rawdata} doesn't exist.")
    else:
        fx = rawdata

    recam = re.compile('^[brzBRZ][\d]$')
    cameras = list()
    for hdu in fx.hdu_list:
        if recam.match(hdu.get_extname()):
            cameras.append(hdu.get_extname().lower())
    return cameras

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
    log = get_logger()
    if args.input is None:
        if args.night is None or args.expid is None:
            raise RuntimeError('Must specify --input or --night AND --expid')

        args.input = findfile('raw', night=args.night, expid=args.expid)

    if not os.path.isfile(args.input):
        raise IOError('Missing input file: {}'.format(args.input))

    log.info(f'Loading header keywords from {args.input}')
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
        cameras = cameras_from_raw_data(fx)
        if len(cameras) == 0:
            raise RuntimeError("No [BRZ][0-9] camera HDUs found in {}".format(args.input))

        args.cameras = cameras
    else:
        camword = parse_cameras(args.cameras)
        args.cameras = decode_camword(camword)

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

def determine_resources(ncameras, jobdesc, queue, nexps=1, forced_runtime=None, system_name=None):
    """
    Determine the resources that should be assigned to the batch script given what
    desi_proc needs for the given input information.

    Args:
        ncameras: int, number of cameras to be processed
        jobdesc: str, type of data being processed
        queue: str, the queue at NERSC to be submitted to. 'realtime' will force
                    restrictions on number of nodes.

    Options:
        nexps: int, the number of exposures processed in this step
        force_runtime: int, the amount of runtime in minutes to allow for the script. Should be left
                            to default heuristics unless needed for some reason.
        system_name: str, batch compute system, e.g. cori-haswell or perlmutter-gpu

    Returns:
        ncores: int, number of cores (actually 2xphysical cores) that should be submitted via "-n {ncores}"
        nodes:  int, number of nodes to be requested in the script. Typically  (ncores-1) // cores_per_node + 1
        runtime: int, the max time requested for the script in minutes for the processing.
    """
    if system_name is None:
        system_name = batch.default_system(jobdesc=jobdesc)

    config = batch.get_config(system_name)
    log = get_logger()
    jobdesc = jobdesc.upper()

    nspectro = (ncameras - 1) // 3 + 1
    nodes = None
    if jobdesc in ('ARC', 'TESTARC'):
        ncores          = 20 * (10*(ncameras+1)//20) # lowest multiple of 20 exceeding 10 per camera
        ncores, runtime = ncores + 1, 45             # + 1 for worflow.schedule scheduler proc
    elif jobdesc in ('FLAT', 'TESTFLAT'):
        runtime = 25
        if system_name[0:10] == 'perlmutter':
            ncores = config['cores_per_node']
        else:
            ncores = 20 * nspectro
    elif jobdesc == 'TILENIGHT':
        runtime  = int(60. / 140. * ncameras * nexps) # 140 frames per node hour
        runtime += 20                                 # overhead
        ncores = config['cores_per_node']
        if system_name[0:10] != 'perlmutter':
            msg = 'tilenight cannot run on system_name={}'.format(system_name)
            log.critical(msg)
            raise ValueError(msg)
    elif jobdesc in ('SKY', 'TWILIGHT', 'SCIENCE','PRESTDSTAR'):
        runtime = 30
        if system_name[0:10] == 'perlmutter':
            ncores = config['cores_per_node']
        else:
            ncores = 20 * nspectro
    elif jobdesc in ('DARK', 'BADCOL'):
        ncores, runtime = ncameras, 5
    elif jobdesc == 'CCDCALIB':
        ncores, runtime = ncameras, 5
    elif jobdesc == 'ZERO':
        ncores, runtime = 2, 5
    elif jobdesc == 'PSFNIGHT':
        ncores, runtime = ncameras, 5
    elif jobdesc == 'NIGHTLYFLAT':
        ncores, runtime = ncameras, 5
    elif jobdesc == 'STDSTARFIT':
        #- Special case hardcode: stdstar parallelism maxes out at ~30 cores
        #- and on KNL, it OOMs above that anyway.
        #- This might be more related to using a max of 30 standards, not that
        #- there are 30 cameras (coincidence).
        #- Use 32 as power of 2 for core packing
        ncores = 32
        runtime = 8+2*nexps
    elif jobdesc == 'POSTSTDSTAR':
        runtime = 10
        ncores = ncameras
    elif jobdesc == 'NIGHTLYBIAS':
        ncores, runtime = 15, 5
        nodes = 2
    else:
        msg = 'unknown jobdesc={}'.format(jobdesc)
        log.critical(msg)
        raise ValueError(msg)

    if forced_runtime is not None:
        runtime = forced_runtime

    if nodes is None:
        nodes = (ncores - 1) // config['cores_per_node'] + 1

    # - Arcs and flats make good use of full nodes, but throttle science
    # - exposures to 5 nodes to enable two to run together in the 10-node
    # - realtime queue, since their wallclock is dominated by less
    # - efficient sky and fluxcalib steps
    if jobdesc in ('ARC', 'TESTARC'):#, 'FLAT', 'TESTFLAT'):
        max_realtime_nodes = 10
    else:
        max_realtime_nodes = 5

    #- Pending further optimizations, use same number of nodes in all queues
    ### if (queue == 'realtime') and (nodes > max_realtime_nodes):
    if (nodes > max_realtime_nodes):
        nodes = max_realtime_nodes
        ncores = config['cores_per_node'] * nodes
        if jobdesc in ('ARC', 'TESTARC'):
            # adjust for workflow.schedule scheduler proc
            ncores = ((ncores - 1) // 20) * 20 + 1

    #- Allow KNL jobs to be slower than Haswell,
    #- except for ARC so that we don't have ridiculously long times
    #- (Normal arc is still ~15 minutes, albeit with a tail)
    if jobdesc not in ['ARC', 'TESTARC']:
        runtime *= config['timefactor']

    #- Do not allow runtime to be less than 5 min
    if runtime < 5:
        runtime = 5

    #- Add additional overhead factor if needed
    if 'NERSC_RUNTIME_OVERHEAD' in os.environ:
        t = os.environ['NERSC_RUNTIME_OVERHEAD']
        log.info(f'Adding $NERSC_RUNTIME_OVERHEAD={t} minutes to batch runtime request')
        runtime += float(runtime)

    return ncores, nodes, runtime

def get_desi_proc_batch_file_path(night,reduxdir=None):
    """
    Returns the default directory location to store a batch script file given a night

    Args:
        night: str or int, defines the night (should be 8 digits)
        reduxdir: str (optional), define the base directory where the /run/scripts directory should or does live.

    Returns:
        batchdir: str, the default location where a batch script file should be written
    """
    if reduxdir is None:
        reduxdir = desispec.io.specprod_root()
    batchdir = os.path.join(reduxdir, 'run', 'scripts', 'night', str(night))
    return batchdir

def get_desi_proc_tilenight_batch_file_name(night, tileid):
    """
    Returns the filename for a tilenight batch script file given a night and tileid

    Args:
        night: str or int, defines the night (should be 8 digits)
        tileid: str or int, defines the tile id relevant to the job

    Returns:
        pathname: str, the default script name for a desi_proc_tilenight batch script file
    """
    if type(tileid) is not str:
        if np.isscalar(tileid):
            tileid = '{}'.format(tileid)
        else:
            raise RuntimeError('tileid should be either int or str')

    jobname = 'tilenight-{}-{}'.format(night, tileid)
    return jobname

def get_desi_proc_batch_file_name(night, exp, jobdesc, cameras):
    """
    Returns the default directory location to store a batch script file given a night

    Args:
        night: str or int, defines the night (should be 8 digits)
        exp: str, int, or array of ints, defines the exposure id(s) relevant to the job
        jobdesc: str, type of data being processed
        cameras: str or list of str. If str, must be camword, If list, must be list of cameras to include in the processing.

    Returns:
        pathname: str, the default script name for a desi_proc batch script file
    """
    camword = parse_cameras(cameras)
    if type(exp) is not str:
        if exp is None:
            expstr = 'none'
        elif np.isscalar(exp):
            expstr = '{:08d}'.format(exp)
        else:
            #expstr = '-'.join(['{:08d}'.format(curexp) for curexp in exp])
            expstr = '{:08d}'.format(exp[0])
    else:
        expstr = exp
    jobname = '{}-{}-{}-{}'.format(jobdesc.lower(), night, expstr, camword)
    return jobname

def get_desi_proc_batch_file_pathname(night, exp, jobdesc, cameras, reduxdir=None):
    """
    Returns the default directory location to store a batch script file given a night

    Args:
        night: str or int, defines the night (should be 8 digits)
        exp: str, int, or array of ints, defines the exposure id(s) relevant to the job
        jobdesc: str, type of data being processed
        cameras: str or list of str. If str, must be camword, If list, must be list of cameras to include in the processing.
        reduxdir: str (optional), define the base directory where the /run/scripts directory should or does live

    Returns:
        pathname: str, the default location and script name for a desi_proc batch script file
    """
    path = get_desi_proc_batch_file_path(night,reduxdir=reduxdir)
    name = get_desi_proc_batch_file_name(night, exp, jobdesc, cameras)
    return os.path.join(path, name)

def get_desi_proc_tilenight_batch_file_pathname(night, tileid, reduxdir=None):
    """
    Returns the default directory location to store a tilenight batch script file given a night and tileid

    Args:
        night: str or int, defines the night (should be 8 digits)
        tileid: str or int, defines the tile id relevant to the job
        reduxdir: str (optional), define the base directory where the /run/scripts directory should or does live

    Returns:
        pathname: str, the default location and script name for a desi_proc_tilenight batch script file
    """
    path = get_desi_proc_batch_file_path(night,reduxdir=reduxdir)
    name = get_desi_proc_tilenight_batch_file_name(night,tileid)
    return os.path.join(path, name)

def create_desi_proc_batch_script(night, exp, cameras, jobdesc, queue, runtime=None, batch_opts=None,\
                                  timingfile=None, batchdir=None, jobname=None, cmdline=None, system_name=None,
                                  gpuspecter=False, gpuextract=False):
    """
    Generate a SLURM batch script to be submitted to the slurm scheduler to run desi_proc.

    Args:
        night: str or int. The night the data was acquired
        exp: str, int, or list of ints. The exposure id(s) for the data.
        cameras: str or list of str. List of cameras to include in the processing.
        jobdesc: str. Description of the job to be performed. Used to determine requested resources
                      and whether to operate in a more mpi parallelism (all except poststdstar) or less (only poststdstar).
                      Directly relate to the obstype, with science exposures being split into two (pre, post)-stdstar,
                         and adding joint fit categories stdstarfit, psfnight, and nightlyflat.
                      Options include:
                     'prestdstar', 'poststdstar', 'stdstarfit', 'arc', 'flat', 'psfnight', 'nightlyflat'
        queue: str. Queue to be used.

    Options:
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
        system_name: name of batch system, e.g. cori-haswell, cori-knl
        gpuspecter: bool. Whether to use gpu_specter.
        gpuextract: bool. Whether to perform gpu extraction with gpu_specter.

    Returns:
        scriptfile: the full path name for the script written.

    Note:
        batchdir and jobname can be used to define an alternative pathname, but may not work with assumptions in desi_proc.
            These optional arguments should be used with caution and primarily for debugging.
    """
    if np.isscalar(cameras):
        camword = cameras
        cameras = decode_camword(camword)

    if batchdir is None:
        batchdir = get_desi_proc_batch_file_path(night)

    os.makedirs(batchdir, exist_ok=True)

    if jobname is None:
        jobname = get_desi_proc_batch_file_name(night, exp, jobdesc, cameras)

    if timingfile is None:
        timingfile = f'{jobname}-timing-$SLURM_JOBID.json'
        timingfile = os.path.join(batchdir, timingfile)

    scriptfile = os.path.join(batchdir, jobname + '.slurm')

    ## If system name isn't specified, guess it
    if system_name is None:
        system_name = batch.default_system(jobdesc=jobdesc)

    batch_config = batch.get_config(system_name)
    threads_per_core = batch_config['threads_per_core']
    gpus_per_node = batch_config['gpus_per_node']
    ncameras = len(cameras)
    nexps = 1
    if exp is not None and not np.isscalar(exp) and type(exp) is not str:
        nexps = len(exp)

    ncores, nodes, runtime = determine_resources(
            ncameras, jobdesc.upper(), queue=queue, nexps=nexps,
            forced_runtime=runtime, system_name=system_name)

    #- derive from cmdline or sys.argv whether this is a nightlybias job
    nightlybias = False
    if cmdline is not None:
        if '--nightlybias' in cmdline:
            nightlybias = True
    elif '--nightlybias' in sys.argv:
        nightlybias = True

    #- nightlybias jobs are memory limited, so throttle number of ranks
    if nightlybias:
        tot_threads = batch_config['threads_per_core'] * batch_config['cores_per_node']
        bias_threads_per_core = tot_threads // 8

        bias_cores, bias_nodes, bias_runtime = determine_resources(
                ncameras, 'NIGHTLYBIAS', queue=queue, nexps=nexps,
                system_name=system_name)

        nodes = max(nodes, bias_nodes)
        runtime += bias_runtime

    #- arc fits require 3.2 GB of memory per bundle, so increase nodes as needed
    if jobdesc.lower() == 'arc':
        cores_per_node = (ncores-1) // nodes + ((ncores-1) % nodes > 0)
        mem_per_node = float(batch_config['memory'])
        mem_per_core = mem_per_node / cores_per_node
        while mem_per_core < 3.2:
            nodes += 1
            cores_per_node = (ncores-1) // nodes + ((ncores-1) % nodes > 0)
            mem_per_core = mem_per_node / cores_per_node
        threads_per_node = batch_config['threads_per_core'] * batch_config['cores_per_node']
        threads_per_core = (threads_per_node * nodes) // ncores

    runtime_hh = int(runtime // 60)
    runtime_mm = int(runtime % 60)

    with open(scriptfile, 'w') as fx:
        fx.write('#!/bin/bash -l\n\n')
        fx.write('#SBATCH -N {}\n'.format(nodes))
        fx.write('#SBATCH --qos {}\n'.format(queue))
        for opts in batch_config['batch_opts']:
            fx.write('#SBATCH {}\n'.format(opts))
        if batch_opts is not None:
            fx.write('#SBATCH {}\n'.format(batch_opts))
        if system_name == 'perlmutter-gpu':
            # perlmutter-gpu requires projects name with "_g" appended
            fx.write('#SBATCH --account desi_g\n')
        else:
            fx.write('#SBATCH --account desi\n')
        fx.write('#SBATCH --job-name {}\n'.format(jobname))
        fx.write('#SBATCH --output {}/{}-%j.log\n'.format(batchdir, jobname))
        fx.write('#SBATCH --time={:02d}:{:02d}:00\n'.format(runtime_hh, runtime_mm))
        fx.write('#SBATCH --exclusive\n')

        fx.write('\n')

        #- Special case CFS readonly mount at NERSC
        if 'DESI_ROOT_READONLY' in os.environ:
            readonlydir = os.environ['DESI_ROOT_READONLY']
        elif os.environ['DESI_ROOT'].startswith('/global/cfs/cdirs'):
            readonlydir = os.environ['DESI_ROOT'].replace(
                    '/global/cfs/cdirs', '/dvs_ro/cfs/cdirs', 1)
        else:
            readonlydir = None

        if readonlydir is not None:
            fx.write(f'export DESI_ROOT_READONLY={readonlydir}\n\n')

        if cmdline is None:
            inparams = list(sys.argv).copy()
        elif np.isscalar(cmdline):
            inparams = []
            for param in cmdline.split(' '):
                for subparam in param.split("="):
                    inparams.append(subparam)
        else:
            inparams = list(cmdline)
        for parameter in ['--queue', '-q', '--batch-opts']:
            ## If a parameter is in the list, remove it and its argument
            ## Elif it is a '--' command, it might be --option=value, which won't be split.
            ##      check for that and remove the whole "--option=value"
            if parameter in inparams:
                loc = np.where(np.array(inparams) == parameter)[0][0]
                # Remove the command
                inparams.pop(loc)
                # Remove the argument of the command (now in the command location after pop)
                inparams.pop(loc)
            elif '--' in parameter:
                for ii,inparam in enumerate(inparams.copy()):
                    if parameter in inparam:
                        inparams.pop(ii)
                        break

        cmd = ' '.join(inparams)
        cmd = cmd.replace(' --batch', ' ').replace(' --nosubmit', ' ')
        if '--mpi' not in cmd:
            cmd += ' --mpi'

        if jobdesc.lower() == 'stdstarfit':
            cmd += ' --mpistdstars'

        if gpuextract and '--gpuextract' not in cmd:
            cmd += ' --gpuextract'
        if gpuspecter and '--gpuspecter' not in cmd:
            cmd += ' --gpuspecter'
        cmd += ' --starttime $(date +%s)'
        cmd += f' --timingfile {timingfile}'

        fx.write(f'# {jobdesc} exposure with {ncameras} cameras\n')
        fx.write(f'# using {ncores} cores on {nodes} nodes\n\n')

        fx.write('echo Starting at $(date)\n')

        mps_wrapper=''
        if jobdesc.lower() == 'arc':
            fx.write("export OMP_NUM_THREADS={}\n".format(threads_per_core))
        else:
            fx.write("export OMP_NUM_THREADS=1\n")
        if system_name == 'perlmutter-gpu' and jobdesc.lower() not in ['arc']:
            fx.write("export MPICH_GPU_SUPPORT_ENABLED=1\n")
            mps_wrapper='desi_mps_wrapper'

        if jobdesc.lower() not in ['science', 'prestdstar', 'stdstarfit', 'poststdstar']:
            if nightlybias:
                tmp = cmd.split()
                has_expid = False
                if '-e' in tmp:
                    has_expid = True
                    i = tmp.index('-e')
                    tmp.pop(i)  # -e
                    tmp.pop(i)  # EXPID
                if '--expid' in tmp:
                    has_expid = True
                    i = tmp.index('--expid')
                    tmp.pop(i)  # --expid
                    tmp.pop(i)  # EXPID
                bias_cmd = ' '.join(tmp)

                fx.write('\n# Run nightlybias first\n')
                srun=f'srun -N {bias_nodes} -n {bias_cores} -c {bias_threads_per_core} {bias_cmd}'
                fx.write('echo Running {}\n'.format(srun))
                fx.write('{}\n'.format(srun))

                if has_expid:
                    fx.write('\nif [ $? -eq 0 ]; then\n')
                    fx.write('  echo nightlybias succeeded at $(date)\n')
                    fx.write('else\n')
                    fx.write('  echo FAILED: nightlybias failed; stopping at $(date)\n')
                    fx.write('  exit 1\n')
                    fx.write('fi\n')

            if ' -e ' in cmd or ' --expid ' in cmd:
                fx.write('\n# Process exposure\n')
                cmd = cmd.replace('--nightlybias', '')
                srun=(f'srun -N {nodes} -n {ncores} -c {threads_per_core} --cpu-bind=cores '
                    +mps_wrapper+f' {cmd}')
                fx.write('echo Running {}\n'.format(srun))
                fx.write('{}\n'.format(srun))

        else:
            if jobdesc.lower() in ['science', 'prestdstar', 'stdstarfit']:
                fx.write('\n# Do steps through stdstarfit at full MPI parallelism\n')
                srun = (f' srun -N {nodes} -n {ncores} -c {threads_per_core} --cpu-bind=cores '
                    +mps_wrapper+f' {cmd}')
                if jobdesc.lower() in ['science', 'prestdstar']:
                    srun += ' --nofluxcalib'
                fx.write('echo Running {}\n'.format(srun))
                fx.write('{}\n'.format(srun))

            if jobdesc.lower() in ['science', 'poststdstar']:
                ntasks=ncameras

                tot_threads = nodes * batch_config['cores_per_node'] * batch_config['threads_per_core']
                threads_per_task = max(int(tot_threads / ntasks), 1)
                fx.write('\n# Use less MPI parallelism for fluxcalib MP parallelism\n')
                fx.write('# This should quickly skip over the steps already done\n')
                #- fluxcalib multiprocessing parallelism needs --cpu-bind=none (or at least not "cores")
                srun = f'srun -N {nodes} -n {ntasks} -c {threads_per_task} --cpu-bind=none {cmd} '
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

def create_desi_proc_tilenight_batch_script(night, exp, tileid, ncameras, queue, runtime=None, batch_opts=None,
                                  system_name=None, mpistdstars=True, gpuspecter=False,
                                  gpuextract=False,
                                  ):
    """
    Generate a SLURM batch script to be submitted to the slurm scheduler to run desi_proc.

    Args:
        night: str or int. The night the data was acquired.
        exp: int, or list of ints. The exposure id(s) for the data.
        tileid: str or int. The tile id for the data.
        ncameras: int. The number of cameras used for joint fitting.
        queue: str. Queue to be used.

    Options:
        runtime: str. Timeout wall clock time.
        batch_opts: str. Other options to give to the slurm batch scheduler (written into the script).
        system_name: name of batch system, e.g. cori-haswell, cori-knl.
        mpistdstars: bool. Whether to use MPI for stdstar fitting.
        gpuspecter: bool. Whether to use gpu_specter.
        gpuextract: bool. Whether to perform gpu extraction with gpu_specter.

    Returns:
        scriptfile: the full path name for the script written.

    """

    batchdir = get_desi_proc_batch_file_path(night)
    os.makedirs(batchdir, exist_ok=True)

    nexps = 1
    if exp is not None and not np.isscalar(exp):
        nexps = len(exp)

    jobname = get_desi_proc_tilenight_batch_file_name(night, tileid)

    timingfile = f'{jobname}-timing-$SLURM_JOBID.json'
    timingfile = os.path.join(batchdir, timingfile)

    scriptfile = os.path.join(batchdir, jobname + '.slurm')

    batch_config = batch.get_config(system_name)
    threads_per_core = batch_config['threads_per_core']
    gpus_per_node = batch_config['gpus_per_node']

    ncores, nodes, runtime = determine_resources(ncameras,'TILENIGHT',
        queue=queue, nexps=nexps, system_name=system_name,forced_runtime=runtime)

    if runtime is None:
        runtime = 30

    runtime_hh = int(runtime // 60)
    runtime_mm = int(runtime % 60)

    with open(scriptfile, 'w') as fx:
        fx.write('#!/bin/bash -l\n\n')
        fx.write('#SBATCH -N {}\n'.format(nodes))
        fx.write('#SBATCH --qos {}\n'.format(queue))
        for opts in batch_config['batch_opts']:
            fx.write('#SBATCH {}\n'.format(opts))
        if batch_opts is not None:
            fx.write('#SBATCH {}\n'.format(batch_opts))
        if system_name == 'perlmutter-gpu':
            # perlmutter-gpu requires projects name with "_g" appended
            fx.write('#SBATCH --account desi_g\n')
        else:
            fx.write('#SBATCH --account desi\n')
        fx.write('#SBATCH --job-name {}\n'.format(jobname))
        fx.write('#SBATCH --output {}/{}-%j.log\n'.format(batchdir, jobname))
        fx.write('#SBATCH --time={:02d}:{:02d}:00\n'.format(runtime_hh, runtime_mm))
        fx.write('#SBATCH --exclusive\n')

        fx.write('\n')

        #- Special case CFS readonly mount at NERSC
        if 'DESI_ROOT_READONLY' in os.environ:
            readonlydir = os.environ['DESI_ROOT_READONLY']
        elif os.environ['DESI_ROOT'].startswith('/global/cfs/cdirs'):
            readonlydir = os.environ['DESI_ROOT'].replace(
                    '/global/cfs/cdirs', '/dvs_ro/cfs/cdirs', 1)
        else:
            readonlydir = None

        if readonlydir is not None:
            fx.write(f'export DESI_ROOT_READONLY={readonlydir}\n\n')

        fx.write('\n')

        cmd = 'desi_proc_tilenight'
        cmd += f' -n {night}'
        cmd += f' -t {tileid}'
        cmd += f' --timingfile {timingfile}'
        cmd += f' --mpi'
        if mpistdstars:
            cmd += f' --mpistdstars'
        if gpuextract:
            cmd += f' --gpuspecter --gpuextract'
        elif gpuspecter:
            cmd += f' --gpuspecter'

        fx.write(f'# running a tile-night\n')
        fx.write(f'# using {ncores} cores on {nodes} nodes\n\n')

        fx.write('echo Starting at $(date)\n')

        mps_wrapper=''
        if system_name == 'perlmutter-gpu':
            fx.write("export MPICH_GPU_SUPPORT_ENABLED=1\n")
            mps_wrapper='desi_mps_wrapper'

        fx.write('\n# Do steps through stdstarfit at full MPI parallelism\n')
        srun = (f' srun -N {nodes} -n {ncores} -c {threads_per_core} --cpu-bind=cores '
                +mps_wrapper+f' {cmd}')
        fx.write('echo Running {}\n'.format(srun))
        fx.write('{}\n'.format(srun))

        fx.write('\nif [ $? -eq 0 ]; then\n')
        fx.write('  echo SUCCESS: done at $(date)\n')
        fx.write('else\n')
        fx.write('  echo FAILED: done at $(date)\n')
        fx.write('  exit 1\n')
        fx.write('fi\n')

    print('Wrote {}'.format(scriptfile))
    print('logfile will be {}/{}-JOBID.log\n'.format(batchdir, jobname))

    return scriptfile

def find_most_recent(night, file_type='psfnight', cam='r', n_nights=30):
    '''
       Searches back in time for either psfnight or fiberflatnight (or anything supported by
       desispec.calibfinder.findcalibfile. This only works on nightly-based files, so exposure id
       information is not used.

       Inputs:
         night : str. YYYYMMDD   night to look back from
         file_type : str. psfnight or fiberflatnight
         cam : str. camera (b, r, or z).
         n_nights : int.  number of nights to step back before giving up

      returns:
         nightfile : str. Full pathname to calibration file of interest.
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
        nightfile = findfile(file_type, test_night_str, camera=cam)
        if os.path.isfile(nightfile) :
            return nightfile

    return None
