"""
desispec.workflow.desi_proc_funcs
=================================

"""
import time

import os, argparse, re

import fitsio
from desispec.io import findfile
from desispec.io.util import decode_camword, \
    parse_cameras
from desiutil.log import get_logger


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
    parser.add_argument("--system-name", type=str, help='Batch system name (cori-haswell, perlmutter-gpu, ...)')
    parser.add_argument("--extract-subcomm-size", type=int, default=None, help="Size to use for GPU extract subcomm")
    parser.add_argument("--no-gpu", action="store_true", help="Do not use GPU for extractions even if available")
    parser.add_argument("--use-specter", action="store_true", help="Use classic specter instead of gpu_specter")
    parser.add_argument("--dont-merge-with-psf-input", action="store_true", help="Do not merge with PSF input")
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
                                                                  ", i.e. [brz][0-9][ABCD]. Example: 'b7D,z8A'."+\
                                                                  " Can be just amps ABCD if processing single camera.")
    parser.add_argument("--fframe", action="store_true", help="Also write non-sky subtracted fframe file")
    parser.add_argument("--nofiberflat", action="store_true", help="Do not apply fiberflat")
    parser.add_argument("--noskysub", action="store_true",
                        help="Do not subtract the sky. Also skips stdstar fit and flux calib")
    parser.add_argument("--noprestdstarfit", action="store_true",
                        help="Do not do any science reductions prior to stdstar fitting")
    parser.add_argument("--nostdstarfit", action="store_true", help="Do not fit standard stars")
    parser.add_argument("--nofluxcalib", action="store_true", help="Do not flux calibrate")
    parser.add_argument("--nightlybias", action="store_true", help="Create nightly bias model from ZEROs")
    # parser.add_argument("--bias-expids", type=str, default=None,
    #                     help="Explicitly name expids of ZEROs to use for nightly bias model")
    parser.add_argument("--nightlycte", action="store_true", help="Fit CTE model from LED exposures")
    parser.add_argument("--cte-expids", type=str, default=None,
                        help="Explicitly name expids of a cte flat and flat to use for cte model")
    return parser

def add_desi_proc_joint_fit_terms(parser):
    """
    Add parameters to the argument parser that are only used by desi_proc_joint_fit
    """
    #parser.add_argument("-n", "--nights", type=str, help="YEARMMDD nights")
    parser.add_argument("-e", "--expids", type=str, help="Exposure IDs")
    parser.add_argument("-i", "--inputs", type=str, help="input raw data files")
    parser.add_argument("--autocal-ff-solve-grad", action="store_true",
                        help="Perform a spatial gradient correction to the fiber flat"
                             + " by running desi_autocalib_fiberflat with --solve-gradient")
    return parser

def add_desi_proc_tilenight_terms(parser):
    """
    Add parameters to the argument parser that are only used by desi_proc_tilenight
    """
    parser.add_argument("-t", "--tileid", type=str, help="Tile ID")
    parser.add_argument("-d", "--dryrun", action="store_true", help="show commands only, do not run")
    parser.add_argument("--laststeps", type=str, default=None,
                        help="Comma separated list of LASTSTEP values "
                             + "(e.g. all, skysub, fluxcalib, ignore); "
                             + "by default, exposures with LASTSTEP "
                             + "'all' and 'fluxcalib' will be processed "
                             + "to the poststdstar step, and all others "
                             + "will not be processed at all.")

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
        tuple: A tuple containing:

        * comm: MPI communicator object
        * rank: int, the numeric number assigned to the currenct MPI rank
        * size: int, the total number of MPI ranks
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
        tuple: A tuple containing:

        * hdr: fitsio header object from the file at pathname
        * fx:  optional, fitsio file object returned only if return_filehandle is True
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
        str: The list of cameras that have data in the given file.
    """
    ## Be flexible on whether input is filepath or a filehandle
    if type(rawdata) is str:
        if os.path.isfile(rawdata):
            fx = fitsio.FITS(rawdata)
        else:
            raise IOError(f"File {rawdata} doesn't exist.")
    else:
        fx = rawdata

    recam = re.compile(r'^[brzBRZ][\d]$')
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
        tuple: A tuple containing:

        * args: modified version of the input args where values have been updated if None using information from
          appropriate headers using either night+expid or an input file.
        * hdr: fitsio header object obtained using ``read_header()`` on input file or file determined from args information.
        * camhdr: dict, dictionary of fitsio header objects for each camera in the input files.

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

  
def find_most_recent(night, file_type='psfnight', cam='r', n_nights=30):
    '''
    Searches back in time for either psfnight or fiberflatnight (or anything supported by
    desispec.calibfinder.findcalibfile. This only works on nightly-based files, so exposure id
    information is not used.

    Args:
        night : str. YYYYMMDD   night to look back from
        file_type : str. psfnight or fiberflatnight
        cam : str. camera (b, r, or z).
        n_nights : int.  number of nights to step back before giving up

    Returns:
        str: Full pathname to calibration file of interest.
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
