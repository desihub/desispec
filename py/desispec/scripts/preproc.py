"""
desispec.scripts.preproc
========================

Command line wrappers for pre-processing a DESI raw exposure
"""

import argparse

import os
import sys
import multiprocessing as mp
import numpy as np
from desispec import io
from desispec.parallel import default_nproc
from desiutil.log import get_logger
log = get_logger()

def parse(options=None):
    parser = argparse.ArgumentParser(
        description="Preprocess DESI raw data",
        epilog='''By default, all HDUs of the input file are processed and
written to pix*.fits files in the current directory; use --outdir to override
the output directory location.  --outfile FILENAME can be used to
override a single output filename but only if a single
camera is given with --cameras.
--bias/--pixflat/--mask can specify the calibration files
to use, but also only if a single camera is specified.
Must specify --infile OR --night and --expid.
        ''')
    parser.add_argument('-i','--infile', type=str, required=False,
                        help = 'path of DESI raw data file')
    parser.add_argument('-n', '--night', type=int, required=False,
                        help = 'YEARMMDD night; must also provide --expid')
    parser.add_argument('-e', '--expid', type=int, required=False,
                        help = 'exposure ID; must also provide --night')
    parser.add_argument('--outdir', type = str, default = None, required=False,
                        help = 'output directory')
    parser.add_argument('--fibermap', type = str, default = None, required=False,
                        help = 'path to fibermap file')
    parser.add_argument('-o','--outfile', type = str, default = None, required=False,
                        help = 'output preprocessed image file')
    parser.add_argument('--cameras', type = str, default = None, required=False,
                        help = 'comma separated list of cameras')
    parser.add_argument('--bias', type = str, default = None, required=False,
                        help = 'bias image calibration file')
    parser.add_argument('--dark', type = str, default = None, required=False,
                        help = 'dark image calibration file')
    parser.add_argument('--pixflat', type = str, default = None, required=False,
                        help = 'pixflat image calibration file')
    parser.add_argument('--mask', type = str, default = None, required=False,
                        help = 'mask image calibration file')
    parser.add_argument('--nobias', action = 'store_true',
                        help = 'no bias subtraction')
    parser.add_argument('--use_savgol', action = 'store_true',
            help='Use Savitsky-Golay filter for the overscan (False/True)')
    parser.add_argument('--nodark', action = 'store_true',
                        help = 'no dark subtraction')
    parser.add_argument('--nopixflat',action = 'store_true',
                        help = 'no pixflat correction')
    parser.add_argument('--nomask', action = 'store_true',
                        help = 'no prior masking of pixels')
    parser.add_argument('--nocrosstalk', action = 'store_true',
                        help = 'no cross-talk correction')
    parser.add_argument('--nocosmic', action='store_true',
                        help = 'do not try and reject cosmic rays')
    parser.add_argument('--nogain', action='store_true',
                        help = 'do not apply gain correction')
    parser.add_argument('--nodarktrail', action='store_true',
                        help = 'do not correct for dark trails if any')
    parser.add_argument('--cosmics-nsig', type = float, default = 6, required=False,
                        help = 'for cosmic ray rejection : number of sigma above background required')
    parser.add_argument('--cosmics-cfudge', type = float, default = 3, required=False,
                        help = 'for cosmic ray rejection : number of sigma inconsistent with PSF required')
    parser.add_argument('--cosmics-c2fudge', type = float, default = 0.5, required=False,
                        help = 'for cosmic ray rejection : fudge factor applied to PSF')

    parser.add_argument('--bkgsub-for-dark', action='store_true',
                        help = 'do a background subtraction prior to cosmic ray rejection')
    parser.add_argument('--bkgsub-for-science', action='store_true',
                        help = 'do a background subtraction in science exposures (measured between blocks of fiber traces)')


    parser.add_argument('--zero-masked', action='store_true',
                        help = 'set to zero the flux of masked pixels (for convenience to display images, no impact on analysis)')
    parser.add_argument('--no-ccd-calib-filename', action='store_true',
                        help = 'do not read calibration data from yaml file in desispec')
    parser.add_argument('--ccd-calib-filename', required=False, default=None,
                        help = 'specify a difference ccd calibration filename (for dev. purpose), default is in desispec/data/ccd')
    parser.add_argument('--fill-header', type = str, default = None,  nargs ='*', help="fill camera header with contents of those of other hdus")
    parser.add_argument('--scattered-light', action="store_true", help="fit and remove scattered light")
    parser.add_argument('--psf', type = str, required=False, default=None, help="psf file to remove scattered light or to compute the variance model")
    parser.add_argument('--model-variance', action="store_true", help="compute a model of the CCD image to derive the Poisson noise")
    parser.add_argument('--no-traceshift', action="store_true", help="do not adjust the trace coordinates when computing a model of the CCD image")
    parser.add_argument('--ncpu', type=int, default=default_nproc,
            help=f"number of parallel processes to use [{default_nproc}]")

    #- uses sys.argv if options=None
    args = parser.parse_args(options)

    return args

def main(args=None):
    if args is None:
        args = parse()
    elif isinstance(args, (list, tuple)):
        args = parse(args)

    # Use bias?
    bias=True
    if args.bias : bias=args.bias
    if args.nobias : bias=False
    # Use dark?
    dark=True
    if args.dark : dark=args.dark
    if args.nodark : dark=False
    pixflat=True
    if args.pixflat : pixflat=args.pixflat
    if args.nopixflat : pixflat=False
    mask=True
    if args.mask : mask=args.mask
    if args.nomask : mask=False

    # infile or night+expid?
    if args.infile is None:
        if args.night is None or args.expid is None:
            log.critical('Must provide --infile or both --night and --expid')
            sys.exit(1)

        args.infile = io.findfile('raw', args.night, args.expid)

    else:
        if args.night is not None or args.expid is not None:
            msg = f'ignoring --night/--expid; using --infile {args.infile}'
            log.warning(msg)

    if args.cameras is None:
        args.cameras = [c+str(i) for c in 'brz' for i in range(10)]
    else:
        args.cameras = args.cameras.split(',')

    if (args.bias is not None) or (args.pixflat is not None) or (args.mask is not None) or (args.dark is not None):
        if len(args.cameras) > 1:
            raise ValueError('must use only one camera with --bias, --dark, --pixflat, --mask options')

    if (args.outfile is not None) and len(args.cameras) > 1:
            raise ValueError('must use only one camera with --outfile option')

    if args.outdir is None:
        args.outdir = os.getcwd()
        log.warning('--outdir not specified; using {}'.format(args.outdir))

    ccd_calibration_filename = None

    if args.no_ccd_calib_filename :
        ccd_calibration_filename = False
    elif args.ccd_calib_filename is not None :
        ccd_calibration_filename = args.ccd_calib_filename

    if args.fibermap and not os.path.exists(args.fibermap):
        raise ValueError('--fibermap {} not found'.format(args.fibermap))

    if args.fibermap is None:
        datadir, infile = os.path.split(os.path.abspath(args.infile))
        fibermapfile = infile.replace('desi-', 'fibermap-').replace('.fits.fz', '.fits')
        args.fibermap = os.path.join(datadir, fibermapfile)

    #- Assemble options to pass to io.read_raw/preproc for each camera
    #- so that they can be optionally parallelized
    opts_array = list()
    for camera in args.cameras:
        opts = dict(
                infile = args.infile,
                camera = camera,
                outfile = args.outfile,
                outdir = args.outdir,
                fibermap = args.fibermap,
                bias=bias, dark=dark, pixflat=pixflat, mask=mask,
                bkgsub_dark=args.bkgsub_for_dark,
                bkgsub_science=args.bkgsub_for_science,
                nocosmic=args.nocosmic,
                cosmics_nsig=args.cosmics_nsig,
                cosmics_cfudge=args.cosmics_cfudge,
                cosmics_c2fudge=args.cosmics_c2fudge,
                ccd_calibration_filename=ccd_calibration_filename,
                nocrosstalk=args.nocrosstalk,
                nogain=args.nogain,
                use_savgol=args.use_savgol,
                nodarktrail=args.nodarktrail,
                fill_header=args.fill_header,
                remove_scattered_light=args.scattered_light,
                psf_filename=args.psf,
                model_variance=args.model_variance,
                zero_masked=args.zero_masked,
                no_traceshift=args.no_traceshift
        )
        opts_array.append(opts)

    num_cameras = len(args.cameras)
    assert num_cameras == len(opts_array)
    if args.ncpu > 1 and num_cameras>1:
        n = min(args.ncpu, num_cameras)
        log.info(f'Processing {num_cameras} cameras with {n} multiprocessing processes')
        pool = mp.Pool(n)
        failed = pool.map(_preproc_file_kwargs_wrapper, opts_array)
        num_failed = np.sum(failed)
        pool.close()
        pool.join()
    else:
        log.info(f'Not using multiprocessing for {num_cameras} cameras')
        num_failed = 0
        for opts in opts_array:
            num_failed += _preproc_file_kwargs_wrapper(opts)

    if num_failed > 0:
        log.error(f'{num_failed}/{num_cameras} cameras failed')
    else:
        log.info(f'All {num_cameras} cameras successfully preprocessed')

    return int(num_failed)

def _preproc_file_kwargs_wrapper(opts):
    """
    This function just unpacks opts dict for preproc_file so that it can be
    used with multiprocessing.Pool.map
    """
    return preproc_file(**opts)

def preproc_file(infile, camera, outfile=None, outdir=None, fibermap=None,
        zero_masked=False, **preproc_opts):
    """
    Preprocess a single camera from a single input file

    Args:
        infile : input raw data file
        camera : camera, e.g. 'b0', 'r1', 'z9'

    Options:
        outfile: output preprocessed image file to write
        outdir: output directory; derive filename from infile NIGHT and EXPID
        fibermap: fibermap filename to include in output
        zero_masked (bool): set masked pixels to 0
        preproc_opts: dictionary to pass to preproc

    Returns error code (1=error, 0=success) but will not raise exception if
    there is an I/O or preprocessing failure (allows other parallel procs to
    proceed).

    Note: either `outfile` or `outdir` must be provided
    """
    try:
        img = io.read_raw(infile, camera,
                          fibermapfile=fibermap,
                          **preproc_opts)
    except IOError as e:
        #- print error and return error code, but don't raise exception so
        #- that this won't block other cameras for multiprocessing
        log.error('Error while reading or preprocessing camera {} in {}'.format(camera, infile))
        log.error(e)
        return 1

    if zero_masked :
        log.info("Setting masked pixels values to zero")
        img.pix *= (img.mask==0)

    if outfile is None:
        night = img.meta['NIGHT']
        if not 'EXPID' in img.meta.keys() :
            if 'EXPNUM' in img.meta.keys() :
                img.meta['EXPID']=img.meta['EXPNUM']
            else :
                mess="no EXPID nor EXPNUM in img.meta, cannot create output filename"
                log.error(mess)
                raise KeyError(mess)
        expid = img.meta['EXPID']
        outfile = io.findfile('preproc', night=night,expid=expid,camera=camera,
                              outdir=outdir)

    io.write_image(outfile, img)
    log.info("Wrote {}".format(outfile))
    return 0
