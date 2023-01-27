"""
desispec.scripts.procexp
========================

This script processes an exposure by applying fiberflat, sky subtraction,
spectro-photometric calibration depending on input.  Optionally, includes
tsnr in the scores hdu.
"""

from desispec.io import read_frame, write_frame
from desispec.io import read_fiberflat
from desispec.io import read_sky
from desispec.io import read_fibermap
from desispec.io.fluxcalibration import read_flux_calibration
from desispec.io import shorten_filename

from desispec.fiberflat import apply_fiberflat
from desispec.sky import subtract_sky
from desispec.fluxcalibration import apply_flux_calibration
from desispec.cosmics import reject_cosmic_rays_1d
from desispec.specscore import compute_and_append_frame_scores, append_frame_scores
from desispec.fiberbitmasking import get_fiberbitmasked_frame
from desispec.fibercrosstalk import correct_fiber_crosstalk

from desispec.tsnr import calc_tsnr2
from desiutil.log import get_logger

import argparse
import sys
import copy

def parse(options=None):
    parser = argparse.ArgumentParser(description="Apply fiberflat, sky subtraction and calibration.")
    parser.add_argument('-i','--infile', type = str, default = None, required=True,
                        help = 'path of DESI exposure frame fits file')
    parser.add_argument('--fiberflat', type = str, default = None,
                        help = 'path of DESI fiberflat fits file')
    parser.add_argument('--sky', type = str, default = None,
                        help = 'path of DESI sky fits file')
    parser.add_argument('--calib', type = str, default = None,
                        help = 'path of DESI calibration fits file')
    parser.add_argument('-o','--outfile', type = str, default = None, required=True,
                        help = 'path of output fits file')
    parser.add_argument('--cosmics-nsig', type = float, default = 0, required=False,
                        help = 'n sigma rejection for cosmics in 1D (default, no rejection)')
    parser.add_argument('--apply-sky-throughput-correction', action='store_true',
                        help =('Apply a throughput correction to the whole sky spectrum, not just the lines '
                               '(default: a correction is applied to the sky lines but not the continuum)'))
    parser.add_argument('--no-sky-line-throughput-correction', action='store_true',
                        help =('Do not apply a throughput correction to the sky spectrum lines or the continuum '
                               '(default: a correction is applied to the sky lines but not the continuum)'))
    parser.add_argument('--no-zero-ivar', action='store_true',
                        help = 'Do NOT set ivar=0 for masked pixels')
    parser.add_argument('--no-tsnr', action='store_true',
                        help = 'Do not compute template SNR')
    parser.add_argument('--no-xtalk', action='store_true',
                        help = 'Do not apply fiber crosstalk correction')
    parser.add_argument('--alpha_only', action='store_true',
                        help = 'Only compute alpha of tsnr calc.')

    args = parser.parse_args(options)
    return args

def main(args):

    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    log = get_logger()

    if (args.fiberflat is None) and (args.sky is None) and (args.calib is None):
        log.critical('no --fiberflat, --sky, or --calib; nothing to do ?!?')
        sys.exit(12)

    if (not args.no_tsnr) and (args.calib is None) :
        log.warning('Need --fiberflat --sky and --calib to compute template SNR. We are not computing it.')
        args.no_tsnr = True

    if args.apply_sky_throughput_correction and args.no_sky_line_throughput_correction:
        msg = "Use --apply-sky-throughput-correction OR --no-sky-line-throughput-correction (or neither) but not both"
        log.critical(msg)
        raise ValueError(msg)

    frame = read_frame(args.infile)

    if not args.no_tsnr :
        # tsnr alpha calc. requires uncalibrated + no substraction rame.
        uncalibrated_frame = copy.deepcopy(frame)

    #- Raw scores already added in extraction, but just in case they weren't
    #- it is harmless to rerun to make sure we have them.
    compute_and_append_frame_scores(frame,suffix="RAW")

    if args.cosmics_nsig>0 and args.sky==None : # Reject cosmics (otherwise do it after sky subtraction)
        log.info("cosmics ray 1D rejection")
        reject_cosmic_rays_1d(frame,args.cosmics_nsig)

    if args.fiberflat!=None :
        log.info("apply fiberflat")
        # read fiberflat
        fiberflat = read_fiberflat(args.fiberflat)

        # apply fiberflat to all fibers
        apply_fiberflat(frame, fiberflat)
        compute_and_append_frame_scores(frame,suffix="FFLAT")
    else :
        fiberflat = None

    if args.no_xtalk :
        zero_ivar = (not args.no_zero_ivar)
    else :
        zero_ivar = False

    if args.sky!=None :

        # read sky
        skymodel=read_sky(args.sky)

        # subtract sky
        subtract_sky(frame, skymodel,
                     apply_throughput_correction_to_lines = (not args.no_sky_line_throughput_correction),
                     apply_throughput_correction = args.apply_sky_throughput_correction,
                     zero_ivar = zero_ivar)

        if args.cosmics_nsig>0 :
            log.info("cosmics ray 1D rejection after sky subtraction")
            reject_cosmic_rays_1d(frame,args.cosmics_nsig)

        compute_and_append_frame_scores(frame,suffix="SKYSUB")

    if not args.no_xtalk :
        log.info("fiber crosstalk correction")
        correct_fiber_crosstalk(frame,fiberflat)

        if not args.no_zero_ivar :
            frame.ivar *= (frame.mask==0)

    if args.calib!=None :
        log.info("calibrate")
        # read calibration
        fluxcalib=read_flux_calibration(args.calib)
        # apply calibration
        apply_flux_calibration(frame, fluxcalib)

        # Ensure that ivars are set to 0 for all values if any designated
        # fibermask bit is set. Also flips a bits for each frame.mask value using specmask.BADFIBER
        frame = get_fiberbitmasked_frame(frame,bitmask="flux",ivar_framemask=(not args.no_zero_ivar))
        compute_and_append_frame_scores(frame,suffix="CALIB")

    if not args.no_tsnr:
        log.info("calculating tsnr")
        results, alpha = calc_tsnr2(uncalibrated_frame, fiberflat=fiberflat, skymodel=skymodel, fluxcalib=fluxcalib, alpha_only=args.alpha_only)

        frame.meta['TSNRALPH'] = alpha

        comments = {k:"from calc_frame_tsnr" for k in results.keys()}
        append_frame_scores(frame,results,comments,overwrite=True)

    # record inputs
    frame.meta['IN_FRAME'] = shorten_filename(args.infile)
    frame.meta['FIBERFLT'] = shorten_filename(args.fiberflat)
    frame.meta['IN_SKY']   = shorten_filename(args.sky)
    frame.meta['IN_CALIB'] = shorten_filename(args.calib)

    # save output
    write_frame(args.outfile, frame, units='10**-17 erg/(s cm2 Angstrom)')
    log.info("successfully wrote %s"%args.outfile)
