"""
This script processes an exposure by applying fiberflat, sky subtraction,
spectro-photometric calibration depending on input.
"""

from desispec.io import read_frame, write_frame
from desispec.io import read_fiberflat
from desispec.io import read_sky
from desispec.io import shorten_filename
from desispec.io.fluxcalibration import read_flux_calibration
from desispec.fiberflat import apply_fiberflat
from desispec.sky import subtract_sky
from desispec.fluxcalibration import apply_flux_calibration
from desiutil.log import get_logger
from desispec.cosmics import reject_cosmic_rays_1d
from desispec.specscore import compute_and_append_frame_scores
from desispec.fiberbitmasking import get_fiberbitmasked_frame
from specter.psf.gausshermite  import  GaussHermitePSF
from desispec.tsnr import get_tsnr

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
    parser.add_argument('--psf', type = str, default=None,
                        help = 'path of DESI calibration psf file (triggers tsnr) ')
    parser.add_argument('-o','--outfile', type = str, default = None, required=True,
                        help = 'path of DESI sky fits file')
    parser.add_argument('--cosmics-nsig', type = float, default = 0, required=False,
                        help = 'n sigma rejection for cosmics in 1D (default, no rejection)')
    parser.add_argument('--no-sky-throughput-correction', action='store_true',
                        help = 'Do NOT apply a throughput correction when subtraction the sky')
    parser.add_argument('--no-zero-ivar', action='store_true',
                        help = 'Do NOT set ivar=0 for masked pixels')

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args):

    log = get_logger()

    if (args.fiberflat is None) and (args.sky is None) and (args.calib is None):
        log.critical('no --fiberflat, --sky, or --calib; nothing to do ?!?')
        sys.exit(12)

    frame = read_frame(args.infile)

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

    if args.sky!=None :

        # read sky
        skymodel=read_sky(args.sky)

        if args.cosmics_nsig>0 :

            # use a copy the frame (not elegant but robust)
            copied_frame = copy.deepcopy(frame)

            # first subtract sky without throughput correction
            subtract_sky(copied_frame, skymodel, apply_throughput_correction = False, zero_ivar = (not args.no_zero_ivar))

            # then find cosmics
            log.info("cosmics ray 1D rejection after sky subtraction")
            reject_cosmic_rays_1d(copied_frame,args.cosmics_nsig)

            # copy mask
            frame.mask = copied_frame.mask

            # and (re-)subtract sky, but just the correction term
            subtract_sky(frame, skymodel, apply_throughput_correction = (not args.no_sky_throughput_correction), zero_ivar = (not args.no_zero_ivar) )

        else :
            # subtract sky
            subtract_sky(frame, skymodel, apply_throughput_correction = (not args.no_sky_throughput_correction), zero_ivar = (not args.no_zero_ivar) )

        compute_and_append_frame_scores(frame,suffix="SKYSUB")

    if args.calib!=None :
        log.info("calibrate")
        # read calibration
        fluxcalib=read_flux_calibration(args.calib)
        # apply calibration
        apply_flux_calibration(frame, fluxcalib)

        # Ensure that ivars are set to 0 for all values if any designated
        # fibermask bit is set. Also flips a bits for each frame.mask value using specmask.BADFIBER
        frame = get_fiberbitmasked_frame(frame,bitmask="flux",ivar_framemask=True)
        compute_and_append_frame_scores(frame,suffix="CALIB")

    if args.psf != None:                
        log.info("calculating tsnr")

        # construct PSF from file. 
        psf=GaussHermitePSF(args.psf)

        tsnr=get_tsnr(frame, psf) 
        
    # record inputs
    frame.meta['IN_FRAME'] = shorten_filename(args.infile)
    frame.meta['FIBERFLT'] = shorten_filename(args.fiberflat)
    frame.meta['IN_SKY']   = shorten_filename(args.sky)
    frame.meta['IN_CALIB'] = shorten_filename(args.calib)

    # save output
    write_frame(args.outfile, frame, units='10**-17 erg/(s cm2 Angstrom)')
    log.info("successfully wrote %s"%args.outfile)
