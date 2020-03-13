
"""
This script processes an exposure by applying fiberflat, sky subtraction,
spectro-photometric calibration depending on input.
"""

from desispec.io import read_frame, write_frame
from desispec.io import read_fiberflat
from desispec.io import read_sky
from desispec.io.fluxcalibration import read_flux_calibration
from desispec.fiberflat import apply_fiberflat
from desispec.sky import subtract_sky
from desispec.fluxcalibration import apply_flux_calibration
from desiutil.log import get_logger
from desispec.cosmics import reject_cosmic_rays_1d
from desispec.specscore import compute_and_append_frame_scores
from desispec.fiberbitmasking import get_fiberbitmasked_frame

import argparse
import sys

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
                        help = 'path of DESI sky fits file')
    parser.add_argument('--cosmics-nsig', type = float, default = 0, required=False,
                        help = 'n sigma rejection for cosmics in 1D (default, no rejection)')
    parser.add_argument('--sky-throughput-correction', action='store_true',
                        help = 'apply a throughput correction when subtraction the sky')

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

            # first subtract sky without throughput correction
            subtract_sky(frame, skymodel, throughput_correction = False)

            # then find cosmics
            log.info("cosmics ray 1D rejection after sky subtraction")
            reject_cosmic_rays_1d(frame,args.cosmics_nsig)

            if args.sky_throughput_correction :
                # and (re-)subtract sky, but just the correction term
                subtract_sky(frame, skymodel, throughput_correction = True, default_throughput_correction = 0.)

        else :
            # subtract sky
            subtract_sky(frame, skymodel, throughput_correction = args.sky_throughput_correction )

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


    # save output
    write_frame(args.outfile, frame, units='10**-17 erg/(s cm2 Angstrom)')
    log.info("successfully wrote %s"%args.outfile)
