#!/usr/bin/env python
#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-

"""
This script processes an exposure by applying fiberflat, sky subtraction, spectro-photometric calibration depending on input.
"""

from desispec.io import read_frame, write_frame
from desispec.io import read_fiberflat
from desispec.io import read_sky
from desispec.io.fluxcalibration import read_flux_calibration
from desispec.fiberflat import apply_fiberflat
from desispec.sky import subtract_sky
from desispec.fluxcalibration import apply_flux_calibration
from desispec.log import get_logger

import argparse
import os
import os.path
import numpy as np
import sys


def main() :

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--infile', type = str, default = None,
                        help = 'path of DESI exposure frame fits file')
    parser.add_argument('--fiberflat', type = str, default = None,
                        help = 'path of DESI fiberflat fits file')
    parser.add_argument('--sky', type = str, default = None,
                        help = 'path of DESI sky fits file')
    parser.add_argument('--calib', type = str, default = None,
                        help = 'path of DESI calibration fits file')
    parser.add_argument('--outfile', type = str, default = None,
                        help = 'path of DESI sky fits file')
    # add calibration here when exists

    args = parser.parse_args()
    log = get_logger()

    if args.infile is None:
        log.critical('Missing input')
        parser.print_help()
        sys.exit(12)

    if args.fiberflat is None and args.sky is None:
        log.critical('Nothing to do ??')
        parser.print_help()
        sys.exit(12)

    if args.outfile is None:
        log.critical('Missing output')
        parser.print_help()
        sys.exit(12)


    frame = read_frame(args.infile)
    
    if args.fiberflat!=None :
        log.info("apply fiberflat")
        # read fiberflat
        fiberflat = read_fiberflat(args.fiberflat)

        # apply fiberflat to sky fibers
        apply_fiberflat(frame, fiberflat)

    if args.sky!=None :
        log.info("subtract sky")
        # read sky
        skyflux,sivar,smask,cskyflux,csivar,swave,skyhdr=read_sky(args.sky)
        # subtract sky
        subtract_sky(flux=frame.flux, ivar=frame.ivar, resolution_data=frame.resolution_data, wave=frame.wave,
            skyflux=skyflux,convolved_skyivar=csivar,skymask=smask,skywave=swave)

    if args.calib!=None :
        log.info("calibrate")
        # read calibration
        calibration,calib_ivar,cmask,convolved_calibration,convolved_calib_ivar,calib_wave=read_flux_calibration(args.calib)
        # apply calibration
        apply_flux_calibration(flux=frame.flux,ivar=frame.ivar,resolution_data=frame.resolution_data,wave=frame.wave,calibration=calibration,civar=convolved_calib_ivar,cmask=cmask,cwave=calib_wave)


    # save output
    write_frame(args.outfile, frame)

    log.info("successfully wrote %s"%args.outfile)



if __name__ == '__main__':
    main()
