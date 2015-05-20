#!/usr/bin/env python
#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-

"""
This script computes the flux calibration for a DESI frame using precomputed spectro-photometrically calibrated stellar models.
"""

from desispec.io import read_frame
from desispec.io import read_fibermap
from desispec.io import read_fiberflat
from desispec.io import read_sky
from desispec.io.fluxcalibration import read_stdstar_models
from desispec.io.fluxcalibration import write_flux_calibration
from desispec.fiberflat import apply_fiberflat
from desispec.sky import subtract_sky
from desispec.fluxcalibration import compute_flux_calibration
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
    parser.add_argument('--fibermap', type = str, default = None,
                        help = 'path of DESI exposure frame fits file')
    parser.add_argument('--fiberflat', type = str, default = None,
                        help = 'path of DESI fiberflat fits file')
    parser.add_argument('--sky', type = str, default = None,
                        help = 'path of DESI sky fits file')
    parser.add_argument('--models', type = str, default = None,
                        help = 'path of spetro-photometric stellar spectra fits file')
    parser.add_argument('--outfile', type = str, default = None,
                        help = 'path of DESI flux calbration fits file')


    args = parser.parse_args()
    log=get_logger()

    if args.infile is None or args.fibermap is None or args.outfile is None or args.fiberflat is None or args.sky is None or args.models is None :
        log.critical('Missing something')
        parser.print_help()
        sys.exit(12)


    log.info("read frame")
    # read frame
    spectra = read_frame(args.infile)

    log.info("apply fiberflat")
    # read fiberflat
    fiberflat = read_fiberflat(args.fiberflat)

    # apply fiberflat
    apply_fiberflat(spectra, fiberflat)
    
    log.info("subtract sky")
    # read sky
    skymodel=read_sky(args.sky)

    # subtract sky
    subtract_sky(spectra, skymodel)

    log.info("compute flux calibration")

    # read models
    model_flux,model_wave,model_fibers=read_stdstar_models(args.models)

    # select fibers
    SPECMIN=spectra.header["SPECMIN"]
    SPECMAX=spectra.header["SPECMAX"]
    selec=np.where((model_fibers>=SPECMIN)&(model_fibers<=SPECMAX))[0]
    if selec.size == 0 :
        log.error("not stellar models for this spectro")
        sys.exit(12)
    fibers=model_fibers[selec]-spectra.header["SPECMIN"]
    log.info("star fibers= %s"%str(fibers))

    table = read_fibermap(args.fibermap)
    bad=np.where(table["OBJTYPE"][fibers]!="STD")[0]
    if bad.size > 0 :
        for fiber in fibers[bad] :
            log.error("inconsistency with fiber %d, OBJTYPE='%s' in fibermap"%(fiber,table["OBJTYPE"][fiber]))
        sys.exit(12)

    fluxcalib = compute_flux_calibration(spectra, fibers, model_wave, model_flux)

    # write result
    write_flux_calibration(args.outfile, fluxcalib, header=spectra.header)


    log.info("successfully wrote %s"%args.outfile)


if __name__ == '__main__':
    main()
