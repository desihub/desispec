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
from desispec.io.fluxcalibration import read_stellar_models
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

    if args.infile is None or args.fibermap is None or args.outfile is None or args.fiberflat is None or args.sky is None or args.models is None :
        print('Missing something')
        parser.print_help()
        sys.exit(12)

    log=get_logger()

    log.info("read frame")
    # read frame
    flux,ivar,wave,resol,head = read_frame(args.infile)

    log.info("apply fiberflat")
    # read fiberflat
    fiberflat,ffivar,ffmask,ffmeanspec,ffwave,ffhdr = read_fiberflat(args.fiberflat)

    # apply fiberflat
    apply_fiberflat(flux=flux,ivar=ivar,wave=wave,fiberflat=fiberflat,ffivar=ffivar,ffmask=ffmask,ffwave=ffwave)

    log.info("subtract sky")
    # read sky
    skyflux,sivar,smask,cskyflux,csivar,swave,skyhdr=read_sky(args.sky)

    # subtract sky
    subtract_sky(flux=flux,ivar=ivar,resolution_data=resol,wave=wave,skyflux=skyflux,convolved_skyivar=csivar,skymask=smask,skywave=swave)

    log.info("compute flux calibration")

    # read models
    model_flux,model_wave,model_fibers=read_stellar_models(args.models)

    # select fibers
    SPECMIN=head["SPECMIN"]
    SPECMAX=head["SPECMAX"]
    selec=np.where((model_fibers>=SPECMIN)&(model_fibers<=SPECMAX))[0]
    if selec.size == 0 :
        log.error("not stellar models for this spectro")
        sys.exit(12)
    fibers=model_fibers[selec]-head["SPECMIN"]
    log.info("star fibers= %s"%str(fibers))

    table, fmhdr = read_fibermap(args.fibermap)
    bad=np.where(table["OBJTYPE"][fibers]!="STD")[0]
    if bad.size > 0 :
        for fiber in fibers[bad] :
            log.error("inconsistency with fiber %d, OBJTYPE='%' in fibermap"%(fiber,table["OBJTYPE"][fiber]))
        sys.exit(12)

    calibration, calibivar, mask, ccalibration, ccalibivar = compute_flux_calibration(wave,flux[fibers],ivar[fibers],resol[fibers],model_wave,model_flux)

    # write result
    write_flux_calibration(args.outfile,calibration, calibivar, mask, ccalibration, ccalibivar,wave, head)


    log.info("successfully wrote %s"%args.outfile)


if __name__ == '__main__':
    main()
