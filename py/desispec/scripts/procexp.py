
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
from desispec.log import get_logger

import argparse
import sys

def parse(options=None):
    parser = argparse.ArgumentParser(description="Apply fiberflat, sky subtraction and calibration.")
    parser.add_argument('--infile', type = str, default = None, required=True,
                        help = 'path of DESI exposure frame fits file')
    parser.add_argument('--fiberflat', type = str, default = None,
                        help = 'path of DESI fiberflat fits file')
    parser.add_argument('--sky', type = str, default = None,
                        help = 'path of DESI sky fits file')
    parser.add_argument('--calib', type = str, default = None,
                        help = 'path of DESI calibration fits file')
    parser.add_argument('--outfile', type = str, default = None, required=True,
                        help = 'path of DESI sky fits file')

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

    if args.fiberflat!=None :
        log.info("apply fiberflat")
        # read fiberflat
        fiberflat = read_fiberflat(args.fiberflat)

        # apply fiberflat to sky fibers
        apply_fiberflat(frame, fiberflat)

    if args.sky!=None :
        log.info("subtract sky")
        # read sky
        skymodel=read_sky(args.sky)
        # subtract sky
        subtract_sky(frame, skymodel)

    if args.calib!=None :
        log.info("calibrate")
        # read calibration
        fluxcalib=read_flux_calibration(args.calib)
        # apply calibration
        apply_flux_calibration(frame, fluxcalib)


    # save output
    write_frame(args.outfile, frame)

    log.info("successfully wrote %s"%args.outfile)








"""
exspec extracts individual bundles of spectra with one bundle per output file.
This script merges them back together into a single file combining all
bundles.

This workflow is hacky.  Release early, release often, but also refactor often.

Stephen Bailey, LBL
March 2014
"""

from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
from astropy.io import fits

from desispec.frame import Frame
import desispec.io
from desispec.log import get_logger

import argparse


def parse(options=None):
    parser = argparse.ArgumentParser(description="Merge extracted spectra bundles into one file.")
    parser.add_argument("-o", "--output", type=str, required=True,
        help="output file name")
    parser.add_argument("-d", "--delete", action="store_true",
        help="delete input files when done")
    parser.add_argument("-f", "--force", action="store_true",
        help="merge files even if some fibers are missing")
    parser.add_argument("files", nargs='*')

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args):

    log = get_logger()

    nspec = 500  #- Hardcode!  Number of DESI fibers per spectrograph

    #- Sanity check that all spectra are represented
    fibers = set()
    for filename in args.files:
        x = fits.getdata(filename, 'FIBERMAP')
        fibers.update( set(x['FIBER']) )

    if len(fibers) != nspec:
        msg = "Input files only have {} instead of {} spectra".format(len(fibers), nspec)
        if args.force:
            log.warn(msg)
        else:
            log.fatal(msg)
            sys.exit(1)

    #- Read a file to get basic dimensions
    w = fits.getdata(args.files[0], 'WAVELENGTH')
    nwave = len(w)
    R1 = fits.getdata(args.files[0], 'RESOLUTION')
    ndiag = R1.shape[1]
    hdr = fits.getheader(args[0])

    camera = hdr['CAMERA']     #- b0, r1, .. z9
    spectrograph = int(camera[1])
    fibermin = spectrograph*nspec

    #- Output arrays to fill
    flux = np.zeros( (nspec, nwave) )
    ivar = np.zeros( (nspec, nwave) )
    R = np.zeros( (nspec, ndiag, nwave) )
    fibermap = desispec.io.empty_fibermap(nspec, specmin=fibermin)

    #- Fill them!
    for filename in args:
        fx = fits.open(filename)
        xhdr = fx[0].header
        xflux = fx['FLUX'].data
        xivar = fx['IVAR'].data
        xR = fx['RESOLUTION'].data
        xfibermap = fx['FIBERMAP'].data
        fx.close()

        ii = xfibermap['FIBER'] % nspec
        
        flux[ii] = xflux
        ivar[ii] = xivar
        R[ii] = xR
        fibermap[ii] = xfibermap
        
    #- Write it out
    print("Writing", args.output)
    frame = Frame(w, flux, ivar, resolution_data=R,
                spectrograph=spectrograph,
                meta=hdr, fibermap=fibermap)
    desispec.io.write_frame(args.output, frame)

    #- Scary!  Delete input files
    if args.delete:
        for filename in args.files:
            os.remove(filename)

