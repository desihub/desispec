"""
desispec.scripts.mergebundles
=============================

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
from astropy.table import Table

from desispec.frame import Frame
import desispec.io
from desiutil.log import get_logger

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
            log.warning(msg)
        else:
            log.fatal(msg)
            sys.exit(1)

    #- Read a file to get basic dimensions
    w = fits.getdata(args.files[0], 'WAVELENGTH')
    nwave = len(w)
    R1 = fits.getdata(args.files[0], 'RESOLUTION')
    ndiag = R1.shape[1]
    hdr = fits.getheader(args.files[0])

    camera = hdr['CAMERA'].lower()     #- b0, r1, .. z9
    spectrograph = int(camera[1])
    fibermin = spectrograph*nspec

    #- Output arrays to fill
    flux = np.zeros( (nspec, nwave) )
    ivar = np.zeros( (nspec, nwave) )
    R = np.zeros( (nspec, ndiag, nwave) )
    fibermap = None
    mask = np.zeros( (nspec, nwave), dtype=np.uint32)
    chi2pix = np.zeros( (nspec, nwave) )

    #- Fill them!
    for filename in args.files :
        fx = fits.open(filename)
        xhdr = fx[0].header
        xflux = fx['FLUX'].data
        xivar = fx['IVAR'].data
        xR = fx['RESOLUTION'].data
        xfibermap = fx['FIBERMAP'].data
        xmask = fx['MASK'].data
        xchi2pix = fx['CHI2PIX'].data
        fx.close()

        ii = xfibermap['FIBER'] % nspec

        flux[ii] = xflux
        ivar[ii] = xivar
        if R.shape[1] < xR.shape[1] :
            # not same number of diagonals, it can happen
            # make sure it's the same wavelength array size
            assert(R.shape[2]==xR.shape[2])
            newR = np.zeros((R.shape[0],xR.shape[1],R.shape[2]))
            ddiag=xR.shape[1]-R.shape[1]
            offset=ddiag//2
            newR[:,offset:-offset,:] = R
            R=newR

        if R.shape[1] > xR.shape[1] :
            # make sure it's the same wavelength array size
            assert(R.shape[2]==xR.shape[2])
            ddiag=R.shape[1]-xR.shape[1]
            offset=ddiag//2
            R[ii,offset:-offset,:] = xR
        else :
            R[ii] = xR

        mask[ii] = xmask
        chi2pix[ii] = xchi2pix

        if fibermap is None:
            fibermap = np.zeros(nspec, dtype=xfibermap.dtype)
            fibermap['FIBER'] = np.arange(fibermin, fibermin+nspec)

        fibermap[ii] = xfibermap

    #- Use fibermap header from first input file
    fm = Table.read(args.files[0], 'FIBERMAP')
    fibermap = Table(fibermap)
    fibermap.meta.update(fm.meta)

    #- Write it out
    print("Writing", args.output)
    frame = Frame(w, flux, ivar, mask=mask, resolution_data=R,
                spectrograph=spectrograph,
                meta=hdr, fibermap=fibermap, chi2pix=chi2pix)
    desispec.io.write_frame(args.output, frame)

    #- Scary!  Delete input files
    if args.delete:
        for filename in args.files:
            os.remove(filename)
