#!/usr/bin/env python

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
log = get_logger()

import optparse

parser = optparse.OptionParser(usage = "%prog [options] bundle1.fits bundle2.fits ...")
parser.add_option("-o", "--output", type="string",  help="output file name")
parser.add_option("-d", "--delete", help="delete input files when done", action="store_true")
parser.add_option("-f", "--force", help="merge files even if some fibers are missing", action="store_true")

opts, args = parser.parse_args()
nspec = 500  #- Hardcode!  Number of DESI fibers per spectrograph

#- Sanity check that all spectra are represented
fibers = set()
for filename in args:
    x = fits.getdata(filename, 'FIBERMAP')
    fibers.update( set(x['FIBER']) )

if len(fibers) != nspec:
    msg = "Input files only have {} instead of {} spectra".format(len(fibers), nspec)
    if opts.force:
        log.warn(msg)
    else:
        log.fatal(msg)
        sys.exit(1)

#- Read a file to get basic dimensions
w = fits.getdata(args[0], 'WAVELENGTH')
nwave = len(w)
R1 = fits.getdata(args[0], 'RESOLUTION')
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
print("Writing", opts.output)
frame = Frame(w, flux, ivar, resolution_data=R,
            spectrograph=spectrograph,
            meta=hdr, fibermap=fibermap)
desispec.io.write_frame(opts.output, frame)

#- Scary!  Delete input files
if opts.delete:
    for filename in args:
        os.remove(filename)
