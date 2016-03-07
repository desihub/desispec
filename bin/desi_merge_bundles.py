#!/usr/bin/env python

"""
exspec extracts individual bundles of spectra with one bundle per output file.
This script merges them back together into a single file combining all
bundles.

This workflow is hacky.  Release early, release often, but also refactor often.

Stephen Bailey, LBL
March 2014
"""

import sys
import os
import numpy as np
from astropy.io import fits

from desispec.frame import Frame
import desispec.io

import optparse

parser = optparse.OptionParser(usage = "%prog [options] bundle1.fits bundle2.fits ...")
parser.add_option("-o", "--output", type="string",  help="output file name")
parser.add_option("-f", "--fibermap", type="string",  help="input fibermap file")
parser.add_option("-d", "--delete", help="delete input files when done", action="store_true")

opts, args = parser.parse_args()
nspec = 500  #- Hardcode!

#- Sanity check that all spectra are represented
specset = set()
for filename in args:
    xhdr = fits.getheader(filename)    
    specset.update( set(range(xhdr['SPECMIN'], xhdr['SPECMAX']+1)) )

if len(specset) != nspec:
    print "Input files only have {} instead of {} spectra".format(len(specset), nspec)
    sys.exit(1)

#- Read a file to get basic dimensions
w = fits.getdata(args[0], 'WAVELENGTH')
nwave = len(w)
R1 = fits.getdata(args[0], 'RESOLUTION')
ndiag = R1.shape[1]
hdr = fits.getheader(args[0])
hdr['SPECMIN'] = 0
hdr['SPECMAX'] = nspec-1
hdr['NSPEC'] = nspec

camera = hdr['CAMERA']     #- b0, r1, .. z9
spectrograph = int(camera[1])
fibermin = spectrograph*nspec
fibers = np.arange(fibermin, fibermin+nspec, dtype='i4')

if opts.fibermap is not None:
    fibermap = desispec.io.read_fibermap(opts.fibermap)
    fibermap = fibermap[fibermin:fibermin+nspec]
else:
    fibermap = None

#- Output arrays to fill
flux = np.zeros( (nspec, nwave) )
ivar = np.zeros( (nspec, nwave) )
R = np.zeros( (nspec, ndiag, nwave) )

#- Fill them!
for filename in args:
    fx = fits.open(filename)
    xhdr = fx[0].header
    xflux = fx['FLUX'].data
    xivar = fx['IVAR'].data
    xR = fx['RESOLUTION'].data
    fx.close()
    
    lo = xhdr['SPECMIN']
    hi = xhdr['SPECMAX']+1
    ### print filename, lo, hi
    flux[lo:hi] = xflux
    ivar[lo:hi] = xivar
    R[lo:hi] = xR
    
#- Write it out
print "Writing", opts.output
frame = Frame(w, flux, ivar, resolution_data=R,
            fibers=fibers, spectrograph=spectrograph,
            meta=hdr, fibermap=fibermap)
desispec.io.write_frame(opts.output, frame)

#- Scary!  Delete input files
if opts.delete:
    for filename in args:
        os.remove(filename)
