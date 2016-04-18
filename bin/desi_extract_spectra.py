#!/usr/bin/env python

"""
Extract spectra from DESI pre-processed raw data
"""

import sys
import os
import os.path
import time
import numpy as np

import specter
from specter.psf import load_psf
from specter.extract import ex2d

from desispec import io
from desispec.frame import Frame

import optparse
parser = optparse.OptionParser(usage = "%prog [options]")
parser.add_option("-i", "--input", type="string",  help="input image")
parser.add_option("-f", "--fibermap", type="string",  help="input fibermap file")
parser.add_option("-p", "--psf", type="string",  help="input psf")
parser.add_option("-o", "--output", type="string",  help="output extracted spectra")
parser.add_option("-w", "--wavelength", type="string",  help="wavemin,wavemax,dw")
parser.add_option("-s", "--specmin", type=int,  help="first spectrum to extract", default=0)
parser.add_option("-n", "--nspec", type=int,  help="number of spectra to extract")
parser.add_option("-r", "--regularize", type="float",  help="regularization amount (%default)", default=0.0)
parser.add_option("--nwavestep", type=int,  help="number of wavelength steps per divide-and-conquer extraction step", default=50)
parser.add_option("-v", "--verbose", action="store_true", help="print more stuff")
### parser.add_option("-x", "--xxx",   help="some flag", action="store_true")

opts, args = parser.parse_args()

#- Load input files
psf = load_psf(opts.psf)
img = io.read_image(opts.input)

camera = img.meta['CAMERA']     #- b0, r1, .. z9
spectrograph = int(camera[1])
fibermin = spectrograph*500+opts.specmin

print('Starting {} spectra {}:{} at {}'.format(os.path.basename(opts.input),
    opts.specmin, opts.specmin+opts.nspec, time.asctime()))

if opts.fibermap is not None:
    fibermap = io.read_fibermap(opts.fibermap)
    fibermap = fibermap[fibermin:fibermin+opts.nspec]
    fibers = fibermap['FIBER']
else:
    fibermap = None
    fibers = np.arange(fibermin, fibermin+opts.nspec, dtype='i4')

#- Get wavelength grid from options
if opts.wavelength is not None:
    wstart, wstop, dw = map(float, opts.wavelength.split(','))
else:
    wstart = np.ceil(psf.wmin_all)
    wstop = np.floor(psf.wmax_all)
    dw = 0.5
    
wave = np.arange(wstart, wstop+dw/2.0, dw)
nwave = len(wave)
bundlesize = 25     #- hardcoded for DESI; could have gotten from desimodel

#- Get specrange from options
specmin, specmax = opts.specmin, opts.specmin + opts.nspec

#- Confirm that this PSF covers these wavelengths for these spectra
psf_wavemin = np.max(psf.wavelength(range(specmin, specmax), y=0))
psf_wavemax = np.min(psf.wavelength(range(specmin, specmax), y=psf.npix_y-1))
if psf_wavemin > wstart:
    raise ValueError, 'Start wavelength {:.2f} < min wavelength {:.2f} for these fibers'.format(wstart, psf_wavemin)
if psf_wavemax < wstop:
    raise ValueError, 'Stop wavelength {:.2f} > max wavelength {:.2f} for these fibers'.format(wstop, psf_wavemax)

#- Print parameters
print """\
#--- Extraction Parameters ---
input:      {input}
psf:        {psf}
output:     {output}
wavelength: {wstart} - {wstop} AA steps {dw}
specmin:    {specmin}
nspec:      {nspec}
regularize: {regularize}
#-----------------------------\
""".format(input=opts.input, psf=opts.psf, output=opts.output,
    wstart=wstart, wstop=wstop, dw=dw,
    specmin=opts.specmin, nspec=opts.nspec,
    regularize=opts.regularize)

#- The actual extraction
flux, ivar, Rdata = ex2d(img.pix, img.ivar, psf, opts.specmin, opts.nspec, wave,
             regularize=opts.regularize, ndecorr=True,
             bundlesize=bundlesize, wavesize=opts.nwavestep, verbose=opts.verbose)

#- Util function to trim path to something that fits in a fits file (!)                            
def _trim(filepath, maxchar=40):
    if len(filepath) > maxchar:
        return '...'+filepath[-maxchar:]

#- Augment input image header for output
img.meta['NSPEC']   = (opts.nspec, 'Number of spectra')
img.meta['WAVEMIN'] = (wstart, 'First wavelength [Angstroms]')
img.meta['WAVEMAX'] = (wstop, 'Last wavelength [Angstroms]')
img.meta['WAVESTEP']= (dw, 'Wavelength step size [Angstroms]')
img.meta['SPECTER'] = (specter.__version__, 'https://github.com/desihub/specter')
img.meta['IN_PSF']  = (_trim(opts.psf), 'Input spectral PSF')
img.meta['IN_IMG']  = (_trim(opts.input), 'Input image')

frame = Frame(wave, flux, ivar, resolution_data=Rdata,
            fibers=fibers, meta=img.meta, fibermap=fibermap)

#- Write output
io.write_frame(opts.output, frame)

print('Done {} spectra {}:{} at {}'.format(os.path.basename(opts.input),
    opts.specmin, opts.specmin+opts.nspec, time.asctime()))



