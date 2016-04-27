"""
Extract spectra from DESI pre-processed raw data
"""

from __future__ import absolute_import, division

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


def parse(options=None):
    parser = argparse.ArgumentParser(description="Extract spectra from pre-processed raw data.")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="input image")
    parser.add_argument("-f", "--fibermap", type=str, required=True,
                        help="input fibermap file")
    parser.add_argument("-p", "--psf", type=str, required=True,
                        help="input psf")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="output extracted spectra")
    parser.add_argument("-w", "--wavelength", type=str, required=False,
                        help="wavemin,wavemax,dw")
    parser.add_argument("-s", "--specmin", type=int, required=False, default=0,
                        help="first spectrum to extract")
    parser.add_argument("-n", "--nspec", type=int, required=False,
                        help="number of spectra to extract")
    parser.add_argument("-r", "--regularize", type="float", required=False, default=0.0,
                        help="regularization amount (%default)")
    parser.add_argument("--nwavestep", type=int, required=False, default=50,
                        help="number of wavelength steps per divide-and-conquer extraction step")
    parser.add_argument("-v", "--verbose", action="store_true", help="print more stuff")

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args):

    psf_file = args.psf
    input_file = args.input
    specmin = args.specmin
    nspec = args.nspec

    #- Load input files
    psf = load_psf(psf_file)
    img = io.read_image(input_file)

    if nspec is None:
        nspec = psf.nspec
    specmax = specmin + nspec

    camera = img.meta['CAMERA']     #- b0, r1, .. z9
    spectrograph = int(camera[1])
    fibermin = spectrograph*500 + specmin

    print('Starting {} spectra {}:{} at {}'.format(os.path.basename(input_file),
        specmin, specmin+nspec, time.asctime()))

    if args.fibermap is not None:
        fibermap = io.read_fibermap(args.fibermap)
        fibermap = fibermap[fibermin:fibermin+nspec]
        fibers = fibermap['FIBER']
    else:
        fibermap = None
        fibers = np.arange(fibermin, fibermin+nspec, dtype='i4')

    #- Get wavelength grid from options
    if args.wavelength is not None:
        wstart, wstop, dw = map(float, args.wavelength.split(','))
    else:
        wstart = np.ceil(psf.wmin_all)
        wstop = np.floor(psf.wmax_all)
        dw = 0.5
        
    wave = np.arange(wstart, wstop+dw/2.0, dw)
    nwave = len(wave)
    bundlesize = 25     #- hardcoded for DESI; could have gotten from desimodel

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
    """.format(input=input_file, psf=psf_file, output=args.output,
        wstart=wstart, wstop=wstop, dw=dw,
        specmin=specmin, nspec=nspec,
        regularize=args.regularize)

    #- The actual extraction
    flux, ivar, Rdata = ex2d(img.pix, img.ivar*(img.mask==0), psf, specmin, nspec, wave,
                 regularize=args.regularize, ndecorr=True,
                 bundlesize=bundlesize, wavesize=args.nwavestep, verbose=args.verbose)

    #- Util function to trim path to something that fits in a fits file (!)                            
    def _trim(filepath, maxchar=40):
        if len(filepath) > maxchar:
            return '...'+filepath[-maxchar:]

    #- Augment input image header for output
    img.meta['NSPEC']   = (nspec, 'Number of spectra')
    img.meta['WAVEMIN'] = (wstart, 'First wavelength [Angstroms]')
    img.meta['WAVEMAX'] = (wstop, 'Last wavelength [Angstroms]')
    img.meta['WAVESTEP']= (dw, 'Wavelength step size [Angstroms]')
    img.meta['SPECTER'] = (specter.__version__, 'https://github.com/desihub/specter')
    img.meta['IN_PSF']  = (_trim(psf_file), 'Input spectral PSF')
    img.meta['IN_IMG']  = (_trim(input_file), 'Input image')

    frame = Frame(wave, flux, ivar, resolution_data=Rdata,
                fibers=fibers, meta=img.meta, fibermap=fibermap)

    #- Write output
    io.write_frame(args.output, frame)

    print('Done {} spectra {}:{} at {}'.format(os.path.basename(input_file),
        specmin, specmin+nspec, time.asctime()))


