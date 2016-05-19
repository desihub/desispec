"""
Extract spectra from DESI pre-processed raw data
"""

from __future__ import absolute_import, division

import sys
import os
import re
import os.path
import time
import argparse
import numpy as np

import specter
from specter.psf import load_psf
from specter.extract import ex2d

from desispec import io
from desispec.frame import Frame

import desispec.scripts.mergebundles as merge


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
    parser.add_argument("-r", "--regularize", type=float, required=False, default=0.0,
                        help="regularization amount (default %(default)s)")
    parser.add_argument("--bundlesize", type=int, required=False, default=25,
                        help="number of spectra per bundle")
    parser.add_argument("--nwavestep", type=int, required=False, default=50,
                        help="number of wavelength steps per divide-and-conquer extraction step")
    parser.add_argument("-v", "--verbose", action="store_true", help="print more stuff")

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


#- Util function to trim path to something that fits in a fits file (!)
def _trim(filepath, maxchar=40):
    if len(filepath) > maxchar:
        return '...{}'.format(filepath[-maxchar:])


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
    fibermin = spectrograph * psf.nspec + specmin

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
    bundlesize = args.bundlesize

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


def main_mpi(args, comm=None):

    psf_file = args.psf
    input_file = args.input

    # these parameters are interpreted as the *global* spec range,
    # to be divided among processes.
    specmin = args.specmin
    nspec = args.nspec

    #- Load input files and broadcast

    psf = None
    img = None
    if comm is None:
        psf = load_psf(psf_file)
        img = io.read_image(input_file)
    else:
        if comm.rank == 0:
            psf = load_psf(psf_file)
            img = io.read_image(input_file)
        psf = comm.bcast(psf, root=0)
        img = comm.bcast(img, root=0)

    # get spectral range

    if nspec is None:
        nspec = psf.nspec
    specmax = specmin + nspec

    camera = img.meta['CAMERA']     #- b0, r1, .. z9
    spectrograph = int(camera[1])
    fibermin = spectrograph * psf.nspec + specmin

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

    #- Confirm that this PSF covers these wavelengths for these spectra
    
    psf_wavemin = np.max(psf.wavelength(range(specmin, specmax), y=0))
    psf_wavemax = np.min(psf.wavelength(range(specmin, specmax), y=psf.npix_y-1))
    if psf_wavemin > wstart:
        raise ValueError, 'Start wavelength {:.2f} < min wavelength {:.2f} for these fibers'.format(wstart, psf_wavemin)
    if psf_wavemax < wstop:
        raise ValueError, 'Stop wavelength {:.2f} > max wavelength {:.2f} for these fibers'.format(wstop, psf_wavemax)

    # Now we divide our spectra into bundles

    bundlesize = args.bundlesize
    checkbundles = set()
    checkbundles.update(np.floor_divide(np.arange(specmin, specmax), bundlesize*np.ones(nspec)).astype(int))
    bundles = sorted(list(checkbundles))
    nbundle = len(bundles)

    bspecmin = {}
    bnspec = {}
    for b in bundles:
        if specmin > b * bundlesize:
            bspecmin[b] = specmin
        else:
            bspecmin[b] = b * bundlesize
        if (b+1) * bundlesize > specmax:
            bnspec[b] = specmax - bspecmin[b]
        else:
            bnspec[b] = bundlesize

    # Now we assign bundles to processes

    nproc = 1
    rank = 0
    if comm is not None:
        nproc = comm.size
        rank = comm.rank

    mynbundle = int(nbundle / nproc)
    myfirstbundle = 0
    leftover = nbundle % nproc
    if rank < leftover:
        mynbundle += 1
        myfirstbundle = rank * mynbundle
    else:
        myfirstbundle = ((mynbundle + 1) * leftover) + (mynbundle * (rank - leftover))

    if rank == 0:
        #- Print parameters
        print "extract:  input = {}".format(input_file)
        print "extract:  psf = {}".format(psf_file)
        print "extract:  specmin = {}".format(specmin)
        print "extract:  nspec = {}".format(nspec)
        print "extract:  wavelength = {},{},{}".format(wstart, wstop, dw)
        print "extract:  nwavestep = {}".format(args.nwavestep)
        print "extract:  regularize = {}".format(args.regularize)

    # get the root output file

    outpat = re.compile(r'(.*)\.fits')
    outmat = outpat.match(args.output)
    if outmat is None:
        raise RuntimeError("extraction output file should have .fits extension")
    outroot = outmat.group(1)

    for b in range(myfirstbundle, myfirstbundle+mynbundle):
        outbundle = "{}_{:02d}.fits".format(outroot, b)

        print('extract:  Starting {} spectra {}:{} at {}'.format(os.path.basename(input_file),
        bspecmin[b], bspecmin[b]+bnspec[b], time.asctime()))

        #- The actual extraction
        flux, ivar, Rdata = ex2d(img.pix, img.ivar*(img.mask==0), psf, bspecmin[b], 
            bnspec[b], wave, regularize=args.regularize, ndecorr=True,
            bundlesize=bundlesize, wavesize=args.nwavestep, verbose=args.verbose)

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
        io.write_frame(outbundle, frame)

        print('extract:  Done {} spectra {}:{} at {}'.format(os.path.basename(input_file),
            specmin, specmin+nspec, time.asctime()))

    if rank == 0:
        opts = [
            '--output', args.output,
            '--force',
            '--delete'
        ]
        opts.extend([ "{}_{:02d}.fits".format(outroot, b) for b in bundles ])
        args = merge.parse(opts)
        merge.main(args)
