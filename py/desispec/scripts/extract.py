"""
Extract spectra from DESI pre-processed raw data
"""

from __future__ import absolute_import, division, print_function

import sys
import traceback
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
from desiutil.log import get_logger
from desispec.frame import Frame
from desispec.maskbits import specmask

import desispec.scripts.mergebundles as mergebundles
from desispec.specscore import compute_and_append_frame_scores


def parse(options=None):
    parser = argparse.ArgumentParser(description="Extract spectra from pre-processed raw data.")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="input image")
    parser.add_argument("-f", "--fibermap", type=str, required=False,
                        help="input fibermap file")
    parser.add_argument("-p", "--psf", type=str, required=True,
                        help="input psf file")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="output extracted spectra file")
    parser.add_argument("-m", "--model", type=str, required=False,
                        help="output 2D pixel model file")
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
    parser.add_argument("--nsubbundles", type=int, required=False, default=6,
                        help="number of extraction sub-bundles")
    parser.add_argument("--nwavestep", type=int, required=False, default=50,
                        help="number of wavelength steps per divide-and-conquer extraction step")
    parser.add_argument("-v", "--verbose", action="store_true", help="print more stuff")
    parser.add_argument("--mpi", action="store_true", help="Use MPI for parallelism")
    parser.add_argument("--decorrelate-fibers", action="store_true", help="Not recommended")

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

    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        return main_mpi(args, comm)

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

    camera = img.meta['CAMERA'].lower()     #- b0, r1, .. z9
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
        wstart, wstop, dw = [float(tmp) for tmp in args.wavelength.split(',')]
    else:
        wstart = np.ceil(psf.wmin_all)
        wstop = np.floor(psf.wmax_all)
        dw = 0.7

    wave = np.arange(wstart, wstop+dw/2.0, dw)
    nwave = len(wave)
    bundlesize = args.bundlesize

    #- Confirm that this PSF covers these wavelengths for these spectra
    psf_wavemin = np.max(psf.wavelength(list(range(specmin, specmax)), y=0))
    psf_wavemax = np.min(psf.wavelength(list(range(specmin, specmax)), y=psf.npix_y-1))
    if psf_wavemin > wstart:
        raise ValueError('Start wavelength {:.2f} < min wavelength {:.2f} for these fibers'.format(wstart, psf_wavemin))
    if psf_wavemax < wstop:
        raise ValueError('Stop wavelength {:.2f} > max wavelength {:.2f} for these fibers'.format(wstop, psf_wavemax))

    #- Print parameters
    print("""\
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
        regularize=args.regularize))

    #- The actual extraction
    results = ex2d(img.pix, img.ivar*(img.mask==0), psf, specmin, nspec, wave,
                 regularize=args.regularize, ndecorr=args.decorrelate_fibers,
                 bundlesize=bundlesize, wavesize=args.nwavestep, verbose=args.verbose,
                 full_output=True, nsubbundles=args.nsubbundles)
    flux = results['flux']
    ivar = results['ivar']
    Rdata = results['resolution_data']
    chi2pix = results['chi2pix']

    mask = np.zeros(flux.shape, dtype=np.uint32)
    mask[results['pixmask_fraction']>0.5] |= specmask.SOMEBADPIX
    mask[results['pixmask_fraction']==1.0] |= specmask.ALLBADPIX
    mask[chi2pix>100.0] |= specmask.BAD2DFIT

    #- Augment input image header for output
    img.meta['NSPEC']   = (nspec, 'Number of spectra')
    img.meta['WAVEMIN'] = (wstart, 'First wavelength [Angstroms]')
    img.meta['WAVEMAX'] = (wstop, 'Last wavelength [Angstroms]')
    img.meta['WAVESTEP']= (dw, 'Wavelength step size [Angstroms]')
    img.meta['SPECTER'] = (specter.__version__, 'https://github.com/desihub/specter')
    img.meta['IN_PSF']  = (_trim(psf_file), 'Input spectral PSF')
    img.meta['IN_IMG']  = (_trim(input_file), 'Input image')

    frame = Frame(wave, flux, ivar, mask=mask, resolution_data=Rdata,
                fibers=fibers, meta=img.meta, fibermap=fibermap,
                chi2pix=chi2pix)
    
    #- Add unit
    #   In specter.extract.ex2d one has flux /= dwave
    #   to convert the measured total number of electrons per
    #   wavelength node to an electron 'density'
    frame.meta['BUNIT'] = 'electrons/Angstrom' 
    
    #- Add scores to frame
    compute_and_append_frame_scores(frame,suffix="RAW",calibrated=False)

    #- Write output
    io.write_frame(args.output, frame)

    if args.model is not None:
        from astropy.io import fits
        fits.writeto(args.model, results['modelimage'], header=frame.meta, clobber=True)

    print('Done {} spectra {}:{} at {}'.format(os.path.basename(input_file),
        specmin, specmin+nspec, time.asctime()))


#- TODO: The level of repeated code from main() is problematic, e.g. the
#- recent addition of mask and chi2pix code required nearly identical edits
#- in two places.  Could main(args) just call main_mpi(args, comm=None) ?

def main_mpi(args, comm=None):

    log = get_logger()

    psf_file = args.psf
    input_file = args.input

    # these parameters are interpreted as the *global* spec range,
    # to be divided among processes.
    specmin = args.specmin
    nspec = args.nspec

    #- Load input files and broadcast

    # FIXME: after we have fixed the serialization
    # of the PSF, read and broadcast here, to reduce
    # disk contention.

    img = None
    if comm is None:
        img = io.read_image(input_file)
    else:
        if comm.rank == 0:
            img = io.read_image(input_file)
        img = comm.bcast(img, root=0)

    psf = load_psf(psf_file)

    # get spectral range

    if nspec is None:
        nspec = psf.nspec
    specmax = specmin + nspec

    camera = img.meta['CAMERA'].lower()     #- b0, r1, .. z9
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
        wstart, wstop, dw = [float(tmp) for tmp in args.wavelength.split(',')]
    else:
        wstart = np.ceil(psf.wmin_all)
        wstop = np.floor(psf.wmax_all)
        dw = 0.7

    wave = np.arange(wstart, wstop+dw/2.0, dw)
    nwave = len(wave)

    #- Confirm that this PSF covers these wavelengths for these spectra

    psf_wavemin = np.max(psf.wavelength(list(range(specmin, specmax)), y=0))
    psf_wavemax = np.min(psf.wavelength(list(range(specmin, specmax)), y=psf.npix_y-1))
    if psf_wavemin > wstart:
        raise ValueError('Start wavelength {:.2f} < min wavelength {:.2f} for these fibers'.format(wstart, psf_wavemin))
    if psf_wavemax < wstop:
        raise ValueError('Stop wavelength {:.2f} > max wavelength {:.2f} for these fibers'.format(wstop, psf_wavemax))

    # Now we divide our spectra into bundles

    bundlesize = args.bundlesize
    checkbundles = set()
    checkbundles.update(np.floor_divide(np.arange(specmin, specmax), bundlesize*np.ones(nspec)).astype(int))
    bundles = sorted(checkbundles)
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

    mynbundle = int(nbundle // nproc)
    myfirstbundle = 0
    leftover = nbundle % nproc
    if rank < leftover:
        mynbundle += 1
        myfirstbundle = rank * mynbundle
    else:
        myfirstbundle = ((mynbundle + 1) * leftover) + (mynbundle * (rank - leftover))

    if rank == 0:
        #- Print parameters
        log.info("extract:  input = {}".format(input_file))
        log.info("extract:  psf = {}".format(psf_file))
        log.info("extract:  specmin = {}".format(specmin))
        log.info("extract:  nspec = {}".format(nspec))
        log.info("extract:  wavelength = {},{},{}".format(wstart, wstop, dw))
        log.info("extract:  nwavestep = {}".format(args.nwavestep))
        log.info("extract:  regularize = {}".format(args.regularize))

    # get the root output file

    outpat = re.compile(r'(.*)\.fits')
    outmat = outpat.match(args.output)
    if outmat is None:
        raise RuntimeError("extraction output file should have .fits extension")
    outroot = outmat.group(1)

    outdir = os.path.normpath(os.path.dirname(outroot))
    if rank == 0:
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

    if comm is not None:
        comm.barrier()

    failcount = 0

    for b in range(myfirstbundle, myfirstbundle+mynbundle):
        outbundle = "{}_{:02d}.fits".format(outroot, b)
        outmodel = "{}_model_{:02d}.fits".format(outroot, b)

        log.info('extract:  Rank {} starting {} spectra {}:{} at {}'.format(
            rank, os.path.basename(input_file),
            bspecmin[b], bspecmin[b]+bnspec[b], time.asctime(),
            ) )
        sys.stdout.flush()

        #- The actual extraction
        try:
            results = ex2d(img.pix, img.ivar*(img.mask==0), psf, bspecmin[b],
                bnspec[b], wave, regularize=args.regularize, ndecorr=args.decorrelate_fibers,
                bundlesize=bundlesize, wavesize=args.nwavestep, verbose=args.verbose,
                full_output=True, nsubbundles=args.nsubbundles)

            flux = results['flux']
            ivar = results['ivar']
            Rdata = results['resolution_data']
            chi2pix = results['chi2pix']

            mask = np.zeros(flux.shape, dtype=np.uint32)
            mask[results['pixmask_fraction']>0.5] |= specmask.SOMEBADPIX
            mask[results['pixmask_fraction']==1.0] |= specmask.ALLBADPIX
            mask[chi2pix>100.0] |= specmask.BAD2DFIT

            #- Augment input image header for output
            img.meta['NSPEC']   = (nspec, 'Number of spectra')
            img.meta['WAVEMIN'] = (wstart, 'First wavelength [Angstroms]')
            img.meta['WAVEMAX'] = (wstop, 'Last wavelength [Angstroms]')
            img.meta['WAVESTEP']= (dw, 'Wavelength step size [Angstroms]')
            img.meta['SPECTER'] = (specter.__version__, 'https://github.com/desihub/specter')
            img.meta['IN_PSF']  = (_trim(psf_file), 'Input spectral PSF')
            img.meta['IN_IMG']  = (_trim(input_file), 'Input image')

            if fibermap is not None:
                bfibermap = fibermap[bspecmin[b]-specmin:bspecmin[b]+bnspec[b]-specmin]
            else:
                bfibermap = None

            bfibers = fibers[bspecmin[b]-specmin:bspecmin[b]+bnspec[b]-specmin]

            frame = Frame(wave, flux, ivar, mask=mask, resolution_data=Rdata,
                        fibers=bfibers, meta=img.meta, fibermap=bfibermap,
                        chi2pix=chi2pix)

            #- Write output
            frame.meta['BUNIT'] = 'photon/bin'
            io.write_frame(outbundle, frame)

            if args.model is not None:
                from astropy.io import fits
                fits.writeto(outmodel, results['modelimage'], header=frame.meta)

            log.info('extract:  Done {} spectra {}:{} at {}'.format(os.path.basename(input_file),
                bspecmin[b], bspecmin[b]+bnspec[b], time.asctime()))
            sys.stdout.flush()
        except:
            # Log the error and increment the number of failures
            log.error("extract:  FAILED bundle {}, spectrum range {}:{}".format(b, bspecmin[b], bspecmin[b]+bnspec[b]))
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            log.error(''.join(lines))
            failcount += 1
            sys.stdout.flush()

    if comm is not None:
        failcount = comm.allreduce(failcount)

    if failcount > 0:
        # all processes throw
        raise RuntimeError("some extraction bundles failed")

    if rank == 0:
        mergeopts = [
            '--output', args.output,
            '--force',
            '--delete'
        ]
        mergeopts.extend([ "{}_{:02d}.fits".format(outroot, b) for b in bundles ])
        mergeargs = mergebundles.parse(mergeopts)
        mergebundles.main(mergeargs)

        if args.model is not None:
            model = None
            for b in bundles:
                outmodel = "{}_model_{:02d}.fits".format(outroot, b)
                if model is None:
                    model = fits.getdata(outmodel)
                else:
                    #- TODO: test and warn if models overlap for pixels with
                    #- non-zero values
                    model += fits.getdata(outmodel)

                os.remove(outmodel)

            fits.writeto(args.model, model)
