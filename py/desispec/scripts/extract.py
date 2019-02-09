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

from distributed import Client

from dask import delayed, compute

from distributed.security import Security

sec = Security(tls_ca_file='./tmp/foo1/foo1.pem',
               tls_client_cert='./tmp/bar1/bar_cert.pem',
               tls_client_key='./tmp/bar1/bar_key.pem',
               require_encryption=True)


#sec=None

#client = Client(..., security=sec)
client = Client(scheduler_file="/global/cscratch1/sd/stephey/scheduler.json", security=sec)


#from distributed import Client, get_client
#client = get_client()
#client = Client(scheduler_file="/global/cscratch1/sd/stephey/scheduler.json")

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
    parser.add_argument("--no-scores", action="store_true", help="Do not compute scores")
    parser.add_argument("--psferr", type=float, default=None, required=False,
                        help="fractional PSF model error used to compute chi2 and mask pixels (default = value saved in psf file)")
    parser.add_argument("--fibermap-index", type=int, default=None, required=False,
                        help="start at this index in the fibermap table instead of using the spectro id from the camera")

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


def main(args, timing=None):

    mark_start = time.time()

    log = get_logger()

    psf_file = args.psf
    input_file = args.input

    # these parameters are interpreted as the *global* spec range,
    # to be divided among processes.
    specmin = args.specmin
    nspec = args.nspec
    #nspec = 50 #for testing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #- Load input files and broadcast

    # FIXME: after we have fixed the serialization
    # of the PSF, read and broadcast here, to reduce
    # disk contention.

    img = None
    #if comm is None:
    img = io.read_image(input_file)
    #else:
    #    if comm.rank == 0:
    #        img = io.read_image(input_file)
    #    img = comm.bcast(img, root=0)

    psf = load_psf(psf_file)

    mark_read_input = time.time()

    # get spectral range

    if nspec is None:
        nspec = psf.nspec
    specmax = specmin + nspec

    if args.fibermap_index is not None :
        fibermin = args.fibermap_index
    else :
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

    psf_wavemin = np.max(psf.wavelength(list(range(specmin, specmax)), y=-0.5))
    psf_wavemax = np.min(psf.wavelength(list(range(specmin, specmax)), y=psf.npix_y-0.5))
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

    #nproc = 1
    #rank = 0

    ##DASKIFY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #nproc = comm.size
    #rank = comm.rank

    ##ALSO THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #mynbundle = int(nbundle // nproc)
    #myfirstbundle = 0
    #leftover = nbundle % nproc
    #if rank < leftover:
    #    mynbundle += 1
    #    myfirstbundle = rank * mynbundle
    #else:
    #    myfirstbundle = ((mynbundle + 1) * leftover) + (mynbundle * (rank - leftover))

    #if rank == 0:
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
    #if rank == 0:
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    #if comm is not None:
    #    comm.barrier()

    mark_preparation = time.time()

    time_total_extraction = 0.0
    time_total_write_output = 0.0

    failcount = 0

    #for b in range(myfirstbundle, myfirstbundle+mynbundle):
    mark_iteration_start = time.time()

    #fix this recordkeeping later
    for b in bundles:
        outbundle = "{}_{:02d}.fits".format(outroot, b)
        outmodel = "{}_model_{:02d}.fits".format(outroot, b)

        #log.info('extract:  Rank {} starting {} spectra {}:{} at {}'.format(
        #    rank, os.path.basename(input_file),
        #    bspecmin[b], bspecmin[b]+bnspec[b], time.asctime(),
        #    ) )
        #sys.stdout.flush()

        #- The actual extraction


    print("bundles")
    print(bundles)

    ###this is the non mpi version, hopefully dask can make sense of this
    #results = ex2d(img.pix, img.ivar*(img.mask==0), psf, specmin, nspec, wave,
    #             regularize=args.regularize, ndecorr=args.decorrelate_fibers,
    #             bundlesize=bundlesize, wavesize=args.nwavestep, verbose=args.verbose,
    #               full_output=True, nsubbundles=args.nsubbundles,psferr=args.psferr).compute()

    ###this is the non mpi version, hopefully dask can make sense of this
    #futures = client.map(ex2d, [(img.pix, img.ivar*(img.mask==0), psf, bspecmin[b], bnspec[b], wave) for b in bundles])

    #results_tup = client.gather(futures)

    #futures = client.map(ex2d, [(img.pix, img.ivar*(img.mask==0), psf, bspecmin[b],
    #    bnspec[b], wave, regularize=args.regularize, ndecorr=args.decorrelate_fibers,
    #    bundlesize=bundlesize, wavesize=args.nwavestep, verbose=args.verbose,
    #    full_output=True, nsubbundles=args.nsubbundles) for b in bundles])

    #results_tup = client.gather(futures)

    ###DASKIFY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #this should become a delayed function
    #just distribute the number of bundles nbundle (usually 20) to tasks
    #starting dask delayed
    print("starting dask delayed for bundles")
    delayed_ex2d = [delayed(ex2d)(img.pix, img.ivar*(img.mask==0), psf, bspecmin[b],
        bnspec[b], wave, regularize=args.regularize, ndecorr=args.decorrelate_fibers,
        bundlesize=bundlesize, wavesize=args.nwavestep, verbose=args.verbose,
        full_output=True, nsubbundles=args.nsubbundles) for b in bundles]
    #
    results_tup = compute(*delayed_ex2d, scheduler='distributed')

    print("type(results_tup)")
    print(type(results_tup))
    print("len(results_tup)")   
    print(len(results_tup))
    #tuple is len 20 (one for each bundle)
    #print("results_tup")
    #print(results_tup)
    #make the results dict we'll eventually fill
    results = dict()
        
    ##preallocate numpy arrays of the correct size
    flux_array = np.zeros([nspec, nwave])
    ivar_array = np.zeros([nspec, nwave])
    resolution_array= np.zeros([nspec, 11, nwave]) #why 11?
    chi2pix_array = np.zeros([nspec, nwave])
    pixmask_array = np.zeros([nspec, nwave])

    ##need to convert tuple of dicts back into single dict i think
    for i, d in enumerate(results_tup):
        istart = i*25
        iend = istart + 25
        flux_array[istart:iend,:] = d['flux']
        ivar_array[istart:iend,:] = d['ivar']
        resolution_array[istart:iend,:,:] = d['resolution_data']
        chi2pix_array[istart:iend,:] = d['chi2pix']
        pixmask_array[istart:iend,:] = d['pixmask_fraction']

    #now put lists back into results dict
    #not elegant but it works, fix later
    results['flux'] = flux_array
    results['ivar'] = ivar_array
    results['resolution_data'] = resolution_array
    results['chi2pix'] = chi2pix_array
    results['pixmask_fraction'] = pixmask_array
            
    #print("type(results)")
    #print(type(results))

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

    #copy this part from the non-mpi code since we have assembled everything
    #back together (i.e. individual ranks aren't writing)
    frame = Frame(wave, flux, ivar, mask=mask, resolution_data=Rdata,
            fibers=fibers, meta=img.meta, fibermap=fibermap,
            chi2pix=chi2pix)

    #- Add unit
    #   In specter.extract.ex2d one has flux /= dwave
    #   to convert the measured total number of electrons per
    #   wavelength node to an electron 'density'
    frame.meta['BUNIT'] = 'count/Angstrom'

    mark_extraction = time.time()

    #- Add scores to frame
    if not args.no_scores :
        compute_and_append_frame_scores(frame,suffix="RAW")

    #- Write output
    io.write_frame(args.output, frame)

    if args.model is not None:
        from astropy.io import fits
        fits.writeto(args.model, results['modelimage'], header=frame.meta, overwrite=True)

    mark_write_output = time.time()

    print('Done {} spectra {}:{} at {}'.format(os.path.basename(input_file),
        specmin, specmin+nspec, time.asctime()))

    time_total_extraction += mark_extraction - mark_iteration_start
    time_total_write_output += mark_write_output - mark_extraction

    # Resolve difference timer data

    #if type(timing) is dict:
    #    timing["read_input"] = mark_read_input - mark_start
    #    timing["preparation"] = mark_preparation - mark_read_input
    #    timing["total_extraction"] = time_total_extraction
    #    timing["total_write_output"] = time_total_write_output
    #    timing["merge"] = time_merge



