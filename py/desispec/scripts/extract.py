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

    if args.fibermap_index is not None :
        fibermin = args.fibermap_index
    else :
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
                   full_output=True, nsubbundles=args.nsubbundles,psferr=args.psferr)
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
    frame.meta['BUNIT'] = 'count/Angstrom'

    #- Add scores to frame
    if not args.no_scores :
        compute_and_append_frame_scores(frame,suffix="RAW")

    #- Write output
    io.write_frame(args.output, frame)

    if args.model is not None:
        from astropy.io import fits
        fits.writeto(args.model, results['modelimage'], header=frame.meta, overwrite=True)

    print('Done {} spectra {}:{} at {}'.format(os.path.basename(input_file),
        specmin, specmin+nspec, time.asctime()))


#- TODO: The level of repeated code from main() is problematic, e.g. the
#- recent addition of mask and chi2pix code required nearly identical edits
#- in two places.  Could main(args) just call main_mpi(args, comm=None) ?

#this should happen for each frame!!!! at the frame level!!!!
def get_subbundles(numfibers, fibers_per_bundle, subbundles_per_bundle):
    #should return specmin_n and keepmin_n
    #where specmin_n is a tuple (specmin, nspec)
    #where keepmin_n is a tuple (keepmin, nkeep)

    #divide fibers_per_bundle by subbundles_per_bundle
    #if we have a remainder add it to the first bundle
    nfloor = np.floor(fibers_per_bundle / subbundles_per_bundle)
    nremainder = fibers_per_bundle % subbundles_per_bundle  
    nbundles = numfibers//fibers_per_bundle    
 
    specmin_array = np.zeros([subbundles_per_bundle,1])
    nspec_array = np.zeros([subbundles_per_bundle,1])
    keepmin_array = np.zeros([subbundles_per_bundle,1])
    nkeep_array = np.zeros([subbundles_per_bundle, 1])
    
    for i in range(subbundles_per_bundle):
        #left end of bundle
        if i == 0:
            specmin_array[i] = 0
            nspec_array[i] = nfloor + nremainder + 1
            keepmin_array[i] = 0
            nkeep_array[i] = nfloor + nremainder
            
        #right end of bundle    
        elif i == (subbundles_per_bundle-1):    
            specmin_array[i] = specmin_array[i-1] + nspec_array[i-1] - 2
            nspec_array[i] = nfloor + 1
            keepmin_array[i] = keepmin_array[i-1] + nkeep_array[i-1]
            nkeep_array[i] = nfloor
            
        #everything else    
        else:
            specmin_array[i] = specmin_array[i-1] + nspec_array[i-1] - 2
            nspec_array[i] = nfloor + 2
            keepmin_array[i] = keepmin_array[i-1] + nkeep_array[i-1]
            nkeep_array[i] = nfloor 
                 
    #combine back into specmin_n (specmin, nspec)
    #combine back into keepmin_n (keepmin, nkeep)
    specmin_n = list()
    keepmin_n = list()
    for j in range(nbundles):
        bundle_offset = fibers_per_bundle*j
        for i in range(subbundles_per_bundle):
            specmin_n.append((int(specmin_array[i])+bundle_offset, int(nspec_array[i]))) 
            keepmin_n.append((int(keepmin_array[i])+bundle_offset, int(nkeep_array[i])))      
        
    return specmin_n, keepmin_n 

def main_mpi(args, comm=None, timing=None):

    mark_start = time.time()

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

    #adopt the usual mpi syntax
    nproc = 1
    rank = 0
    if comm is not None:
        nproc = comm.size
        rank = comm.rank

    img = None
    if comm is None:
        img = io.read_image(input_file)
    else:
        if comm.rank == 0:
            img = io.read_image(input_file)
        img = comm.bcast(img, root=0)

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

    mark_preparation = time.time()

    time_total_extraction = 0.0
    time_total_write_output = 0.0

    failcount = 0
    
    #probably these are args or could be determined from args
    #numfibers = 500 #this is the global nspec which i think is 500?
    numfibers = nspec
    fibers_per_bundle = args.bundlesize
    subbundles_per_bundle = args.nsubbundles
    nbundles = numfibers // fibers_per_bundle
    
    #instead of bundles lets divide into subbundles  
    specmin_n, keepmin_n = get_subbundles(numfibers, fibers_per_bundle, subbundles_per_bundle) 

    if comm is not None: 
        list_of_subbundles = np.concatenate((specmin_n, keepmin_n), axis=1)
    else:
        list_of_subbundles = np.concatenate((specmin_n, keepmin_n), axis=0)

    #based on this info we can create lists of subbundles for each rank to process
    #get size of frame communicator (32 for haswell, 68 for knl)
    #now distribute the subbundles among the ranks
    #here comm is the frame communicator
    my_subbundles = np.array_split(list_of_subbundles, nproc)[rank]

    #a little more bookkeeping for later
    each_nkeep = []
    my_subbundles_keepmin = my_subbundles[0][2]
    for b in my_subbundles:
        each_nkeep.append(b[3])
    my_subbundles_nkeep = np.sum(each_nkeep)

    #i think we do want to preallocate so we write only once per rank
    #not once per subbundle
    flux = np.zeros((my_subbundles_nkeep, nwave))
    ivar = np.zeros((my_subbundles_nkeep, nwave))
    chi2pix = np.zeros((my_subbundles_nkeep, nwave))
    Rdata = np.zeros((my_subbundles_nkeep, 11, nwave)) #why 11?
    pixmask_fraction = np.zeros((my_subbundles_nkeep, nwave))

    #we want this only once per rank
    outmysubbundles = "{}_{:02d}.fits".format(outroot, rank)
    outmodel = "{}_model_{:02d}.fits".format(outroot, rank)

    for b in my_subbundles:
        mark_iteration_start = time.time()
        
        #the four items stored for each subbundle are:
        #specmin, nspec, keepmin, nkeep
        sbspecmin = b[0] #subbundle specmin
        sbnspec = b[1] #subbundle nspec
        sbkeepmin = b[2] #subbundle keepmin
        sbnkeep = b[3] #subbundle nkeep

        log.info('extract:  Rank {} starting {} spectra {}:{} at {}'.format(
            rank, os.path.basename(input_file),
            sbspecmin, sbspecmin+sbnspec, time.asctime(),
            ) )
        sys.stdout.flush()

        #- The actual extraction, now over subbundles!
        #be careful with nsubbundles here... this is nsubbundles for specter
        #as a confusing workaround to avoid changing code in specter set nsubbundles=1
        try:
            results = ex2d(img.pix, img.ivar*(img.mask==0), psf, sbspecmin,
                sbnspec, wave, regularize=args.regularize, ndecorr=args.decorrelate_fibers,
                bundlesize=sbnkeep, wavesize=args.nwavestep, verbose=args.verbose,
                full_output=True, nsubbundles=1)
            
            #flux.shape(6, 3022) so it's sbnspec by len(wave)         
            sbflux = results['flux']
            sbivar = results['ivar']
            sbRdata = results['resolution_data']
            sbchi2pix = results['chi2pix']
            sbpixmask_fraction = results['pixmask_fraction']

            #print("sbflux.shape")
            #print(sbflux.shape)
            #print("sbivar.shape")
            #print(sbivar.shape)
            #print("sbRdata.shape")
            #print(sbRdata.shape)
            #print("sbchi2pix shape")
            #print(sbchi2pix.shape)

            #now trim down the results using sbkeepmin and sbnkeep
            #insert trimmed results into the right place in larger mysubbundle arrays
            trim = sbkeepmin - sbspecmin
            sboffset = sbkeepmin - my_subbundles_keepmin

            flux[sboffset:sboffset+sbnkeep, :] = sbflux[trim:trim + sbnkeep,:]
            ivar[sboffset:sboffset+sbnkeep, :] = sbivar[trim:trim + sbnkeep,:]
            #Rdata has an extra dimension, for example(6,11,3022)
            Rdata[sboffset:sboffset+sbnkeep] = sbRdata[trim:trim + sbnkeep, :, :]
            chi2pix[sboffset:sboffset+sbnkeep, :] = sbchi2pix[trim:trim + sbnkeep, :]
            pixmask_fraction[sboffset:sboffset+sbnkeep, :] = sbpixmask_fraction[trim:trim + sbnkeep, :]
        except:
            # Log the error and increment the number of failures
            log.error("extract:  FAILED bundle {}, spectrum range {}:{}".format(b, sbkeepmin, sbkeepmin+sbnkeep))
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            log.error(''.join(lines))
            failcount += 1
            sys.stdout.flush()

    #use our fully assembled data
    mask = np.zeros(flux.shape, dtype=np.uint32)
    mask[pixmask_fraction>0.5] |= specmask.SOMEBADPIX
    mask[pixmask_fraction==1.0] |= specmask.ALLBADPIX
    mask[chi2pix>100.0] |= specmask.BAD2DFIT

    #- Augment input image header for output
    #### not sure if i need to edit this or not
    img.meta['NSPEC']   = (nspec, 'Number of spectra')
    img.meta['WAVEMIN'] = (wstart, 'First wavelength [Angstroms]')
    img.meta['WAVEMAX'] = (wstop, 'Last wavelength [Angstroms]')
    img.meta['WAVESTEP']= (dw, 'Wavelength step size [Angstroms]')
    img.meta['SPECTER'] = (specter.__version__, 'https://github.com/desihub/specter')
    img.meta['IN_PSF']  = (_trim(psf_file), 'Input spectral PSF')
    img.meta['IN_IMG']  = (_trim(input_file), 'Input image')

    #use our bookkeeping from earlier
    if fibermap is not None:
        sbfibermap = fibermap[my_subbundles_keepmin-specmin:my_subbundles_keepmin+my_subbundles_nkeep-specmin]
    else:
        sbfibermap = None

    sbfibers = fibers[my_subbundles_keepmin-specmin:my_subbundles_keepmin+my_subbundles_nkeep-specmin]

    #do this once for all subbundles handled by a rank
    frame = Frame(wave, flux, ivar, mask=mask, resolution_data=Rdata,
                fibers=sbfibers, meta=img.meta, fibermap=sbfibermap,
                chi2pix=chi2pix)

    #- Add unit
    #   In specter.extract.ex2d one has flux /= dwave
    #   to convert the measured total number of electrons per
    #   wavelength node to an electron 'density'
    frame.meta['BUNIT'] = 'count/Angstrom'

    #- Add scores to frame
    compute_and_append_frame_scores(frame,suffix="RAW")

    mark_extraction = time.time()

    #- Write output
    #writes for each subbundle?-- changed this, be careful
    io.write_frame(outmysubbundles, frame)

    if args.model is not None:
        from astropy.io import fits
        #outmodel now inclues the rank number instead of the bundle number
        fits.writeto(outmodel, results['modelimage'], header=frame.meta)

    log.info('extract:  Done {} spectra {}:{} at {}'.format(os.path.basename(input_file),
        my_subbundles_keepmin, my_subbundles_keepmin + my_subbundles_nkeep, time.asctime()))
    sys.stdout.flush()

    mark_write_output = time.time()

    time_total_extraction += mark_extraction - mark_iteration_start
    time_total_write_output += mark_write_output - mark_extraction

    if comm is not None:
        failcount = comm.allreduce(failcount)

    if failcount > 0:
        # all processes throw
        raise RuntimeError("some extraction bundles failed")


    time_merge = None
    if rank == 0:
        mark_merge_start = time.time()
        mergeopts = [
            '--output', args.output,
            '--force',
            '--delete'
        ]

        #make a list of ranks
        rank_list = []
        for i in range(nproc):
            rank_list.append(i)
        #instead of bundles we now have work assigned to ranks
        mergeopts.extend([ "{}_{:02d}.fits".format(outroot, b) for b in rank_list])
        mergeargs = mergebundles.parse(mergeopts)
        mergebundles.main(mergeargs)

        if args.model is not None:
            model = None
            for b in outmysubbundles:
                outmodel = "{}_model_{:02d}.fits".format(outroot, b)
                if model is None:
                    model = fits.getdata(outmodel)
                else:
                    #- TODO: test and warn if models overlap for pixels with
                    #- non-zero values
                    model += fits.getdata(outmodel)

                os.remove(outmodel)

            fits.writeto(args.model, model)
        mark_merge_end = time.time()
        time_merge = mark_merge_end - mark_merge_start

    # Resolve difference timer data

    if type(timing) is dict:
        timing["read_input"] = mark_read_input - mark_start
        timing["preparation"] = mark_preparation - mark_read_input
        timing["total_extraction"] = time_total_extraction
        timing["total_write_output"] = time_total_write_output
        timing["merge"] = time_merge

