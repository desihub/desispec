"""
desispec.fibercrosstalk
=======================

Utility functions to correct for the fibercrosstalk

"""
from __future__ import absolute_import, division

import numpy as np
from importlib import resources
import yaml
from scipy.signal import fftconvolve

from desiutil.log import get_logger

from desispec.io import read_xytraceset
from desispec.calibfinder import CalibFinder
from desispec.specscore import append_frame_scores
from desispec.maskbits import specmask,fibermask

def compute_crosstalk_kernels(max_fiber_offset=2,fiber_separation_in_pixels=7.3,asymptotic_power_law_index = 2.5):
    """
    Computes the fiber crosstalk convolution kernels assuming a power law PSF tail

    Args:
       max_fiber_offset, optional : positive int, maximum fiber offset, 2 by default
       fiber_separation_in_pixels, optional : float, distance between neighboring fiber traces in the CCD in pixels, default=7.3
       asymptotic_power_law_index, optional : float, power law index of PSF tail

    Returns:
        A dictionnary of kernels, with key the positive fiber offset 1,2,.... Each entry is an 1D array.
    """
    # assume PSF tail shape (tuned to measured PSF tail in NIR)
    asymptotic_power_law_index = 2.5

    hw=100 # pixel
    dy  = np.linspace(-hw,hw,2*hw+1)

    kernels={}
    for fiber_offset in range(1,max_fiber_offset+1) :
        dx  = fiber_offset * fiber_separation_in_pixels
        r2 = dx**2+dy**2
        kern = r2 / (1. + r2)**(1+asymptotic_power_law_index/2.0)
        kern /= np.sum(kern)
        kernels[fiber_offset]=kern
    return kernels

def eval_crosstalk(camera,wave,fibers,dfiber,params,apply_scale=True,nfiber_per_bundle=25) :
    """
    Computes the crosstalk as a function of wavelength from a fiber offset dfiber (positive and negative) for an input set of fibers

    Args:
        camera : str, camera identifier (b8,r7,z3, ...)
        wave : 1D array, wavelength
        fibers : list or 1D array of int, list of contaminated fibers
        dfiber : int, positive or negative fiber offset, contaminating fibers = contaminated fibers + dfiber
        params : nested dictionnary, parameters of the crosstalk model
        apply_scale : boolean, optional apply or not the scale factor if found in the list of parameters
        nfiber_per_bundle : int, optional number of fibers per bundle, only the fibers in the same bundle are considered

    Returns:
        2D array of crosstalk fraction (between 0 and 1) of shape ( len(fibers),len(wave) )
    """
    log = get_logger()

    camera=camera.upper()
    if camera in params :
        cam=camera
    else :
        cam=camera[0] # same for all

    if cam[0] == "B" or cam[0] == "R" :
        W0=params[cam]["W0"]
        W1=params[cam]["W1"]
        DFIBER="F{:+d}".format(dfiber)
        P0=params[cam][DFIBER]["P0"]
        P1=params[cam][DFIBER]["P1"]
    elif cam[0] == "Z" :
        W0=params[cam]["W0"]
        W1=params[cam]["W1"]
        WP=params[cam]["WP"]
        WP2=params[cam]["WP2"]
        DFIBER="F{:+d}".format(dfiber)
        P0=params[cam][DFIBER]["P0"]
        P1=params[cam][DFIBER]["P1"]
        P2=params[cam][DFIBER]["P2"]
        P3=params[cam][DFIBER]["P3"]
        P4=params[cam][DFIBER]["P4"]
        P5=params[cam][DFIBER]["P5"]
    else :
        mess = "not implemented!"
        log.critical(mess)
        raise RuntimeError(mess)

    nfibers=fibers.size
    xtalk=np.zeros((nfibers,wave.size))

    for index,into_fiber in enumerate(fibers) :
        from_fiber = into_fiber + dfiber

        if from_fiber//nfiber_per_bundle != into_fiber//nfiber_per_bundle : continue # not same bundle
        if cam[0] == "B" or cam[0] == "R" :
            fraction = P0*(W1-wave)/(W1-W0) +P1*(wave-W0)/(W1-W0)
        elif cam[0] == "Z" :
            dw=(wave>W0)*(np.abs(wave-W0)/(W1-W0))
            fraction = P0 + P1*(into_fiber/250-1) + (P2 + P3*(into_fiber/250-1)) * dw**WP + (P4 + P5*(into_fiber/250-1)) * dw**WP2
        fraction *= (fraction>0)
        xtalk[index]=fraction


    if apply_scale :
        if camera in params :
            if DFIBER in params[camera] :
                if "SCALE" in params[camera][DFIBER] :
                    scale=float(params[camera][DFIBER]["SCALE"])
                    log.debug("apply scale={:3.2f} for camera={} {}".format(scale,camera,DFIBER))
                    xtalk *= scale

    return xtalk


def compute_contamination(frame,dfiber,kernel,params,xyset,fiberflat=None,fractional_error=0.1) :
    """
    Computes the contamination of a frame from a given fiber offset

    Args:
        frame : a desispec.frame.Frame object
        dfiber : int, fiber offset (-2,-1,1,2)
        kernel : 1D numpy array, convolution kernel
        params : nested dictionnary, parameters of the crosstalk model
        xyset : desispec.xytraceset.XYTraceSet object with trace coordinates to shift the spectra
       fiberflat : desispec.fiberflat.FiberFlat object, optional if the frame has already been fiber flatfielded
       fractionnal_error : float, optional consider this systematic relative error on the correction

    Returns:
        contamination , contamination_var: the contamination of the frame,
        2D numpy array of same shape as frame.flux, and its variance
    """
    log = get_logger()

    camera = frame.meta["camera"]
    fibers = np.arange(frame.nspec,dtype=int)
    xtalk  = eval_crosstalk(camera,frame.wave,fibers,dfiber,params)

    contamination=np.zeros(frame.flux.shape)
    contamination_var=np.zeros(frame.flux.shape)
    central_y = xyset.npix_y//2
    nfiber_per_bundle = 25

    # dfiber = from_fiber - into_fiber
    # into_fiber = from_fiber - dfiber

    # do a simplified achromatic correction for the fiberflat here
    if fiberflat is not None :
        medflat=np.median(fiberflat.fiberflat,axis=1)

    # we can use the signal from the following fibers to compute the cross talk
    # because the only think that is bad about them is their position in the focal plane.
    would_be_ok = fibermask.STUCKPOSITIONER|fibermask.UNASSIGNED|fibermask.MISSINGPOSITION|fibermask.BADPOSITION

    fiberstatus = frame.fibermap["FIBERSTATUS"]
    fiber_should_be_considered = (fiberstatus==(fiberstatus&would_be_ok))

    for index,into_fiber in enumerate(fibers) :
        from_fiber = into_fiber + dfiber

        if from_fiber not in fibers : continue
        if from_fiber//nfiber_per_bundle != into_fiber//nfiber_per_bundle : continue # not same bundle
        if not fiber_should_be_considered[from_fiber] : continue

        fraction = xtalk[index]

        # keep fibers with mask=BADFIBER because we already discarded the fiber with bad status
        jj=(frame.ivar[from_fiber]>0)&((frame.mask[from_fiber]==0)|(frame.mask[from_fiber]==specmask.BADFIBER))

        from_fiber_central_wave = xyset.wave_vs_y(from_fiber,central_y)
        into_fiber_central_wave = xyset.wave_vs_y(into_fiber,central_y)

        nok=np.sum(jj)
        if nok<10 :
            log.warning("skip contaminating fiber {} because only {} valid flux values".format(from_fiber,nok))
            continue
        tmp=np.interp(frame.wave+from_fiber_central_wave-into_fiber_central_wave,frame.wave[jj],frame.flux[from_fiber,jj],left=0,right=0)
        if fiberflat is not None :
            tmp *= medflat[from_fiber] # apply median transmission of the contaminating fiber, i.e. undo the fiberflat correction

        convolved_flux=fftconvolve(tmp,kernel,mode="same")
        contamination[into_fiber] = fraction * convolved_flux

        # we cannot easily use the variance of the contaminant spectrum
        # we consider only a fractional error to reflect systematic errors in the fiber cross-talk correction
        contamination_var[into_fiber] = (fractional_error*contamination[into_fiber])**2

    if fiberflat is not None :
        # apply the fiberflat correction of the contaminated fibers
        for fiber in range(contamination.shape[0]) :
            if medflat[fiber]>0.1 :
                contamination[fiber] = contamination[fiber] / medflat[fiber]

    return contamination , contamination_var



def read_crosstalk_parameters(parameter_filename = None) :
    """
    Reads the crosstalk parameters in desispec/data/fiber-crosstalk.yaml

    Returns:
       nested dictionary with parameters per camera
    """
    log=get_logger()
    if parameter_filename is None :
        parameter_filename = resources.files('desispec').joinpath("data/fiber-crosstalk.yaml")
    log.info("read parameters in {}".format(parameter_filename))
    stream = open(parameter_filename, 'r')
    params = yaml.safe_load(stream)
    stream.close()
    log.debug("params= {}".format(params))
    return params

def correct_fiber_crosstalk(frame,fiberflat=None,xyset=None,parameter_filename=None):
    """Apply a fiber cross talk correction. Modifies frame.flux and frame.ivar.

    Args:
        frame : desispec.frame.Frame object
        fiberflat, optional : desispec.fiberflat.FiberFlat object
        xyset, optional : desispec.xytraceset.XYTraceSet object with trace
            coordinates to shift the spectra
            (automatically found with calibration finder otherwise)
        parameter_filename, optional : path to yaml file with correction parameters
    """
    log=get_logger()

    cfinder = None
    if parameter_filename is None :
        cfinder = CalibFinder([frame.meta])
        if cfinder.haskey("FIBERCROSSTALK") :
            parameter_filename = cfinder.findfile("FIBERCROSSTALK")
            log.debug("Using custom file "+parameter_filename)

    params = read_crosstalk_parameters(parameter_filename = parameter_filename)

    if xyset is None :
        if cfinder is None :
            cfinder = CalibFinder([frame.meta])
        psf_filename = cfinder.findfile("PSF")
        xyset  = read_xytraceset(psf_filename)

    log.info("compute kernels")
    kernels = compute_crosstalk_kernels()

    contamination     = np.zeros(frame.flux.shape)
    contamination_var = np.zeros(frame.flux.shape)

    for dfiber in [-2,-1,1,2] :
        log.info("F{:+d}".format(dfiber))
        kernel = kernels[np.abs(dfiber)]
        cont,var = compute_contamination(frame,dfiber,kernel,params,xyset,fiberflat)
        contamination     += cont
        contamination_var += var

    frame.flux -= contamination
    frame_var  = 1./(frame.ivar + (frame.ivar==0))
    frame.ivar = (frame.ivar>0)/( frame_var + contamination_var )

    ## document the contamination in the frame's scores table
    ## create ivar
    ivar = np.zeros(frame.flux.shape)

    ## null out masked values and zero variances
    valid_entries = (frame.mask==0)&(contamination_var>0.)
    ## bool arrays unravel 2d into 1d, so we need to get the indices
    valid_entries_inds = np.where(valid_entries)
    not_valid_entries_inds = np.where(~valid_entries)
    ## calc ivar for all valid entries and set rest to small value to avoid div by 0
    ivar[valid_entries_inds] = 1./contamination_var[valid_entries_inds]
    ivar[not_valid_entries_inds] = 1.0e-63 # to avoid division by zero later
    ## set contamination to zero for invalid entries
    contamination[not_valid_entries_inds] = 0.

    ## Compute the weighted mean and median
    band = frame.meta['CAMERA'][0].upper()

    ## Add them to the frame's scores table
    entries = {
        f'MEAN_FIB_XTALK_{band}': (np.sum(contamination*ivar, axis=1)/np.sum(ivar, axis=1)).astype(np.float32),
        f'MEDIAN_FIB_XTALK_{band}':  np.median(contamination, axis=1).astype(np.float32),
    }
    comments = {
        f'MEAN_FIB_XTALK_{band}': 'Inverse variance weighted mean fiber crosstalk contamination (in flux units)',
        f'MEDIAN_FIB_XTALK_{band}': 'Median fiber crosstalk contamination (in flux units)',
    }
    append_frame_scores(frame,entries,comments,overwrite=True)

    #################### TODO Hacks that should be removed later ##########################
    # from astropy.io import fits
    # from astropy.table import Table
    # import os
    # # Create the primary HDU
    # header = fits.Header()
    # for key in frame.meta.keys():
    #     header[key] = frame.meta[key]
    # primary_hdu = fits.PrimaryHDU(data=None, header=header)

    # # Create additional HDUs from the ndarrays
    # hdu1 = fits.ImageHDU(data=frame.flux.data, name='CONTAM_SUBD_FLUX')
    # hdu2 = fits.ImageHDU(data=frame.ivar.data, name='IVAR')
    # hdu3 = fits.ImageHDU(data=contamination, name='CONTAMINATION')
    # hdu4 = fits.ImageHDU(data=frame.mask.data, name='MASK')
    # hdu5 = fits.ImageHDU(data=frame.wave.data, name='WAVELENGTH')
    # scores_table = Table()
    # for key in frame.scores.keys():
    #     scores_table[key] = frame.scores[key]
    # hdu6 = fits.BinTableHDU(data=scores_table, name='SCORES')

    # # Create a FITS file, add the HDUs, and save it
    # specprod = os.environ.get('SPECPROD','unknownspecprod')
    # night = frame.meta.get('NIGHT','unknownnight')
    # tileid = frame.meta.get('TILEID','unknowntileid')
    # expid = frame.meta.get('EXPID','unknownexpid')
    # camera = frame.meta.get('CAMERA','unknowncamera')
    # with fits.HDUList([primary_hdu, hdu1, hdu2, hdu3, hdu4, hdu5, hdu6]) as hdul:
    #     hdul.writeto(os.path.join('/global/cfs/projectdirs/desi/users/kremin/workspace/contaminated_fibers/modified_cframes', f"mod_cframe_{specprod}_{camera}_{night}_{tileid}_{expid}.fits"), overwrite=True)
    #################### /end Hacks that should be removed later ##########################

    ## if the 70th percentile of the absolute percent contamination is greater than 0.5, flag it
    contam_frac = np.abs(contamination/frame.flux)
    contaminated_fibers = np.quantile(contam_frac, 0.7, axis=1) > 0.5
    frame.fibermap['FIBERSTATUS'][contaminated_fibers] |= fibermask.MANYREJECTED  # set MANYREJECTED flag

    ## Flag spectral bins that are highly contaminated by fiber crosstalk
    frame.mask[contam_frac > 0.5] |= specmask.CONTAMINATED  # set CONTAMINATED flag

    # #################### TODO Hacks that should be removed later ##########################
    # quants, cuts, ncontam, fibs = [], [], [], []
    # for quant in np.arange(0.1,1.,0.1):
    #     for cut in np.arange(0.1,1.1,0.1):
    #         contam_mask = np.quantile(contam_frac, quant, axis=1) > cut
    #         quants.append(quant)
    #         cuts.append(cut)
    #         ncontam.append(np.sum(contam_mask))
    #         fibs.append(';'.join(frame.fibermap['FIBER'][contam_mask].data.astype(str)))
    #         print(f"{quant=:0.1f}, {cut=:0.1f}, ncontaminated={np.sum(contam_mask)}, fibers={frame.fibermap['FIBER'][contam_mask].data}")
    # from astropy.table import Table
    # import os
    # specprod = os.environ.get('SPECPROD','unknownspecprod')
    # night = frame.meta.get('NIGHT','unknownnight')
    # tileid = frame.meta.get('TILEID','unknowntileid')
    # expid = frame.meta.get('EXPID','unknownexpid')
    # camera = frame.meta.get('CAMERA','unknowncamera')
    # t = Table()
    # t['QUANTILE'] = np.array(quants).astype(np.float16)
    # t['CUT'] = np.array(cuts).astype(np.float16)
    # t['NCONTAMINATED'] = np.array(ncontam).astype(np.int16)
    # t['FIBERS'] = fibs
    # t['NIGHT'] = np.int32(night) if str(night).isdigit() else -1
    # t['TILEID'] = np.int32(tileid) if str(tileid).isdigit() else -1
    # t['EXPID'] = np.int32(expid) if str(expid).isdigit() else -1
    # t['CAMERA'] = camera
    # t.write(os.path.join(os.environ['SCRATCH'], 'contaminated_fiber_tables', f"contaminated_fibers_table_{specprod}_{camera}_{night}_{tileid}_{expid}.fits"), overwrite=True)
    # #################### /end Hacks that should be removed later ##########################