"""
desispec.fibercrosstalk
=======================

Utility functions to correct for the fibercrosstalk

"""
from __future__ import absolute_import, division

import numpy as np
from pkg_resources import resource_filename
import yaml
from scipy.signal import fftconvolve

from desiutil.log import get_logger

from desispec.io import read_xytraceset
from desispec.calibfinder import findcalibfile
from desispec.maskbits import specmask,fibermask

def compute_crosstalk_kernels(max_fiber_offset=2,fiber_separation_in_pixels=7.3,asymptotic_power_law_index = 2.5):
    """
    Computes the fiber crosstalk convolution kernels assuming a power law PSF tail
    Returns a dictionnary of kernels, with key the positive fiber offset 1,2,.... Each entry is an 1D array.

    Optionnal arguments:
       max_fiber_offset : positive int, maximum fiber offset, 2 by default
       fiber_separation_in_pixels : float, distance between neighboring fiber traces in the CCD in pixels, default=7.3
       asymptotic_power_law_index : float, power law index of PSF tail
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

    Optionnal:
      apply_scale : boolean, apply or not the scale factor if found in the list of parameters
      nfiber_per_bundle : number of fibers per bundle, only the fibers in the same bundle are considered

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
    Optionnal:
       fiberflat : desispec.fiberflat.FiberFlat object, if the frame has already been fiber flatfielded
       fractionnal_error : float, consider this systematic relative error on the correction
    Returns: contamination , contamination_var
       the contamination of the frame, 2D numpy array of same shape as frame.flux, and its variance
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

    # also include BADCOLUMN since those are a small effect (really bad columns
    # are part of the BADFIBER mask) so it is better to make some correction
    would_be_ok |= fibermask.BADCOLUMN

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



def read_crosstalk_parameters() :
    """
    Reads the crosstalk parameters in desispec/data/fiber-crosstalk.yaml
    Returns:
       nested dictionary with parameters per camera
    """
    log=get_logger()
    parameter_filename = resource_filename('desispec', "data/fiber-crosstalk.yaml")
    log.info("read parameters in {}".format(parameter_filename))
    stream = open(parameter_filename, 'r')
    params = yaml.safe_load(stream)
    stream.close()
    log.debug("params= {}".format(params))
    return params

def correct_fiber_crosstalk(frame,fiberflat=None,xyset=None):
    """Apply a fiber cross talk correction. Modifies frame.flux and frame.ivar.

    Args:
        frame : desispec.frame.Frame object

    Optionnal:
    fiberflat : desispec.fiberflat.FiberFlat object
        xyset : desispec.xytraceset.XYTraceSet object with trace coordinates to shift the spectra
                (automatically found with calibration finder otherwise)
    """
    log=get_logger()

    params = read_crosstalk_parameters()

    if xyset is None :
        psf_filename = findcalibfile([frame.meta,],"PSF")
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
