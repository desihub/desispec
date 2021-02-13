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

def compute_crosstalk_kernels(max_fiber_offset=2,fiber_separation_in_pixels=7.3,asymptotic_power_law_index = 2.5):
    """
    Computes the fiber crosstalk convolution kernels assuming a power law PSF tail
    Returns a dictionnary of kernels, with key the positive fiber offset 1,2,.... Each entry is an 1D array.
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

def compute_contamination(frame,dfiber,kernel,params,xyset) :

    cam = frame.meta["camera"][0].upper()


    if cam == "B" or cam == "R" :
        W0=params[cam]["W0"]
        W1=params[cam]["W1"]
        DFIBER="FIBER{:+d}".format(dfiber)
        P0=params[cam][DFIBER]["P0"]
        P1=params[cam][DFIBER]["P1"]
    elif cam == "Z" :
        W0=params[cam]["W0"]
        W1=params[cam]["W1"]
        WP=params[cam]["WP"]
        WP2=params[cam]["WP2"]
        DFIBER="FIBER{:+d}".format(dfiber)
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

    contamination=np.zeros(frame.flux.shape)
    central_y = xyset.npix_y//2
    nfiber_per_bundle = 25

    # dfiber = from_fiber - into_fiber
    # into_fiber = from_fiber - dfiber
    minfiber = max(0,-dfiber)
    maxfiber = min(frame.flux.shape[0],frame.flux.shape[0]-dfiber)
    for index,into_fiber in enumerate(np.arange(minfiber,maxfiber)) :
        from_fiber = into_fiber + dfiber

        if from_fiber//nfiber_per_bundle != into_fiber//nfiber_per_bundle : continue # not same bundle

        if cam == "B" or cam == "R" :
            fraction = P0*(W1-frame.wave)/(W1-W0) +P1*(frame.wave-W0)/(W1-W0)
        elif cam == "Z" :
            dw=(frame.wave>W0)*(np.abs(frame.wave-W0)/(W1-W0))
            fraction = P0 + P1*(into_fiber/250-1) + (P2 + P3*(into_fiber/250-1)) * dw**WP + (P4 + P5*(into_fiber/250-1)) * dw**WP2
        fraction *= (fraction>0)
        jj=(frame.ivar[from_fiber]>0)&(frame.mask[from_fiber]==0)

        from_fiber_central_wave = xyset.wave_vs_y(from_fiber,central_y)
        into_fiber_central_wave = xyset.wave_vs_y(into_fiber,central_y)

        tmp=np.interp(frame.wave+from_fiber_central_wave-into_fiber_central_wave,frame.wave[jj],frame.flux[from_fiber,jj],left=0,right=0)
        convolved_flux=fftconvolve(tmp,kernel,mode="same")
        contamination[into_fiber] = fraction * convolved_flux
    return contamination





def correct_fiber_crosstalk(frame,xyset=None):
    """Apply a fiber cross talk correction. Modifies frame.flux and frame.ivar.

    Args:
        frame : `desispec.Frame` object
    """
    log=get_logger()

    parameter_filename = resource_filename('desispec', "data/fiber-crosstalk.yaml")
    log.info("read parameters in {}".format(parameter_filename))
    stream = open(parameter_filename, 'r')
    params = yaml.safe_load(stream)
    stream.close()
    log.debug("params= {}".format(params))

    if xyset is None :
        psf_filename = findcalibfile([frame.meta,],"PSF")
        xyset  = read_xytraceset(psf_filename)

    log.info("compute kernels")
    kernels = compute_crosstalk_kernels()

    contamination = np.zeros(frame.flux.shape)

    for dfiber in [-2,-1,1,2] :
        log.info("FIBER{:+d}".format(dfiber))
        kernel = kernels[np.abs(dfiber)]
        contamination += compute_contamination(frame,dfiber,kernel,params,xyset)

    frame.flux -= contamination
