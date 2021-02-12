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

def compute_contamination(frame,dfiber,kernel,params) :

    cam = frame.meta["camera"][0].upper()
    if cam == "Z" :
        W0=params[cam]["W0"]
        W1=params[cam]["W1"]
        WP=params[cam]["WP"]
        DFIBER="FIBER{:+d}".format(dfiber)
        P0=params[cam][DFIBER]["P0"]
        P1=params[cam][DFIBER]["P1"]
        P2=params[cam][DFIBER]["P2"]
        P3=params[cam][DFIBER]["P3"]

        contamination=np.zeros(frame.flux.shape)

        # dfiber = from_fiber - into_fiber
        # into_fiber = from_fiber - dfiber
        minfiber = max(0,-dfiber)
        maxfiber = min(frame.flux.shape[0],frame.flux.shape[0]-dfiber)
        for index,into_fiber in enumerate(np.arange(minfiber,maxfiber)) :
            from_fiber = into_fiber + dfiber
            fraction = P0 + P1*(into_fiber/250-1) + (P2 + P3*(into_fiber/250-1)) * (frame.wave>W0)*(np.abs(frame.wave-W0)/(W1-W0))**WP
            jj=(frame.ivar[from_fiber]>0)&(frame.mask[from_fiber]==0)
            tmp=np.interp(frame.wave,frame.wave[jj],frame.flux[from_fiber,jj],left=0,right=0)
            convolved_flux=fftconvolve(tmp,kernel,mode="same")
            contamination[into_fiber] = fraction * convolved_flux
        return contamination

    else :
        mess = "not implemented!"
        log.critical(mess)
        raise RuntimeError(mess)



def correct_fiber_crosstalk(frame):
    """Apply a fiber cross talk correction. Modifies frame.flux and frame.ivar.

    Args:
        frame : `desispec.Frame` object
    """
    log=get_logger()

    parameter_filename = resource_filename('desispec', "data/fiber-crosstalk.yaml")
    log.info("reading parameters in {}".format(parameter_filename))
    stream = open(parameter_filename, 'r')
    params = yaml.safe_load(stream)
    stream.close()

    log.debug("params= {}".format(params))

    log.info("compute kernels")
    kernels = compute_crosstalk_kernels()

    contamination = np.zeros(frame.flux.shape)

    for dfiber in [-2,-1,1,2] :
        log.info("FIBER{:+d}".format(dfiber))
        kernel = kernels[np.abs(dfiber)]
        contamination += compute_contamination(frame,dfiber,kernel,params)

    import matplotlib.pyplot as plt
    for fiber in range(500) :
        plt.plot(frame.wave,contamination[fiber])
    plt.show()

    frame.flux -= contamination
