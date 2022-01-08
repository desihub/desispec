"""
desispec.fiberflat_vs_humidity
==================

Utility functions to compute a fiber flat corrected for variations with humidity in the shack
"""
from __future__ import absolute_import, division


import numpy as np
import copy

from desiutil.log import get_logger
from desispec.fiberflat import apply_fiberflat
from desispec.fiberbitmasking import get_skysub_fiberbitmask_val

def _interpolated_fiberflat_vs_humidity(fiberflat_vs_humidity , humidity_array, humidity_point) :
    """
    Interpolates between fiberflat templates indexed by humidity.

    Args:
        fiberflat_vs_humidity: 3D numpy array (n_humidity,n_fibers,n_wavelength)
        humidity_array: 1D numpy array (n_humidity)
        humidity_point: float, humidity value (same unit as humidity_array)

    Returns 2D numpy array (n_fibers,n_wavelength)
    """
    if humidity_point<=humidity_array[0] :
        i1=0
    else :
        i1=np.where(humidity_array<humidity_point)[0][-1]
    i2=i1+1
    if i2>=humidity_array.size : # return largest value
        return fiberflat_vs_humidity[-1]

    w1=(humidity_array[i2]-humidity_point)/(humidity_array[i2]-humidity_array[i1])
    w2=(humidity_point-humidity_array[i1])/(humidity_array[i2]-humidity_array[i1])
    return w1*fiberflat_vs_humidity[i1]+w2*fiberflat_vs_humidity[i2]

def _fit_flat(wavelength,flux,ivar,fibers,mean_fiberflat_vs_humidity,humidity_array) :
    """
    Finds best fit interpolation of fiberflat templates that matches an input flux frame
    Works only if wavelength array intersects the range [4000,4600]A, i.e. the blue cameras

    Args:
        wavelength: 1D numpy array (n_wavelength) in Angstrom
        flux: 2D numpy array (n_fibers,n_wavelength) unit does not matter
        ivar: 2D numpy array (n_fibers,n_wavelength) inverse variance of flux
        fibers: list or 1D number arrays of fibers to use among range(n_fibers)
        mean_fiberflat_vs_humidity: 3D numpy array (n_humidity,n_fibers,n_wavelength)
        humidity_array: 1D numpy array (n_humidity)

    Returns best_fit_flat best_fit_humidity (2D numpy array (n_fibers,n_wavelength) and float)
    """
    log = get_logger()
    selection = (wavelength > 4000.) & (wavelength < 4600)
    if np.sum(selection)==0 :
        message="incorrect wavelength range"
        log.error(message)
        raise RuntimeError(message)
    waveindex = np.where(selection)[0]
    tmp_flux = flux[fibers][:,waveindex].copy()
    tmp_ivar = ivar[fibers][:,waveindex].copy()

    for loop in range(2) :
        # remove mean variation from fiber to fiber
        med   = np.median(tmp_flux,axis=-1)
        tmp_flux /= med[:,None]
        tmp_ivar *= med[:,None]**2
        # remove average over fibers
        med =  np.median(tmp_flux,axis=0)
        tmp_flux /= med[None,:]
        tmp_ivar *= med[None,:]**2

    tmp_flat = mean_fiberflat_vs_humidity[:,fibers][:,:,waveindex].copy()

    for loop in range(2) :
        # remove mean variation from fiber to fiber
        med   = np.median(tmp_flat,axis=-1)
        tmp_flat /= med[:,:,None]
        # remove average over fibers
        for index in range(tmp_flat.shape[0]) :
            med =  np.median(tmp_flat[index],axis=0)
            tmp_flat[index] /= med[None,:]

    # chi2 between all fiberflat templates (one per humidity bin)
    # with current flux value
    # summed over all fibers and all wavelength
    # after having both the fiberflat and the flux normalized to 1
    # per fiber when averaged over all wavelength
    # and per wavelength when averaged over all fibers
    chi2 = np.sum(np.sum(tmp_ivar*(tmp_flux-tmp_flat)**2,axis=-1),axis=-1)

    # chi2 is a 1D array with size = number of humidity bins

    # index of minimum, but then we refine
    minindex=np.argmin(chi2)

    bb=minindex-1
    ee=minindex+2
    if bb<0 :
        bb+=1
        ee+=1
    if ee>=chi2.size :
        bb-=1
        ee-=1

    # get the chi2 minimum
    c=np.polyfit(humidity_array[bb:ee],chi2[bb:ee],2)
    best_humidity = -c[1]/2./c[0]
    best_humidity = max(humidity_array[0],best_humidity)
    best_humidity = min(humidity_array[-1],best_humidity)
    log.info("best fit humidity = {:.2f}".format(best_humidity))

    # simple linear interpolation indexed by the humidity
    flat = _interpolated_fiberflat_vs_humidity(mean_fiberflat_vs_humidity , humidity_array, best_humidity)

    return flat , best_humidity

def compute_humidity_corrected_fiberflat(calib_fiberflat, mean_fiberflat_vs_humidity , humidity_array, current_humidity, frame) :
    """
    Apply a humidity-dependent correction to an input fiber flat
    Returns frame_fiberflat = calib_fiberflat / flat_vs_humidity_model(calib) * flat_vs_humidity_model(frame)

    Args:
        calib_fiberflat: desispec.FiberFlat object
        mean_fiberflat_vs_humidity: 3D numpy array (n_humidity,n_fibers,n_wavelength)
        humidity_array: 1D numpy array (n_humidity)
        current_humidity: float (same unit as humidity_array)
        frame: desispec.Frame object

    Returns modified desispec.FiberFlat object
    """
    log = get_logger()

    best_humidity = current_humidity

    log.info("using nightly flat to fit for the best fit nightly flat humidity")
    selection = np.sum(calib_fiberflat.ivar!=0,axis=1)>10
    good_flat_fibers = np.where(selection)[0]
    flat2 , hum2 = _fit_flat(calib_fiberflat.wave,calib_fiberflat.fiberflat,calib_fiberflat.ivar,good_flat_fibers,mean_fiberflat_vs_humidity,humidity_array)

    flat1 = None
    hum1  = current_humidity
    if frame is not None :
        log.info("using frame to fit for the best fit current humidity")
        ivar = frame.ivar*(frame.mask==0)
        badfibermask = get_skysub_fiberbitmask_val()
        selection = (frame.fibermap["OBJTYPE"]=="SKY") & (frame.fibermap["FIBERSTATUS"] & badfibermask == 0) & (np.sum(ivar!=0,axis=1)>10)
        if np.sum(selection)>0 :
            good_sky_fibers = np.where(selection)[0]
            heliocor=frame.meta['HELIOCOR']
            frame_wave_in_fiberflat_system  = frame.wave/heliocor
            tmp_flux = frame.flux.copy()
            tmp_ivar = ivar.copy()
            for fiber in good_sky_fibers:
                ok=(ivar[fiber]>0)
                tmp_flux[fiber] = np.interp(frame.wave,frame_wave_in_fiberflat_system[ok],frame.flux[fiber][ok])
                tmp_ivar[fiber] = np.interp(frame.wave,frame_wave_in_fiberflat_system[ok],ivar[fiber][ok])
            flat1 , hum1  = _fit_flat(frame.wave,tmp_flux*flat2/calib_fiberflat.fiberflat,tmp_ivar,good_sky_fibers,mean_fiberflat_vs_humidity,humidity_array)
    if flat1 is None :
        log.info("use input humidity = {:.2f}".format(current_humidity))
        flat1  = _interpolated_fiberflat_vs_humidity(mean_fiberflat_vs_humidity , humidity_array, current_humidity)

    # apply humidity correction to current calib fiberflat
    fiberflat = copy.deepcopy(calib_fiberflat)
    fiberflat.fiberflat = calib_fiberflat.fiberflat/flat2*flat1
    fiberflat.header["EXPTHUM"] = (current_humidity,"exposure humidity from telemetry")
    fiberflat.header["EXPFHUM"] = (hum1,"exposure humidity from flat fit")
    fiberflat.header["CALFHUM"] = (hum2,"dome flat humidity from flat fit")

    if np.abs(hum1-current_humidity)>10 :
        message="large difference between best fit humidity during science exposure ({:.1f}) and value from telemetry ({:.1f})".format(hum1,current_humidity)
        if np.abs(hum1-current_humidity)>20 :
            log.error(message)
            raise RuntimeError(message)
        log.warning(message)


    return fiberflat
