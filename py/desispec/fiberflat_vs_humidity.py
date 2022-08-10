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

def fit_wave_of_dip(wave1d,flux2d) :

    assert(len(flux2d.shape)==2)
    assert(flux2d.shape[0]==500)
    assert(flux2d.shape[1]==wave1d.shape[0])

    # 2D -> 1D
    mflux=(np.median(flux2d[0:200],axis=0)+np.median(flux2d[300:500],axis=0))/2.
    flux1d=np.median(flux2d[200:300],axis=0)
    flux1d/=mflux

    # minimum
    ii=np.where((wave1d>4250)&(wave1d<4500))[0]
    i=np.argmin(flux1d[ii])
    mwave1=wave1d[ii[i]]

    # parabola fit
    ii=np.abs(wave1d-mwave1)<10
    x=wave1d[ii]
    y=flux1d[ii]
    c=np.polyfit(x,y,2)
    mwave2=-c[1]/(2*c[0])

    return mwave2

def _interpolated_fiberflat_vs_humidity(fiberflat_vs_humidity , humidity_indices, humidity_index) :
    """
    Interpolates between fiberflat templates indexed by humidity.

    Args:
        fiberflat_vs_humidity: 3D numpy array (n_humidity,n_fibers,n_wavelength)
        humidity_indices: 1D numpy array (n_humidity)
        humidity_index: float, humidity index value (same unit as humidity_indices)

    Returns 2D numpy array (n_fibers,n_wavelength)
    """
    if humidity_index<=humidity_indices[0] :
        i1=0
    else :
        i1=np.where(humidity_indices<humidity_index)[0][-1]
    i2=i1+1
    if i2>=humidity_indices.size : # return largest value
        return fiberflat_vs_humidity[-1]

    w1=(humidity_indices[i2]-humidity_index)/(humidity_indices[i2]-humidity_indices[i1])
    w2=(humidity_index-humidity_indices[i1])/(humidity_indices[i2]-humidity_indices[i1])
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
    best_index=np.argmin(chi2)
    humidity_indices = np.arange(humidity_array.size,dtype=float)

    if best_index==0 or best_index==humidity_array.size-1 :

        best_humidity = humidity_array[best_index]
        flat = mean_fiberflat_vs_humidity[best_index]
        log.warning("best fit at edge of model humidity range")

    else :

        bb=best_index-1
        ee=best_index+2
        if bb<0 :
            bb+=1
            ee+=1
        if ee>chi2.size :
            bb-=1
            ee-=1

        # get the chi2 minimum, using the indices, not the humidity values which can fluctuate
        c=np.polyfit(humidity_indices[bb:ee],chi2[bb:ee],2)
        best_index = -c[1]/2./c[0]
        best_index = max(0,best_index)
        best_index = min(chi2.size-1,best_index)
        best_humidity = np.interp(best_index,humidity_indices,humidity_array)

        # simple linear interpolation indexed by the humidity
        flat = _interpolated_fiberflat_vs_humidity(mean_fiberflat_vs_humidity , humidity_indices, best_index)

    log.info("best fit index = {} , humidity = {:.2f}".format(best_index, best_humidity))

    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.plot(humidity_indices,chi2,"o")
    #plt.axvline(best_index)
    #plt.show()

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

    # only consider model humidity within 20% of measured humidity
    humidity_selection = np.abs(humidity_array - current_humidity)<20

    log.info("using nightly flat to fit for the best fit nightly flat humidity")
    selection = np.sum(calib_fiberflat.ivar!=0,axis=1)>10
    good_flat_fibers = np.where(selection)[0]
    flat2, hum2 = _fit_flat(calib_fiberflat.wave, calib_fiberflat.fiberflat, calib_fiberflat.ivar,
            good_flat_fibers, mean_fiberflat_vs_humidity[humidity_selection], humidity_array[humidity_selection])

    flat1 = None
    hum1  = current_humidity
    if frame is not None :
        log.info("using frame to fit for the best fit current humidity")
        ivar = frame.ivar*(frame.mask==0)
        band = 'brz' # all by default
        if frame.meta is not None :
            if "CAMERA" in frame.meta.keys() :
                camera = frame.meta["CAMERA"].lower()
                band   = camera[0]
        badfibermask = get_skysub_fiberbitmask_val(band=band)
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
            flat1, hum1  = _fit_flat(frame.wave, tmp_flux*flat2/calib_fiberflat.fiberflat, tmp_ivar, good_sky_fibers,
                    mean_fiberflat_vs_humidity[humidity_selection], humidity_array[humidity_selection])
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
