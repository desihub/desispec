"""
Utility functions for desispec.tests
"""
import numpy as np

from desispec.resolution import Resolution
from desispec.frame import Frame
from desispec.spectra import Spectra
from desispec.io import empty_fibermap

from desitarget.targetmask import desi_mask

def get_calib_from_frame(frame):
    """ Generate a FluxCalib object given an input Frame
    fc with essentially no error
    Args:
        frame: Frame

    Returns:
        fcalib: FluxCalib

    """
    from desispec.fluxcalibration import FluxCalib
    calib = np.ones_like(frame.flux)
    mask = np.zeros(frame.flux.shape, dtype=np.uint32)
    calib[0] *= 0.5
    fcivar = 1e20 * np.ones_like(frame.flux)
    fluxcalib = FluxCalib(frame.wave, calib, fcivar,mask)
    # Return
    return fluxcalib

def get_fiberflat_from_frame(frame):
    from desispec.fiberflat import FiberFlat
    flux = frame.flux
    fiberflat = np.ones_like(flux)
    ffivar = 2*np.ones_like(flux)
    fiberflat[0] *= 0.8
    fiberflat[1] *= 1.2
    ff = FiberFlat(frame.wave, fiberflat, ffivar)
    return ff

def get_frame_data(nspec=10, wavemin=4000, wavemax=4100, nwave=100, meta={}):
    """
    Return basic test data for desispec.frame object:
    """
    wave, model_flux = get_models(nspec, nwave, wavemin=wavemin, wavemax=wavemax)
    resol_data=set_resolmatrix(nspec,nwave)

    calib = np.sin((wave-wavemin) * np.pi / np.max(wave))
    flux = np.zeros((nspec, nwave))
    for i in range(nspec):
        flux[i] = Resolution(resol_data[i]).dot(model_flux[i] * calib)

    sigma = 0.01
    # flux += np.random.normal(scale=sigma, size=flux.shape)

    ivar = np.ones(flux.shape) / sigma**2
    mask = np.zeros(flux.shape, dtype=int)
    fibermap = empty_fibermap(nspec, 1500)
    fibermap['OBJTYPE'] = 'TGT'
    fibermap['DESI_TARGET'] = desi_mask.QSO
    fibermap['DESI_TARGET'][0:3] = desi_mask.STD_FAINT  # For flux tests
    fibermap['FIBER_X'] = np.arange(nspec)*400./nspec  #mm
    fibermap['FIBER_Y'] = np.arange(nspec)*400./nspec  #mm
    fibermap['DELTA_X'] = 0.005*np.ones(nspec)  #mm
    fibermap['DELTA_Y'] = 0.003*np.ones(nspec)  #mm


    if "EXPTIME" not in meta.keys():
        meta['EXPTIME'] = 1.0

    frame = Frame(wave, flux, ivar, mask,resol_data,fibermap=fibermap, meta=meta)
    return frame


def get_models(nspec=10, nwave=1000, wavemin=4000, wavemax=5000):
    """
    Returns basic model data:
    - [1D] modelwave [nmodelwave]
    - [2D] modelflux [nmodel,nmodelwave]
    """
    #make 20 models

    model_wave=np.linspace(wavemin, wavemax, nwave)
    y=np.sin(10*(model_wave-wavemin)/(wavemax-wavemin))+5.0
    model_flux=np.tile(y,nspec).reshape(nspec,len(model_wave))
    return model_wave,model_flux


def set_resolmatrix(nspec,nwave):
    """arguably typo function name, retained for backwards compatibility"""
    return get_resolmatrix(nspec, nwave)

def get_resolmatrix(nspec,nwave):
    """ Generate a Resolution Matrix
    Args:
        nspec: int
        nwave: int

    Returns:
        Rdata: np.array

    """
    sigma = np.linspace(2,10,nwave*nspec)
    ndiag = 21
    xx = np.linspace(-ndiag/2.0, +ndiag/2.0, ndiag)
    Rdata = np.zeros( (nspec, len(xx), nwave) )

    for i in range(nspec):
        for j in range(nwave):
            kernel = np.exp(-xx**2/(2*sigma[i*nwave+j]**2))
            kernel /= sum(kernel)
            Rdata[i,:,j] = kernel
    return Rdata

def get_resolmatrix_fixedsigma(nspec,nwave):
    """ Generate a Resolution Matrix with fixed sigma
    Args:
        nspec: int
        nwave: int

    Returns:
        Rdata: np.array

    """
    sigma = 3.0
    ndiag = 21
    xx = np.linspace(-ndiag/2.0, +ndiag/2.0, ndiag)
    kernel = np.exp(-xx**2/(2*sigma**2))
    kernel /= sum(kernel)
    Rdata = np.zeros( (nspec, len(xx), nwave) )

    for i in range(nspec):
        for j in range(nwave):
            Rdata[i,:,j] = kernel

    return Rdata

def get_blank_spectra(nspec):
    """Generate a blank spectrum object with realistic wavelength coverage"""

    wave = dict(
            b=np.arange(3600, 5800.1, 0.8),
            r=np.arange(5760, 7620.1, 0.8),
            z=np.arange(7520, 9824.1, 0.8),
            )
    bands = tuple(wave.keys())
    flux = dict()
    ivar = dict()
    mask = dict()
    rdat = dict()
    for band in bands:
        nwave = len(wave[band])
        flux[band] = np.ones((nspec, nwave))
        ivar[band] = np.zeros((nspec, nwave))
        mask[band] = np.zeros((nspec, nwave), dtype=np.int32)
        rdat[band] = get_resolmatrix_fixedsigma(nspec, nwave)

    fm = empty_fibermap(nspec)
    fm['FIBER'] = np.arange(nspec, dtype=np.int32)
    fm['TARGETID'] = np.arange(nspec, dtype=np.int64)

    sp = Spectra(bands=bands, wave=wave, flux=flux, ivar=ivar, mask=mask,
                 resolution_data=rdat, fibermap=fm)

    return sp


