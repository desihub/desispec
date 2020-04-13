"""
Utility functions for desispec.tests
"""
import numpy as np

from desispec.resolution import Resolution
from desispec.frame import Frame
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
