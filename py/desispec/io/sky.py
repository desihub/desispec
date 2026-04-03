"""
desispec.io.sky
===============

IO routines for sky.
"""
from __future__ import absolute_import, division
import os
import time
from astropy.io import fits
from desiutil.log import get_logger

from . import iotime
from .util import get_tempfilename

def write_sky(outfile, skymodel, header=None):
    """Write sky model.

    Args:
        outfile : filename or (night, expid, camera) tuple
        skymodel : SkyModel object, with the following attributes
        wave : 1D wavelength in vacuum Angstroms
        flux : 2D[nspec, nwave] sky flux
        ivar : 2D inverse variance of sky flux
        mask : 2D mask for sky flux
        stat_ivar : 2D inverse variance of sky flux (statistical only)
        dwavecoeff : 1D[ncoeff] array of PCA dwavelength coefficients
            (optional)
        dlsfcoeff : 1D[ncoeff] array of PCA dlsf coefficients (optional)
        peak_dw_before : 2D[nspec, npeaks] wavelength offsets at peaks before PCA correction (optional)
        peak_dlsf_before : 2D[nspec, npeaks] LSF width changes at peaks before PCA correction (optional)
        peak_chi2pdf_before : 2D[nspec, npeaks] chi2/dof at peaks before PCA correction (optional)
        peak_dw_after : 2D[nspec, npeaks] wavelength offsets at peaks after PCA correction (optional)
        peak_dlsf_after : 2D[nspec, npeaks] LSF width changes at peaks after PCA correction (optional)
        peak_chi2pdf_after : 2D[nspec, npeaks] chi2/dof at peaks after PCA correction (optional)
        header : optional fits header data (fits.Header, dict, or list)
    """
    from desiutil.depend import add_dependencies
    from .util import fitsheader, makepath

    log = get_logger()
    outfile = makepath(outfile, 'sky')

    #- Convert header to fits.Header if needed
    if header is not None:
        hdr = fitsheader(header)
    else:
        hdr = fitsheader(skymodel.header)

    add_dependencies(hdr)

    hx = fits.HDUList()

    hdr['EXTNAME'] = ('SKY', 'no dimension')
    hx.append( fits.PrimaryHDU(skymodel.flux.astype('f4'), header=hdr) )
    hx.append( fits.ImageHDU(skymodel.ivar.astype('f4'), name='IVAR') )
    # hx.append( fits.CompImageHDU(skymodel.mask, name='MASK') )
    hx.append( fits.ImageHDU(skymodel.mask, name='MASK') )
    hx.append( fits.ImageHDU(skymodel.wave.astype('f4'), name='WAVELENGTH') )
    hx[-1].header['BUNIT'] = 'Angstrom'

    if skymodel.stat_ivar is not None :
       hx.append( fits.ImageHDU(skymodel.stat_ivar.astype('f4'), name='STATIVAR') )
    if skymodel.throughput_corrections is not None:
        hx.append( fits.ImageHDU(skymodel.throughput_corrections.astype('f4'), name='THRPUTCORR') )
    if skymodel.throughput_corrections_model is not None:
        hx.append( fits.ImageHDU(skymodel.throughput_corrections_model.astype('f4'), name='THRPUTCORR_MOD') )
    if skymodel.dwavecoeff is not None:
        hx.append(fits.ImageHDU(skymodel.dwavecoeff.astype('f4'),
                                name='DWAVECOEFF'))
    if skymodel.dlsfcoeff is not None:
        hx.append(fits.ImageHDU(skymodel.dlsfcoeff.astype('f4'),
                                name='DLSFCOEFF'))
    if skymodel.skygradpcacoeff is not None:
        hx.append(fits.ImageHDU(skymodel.skygradpcacoeff.astype('f4'),
                                name='SKYGRADPCACOEFF'))
    if skymodel.skytargetid is not None:
        hx.append(fits.ImageHDU(skymodel.skytargetid,
                                name='SKYTARGETID'))
    if skymodel.peak_dw_before is not None:
        hx.append(fits.ImageHDU(skymodel.peak_dw_before.astype('f4'),
                                name='PEAK_DW_BEFORE'))
    if skymodel.peak_dlsf_before is not None:
        hx.append(fits.ImageHDU(skymodel.peak_dlsf_before.astype('f4'),
                                name='PEAK_DLSF_BEFORE'))
    if skymodel.peak_chi2pdf_before is not None:
        hx.append(fits.ImageHDU(skymodel.peak_chi2pdf_before.astype('f4'),
                                name='PEAK_CHI2PDF_BEFORE'))
    if skymodel.peak_dw_after is not None:
        hx.append(fits.ImageHDU(skymodel.peak_dw_after.astype('f4'),
                                name='PEAK_DW_AFTER'))
    if skymodel.peak_dlsf_after is not None:
        hx.append(fits.ImageHDU(skymodel.peak_dlsf_after.astype('f4'),
                                name='PEAK_DLSF_AFTER'))
    if skymodel.peak_chi2pdf_after is not None:
        hx.append(fits.ImageHDU(skymodel.peak_chi2pdf_after.astype('f4'),
                                name='PEAK_CHI2PDF_AFTER'))


    t0 = time.time()
    tmpfile = get_tempfilename(outfile)
    hx.writeto(tmpfile, overwrite=True, checksum=True)
    os.rename(tmpfile, outfile)
    duration = time.time() - t0
    log.info(iotime.format('write', outfile, duration))

    return outfile

def read_sky(filename):
    """Read sky model and return SkyModel object with attributes
    wave, flux, ivar, mask, header.

    skymodel.wave is 1D common wavelength grid, the others are 2D[nspec, nwave]
    """
    from .meta import findfile
    from .util import native_endian, checkgzip
    from ..sky import SkyModel
    log = get_logger()
    #- check if filename is (night, expid, camera) tuple instead
    if not isinstance(filename, str):
        night, expid, camera = filename
        filename = findfile('sky', night, expid, camera)

    t0 = time.time()
    filename = checkgzip(filename)
    fx = fits.open(filename, memmap=False, uint=True)

    hdr = fx[0].header
    wave = native_endian(fx["WAVELENGTH"].data.astype('f8'))
    skyflux = native_endian(fx["SKY"].data.astype('f8'))
    ivar = native_endian(fx["IVAR"].data.astype('f8'))
    mask = native_endian(fx["MASK"].data)
    if "STATIVAR" in fx :
        stat_ivar = native_endian(fx["STATIVAR"].data.astype('f8'))
    else :
        stat_ivar = None
    if "THRPUTCORR" in fx :
        throughput_corrections = native_endian(fx["THRPUTCORR"].data.astype('f8'))
    else :
        throughput_corrections = None
    if "THRPUTCORR_MOD" in fx :
        throughput_corrections_model = native_endian(fx["THRPUTCORR_MOD"].data.astype('f8'))
    else :
        throughput_corrections_model = None
    if "DWAVECOEFF" in fx :
        dwavecoeff = native_endian(fx["DWAVECOEFF"].data.astype('f8'))
    else :
        dwavecoeff = None
    if "DLSFCOEFF" in fx :
        dlsfcoeff = native_endian(fx["DLSFCOEFF"].data.astype('f8'))
    else :
        dlsfcoeff = None
    if "SKYGRADPCACOEFF" in fx:
        skygradpcacoeff = native_endian(fx['SKYGRADPCACOEFF'].data.astype('f8'))
    else:
        skygradpcacoeff = None
    if "PEAK_DW_BEFORE" in fx:
        peak_dw_before = native_endian(fx['PEAK_DW_BEFORE'].data.astype('f8'))
    else:
        peak_dw_before = None
    if "PEAK_DLSF_BEFORE" in fx:
        peak_dlsf_before = native_endian(fx['PEAK_DLSF_BEFORE'].data.astype('f8'))
    else:
        peak_dlsf_before = None
    if "PEAK_CHI2PDF_BEFORE" in fx:
        peak_chi2pdf_before = native_endian(fx['PEAK_CHI2PDF_BEFORE'].data.astype('f8'))
    else:
        peak_chi2pdf_before = None
    if "PEAK_DW_AFTER" in fx:
        peak_dw_after = native_endian(fx['PEAK_DW_AFTER'].data.astype('f8'))
    else:
        peak_dw_after = None
    if "PEAK_DLSF_AFTER" in fx:
        peak_dlsf_after = native_endian(fx['PEAK_DLSF_AFTER'].data.astype('f8'))
    else:
        peak_dlsf_after = None
    if "PEAK_CHI2PDF_AFTER" in fx:
        peak_chi2pdf_after = native_endian(fx['PEAK_CHI2PDF_AFTER'].data.astype('f8'))
    else:
        peak_chi2pdf_after = None
    fx.close()
    duration = time.time() - t0
    log.info(iotime.format('read', filename, duration))

    skymodel = SkyModel(wave, skyflux, ivar, mask, header=hdr,stat_ivar=stat_ivar,
                        throughput_corrections=throughput_corrections,
                        throughput_corrections_model=throughput_corrections_model,
                        dwavecoeff=dwavecoeff, dlsfcoeff=dlsfcoeff,
                        skygradpcacoeff=skygradpcacoeff,
                        peak_dw_before=peak_dw_before, peak_dlsf_before=peak_dlsf_before,
                        peak_chi2pdf_before=peak_chi2pdf_before,
                        peak_dw_after=peak_dw_after, peak_dlsf_after=peak_dlsf_after,
                        peak_chi2pdf_after=peak_chi2pdf_after)

    return skymodel
