"""
io routines for flux calibration

"""
import os
from astropy.io import fits
from desispec.io.util import fitsheader, native_endian, makepath
import numpy,scipy

# this is really temporary
# the idea is to have a datamodel for calibration stars spectra
def read_stellar_models(filename) :
    """
    read stellar models from filename
    
    returns flux[nspec, nwave], wave[nwave], fibers[nspec]
    """
    flux = native_endian(fits.getdata(filename, 0))
    wave = native_endian(fits.getdata(filename, 1))
    fibers = native_endian(fits.getdata(filename, 2))
    return flux,wave,fibers


def write_flux_calibration(outfile,calibration, calibration_ivar, mask, convolved_calibration, convolved_calibration_ivar,wave,header=None):
    """
    writes  flux calibration 
    """
    hdr = fitsheader(header)
    hdr['EXTNAME'] = ('CALIB', 'CHECK UNIT')
    fits.writeto(outfile,calibration,header=hdr, clobber=True)
    
    hdr['EXTNAME'] = ('IVAR', 'CHECK UNIT')
    hdu = fits.ImageHDU(calibration_ivar, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('MASK', 'no dimension')
    hdu = fits.ImageHDU(mask, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    
    hdr['EXTNAME'] = ('CCALIB', 'CHECK UNIT')
    hdu = fits.ImageHDU(convolved_calibration, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('CIVAR', 'CHECK UNIT')
    hdu = fits.ImageHDU(convolved_calibration_ivar, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('WAVELENGTH', '[Angstroms]')
    hdu = fits.ImageHDU(wave, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
def read_flux_calibration(filename) :

    """
    read flux calibration
    """
    calibration=native_endian(fits.getdata(filename, 0))
    calib_ivar=native_endian(fits.getdata(filename, "IVAR"))
    mask=native_endian(fits.getdata(filename, "MASK"))
    convolved_calibration=native_endian(fits.getdata(filename, "CCALIB"))
    convolved_calib_ivar=native_endian(fits.getdata(filename, "CIVAR"))
    wave=native_endian(fits.getdata(filename, "WAVELENGTH"))
    
    return calibration,calib_ivar,mask,convolved_calibration,convolved_calib_ivar,wave


def loadStellarModels(stellarmodelfile):

    phdu=fits.open(stellarmodelfile)
    hdr0=phdu[0].header
    crpix1=hdr0['CRPIX1']
    crval1=hdr0['CRVAL1']
    cdelt1=hdr0['CDELT1']
    if hdr0["LOGLAM"]==1: #log bins
        wavebins=10**(crval1+cdelt1*numpy.arange(len(phdu[0].data[0])))
    else: #lin bins
        model_wave_step   = cdelt1
        model_wave_offset = (crval1-cdelt1*(crpix1-1))
        wavebins=model_wave_step*numpy.arange(n_model_wave) + model_wave_offset
    paramData=phdu[1].data
    templateid=paramData["TEMPLATEID"]
    fluxData=phdu[0].data
    #modelFlux=ergs2photons(fluxData,wavebins)# Not required, will be normalized anyway while comparison
    modelData=[]
    nspec=fluxData.shape[0]
    for i in range(nspec):
        modelData.append((i,fluxData[i],templateid[i]))
    models={"WAVEBINS":wavebins,"FLUX":modelData}
    phdu.close()
    
    return wavebins,fluxData,templateid

# reading filter quantum efficiency
def read_filter_response(given_filter,basepath):
    filterNameMap={}
    
    filttype=str.split(given_filter,'_')
    if filttype[0]=='SDSS':
        filterNameMap=given_filter.lower()+"0.txt"
    else: #if breakfilt[0]=='DECAM':
        filterNameMap=given_filter.lower()+".txt"
    print filterNameMap
    filter_response={}
    fileName=basepath+filterNameMap
    filt=numpy.loadtxt(fileName,unpack=True)
    tck=scipy.interpolate.splrep(filt[0],filt[1],s=0)
    filter_response=(filt[0],filt[1],tck)
    return filter_response
