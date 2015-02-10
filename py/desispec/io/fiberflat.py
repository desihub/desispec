"""
io routines for fiberflat

"""
import os
from astropy.io import fits

from desispec.io import findfile
from desispec.io.util import fitsheader, native_endian, makepath

def write_fiberflat(outfile,fiberflat,fiberflat_ivar,fiberflat_mask,mean_spectrum,wave, header=None) :
    """
    write fiberflat
    """
    outfile = makepath(outfile, 'fiberflat')
    
    hdr = fitsheader(header)
    hdr['EXTNAME'] = ('FIBERFLAT', 'no dimension')
    fits.writeto(outfile,fiberflat,header=hdr, clobber=True)
    
    hdr['EXTNAME'] = ('IVAR', 'no dimension')
    hdu = fits.ImageHDU(fiberflat_ivar, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('MASK', 'no dimension')
    hdu = fits.ImageHDU(fiberflat_mask, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('MEANSPEC', 'electrons')
    hdu = fits.ImageHDU(mean_spectrum, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('WAVELENGTH', '[Angstroms]')
    hdu = fits.ImageHDU(wave, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    
def read_fiberflat(filename) :
    """
    read fiberflat
    """
    #- check if outfile is (night, expid, camera) tuple instead
    if not isinstance(filename, (str, unicode)):
        night, expid, camera = filename
        filename = findfile('fiberflat', night, expid, camera)
    
    header = fits.getheader(filename, 0)
    fiberflat = native_endian(fits.getdata(filename, 0))
    ivar = native_endian(fits.getdata(filename, "IVAR"))
    mask = native_endian(fits.getdata(filename, "MASK"))
    meanspec = native_endian(fits.getdata(filename, "MEANSPEC"))
    wave = native_endian(fits.getdata(filename, "WAVELENGTH"))
    
    return fiberflat,ivar,mask,meanspec,wave, header
    
