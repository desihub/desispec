"""
io routines for frame

"""
import os
from astropy.io import fits


def write_fiberflat(outfile,head,fiberflat,fiberflat_ivar,mean_spectrum,wave) :
    """
    write fiberflat
    """
    hdr = head
    hdr['EXTNAME'] = ('FIBERFLAT', 'no dimension')
    fits.writeto(outfile,fiberflat,header=hdr, clobber=True)
    
    hdr['EXTNAME'] = ('IVAR', 'no dimension')
    hdu = fits.ImageHDU(fiberflat_ivar, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('MEANSPEC', 'electrons')
    hdu = fits.ImageHDU(mean_spectrum, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('WAVELENGTH', '[Angstroms]')
    hdu = fits.ImageHDU(wave, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    
