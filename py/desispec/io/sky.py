"""
io routines for sky

"""
import os
from astropy.io import fits


def write_sky(outfile,head,skyflux,skyivar,skymask,wave) :
    """
    write fiberflat
    """
    hdr = head
    hdr['EXTNAME'] = ('SKY', 'no dimension')
    fits.writeto(outfile,skyflux,header=hdr, clobber=True)
    
    hdr['EXTNAME'] = ('IVAR', 'no dimension')
    hdu = fits.ImageHDU(skyivar, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('MASK', 'no dimension')
    hdu = fits.ImageHDU(skymask, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('WAVELENGTH', '[Angstroms]')
    hdu = fits.ImageHDU(wave, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    
