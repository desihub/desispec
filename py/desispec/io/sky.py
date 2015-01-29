"""
io routines for sky

"""
import os
from astropy.io import fits


def write_sky(outfile,head,skyflux,skyivar,skymask,cskyflux,cskyivar,wave) :
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
    
    
    hdr['EXTNAME'] = ('CSKY', 'convolved sky at average resolution')
    hdu = fits.ImageHDU(cskyflux, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('CIVAR', 'convolved sky inverse variance')
    hdu = fits.ImageHDU(cskyivar, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('WAVELENGTH', '[Angstroms]')
    hdu = fits.ImageHDU(wave, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
def read_sky(filename) :

    """
    read sky
    """
    skyflux=fits.getdata(filename, 0).astype('float64')
    ivar=fits.getdata(filename, "IVAR").astype('float64')
    mask=fits.getdata(filename, "MASK").astype('int') # ??? SOMEONE CHECK THIS ???
    cskyflux=fits.getdata(filename, "CSKY").astype('float64')
    civar=fits.getdata(filename, "CIVAR").astype('float64')
    wave=fits.getdata(filename, "WAVELENGTH").astype('float64')
    
    return skyflux,ivar,mask,cskyflux,civar,wave

