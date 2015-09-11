"""
desispec.io.image
=================

I/O routines for Image objects
"""

from desispec.image import Image

def write_image(outfile, image, meta=meta):
    """Writes image object to outfile
    
    Args:
        outfile : output file string
        image : desispec.image.Image object
            (or any object with 2D array attributes image, ivar, mask)
    
    Optional:
        meta : dict-like object with metadata key/values (e.g. FITS header)
    """

    if meta is not None:
        hdr = fitsheader(meata)
    else:
        hdr = fitsheader(image.meta)
        
    hx = fits.HDUList()
    hdu = fits.ImageHDU(image.image.astype(np.float32), name='IMAGE')
    hdu.header.append( ('CAMERA', image.camera, 'Spectograph Camera') )
    hdu.header.append( ('VSPECTER', '0.0.0', 'TODO: Specter version') )    
    hdu.header.append( ('RDNOISE', image.rdnoise, 'Read noise [electrons]'))
    hx.append(hdu)
    
    hx.append(fits.ImageHDU(image.ivar.astype(np.float32), name='IVAR'))
    hx.append(fits.CompImageHDU(image.mask.astype(np.uint16), name='MASK'))
    hx.writeto(outfile, clobber=True)
    
    return outfile
    
def read_image(filename):
    """
    Returns desispec.image.Image object from input file
    """
    fx = fits.open(filename)
    image = fx['IMAGE'].data.astype(np.float64)
    ivar = fx['IVAR'].data.astype(np.float64)
    mask = fx['MASK'].data
    
    return Image(image, ivar, mask=mask)
