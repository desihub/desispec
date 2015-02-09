import os
import astropy.io

def fitsheader(header):
    """
    Convert header into astropy.io.fits.Header object
    
    header can be:
      - None: return a blank Header
      - list of (key, value) or (key, (value,comment)) entries
      - dict d[key] -> value or (value, comment)
      - Header: just return it unchanged
      
    Returns fits.Header object
    """
    if header is None:
        return astropy.io.fits.Header()
        
    if isinstance(header, list):
        hdr = astropy.io.fits.Header()
        for key, value in header:
            hdr[key] = value
            
        return hdr
        
    if isinstance(header, dict):
        hdr = astropy.io.fits.Header()
        for key, value in header.items():
            hdr[key] = value
        return hdr
        
    if isinstance(header, astropy.io.fits.Header):
        return header
        
    raise ValueError("Can't convert {} into fits.Header".format(type(header)))

def native_endian(data):
    """
    Convert numpy array data to native endianness if needed
    
    Returns new array if endianness is swapped, otherwise returns input data
    
    Context:
    By default, FITS data from astropy.io.fits.getdata() are not Intel
    native endianness and scipy 0.14 sparse matrices have a bug with
    non-native endian data.
    """
    if data.dtype.isnative:
        return data
    else:
        return data.byteswap().newbyteorder()

def makepath(outfile, filetype=None):
    """
    Create path to outfile if needed.
    
    If outfile isn't a string and filetype is set, interpret outfile as
    a tuple of parameters to locate the file via findfile().
    
    Returns /path/to/outfile
    
    TODO: maybe a different name?
    """
    #- if this doesn't look like a filename, interpret outfile as a tuple of
    #- (night, expid, ...) via findfile.  Only works if filetype is set.
    if (filetype is not None) and (not isinstance(outfile, (str, unicode))):
        outfile = findfile(filetype, *outfile)

    #- Create the path to that file if needed
    path = os.path.normpath(os.path.dirname(outfile))
    if not os.path.exists(path):
        os.makedirs(path)

    return outfile
