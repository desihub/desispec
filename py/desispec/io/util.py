import astropy.io

def fitsheader(header):
    """
    Utility function to convert header into astropy.io.fits.Header
    
    header can be:
      - None: return blank Header
      - list of (key, value) or (key, (value,comment)) entries
      - dict d[key] -> value or (value, comment)
      - Header: just return it unchanged
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
