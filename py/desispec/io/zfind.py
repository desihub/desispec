"""
I/O routines for zfind

Things to output

brick
targetid
z : 1D[nspec] best fit redshift
zerr : 1D[nspec] redshift uncertainty estimate
zwarn : 1D[nspec] integer redshift warning bitmask (details TBD)
type : 1D[nspec] classification [GALAXY, QSO, STAR, ...]
subtype : 1D[nspec] sub-classification
"""

import numpy as np
from astropy.io import fits

def write_zbest(filename, brickname, targetids, zfind):
    
    dtype = [
        ('BRICKNAME', 'S8'),
        ('TARGETID',  np.int64),
        ('Z',         zfind.z.dtype),
        ('ZERR',      zfind.zerr.dtype),
        ('ZWARN',     zfind.zwarn.dtype),
        ('TYPE',      zfind.type.dtype),
        ('SUBTYPE',   zfind.subtype.dtype),
    ]
    
    data = np.empty(zfind.nspec, dtype=dtype)
    data['BRICKNAME'] = brickname
    data['TARGETID']  = targetids
    data['Z']         = zfind.z
    data['ZERR']      = zfind.zerr
    data['ZWARN']     = zfind.zwarn
    data['TYPE']      = zfind.type
    data['SUBTYPE']   = zfind.subtype
    
    hdu = fits.BinTableHDU(data, name='ZBEST', uint=True)
    fits.writeto(filename, hdu.data, hdu.header, clobber=True)