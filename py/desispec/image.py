'''
Lightweight wrapper class for preprocessed image data
'''
import copy
import numpy as np

class Image(object):
    def __init__(self, pix, ivar, mask=None, readnoise=0.0, camera='unknown',
        meta=None):
        """
        Create Image object
        
        Args:
            pix : 2D numpy.ndarray of image pixels
            ivar : inverse variance of pix, same shape as pix
            
        Optional:
            mask : 0 is good, non-0 is bad; default is (ivar==0)
            readnoise : CCD readout noise in electrons/pixel (float)
            camera : e.g. 'b0', 'r1', 'z9'
            meta : dict-like metadata key/values, e.g. from FITS header
            
        Notes:
            TODO: expand readnoise be an array instead of a single float
        """
        if pix.ndim != 2:
            raise ValueError('pix must be 2D, not {}D'.format(pix.ndim))
        if pix.shape != ivar.shape:
            raise ValueError('pix.shape{} != ivar.shape{}'.format(pix.shape, ivar.shape))            
        if (mask is not None) and (pix.shape != mask.shape):
            raise ValueError('pix.shape{} != mask.shape{}'.format(pix.shape, mask.shape))
            
        self.pix = pix
        self.ivar = ivar
        if mask is not None:
            self._mask = mask.astype(np.uint16)
        else:
            self._mask = None
        self.meta = meta
        
        #- Optional parameters
        self.readnoise = readnoise
        self.camera = camera
    
    #- Image.mask = (ivar==0) if input mask was None    
    @property
    def mask(self):
        if self._mask is None:
            return (self.ivar == 0)
        else:
            return self._mask
            
    #- Allow image slicing
    def __getitem__(self, xyslice):

        #- Slices must be a slice object, or a tuple of (slice, slice)
        if isinstance(xyslice, slice):
            pass #- valid slice
        elif isinstance(xyslice, tuple):
            #- tuples of (slice, slice) are valid
            if len(xyslice) > 2:
                raise ValueError('Must slice in 1D or 2D, not {}D'.format(len(xyslice)))
            else:
                if not isinstance(xyslice[0], slice) or \
                   not isinstance(xyslice[1], slice):
                    raise ValueError('Invalid slice for Image objects')
        else:
            raise ValueError('Invalid slice for Image objects')

        pix = self.pix[xyslice]
        ivar = self.ivar[xyslice]
        if self._mask is not None:
            mask = self.mask[xyslice]
        else:
            mask = None
        
        meta = copy.copy(self.meta)
    
        #- NAXIS1 = x, NAXIS2 = y; python slices[y,x] = [NAXIS2, NAXIS1]
        if meta is not None and (('NAXIS1' in meta) or ('NAXIS2' in meta)):
            #- image[a:b] instead of image[a:b, c:d]
            if isinstance(xyslice, slice):
                ny = xyslice.stop - xyslice.start
                meta['NAXIS2'] = ny
            else:
                slicey, slicex = xyslice
                #- slices ranges could be None if using : instead of a:b
                if (slicex.stop is not None):
                    nx = slicex.stop - slicex.start
                    meta['NAXIS1'] = nx
                if (slicey.stop is not None):
                    ny = slicey.stop - slicey.start
                    meta['NAXIS2'] = ny
            
        return Image(pix, ivar, mask, \
            readnoise=self.readnoise, camera=self.camera, meta=meta)
