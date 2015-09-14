'''
Lightweight wrapper class for preprocessed image data
'''

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
        self._mask = mask
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
