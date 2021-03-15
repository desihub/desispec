'''
Utility function to perform a median of images with masks
'''

from desiutil.log import get_logger
import numpy as np

def masked_median(images,masks=None) :
    '''
    Perfomes a median of an list of input images. If a list of mask is provided,
    the median is performed only on unmasked pixels.

    Args:
       images : 3D numpy array : list of images of same shape
    Options:
       masks : list of mask images of same shape as the images. Only pixels with mask==0 are considered in the median.

    Returns : median image
    '''
    log = get_logger()

    if masks is None :
        log.info("simple median of %d images"%len(images))
        return np.median(images,axis=0)
    else :
        log.info("masked array median of %d images"%len(images))
        return np.ma.median(np.ma.masked_array(data=images,mask=(masks!=0)),axis=0).data
