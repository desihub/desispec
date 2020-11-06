"""
desispec.pixflat
========================

Routines for pixel flat fielding
"""

import numpy as np
import scipy.ndimage,scipy.signal


def convolve2d(image,k,weight=None) :
    """ Return a 2D convolution of image with kernel k, optionally with a weight image

    Args:
        image : 2D np.array image
        k : 2D np.array kernel, each dimension must be odd and greater than 1
    Options:
        weight : 2D np.array of same shape as image
    Returns:
        cimage : 2D np.array convolved image of same shape as input image
    """
    if weight is not None :
        if weight.shape != image.shape :
            raise ValueError("weight and image should have same shape")
        sw=convolve2d(weight,k,None)
        swim=convolve2d(weight*image,k,None)
        return swim/(sw+(sw==0))

    if len(k.shape) != 2 or len(image.shape) != 2:
        raise ValueError("kernel and image should have 2 dimensions")
    for d in range(2) :
        if k.shape[d]<=1 or k.shape[d]-(k.shape[d]//2)*2 != 1 :
            raise ValueError("kernel dimensions should both be odd and >1, and input as shape %s"%str(k.shape))
    m0=k.shape[0]//2
    m1=k.shape[1]//2
    eps0=m0
    eps1=m1
    tmp=np.zeros((image.shape[0]+2*m0,image.shape[1]+2*m1))
    tmp[m0:-m0,m1:-m1]=image
    tmp[:m0+1,m1:-m1]=np.tile(np.median(image[:eps0,:],axis=0),(m0+1,1))
    tmp[-m0-1:,m1:-m1]=np.tile(np.median(image[-eps0:,:],axis=0),(m0+1,1))
    tmp[m0:-m0,:m1+1]=np.tile(np.median(image[:,:eps1],axis=1),(m1+1,1)).T
    tmp[m0:-m0,-m1-1:]=np.tile(np.median(image[:,-eps1:],axis=1),(m1+1,1)).T
    tmp[:m0,:m1]=np.median(tmp[:m0,m1])
    tmp[-m0:,:m1]=np.median(tmp[-m0:,m1])
    tmp[-m0:,-m1:]=np.median(tmp[-m0:,-m1-1])
    tmp[:m0,-m1:]=np.median(tmp[:m0,-m1-1])
    return scipy.signal.fftconvolve(tmp,k,"valid")
