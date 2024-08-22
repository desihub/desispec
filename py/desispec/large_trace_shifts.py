"""
desispec.trace_shifts
=====================

"""

from __future__ import absolute_import, division

import os
import sys
import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.signal import fftconvolve

from desiutil.log import get_logger


def detect_spots_in_image(image) :
    '''
    Detection of spots in preprocessed arc lamp image
    Args:
        image : preprocessed arc lamp image (desispec.Image object)
    returns:
        xc: 1D float numpy array with xccd spot coordinates in the image (CCD column number)
        yc: 1D float numpy array with yccd spot coordinates in the image (CCD row number)
    '''

    log = get_logger()

    # set to zero masked pixels
    image.ivar *= (image.mask==0)
    image.ivar *= (image.ivar>0)
    image.pix  *= (image.ivar>0)

    # convolve with Gaussian kernel
    hw = 3
    sigma = 1.
    x = np.tile(np.arange(-hw,hw+1),(2*hw+1,1))
    y = x.T.copy()
    kernel = np.exp(-(x**2+y**2)/2/sigma**2)
    kernel /= np.sum(kernel)
    simg  = fftconvolve(image.pix,kernel,mode='same')
    sivar = fftconvolve(image.ivar,kernel**2,mode='same')
    sivar *= (sivar>0)

    log.info("detections")
    nsig = 6
    detections = (simg*np.sqrt(sivar))>nsig
    peaks=np.zeros(simg.shape)
    peaks[1:-1,1:-1] = (detections[1:-1,1:-1]>0)\
        *(simg[1:-1,1:-1]>simg[2:,1:-1])\
        *(simg[1:-1,1:-1]>simg[:-2,1:-1])\
        *(simg[1:-1,1:-1]>simg[1:-1,2:])\
        *(simg[1:-1,1:-1]>simg[1:-1,:-2])

    log.info("peak coordinates")
    x=np.tile(np.arange(simg.shape[1]),(simg.shape[0],1))
    y=np.tile(np.arange(simg.shape[0]),(simg.shape[1],1)).T
    xp=x[peaks>0]
    yp=y[peaks>0]

    nspots=xp.size
    if nspots>1e5 :
        message="way too many spots detected: {}. Aborting".format(nspots)
        log.error(message)
        raise RuntimeError(message)

    log.info("refit {} spots centers".format(nspots))
    xc=np.zeros(nspots)
    yc=np.zeros(nspots)
    for p in range(nspots) :
        b0=yp[p]-3
        e0=yp[p]+4
        b1=xp[p]-3
        e1=xp[p]+4
        spix=np.sum(image.pix[b0:e0,b1:e1])
        xc[p]=np.sum(image.pix[b0:e0,b1:e1]*x[b0:e0,b1:e1])/spix
        yc[p]=np.sum(image.pix[b0:e0,b1:e1]*y[b0:e0,b1:e1])/spix
    log.info("done")

    return xc,yc


# copied from desimeter to avoid dependencies


def match_same_system(x1,y1,x2,y2,remove_duplicates=True) :
    '''
    match two catalogs, assuming the coordinates are in the same coordinate system (no transfo)
    Args:
        x1 : float numpy array of coordinates along first axis of cartesian coordinate system
        y1 : float numpy array of coordinates along second axis in same system
        x2 : float numpy array of coordinates along first axis in same system
        y2 : float numpy array of coordinates along second axis in same system
    returns:
        indices_2 : integer numpy array. if ii is a index array for entries in the first catalog,
                            indices_2[ii] is the index array of best matching entries in the second catalog.
                            (one should compare x1[ii] with x2[indices_2[ii]])
                            negative indices_2 indicate unmatched entries
        distances : distances between pairs. It can be used to discard bad matches
    '''

    xy1=np.array([x1,y1]).T
    xy2=np.array([x2,y2]).T
    tree2 = KDTree(xy2)
    distances,indices_2 = tree2.query(xy1,k=1)

    if remove_duplicates :
        unique_indices_2 = np.unique(indices_2)
        n_duplicates = np.sum(indices_2>=0)-np.sum(unique_indices_2>=0)
        if n_duplicates > 0 :
            for i2 in unique_indices_2 :
                jj=np.where(indices_2==i2)[0]
                if jj.size>1 :
                    kk=np.argsort(distances[jj])
                    indices_2[jj[kk[1:]]] = -1

    distances[indices_2<0] = np.inf
    return indices_2,distances
