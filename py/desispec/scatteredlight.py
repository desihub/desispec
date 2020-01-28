'''
Try to model and remove the scattered light
'''
import time
import numpy as np
import scipy.interpolate
import astropy.io.fits as pyfits
from scipy.signal import fftconvolve
from desispec.image import Image
from desiutil.log import get_logger
from desispec.qproc.qextract import numba_extract


def model_scattered_light(image,xyset) :
    """
    Args:
      
      image: desispec.image.Image object
      xyset: desispec.xytraceset.XYTraceSet object
      sigma: sigma of Gaussian convolution in pixels
    """  

    log = get_logger()

    log.info("compute mask")
    mask_in  = np.zeros(image.pix.shape,dtype=int)
    
    yy = np.arange(image.pix.shape[0],dtype=int)
    for fiber in range(xyset.nspec) :
        xx = xyset.x_vs_y(fiber,yy).astype(int)
        for x,y in zip(xx,yy) :
            mask_in[y,x-1:x+2] = 1
    mask_in *= (image.mask==0)*(image.ivar>0)
              
    log.info("convolving mask*image")
    hw = 1000
    x1d = np.linspace(-hw,hw,2*hw+1)
    x2d = np.tile(x1d,(2*hw+1,1))
    r   = np.sqrt(x2d**2+x2d.T**2)

    # convolution kernel shape found empirically
    # by looking at one arc lamp image, preproc-z0-00043688.fits
    ##################################################################
    scale=70.
    kern1 = (1-r/scale)*(r<scale)
    kern2 = np.exp(-0.5*(r/150)**2)
    kern1 /= np.sum(kern1)
    kern2 /= np.sum(kern2)
    kern = 0.4*kern1+kern2
    kern /= np.sum(kern)
    ##################################################################

    model  = fftconvolve(image.pix*mask_in,kern,mode="same")
    model *= (model>0)

    log.info("calibrating scattered light model between fiber bundles")
    ny=image.pix.shape[0]    
    yy=np.arange(ny)
    xinter = np.zeros((21,ny))
    ratio = np.zeros((21,ny))
    for i in range(21) :
        if i==0 : xinter[i] =  xyset.x_vs_y(0,yy)-7
        elif i==20 : xinter[i] =  xyset.x_vs_y(499,yy)+7
        else : xinter[i] = (xyset.x_vs_y(i*25-1,yy)+xyset.x_vs_y(i*25,yy))/2.
        meas,junk = numba_extract(image.pix,image.ivar,xinter[i],hw=3)
        mod,junk  = numba_extract(model,image.ivar,xinter[i],hw=3)
        # compute median ratio in bins of y
        bins=np.linspace(0,ny,10).astype(int)
        tmpratios=np.zeros(bins.size-1)
        for b in range(bins.size-1) :
            yb=bins[b]
            ye=bins[b+1]
            tmpratios[b] = np.median(meas[yb:ye]/mod[yb:ye])
        # fit ratio in bins with a deg2 polynomial
        centers=(bins[:-1]+bins[1:])/2.
        pol=np.poly1d(np.polyfit(centers,tmpratios,4))
        log.info("#{} x[2000]={} ratio={}".format(i,xinter[i,2000],pol(2000.)))
        ratio[i] = pol(yy)

    log.info("interpolating over bundles, using fitted calibration")
    xx = np.arange(image.pix.shape[1])
    for j in range(ny) :
        model[j] *= np.interp(xx,xinter[:,j],ratio[:,j])
    model *= (model>0)

    return model
