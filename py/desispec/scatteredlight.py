'''
Try to model and remove the scattered light
'''
import time
import numpy as np
import scipy.interpolate
import astropy.io.fits as pyfits
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from desispec.image import Image
from desiutil.log import get_logger
from desispec.qproc.qextract import numba_extract

def model_scattered_light(image,xyset) :
    """
    Model the scattered light in a preprocessed image.
    The method consist in convolving the "direct" light
    image (image * mask along spectral traces) and
    calibrating this convolved image using the data
    between the fiber bundles.
    
    Args:
      
      image: desispec.image.Image object
      xyset: desispec.xytraceset.XYTraceSet object

    Returns:

      model: np.array of same shape as image.pix
    """  

    log = get_logger()

    log.info("compute mask")
    mask_in  = np.zeros(image.pix.shape,dtype=int)
    
    yy = np.arange(image.pix.shape[0],dtype=int)
    for fiber in range(xyset.nspec) :
        xx = xyset.x_vs_y(fiber,yy).astype(int)
        for x,y in zip(xx,yy) :
            mask_in[y,x-2:x+3] = 1
    mask_in *= (image.mask==0)*(image.ivar>0)
              
    log.info("convolving mask*image")
    hw = 1000
    x1d = np.linspace(-hw,hw,2*hw+1)
    x2d = np.tile(x1d,(2*hw+1,1))
    r   = np.sqrt(x2d**2+x2d.T**2)

    # convolution kernel shape found empirically
    # by looking at one arc lamp image, preproc-z0-00043688.fits
    ##################################################################

    camera = image.meta["CAMERA"].upper()[0]
    log.info("camera= '{}'".format(camera))

    # generic
    if camera[0]=="B" :
        scale1=17.
        scale2=90.
    elif camera[0]=="R" :
        scale1=20.
        scale2=90.
    else :
        scale1=65.
        scale2=110.

    # now specific params for some cameras
    if camera=="R2" :
        scale1=16.
        scale2=110.
    
       
    kern1 = np.exp(-0.5*(r/scale1)**2)
    kern2 = np.exp(-0.5*(r/scale2)**2)
    kern1 /= np.sum(kern1)
    kern2 /= np.sum(kern2)
    
    ##################################################################

    model1  = fftconvolve(image.pix*mask_in,kern1,mode="same")
    model1 *= (model1>0)
    model2  = fftconvolve(image.pix*mask_in,kern2,mode="same")
    model2 *= (model2>0)
    
    log.info("calibrating scattered light model between fiber bundles")
    ny=image.pix.shape[0]    
    yy=np.arange(ny)
    xinter = np.zeros((21,ny))
    #ratio = np.zeros((21,ny))
    mod1_scale = np.zeros((21,ny))
    mod2_scale = np.zeros((21,ny))

    meas_inter = np.zeros((21,ny))
    mod1_inter = np.zeros((21,ny))
    mod2_inter = np.zeros((21,ny))
    
    
    nbins=ny//5
    # compute median ratio in bins of y
    bins=np.linspace(0,ny,nbins+1).astype(int)
    meas_bins=np.zeros((21,nbins))
    mod1_bins=np.zeros((21,nbins))
    mod2_bins=np.zeros((21,nbins))
    y_bins=np.tile((bins[:-1]+bins[1:])/2.,(21,1))

    
    ivar=image.ivar*(image.mask==0)
    for i in range(21) :
        if i==0 : xinter[i] =  xyset.x_vs_y(0,yy)-7.5
        elif i==20 : xinter[i] =  xyset.x_vs_y(499,yy)+7.5
        else : xinter[i] = (xyset.x_vs_y(i*25-1,yy)+xyset.x_vs_y(i*25,yy))/2.
        meas,junk = numba_extract(image.pix,ivar,xinter[i],hw=3)
        mod1,junk  = numba_extract(model1,ivar,xinter[i],hw=3)
        mod2,junk  = numba_extract(model2,ivar,xinter[i],hw=3)
        for b in range(bins.size-1) :
            yb=bins[b]
            ye=bins[b+1]
            meas_bins[i,b]=np.median(meas[yb:ye])
            mod1_bins[i,b]=np.median(mod1[yb:ye])
            mod2_bins[i,b]=np.median(mod2[yb:ye])

        meas_inter[i] = meas
        mod1_inter[i] = mod1
        mod2_inter[i] = mod2
        
    # first fit a low deg to get ratio of the two terms among all central bins
    deg=0
    npar=(deg+1)*2
    H=np.zeros((npar,nbins*(21-4)))
    for d in range(deg+1) :
        H[d] = (mod1_bins[2:-2]*y_bins[2:-2]**d).ravel()
        H[d+deg+1] = (mod2_bins[2:-2]*y_bins[2:-2]**d).ravel()
    A=H.dot(H.T)
    B=H.dot(meas_bins[2:-2].ravel())
    Ai=np.linalg.inv(A)
    Par=Ai.dot(B)
    Par[Par<0]=0.
    a1=Par[0]/(Par[0]+Par[1])
    a2=Par[1]/(Par[0]+Par[1])
    log.info("a1={} a2={}".format(a1,a2))
    
    # second fit a higher degree correction of a fixed combination of both terms
    # per inter-bundle space
    for i in range(21) :
        deg=2
        npar=(deg+1)
        H=np.zeros((npar,nbins))
        for d in range(deg+1) :
            H[d] = (a1*mod1_bins[i]+a2*mod2_bins[i])*y_bins[i]**d
        A=H.dot(H.T)
        B=H.dot(meas_bins[i])
        Ai=np.linalg.inv(A)
        Par=Ai.dot(B)
        for d in range(deg+1) :
            mod1_scale[i] += a1*Par[d]*yy**d
            mod2_scale[i] += a2*Par[d]*yy**d
        
        
        #     import matplotlib.pyplot as plt
        #     plt.plot(meas_inter[i])
        #     plt.plot(mod1_scale[i]*mod1_inter[i])
        #     plt.plot(mod1_scale[i]*mod1_inter[i]+mod2_scale[i]*mod2_inter[i])
        #     print(i)
        #     plt.show()

             
    log.info("interpolating over bundles, using fitted calibration")
    xx = np.arange(image.pix.shape[1])

    model=np.zeros(model1.shape)
    for j in range(ny) :
        func1 = interp1d(xinter[:,j],mod1_scale[:,j], kind='linear', bounds_error = False, fill_value=0.)
        func2 = interp1d(xinter[:,j],mod2_scale[:,j], kind='linear', bounds_error = False, fill_value=0.)
        model[j] = func1(xx)*model1[j]+func2(xx)*model2[j]
    model *= (model>0)
    
    return model
