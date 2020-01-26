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

def model_scattered_light(image,xyset,sigma1=20,sigma2=100,a1=0.8) :
    """
    Args:
      
      image: desispec.image.Image object
      xyset: desispec.xytraceset.XYTraceSet object
      sigma: sigma of Gaussian convolution in pixels
    """  

    log = get_logger()

    log.info("compute mask")
    mask_in  = np.zeros(image.pix.shape,dtype=int)
    mask_out = np.ones(image.pix.shape,dtype=float)

    yy = np.arange(image.pix.shape[0],dtype=int)
    for fiber in range(xyset.nspec) :
        xx = xyset.x_vs_y(fiber,yy).astype(int)
        for x,y in zip(xx,yy) :
            mask_in[y,x-1:x+2] = 1
            mask_out[y,x-5:x+6] = 0

    mask_in *= (image.mask==0)*(image.ivar>0)
    mask_out *= (image.mask==0)*image.ivar
             
    log.info("convolving mask_in*image")
    hw = int(3*max(sigma1,sigma2))
    x1d = np.linspace(-hw,hw,2*hw+1)
    x2d = np.tile(x1d,(2*hw+1,1))
    r2  = x2d**2+x2d.T**2
    kern = a1*np.exp(-0.5*r2/sigma1**2)+(1-a1)*np.exp(-0.5*r2/sigma2**2)
    kern /= np.sum(kern)
    model  = fftconvolve(image.pix*mask_in,kern,mode="same")
    model *= (model>0)
    model2 = model**2  
    n0=image.pix.shape[0]
    n1=image.pix.shape[1]
    yy = np.tile(np.linspace(-1,1,n0),(n1,1)).T
    xx = np.tile(np.linspace(-1,1,n1),(n0,1))
    
    # adjust this model weighted by a polynomial of x and y
    xp=[0,1,2,0,1,0]
    yp=[0,0,0,1,1,2]
    log.info("compute linear system matrices")
    npar=len(xp)
    B=np.zeros(npar)
    A=np.zeros((npar,npar))
    for i in range(npar) :
        B[i] = np.sum(mask_out*image.pix*model*xx**xp[i]*yy**yp[i])        
        for j in range(i+1) :
            A[i,j] = A[j,i] = np.sum(mask_out*model2*(xx**(xp[i]+xp[j])*yy**(yp[i]+yp[j])))

    log.info("solve and apply")
    Ai = np.linalg.inv(A)
    alpha = Ai.dot(B)
    log.info("coefficients= {}".format(alpha))
    pmod = np.zeros(mod.shape)
    for i in range(npar) :
        pmod += alpha[i] * model * xx**xp[i] * yy**yp[i]
    pmod[pmod<0]=0.
    return pmod
