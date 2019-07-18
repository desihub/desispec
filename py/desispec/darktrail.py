"""
desispec.darktrail
================

Utility functions to correct dark trails found for some LBNL CCD
amplifiers due to an imperfect CDS (correlated double sampling) suppression of the reset level.
The cause of the effect is unknown (as this is written), and we have
not been able to correcting it with adjustment of the readout electronic parameters.

The function here are used to do a software correction of the effect.
"""

import numpy as np
import scipy.signal
from desiutil.log import get_logger

def correct_dark_trail(image,xyslice,left,width,amplitude) :
    """
    remove the dark trails from a preprocessed image with a processing of the form
    image[j,i] += sum_{k<i} image[j,k] * a * |i-k|/width * exp( -|i-k|/width)

    Args:
        image : desispec.Image class instance 
        xyslice : tuple of python slices (yy,xx) where yy is indexing CCD rows (firt index in 2D numpy array) and xx is the other
        left : if true , the trails are to the left of the image spots, as is the case for amplifiers B and D.
        width : width parameter in unit of pixels
        amplitude : amplitude of the dark trail

    """
    hwidth=int(5*width+1)
    kernel = np.zeros((3,2*hwidth+1))
    dx=np.arange(-hwidth,hwidth+1)
    adx=np.abs(dx)   
    if left :
        kernel[1]=amplitude*(adx/width)*np.exp(-adx/width)*(dx<0)
    else :
        kernel[1]=amplitude*(adx/width)*np.exp(-adx/width)*(dx>0)
    trail = scipy.signal.fftconvolve(image[xyslice],kernel,"same")
    image[xyslice] += trail

    
def compute_fiber_cross_profile(image,tset,xyslice) :
    """
    Computes a cross dispersion profile averaged over fibers and CCD rows
    for pixels in a subsample of the image

    Args:
      image : desispec.Image class instance 
      xyslice : tuple of python slices (yy,xx) where yy is indexing CCD rows (firt index in 2D numpy array) and xx is the other
              
    Returns: 
      x: 1D numpy array with the pixel grid
      prof: 1D numpy array with the cross dispersion profile sampled on x
    
    """

    log = get_logger()
    
    log.info("compute_fiber_cross_profile")

    xmargin = 100
    ymargin = 300
    dx = np.arange(-xmargin,xmargin+1)
    yy=xyslice[0]
    xx=xyslice[1]
    
    fprofs=[]
    for fiber in range(tset.nspec) :
        xc = tset.x_vs_y(fiber,(yy.start+yy.stop)/2.)
        if xc < xx.start+xmargin : continue
        if xc > xx.stop-xmargin : continue
        log.debug("using fiber={} xc={}".format(fiber,xc))
        profs = []
        for y in range(yy.start+ymargin,yy.stop-ymargin) :
            xc = tset.x_vs_y(fiber,y)
            xb = int(xc)-xmargin
            xe = xb + 2*xmargin + 1
            if xb<xx.start : continue
            if xe>xx.stop : continue
            xslice = np.arange(xb,xe)
            prof = np.interp(dx,xslice-xc,image.pix[y,xb:xe])
            prof /=  np.sum(prof[xmargin-3:xmargin+4])
            prof -= np.median(prof)
            profs.append(prof)
        prof = np.median(np.vstack(profs),axis=0)
        fprofs.append(prof)
    prof = np.median(np.vstack(fprofs),axis=0)
    
    return dx,prof
 
def compute_dark_trail(input_profile,width,amplitude=1,data_profile=None) :
    """
    Computes the dark trail profile given an input cross-dispersion profile.
    The amplitude is fitted if the data_profile is given.
    
    Args:
      input_profile: 1D numpy array with the input cross dispersion profile
      width: the trail width parameter
      amplitude: the amplitude of the trail (overwritten if data_profile is not None)
    
    Optional argument:
      data_profile: 1D numpy array with the data cross dispersion profile. It is
      used to fit the trail amplitude.

    Returns:
      trail: 1D numpy array with the trail profile on the same grid as the input profile
      amplitude: the fitted amplitude (or the input amplitude if data profile = None)
    """
    nn = int(input_profile.size)
    trail = np.zeros(nn)
    for i in range(nn) :
        dxi = i-np.arange(i)
        trail[:i] -= (dxi/width)*np.exp(-dxi/width)*input_profile[i]
    if data_profile is not None :
        # fit the normalization
        a = np.sum(trail[:nn//2-15]**2)
        b = np.sum(trail[:nn//2-15]*data_profile[:nn//2-15])
        amplitude = b/a
    trail *= amplitude
    return trail,amplitude

  
def fit_dark_trail(x,prof,left) :
    """
    Fit the width and amplitude parameters of a dark trail given a cross-dispersion profile.
    It uses the unaffect side of the profile as a model for the other side assuming
    a symetric profile. 
    
    Args:
      x: 1D numpy array giving the pixel grid of the cross-dispersion profile.
      prof: 1D numpy array representing the cross dispersion profile.
      left: if true , the trail is to the left of the profile, as is the case for amplifiers B and D.

    Returns:
      amplitude: the fitted amplitude of the trail profile
      width: the fitted width of the trail profile
    """
    log = get_logger()
    
    log.info("Fitting dark trail function")
    
    xmargin=prof.size//2
    if not left :
        # flip the profile so we always deal with same orientation
        prof = prof[::-1]
    
    # use unaffected right side of profile for pedestal subtraction
    prof -= np.median(prof[-xmargin//2:]) 
    
    nn=prof.size
    nh=nn//2

    model=prof.copy()
    model[1:nh+1] = model[nh:nn-1][::-1]

    for loop in range(2) :
        if loop == 0 : 
            width_array=np.linspace(1,20,21)
        else :
            width_array=np.linspace(max(1.,width-2),width+3,50)
        chi2=np.zeros(width_array.size)
        for iw,width in enumerate(width_array) :
            trail,amp = compute_dark_trail(model,width=width,data_profile=prof)
            chi2[iw]=np.sum((prof-(model+trail)))**2
            log.debug("width={} amp={}, chi2={}".format(width,amp,chi2[iw]))
            
        width=width_array[np.argmin(chi2)]
        log.info("loop {}/2 best fit width = {:3.1f} pixels".format(loop+1,width))

   

    trail,amplitude = compute_dark_trail(model,width=width,data_profile=prof)
    log.info("amplitude={:6.5f} width={:3.2f} pix".format(amplitude,width))
    
    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.plot(x,prof,label="data")
    #plt.plot(x,model,label="model without trail")
    #plt.plot(x,model+trail,label="model with trail")
    #plt.plot(x,trail,c="r",label="trail")
    #plt.grid()
    #plt.legend()
    #plt.show()
    
    return amplitude,width
    
    



