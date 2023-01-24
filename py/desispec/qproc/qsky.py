"""
desispec.qproc.qsky
===================

Please add module-level documentation.
"""
import time
import numpy as np
import scipy.ndimage

from desiutil.log import get_logger
from desispec.qproc.qframe import QFrame

def qproc_sky_subtraction(qframe,return_skymodel=False) :
    """
    Fast sky subtraction directly applied to the input qframe.

     Args:
        qframe : DESI QFrame object
     Optional:
        return_skymodel returns the skymodel as an array of same shape as qframe.flux
    """


    """
The algorithm is an iterative subtraction of the mean spectrum
of the sky fibers. It involves 2 resampling at each step,
- from the fiber wavelength array to the model wavelength array
- from the model wavelength array back to the fiber wavelength array

Here all spectra have the same size. It is the number of CCD rows
(maybe trimed), but the wavelength array of each fiber is different.

Let's call R_ki the resampling matrix of the spectrum F_i of a fiber
with wavelength array (i) on the wavelength array (k).

The sky model defined on the wavelength array (k)
after the first iteration is the mean of the resampled sky fiber spectra F_j :
F^(1)_k = 1/n sum_j R_kj F_j

The model resampled back to a fiber (i) is :
F^(1)_i = R_ik F^(1)_k = (sum_j 1/n R_ik R_kj) F_j

We call A^(1) this matrix that gives a model F^(1)_i = A^(1)_ij F_j
from the measurements.
Omitting the indices, the residuals are R^(1) = (1-A^(1)) F.

Let's call A^(n) this matrix that gives a model F^(n) = A^(n) F
after n iterations. The residuals are R^(n) = (1-A^(n)) F.

If one applies at iteration (n+1) the algorithm on those residuals,
one gets an increment to the model A^(1) R^(n) = A^(1) (1-A^(n)) F ,
so that the total model F^(n+1) =  ( A^(n) + A^(1) (1-A^(n)) ) F
and so, A^(n+1) = A^(n) + A^(1) (1-A^(n))

For our fast algorithm A^(n+1) != A^(n); there is a gain at each
iteration.

This is different from a standard chi2 fit of a linear model.
In the standard chi2 fit, if H_pi is the derivative of the model
for the data point 'i' with respect to a parameter 'p',
ignoring the weights ,
the best fit parameters are X = ( H H^t )^{-1} H D , so that
the value of the model for the data points
is  H^t X = H^t ( H H^t )^{-1} H D .
such that we have the matrix A = H^t  ( H H^t )^{-1} H

It's trivial to see that A is a projector, A^2 = A,
so applying a second time the algorithm on the residuals
give the model parameter increment dX = A (1-A) D = (A-A^2) D = 0,
i.e. not improvement after the first fit.

    """


    log=get_logger()
    t0=time.time()
    log.info("Starting...")

    twave=np.linspace(np.min(qframe.wave),np.max(qframe.wave),qframe.wave.shape[1]*2) # oversampling
    tflux=np.zeros((qframe.flux.shape[0],twave.size))
    tivar=np.zeros((qframe.flux.shape[0],twave.size))

    if return_skymodel :
        sky=np.zeros(qframe.flux.shape)

    if qframe.mask is not None :
        qframe.ivar *= (qframe.mask==0)

    if qframe.fibermap is None :
        log.error("Empty fibermap in qframe, cannot know which are the sky fibers!")
        raise RuntimeError("Empty fibermap in qframe, cannot know which are the sky fibers!")

    skyfibers = np.where(qframe.fibermap["OBJTYPE"]=="SKY")[0]
    if skyfibers.size==0 :
       log.warning("No sky fibers! I am going to declare the faintest half of the fibers as sky fibers")
       mflux = np.median(qframe.flux,axis=1)
       ii = np.argsort(mflux)
       qframe.fibermap["OBJTYPE"][ii[:ii.size//2]] = "SKY"
       skyfibers = np.where(qframe.fibermap["OBJTYPE"]=="SKY")[0]

    log.info("Sky fibers: {}".format(skyfibers))

    for loop in range(5) : # I need several iterations to remove the effect of the wavelength solution noise

        for i in skyfibers :
            jj=(qframe.ivar[i]>0)
            if np.sum(jj)>0 :
                tflux[i]=np.interp(twave,qframe.wave[i,jj],qframe.flux[i,jj])

        ttsky  = np.median(tflux[skyfibers],axis=0)

        if return_skymodel :
            for i in range(qframe.flux.shape[0]) :
                tmp=np.interp(qframe.wave[i],twave,ttsky)
                jj=(qframe.flux[i]!=0)
                if np.sum(jj)>0 :
                    qframe.flux[i,jj] -= tmp[jj]
                    sky[i] += tmp
        else :
            for i in range(qframe.flux.shape[0]) :
                jj=(qframe.flux[i]!=0)
                if np.sum(jj)>0 :
                    qframe.flux[i,jj] -= np.interp(qframe.wave[i,jj],twave,ttsky)


    t1=time.time()
    log.info(" done in {:3.1f} sec".format(t1-t0))

    if return_skymodel :
        return sky
