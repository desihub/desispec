'''
desispec.scatteredlight
=======================

Try to model and remove the scattered light.
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


    # convolution kernel shape found by fitting
    # one arc lamp image, preproc-z0-00043688.fits
    # with the code desi_fit_scattered_light_kernel
    ##################################################################
    nodes=np.array([0,5,10,20,30,40,50,100,150,200,300])
    hw = nodes[-1]
    x1d = np.linspace(-hw,hw,2*hw+1)
    x2d = np.tile(x1d,(2*hw+1,1))
    r   = np.sqrt(x2d**2+x2d.T**2)

    params=dict()
    params['b0']=np.array([1.0000,0.5112,1.1413,0.6052,0.3623,0.3427,0.2278,0.0841,0.0352,0.0431,0.0000,])
    params['b1']=np.array([1.0000,0.5314,1.3406,0.6678,0.2432,0.1110,0.0370,0.0202,0.0254,0.0284,0.0000,])
    params['b2']=np.array([1.0000,0.8836,1.0667,1.3196,0.6735,0.2275,0.1053,0.1047,0.0521,0.0229,0.0000,])
    params['b3']=np.array([1.0000,0.6919,1.0962,0.7958,0.5117,0.3656,0.1786,0.0596,0.0365,0.0188,0.0000,])
    params['b4']=np.array([1.0000,0.2730,1.0030,0.4884,0.1662,0.1428,0.0620,0.0190,0.0353,0.0266,0.0000,])
    params['b5']=np.array([1.0000,0.5347,1.1061,0.6693,0.4260,0.3748,0.2058,0.0450,0.0230,0.0192,0.0000,])
    params['b6']=np.array([1.0000,0.7443,1.0783,0.8645,0.5807,0.4181,0.1993,0.0539,0.0317,0.0186,0.0000,])
    params['b7']=np.array([1.0000,0.2317,1.0386,0.3037,0.1214,0.0642,0.0330,0.0143,0.0328,0.0236,0.0000,])
    params['b8']=np.array([1.0000,0.4321,1.2369,0.4828,0.2282,0.1275,0.0506,0.0209,0.0423,0.0342,0.0000,])
    params['b9']=np.array([1.0000,0.7482,1.1079,0.9641,0.4154,0.2291,0.0711,0.0611,0.0352,0.0175,0.0000,])
    params['r0']=np.array([1.0000,1.0634,1.1767,0.9618,0.5883,0.6366,0.7888,0.0621,-0.0000,0.0000,0.0000,])
    params['r1']=np.array([1.0000,1.1019,1.2270,0.9856,0.1717,0.1558,0.2142,-0.0000,0.0000,-0.0000,0.0000,])
    params['r2']=np.array([1.0000,1.3130,1.1483,1.2472,0.7480,0.6697,0.1524,0.1562,0.0877,0.0292,0.0000,])
    params['r3']=np.array([1.0000,1.2498,1.4043,1.0969,0.0860,0.1838,0.2942,0.0723,0.0682,-0.0000,0.0000,])
    params['r4']=np.array([1.0000,1.3722,1.2573,1.6649,1.4019,1.6920,1.7492,0.4951,0.2254,0.1171,0.0000,])
    params['r5']=np.array([1.0000,1.2123,1.2441,0.9732,0.0766,0.1110,0.0164,0.0174,0.0159,-0.0000,0.0000,])
    params['r6']=np.array([1.0000,0.9967,1.0351,0.8091,0.3163,0.3794,0.5617,-0.0000,-0.0000,0.0000,0.0000,])
    params['r7']=np.array([1.0000,0.9842,1.3198,1.1438,0.5061,0.5434,0.3175,0.0574,0.0156,0.0045,0.0000,])
    params['r8']=np.array([1.0000,1.6838,0.9991,1.3340,0.5631,0.6579,0.5212,0.0942,-0.0000,0.0000,0.0000,])
    params['r9']=np.array([1.0000,1.2776,1.2246,1.2775,0.9397,1.0657,0.6500,0.1212,0.0663,0.0303,0.0000,])
    params['z0']=np.array([1.0000,1.2187,1.3041,0.8125,0.5324,0.5660,0.2830,0.0637,0.0915,0.0045,0.0000,])
    params['z1']=np.array([1.0000,0.4882,0.9247,0.4095,0.1006,0.1259,0.0635,0.0022,0.0028,-0.0000,0.0000,])
    params['z2']=np.array([1.0000,0.8952,1.0721,0.6368,0.3757,0.4195,0.1516,0.0792,0.1059,-0.0000,0.0000,])
    params['z3']=np.array([1.0000,0.9608,1.0748,0.6764,0.4919,0.4806,0.3256,0.0496,0.0401,0.0000,0.0000,])
    params['z4']=np.array([1.0000,1.6451,1.4325,1.1856,1.0914,1.2408,0.8704,0.1198,0.0826,0.0037,0.0000,])
    params['z5']=np.array([1.0000,0.4804,0.7011,0.3784,0.0633,0.0954,0.1605,0.0275,0.0182,-0.0000,0.0000,])
    params['z6']=np.array([1.0000,0.9678,1.2216,0.7484,0.4634,0.4786,0.3571,0.0569,0.0382,-0.0000,0.0000,])
    params['z7']=np.array([1.0000,0.5858,0.8845,0.4246,0.1513,0.1859,0.1553,0.0152,0.0197,-0.0000,0.0000,])
    params['z8']=np.array([1.0000,0.9139,1.0364,0.7501,0.5155,0.3914,0.1731,0.0134,0.0307,-0.0000,0.0000,])
    params['z9']=np.array([1.0000,0.8205,1.0013,0.7501,0.4873,0.3066,0.1290,0.0576,0.0158,-0.0000,0.0000,])

    camera = image.meta["CAMERA"].strip().lower()


    log.info("camera= '{}'".format(camera))
    par=params[camera]

    kern = np.interp(r,nodes,par)
    kern /= np.sum(kern)

    ##################################################################

    model  = fftconvolve(image.pix*mask_in,kern,mode="same")
    model *= (model>0)

    log.info("calibrating scattered light model between fiber bundles")
    ny=image.pix.shape[0]
    yy=np.arange(ny)
    xinter = np.zeros((21,ny))
    mod_scale = np.zeros((21,ny))

    nbins=ny//100
    # compute median ratio in bins of y
    bins=np.linspace(0,ny,nbins+1).astype(int)
    meas_bins=np.zeros(nbins)
    mod_bins=np.zeros(nbins)
    y_bins=(bins[:-1]+bins[1:])/2.

    ivar=image.ivar*(image.mask==0)
    for i in range(21) :
        if i==0 : xinter[i] =  xyset.x_vs_y(0,yy)-7.5
        elif i==20 : xinter[i] =  xyset.x_vs_y(499,yy)+7.5
        else : xinter[i] = (xyset.x_vs_y(i*25-1,yy)+xyset.x_vs_y(i*25,yy))/2.
        meas,junk = numba_extract(image.pix,ivar,xinter[i],hw=3)
        mod,junk  = numba_extract(model,ivar,xinter[i],hw=3)
        for b in range(bins.size-1) :
            yb=bins[b]
            ye=bins[b+1]
            meas_bins[b]=np.median(meas[yb:ye])
            mod_bins[b]=np.median(mod[yb:ye])
        deg=4
        npar=(deg+1)
        H=np.zeros((npar,nbins))
        for d in range(deg+1) :
            H[d] = mod_bins*y_bins**d
        A=H.dot(H.T)
        B=H.dot(meas_bins)
        Ai=np.linalg.inv(A)
        Par=Ai.dot(B)
        for d in range(deg+1) :
            mod_scale[i] += Par[d]*yy**d

    log.info("interpolating over bundles, using fitted calibration")
    xx = np.arange(image.pix.shape[1])

    for j in range(ny) :
        func = interp1d(xinter[:,j],mod_scale[:,j], kind='linear', bounds_error = False, fill_value=0.)
        model[j] *= func(xx)
    model *= (model>0)

    if camera == 'r2' :
        log.warning("do not try to remove scattered light for this one on the left hand side of the image")
        model[:,:model.shape[1]//2] = 0.

    return model
