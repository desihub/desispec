"""
desispec.sky
============

Utility functions to compute a sky model and subtract it.
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import copy, pdb
import warnings

from astropy.modeling import models, fitting
from astropy.stats import sigma_clip

from matplotlib import pyplot as plt

from xastropy.xutils import afits as xafits
from xastropy.xutils import xdebug as xdb

def xpos_image(shape, xtrc, box_radius):
    """Generates an xpos image which is the offset from a given trace of each pixel
    Parameters:
    ----------
    shape: tuple
    xtrc: trace
    box_radius: float
    """
    # Generate mask


def fiber_gauss(flat, xtrc, xerr, box_radius=2, debug=False, verbose=False):
    """Find the PSF sigma for each fiber
    This serves as an initial guess to what follows

    args:
        flat : ndarray of fiber flat image 
        xtrc: ndarray of fiber traces
        xerr: ndarray of error in fiber traces
        box_radius: int, optinal 
          Radius of boxcar extraction in pixels

    returns gauss
      list of Gaussian sigma
    """
    warnings.warn("fiber_gauss uses astropy.modeling.  Consider an alternative")
    # Init
    nfiber = xtrc.shape[1]
    ny = xtrc.shape[0]
    iy = np.arange(ny).astype(int)
    # Mask
    mask = np.zeros_like(flat,dtype=int)
    bad_mask = np.ones_like(flat,dtype=int)
    badx = np.any([xerr > 900.,xerr < 0.],axis=0)
    bad_mask[badx] = 0
    nbox = box_radius*2 + 1
    # Sub images
    #dx_img = np.zeros((ny,nbox))
    #nrm_img = np.zeros((ny,nbox))
    xpix_img = np.outer(np.ones(flat.shape[0]),np.arange(flat.shape[1]))
    # Gaussian fit
    g_init = models.Gaussian1D(amplitude=1., mean=0., stddev=1.)
    g_init.amplitude.fixed = True
    g_init.mean.fixed = True
    fitter = fitting.LevMarLSQFitter()

    # Loop on fibers
    gauss = []
    for ii in xrange(nfiber):
        if verbose:
            print("Working on fiber {:d}".format(ii))
        mask[:] = 0
        #dx_img[:] = 0
        ixt = np.round(xtrc[:,ii]).astype(int)
        for jj,ibox in enumerate(range(-box_radius,box_radius+1)):
            ix = ixt + ibox
            mask[iy,ix] = 1
            #dx_img[:,jj] = ix-xtrc[:,ii]
            #nrm_img[:,jj] = flat[iy,ix]
        dx_img = xpix_img - np.outer(xtrc[:,ii],np.ones(flat.shape[1]))
        # Sum
        flux = np.sum(mask*flat,axis=1)
        # Normalize
        #nrm_img /= np.outer(flux,np.ones(nbox))
        nrm_img = flat / np.outer(flux,np.ones(flat.shape[1]))
        # Gaussian
        amp = np.median(nrm_img[np.where(np.abs(dx_img)<0.05)])
        g_init.amplitude.value = amp # Fixed
        #fdimg = dx_img.flatten()
        #fnimg = nrm_img.flatten()
        fdimg = dx_img[mask==1].flatten()
        fnimg = nrm_img[mask==1].flatten()
        all_sig =  np.abs(fdimg) / np.sqrt(
            np.log(amp)-np.log(fnimg) )
        g_init.stddev.value = np.median(
            all_sig[np.where((np.abs(fdimg)>1) & (np.abs(fdimg)<1.5))])

        # Initial fit (need to mask!)
        parm = fitter(g_init, fdimg, fnimg)
        # Iterate
        iterate = True
        nrej = 0
        niter = 0
        while iterate:
            # Clip
            resid = parm(fdimg) - fnimg
            resid_mask = sigma_clip(resid,sig=4.)
            # Fit
            gdp = ~resid_mask.mask
            parm = fitter(g_init, fdimg[gdp], fnimg[gdp])
            # Again?
            if np.sum(resid_mask.mask) <= nrej:
                iterate = False
            else:
                nrej = np.sum(resid_mask.mask)
                niter += 1
        if verbose:
            print("Rejected {:d} in {:d} iterations".format(nrej,niter))

        #debug = False
        if debug:
            plt.clf()
            plt.scatter(fdimg[gdp], fnimg[gdp])
            x= np.linspace(-box_radius, box_radius, 200)
            plt.plot(x, parm(x), 'r-')
            plt.show()
            plt.close()
            xdb.set_trace()
        # Save
        gauss.append(parm.stddev.value)
    #
    return np.array(gauss)

def find_fiber_peaks(flat, ypos=None, nwidth=5, debug=False) :
    """Find the peaks of the fiber flat spectra
    Preforms book-keeping error checking

    args:
        flat : ndarray of fiber flat image 
        ypos : int [optional] Row for finding peaks
           Default is half-way up the image
        nwidth : int [optional] Width of peak (end-to-end)
        debug: bool, optional

    returns xpk, ypos, cut
      list of xpk (nearest pixel) at ypos
      ndarray of cut through the image
    """
    # Init
    Nbundle = 20
    Nfiber = 25 # Fibers per bundle
    # Set ypos for peak finding
    if ypos is None:
        ypos = flat.shape[0]//2

    # Cut image
    cutimg = flat[ypos-15:ypos+15,:]

    # Smash
    cut = np.median(cutimg,axis=0)

    # Set flux threshold
    srt = np.sort(cutimg.flatten())
    thresh = srt[int(cutimg.size*0.95)] / 2.
    gdp = cut > thresh

    # Roll to find peaks (simple algorithm)
    nstep = nwidth // 2
    for kk in xrange(-nstep,nstep):
        if kk < 0:
            test = np.roll(cut,kk) < np.roll(cut,kk+1)
        else:
            test = np.roll(cut,kk) > np.roll(cut,kk+1)
        # Compare
        gdp = gdp & test
    xpk = np.where(gdp)[0]
    if debug:
        xdb.xplot(cut, xtwo=xpk, ytwo=cut[xpk],mtwo='o')
        xdb.set_trace()

    # Book-keeping and some error checking
    if len(xpk) != Nbundle*Nfiber:
        raise ValueError('Found the wrong number of total fibers: {:d}'.format(len(xpk)))
    else:
        print('Found {:d} fibers'.format(len(xpk)))
    # Find bundles
    xsep = np.roll(xpk,-1) - xpk
    medsep = np.median(xsep)
    bundle_ends = np.where(np.abs(xsep-medsep) > 0.5*medsep)[0]
    if len(bundle_ends) != Nbundle:
        raise ValueError('Found the wrong number of bundles: {:d}'.format(len(bundle_ends)))
    else:
        print('Found {:d} bundles'.format(len(bundle_ends)))
    # Confirm correct number of fibers per bundle
    bad = ((bundle_ends+1) % Nfiber) != 0
    if np.sum(bad) > 0:
        raise ValueError('Wrong number of fibers in a bundle')

    # Return
    return xpk, ypos, cut


def fit_traces(xset, xerr, func='legendre', order=6, sigrej=20.,
    RMS_TOLER=0.02, verbose=False):
    '''Fit the traces
    Default is 6th order Legendre polynomials

    Parameters:
    -----------
    xset: ndarray
      traces
    xerr: ndarray
      Error in the trace values (999.=Bad)
    RMS_TOLER: float, optional [0.02]
      Tolerance on size of RMS in fit

    Returns:
    -----------
    xnew, fits
    xnew: ndarray
      New fit values (without error)
    fits: list
      List of the fit dicts
    '''
    reload(xafits)

    ny = xset.shape[0]
    ntrace = xset.shape[1]
    xnew = np.zeros_like(xset)
    fits = []
    yval = np.arange(ny)
    for ii in xrange(ntrace):
        mask = xerr[:,ii] > 900.
        nmask = np.sum(mask)
        # Fit with rejection
        dfit, mask = xafits.iter_fit(yval, xset[:,ii], func, order, sig_rej=sigrej,
            weights=1./xerr[:,ii], initialmask=mask, maxone=True)#, sigma=xerr[:,ii])
        # Stats on residuals
        nmask_new = np.sum(mask)-nmask 
        if nmask_new > 10:
            raise ValueError('Rejected too many points: {:d}'.format(nmask_new))
        # Save
        xnew[:,ii] = xafits.func_val(yval,dfit)
        fits.append(dfit)
        # Residuas
        gdval = mask==0
        resid = xnew[:,ii][gdval] - xset[:,ii][gdval]
        rms = np.std(resid)
        if verbose:
            print('RMS of FIT= {:g}'.format(rms))
        if rms > RMS_TOLER:
            #xdb.xplot(yval, xnew[:,ii], xtwo=yval[gdval],ytwo=xset[:,ii][gdval], mtwo='o')
            pdb.set_trace()
    # Return
    return xnew, fits


def trace_crude_init(image, xinit0, ypass, invvar=None, radius=3.,
    maxshift0=0.5, maxshift=0.2, maxerr=0.2):
#                   xset, xerr, maxerr, maxshift, maxshift0
    '''Python port of trace_crude_idl.pro from IDLUTILS
    Modified for initial guess
    image: 2D ndarray
      Image for tracing
    xinit: ndarray
      Initial guesses for trace peak at ypass
    ypass: int
      Row for initial guesses
    Returns:
    ---------
    xset: Trace for each fiber
    xerr: Estimated error in that trace
    '''
    # Init
    xinit = xinit0.astype(float)
    #xinit = xinit[0:3]
    ntrace = xinit.size
    ny = image.shape[0]
    xset = np.zeros((ny,ntrace))
    xerr = np.zeros((ny,ntrace))
    if invvar is None: 
        invvar = np.zeros_like(image) + 1. 

    #
    #  Recenter INITIAL Row for all traces simultaneously
    #
    iy = ypass * np.ones(ntrace,dtype=int)
    xfit,xfiterr = trace_fweight(image, xinit, iy, invvar=invvar, radius=radius)
    # Shift
    xshift = np.clip(xfit-xinit, -1*maxshift0, maxshift0) * (xfiterr < maxerr)
    xset[ypass,:] = xinit + xshift
    xerr[ypass,:] = xfiterr * (xfiterr < maxerr)  + 999.0 * (xfiterr >= maxerr)

    #    /* LOOP FROM INITIAL (COL,ROW) NUMBER TO LARGER ROW NUMBERS */
    for iy in range(ypass+1, ny):
        xinit = xset[iy-1, :]
        ycen = iy * np.ones(ntrace,dtype=int)
        xfit,xfiterr = trace_fweight(image, xinit, ycen, invvar=invvar, radius=radius)
        # Shift
        xshift = np.clip(xfit-xinit, -1*maxshift, maxshift) * (xfiterr < maxerr)
        # Save
        xset[iy,:] = xinit + xshift
        xerr[iy,:] = xfiterr * (xfiterr < maxerr)  + 999.0 * (xfiterr >= maxerr)
    #      /* LOOP FROM INITIAL (COL,ROW) NUMBER TO SMALLER ROW NUMBERS */
    for iy in range(ypass-1, -1,-1):
        xinit = xset[iy+1, :]
        ycen = iy * np.ones(ntrace,dtype=int)
        xfit,xfiterr = trace_fweight(image, xinit, ycen, invvar=invvar, radius=radius)
        # Shift      
        xshift = np.clip(xfit-xinit, -1*maxshift, maxshift) * (xfiterr < maxerr)
        # Save
        xset[iy,:] = xinit + xshift
        xerr[iy,:] = xfiterr * (xfiterr < maxerr)  + 999.0 * (xfiterr >= maxerr)
        
    return xset, xerr

def trace_fweight(fimage, xinit, ycen=None, invvar=None, radius=3.):
    '''Python port of trace_fweight.pro from IDLUTILS

    Parameters:
    -----------
    fimage: 2D ndarray
      Image for tracing
    xinit: ndarray
      Initial guesses for x-trace
    invvar: ndarray, optional
      Inverse variance array for the image
    radius: float, optional
      Radius for centroiding; default to 3.0
    '''
    # Definitions for Cython
    #cdef int nx,ny,ncen

    # Init
    nx = fimage.shape[1]
    ny = fimage.shape[0]
    ncen = len(xinit)
    # Create xnew, xerr
    xnew = xinit.astype(float)
    xerr = np.zeros(ncen) + 999.

    # ycen
    if ycen is None:
        if ncen != ny:
            raise ValueError('Bad input')
        ycen = np.arange(ny, dtype=int)
    else:
        if len(ycen) != ncen:
            raise ValueError('Bad ycen input.  Wrong length')
    x1 = xinit - radius + 0.5
    x2 = xinit + radius + 0.5
    ix1 = np.floor(x1).astype(int)
    ix2 = np.floor(x2).astype(int)

    fullpix = int(np.maximum(np.min(ix2-ix1)-1,0))
    sumw = np.zeros(ncen)
    sumxw = np.zeros(ncen)
    sumwt = np.zeros(ncen)
    sumsx1 = np.zeros(ncen)
    sumsx2 = np.zeros(ncen)
    qbad = np.array([False]*ncen) 

    if invvar is None: 
        invvar = np.zeros_like(fimage) + 1. 

    # Compute
    for ii in range(0,fullpix+3):
        spot = ix1 - 1 + ii
        ih = np.clip(spot,0,nx-1)
        xdiff = spot - xinit
        #
        wt = np.clip(radius - np.abs(xdiff) + 0.5,0,1) * ((spot >= 0) & (spot < nx))
        sumw = sumw + fimage[ycen,ih] * wt
        sumwt = sumwt + wt
        sumxw = sumxw + fimage[ycen,ih] * xdiff * wt
        var_term = wt**2 / (invvar[ycen,ih] + (invvar[ycen,ih] == 0))
        sumsx2 = sumsx2 + var_term
        sumsx1 = sumsx1 + xdiff**2 * var_term
        #qbad = qbad or (invvar[ycen,ih] <= 0)
        qbad = np.any([qbad, invvar[ycen,ih] <= 0], axis=0)

    # Fill up
    good = (sumw > 0) &  (~qbad)
    if np.sum(good) > 0:
        delta_x = sumxw[good]/sumw[good]
        xnew[good] = delta_x + xinit[good]
        xerr[good] = np.sqrt(sumsx1[good] + sumsx2[good]*delta_x**2)/sumw[good]

    bad = np.any([np.abs(xnew-xinit) > radius + 0.5,xinit < radius - 0.5,xinit > nx - 0.5 - radius],axis=0)
    if np.sum(bad) > 0:
        xnew[bad] = xinit[bad]
        xerr[bad] = 999.0

    # Return
    return xnew, xerr

#####################################################################            
#####################################################################            
# Utilities
#####################################################################            


#####################################################################            
#####################################################################            
#####################################################################            
# QA
#####################################################################            

def fiber_trace_qa(flat, xtrc, outfil=None, Nfiber=25, isclmin=0.5):
    ''' Generate a QA plot for the traces.  Bundle by bundle
    Parameters:
    ------------
    flat: ndarray
      image
    xtrc: ndarray
      Trace array
    isclmin: float, optional [0.5]
      Fraction of 90 percentile flux to scale image by
    outfil: str, optional
      Output file
    normalize: bool, optional
      Normalize the flat?  If not, use zscale for output
    '''
    import matplotlib
    import matplotlib.gridspec as gridspec
    import matplotlib.cm as cm
    from matplotlib.backends.backend_pdf import PdfPages

    ticks_font = matplotlib.font_manager.FontProperties(family='times new roman', 
       style='normal', size=16, weight='normal', stretch='normal')
    plt.rcParams['font.family']= 'times new roman'
    cmm = cm.Greys_r
    # Outfil
    if outfil is None:
        outfil = 'fiber_trace_qa.pdf'
    ntrc = xtrc.shape[1]
    ycen = np.arange(flat.shape[0])

    # Plot
    pp = PdfPages(outfil)
    plt.clf()
    fig = plt.figure(figsize=(8, 5.0),dpi=1200)
    #fig.set_size_inches(10.0,6.5)
    Nbundle = ntrc // Nfiber + (ntrc%Nfiber > 0)
    for qq in range(Nbundle):
        ax = plt.gca()
        for label in ax.get_yticklabels() :
            label.set_fontproperties(ticks_font)
        for label in ax.get_xticklabels() :
            label.set_fontproperties(ticks_font)

        # Cut image
        i0 = qq*Nfiber
        i1 = np.minimum((qq+1)*Nfiber,ntrc)
        x0 = np.maximum(int(np.min(xtrc[:,i0:i1]))-3,0)
        x1 = np.minimum(int(np.max(xtrc[:,i0:i1]))+3,flat.shape[1])
        sub_flat = flat[:,x0:x1].T
        # Scale
        srt = np.sort(sub_flat.flatten())
        sclmax = srt[int(sub_flat.size*0.9)]
        sclmin = isclmin * sclmax
        # Plot
        #xdb.set_trace()
        mplt = plt.imshow(sub_flat,origin='lower', cmap=cmm, 
            extent=(0., sub_flat.shape[1]-1, x0,x1-1), aspect='auto')
            #extent=(0., sub_flat.shape[1]-1, x0,x1))
        #mplt.set_clim(vmin=sclmin, vmax=sclmax)

        # Axes
        #xdb.set_trace()
        #plt.xlim(0., sub_flat.shape[1]-1)
        plt.xlim(0., sub_flat.shape[1]-1)
        plt.ylim(x0,x1)

        # Traces
        for ii in xrange(i0,i1):
            # Left
            plt.plot(ycen, xtrc[:,ii], 'r-',alpha=0.7, linewidth=0.5)
            # Label
            #iy = int(frame.shape[0]/2.)
            #plt.text(ltrace[iy,ii], ycen[iy], '{:d}'.format(ii+1), color='red', ha='center')
            #plt.text(rtrace[iy,ii], ycen[iy], '{:d}'.format(ii+1), color='green', ha='center')

        pp.savefig(bbox_inches='tight')
        plt.close()
    # Finish
    print('Writing {:s} QA for fiber trace'.format(outfil))
    pp.close()
