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


########################################################
# Arc/Wavelength Routines
########################################################

def find_arc_lines(spec,rms_thresh=10.,nwidth=5):
    """Find and centroid arc lines in an input spectrum

    Parameters
    ----------
    spec : ndarray
      Arc line spectrum
    thresh : float
      RMS threshold
    nwidth : int
      Line width to test over
    """
    # Threshold criterion
    npix = spec.size
    spec_mask = sigma_clip(spec,sigma=4.)
    rms = np.std(spec_mask)
    thresh = 10*rms
    #print("thresh = {:g}".format(thresh))
    gdp = spec > thresh

    # Avoid edges
    gdp = gdp & (np.arange(npix) > 2.*nwidth) & (np.arange(npix) < (npix-2.*nwidth))

    # Roll to find peaks (simple algorithm)
    nwidth = 5
    nstep = nwidth // 2
    for kk in xrange(-nstep,nstep):
        if kk < 0:
            test = np.roll(spec,kk) < np.roll(spec,kk+1)
        else:
            test = np.roll(spec,kk) > np.roll(spec,kk+1)
        # Compare
        gdp = gdp & test

    # Center
    gdpix = np.where(gdp)[0]
    ngd = gdpix.size
    xpk = np.zeros(ngd)
    for jj,igdpix in enumerate(gdpix):
        # Simple flux-weight
        pix = np.arange(igdpix-nstep,igdpix+nstep+1,dtype=int)
        xpk[jj] = np.sum(pix*spec[pix]) / np.sum(spec[pix])

    # Finish
    return xpk

def load_gdarc_lines(camera):
    """Loads a select set of arc lines for initial calibrating

    Parameters
    ----------
    cameara : str
      Camera ('b', 'g', 'r')

    Returns
    -------
    dlamb : float
      Dispersion for input camera
    gd_lines : ndarray
      Array of lines expected to be recorded and good for ID
    """
    if camera == 'b':
        HgI = [4046.57, 4077.84, 4358.34, 5460.75, 5769.598]
        CdI = [3610.51, 3650.157, 4678.15, 4799.91, 5085.822]
        NeI = [5881.895, 5944.834]
        dlamb = 0.589
        wmark = 4358.34 # Hg
        gd_lines = np.array(HgI + CdI + NeI)
    else:
        raise ValueError('Bad camera')

    # Sort and return
    gd_lines.sort()
    return dlamb, wmark, gd_lines

def add_arc_lines(id_dict, pixpk, gd_lines, npoly=2, verbose=True):
    """Attempt to identify and add additionally detected lines

    Parameters
    ----------
    id_dict : dict
      dict of ID info
    pixpk : ndarray
      Pixel locations of detected arc lines
    gd_lines : ndarray
      array of expected arc lines to be detected and identified
    npoly : int, optional
      Order of polynomial for fitting

    Returns
    -------
    id_dict : dict
      Filled with complete set of IDs and the final polynomial fit
    """
    # Init
    ilow = id_dict['icen']-(len(id_dict['first_id_pix'])//2)
    ihi = id_dict['icen']+(len(id_dict['first_id_pix'])//2)
    pos=True
    idx = list(id_dict['first_id_idx'])
    wvval = list(id_dict['first_id_wave'])
    xval = list(id_dict['first_id_pix'])
    if 'first_fit' not in id_dict.keys():
        id_dict['first_fit'] = copy.deepcopy(id_dict['fit'])

    # Loop on additional lines for identification
    while ((ilow > 0) or (ihi < gd_lines.size-1)):
        # index to add (step on each side)
        if pos:
            ihi += 1
            inew = ihi
            pos=False
        else:
            ilow -= 1
            inew = ilow
            pos=True
        if ilow < 0:
            continue
        if ihi > (gd_lines.size-1):
            continue
        # New line
        new_wv = gd_lines[inew]
        wvval.append(new_wv)
        wvval.sort()
        # newx
        newx = xafits.func_val(new_wv,id_dict['fit'])
        # Match and add
        imin = np.argmin(np.abs(pixpk-newx))
        newtc = pixpk[imin]
        idx.append(imin)
        idx.sort()
        xval.append(newtc)
        xval.sort()
        # Fit
        # Should reject 1
        if len(xval) > 7:
            npoly = 3
        new_fit = xafits.func_fit(np.array(wvval),np.array(xval),'polynomial',npoly,xmin=0.,xmax=1.)
        id_dict['fit'] = new_fit
    # RMS
    resid2 = (np.array(xval)-xafits.func_val(np.array(wvval),id_dict['fit']))**2
    rms = np.sqrt(np.sum(resid2)/len(xval))
    id_dict['rms'] = rms
    if verbose:
        print('rms = {:g}'.format(rms))
    # Finish
    id_dict['id_idx'] = idx
    id_dict['id_pix'] = xval
    id_dict['id_wave'] = wvval

def id_arc_lines(pixpk, gd_lines, dlamb, wmark, toler=0.2, verbose=False):
    """Match (as best possible), a set of the input list of expected arc lines to the detected list

    Parameters
    ----------
    pixpk : ndarray
      Pixel locations of detected arc lines
    gd_lines : ndarray
      array of expected arc lines to be detected and identified
    dlamb : float
      Average disperion in the spectrum
    wmark : float
      Center of 5 gd_lines to key on (camera dependent)
    toler : float, optional
      Tolerance for matching (20%)

    Returns
    -------
    id_dict : dict
      dict of identified lines
    """
    # List of dicts for diagnosis
    rms_dicts = []
    ## Assign a line to center on
    icen = np.where(np.abs(gd_lines-wmark)<1e-3)[0]
    icen = icen[0]
    ##
    ndet = pixpk.size
    # Set up detected lines to search over
    guesses = ndet//2 + np.arange(-4,7)
    for guess in guesses:
        # Setup
        tc = pixpk[guess]
        if verbose:
            print('tc = {:g}'.format(tc))
        rms_dict = dict(tc=tc,guess=guess)
        # Match to i-2 line
        line_m2 = gd_lines[icen-2]
        Dm2 = wmark-line_m2
        mtm2 = np.where(np.abs((tc-pixpk)*dlamb - Dm2)/Dm2 < toler)[0]
        if len(mtm2) == 0:
            if verbose:
                print('No im2 lines found for guess={:g}'.format(tc))
            continue
        # Match to i+2 line
        line_p2 = gd_lines[icen+2]
        Dp2 = line_p2-wmark
        mtp2 = np.where(np.abs((pixpk-tc)*dlamb - Dp2)/Dp2 < toler)[0]
        if len(mtp2) == 0:
            if verbose:
                print('No ip2 lines found for guess={:g}'.format(tc))
            continue
            #
        all_guess_rms = [] # One per set of guesses, of course
        # Loop on i-2 line
        for imtm2 in mtm2:
            if imtm2==(icen-1):
                if verbose:
                    print('No im1 lines found for guess={:g}'.format(tc))
                continue
            # Setup
            tcm2 = pixpk[imtm2]
            xm1_wv = (gd_lines[icen-1]-gd_lines[icen-2])/(wmark-gd_lines[icen-2])
            # Best match
            xm1 = (pixpk-tcm2)/(tc-tcm2)
            itm1 = np.argmin(np.abs(xm1-xm1_wv))
            # Now loop on ip2
            for imtp2 in mtp2:
                guess_rms = dict(guess=guess, im1=itm1,im2=imtm2, rms=999., ip1=None, ip2=imtp2)
                all_guess_rms.append(guess_rms)
                if imtp2==(icen-1):
                    if verbose:
                        print('No ip1 lines found for guess={:g}'.format(tc))
                    continue
                #
                tcp2 = float(pixpk[imtp2])
                xp1_wv = (gd_lines[icen+2]-gd_lines[icen+1])/(gd_lines[icen+2]-wmark)
                # Best match
                xp1 = (tcp2-pixpk)/(tcp2-tc)
                itp1 = np.argmin(np.abs(xp1-xp1_wv))
                guess_rms['ip1'] = itp1
                # Time to fit
                xval = np.array([tcm2,pixpk[itm1],tc,pixpk[itp1],tcp2])
                wvval = gd_lines[icen-2:icen+3]
                pfit = xafits.func_fit(wvval,xval,'polynomial',2,xmin=0.,xmax=1.)
                # Clip one here and refit
                #   NOT IMPLEMENTED YET
                # RMS (in pixel space)
                rms = np.sqrt(np.sum((xval-xafits.func_val(wvval,pfit))**2)/xval.size)
                guess_rms['rms'] = rms
                # Save fit too
                guess_rms['fit'] = pfit
        # Take best RMS
        if len(all_guess_rms) > 0:
            all_rms = np.array([idict['rms'] for idict in all_guess_rms])
            imn = np.argmin(all_rms)
            rms_dicts.append(all_guess_rms[imn])
    # Find the best one
    all_rms = np.array([idict['rms'] for idict in rms_dicts])
    imin = np.argmin(all_rms)
    id_dict = rms_dicts[imin]
    # Finish
    id_dict['wmark'] = wmark
    id_dict['icen'] = icen
    id_dict['first_id_wave'] = wvval
    id_idx = []
    id_pix = []
    for key in ['im2','im1','guess','ip1','ip2']:
        id_idx.append(id_dict[key])
        id_pix.append(pixpk[id_dict[key]])
    id_dict['first_id_idx'] = id_idx
    id_dict['first_id_pix'] = np.array(id_pix)
    # Return
    return id_dict

########################################################
# Fiber routines
########################################################

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
    #bad_mask = np.ones_like(flat,dtype=int)
    #badx = np.any([xerr > 900.,xerr < 0.],axis=0)
    #bad_mask[badx] = 0
    #nbox = box_radius*2 + 1
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
        flux = np.maximum(flux,1.)
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
        # Guess at sigma
        all_sig = np.abs(fdimg) / np.sqrt( np.log(amp)-np.log(fnimg) )
        g_init.stddev.value = np.median(all_sig[np.where((np.abs(fdimg)>1) & (np.abs(fdimg)<1.5) & (np.isfinite(all_sig)))])

        # Initial fit (need to mask!)
        parm = fitter(g_init, fdimg, fnimg)
        # Iterate
        iterate = True
        nrej = 0
        niter = 0
        while iterate:
            # Clip
            resid = parm(fdimg) - fnimg
            resid_mask = sigma_clip(resid,sigma=4.)
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

def extract_sngfibers_gaussianpsf(img, xtrc, sigma, box_radius=2):
    """Extract spectrum for fibers one-by-one using a Gaussian PSF

    Parameters
    ----------
    img : ndarray
      Image
    xtrc : ndarray
      fiber trace
    sigma : float
      Gaussian sigma for PSF
    box_radius : int, optional
      Radius for extraction (+/-)

    Returns
    -------
    spec : ndarray
      Extracted spectrum
    """
    # Init
    xpix_img = np.outer(np.ones(img.shape[0]),np.arange(img.shape[1]))
    mask = np.zeros_like(img,dtype=int)
    iy = np.arange(img.shape[0],dtype=int)
    #
    all_spec = np.zeros_like(xtrc)
    for qq in range(xtrc.shape[1]):
        if qq%10 == 0:
            print(qq)
        # Mask
        mask[:,:] = 0
        ixt = np.round(xtrc[:,qq]).astype(int)
        for jj,ibox in enumerate(range(-box_radius,box_radius+1)):
            ix = ixt + ibox
            mask[iy,ix] = 1

        # Generate PSF
        dx_img = xpix_img - np.outer(xtrc[:,qq],np.ones(img.shape[1]))
        g_init = models.Gaussian1D(amplitude=1., mean=0., stddev=sigma[qq])
        psf = mask * g_init(dx_img)
        # Extract
        all_spec[:,qq] = np.sum(psf*img,axis=1) / np.sum(psf,axis=1)
    # Return
    return all_spec

def trace_crude_init(image, xinit0, ypass, invvar=None, radius=3.,
    maxshift0=0.5, maxshift=0.2, maxerr=0.2):
#                   xset, xerr, maxerr, maxshift, maxshift0
    """Python port of trace_crude_idl.pro from IDLUTILS

    Modified for initial guess
    Parameters
    ----------
    image : 2D ndarray
      Image for tracing
    xinit : ndarray
      Initial guesses for trace peak at ypass
    ypass : int
      Row for initial guesses

    Returns
    -------
    xset : Trace for each fiber
    xerr : Estimated error in that trace
    """
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
