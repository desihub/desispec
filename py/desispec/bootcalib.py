"""
desispec.bootcalib
==================

Utility functions to perform a quick calibration of DESI data

TODO:
1. Expand to r, i cameras
2. QA plots
3. Test with CR data
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import copy
import pdb
import imp
import yaml
import glob

from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
from astropy.table import Table, Column, vstack
from astropy.io import fits

from matplotlib import pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages

from desispec.log import get_logger
from desiutil import funcfits as dufits

try:
    from xastropy.xutils import xdebug as xdb
except:
    pass

desispec_path = imp.find_module('desispec')[1]+'/../../'
glbl_figsz = (16,9)

########################################################
# Arc/Wavelength Routines (Linelists come next)
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
    spec_mask = sigma_clip(spec, sig=4.)
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
    camera : str
      Camera ('b', 'g', 'r')

    Returns
    -------
    dlamb : float
      Dispersion for input camera
    gd_lines : ndarray
      Array of lines expected to be recorded and good for ID
    """
    log=get_logger()
    if camera[0] == 'b':
        HgI = [4046.57, 4077.84, 4358.34, 5460.75, 5769.598]
        CdI = [3610.51, 3650.157, 4678.15, 4799.91, 5085.822]
        NeI = [5881.895, 5944.834]
        dlamb = 0.589
        wmark = 4358.34 # Hg
        gd_lines = np.array(HgI + CdI + NeI)
    if camera[0] == 'r':
        NeI = [5881.895, 5944.834]
    else:
        log.error('Bad camera')

    # Sort and return
    gd_lines.sort()
    return dlamb, wmark, gd_lines

def add_gdarc_lines(id_dict, pixpk, gd_lines, npoly=2, verbose=False):
    """Attempt to identify and add additional goodlines

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
        newx = dufits.func_val(new_wv,id_dict['fit'])
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
        new_fit = dufits.func_fit(np.array(wvval),np.array(xval),'polynomial',npoly,xmin=0.,xmax=1.)
        id_dict['fit'] = new_fit
    # RMS
    resid2 = (np.array(xval)-dufits.func_val(np.array(wvval),id_dict['fit']))**2
    rms = np.sqrt(np.sum(resid2)/len(xval))
    id_dict['rms'] = rms
    if verbose:
        log=get_logger()
        log.info('rms = {:g}'.format(rms))
    # Finish
    id_dict['id_idx'] = idx
    id_dict['id_pix'] = xval
    id_dict['id_wave'] = wvval

def id_remainder(id_dict, pixpk, llist, toler=3., verbose=False):
    """Attempt to identify the remainder of detected lines
    Parameters
    ----------
    id_dict : dict
      dict of ID info
    pixpk : ndarray
      Pixel locations of detected arc lines
    llist : Table
      Line list
    toler : float
      Tolerance for matching
    """
    wv_toler = 3.*id_dict['dlamb'] # Ang
    # Generate a fit for pixel to wavelength
    pixwv_fit = dufits.func_fit(np.array(id_dict['id_pix']),np.array(id_dict['id_wave']),'polynomial',3,xmin=0.,xmax=1.)
    # Loop on detected lines, skipping those with an ID
    for ii,ixpk in enumerate(pixpk):
        # Already ID?
        if ii in id_dict['id_idx']:
            continue
        # Predict wavelength
        wv_pk = dufits.func_val(ixpk,pixwv_fit)
        # Search for a match
        mt = np.where(np.abs(llist['wave']-wv_pk)<wv_toler)[0]
        if len(mt) == 1:
            if verbose:
                log=get_logger()
                log.info('Matched {:g} to {:g}'.format(ixpk,llist['wave'][mt[0]]))
            id_dict['id_idx'].append(ii)
            id_dict['id_pix'].append(ixpk)
            id_dict['id_wave'].append(llist['wave'][mt[0]])
    # Sort
    id_dict['id_idx'].sort()
    id_dict['id_pix'].sort()
    id_dict['id_wave'].sort()


def id_arc_lines(pixpk, gd_lines, dlamb, wmark, toler=0.2, verbose=False):
    """Match (as best possible), a set of the input list of expected arc lines to the detected list

    Parameters
    ----------
    pixpk : ndarray
      Pixel locations of detected arc lines
    gd_lines : ndarray
      array of expected arc lines to be detected and identified
    dlamb : float
      Average dispersion in the spectrum
    wmark : float
      Center of 5 gd_lines to key on (camera dependent)
    toler : float, optional
      Tolerance for matching (20%)

    Returns
    -------
    id_dict : dict
      dict of identified lines
    """
    log=get_logger()
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
            log.info('tc = {:g}'.format(tc))
        rms_dict = dict(tc=tc,guess=guess)
        # Match to i-2 line
        line_m2 = gd_lines[icen-2]
        Dm2 = wmark-line_m2
        mtm2 = np.where(np.abs((tc-pixpk)*dlamb - Dm2)/Dm2 < toler)[0]
        if len(mtm2) == 0:
            if verbose:
                log.info('No im2 lines found for guess={:g}'.format(tc))
            continue
        # Match to i+2 line
        line_p2 = gd_lines[icen+2]
        Dp2 = line_p2-wmark
        mtp2 = np.where(np.abs((pixpk-tc)*dlamb - Dp2)/Dp2 < toler)[0]
        if len(mtp2) == 0:
            if verbose:
                log.info('No ip2 lines found for guess={:g}'.format(tc))
            continue
            #
        all_guess_rms = [] # One per set of guesses, of course
        # Loop on i-2 line
        for imtm2 in mtm2:
            if imtm2==(icen-1):
                if verbose:
                    log.info('No im1 lines found for guess={:g}'.format(tc))
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
                pfit = dufits.func_fit(wvval,xval,'polynomial',2,xmin=0.,xmax=1.)
                # Clip one here and refit
                #   NOT IMPLEMENTED YET
                # RMS (in pixel space)
                rms = np.sqrt(np.sum((xval-dufits.func_val(wvval,pfit))**2)/xval.size)
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
    id_dict['dlamb'] = dlamb
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
# Linelist routines
########################################################

def parse_nist(ion):
    """Parse a NIST ASCII table.

    Note that the long ---- should have
    been commented out and also the few lines at the start.

    Taken from PYPIT

    Parameters
    ----------
    ion : str
      Name of ion
    """
    log=get_logger()
    # Find file
    srch_file = desispec_path + '/data/arc_lines/'+ion+'_air.ascii'
    nist_file = glob.glob(srch_file)
    if len(nist_file) != 1:
        log.error("Cannot find NIST file {:s}".format(srch_file))
    # Read
    nist_tbl = Table.read(nist_file[0], format='ascii.fixed_width')
    gdrow = nist_tbl['Observed'] > 0.  # Eliminate dummy lines
    nist_tbl = nist_tbl[gdrow]
    # Now unique values only (no duplicates)
    uniq, indices = np.unique(nist_tbl['Observed'],return_index=True)
    nist_tbl = nist_tbl[indices]
    # Deal with Rel
    agdrel = []
    for row in nist_tbl:
        try:
            gdrel = int(row['Rel.'])
        except:
            try:
                gdrel = int(row['Rel.'][:-1])
            except:
                gdrel = 0
        agdrel.append(gdrel)
    agdrel = np.array(agdrel)
    # Remove and add
    nist_tbl.remove_column('Rel.')
    nist_tbl.remove_column('Ritz')
    nist_tbl.add_column(Column(agdrel,name='RelInt'))
    nist_tbl.add_column(Column([ion]*len(nist_tbl), name='Ion', dtype='S5'))
    nist_tbl.rename_column('Observed','wave')
    # Return
    return nist_tbl

def load_arcline_list(camera):
    """Loads arc line list from NIST files
    Parses and rejects

    Taken from PYPIT

    Parameters
    ----------
    lines : list
      List of ions to load

    Returns
    -------
    alist : Table
      Table of arc lines
    """
    log=get_logger()
    wvmnx = None
    if camera[0] == 'b':
        lamps = ['CdI','ArI','HgI','NeI']
    else:
        log.error("Not ready for this camera")
    # Get the parse dict
    parse_dict = load_parse_dict()
    # Read rejection file
    with open(desispec_path+'/data/arc_lines/rejected_lines.yaml', 'r') as infile:
        rej_dict = yaml.load(infile)
    # Loop through the NIST Tables
    tbls = []
    for iline in lamps:
        # Load
        tbl = parse_nist(iline)
        # Parse
        if iline in parse_dict.keys():
            tbl = parse_nist_tbl(tbl,parse_dict[iline])
        # Reject
        if iline in rej_dict.keys():
            log.info("Rejecting select {:s} lines".format(iline))
            tbl = reject_lines(tbl,rej_dict[iline])
        tbls.append(tbl[['Ion','wave','RelInt']])
    # Stack
    alist = vstack(tbls)

    # wvmnx?
    if wvmnx is not None:
        print('Cutting down line list by wvmnx: {:g},{:g}'.format(wvmnx[0],wvmnx[1]))
        gdwv = (alist['wave'] >= wvmnx[0]) & (alist['wave'] <= wvmnx[1])
        alist = alist[gdwv]
    # Return
    return alist


def reject_lines(tbl,rej_dict):
    """Parses a NIST table using various criteria

    Taken from PYPIT

    Parameters
    ----------
    tbl : Table
      Read previously from NIST ASCII file
    rej_dict : dict
      Dict of rejected lines

    Returns
    -------
    tbl : Table
      Rows not rejected
    """
    msk = tbl['wave'] == tbl['wave']
    # Loop on rejected lines
    for wave in rej_dict.keys():
        close = np.where(np.abs(wave-tbl['wave']) < 0.1)[0]
        if rej_dict[wave] == 'all':
            msk[close] = False
        else:
            raise ValueError('Not ready for this')
    # Return
    return tbl[msk]

def parse_nist_tbl(tbl,parse_dict):
    """Parses a NIST table using various criteria
    Parameters
    ----------
    tbl : Table
      Read previously from NIST ASCII file
    parse_dict : dict
      Dict of parsing criteria.  Read from load_parse_dict

    Returns
    -------
    tbl : Table
      Rows meeting the criteria
    """
    # Parse
    gdI = tbl['RelInt'] >= parse_dict['min_intensity']
    gdA = tbl['Aki'] >= parse_dict['min_Aki']
    gdw = tbl['wave'] >= parse_dict['min_wave']
    # Combine
    allgd = gdI & gdA & gdw
    # Return
    return tbl[allgd]

def load_parse_dict():
    """Dicts for parsing Arc line lists from NIST

    Rejected lines are in the rejected_lines.yaml file
    """
    dict_parse = dict(min_intensity=0., min_Aki=0., min_wave=0.)
    arcline_parse = {}
    # ArI
    arcline_parse['ArI'] = copy.deepcopy(dict_parse)
    arcline_parse['ArI']['min_intensity'] = 1000. # NOT PICKING UP REDDEST LINES
    # HgI
    arcline_parse['HgI'] = copy.deepcopy(dict_parse)
    arcline_parse['HgI']['min_intensity'] = 800.
    # HeI
    arcline_parse['HeI'] = copy.deepcopy(dict_parse)
    arcline_parse['HeI']['min_intensity'] = 20.
    # NeI
    arcline_parse['NeI'] = copy.deepcopy(dict_parse)
    arcline_parse['NeI']['min_intensity'] = 500.
    arcline_parse['NeI']['min_Aki']  = 1. # NOT GOOD FOR DEIMOS, DESI
    #arcline_parse['NeI']['min_wave'] = 5700.
    arcline_parse['NeI']['min_wave'] = 5850. # NOT GOOD FOR DEIMOS?
    # ZnI
    arcline_parse['ZnI'] = copy.deepcopy(dict_parse)
    arcline_parse['ZnI']['min_intensity'] = 50.
    #
    return arcline_parse

########################################################
# Fiber routines
########################################################

def fiber_gauss(flat, xtrc, xerr, box_radius=2, max_iter=5, debug=False, verbose=False):
    """Find the PSF sigma for each fiber
    This serves as an initial guess to what follows

    Parameters
    ----------
    flat : ndarray of fiber flat image
    xtrc: ndarray of fiber traces
    xerr: ndarray of error in fiber traces
    box_radius: int, optinal
          Radius of boxcar extraction in pixels
    max_iter : int, optional
      Maximum number of iterations for rejection

    Returns
    -------
    gauss
      list of Gaussian sigma
    """
    log=get_logger()
    log.warn("fiber_gauss uses astropy.modeling.  Consider an alternative")
    # Init
    nfiber = xtrc.shape[1]
    ny = xtrc.shape[0]
    iy = np.arange(ny).astype(int)
    # Mask
    mask = np.zeros_like(flat,dtype=int)
    # Sub images
    xpix_img = np.outer(np.ones(flat.shape[0]),np.arange(flat.shape[1]))
    # Gaussian fit
    g_init = models.Gaussian1D(amplitude=1., mean=0., stddev=1.)
    g_init.amplitude.fixed = True
    g_init.mean.fixed = True
    fitter = fitting.LevMarLSQFitter()

    # Loop on fibers
    gauss = []
    for ii in xrange(nfiber):
        if (ii % 25 == 0): # & verbose:
            log.info("Working on fiber {:d} of {:d}".format(ii,nfiber))
        mask[:] = 0
        ixt = np.round(xtrc[:,ii]).astype(int)
        for jj,ibox in enumerate(range(-box_radius,box_radius+1)):
            ix = ixt + ibox
            mask[iy,ix] = 1
        dx_img = xpix_img - np.outer(xtrc[:,ii],np.ones(flat.shape[1]))
        # Sum
        flux = np.sum(mask*flat,axis=1)
        flux = np.maximum(flux,1.)
        # Normalize
        nrm_img = flat / np.outer(flux,np.ones(flat.shape[1]))
        # Gaussian
        cpix = np.where(np.abs(dx_img)<0.10)
        if len(cpix[0]) < 50:
            cpix = np.where(np.abs(dx_img)<0.40)
        amp = np.median(nrm_img[cpix])
        g_init.amplitude.value = amp # Fixed
        fdimg = dx_img[mask==1].flatten()
        fnimg = nrm_img[mask==1].flatten()
        # Guess at sigma
        gdfn = (fnimg < amp) & (fnimg > 0.)
        all_sig = np.abs(fdimg[gdfn]) / np.sqrt( np.log(amp)-np.log(fnimg[gdfn]) )
        g_init.stddev.value = np.median(all_sig[np.where((np.abs(fdimg[gdfn])>1) & (np.abs(fdimg[gdfn])<1.5) & (np.isfinite(all_sig)))])

        # Initial fit (need to mask!)
        parm = fitter(g_init, fdimg, fnimg)
        # Iterate
        iterate = True
        nrej = 0
        niter = 0
        while iterate & (niter < max_iter):
            # Clip
            resid = parm(fdimg) - fnimg
            resid_mask = sigma_clip(resid, sig=4.)
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
            log.info("Rejected {:d} in {:d} iterations".format(nrej,niter))

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
    log=get_logger()
    log.info("starting")
    # Init
    Nbundle = 20
    Nfiber = 25 # Fibers per bundle
    # Set ypos for peak finding
    if ypos is None:
        ypos = flat.shape[0]//2

    # Cut image
    cutimg = flat[ypos-15:ypos+15, :]

    # Smash
    cut = np.median(cutimg, axis=0)

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
        log.warn('Found the wrong number of total fibers: {:d}'.format(len(xpk)))
    else:
        log.info('Found {:d} fibers'.format(len(xpk)))
    # Find bundles
    xsep = np.roll(xpk,-1) - xpk
    medsep = np.median(xsep)
    bundle_ends = np.where(np.abs(xsep-medsep) > 0.5*medsep)[0]
    if len(bundle_ends) != Nbundle:
        log.warn('Found the wrong number of bundles: {:d}'.format(len(bundle_ends)))
    else:
        log.info('Found {:d} bundles'.format(len(bundle_ends)))
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
    ny = xset.shape[0]
    ntrace = xset.shape[1]
    xnew = np.zeros_like(xset)
    fits = []
    yval = np.arange(ny)
    for ii in xrange(ntrace):
        mask = xerr[:,ii] > 900.
        nmask = np.sum(mask)
        # Fit with rejection
        dfit, mask = dufits.iter_fit(yval, xset[:,ii], func, order, sig_rej=sigrej,
            weights=1./xerr[:,ii], initialmask=mask, maxone=True)#, sigma=xerr[:,ii])
        # Stats on residuals
        nmask_new = np.sum(mask)-nmask 
        if nmask_new > 50:
            pdb.set_trace()
            raise ValueError('Rejected too many points [may need to increase for z camera with CRs: {:d}'.format(nmask_new))
        # Save
        xnew[:,ii] = dufits.func_val(yval,dfit)
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
# Output
#####################################################################

def write_psf(outfile, xfit, fdicts, gauss, wv_solns, ncoeff=5):
    """ Write the output to a Base PSF format

    Parameters
    ----------
    outfile : str
      Output file
    xfit : ndarray
      Traces
    gauss : list
      List of gaussian sigmas
    fdicts : list
      List of trace fits
    wv_solns : list
      List of wavelength calibrations
    ncoeff : int
      Number of Legendre coefficients in fits
    """
    #
    ny = xfit.shape[0]
    nfiber = xfit.shape[1]
    XCOEFF = np.zeros((nfiber, ncoeff))
    YCOEFF = np.zeros((nfiber, ncoeff))
    # Find WAVEMIN, WAVEMAX
    WAVEMIN = np.min([id_dict['wave_min'] for id_dict in wv_solns]) - 1.
    WAVEMAX = np.min([id_dict['wave_max'] for id_dict in wv_solns]) + 1.
    wv_array = np.linspace(WAVEMIN, WAVEMAX, num=ny)
    # Fit Legendre to y vs. wave
    for ii,id_dict in enumerate(wv_solns):
        # Fit y vs. wave
        yleg_fit, mask = dufits.iter_fit(np.array(id_dict['id_wave']), np.array(id_dict['id_pix']), 'legendre', ncoeff-1, xmin=WAVEMIN, xmax=WAVEMAX, niter=5)
        YCOEFF[ii, :] = yleg_fit['coeff']
        # Fit x vs. wave
        yval = dufits.func_val(wv_array, yleg_fit)
        xtrc = dufits.func_val(yval, fdicts[ii])
        xleg_fit,mask = dufits.iter_fit(wv_array, xtrc, 'legendre', ncoeff-1, xmin=WAVEMIN, xmax=WAVEMAX, niter=5)
        XCOEFF[ii, :] = xleg_fit['coeff']

    # Write the FITS file
    prihdu = fits.PrimaryHDU(XCOEFF)
    prihdu.header['WAVEMIN'] = WAVEMIN
    prihdu.header['WAVEMAX'] = WAVEMAX

    yhdu = fits.ImageHDU(YCOEFF)
    gausshdu = fits.ImageHDU(np.array(gauss))

    hdulist = fits.HDUList([prihdu, yhdu, gausshdu])
    hdulist.writeto(outfile, clobber=True)



#####################################################################
#####################################################################            
# Utilities
#####################################################################            


#####################################################################            
#####################################################################            
#####################################################################            
# QA
#####################################################################            

def qa_fiber_peaks(xpk, cut, pp=None, figsz=None, nper=100):
    """ Generate a QA plot for the fiber peaks

    Args:
        xpk: x positions on the CCD of the fiber peaks at a ypos
        cut: Spatial cut through the detector
        pp: PDF file pointer
        figsz: figure size, optional
        nper: number of fibers per row in the plot, optional

    """
    # Init
    if figsz is None:
        figsz = glbl_figsz

    nfiber = xpk.size
    nrow = (nfiber // nper) + ((nfiber % nper) > 0)
    xplt = np.arange(cut.size)
    # Plots
    gs = gridspec.GridSpec(nrow, 1)
    plt.figure(figsize=figsz)
    # Loop
    for ii in range(nrow):
        ax = plt.subplot(gs[ii])
        i0 = ii*nper
        i1 = i0 + nper
        ax.plot(xplt,cut, 'k-')
        ax.plot(xpk, cut[xpk],'go')
        xmin = np.min(xpk[i0:i1])-10.
        xmax = np.max(xpk[i0:i1])+10.
        ax.set_xlim(xmin,xmax)
    # Save and close
    if pp is not None:
        pp.savefig(bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def qa_fiber_Dx(xfit, fdicts, pp=None, figsz=None):
    """ Show the spread in the trace per fiber

    Used to diagnose the traces

    Args:
        xfit: traces
        fdicts: dict of the traces
        pp: PDF file pointer
        figsz: figure size, optional

    """
    #
    if figsz is None:
        figsz = glbl_figsz
    # Calculate Dx
    nfiber = xfit.shape[1]
    Dx = []
    for ii in range(nfiber):
        Dx.append(np.max(xfit[:, ii])-np.min(xfit[:, ii]))
    # Plot
    plt.figure(figsize=figsz)
    plt.scatter(np.arange(nfiber), np.array(Dx))
    # Label
    plt.xlabel('Fiber', fontsize=17.)
    plt.ylabel(r'$\Delta x$ (pixels)', fontsize=17.)
    # Save and close
    if pp is None:
        plt.show()
    else:
        pp.savefig(bbox_inches='tight')
    plt.close()

def qa_fiber_gauss(gauss, pp=None, figsz=None):
    """ Show the Gaussian (sigma) fits to each fiber

    Args:
        gauss: Gaussian of each fiber
        pp: PDF file pointer
        figsz: figure size, optional

    """
    #
    if figsz is None:
        figsz = glbl_figsz
    # Calculate Dx
    nfiber = gauss.size
    # Plot
    plt.figure(figsize=figsz)
    plt.scatter(np.arange(nfiber), gauss)
    # Label
    plt.xlabel('Fiber', fontsize=17.)
    plt.ylabel('Gaussian sigma (pixels)', fontsize=17.)
    # Save and close
    if pp is None:
        plt.show()
    else:
        pp.savefig(bbox_inches='tight')
    plt.close()

def qa_arc_spec(all_spec, all_soln, pp, figsz=None):
    """ Generate QA plots of the arc spectra with IDs

    Args:
        all_spec: Arc 1D fiber spectra
        all_soln: Wavelength solutions
        pp: PDF file pointer
        figsz: figure size, optional

    """
    # Init
    if figsz is None:
        figsz = glbl_figsz
    nfiber = len(all_soln)
    npix = all_spec.shape[0]
    #
    nrow = 2
    ncol = 3
    # Plots
    gs = gridspec.GridSpec(nrow, ncol)
    plt.figure(figsize=figsz)
    # Loop
    for ii in range(nrow*ncol):
        ax = plt.subplot(gs[ii])
        idx = ii * (nfiber//(nrow*ncol))
        yspec = np.log10(np.maximum(all_spec[:,idx],1))
        ax.plot(np.arange(npix), yspec, 'k-')
        ax.set_xlabel('Pixel')
        ax.set_ylabel('log Flux')
        # ID
        id_dict = all_soln[idx]
        for jj,xpixpk in enumerate(id_dict['id_pix']):
            ax.text(xpixpk, yspec[int(np.round(xpixpk))], '{:g}'.format(id_dict['id_wave'][jj]), ha='center',color='red', rotation=90.)

    # Save and close
    pp.savefig(bbox_inches='tight')
    plt.close()


def qa_fiber_arcrms(all_soln, pp, figsz=None):
    """ Show the RMS of the wavelength solutions vs. fiber

    Args:
        all_soln: Wavelength solutions
        pp: PDF file pointer
        figsz: figure size, optional

    """
    #
    if figsz is None:
        figsz = glbl_figsz
    # Calculate Dx
    nfiber = len(all_soln)
    rms = [id_dict['rms'] for id_dict in all_soln]
    # Plot
    plt.figure(figsize=figsz)
    plt.scatter(np.arange(nfiber), np.array(rms))
    # Label
    plt.xlabel('Fiber', fontsize=17.)
    plt.ylabel('RMS (pixels)', fontsize=17.)
    # Save and close
    pp.savefig(bbox_inches='tight')
    plt.close()


def qa_fiber_dlamb(all_spec, all_soln, pp, figsz=None):
    """ Show the Dlamb of the wavelength solutions vs. fiber

    Args:
        all_soln: Wavelength solutions
        pp: PDF file pointer
        figsz: figure size, optional

    """
    #
    if figsz is None:
        figsz = glbl_figsz
    # Calculate Dx
    nfiber = len(all_soln)
    npix = all_spec.shape[0]
    xval = np.arange(npix)
    dlamb = []
    for ii in range(nfiber):
        idict = all_soln[ii]
        wave = dufits.func_val(xval,idict['final_fit_pix'])
        dlamb.append(np.median(np.abs(wave-np.roll(wave,1))))
    # Plot
    plt.figure(figsize=figsz)
    plt.scatter(np.arange(nfiber), np.array(dlamb))
    # Label
    plt.xlabel('Fiber', fontsize=17.)
    plt.ylabel(r'$\Delta \lambda$ (Ang)', fontsize=17.)
    # Save and close
    pp.savefig(bbox_inches='tight')
    plt.close()


def qa_fiber_trace_qa(flat, xtrc, outfil=None, Nfiber=25, isclmin=0.5):
    ''' Generate a QA plot for the fiber traces

    Parameters
    ----------
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
