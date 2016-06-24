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
import math
import time
import os
import argparse
from pkg_resources import resource_exists, resource_filename

from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
from astropy.table import Table, Column, vstack
from astropy.io import fits

from desispec.util import set_backend
set_backend()

from matplotlib import pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages

from desispec.log import get_logger
from desiutil import funcfits as dufits

glbl_figsz = (16,9)

########################################################
# High level wrapper
# TODO: This was ported from the original bin/desi_bootcalib so that it could
# be called independently by quicklook, but it needs to be coordinated with
# desispec.scripts.bootcalib.main()
########################################################

def bootcalib(deg,flatimage,arcimage):
    """
    Args:
        deg: Legendre polynomial degree to use to fit
        flatimage: desispec.image.Image object of flatfield
        arcimage: desispec.image.Image object of arc

    Mostly inherited from desispec/bin/desi_bootcalib directly as needed

    Returns:
        xfit, fdicts, gauss, all_wave_soln

    TODO: document what those return objects are
    """

    camera=flatimage.camera
    flat=flatimage.pix
    ny=flat.shape[0]

    xpk,ypos,cut=find_fiber_peaks(flat)
    xset,xerr=trace_crude_init(flat,xpk,ypos)
    xfit,fdicts=fit_traces(xset,xerr)
    gauss=fiber_gauss(flat,xfit,xerr)

    #- Also need wavelength solution not just trace

    arc=arcimage.pix
    arc_ivar=arcimage.ivar
    all_spec=extract_sngfibers_gaussianpsf(arc,arc_ivar,xfit,gauss)
    llist=load_arcline_list(camera)
    dlamb,wmark,gd_lines,line_guess=load_gdarc_lines(camera)

    #- Solve for wavelengths
    all_wv_soln=[]
    all_dlamb=[]
    for ii in range(all_spec.shape[1]):
        spec=all_spec[:,ii]
        pixpk=find_arc_lines(spec)
        id_dict=id_arc_lines(pixpk,gd_lines,dlamb,wmark,line_guess=line_guess)
        id_dict['fiber']=ii
        #- Find the other good ones
        if camera == 'z':
            inpoly = 3  # The solution in the z-camera has greater curvature
        else:
            inpoly = 2
        add_gdarc_lines(id_dict, pixpk, gd_lines, inpoly=inpoly)
        #- Now the rest
        id_remainder(id_dict, pixpk, llist)
        #- Final fit wave vs. pix too
        final_fit, mask = dufits.iter_fit(np.array(id_dict['id_wave']), np.array(id_dict['id_pix']), 'polynomial', 3, xmin=0., xmax=1.)
        rms = np.sqrt(np.mean((dufits.func_val(np.array(id_dict['id_wave'])[mask==0],final_fit)-np.array(id_dict['id_pix'])[mask==0])**2))
        final_fit_pix,mask2 = dufits.iter_fit(np.array(id_dict['id_pix']), np.array(id_dict['id_wave']),'legendre',deg, niter=5)

        id_dict['final_fit'] = final_fit
        id_dict['rms'] = rms
        id_dict['final_fit_pix'] = final_fit_pix
        id_dict['wave_min'] = dufits.func_val(0,final_fit_pix)
        id_dict['wave_max'] = dufits.func_val(ny-1,final_fit_pix)
        id_dict['mask'] = mask
        all_wv_soln.append(id_dict)

    return xfit, fdicts, gauss,all_wv_soln


########################################################
# Arc/Wavelength Routines (Linelists come next)
########################################################

def find_arc_lines(spec,rms_thresh=7.,nwidth=5):
    """Find and centroid arc lines in an input spectrum

    Parameters
    ----------
    spec : ndarray
      Arc line spectrum
    rms_thresh : float
      RMS threshold scale
    nwidth : int
      Line width to test over
    """
    # Threshold criterion
    npix = spec.size
    spec_mask = sigma_clip(spec, sigma=4., iters=5)
    rms = np.std(spec_mask)
    thresh = rms*rms_thresh
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


def load_gdarc_lines(camera, vacuum=True,lamps=None):

    """Loads a select set of arc lines for initial calibrating

    Parameters
    ----------
    camera : str
      Camera ('b', 'g', 'r')
    vacuum : bool, optional
      Use vacuum wavelengths
    lamps : optional numpy array of ions, ex np.array(["HgI","CdI","ArI","NeI"])

    Returns
    -------
    dlamb : float
      Dispersion for input camera
    wmark : float
      wavelength to key off of [???]
    gd_lines : ndarray
      Array of lines expected to be recorded and good for ID
    line_guess : int or None
      Guess at the line index corresponding to wmark (default is to guess the 1/2 way point)
    """
    log=get_logger()

    if lamps is None :
        lamps=np.array(["HgI","CdI","ArI","NeI"])

    if camera[0] == 'b':

        lines={}
        if vacuum :
            lines["HgI"]=[3651.198, 3655.883, 3664.327, 4047.708, 4078.988, 4359.56, 5462.268, 5771.210, 5792.276]
            lines["CdI"]=[3611.5375, 4679.4587, 4801.2540, 5087.2393]
            lines["NeI"]=[5854.1101, 5883.5252, 5946.4810]
            lines["ArI"]=[]
            lines["KrI"]=[]
            wmark = 4359.56  # Hg
        else :
            lines["HgI"]=[4046.57, 4077.84, 4358.34, 5460.75, 5769.598, 5790.670]
            lines["CdI"]=[3610.51, 3650.157, 4678.15, 4799.91, 5085.822]
            lines["NeI"]=[5881.895, 5944.834]
            lines["ArI"]=[]
            lines["KrI"]=[]
            wmark = 4358.34  # Hg

        gd_lines=np.array([])
        for lamp in lamps :
            gd_lines=np.append(gd_lines,lines[lamp])

        dlamb = 0.589
        line_guess = None

    elif camera[0] == 'r':

        lines={}
        if vacuum :
            lines["HgI"] = [5771.210]
            lines["NeI"] = [5854.1101, 5946.481, 6144.7629, 6404.018, 6508.3255,
                   6680.1205, 6718.8974, 6931.3787, 7034.3520,
                   7175.9154, 7247.1631, 7440.9469]
            lines["ArI"] = [6965.431]#, 7635.106, 7723.761]
            lines["KrI"] = [7603.6384,7696.6579]
            lines["CdI"] = []
            wmark = 6718.8974 # Ne

        else :
            lines["HgI"] = [5769.598]
            lines["NeI"] = [5852.4878, 5944.834, 6143.062, 6402.246, 6506.528,
                            #6532.8824, #6598.9528,
                            6678.2766, 6717.043, 6929.4672, 7032.4128,
                            7173.9380, 7245.1665, 7438.898]
            lines["ArI"] = [6965.431]#, 7635.106, 7723.761]
            lines["KrI"] = [7601.55,7694.54]
            lines["CdI"] = []
            wmark = 6717.043  # Ne

        gd_lines=np.array([])
        for lamp in lamps :
            gd_lines=np.append(gd_lines,lines[lamp])

        dlamb = 0.527
        line_guess = 24

    elif camera[0] == 'z':

        lines={}
        if vacuum :
            lines["NeI"] = [7440.9469, 7490.9335, 7537.8488,
                            7945.3654, 8138.6432, 8302.6062,
                            8379.9093, 8497.6932, 8593.6184, 8637.0190, 8656.7599,
                            8786.1660, 8921.9496, 9151.1829, 9204.2841, 9427.9655,9536.7793,9668.0709]
                            #8786.1660, 8921.9496, 8991.024,  9151.1829, 9204.2841, 9222.59, 9227.03, 9303.4053,
                            #9329.0663, 9375.8796, 9427.9655,9461.806,9489.2849,9536.7793,9550.0241,9668.0709]
            lines["KrI"] = [7603.6384,7696.6579,7856.9844,8061.7211,8106.5945,8192.3082,8300.3907,8779.1607,8931.1447]
            lines["ArI"] = [9125.471]
            lines["HgI"] = []
            lines["CdI"] = []
            wmark = 8593.6184 # Ne
        else :
            lines["NeI"] = [7438.898, 7488.8712, 7535.7739,
                            7943.1805, 8136.4061, 8300.3248,
                            8377.6070, 8495.3591, 8591.2583, 8634.6472, 8654.3828,
                            8783.7539, 8919.5007, 9148.6720, 9201.7588, 9425.3797]
            lines["KrI"] = [7601.55,7694.54,7854.82,8059.50,8104.36,8190.06,8298.11,8776.75,8928.69]
            lines["ArI"] = [9122.97,9784.50]
            lines["HgI"] = []
            lines["CdI"] = []
            wmark = 8591.2583

        gd_lines=np.array([])
        for lamp in lamps :
            gd_lines=np.append(gd_lines,lines[lamp])

        dlamb = 0.599  # Ang
        line_guess = 17

    else:
        log.error('Bad camera')

    # Sort and return
    gd_lines.sort()
    return dlamb, wmark, gd_lines, line_guess


def add_gdarc_lines(id_dict, pixpk, gd_lines, inpoly=2, toler=10., verbose=False, debug=False):
    """Attempt to identify and add additional goodlines

    Parameters
    ----------
    id_dict : dict
      dict of ID info
    pixpk : ndarray
      Pixel locations of detected arc lines
    toler : float
      Tolerance for a match (pixels)
    gd_lines : ndarray
      array of expected arc lines to be detected and identified
    inpoly : int, optional
      Order of polynomial for fitting for initial set of lines

    Returns
    -------
    id_dict : dict
      Filled with complete set of IDs and the final polynomial fit
    """
    log=get_logger()
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
        # newx
        newx = dufits.func_val(new_wv, id_dict['fit'])
        # Match
        mnm = np.min(np.abs(pixpk-newx))
        if mnm > toler:
            log.warn("No match for {:g} in fiber {:d}".format(new_wv, id_dict['fiber']))
            continue
        imin = np.argmin(np.abs(pixpk-newx))
        if debug:
            print(new_wv, np.min(np.abs(pixpk-newx)))
        # Append and sort
        wvval.append(new_wv)
        wvval.sort()
        newtc = pixpk[imin]
        idx.append(imin)
        idx.sort()
        xval.append(newtc)
        xval.sort()
        # Fit (should reject 1 someday)
        if len(xval) > 7:
            npoly = inpoly+1
        else:
            npoly = inpoly
        npoly=min(len(xval)-1,npoly)
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
    deg=max(1,min(len(id_dict['id_pix'])-2,3))
    pixwv_fit = dufits.func_fit(np.array(id_dict['id_pix']),np.array(id_dict['id_wave']),'polynomial',deg,xmin=0.,xmax=1.)
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


def id_arc_lines(pixpk, gd_lines, dlamb, wmark, toler=0.2,
                 line_guess=None, verbose=False):
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
    line_guess : int, optional
      Guess at the line index corresponding to wmark (default is to guess the 1/2 way point)

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
    if line_guess is None:
        line_guess = ndet//2
    guesses = line_guess + np.arange(-4, 7)
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
                guess_rms = dict(guess=guess, im1=itm1, im2=imtm2,
                                 rms=999., ip1=None, ip2=imtp2)
                all_guess_rms.append(guess_rms)
                if imtp2 == (icen-1):
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
                deg=max(1,min(wvval.size-2,2))
                pfit = dufits.func_fit(wvval,xval,'polynomial',deg,xmin=0.,xmax=1.)
                # Clip one here and refit
                #   NOT IMPLEMENTED YET
                # RMS (in pixel space)
                rms = np.sqrt(np.sum((xval-dufits.func_val(wvval,pfit))**2)/xval.size)
                guess_rms['rms'] = rms
                #if guess == 22:
                #    pdb.set_trace()
                # Save fit too
                guess_rms['fit'] = pfit
        # Take best RMS
        if len(all_guess_rms) > 0:
            all_rms = np.array([idict['rms'] for idict in all_guess_rms])
            imn = np.argmin(all_rms)
            rms_dicts.append(all_guess_rms[imn])
    # Find the best one
    all_rms = np.array([idict['rms'] for idict in rms_dicts])
    # Allow for (very rare) failed solutions
    imin = np.argmin(all_rms)
    id_dict = rms_dicts[imin]
    id_dict['status'] = 'ok'
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

def remove_duplicates(wy,w,y_id,w_id) :
    # might be several identical w_id
    y_id=np.array(y_id).astype(int)
    w_id=np.array(w_id).astype(int)
    y_id2=[]
    w_id2=[]
    for j in np.unique(w_id) :
        w_id2.append(j)
        ii=y_id[w_id==j]
        if ii.size==1 :
            y_id2.append(ii[0])
        else :
            i=np.argmin(np.abs(wy[ii]-w[j]))
            #print("rm duplicate %d : %s -> %d"%(j,str(ii),ii[i]))
            y_id2.append(ii[i])
    y_id2=np.array(y_id2).astype(int)
    w_id2=np.array(w_id2).astype(int)
    tmp=np.argsort(w[w_id2])
    y_id2=y_id2[tmp]
    w_id2=w_id2[tmp]
    return y_id2,w_id2


def refine_solution(y,w,y_id,w_id,deg=3) :
    log=get_logger()
    transfo=np.poly1d(np.polyfit(y[y_id],w[w_id],deg=deg))
    wy=transfo(y)
    y_id,w_id=remove_duplicates(wy,w,y_id,w_id)
    nmatch=len(y_id)
    #log.info("init nmatch=%d rms=%f wave=%s"%(nmatch,np.std(wy[y_id]-w[w_id]),w[w_id]))
    #log.info("init nmatch=%d rms=%f"%(nmatch,np.std(wy[y_id]-w[w_id])))
    if nmatch<deg+1 :
        log.error("error : init nmatch too small")
        return y_id,w_id,1000.,0

    rms=0.

    # loop on fit of transfo, pairing, cleaning
    for loop in range(200) :

        # compute transfo
        transfo=np.poly1d(np.polyfit(y[y_id],w[w_id],deg=deg))

        # apply transfo to measurements
        wy=transfo(y)
        previous_rms = rms+0.
        rms=np.std(wy[y_id]-w[w_id])

        # match lines
        mdiff0=max(2.,rms*2.) # this is a difficult parameter to tune, either loose lever arm, or have false matches !!
        mdiff1=10. # this is a difficult parameter to tune, either loose lever arm, or have false matches !!
        unmatched_indices=np.setdiff1d(np.arange(y.size),y_id)
        for i,wi in zip(unmatched_indices,wy[unmatched_indices]) :
            dist=np.abs(wi-w)
            jj=np.argsort(dist)
            for j,o in enumerate(jj) :
                if j in w_id :
                    continue
                if dist[j]<mdiff0 or ( o<jj.size-1 and dist[j]<mdiff1 and dist[j]<0.3*dist[jj[o+1]]) :
                    y_id=np.append(y_id,i)
                    w_id=np.append(w_id,j)
        previous_nmatch = nmatch+0
        nmatch=len(y_id)

        #log.info("iter #%d nmatch=%d rms=%f"%(loop,nmatch,rms))
        if nmatch < deg+1 :
           log.error("error init nmatch too small")
           y_id=[]
           w_id=[]
           rms=100000.
           return y_id,w_id,rms,loop

        if nmatch==previous_nmatch and abs(rms-previous_rms)<0.01 and loop>=1 :
            break
        if nmatch>=min(w.size,y.size) :
            #print("break because %d>=min(%d,%d)"%(nmatch,w.size,y.size))
            break

    return y_id,w_id,rms,loop


def compute_triplets(wave) :

    triplets=[]
    wave=np.sort(wave)
    for i1,w1 in enumerate(wave[:-1]) :
        for i2,w2 in enumerate(wave[i1+1:]) :
            for i3,w3 in enumerate(wave[i1+i2+2:]) :
                triplet=[w1,w2,w3,i1,i1+1+i2,i1+i2+2+i3,w2-w1,w3-w1,w2**2-w1**2,w3**2-w1**2]
                #print(triplet)
                #print(wave[i1],wave[i1+1+i2],wave[i1+i2+2+i3])
                triplets.append(triplet)
    return np.array(triplets)

def id_arc_lines_using_triplets(y,w,dwdy_prior,d2wdy2_prior=1.5e-5,toler=0.2,ntrack=50):
    """Match (as best possible), a set of the input list of expected arc lines to the detected list

    Parameters
    ----------
    y : ndarray
      Pixel locations of detected arc lines
    w : ndarray
      array of expected arc lines to be detected and identified
    dwdy : float
      Average dispersion in the spectrum
    d2wdy2_prior : float
      Prior on second derivative
    toler : float, optional
      Tolerance for matching (20%)
    ntrack : max. number of solutions to be tracked

    Returns
    -------
    id_dict : dict
      dict of identified lines
    """

    log=get_logger()
    #log.info("y=%s"%str(y))
    #log.info("w=%s"%str(w))

    # compute triplets of waves of y positions
    y_triplets = compute_triplets(y)
    w_triplets = compute_triplets(w)

    # each pair of triplet defines a 2nd order polynomial (chosen centered on y=2000)
    # w = a*(y-2000)**2+b*(y-2000)+c
    # w = a*y**2-4000*a*y+b*y+cst
    # w = a*(y**2-4000*y)+b*y+cst
    # dw_12 = a*(dy2_12-4000*dy_12)+b*dy_12
    # dw_13 = a*(dy2_13-4000*dy_13)+b*dy_12
    # dw_12 = a*cdy2_12+b*dy_12
    # dw_13 = a*cdy2_13+b*dy_13
    # with cdy2_12=dy2_12-4000*dy_12
    # and  cdy2_13=dy2_13-4000*dy_13
    # idet = 1./(dy_13*cdy2_12-dy_12*cdy2_13)
    # a = idet*(dy_13*dw_12-dy_12*dw_13)
    # b = idet*(-cdy2_13*dw_12+cdy2_12*dw_13)

    #triplet=[w1,w2,w3,i1,i1+1+i2,i1+i2+2+i3,w2-w1,w3-w1,w2**2-w1**2,w3**2-w1**2]
    dy_12=y_triplets[:,6]
    dy_13=y_triplets[:,7]
    #dy2_12=y_triplets[:,8]
    #dy2_13=y_triplets[:,9]
    # centered version
    cdy2_12=y_triplets[:,8]-4000.*y_triplets[:,6]
    cdy2_13=y_triplets[:,9]-4000.*y_triplets[:,7]
    idet=1./(dy_13*cdy2_12-dy_12*cdy2_13)

    # fill histogram with polynomial coefs and first index of each triplet in the pair for all pairs of triplets(y,w)
    # create the 4D histogram
    ndwdy    = 41
    nd2wdy2  = 21
    dwdy_min = dwdy_prior*(1-toler)
    dwdy_max = dwdy_prior*(1+toler)
    dwdy_step = (dwdy_max-dwdy_min)/ndwdy
    d2wdy2_min = -d2wdy2_prior
    d2wdy2_max = +d2wdy2_prior
    d2wdy2_step = (d2wdy2_max-d2wdy2_min)/nd2wdy2
    histogram = np.zeros((ndwdy,nd2wdy2,len(y),len(w))) # definition of the histogram

    # fill the histogram
    for w_triplet in w_triplets :
        #d2wdy2 = idet*(dy_13*w_triplet[6]-dy_12*w_triplet[7])
        #dwdy   = idet*(-cdy2_13*w_triplet[6]+cdy2_12*w_triplet[7])
        # bins in the histogram
        dwdy_bin   = ((idet*(-cdy2_13*w_triplet[6]+cdy2_12*w_triplet[7])-dwdy_min)/dwdy_step).astype(int)
        d2wdy2_bin = ((idet*(dy_13*w_triplet[6]-dy_12*w_triplet[7])-d2wdy2_min)/d2wdy2_step).astype(int)
        pairs_in_histo=np.where((dwdy_bin>=0)&(dwdy_bin<ndwdy)&(d2wdy2_bin>=0)&(d2wdy2_bin<nd2wdy2))[0]
        # fill histo
        iw=w_triplet[3]
        for a,b,c in zip(dwdy_bin[pairs_in_histo],d2wdy2_bin[pairs_in_histo],y_triplets[pairs_in_histo,3]) :
            histogram[a,b,c,iw] += 1

    # find max bins in the histo
    histogram_ravel = histogram.ravel()
    best_histo_bins = histogram_ravel.argsort()[::-1]
    #log.info("nmatch in first bins=%s"%histogram.ravel()[best_histo_bins[:3]])

    best_y_id=[]
    best_w_id=[]
    best_rms=1000.

    # loop on best matches ( = most populated bins)
    count=0
    for histo_bin in best_histo_bins[:ntrack] :
        if  histogram_ravel[histo_bin]<4 :
            log.warning("stopping here")
            break
        count += 1
        dwdy_best_bin,d2wdy2_best_bin,iy_best_bin,iw_best_bin = np.unravel_index(histo_bin, histogram.shape) # bin coord
        #print("bins=",dwdy_best_bin,d2wdy2_best_bin,iy_best_bin,iw_best_bin)

        # pairs of triplets in this histo bin
        w_id=np.array([])
        y_id=np.array([])
        wok=np.where(w_triplets[:,3]==iw_best_bin)[0]
        yok=np.where(y_triplets[:,3]==iy_best_bin)[0]
        for w_triplet in w_triplets[wok] :
            #d2wdy2 = idet[yok]*(dy_13[yok]*w_triplet[6]-dy_12[yok]*w_triplet[7])
            #dwdy   = idet[yok]*(-cdy2_13[yok]*w_triplet[6]+cdy2_12[yok]*w_triplet[7])
            # bins in the histogram
            dwdy_bin   = ((idet[yok]*(-cdy2_13[yok]*w_triplet[6]+cdy2_12[yok]*w_triplet[7])-dwdy_min)/dwdy_step).astype(int)
            d2wdy2_bin = ((idet[yok]*(dy_13[yok]*w_triplet[6]-dy_12[yok]*w_triplet[7])-d2wdy2_min)/d2wdy2_step).astype(int)
            wyok=yok[np.where((dwdy_bin==dwdy_best_bin)&(d2wdy2_bin==d2wdy2_best_bin))[0]]
            for y_triplet in y_triplets[wyok] :
                y_id=np.append(y_id,y_triplet[3:6])
                w_id=np.append(w_id,w_triplet[3:6])

        # now need to rm duplicates
        nw=len(w)
        ny=len(y)
        unique_common_id=np.unique(y_id.astype(int)*nw+w_id.astype(int))
        y_id=(unique_common_id/nw).astype(int)
        w_id=(unique_common_id%nw).astype(int)
        ordering=np.argsort(y[y_id])
        y_id=y_id[ordering]
        w_id=w_id[ordering]
        # refine
        y_id,w_id,rms,niter=refine_solution(y,w,y_id,w_id)
        #log.info("get solution with %d match and rms=%f (niter=%d)"%(len(y_id),rms,niter))
        if (len(y_id)>len(best_y_id) and rms<max(1,best_rms)) or (len(y_id)==len(best_y_id) and rms<best_rms) or (best_rms>1 and rms<1 and len(y_id)>=8) :
            log.info("new best solution #%d with %d match and rms=%f (niter=%d)"%(count,len(y_id),rms,niter))
            #log.info("previous had %d match and rms=%f"%(len(best_y_id),best_rms))
            best_y_id = y_id
            best_w_id = w_id
            best_rms = rms

        # stop at some moment
        if best_rms<0.2 and len(y_id)>=min(15,min(len(y),len(w))) :
            #log.info("stop here because we have a correct solution")
            break

    id_dict={}
    id_dict["status"]="ok"
    id_dict["first_id_idx"]=best_y_id
    id_dict["first_id_pix"]=y[best_y_id]
    id_dict["first_id_wave"]=w[best_w_id]
    id_dict["rms"]=best_rms

    id_dict["dlamb"]=dwdy_prior
    id_dict["icen"]=best_w_id[best_w_id.size/2]
    id_dict["wmark"]=w[id_dict["icen"]]

    deg=max(1,min(3,best_y_id.size-2))
    id_dict["fit"]= dufits.func_fit(w[best_w_id],y[best_y_id],'polynomial',deg,xmin=0.,xmax=1.)

    log.info("{:d} matched for {:d} detected and {:d} known as good, rms = {:g}".format(len(best_y_id),len(y),len(w),best_rms))

    #message="matched :"
    #for i,j in zip(best_y_id,best_w_id) :
    #    message += " y=%d:w=%d"%(y[i],w[j])
    #log.info(message)

    return id_dict

def use_previous_wave(new_id, old_id, new_pix, old_pix, tol=0.5):
    """ Uses the previous wavelength solution to fix the current

    Args:
        new_id:
        old_id:
        new_pix:
        old_pix:

    Returns:
        Stuff
    """
    log=get_logger()
    # Find offset in pixels
    min_off = []
    for pix in new_pix:
        imin = np.argmin(np.abs(old_pix-pix))
        min_off.append(old_pix[imin]-pix)
    off = np.median(min_off)

    # Find closest with small tolerance
    id_pix = []
    id_wave = []
    # Insure enough pixels (some failures are bad extractions)
    if len(new_pix) > len(old_pix)-5:
        for kk,oldpix in enumerate(old_id['id_pix']):
            mt = np.where(np.abs(new_pix-(oldpix-off)) < tol)[0]
            if len(mt) == 1:
                id_pix.append(new_pix[mt][0])
                id_wave.append(old_id['id_wave'][kk])
    else:  # Just apply offset
        log.warn("Completely kludging this fiber wavelength")
        id_wave = old_id['id_wave']
        id_pix = old_id['id_pix']-off
    # Fit
    new_id['id_wave'] = id_wave
    new_id['id_pix'] = id_pix


def fix_poor_solutions(all_wv_soln, all_dlamb, ny, ldegree):
    """ Identify solutions with poor RMS and replace

    Args:
        all_wv_soln: list of solutions
        all_dlamb: list of dispersion values

    Returns:
        Updated lists if there were poor RMS solutions

    """
    from scipy.signal import medfilt

    log=get_logger()
    #
    nfiber = len(all_dlamb)
    #med_dlamb = np.median(all_dlamb)
    dlamb_fit, dlamb_mask = dufits.iter_fit(np.arange(nfiber), np.array(all_dlamb), 'legendre', 4, xmin=0., xmax=1., sig_reg=10., max_rej=20)
    #med_res = np.median(np.abs(med_dlamb-np.array(all_dlamb)))
    #xval = np.linspace(0,nfiber,num=1000)
    #yval = dufits.func_val(xval, dlamb_fit)

    for ii,dlamb in enumerate(all_dlamb):
        id_dict = all_wv_soln[ii]
        #if (np.abs(dlamb - med_dlamb)/med_dlamb > 0.1) or (id_dict['rms'] > 0.7):
        #if (np.abs(dlamb - med_dlamb[ii]) > 10*med_res) or (id_dict['rms'] > 0.7):
        if (dlamb_mask[ii] == 1) or (id_dict['rms'] > 0.7):
            log.warn('Bad wavelength solution for fiber {:d}.  Using closest good one to guide..'.format(ii))
            if ii > nfiber/2:
                off = -1
            else:
                off = +1
            jj = ii + off

            jdict = all_wv_soln[jj]
            jdlamb = all_dlamb[jj]
            #while (np.abs(dlamb - med_dlamb)/med_dlamb > 0.1) or (jdict['rms'] > 0.7):
            #while (np.abs(jdlamb - med_dlamb[jj]) > 10*med_res) or (jdict['rms'] > 0.7):
            while (dlamb_mask[jj] == 1) or (jdict['rms'] > 0.7):
                jj += off
                jdict = all_wv_soln[jj]
                #jdlamb = all_dlamb[jj]
            # Bad solution; shifting to previous
            use_previous_wave(id_dict, jdict, id_dict['pixpk'], jdict['pixpk'])
            final_fit, mask = dufits.iter_fit(np.array(id_dict['id_wave']),
                                              np.array(id_dict['id_pix']), 'polynomial', 3, xmin=0., xmax=1.)
            rms = np.sqrt(np.mean((dufits.func_val(np.array(id_dict['id_wave'])[mask==0], final_fit)-
                                   np.array(id_dict['id_pix'])[mask==0])**2))
            final_fit_pix,mask2 = dufits.iter_fit(np.array(id_dict['id_pix']),
                                                  np.array(id_dict['id_wave']),'legendre',ldegree , niter=5)
            # Save
            id_dict['final_fit'] = final_fit
            id_dict['rms'] = rms
            id_dict['final_fit_pix'] = final_fit_pix
            id_dict['wave_min'] = dufits.func_val(0,final_fit_pix)
            id_dict['wave_max'] = dufits.func_val(ny-1,final_fit_pix)
            id_dict['mask'] = mask

########################################################
# Linelist routines
########################################################

def parse_nist(ion, vacuum=True):
    """Parse a NIST ASCII table.

    Note that the long ---- should have
    been commented out and also the few lines at the start.

    Taken from PYPIT

    Parameters
    ----------
    ion : str
      Name of ion
    vaccuum : bool, optional
      Use vacuum wavelengths
    """
    log=get_logger()
    # Find file
    medium = 'vacuum'
    if not vacuum:
        log.info("Using air wavelengths")
        medium = 'air'
    srch_file = "data/arc_lines/{0}_{1}.ascii".format(ion, medium)
    if not resource_exists('desispec', srch_file):
        log.error("Cannot find NIST file {:s}".format(srch_file))
        raise Exception("Cannot find NIST file {:s}".format(srch_file))
    # Read
    nist_file = resource_filename('desispec', srch_file)
    log.info("reading NIST file {:s}".format(nist_file))
    nist_tbl = Table.read(nist_file, format='ascii.fixed_width')
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


def load_arcline_list(camera, vacuum=True,lamps=None):

    """Loads arc line list from NIST files
    Parses and rejects

    Taken from PYPIT

    Parameters
    ----------
    lines : list
      List of ions to load
    vacuum : bool, optional
      Use vacuum wavelengths
    lamps : optional numpy array of ions, ex np.array(["HgI","CdI","ArI","NeI"])

    Returns
    -------
    alist : Table
      Table of arc lines
    """
    log=get_logger()
    wvmnx = None
    if lamps is None :
        if camera[0] == 'b':
            lamps = ['CdI','ArI','HgI','NeI','KrI']
        elif camera[0] == 'r':
            lamps = ['CdI','ArI','HgI','NeI','KrI']
        elif camera[0] == 'z':
            lamps = ['CdI','ArI','HgI','NeI','KrI']
        elif camera == 'all': # Used for specex
            lamps = ['CdI','ArI','HgI','NeI','KrI']
        else:
            log.error("Not ready for this camera")

    # Get the parse dict
    parse_dict = load_parse_dict()
    # Read rejection file
    medium = 'vacuum'
    if not vacuum:
        log.info("Using air wavelengths")
        medium = 'air'
    rej_file = resource_filename('desispec', "data/arc_lines/rejected_lines_{0}.yaml".format(medium))
    with open(rej_file, 'r') as infile:
        rej_dict = yaml.load(infile)
    # Loop through the NIST Tables
    tbls = []
    for iline in lamps:
        # Load
        tbl = parse_nist(iline, vacuum=vacuum)
        # Parse
        if iline in parse_dict.keys():
            tbl = parse_nist_tbl(tbl,parse_dict[iline])
        # Reject
        if iline in rej_dict.keys():
            log.info("Rejecting select {:s} lines".format(iline))
            tbl = reject_lines(tbl,rej_dict[iline])
        #print("DEBUG",iline)
        #print("DEBUG",tbl[['Ion','wave','RelInt']])
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


def reject_lines(tbl,rej_dict, rej_tol=0.1):
    """Rejects lines from a NIST table

    Taken from PYPIT

    Parameters
    ----------
    tbl : Table
      Read previously from NIST ASCII file
    rej_dict : dict
      Dict of rejected lines
    rej_tol : float, optional
      Tolerance for matching a line to reject to linelist (Angstroms)

    Returns
    -------
    tbl : Table
      Rows not rejected
    """
    msk = tbl['wave'] == tbl['wave']
    # Loop on rejected lines
    for wave in rej_dict.keys():
        close = np.where(np.abs(wave-tbl['wave']) < rej_tol)[0]
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
    arcline_parse['NeI']['min_intensity'] = 999.
    #arcline_parse['NeI']['min_Aki']  = 1. # NOT GOOD FOR DEIMOS, DESI
    #arcline_parse['NeI']['min_wave'] = 5700.
    arcline_parse['NeI']['min_wave'] = 5850. # NOT GOOD FOR DEIMOS?
    # ZnI
    arcline_parse['ZnI'] = copy.deepcopy(dict_parse)
    arcline_parse['ZnI']['min_intensity'] = 50.
    # KrI
    arcline_parse['KrI'] = copy.deepcopy(dict_parse)
    arcline_parse['KrI']['min_intensity'] = 50.
    return arcline_parse

########################################################
# Fiber routines
########################################################

def fiber_gauss(flat, xtrc, xerr, box_radius=2, max_iter=5, debug=False, verbose=False) :
    return fiber_gauss_new(flat, xtrc, xerr, box_radius, max_iter)

def fiber_gauss_new(flat, xtrc, xerr, box_radius=2, max_iter=5, debug=False, verbose=False):
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

    npix_y  = flat.shape[0]
    npix_x  = flat.shape[1]
    ny      = xtrc.shape[0] # number of ccd rows in trace
    assert(ny==npix_y)

    nfiber = xtrc.shape[1]

    minflux=1. # minimal flux in a row to include in the fit

    # Loop on fibers
    gauss = []
    start = 0
    for ii in xrange(nfiber):
        if (ii % 25 == 0): # & verbose:
            stop=time.time()
            if start==0 :
                log.info("Working on fiber {:d} of {:d}".format(ii,nfiber))
            else :
                log.info("Working on fiber %d of %d (25 done in %3.2f sec)"%(ii,nfiber,stop-start))
            start=stop

        # collect data
        central_xpix=np.floor(xtrc[:,ii]+0.5)
        begin_xpix=central_xpix-box_radius
        end_xpix=central_xpix+box_radius+1
        dx=[]
        flux=[]
        for y in xrange(ny) :
            yflux=flat[y,begin_xpix[y]:end_xpix[y]]
            syflux=np.sum(yflux)
            if syflux<minflux :
                continue
            dx.append(np.arange(begin_xpix[y],end_xpix[y])-(xtrc[y,ii]))
            flux.append(yflux/syflux)
        dx=np.array(dx)
        flux=np.array(flux)

        # compute profile
        # one way to get something robust is to compute median in bins
        # it's a bit biasing but the PSF is not a Gaussian anyway
        bins=np.linspace(-box_radius,box_radius,100)
        bstep=bins[1]-bins[0]
        bdx=[]
        bflux=[]
        for b in bins :
            ok=(dx>=b)&(dx<(b+bstep))
            if np.sum(ok)>0 :
                bdx.append(np.mean(dx[ok]))
                bflux.append(np.median(flux[ok]))
        if len(bdx)<10 :
            log.error("sigma fit failed for fiber #%02d"%ii)
            log.error("this should only occur for the fiber near the center of the detector (if at all)")
            log.error("using the sigma value from the previous fiber")
            gauss.append(gauss[-1])
            continue
        # this is the profile :
        bdx=np.array(bdx)
        bflux=np.array(bflux)

        # fast iterative gaussian fit
        sigma = 1.0
        sq2 = math.sqrt(2.)
        for i in xrange(10) :
            nsigma = sq2*np.sqrt(np.mean(bdx**2*bflux*np.exp(-bdx**2/2/sigma**2))/np.mean(bflux*np.exp(-bdx**2/2/sigma**2)))
            if abs(nsigma-sigma) < 0.001 :
                break
            sigma = nsigma
        gauss.append(sigma)

    return np.array(gauss)



def fiber_gauss_old(flat, xtrc, xerr, box_radius=2, max_iter=5, debug=False, verbose=False):
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
    start = 0
    for ii in xrange(nfiber):
        if (ii % 25 == 0): # & verbose:
            stop=time.time()
            if start==0 :
                log.info("Working on fiber {:d} of {:d}".format(ii,nfiber))
            else :
                log.info("Working on fiber %d of %d (done 25 in %3.2f sec)"%(ii,nfiber,stop-start))
            start=stop

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
            resid_mask = sigma_clip(resid, sigma=4., iters=5)
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
            pdb.set_trace()
        # Save
        gauss.append(parm.stddev.value)
    #
    return np.array(gauss)

def find_fiber_peaks(flat, ypos=None, nwidth=5, debug=False) :
    """Find the peaks of the fiber flat spectra
    Preforms book-keeping error checking

    Args:
        flat : ndarray of fiber flat image
        ypos : int [optional] Row for finding peaks
           Default is half-way up the image
        nwidth : int [optional] Width of peak (end-to-end)
        debug: bool, optional

    Returns:
        xpk, ypos, cut
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
    
    # JG : this algorithm is fragile 
    # 
    #gdp = cut > thresh
    # Roll to find peaks (simple algorithm)
    #nstep = nwidth // 2
    # 
    #for kk in xrange(-nstep,nstep):
    #    if kk < 0:
    #        test = np.roll(cut,kk) < np.roll(cut,kk+1)
    #    else:
    #        test = np.roll(cut,kk) > np.roll(cut,kk+1)
    #    # Compare
    #    gdp = gdp & test
    # xpk = np.where(gdp)[0]

    # Find clusters of adjacent points
    clusters=[]
    gdp=np.where(cut > thresh)[0]
    cluster=[gdp[0]]
    for i in gdp[1:] :
        if i==cluster[-1]+1 :
            cluster.append(i)
        else :
            clusters.append(cluster)
            cluster=[i]
    clusters.append(cluster)
    
    # Record max of each cluster
    xpk=np.zeros((len(clusters)))
    for i in xrange(len(clusters)) :
        t=np.argmax(cut[clusters[i]])
        xpk[i]=clusters[i][t]
    
    #print ("xpk.size=",xpk.size)
    #print ("xpk=",xpk)
    
    if debug:
        #pdb.xplot(cut, xtwo=xpk, ytwo=cut[xpk],mtwo='o')
        pdb.set_trace()

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
        log.warn('Wrong number of fibers in a bundle')
        #raise ValueError('Wrong number of fibers in a bundle')

    # Return
    return xpk, ypos, cut


def fit_traces(xset, xerr, func='legendre', order=6, sigrej=20.,
    RMS_TOLER=0.03, verbose=False):
    """Fit the traces
    Default is 6th order Legendre polynomials

    Parameters
    ----------
    xset : ndarray
      traces
    xerr : ndarray
      Error in the trace values (999.=Bad)
    RMS_TOLER : float, optional [0.02]
      Tolerance on size of RMS in fit

    Returns
    -------
    xnew, fits
    xnew : ndarray
      New fit values (without error)
    fits : list
      List of the fit dicts
    """
    log=get_logger()
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
        if nmask_new > 200:
            log.error("Rejected many points ({:d}) in fiber {:d}".format(nmask_new, ii))
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
            #from xastropy.xutils import xdebug as xdb
            #xdb.xplot(yval, xnew[:,ii], xtwo=yval[gdval],ytwo=xset[:,ii][gdval], mtwo='o')
            log.error("RMS {:g} exceeded tolerance for fiber {:d}".format(rms, ii))
    # Return
    return xnew, fits


def extract_sngfibers_gaussianpsf(img, img_ivar, xtrc, sigma, box_radius=2, verbose=True):
    """Extract spectrum for fibers one-by-one using a Gaussian PSF

    Parameters
    ----------
    img : ndarray
      Image
    img_ivar : ndarray
      Image inverse variance
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

    log = get_logger()

    #
    all_spec = np.zeros_like(xtrc)
    cst = 1./np.sqrt(2*np.pi)
    start=0
    for qq in range(xtrc.shape[1]):
        if verbose & (qq % 25 == 0):
            stop=time.time()
            if start>0 :
                log.info("Working on fiber %d of %d (done 25 in %3.2f sec)"%(qq,xtrc.shape[1],stop-start))
            else :
                log.info("Working on fiber %d of %d"%(qq,xtrc.shape[1]))
            start=stop

        # Mask
        mask[:,:] = 0
        ixt = np.round(xtrc[:,qq]).astype(int)
        for jj,ibox in enumerate(range(-box_radius,box_radius+1)):
            ix = ixt + ibox
            mask[iy,ix] = 1
        # Sub-image (for speed, not convenience)
        gdp = np.where(mask == 1)
        minx = np.min(gdp[1])
        maxx = np.max(gdp[1])
        nx = (maxx-minx)+1

        # Generate PSF
        dx_img = xpix_img[:,minx:maxx+1] - np.outer(xtrc[:,qq], np.ones(nx))
        psf = cst*np.exp(-0.5 * (dx_img/sigma[qq])**2)/sigma[qq]
        #dx_img = xpix_img[:,minx:maxx+1] - np.outer(xtrc[:,qq],np.ones(img.shape[1]))
        #g_init = models.Gaussian1D(amplitude=1., mean=0., stddev=sigma[qq])
        #psf = mask * g_init(dx_img)
        # Extract
        #all_spec[:,qq] = np.sum(psf*img,axis=1) / np.sum(psf,axis=1)
        #all_spec[:,qq] = np.sum(psf*img[:,minx:maxx+1],axis=1) / np.sum(psf,axis=1)
        a=np.sum(img_ivar[:,minx:maxx+1]*psf**2,axis=1)
        ok=(a>0)
        all_spec[ok,qq] = np.sum(img_ivar[ok,minx:maxx+1]*psf[ok]*img[ok,minx:maxx+1],axis=1) / a[ok]

    # Return
    return all_spec


def trace_crude_init(image, xinit0, ypass, invvar=None, radius=2.,
    maxshift0=0.5, maxshift=0.15, maxerr=0.2):
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
        xerr[iy,:] = xfiterr * (xfiterr < maxerr) + 999.0 * (xfiterr >= maxerr)

    return xset, xerr


def trace_fweight(fimage, xinit, ycen=None, invvar=None, radius=2., debug=False):
    '''Python port of trace_fweight.pro from IDLUTILS

    Parameters
    ----------
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

    if debug:
        pdb.set_trace()

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

def write_psf(outfile, xfit, fdicts, gauss, wv_solns, legendre_deg=5, without_arc=False,
              XCOEFF=None):
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

    # check legendre degree makes sense based on number of lines
    if not without_arc:
        nlines=10000
        for ii,id_dict in enumerate(wv_solns):
            nlines_in_fiber=(np.array(id_dict['id_pix'])[id_dict['mask']==0]).size
            #print("fiber #%d nlines=%d"%(ii,nlines_in_fiber))
            nlines=min(nlines,nlines_in_fiber)
        if nlines < legendre_deg+2 :
            legendre_deg=nlines-2
            print("reducing legendre degree to %d because the min. number of emission lines found is %d"%(legendre_deg,nlines))

    ny = xfit.shape[0]
    nfiber = xfit.shape[1]
    ncoeff=legendre_deg+1
    if XCOEFF is None:
        XCOEFF = np.zeros((nfiber, ncoeff))
    YCOEFF = np.zeros((nfiber, ncoeff))

    # Find WAVEMIN, WAVEMAX
    if without_arc:
        WAVEMIN = 0.
        WAVEMAX = ny-1.
        wv_solns = [None]*nfiber
    else:
        WAVEMIN = np.min([id_dict['wave_min'] for id_dict in wv_solns]) - 1.
        WAVEMAX = np.max([id_dict['wave_max'] for id_dict in wv_solns]) + 1.
    wv_array = np.linspace(WAVEMIN, WAVEMAX, num=ny)
    # Fit Legendre to y vs. wave
    for ii,id_dict in enumerate(wv_solns):
        # Fit y vs. wave
        if without_arc:
            yleg_fit, mask = dufits.iter_fit(wv_array, np.arange(ny), 'legendre', ncoeff-1, xmin=WAVEMIN, xmax=WAVEMAX, niter=1)
        else:
            yleg_fit, mask = dufits.iter_fit(np.array(id_dict['id_wave'])[id_dict['mask']==0], np.array(id_dict['id_pix'])[id_dict['mask']==0], 'legendre', ncoeff-1, xmin=WAVEMIN, xmax=WAVEMAX, sig_rej=100000.)
        YCOEFF[ii, :] = yleg_fit['coeff']
        # Fit x vs. wave
        yval = dufits.func_val(wv_array, yleg_fit)
        if fdicts is None:
            if XCOEFF is None:
                raise IOError("Need to set either fdicts or XCOEFF!")
        else:
            xtrc = dufits.func_val(yval, fdicts[ii])
            xleg_fit,mask = dufits.iter_fit(wv_array, xtrc, 'legendre', ncoeff-1, xmin=WAVEMIN, xmax=WAVEMAX, niter=5, sig_rej=100000.)
            XCOEFF[ii, :] = xleg_fit['coeff']

    # Write the FITS file
    prihdu = fits.PrimaryHDU(XCOEFF)
    prihdu.header['WAVEMIN'] = WAVEMIN
    prihdu.header['WAVEMAX'] = WAVEMAX

    yhdu = fits.ImageHDU(YCOEFF)

    # also save wavemin wavemax in yhdu
    yhdu.header['WAVEMIN'] = WAVEMIN
    yhdu.header['WAVEMAX'] = WAVEMAX

    gausshdu = fits.ImageHDU(np.array(gauss))

    hdulist = fits.HDUList([prihdu, yhdu, gausshdu])
    hdulist.writeto(outfile, clobber=True)



#####################################################################
#####################################################################
# Utilities
#####################################################################

def script_bootcalib(arc_idx, flat_idx, cameras=None, channels=None, nproc=10):
    """ Runs desi_bootcalib on a series of pix files

    Returns:
        script_bootcalib([0,1,2,3,4,5,6,7,8,9], [10,11,12,13,14])

    """
    from subprocess import Popen
    #
    if cameras is None:
        cameras = ['0','1','2','3','4','5','6','7','8','9']
    if channels is None:
        channels = ['b','r','z']
        #channels = ['b']#,'r','z']
    nchannels = len(channels)
    ncameras = len(cameras)
    #
    narc = len(arc_idx)
    nflat = len(flat_idx)
    ntrial = narc*nflat*ncameras*nchannels

    # Loop on the systems
    nrun = -1
    #nrun = 123
    while(nrun < ntrial):

        proc = []
        ofiles = []
        for ss in range(nproc):
            nrun += 1
            iarc = nrun % narc
            jflat = (nrun//narc) % nflat
            kcamera = (nrun//(narc*nflat)) % ncameras
            lchannel = nrun // (narc*nflat*ncameras)
            #pdb.set_trace()
            if nrun == ntrial:
                break
            # Names
            afile = str('pix-{:s}{:s}-{:08d}.fits'.format(channels[lchannel], cameras[kcamera], arc_idx[iarc]))
            ffile = str('pix-{:s}{:s}-{:08d}.fits'.format(channels[lchannel], cameras[kcamera], flat_idx[jflat]))
            ofile = str('boot_psf-{:s}{:s}-{:d}{:d}.fits'.format(channels[lchannel], cameras[kcamera],
                                                                 arc_idx[iarc], flat_idx[jflat]))
            qfile = str('qa_boot-{:s}{:s}-{:d}{:d}.pdf'.format(channels[lchannel], cameras[kcamera],
                                                                 arc_idx[iarc], flat_idx[jflat]))
            lfile = str('boot-{:s}{:s}-{:d}{:d}.log'.format(channels[lchannel], cameras[kcamera],
                                                               arc_idx[iarc], flat_idx[jflat]))
            ## Run
            script = [str('desi_bootcalib.py'), str('--fiberflat={:s}'.format(ffile)),
                      str('--arcfile={:s}'.format(afile)),
                      str('--outfile={:s}'.format(ofile)),
                      str('--qafile={:s}'.format(qfile))]#,
                      #str('>'),
                      #str('{:s}'.format(lfile))]
            f = open(lfile, "w")
            proc.append(Popen(script, stdout=f))
            ofiles.append(f)
        exit_codes = [p.wait() for p in proc]
        for ofile in ofiles:
            ofile.close()


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


def qa_fiber_trace(flat, xtrc, outfil=None, Nfiber=25, isclmin=0.5):
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
        mplt = plt.imshow(sub_flat,origin='lower', cmap=cmm,
            extent=(0., sub_flat.shape[1]-1, x0,x1-1), aspect='auto')
            #extent=(0., sub_flat.shape[1]-1, x0,x1))
        #mplt.set_clim(vmin=sclmin, vmax=sclmax)

        # Axes
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
