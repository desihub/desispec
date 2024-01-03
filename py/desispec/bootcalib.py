"""
desispec.bootcalib
==================

Utility functions to perform a quick calibration of DESI data

TODO:
1. Expand to r, i cameras
2. QA plots
3. Test with CR data
"""
from __future__ import print_function, absolute_import, division

import numpy as np
import copy
import pdb
import yaml
import glob
import math
import time
import os
import sys
import argparse
import locale
from importlib import resources

from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
from astropy.table import Table, Column, vstack
from astropy.io import fits

#- support astropy 2.x sigma_clip syntax with `iters` instead of `maxiters`
import astropy
if astropy.__version__.startswith('2.'):
    astropy_sigma_clip = sigma_clip
    def sigma_clip(data, sigma=None, maxiters=5):
        return astropy_sigma_clip(data, sigma=sigma, iters=maxiters)

from desispec.util import set_backend
set_backend()

from matplotlib import pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages

from desiutil.log import get_logger
from desiutil import funcfits as dufits
from numpy.polynomial.legendre import legval

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
    flat[flat<-20]=-20.
    ny=flat.shape[0]

    xpk,ypos,cut=find_fiber_peaks(flat)
    xset,xerr=trace_crude_init(flat,xpk,ypos)
    xfit,fdicts=fit_traces(xset,xerr)
    gauss=fiber_gauss(flat,xfit,xerr)

    #- Also need wavelength solution not just trace

    arc=arcimage.pix
    arc[arc<-20]=-20.
    arc_ivar=arcimage.ivar*(arcimage.mask==0)
    all_spec=extract_sngfibers_gaussianpsf(arc,arc_ivar,xfit,gauss)
    llist=load_arcline_list(camera)
    ### dlamb,wmark,gd_lines,line_guess=load_gdarc_lines(camera)
    dlamb, gd_lines = load_gdarc_lines(camera, llist)

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
    spec_mask = sigma_clip(spec, sigma=4., maxiters=5)
    rms = np.std(spec_mask)
    thresh = rms*rms_thresh
    #print("thresh = {:g}".format(thresh))
    gdp = spec > thresh

    # Avoid edges
    gdp = gdp & (np.arange(npix) > 2.*nwidth) & (np.arange(npix) < (npix-2.*nwidth))

    # Roll to find peaks (simple algorithm)
    # nwidth = 5
    nstep = max(1,nwidth // 2)
    for kk in range(-nstep,nstep):
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
    flux = np.zeros(ngd)

    for jj,igdpix in enumerate(gdpix):
        # Simple flux-weight
        pix = np.arange(igdpix-nstep,igdpix+nstep+1,dtype=int)
        flux[jj] =  np.sum(spec[pix])
        xpk[jj] = np.sum(pix*spec[pix]) / flux[jj]

    # Finish
    return xpk , flux



def remove_duplicates_w_id(wy,w,y_id,w_id) :
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
            y_id2.append(ii[i])
    y_id2=np.array(y_id2).astype(int)
    w_id2=np.array(w_id2).astype(int)
    tmp=np.argsort(w[w_id2])
    y_id2=y_id2[tmp]
    w_id2=w_id2[tmp]
    return y_id2,w_id2

def remove_duplicates_y_id(yw,y,y_id,w_id) :
    # might be several identical y_id
    w_id=np.array(w_id).astype(int)
    y_id=np.array(y_id).astype(int)
    w_id2=[]
    y_id2=[]
    for j in np.unique(y_id) :
        y_id2.append(j)
        ii=w_id[y_id==j]
        if ii.size==1 :
            w_id2.append(ii[0])
        else :
            i=np.argmin(np.abs(yw[ii]-y[j]))
            w_id2.append(ii[i])
    w_id2=np.array(w_id2).astype(int)
    y_id2=np.array(y_id2).astype(int)
    tmp=np.argsort(y[y_id2])
    w_id2=w_id2[tmp]
    y_id2=y_id2[tmp]
    return y_id2,w_id2


def refine_solution(y,w,y_id,w_id,deg=3,tolerance=5.) :

    log = get_logger()

    # remove duplicates
    transfo=np.poly1d(np.polyfit(y[y_id],w[w_id],deg=deg))
    wy=transfo(y)
    y_id,w_id=remove_duplicates_w_id(wy,w,y_id,w_id)
    transfo=np.poly1d(np.polyfit(w[w_id],y[y_id],deg=deg))
    yw=transfo(w)
    y_id,w_id=remove_duplicates_y_id(yw,y,y_id,w_id)

    if len(y_id) != len(np.unique(y_id)) :
        log.error("duplicate AT INIT y_id={:s}".format(str(y_id)))
    if len(w_id) != len(np.unique(w_id)) :
        log.error("duplicate AT INIT  w_id={:s}".format(str(w_id)))

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
        mdiff0=min(tolerance,max(2.,rms*2.)) # this is a difficult parameter to tune, either loose lever arm, or have false matches !!
        mdiff1=tolerance # this is a difficult parameter to tune, either loose lever arm, or have false matches !!
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
                    break

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

def id_remainder(id_dict, llist, deg=4, tolerance=1., verbose=False) :

    log = get_logger()

    y_id=np.array(id_dict['id_idx']).astype(int)
    all_y=np.array(id_dict['pixpk'])

    all_known_waves  = np.sort(np.array(llist["wave"]))
    identified_waves = np.array(id_dict["id_wave"]) # lines identified at previous steps

    w_id=[]
    for w in identified_waves :
        i=np.argmin(np.abs(all_known_waves-w))
        diff=np.abs(all_known_waves[i]-w)
        if diff>0.1 :
            log.warning("discrepant wavelength".format(w,all_known_waves[i]))
        w_id.append(i)
    w_id = np.array(w_id).astype(int)
    y_id,w_id,rms,niter=refine_solution(all_y,all_known_waves,y_id,w_id,deg=deg,tolerance=tolerance)

    id_dict['id_idx']  = np.sort(y_id)
    id_dict['id_pix']  = np.sort(all_y[y_id])
    id_dict['id_wave'] = np.sort(all_known_waves[w_id])
    id_dict['rms'] = rms

    log.info("{:d} matched for {:d} detected and {:d} known, rms = {:g}".format(len(y_id),len(all_y),len(all_known_waves),rms))


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

def id_arc_lines_using_triplets(id_dict,w,dwdy_prior,d2wdy2_prior=1.5e-5,toler=0.2,ntrack=50,nmax=40):
    """Match (as best possible), a set of the input list of expected arc lines to the detected list

    Parameters
    ----------
    id_dict : dictionnary with Pixel locations of detected arc lines in "pixpk" and fluxes in "flux"
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


    y = id_dict["pixpk"]

    log.info("ny=%d nw=%d"%(len(y),len(w)))

    if nmax<10 :
        nmax=10
        log.warning("force nmax=10 (arg was too small: {:d})".format(nmax))

    if len(y)>nmax :
        # log.info("down-selecting the number of detected lines from {:d} to {:d}".format(len(y),nmax))
        # keep at least the edges
        margin=3
        new_y=np.append(y[:margin],y[-margin:])
        # now look at the flux to select the other ones
        flux=id_dict["flux"][margin:-margin]
        ii=np.argsort(flux)
        new_y=np.append(new_y,y[margin:-margin][ii[-(nmax-2*margin):]])
        y = np.sort(new_y)

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
        iw=int(w_triplet[3])
        for a,b,c in zip(dwdy_bin[pairs_in_histo],d2wdy2_bin[pairs_in_histo],y_triplets[pairs_in_histo,3].astype(int)) :
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

        if  histogram_ravel[histo_bin]<4 and count>3 :
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
            #log.info("new best solution #%d with %d match and rms=%f (niter=%d)"%(count,len(y_id),rms,niter))
            #log.info("previous had %d match and rms=%f"%(len(best_y_id),best_rms))
            best_y_id = y_id
            best_w_id = w_id
            best_rms = rms

        # stop at some moment
        if best_rms<0.2 and len(y_id)>=min(15,min(len(y),len(w))) :
            #log.info("stop here because we have a correct solution")
            break

    if len(y) != len(id_dict["pixpk"]) :
        #log.info("re-indexing the result")
        tmp_y_id = []
        for i in best_y_id :
            tmp_y_id.append(np.argmin(np.abs(id_dict["pixpk"]-y[i])))
        best_y_id = np.array(tmp_y_id).astype(int)
        y = id_dict["pixpk"]

    if len(best_w_id) == 0 :
        log.error("failed, no match")
        id_dict["status"]="failed"
        id_dict["id_idx"]=[]
        id_dict["id_pix"]=[]
        id_dict["id_wave"]=[]
        id_dict["rms"]=999.
        id_dict["fit"]=None
        return

    id_dict["status"]="ok"
    id_dict["id_idx"]=best_y_id
    id_dict["id_pix"]=y[best_y_id]
    id_dict["id_wave"]=w[best_w_id]
    id_dict["rms"]=best_rms
    deg=max(1,min(3,best_y_id.size-2))
    id_dict["fit"]= dufits.func_fit(w[best_w_id],y[best_y_id],'polynomial',deg,xmin=0.,xmax=1.)

    log.info("{:d} matched for {:d} detected and {:d} known as good, rms = {:g}".format(len(best_y_id),len(y),len(w),best_rms))



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
        Name of ion.
    vacuum : bool, optional
        Use vacuum wavelengths.

    Returns
    -------
    :class:`astropy.table.Table`
        A Table obtained from the data file with some columns added or renamed.
    """
    log=get_logger()
    # Find file
    medium = 'vacuum'
    if not vacuum:
        log.info("Using air wavelengths")
        medium = 'air'
    srch_file = "data/arc_lines/{0}_{1}.ascii".format(ion, medium)
    if not resources.files('desispec').joinpath(srch_file).is_file():
        log.error("Cannot find NIST file {:s}".format(srch_file))
        raise Exception("Cannot find NIST file {:s}".format(srch_file))
    # Read, while working around non-ASCII characters in NIST line lists
    nist_file = str(resources.files('desispec').joinpath(srch_file))
    log.info("reading NIST file {:s}".format(nist_file))
    # The data files contain the non-ASCII character 'Å', so explicitly set the
    # encoding when reading the table.
    #
    # cupy is known to unexpectedly alter the default encoding,
    # so we need this for both of those reasons.
    nist_tbl = Table.read(nist_file, format='ascii.fixed_width', encoding='UTF-8')
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
    nist_tbl.add_column(Column([ion]*len(nist_tbl), name='Ion', dtype=(str, 5)))
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
            lamps = ['CdI','ArI','HgI','NeI','KrI','XeI']
        elif camera == 'all': # Used for specex
            lamps = ['CdI','ArI','HgI','NeI','KrI','XeI']
        else:
            log.error("Not ready for this camera")

    # Get the parse dict
    parse_dict = load_parse_dict()
    # Read rejection file
    medium = 'vacuum'
    if not vacuum:
        log.info("Using air wavelengths")
        medium = 'air'
    rej_file = resources.files('desispec').joinpath(f"data/arc_lines/rejected_lines_{medium}.yaml")
    with open(rej_file, 'r') as infile:
        rej_dict = yaml.safe_load(infile)
    # Loop through the NIST Tables
    tbls = []
    for iline in lamps:
        # Load
        tbl = parse_nist(iline, vacuum=vacuum)
        # Parse
        if iline in parse_dict:
            tbl = parse_nist_tbl(tbl,parse_dict[iline])
        # Reject
        if iline in rej_dict:
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
    for wave in rej_dict:
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


def load_gdarc_lines(camera, llist, vacuum=True,lamps=None,good_lines_filename=None):

    """Loads a select set of arc lines for initial calibrating

    Parameters
    ----------
    camera : str
      Camera ('b', 'g', 'r')
    llist : table of lines to use, with columns Ion, wave
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

    lines={}

    dlamb=0.6
    if camera[0] == 'b':
        dlamb = 0.589
    elif camera[0] == 'r':
        dlamb = 0.527
    elif camera[0] == 'z':
        #dlamb = 0.599  # Ang
        dlamb = 0.608  # Ang (from teststand, ranges (fiber & wave) from 0.54 to 0.66)
    # read good lines
    if good_lines_filename is not None :
        filename = good_lines_filename
    else :
        if vacuum :
            filename = str(resources.files('desispec').joinpath("data/arc_lines/goodlines_vacuum.ascii"))
        else :
            filename = str(resources.files('desispec').joinpath("data/arc_lines/goodlines_air.ascii"))

    log.info("Reading good lines in {:s}".format(filename))
    lines={}
    ifile=open(filename)
    for line in ifile.readlines() :
        if line[0]=="#" :
            continue
        vals=line.strip().split()
        if len(vals)<3 :
            log.warning("ignoring line '{:s}' in {:s}".format(line.strip(),filename))
            continue
        cameras=vals[2]
        if cameras.find(camera[0].upper()) < 0 :
            continue
        ion=vals[1]
        wave=float(vals[0])
        if ion in lines:
            lines[ion].append(wave)
        else :
            lines[ion]=[wave,]
    ifile.close()
    log.info("Good lines = {:s}".format(str(lines)))

    log.info("Checking consistency with full line list")
    nbad=0
    for ion in lines:
        ii=np.where(llist["Ion"]==ion)[0]
        if ii.size == 0 :
            continue
        all_waves=np.array(llist["wave"][ii])
        for j,w in enumerate(lines[ion]) :
            i=np.argmin(np.abs(w-all_waves))
            if np.abs(w-all_waves[i])>0.2 :
                log.error("cannot find good line {:f} of {:s} in full line list. nearest is {:f}".format(w,ion,all_waves[i]))
                nbad += 1
            elif np.abs(w-all_waves[i])>0.001 :
                log.warning("adjusting hardcoded {:s} line {:f} -> {:f} (the NIST line list is the truth)".format(w,ion,all_waves[i]))
                lines[ion][j]=all_waves[i]
    if nbad>0 :
        log.error("{:d} inconsistent hardcoded lines, exiting".format(nbad))
        sys.exit(12)

    gd_lines=np.array([])
    for lamp in lamps :
        if lamp in lines:
            gd_lines=np.append(gd_lines,lines[lamp])

    # Sort and return
    gd_lines.sort()
    return dlamb, gd_lines

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
    for ii in range(nfiber):
        if (ii % 25 == 0): # & verbose:
            stop=time.time()
            if start==0 :
                log.info("Working on fiber {:d} of {:d}".format(ii,nfiber))
            else :
                log.info("Working on fiber %d of %d (25 done in %3.2f sec)"%(ii,nfiber,stop-start))
            start=stop

        # collect data
        central_xpix=np.floor(xtrc[:,ii]+0.5)
        begin_xpix=(central_xpix-box_radius).astype(int)
        end_xpix=(central_xpix+box_radius+1).astype(int)
        dx=[]
        flux=[]
        for y in range(ny) :
            yflux=np.zeros(2*box_radius+1)
            tmp=flat[y,begin_xpix[y]:end_xpix[y]]
            yflux[:tmp.size] = tmp
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
            if np.sum(ok)>1 :
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
        for i in range(10) :
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
    log.warning("fiber_gauss uses astropy.modeling.  Consider an alternative")
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
    for ii in range(nfiber):
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
            try :
                mask[iy,ix] = 1
            except IndexError :
                pass
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
            resid_mask = sigma_clip(resid, sigma=4., maxiters=5)
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

def find_fiber_peaks(flat, ypos=None, nwidth=5, debug=False,thresh=None) :
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
    cutimg = flat[ypos-50:ypos+50, :]

    # Smash
    cut = np.median(cutimg, axis=0)

    # Set flux threshold
    #srt = np.sort(cutimg.flatten()) # this does not work for sparse fibers
    #thresh = srt[int(cutimg.size*0.95)] / 2. # this does not work for sparse fibers

    if thresh is None :
        thresh = np.max(cut)/20.
        log.info("Threshold: {:f}".format(thresh))
        pixels_below_threshold=np.where(cut<thresh)[0]
        if pixels_below_threshold.size>2 :
            values_below_threshold = sigma_clip(cut[pixels_below_threshold],sigma=3,maxiters=200)
            if values_below_threshold.size>2 :
                rms=np.std(values_below_threshold)
                nsig=7
                new_thresh=max(thresh,nsig*rms)
                log.info("Threshold: {:f} -> {:f} ({:d}*rms: {:f})".format(thresh,new_thresh,nsig,nsig*rms))
                thresh=new_thresh
    else :
        log.info("Using input threshold: {:f})".format(thresh))
    #gdp = cut > thresh
    # Roll to find peaks (simple algorithm)
    #nstep = nwidth // 2
    #for kk in range(-nstep,nstep):
    #    if kk < 0:
    #        test = np.roll(cut,kk) < np.roll(cut,kk+1)
    #    else:
    #        test = np.roll(cut,kk) > np.roll(cut,kk+1)
    #    # Compare
    #    gdp = gdp & test
    #xpk = np.where(gdp)[0]

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


    log.info("Number of clusters found: {:d}".format(len(clusters)))

    # Record max of each cluster
    xpk=np.zeros((len(clusters)), dtype=np.int64)
    for i in range(len(clusters)) :
        t=np.argmax(cut[clusters[i]])
        xpk[i]=clusters[i][t]

    if debug:
        #pdb.xplot(cut, xtwo=xpk, ytwo=cut[xpk],mtwo='o')
        pdb.set_trace()

    # Book-keeping and some error checking
    if len(xpk) != Nbundle*Nfiber:
        log.warning('Found the wrong number of total fibers: {:d}'.format(len(xpk)))
    else:
        log.info('Found {:d} fibers'.format(len(xpk)))
    # Find bundles
    xsep = np.roll(xpk,-1) - xpk
    medsep = np.median(xsep)
    bundle_ends = np.where(np.abs(xsep-medsep) > 0.5*medsep)[0]
    if len(bundle_ends) != Nbundle:
        log.warning('Found the wrong number of bundles: {:d}'.format(len(bundle_ends)))
    else:
        log.info('Found {:d} bundles'.format(len(bundle_ends)))
    # Confirm correct number of fibers per bundle
    bad = ((bundle_ends+1) % Nfiber) != 0
    if np.sum(bad) > 0:
        log.warning('Wrong number of fibers in a bundle')
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
    for ii in range(ntrace):
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
            try :
                mask[iy,ix] = 1
            except IndexError :
                pass

        # Sub-image (for speed, not convenience)
        gdp = np.where(mask == 1)
        if len(gdp[1])<2: continue
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
        b=np.sum(img_ivar[:,minx:maxx+1]*psf*img[:,minx:maxx+1],axis=1)
        ok=(a>1.e-6)
        all_spec[ok,qq] = b[ok] / a[ok]

        #import astropy.io.fits as pyfits
        #h=pyfits.HDUList([pyfits.PrimaryHDU(),
        #                  pyfits.ImageHDU(img[:,minx:maxx+1],name="FLUX"),
        #                  pyfits.ImageHDU(img_ivar[:,minx:maxx+1],name="IVAR"),
        #                  pyfits.ImageHDU(psf,name="PSF"),
        #                  pyfits.ImageHDU(a,name="A"),
        #                  pyfits.ImageHDU(b,name="B")])
        #h.writeto("test.fits")
        #sys.exit(12)





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


def fix_ycoeff_outliers(xcoeff, ycoeff, deg=5, tolerance=2):
    '''
    Fix outliers in coefficients for wavelength solution, assuming a continuous function of CCD coordinates

    Args:
        xcoeff[nfiber, ncoeff] : 2D array of Legendre coefficients for X(wavelength)
        ycoeff[nfiber, ncoeff] : 2D array of Legendre coefficients for Y(wavelength)

    Options:
        deg : integer degree of polynomial to fit
        tolerance : replace fibers with difference of wavelength solution larger than this number of pixels after interpolation

    Returns:
        new_ycoeff[nfiber, ncoeff] with outliers replaced by interpolations

    For each coefficient, fit a polynomial vs. fiber number with one
    pass of sigma clipping.  Remaining outliers are than replaced with
    the interpolated fit value.
    '''

    log = get_logger()

    nfibers=ycoeff.shape[0]
    if nfibers < 3 :
        log.warning("only {:d} fibers, cannot interpolate coefs".format(nfibers))
        return ycoeff
    deg=min(deg,nfibers-1)

    nwave=ycoeff.shape[1]+1
    wave_nodes = np.linspace(-1,1,nwave)

    # get traces using fit coefs
    x=np.zeros((nfibers,nwave))
    y=np.zeros((nfibers,nwave))

    for i in range(nfibers) :
        x[i] = legval(wave_nodes,xcoeff[i])
        y[i] = legval(wave_nodes,ycoeff[i])

    new_ycoeff=ycoeff.copy()

    bad_fibers=None
    while True : # loop to discard one fiber at a time

        # polynomial fit as a function of x for each wave
        yf=np.zeros((nfibers,nwave))
        xx=2*(x - np.min(x)) / (np.max(x) - np.min(x)) - 1
        for i in range(nwave) :
            c=np.polyfit(xx[:,i], y[:,i], deg)
            yf[:,i]=np.polyval(c, xx[:,i])

        diff=np.max(np.abs(y-yf),axis=1)

        for f in range(nfibers) :
            log.info("fiber {:d} maxdiff= {:f}".format(f,diff[f]))


        worst = np.argmax(diff)
        if diff[worst] > tolerance :
            log.warning("replace fiber {:d} trace by interpolation".format(worst))
            leg_fit = dufits.func_fit(wave_nodes, yf[worst], 'legendre', ycoeff.shape[1]-1, xmin=-1, xmax=1)
            new_ycoeff[worst] = leg_fit['coeff']
            y[worst] = legval(wave_nodes,new_ycoeff[worst])
            if bad_fibers is None :
                bad_fibers = np.array([worst])
            else :
                bad_fibers=np.append(bad_fibers, worst)
                bad_fibers=np.unique(bad_fibers)
            continue
        break

    return new_ycoeff


#####################################################################
#####################################################################
# Output
#####################################################################

def write_psf(outfile, xfit, fdicts, gauss, wv_solns, legendre_deg=5, without_arc=False,
              XCOEFF=None, fiberflat_header=None, arc_header=None, fix_ycoeff=True):
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
            if len(id_dict['id_pix']) > 0 :
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
        WAVEMIN = 10000000.
        WAVEMAX = 0.
        for id_dict in wv_solns :
            if 'wave_min' in id_dict :
                WAVEMIN = min(WAVEMIN,id_dict['wave_min'])
            if 'wave_max' in id_dict :
                WAVEMAX = max(WAVEMAX,id_dict['wave_max'])
        WAVEMIN -= 1.
        WAVEMAX += 1.

    wv_array = np.linspace(WAVEMIN, WAVEMAX, num=ny)
    # Fit Legendre to y vs. wave
    for ii,id_dict in enumerate(wv_solns):



        # Fit y vs. wave
        if without_arc:
            yleg_fit, mask = dufits.iter_fit(wv_array, np.arange(ny), 'legendre', ncoeff-1, xmin=WAVEMIN, xmax=WAVEMAX, niter=1)
        else:
            if len(id_dict['id_wave']) > 0 :
                yleg_fit, mask = dufits.iter_fit(np.array(id_dict['id_wave'])[id_dict['mask']==0], np.array(id_dict['id_pix'])[id_dict['mask']==0], 'legendre', ncoeff-1, xmin=WAVEMIN, xmax=WAVEMAX, sig_rej=100000.)
            else :
                yleg_fit = None
                mask = None

        if yleg_fit is None :
            continue

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

    # Fix outliers assuming that coefficients vary smoothly vs. CCD coordinates
    if fix_ycoeff :
        YCOEFF = fix_ycoeff_outliers(XCOEFF,YCOEFF,tolerance=2)

    # Write the FITS file
    prihdu = fits.PrimaryHDU(XCOEFF)
    prihdu.header['WAVEMIN'] = WAVEMIN
    prihdu.header['WAVEMAX'] = WAVEMAX
    prihdu.header['EXTNAME'] = 'XTRACE'
    prihdu.header['PSFTYPE'] = 'bootcalib'

    from desiutil.depend import add_dependencies
    add_dependencies(prihdu.header)

    # Add informations for headers
    if arc_header is not None :
        if "NIGHT" in arc_header:
            prihdu.header["ARCNIGHT"] = arc_header["NIGHT"]
        if "EXPID" in arc_header:
            prihdu.header["ARCEXPID"] = arc_header["EXPID"]
        if "CAMERA" in arc_header:
            prihdu.header["CAMERA"] = arc_header["CAMERA"]
        prihdu.header['NPIX_X'] = arc_header['NAXIS1']
        prihdu.header['NPIX_Y'] = arc_header['NAXIS2']
    if fiberflat_header is not None :
        if 'NPIX_X' not in prihdu.header:
            prihdu.header['NPIX_X'] = fiberflat_header['NAXIS1']
            prihdu.header['NPIX_Y'] = fiberflat_header['NAXIS2']
        if "NIGHT" in fiberflat_header:
            prihdu.header["FLANIGHT"] = fiberflat_header["NIGHT"]
        if "EXPID" in fiberflat_header:
            prihdu.header["FLAEXPID"] = fiberflat_header["EXPID"]

    yhdu = fits.ImageHDU(YCOEFF, name='YTRACE')

    # also save wavemin wavemax in yhdu
    yhdu.header['WAVEMIN'] = WAVEMIN
    yhdu.header['WAVEMAX'] = WAVEMAX

    gausshdu = fits.ImageHDU(np.array(gauss), name='XSIGMA')

    hdulist = fits.HDUList([prihdu, yhdu, gausshdu])


    hdulist.writeto(outfile, overwrite=True)

def write_line_list(filename,all_wv_soln,llist) :
    wave = np.array([])
    for id_dict in all_wv_soln :
        wave=np.append(wave,id_dict["id_wave"])
    wave=np.unique(wave)

    ofile=open(filename,"w")
    ofile.write("# from bootcalib\n")
    ofile.write("Ion wave score RelInt\n")
    for w in wave :
        ii=np.argmin(np.abs(llist["wave"]-w))
        print(w,llist["wave"][ii],llist["Ion"][ii])
        ofile.write("{:s} {:f} 1 1\n".format(llist["Ion"][ii],w))
    ofile.close()





#####################################################################
#####################################################################
# Utilities
#####################################################################

def script_bootcalib(arc_idx, flat_idx, cameras=None, channels=None, nproc=10):
    """ Runs desi_bootcalib on a series of preproc files

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
            #- TODO: update to use desispec.io.findfile instead
            afile = str('preproc-{:s}{:s}-{:08d}.fits'.format(channels[lchannel], cameras[kcamera], arc_idx[iarc]))
            ffile = str('preproc-{:s}{:s}-{:08d}.fits'.format(channels[lchannel], cameras[kcamera], flat_idx[jflat]))
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
        for ii in range(i0,i1):
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
