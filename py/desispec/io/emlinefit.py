#!/usr/bin/env python

"""
desispec.io.emlinefit
====================
IO routines for desi_emlinefit_afterburner.
"""

import os
import sys
from astropy.io import fits
import numpy as np
from scipy.optimize import curve_fit
from desitarget.geomask import match_to
from desiutil.dust import ext_odonnell, ext_fitzpatrick
from desiutil.dust import ebv as dust_ebv
from desiutil.log import get_logger
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import gridspec

allowed_emnames = ["OII", "HDELTA", "HGAMMA", "HBETA", "OIII", "HALPHA"]

def get_rf_em_waves(emname):
    """
    Returns the rest-frame, vacuum, wavelengths.
    Args:
        emname, from: "OII", "HDELTA", "HGAMMA", "HBETA", "OIII", "HALPHA" (string)

    Returns:
        rf_em_waves: rest-frame wavelength(s) (np array)

    Notes:
        For OII and OIII returns a two-elements array; one-element array otherwise.
        https://github.com/desihub/fastspecfit/blob/60393296e0cc466858f70a5d021d02693eff9375/py/fastspecfit/data/emlines.ecsv
    """
    if emname == "OII":
        rf_em_waves = np.array([3727.092, 3729.874])
    if emname == "OIII":
        rf_em_waves = np.array([4960.295, 5008.239])
    if emname == "HALPHA":
        rf_em_waves = np.array([6564.613])
    if emname == "HBETA":
        rf_em_waves = np.array([4862.683])
    if emname == "HGAMMA":
        rf_em_waves = np.array([4341.684])
    if emname == "HDELTA":
        rf_em_waves = np.array([4102.892])
    return rf_em_waves


def emlines_gaussfit(
    emname,
    targetid,
    zspec,
    waves,
    fluxes,
    ivars,
    rf_fit_hw=40,
    min_rf_fit_hw=20,
    rf_cont_w=200,
    p0_sigma=2.5,
    p0_flux=10,
    p0_share=0.58,
    min_sigma=1e-5,
    max_sigma=10.,
    min_flux=1e-5,
    max_flux=1e9,
    min_share=1e-1,
    max_share=1,
    balmerfit="em",
    log=get_logger(),
):
    """
    Fits the [OII] doublet line profile with 2 related Gaussians.

    Args:
        emname: "OII" or "OIII" or "HALPHA", "HBETA", "HGAMMA", "HDELTA" (string)
        targetid: targetid (int)
        zspec: redshift (float)
        waves: wavelength in Angstroms (numpy array)
        fluxes: flux observed in the broad band (in erg/s/cm2/A) (numpy array)
        ivars: inverse variance on the flux (numpy array)
        rf_fit_hw (optional, defaults to 40): *rest-frame* wavelength width (in A) used for fitting on each side of the line (float)
        min_rf_fit_hw (optional, defaults to 20): minimum requested *rest-frame* width (in A) on each side of the line to consider the fitting (float)
        rf_cont_w (optional, defaults to 200): *rest-frame* wavelength extent (in A) to fit the continuum (float)
        p0_sigma (optional, defaults to 2.5): initial guess on the line width in A (float)
        p0_flux (optional, defaults to 0.1): initial guess on the line flux in 1e-17 * erg/cm2/s/A (float)
        p0_share (optional, defaults to 0.58): initial guess on the share between the two [OII] lines (float)
        min_sigma (optional, defaults to 1e-5): minimum allowed value for the line width in A (float)
        max_sigma (optional, defaults to 10.): maximum allowed value for the line width in A (float)
        min_flux (optional, defaults to 1e-5): minimum allowed value for the flux in e-17 * erg/cm2/s/A (float)
        max_flux (optional, defaults to 1e9): maximum allowed value for the flux in e-17 * erg/cm2/s/A (float)
        min_share (optional, defaults to 1e-1): minimum allowed value for the share (float)
        max_share (optional, defaults to 1): maximum allowed value for the share (float)
        balmerfit (optional, defaults to "em"): how to fit Balmer lines? "em": emission line only; "em+abs": emission+absorption lines (string)
        log (optional, defaults to get_logger()): Logger object

    Returns:
        mydict: a dictionary with various quantities, noticely "FLUX" and "FLUX_IVAR" (dictionary of floats)
            list of all keys:
                TARGETID, ZSPEC, OBSEMWAVES,
                CHI2, NDOF: *reduced* chi2 and nb of degrees of freedom
                CONT, CONT_IVAR: continuum in 1e-17 * erg/cm2/s/A
                FLUX, FLUX_IVAR: flux in 1e-17 * erg/cm2/s/A
                SIGMA, SIGMA_IVAR: line width in A (observed frame)
                SHARE, SHARE_IVAR: f1/(f0+f1) for OII and OIII doublets
                EW, EW_IVAR: rest-frame equivalent width
        succeed: did the fit succeed? (boolean)
        waves: wavelength values (in A) used for the fitting (numpy array of floats)
        fluxes: flux values (in 1e-17 * erg/cm2/s/A) used for the fitting (numpy array of floats)
        models: model flux values (in 1e-17 * erg/cm2/s/A) from the fit (numpy array of floats)

    Notes:
        Adapted/simplified from elgredshiftflag from J. Comparat (used for eBOSS/ELG):
            https://svn.sdss.org/repo/eboss/elgredshiftflag/
        Returns np.nan in mydict (and NDOF=-99) if not enough pixels to fit or if fit fails.
        For "OII", let the doublet line ratio free during the fit.
        For "OIII", fits the 4960 and 5007 lines with a fixed line ratio.
        For the Balmer lines, SHARE is not fitted and set to np.nan.
    """
    # AR allowed arguments
    if emname not in allowed_emnames:
        log.error("{} not in {}; exiting".format(emname, allowed_emnames))
        sys.exit(1)
    if balmerfit not in ["em", "em+abs"]:
        log.error("{} not in ['em', 'em+abs']".format(balmerfit))
        sys.exit(1)
    # AR Line models
    gauss_nocont = lambda ws, sigma, F0, w0 : F0 * (np.e ** (- (ws - w0) ** 2. / (2. * sigma ** 2.))) / (sigma * (2. * np.pi) ** 0.5)
    # AR vacuum rest-frame wavelength(s)
    rf_em_waves = get_rf_em_waves(emname)
    if emname == "OII":
        cont_choice = "left"
        min_n_lines = 2
    if emname == "OIII":
        cont_choice = "center"
        min_n_lines = 2
    if emname in ["HALPHA", "HBETA", "HGAMMA", "HDELTA"]:
        cont_choice = "center"
        min_n_lines = 1
    # AR expected position of the peak of the line in the observed frame (redshifted). 2 positions given.
    obs_em_waves = (1. + zspec) * rf_em_waves
    # AR *observed-frame* wavelength extents
    obs_fit_hw = (1. + zspec) * rf_fit_hw
    min_obs_fit_hw = (1. + zspec) * min_rf_fit_hw
    obs_cont_w = (1. + zspec) * rf_cont_w
    # AR initializing
    mydict = {}
    mydict["TARGETID"] = targetid
    mydict["Z"] = zspec
    mydict["OBSEMWAVES"] = obs_em_waves
    keys = [
        "FLUX", "FLUX_IVAR", "SIGMA", "SIGMA_IVAR",
        "CONT", "CONT_IVAR", "SHARE", "SHARE_IVAR",
        "EW", "EW_IVAR",
        "CHI2", "NDOF",
    ]
    for key in keys:
        if key == "NDOF":
            mydict[key] = -99
        else:
            mydict[key] = np.nan
    # AR picking wavelengths
    keep_line = np.zeros(len(waves), dtype=bool)
    keep_cont = np.zeros(len(waves), dtype=bool)
    n_cov_lines = 0
    for obs_em_wave in obs_em_waves:
        # AR used for line fitting
        keep_line |= (waves > obs_em_wave - obs_fit_hw) & (waves < obs_em_wave + obs_fit_hw)
        # AR used for continuum
        if cont_choice == "left":
            keep_cont |=  (waves > obs_em_wave - obs_cont_w) & (waves < obs_em_wave)
        if cont_choice == "center":
            keep_cont |= (waves > obs_em_wave - obs_cont_w / 2.) & (waves < obs_em_wave + obs_cont_w / 2.)
        if cont_choice == "right":
            keep_cont |=  (waves > obs_em_wave) & (waves < obs_em_wave + obs_cont_w)
        # AR has the line minimal coverage?
        n_cov_lines += int((waves.min() < obs_em_wave - min_obs_fit_hw) & (waves.max() > obs_em_wave + min_obs_fit_hw))
    # AR excluding for the continuum estimation wavelengths used for line(s) fiting
    keep_cont &= ~keep_line
    # AR discarding flux=nan and ivars == 0 pixels
    keep_line &= (np.isfinite(fluxes)) & (ivars > 0)
    keep_cont &= (np.isfinite(fluxes)) & (ivars > 0)
    # AR
    models = np.nan+np.zeros(keep_line.sum())
    # AR enough pixels to fit?
    succeed = False
    if (
        (keep_line.sum() >= 3) &
        (keep_cont.sum() >= 3) &
        (n_cov_lines >= min_n_lines)
    ):
        # AR continuum flux, ivar
        mydict["CONT"] = np.median(fluxes[keep_cont])
        mydict["CONT_IVAR"] = np.median(ivars[keep_cont])
        #
        # AR OII
        # AR fitting a doublet with line ratio in the fitting
        # AR sh = f3729 / (f3727 + f3729)
        if emname == "OII":
            myfunc = lambda ws, sigma, F0, sh : mydict["CONT"] + gauss_nocont(ws, sigma, (1-sh) * F0, obs_em_waves[0]) + gauss_nocont(ws, sigma, sh * F0, obs_em_waves[1])
            p0 = np.array([p0_sigma, p0_flux, p0_share])
            bounds = ((min_sigma, min_flux, min_share), (max_sigma, max_flux, max_share))
        # AR OIII
        # AR fitting two lines with a fixed line ratio
        # AR f5007 = 2.9 * f4960
        # AR sh = f5007 / (f4960 + f5007)
        # AR sh = 2.9 / (1 + 2.9) = 0.744
        # AR using same F0 and sigma for both lines
        if emname == "OIII":
            sh = 0.744
            myfunc = lambda ws, sigma, F0: mydict["CONT"] + gauss_nocont(ws, sigma, (1-sh) * F0, obs_em_waves[0]) + gauss_nocont(ws, sigma, sh * F0, obs_em_waves[1])
            p0 = np.array([p0_sigma, p0_flux])
            bounds = ((min_sigma, min_flux), (max_sigma, max_flux))
        # AR Balmer lines
        # AR fitting an absorption + emisison line
        if emname in ["HALPHA", "HBETA", "HGAMMA", "HDELTA"]:
            if balmerfit == "em":
                myfunc = lambda ws, sigma, F0 : mydict["CONT"] + gauss_nocont(ws, sigma, F0, obs_em_waves[0])
                p0 = np.array([p0_sigma, p0_flux])
                bounds = ((min_sigma, min_flux), (max_sigma, max_flux))
            if balmerfit == "em+abs":
                myfunc = lambda ws, sigma, F0, abs_sigma, abs_F0 : mydict["CONT"] + gauss_nocont(ws, sigma, F0, obs_em_waves[0]) - gauss_nocont(ws, abs_sigma, abs_F0, obs_em_waves[0])
                p0 = np.array([p0_sigma, p0_flux, p0_sigma, p0_flux])
                bounds = ((min_sigma, min_flux, min_sigma, min_flux), (max_sigma, max_flux, max_sigma, max_flux))
        # AR flux at observed wavelength(s)
        obs_em_fluxes = np.array([fluxes[np.searchsorted(waves, obs_em_wave)] for obs_em_wave in obs_em_waves])
        # AR is the flux above continuum for at least one line?
        if obs_em_fluxes.max() > mydict["CONT"]:
            # AR maxfev and gtol set by JC; seems to work; not touching those...
            popt, pcov = curve_fit(
                myfunc,
                waves[keep_line],
                fluxes[keep_line],
                p0 = p0,
                sigma = 1. / np.sqrt(ivars[keep_line]),
                maxfev = 10000000,
                gtol = 1.49012e-8,
                bounds = bounds,
            )
            # AR fit succeeded?
            # AR - pcov.__class__ criterion => dates from JC, not sure how relevant.. keeping it
            # AR - then we decide that the fit is no good if:
            # AR   - var = 0,
            # AR   - or var(flux) = 0
            # AR   - or flux closer than 1% to the allowed boundaries
            if pcov.__class__ == np.ndarray:
                diag = np.diag(pcov)
                if (diag.sum() > 0) & (diag[1] > 0) & (popt[1] > 1.01 * min_flux) & (popt[1] < 0.99 * max_flux):
                    succeed = True
                    if emname == "OII":
                        models = myfunc(waves[keep_line], popt[0], popt[1], popt[2])
                        mydict["SHARE"] = popt[2]
                        mydict["SHARE_IVAR"] = diag[2] ** -1
                    if emname == "OIII":
                        models = myfunc(waves[keep_line], popt[0], popt[1])
                        mydict["SHARE"] = sh
                        mydict["SHARE_IVAR"] = 0.
                    if emname in ["HALPHA", "HBETA", "HGAMMA", "HDELTA"]:
                        if balmerfit == "em":
                            models = myfunc(waves[keep_line], popt[0], popt[1])
                        if balmerfit == "em+abs":
                            models = myfunc(waves[keep_line], popt[0], popt[1], popt[2], popt[3])
                    mydict["NDOF"]    = keep_line.sum() - len(p0)
                    mydict["CHI2"]    = np.sum(np.abs(models - fluxes[keep_line]) ** 2. / ivars[keep_line] ** 2.)
                    mydict["CHI2"]   /= mydict["NDOF"] # AR we define CHI2 as the reduced chi2, as in fastspecfit
                    mydict["SIGMA"]   = popt[0]
                    mydict["SIGMA_IVAR"]= diag[0] ** -1
                    mydict["FLUX"]    = popt[1]
                    mydict["FLUX_IVAR"] = diag[1] ** -1
                    # AR rest-frame equivalent width
                    factor = (1 + zspec) / mydict["CONT"]
                    mydict["EW"] = mydict["FLUX"] * factor
                    mydict["EW_IVAR"] = mydict["FLUX_IVAR"] / factor ** 2
    #
    return mydict, succeed, waves[keep_line], fluxes[keep_line], ivars[keep_line], models

def get_emlines(
    redrockfn,
    coaddfn,
    targetids=None,
    emnames=["OII", "HDELTA", "HGAMMA", "HBETA", "OIII", "HALPHA"],
    rf_fit_hw=40,
    min_rf_fit_hw=20,
    rf_cont_w=200,
    rv=3.1,
    balmerfit="em",
    outpdf=None,
    ylim=(None, None),
    nrow=10,
    log=get_logger(),
):
    """
    Get Gaussian-fitted emission line flux for a coadd; corrected for foreground MW extinction before
fitting.

    Args:
        redrockfn: full path to a redrock/zbest file
        coaddfn: full path to a coadd file (everest-format)
        targetids (optional, defaults to None): list of TARGETIDs to restrict to (list or numpy array)
        emnames (optional, defaults to ["OII", "HDELTA", "HGAMMA", "HBETA", "OIII", "HALPHA"]): list of lines to fit (list of string)
        rf_fit_hw (optional, defaults to 40): *rest-frame* wavelength width (in A) used for fitting on each side of the line (float)
        min_rf_fit_hw (optional, defaults to 20): minimum requested *rest-frame* width (in A) on each side of the line to consider the fitting (float)
        rf_cont_w (optional, defaults to 200): *rest-frame* wavelength extent (in A) to fit the continuum (float)
        rv (optional, defaults to 3.1): value of R_V (float)
        balmerfit (optional, defaults to "em"): how to fit Balmer lines? "em": emission line only; "em+abs": emission+absorption lines (string)
        outpdf (optional, defaults to None): PDF filename for plotting the OII doublet (data + fit) (string)
        ylim (optional, defaults to (None, None)): ylim for plotting (float doublet)
        nrow (optional, defaults to 10): number of rows, i.e. galaxy, per pdf page (int)
        log (optional, defaults to get_logger()): Logger object

    Returns:
        mydict: a dictionary with various quantities,
            list of all keys:
                TARGETID,TARGET_RA,TARGET_DEC,OBJTYPE,Z,ZWARN,SPECTYPE,DELTACHI2: numpy array with len(targetids) objects
                subdictionaries for each emname in emnames, with:
                    OBSEMWAVES
                    CHI2, NDOF: *reduced* chi2 and nb of degrees of freedom
                    CONT, CONT_IVAR: continuum in 1e-17 * erg/cm2/s/A
                    FLUX, FLUX_IVAR: flux in 1e-17 * erg/cm2/s/A
                    SIGMA, SIGMA_IVAR: line width in A (observed frame)
                    SHARE, SHARE_IVAR: f1/(f0+f1) for OII and OIII doublets
                    EW, EW_IVAR: rest-frame equivalent width
                    wave, data, ivar, model: data used for fitting + fitted model
    """
    # AR sanity checks
    for fn in [redrockfn, coaddfn]:
        if not os.path.isfile(fn):
            log.error("no {} file; exiting".format(fn))
            sys.exit(1)

    # AR redrock: reading TARGETID and ZSPEC (+ cut on TARGETIDs if requested)
    h = fits.open(redrockfn)
    extnames = [h[i].header["EXTNAME"] for i in range(1, len(h))]
    if "REDSHIFTS" in extnames:
        d_rr = h["REDSHIFTS"].data
    elif "ZBEST" in extnames:
        d_rr = h["ZBEST"].data
    else:
        log.error("{} has neither REDSHIFTS or ZBEST extension; exiting".format(redrockfn))
        sys.exit(1)
    # AR cutting on TARGETID?
    if targetids is None:
        targetids = d_rr["TARGETID"]
    nspec = len(targetids)
    log.info("Dealing with {} spectra".format(nspec))
    ii_rr = match_to(d_rr["TARGETID"], targetids)
    d_rr = d_rr[ii_rr]
    if len(d_rr) != nspec:
        log.error("{} TARGETIDs are not in {}; exiting".format(nspec - len(d_rr), redrockfn))
        sys.exit(1)

    # AR coadd: fibermap cut + sanity check
    # AR coadd: fibermap has 500 rows (even in pre-everest)
    h = fits.open(coaddfn)
    # AR requested TARGETIDs
    ii_co = match_to(h["FIBERMAP"].data["TARGETID"], targetids)
    if ii_co.size != nspec:
        log.error("{} TARGETIDs are not in {}; exiting".format(nspec - ii_co.size, coaddfn))
        sys.exit(1)
    d_fm = h["FIBERMAP"].data[ii_co]
    ebvs = dust_ebv(d_fm["TARGET_RA"], d_fm["TARGET_DEC"])

    # AR available cameras
    cameras = []
    extnames = [h[i].header['extname'] for i in range(1, len(h))]
    for camera in ["B", "R", "Z"]:
        if "{}_FLUX".format(camera) in extnames:
            cameras.append(camera)
    # AR cutting FLUX and IVAR arrays on TARGETIDs
    for camera in cameras:
        h["{}_FLUX".format(camera)].data = h["{}_FLUX".format(camera)].data[ii_co, :]
        h["{}_IVAR".format(camera)].data = h["{}_IVAR".format(camera)].data[ii_co, :]
    # AR b : 3600,5800  , r : 5760,7620 , z : 7520,9824
    # AR we keep all data points (i.e. including overlaps)
    # AR note that waves is not ordered, as a consequence
    waves = np.zeros(0)
    fluxes = np.zeros(0).reshape(nspec, 0)
    ivars = np.zeros(0).reshape(nspec, 0)
    for camera in cameras:
        tmpw = h["{}_WAVELENGTH".format(camera)].data
        tmpfl = h["{}_FLUX".format(camera)].data
        tmpiv = h["{}_IVAR".format(camera)].data
        # AR TBD: use a smarter way with no loop on tids...
        # AR TBD: but as 500 rows at most, ~ok
        tmpexts = ext_odonnell(tmpw, Rv=rv)
        for i in range(nspec):
            tmp_mw_trans =  10 ** (-0.4 * ebvs[i] * rv * tmpexts)
            tmpfl[i, :] /= tmp_mw_trans
            tmpiv[i, :] *= tmp_mw_trans ** 2
        waves = np.append(waves, tmpw)
        fluxes = np.append(fluxes, tmpfl, axis=1)
        ivars = np.append(ivars, tmpiv, axis=1)
    # AR dictionary to store results
    # AR keys returned by emlines_gaussfit
    emkeys = [
        "OBSEMWAVES",
        "FLUX", "FLUX_IVAR", "SIGMA", "SIGMA_IVAR",
        "CONT", "CONT_IVAR", "SHARE", "SHARE_IVAR",
        "EW", "EW_IVAR",
        "CHI2", "NDOF",
    ]
    mydict = {}
    mydict["TARGETID"] = targetids
    for key in ["TARGET_RA", "TARGET_DEC", "OBJTYPE"]:
        mydict[key] = d_fm[key]
    for key in ["Z", "ZWARN", "SPECTYPE", "DELTACHI2"]:
        mydict[key] = d_rr[key]
    for emname in emnames:
        mydict[emname] = {}
        for key in emkeys:
            if key == "OBSEMWAVES":
                mydict[emname][key] = np.zeros(nspec, dtype=object)
            elif key == "NDOF":
                mydict[emname][key] = -99 + np.zeros(nspec, dtype=int)
            else:
                mydict[emname][key] = np.nan + np.zeros(nspec)
        for key in ["wave", "fitdata", "fitivar", "model"]:
            mydict[emname][key] = np.zeros(nspec, dtype=object)
    for i_emname, emname in enumerate(emnames):
        for i in range(nspec):
            tmpmydict, _, w, d, iv, m = emlines_gaussfit(
                emname,
                targetids[i],
                d_rr["Z"][i],
                waves,
                fluxes[i, :],
                ivars[i, :],
                rf_fit_hw=rf_fit_hw,
                min_rf_fit_hw=min_rf_fit_hw,
                rf_cont_w=rf_cont_w,
                balmerfit=balmerfit,
                log=log,
            )
            for key in emkeys:
                mydict[emname][key][i] = tmpmydict[key]
            mydict[emname]["wave"][i] = w
            mydict[emname]["fitdata"][i] = d
            mydict[emname]["fitivar"][i] = iv
            mydict[emname]["model"][i] = m
    # AR plot?
    if outpdf is not None:
        # AR plotting by increasing redshift
        ii = mydict["Z"].argsort()
        with PdfPages(outpdf) as pdf:
            for ix, i in enumerate(ii):
                if ix % nrow == 0:
                    fig = plt.figure(figsize = (25,15))
                    gs = gridspec.GridSpec(nrow, len(emnames), wspace=0.1, hspace=0.1)
                for i_emname, emname in enumerate(emnames):
                    ax = plt.subplot(gs[ix % nrow, i_emname])
                    w = mydict[emname]["wave"][i]
                    d = mydict[emname]["fitdata"][i]
                    iv = mydict[emname]["fitivar"][i]
                    m = mydict[emname]["model"][i]
                    jj = w.argsort() # AR case of camera overlap
                    w, d, iv, m = w[jj], d[jj], iv[jj], m[jj]
                    jj = np.where(w[1:] - w[:-1] > 10)[0]
                    for j in jj:
                        d[j], iv[j], m[j] = np.nan, np.nan, np.nan
                    obs_em_waves = mydict[emname]["OBSEMWAVES"][i]
                    ax.plot(w, d, c="k", lw=0.5, alpha=1.0, label="data")
                    ax.fill_between(w, d - iv ** (-0.5), d + iv ** (-0.5), color="k", alpha=0.3)
                    ax.plot(w, m, c="r", lw=0.5, alpha=1.0, label="model")
                    ax.axhline(mydict[emname]["CONT"][i], c="r", lw=0.5, ls="--", alpha=1.0)
                    for obs_em_wave in obs_em_waves:
                        ax.axvline(obs_em_wave, c="r", lw=0.5, ls="--", alpha=1.0)
                    fs = 8
                    ax.text(0.05, 0.85, "{}".format(mydict["TARGETID"][i]), fontsize=fs, transform=ax.transAxes)
                    ax.text(0.05, 0.75, "Z={:.3f}".format(mydict["Z"][i]), fontsize=fs, transform=ax.transAxes)
                    ax.text(0.05, 0.05, "SNR({})={:.1f}".format(emname, mydict[emname]["FLUX"][i] * mydict[emname]["FLUX_IVAR"][i] ** 0.5), fontsize=fs, transform=ax.transAxes)
                    ax.text(0.95, 0.85, "OBJTYPE={}".format(mydict["OBJTYPE"][i]), fontsize=fs, ha="right", transform=ax.transAxes)
                    ax.text(0.95, 0.75, "SPECTYPE={}".format(mydict["SPECTYPE"][i]), fontsize=fs, ha="right", transform=ax.transAxes)
                    ax.text(0.95, 0.05, "log10(DELTACHI2)={:.1f}".format(np.log10(mydict["DELTACHI2"][i])), fontsize=fs, ha="right", transform=ax.transAxes)
                    ax.grid(True)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_ylim(ylim)
                if (ix % nrow == nrow - 1) | (i == ii[-1]):
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close()
    return mydict
