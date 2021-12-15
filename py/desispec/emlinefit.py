#!/usr/bin/env python

"""
desispec.io.emlinefit
====================
Routines for desi_emlinefit_afterburner.
"""

import numpy as np
from scipy.optimize import curve_fit
from desiutil.log import get_logger

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
    log=None,
):
    """
    Fits the [OII] doublet line profile with 2 related Gaussians.

    Args:
        emname: "OII" or "OIII" or "HALPHA", "HBETA", "HGAMMA", "HDELTA" (string)
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
        log (optional, defaults to get_logger()): Logger object

    Returns:
        emdict: a dictionary with various quantities, noticely "FLUX" and "FLUX_IVAR" (dictionary of floats)
            list of all keys:
                CHI2, NDOF: *reduced* chi2 and nb of degrees of freedom
                CONT, CONT_IVAR: continuum in 1e-17 * erg/cm2/s/A
                FLUX, FLUX_IVAR: flux in 1e-17 * erg/cm2/s/A
                SIGMA, SIGMA_IVAR: line width in A (observed frame)
                SHARE, SHARE_IVAR: f1/(f0+f1) for OII and OIII doublets
                EW, EW_IVAR: rest-frame equivalent width
                waves: wavelength values (in A) used for the fitting (numpy array of floats)
                fluxes: flux values (in 1e-17 * erg/cm2/s/A) used for the fitting (numpy array of floats)
                ivars: ivar values used for the fitting (numpy array of floats)
                models: model flux values (in 1e-17 * erg/cm2/s/A) from the fit (numpy array of floats)
        succeed: did the fit succeed? (boolean)

    Notes:
        Adapted/simplified from elgredshiftflag from J. Comparat (used for eBOSS/ELG):
            https://svn.sdss.org/repo/eboss/elgredshiftflag/
        Returns np.nan in emdict (and NDOF=-99) if not enough pixels to fit or if fit fails.
        Default settings designed for ELGs (e.g. max_sigma); need to re-assess if run on other targets.
        For "OII", let the doublet line ratio free during the fit.
        For "OIII", fits the 4960 and 5007 lines with a fixed line ratio.
        For the Balmer lines, SHARE is not fitted and set to np.nan.
    """
    # AR log
    if log is None:
        log = get_logger()
    # AR allowed arguments
    if emname not in allowed_emnames:
        msg = "{} not in {}".format(emname, allowed_emnames)
        log.error(msg)
        raise ValueError(msg)
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
    # AR "waves", "fluxes", "ivars", "models" are dealt separately
    emdict = {}
    keys = [
        "FLUX", "FLUX_IVAR", "SIGMA", "SIGMA_IVAR",
        "CONT", "CONT_IVAR", "SHARE", "SHARE_IVAR",
        "EW", "EW_IVAR",
        "CHI2", "NDOF",
    ]
    for key in keys:
        if key == "NDOF":
            emdict[key] = -99
        else:
            emdict[key] = np.nan
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
        emdict["CONT"] = np.median(fluxes[keep_cont])
        emdict["CONT_IVAR"] = np.median(ivars[keep_cont])
        #
        # AR OII
        # AR fitting a doublet with line ratio in the fitting
        # AR sh = f3729 / (f3727 + f3729)
        if emname == "OII":
            myfunc = lambda ws, sigma, F0, sh : emdict["CONT"] + gauss_nocont(ws, sigma, (1-sh) * F0, obs_em_waves[0]) + gauss_nocont(ws, sigma, sh * F0, obs_em_waves[1])
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
            myfunc = lambda ws, sigma, F0: emdict["CONT"] + gauss_nocont(ws, sigma, (1-sh) * F0, obs_em_waves[0]) + gauss_nocont(ws, sigma, sh * F0, obs_em_waves[1])
            p0 = np.array([p0_sigma, p0_flux])
            bounds = ((min_sigma, min_flux), (max_sigma, max_flux))
        # AR Balmer lines
        # AR fitting emission line only (no absorption)
        if emname in ["HALPHA", "HBETA", "HGAMMA", "HDELTA"]:
            myfunc = lambda ws, sigma, F0 : emdict["CONT"] + gauss_nocont(ws, sigma, F0, obs_em_waves[0])
            p0 = np.array([p0_sigma, p0_flux])
            bounds = ((min_sigma, min_flux), (max_sigma, max_flux))
        # AR flux at observed wavelength(s)
        obs_em_fluxes = np.array([fluxes[np.searchsorted(waves, obs_em_wave)] for obs_em_wave in obs_em_waves])
        # AR is the flux above continuum for at least one line?
        if obs_em_fluxes.max() > emdict["CONT"]:
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
                        emdict["SHARE"] = popt[2]
                        emdict["SHARE_IVAR"] = diag[2] ** -1
                    if emname == "OIII":
                        models = myfunc(waves[keep_line], popt[0], popt[1])
                        emdict["SHARE"] = sh
                        emdict["SHARE_IVAR"] = np.inf
                    if emname in ["HALPHA", "HBETA", "HGAMMA", "HDELTA"]:
                        models = myfunc(waves[keep_line], popt[0], popt[1])
                    emdict["NDOF"]    = keep_line.sum() - len(p0)
                    emdict["CHI2"]    = np.sum(np.abs(models - fluxes[keep_line]) ** 2. / ivars[keep_line] ** 2.)
                    emdict["CHI2"]   /= emdict["NDOF"] # AR we define CHI2 as the reduced chi2, as in fastspecfit
                    emdict["SIGMA"]   = popt[0]
                    emdict["SIGMA_IVAR"]= diag[0] ** -1
                    emdict["FLUX"]    = popt[1]
                    emdict["FLUX_IVAR"] = diag[1] ** -1
                    # AR rest-frame equivalent width
                    factor = (1 + zspec) / emdict["CONT"]
                    emdict["EW"] = emdict["FLUX"] * factor
                    emdict["EW_IVAR"] = emdict["FLUX_IVAR"] / factor ** 2
    # AR fitted waves / fluxes / ivars / models
    emdict["waves"] = waves[keep_line]
    emdict["fluxes"] = fluxes[keep_line]
    emdict["ivars"] = ivars[keep_line]
    emdict["models"] = models
    #
    return emdict, succeed



def get_emlines(
    zspecs,
    waves,
    fluxes,
    ivars,
    emnames=["OII", "HDELTA", "HGAMMA", "HBETA", "OIII", "HALPHA"],
    rf_fit_hw=40,
    min_rf_fit_hw=20,
    rf_cont_w=200,
    log=None,
):
    """
    Get Gaussian-fitted emission line fluxes.

    Args:
        zspecs: redshifts (numpy array of shape (Nspec))
        waves: wavelengths (numpy array of shape (Nwave))
        fluxes: fluxes (numpy array of shape (Nspec, Nwave))
        ivars: inverse variances (numpy array of shape (Nspec, Nwave))
        emnames (optional, defaults to ["OII", "HDELTA", "HGAMMA", "HBETA", "OIII", "HALPHA"]): list of lines to fit (list of string)
        rf_fit_hw (optional, defaults to 40): *rest-frame* wavelength width (in A) used for fitting on each side of the line (float)
        min_rf_fit_hw (optional, defaults to 20): minimum requested *rest-frame* width (in A) on each side of the line to consider the fitting (float)
        rf_cont_w (optional, defaults to 200): *rest-frame* wavelength extent (in A) to fit the continuum (float)
        log (optional, defaults to get_logger()): Logger object

    Returns:
        emdict: a dictionary with a subdictionary for each emname in emnames, with:,
            CHI2, NDOF: *reduced* chi2 and nb of degrees of freedom
            CONT, CONT_IVAR: continuum in 1e-17 * erg/cm2/s/A
            FLUX, FLUX_IVAR: flux in 1e-17 * erg/cm2/s/A
            SIGMA, SIGMA_IVAR: line width in A (observed frame)
            SHARE, SHARE_IVAR: f1/(f0+f1) for OII and OIII doublets
            EW, EW_IVAR: rest-frame equivalent width
            waves, fluxes, ivars, models: data used for fitting + fitted model
    """
    # AR log
    if log is None:
        log = get_logger()

    # AR number of spectra
    nspec = len(zspecs)

    # AR assert consistency
    if (
        (fluxes.shape[0] != nspec) |
        (fluxes.shape[1] != waves.shape[0]) |
        (ivars.shape[0] != nspec) |
        (ivars.shape[1] != waves.shape[0])
    ):
        log.error(
            "Shape inconsistencies for inputs: zspecs={}, waves={}, fluxes={}, ivars={}".format(
                zspecs.shape, waves.shape, fluxes.shape, ivars.shape,
            )
        )
        raise RuntimeError()

    # AR dictionary to store results
    # AR keys returned by emlines_gaussfit
    emkeys = [
        "FLUX", "FLUX_IVAR", "SIGMA", "SIGMA_IVAR",
        "CONT", "CONT_IVAR", "SHARE", "SHARE_IVAR",
        "EW", "EW_IVAR",
        "CHI2", "NDOF",
        "waves", "fluxes", "ivars", "models",
    ]
    emdict = {}

    # AR prepare columns
    for emname in emnames:
        emdict[emname] = {}
        for key in emkeys:
            if key == "NDOF":
                emdict[emname][key] = -99 + np.zeros(nspec, dtype=int)
            elif key in ["waves", "fluxes", "ivars", "models"]:
                emdict[emname][key] = np.zeros(nspec, dtype=object)
            else:
                emdict[emname][key] = np.nan + np.zeros(nspec)

    # AR fit Gaussians
    for i_emname, emname in enumerate(emnames):
        for i in range(nspec):
            tmpemdict, _ = emlines_gaussfit(
                emname,
                zspecs[i],
                waves,
                fluxes[i, :],
                ivars[i, :],
                rf_fit_hw=rf_fit_hw,
                min_rf_fit_hw=min_rf_fit_hw,
                rf_cont_w=rf_cont_w,
                log=log,
            )
            for key in emkeys:
                emdict[emname][key][i] = tmpemdict[key]
    return emdict
