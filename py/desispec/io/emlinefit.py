#!/usr/bin/env python

"""
desispec.io.emlinefit
====================
IO routines for desi_emlinefit_afterburner.
"""

import os
import sys
from astropy.io import fits
import fitsio
import numpy as np
from desitarget.geomask import match_to
from desiutil.dust import ext_odonnell
from desiutil.dust import ebv as dust_ebv
from desiutil.log import get_logger
from desispec.emlinefit import get_rf_em_waves
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import gridspec


def get_targetids(redrockfn, bitnames, log=get_logger()):
    """
    Returns the TARGETIDs passing the bitnames for the CMX_TARGET, SV{1,2,3}_DESI_TARGET, or DESI_TARGET mask.

    Args:
        redrockfn: full path to a redrock/zbest file
        bitnames: comma-separated list of target bitnames to fit from the *DESI_TARGET mask (string)
        log (optional, defaults to get_logger()): Logger object

    Returns:
        targetids: list of TARGETIDs passing the *DESI_TARGET mask (np.array)

    Notes:
        If several bitnames are provided, selects the union of those.
        Use the FIBERMAP extension.
        Do not use desitarget.targets.main_cmx_or_sv, as we only read TARGETID and the mask column,
            so it is faster; but we use same philosophy.
    """
    # AR get the *DESI_TARGET colum
    dtkeys = [
        key for key in fits.open(redrockfn)["FIBERMAP"].columns.names
            if key in ["CMX_TARGET", "SV1_DESI_TARGET", "SV2_DESI_TARGET", "SV3_DESI_TARGET"]
    ]
    # AR if no match, then assume it is a main catalog
    if len(dtkeys) == 0:
        dtkey = "DESI_TARGET"
        from desitarget.targetmask import desi_mask as mask
    elif len(dtkeys) == 1:
        dtkey = dtkeys[0]
        if dtkey == "CMX_TARGET":
            from desitarget.cmx.cmx_targetmask import cmx_mask as mask
        if dtkey == "SV1_DESI_TARGET":
            from desitarget.sv1.sv1_targetmask import desi_mask as mask
        if dtkey == "SV2_DESI_TARGET":
            from desitarget.sv2.sv2_targetmask import desi_mask as mask
        if dtkey == "SV3_DESI_TARGET":
            from desitarget.sv3.sv3_targetmask import desi_mask as mask
    else:
        log.info(
            "found {}>1 matching keys: {}; 0 or 1 key expected; exiting".format(
                len(dtkeys), dtkeys
            )
        )
        sys.exit(1)
    allowed_bitnames = mask.names()
    # AR read + select the targetids
    d = fitsio.read(redrockfn, columns=["TARGETID", dtkey], ext="FIBERMAP")
    sel = np.zeros(len(d), dtype=bool)
    for bitname in bitnames.split(","):
        if bitname not in allowed_bitnames:
            log.info(
                "{} not in allowed bitnames for {} ({}; exiting)".format(
                    bitname, dtkey, allowed_bitnames
                )
            )
            sys.exit(1)
        sel |= (d[dtkey] & mask[bitname]) > 0
    targetids = d["TARGETID"][sel]
    log.info("selecting {} targets with {} in {}".format(sel.sum(), bitnames, dtkey))
    return targetids



def read_emlines_inputs(
    redrockfn,
    coaddfn,
    mwext_corr=True,
    rv=3.1,
    targetids=None,
    rr_keys="TARGETID,Z,ZWARN,SPECTYPE,DELTACHI2",
    fm_keys="TARGET_RA,TARGET_DEC,OBJTYPE",
    log=get_logger(),
):
    """
    Read the columns and spectra information (waves, fluxes, ivars) from the redrock and coadd files,
    which will go as input to get_emlines().

    Args:
        redrockfn: full path to a redrock/zbest file
        coaddfn: full path to a coadd file (everest-format)
        mwext_corr (optional, defaults to True): correct flux for foreground MW extinction? (boolean)
        rv (optional, defaults to 3.1): value of R_V, used if mwext_corr=True (float)
        targetids (optional, defaults to None): list of TARGETIDs to restrict to (list or numpy array)
        rr_keys (optional, defaults to "TARGETID,Z,ZWARN,SPECTYPE,DELTACHI2"): comma-separated list of columns from REDSHIFTS to propagate (string)
        fm_keys (optional, defaults to "TARGET_RA,TARGET_DEC,OBJTYPE"): comma-separated list of columns from FIBERMAP to propagate (string)
        log (optional, defaults to get_logger()): Logger object

    Returns:
        rr: structured array with the desired redrock columns for the desired TARGETIDs
        fm: structured array with the desired fibermap columns for the desired TARGETIDs
        waves: wavelengths (numpy array of shape (Nwave))
        fluxes: Galactic-extinction-corrected fluxes (numpy array of shape (Nspec, Nwave))
        ivars: inverse variances (numpy array of shape (Nspec, Nwave))

    Notes:
        We add TARGETID and Z to rr_keys if TARGETID not present in rr_keys nor in fm_keys.
        If keys in rr_keys or fm_keys are not present in the redrockfn, those will be ignored.
    """
    # AR sanity checks
    for fn in [redrockfn, coaddfn]:
        if not os.path.isfile(fn):
            log.error("no {} file; exiting".format(fn))
            sys.exit(1)
    keys = [key for key in rr_keys.split(",") if key in fm_keys.split(",")]
    if len(keys) > 0:
        log.error("the following columns are both present in rr_keys and fm_keys: {}; exiting".format(",".join(keys)))
        sys.exit(1)
    # AR grab TARGETID from the REDSHIFTS/ZBEST extension
    if "TARGETID" in fm_keys.split(","):
        log.info("removing TARGETID from fm_keys")
        fm_keys = ",".join([key for key in fm_keys.split(",") if key != "TARGETID"])
    for key in ["TARGETID", "Z"]:
        if key not in rr_keys.split(","):
            log.info("adding {} to rr_keys".format(key))
            rr_keys = "{},{}".format(key, rr_keys)

    # AR redrock: extension name + columns
    h = fits.open(redrockfn)
    extnames = [h[i].header["EXTNAME"] for i in range(1, len(h))]
    if "REDSHIFTS" in extnames:
        rr_extname = "REDSHIFTS"
    elif "ZBEST" in extnames:
        rr_extname = "ZBEST"
    else:
        log.error("{} has neither REDSHIFTS or ZBEST extension; exiting".format(redrockfn))
        sys.exit(1)
    # AR rr_keys, fm_keys: restrict to existing ones
    rmv_rr_keys = [key for key in rr_keys.split(",") if key not in h[rr_extname].columns.names]
    if len(rmv_rr_keys) > 0:
        log.info("{} removed from rr_keys, as not present in {}".format(",".join(rmv_rr_keys), rr_extname))
    rr_keys = ",".join([key for key in rr_keys.split(",") if key not in rmv_rr_keys])
    # AR redrock: reading
    rr = fitsio.read(redrockfn, ext=rr_extname, columns=rr_keys.split(","))
    # AR redrock: cutting on TARGETID if requested
    if targetids is None:
        targetids = rr["TARGETID"]
    nspec = len(targetids)
    log.info("Dealing with {} spectra".format(nspec))
    ii_rr = match_to(rr["TARGETID"], targetids)
    rr = rr[ii_rr]
    if len(rr) != nspec:
        log.error("{} TARGETIDs are not in {}; exiting".format(nspec - len(rr), redrockfn))
        sys.exit(1)

    # AR coadd: fibermap cut + sanity check
    # AR coadd: fibermap has 500 rows (even in pre-everest)
    h = fits.open(coaddfn)
    # AR coadd: fibermap columns
    rmv_fm_keys = [key for key in fm_keys.split(",") if key not in h["FIBERMAP"].columns.names]
    if len(rmv_fm_keys) > 0:
        log.info("{} removed from fm_keys, as not present in FIBERMAP".format(",".join(rmv_fm_keys)))
    fm_keys = ",".join([key for key in fm_keys.split(",") if key not in rmv_fm_keys])
    # AR coadd: fibermap read (and TARGETID separately)
    fm = fitsio.read(coaddfn, ext="FIBERMAP", columns=fm_keys.split(","))
    fm_tids = fitsio.read(coaddfn, ext="FIBERMAP", columns=["TARGETID"])["TARGETID"]
    # AR requested TARGETIDs
    ii_co = match_to(fm_tids, targetids)
    if ii_co.size != nspec:
        log.error("{} TARGETIDs are not in {}; exiting".format(nspec - ii_co.size, coaddfn))
        sys.exit(1)
    fm = fm[ii_co]
    if mwext_corr:
        ebvs = dust_ebv(fm["TARGET_RA"], fm["TARGET_DEC"])

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
        if mwext_corr:
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

    return rr, fm, waves, fluxes, ivars



def write_emlines(
    outfn,
    emdict,
    rr=None,
    fm=None,
    redrockfn=None,
    coaddfn=None,
    rf_fit_hw=None,
    min_rf_fit_hw=None,
    rf_cont_w=None,
    rv=None,
    log=get_logger(),
):
    """
    Writes the emission line fitting to a file.

    Args:
        outfn: output fits file (string)
        emdict: dictionary output by get_emlines()
        rr (optional, defaults to None): redrock REDSHIFTS/ZBEST extension
        fm (optional, defaults to None): redrock FIBERMAP extension
        redrockfn (optional, defaults to None): used redrock/zbest file
        coaddfn (optional, defaults to None): used coadd file (everest-format)
        rf_fit_hw (optional, defaults to None): *rest-frame* wavelength width (in A) used for fitting on each side of the line (float)
        min_rf_fit_hw (optional, defaults to None): minimum requested *rest-frame* width (in A) on each side of the line to consider the fitting (float)
        rf_cont_w (optional, defaults to None): *rest-frame* wavelength extent (in A) to fit the continuum (float)
        rv (optional, defaults to None): value of R_V (float)
        log (optional, defaults to get_logger()): Logger object
    """
    # AR fitted emission lines
    emnames = list(emdict.keys())

    # AR number of fitted spectra
    nspec = emdict[emnames[0]]["FLUX"].size

    # AR data model
    dtype = []
    if rr is not None:
        dtype += [(key, rr.dtype[key]) for key in rr.dtype.names]
    if fm is not None:
        dtype += [(key, fm.dtype[key]) for key in fm.dtype.names]
    for emname in emnames:
        for key in list(emdict[emname].keys()):
            if key not in ["waves", "fluxes", "ivars", "models"]:
                if key == "NDOF":
                    dtype.append(("{}_{}".format(emname, key), ">i4"))
                else:
                    dtype.append(("{}_{}".format(emname, key), ">f4"))

    # AR store in table
    d = np.zeros(nspec, dtype=dtype)

    if rr is not None:
        for key in rr.dtype.names:
            d[key] = rr[key]
    if fm is not None:
        for key in fm.dtype.names:
            d[key] = fm[key]
    for emname in emnames:
        for key in list(emdict[emname].keys()):
            if key not in ["waves", "fluxes", "ivars", "models"]:
                d["{}_{}".format(emname, key)] = emdict[emname][key]

    # AR header
    hdr = fitsio.FITSHDR()
    if redrockfn is not None:
        hdr["RRFN"] = redrockfn
    if coaddfn is not None:
        hdr["COADDFN"] = coaddfn
    if rf_fit_hw is not None:
        hdr["RFHW"] = rf_fit_hw
    if min_rf_fit_hw is not None:
        hdr["MINRFHW"] = min_rf_fit_hw
    if rf_cont_w is not None:
        hdr["RFCONTW"] = rf_cont_w
    if rv is not None:
        hdr["RV"] = rv
    hdr["EMNAMES"] = emnames
    for i_emname, emname in enumerate(emnames):
        hdr["RFWAVE{:02d}".format(i_emname)] = ",".join(get_rf_em_waves(emname).astype(str))

    # AR write to fits
    if os.path.isfile(outfn):
        log.info("Removing existing {}".format(outfn))
        os.remove(outfn)
    fd = fitsio.FITS(outfn, "rw")
    fd.write(d, extname="EMLINEFIT", header=hdr)
    fd.close()



def plot_emlines(
    outpdf,
    zspecs,
    emdict,
    emnames=["OII", "HDELTA", "HGAMMA", "HBETA", "OIII", "HALPHA"],
    targetids=None,
    objtypes=None,
    spectypes=None,
    deltachi2s=None,
    ylim=(None, None),
    nrow=10,
    rowsort_byz=True,
):
    """
    Plot the fitted data from get_emlines().

    Args:
        outpdf: output pdf filename (string)
        zspecs: list of redrock redshifts (numpy array)
        emdict: a dictionary with various quantities, output by get_lines().
            used content:
            - each emission line has its own subdictionary.
            list of used keys:
                subdictionaries for each emname in emnames, with:
                    CONT : continuum in 1e-17 * erg/cm2/s/A
                    FLUX, FLUX_IVAR: flux in 1e-17 * erg/cm2/s/A
                    waves, fluxes, ivars, models: data used for fitting + fitted model
        emnames (optional, defaults to ["OII", "HDELTA", "HGAMMA", "HBETA", "OIII", "HALPHA"]): list of plotted lines (list of string)
        targetids (optional, defaults to None): list of TARGETIDs (numpy array)
        objtypes (optional, defaults to None): list of fibermap OBJTYPEs (numpy array)
        spectypes (optional, defaults to None): list of redrock SPECTYPEs (numpy array)
        deltachi2s (optional, defaults to None): list of redrock DELTACHI2s (numpy array)
        ylim (optional, defaults to (None, None)): ylim for plotting (float doublet)
        nrow (optional, defaults to 10): number of rows, i.e. galaxy, per pdf page (int)
        rowsort_byz (optional, defaults to True): the spectra are displayed by increasing redshifts (boolean)

    Notes:
        If some emname is not present in emdict, it will be discarded.
        Each spectra will be one row of plot, each emission line corresponds to a column.
        If rowsort_byz=False, then the spectra will be displayed by order of appearance.
    """
    # AR emnames present in emdict
    emnames = [emname for emname in emnames if emname in emdict]

    # AR plotting by increasing redshift?
    if rowsort_byz:
        ii = zspecs.argsort()
    else:
        ii = np.arange(zspecs.size, dtype=int)

    # AR start plot
    with PdfPages(outpdf) as pdf:
        for ix, i in enumerate(ii):
            if ix % nrow == 0:
                fig = plt.figure(figsize = (25,15))
                gs = gridspec.GridSpec(nrow, len(emnames), wspace=0.1, hspace=0.1)
            for i_emname, emname in enumerate(emnames):
                ax = plt.subplot(gs[ix % nrow, i_emname])
                w = emdict[emname]["waves"][i]
                d = emdict[emname]["fluxes"][i]
                iv = emdict[emname]["ivars"][i]
                m = emdict[emname]["models"][i]
                jj = w.argsort() # AR case of camera overlap
                w, d, iv, m = w[jj], d[jj], iv[jj], m[jj]
                jj = np.where(w[1:] - w[:-1] > 10)[0]
                for j in jj:
                    d[j], iv[j], m[j] = np.nan, np.nan, np.nan
                obs_em_waves = (1. + zspecs[i]) * get_rf_em_waves(emname)
                ax.plot(w, d, c="k", lw=0.5, alpha=1.0, label="data")
                ax.fill_between(w, d - iv ** (-0.5), d + iv ** (-0.5), color="k", alpha=0.3)
                ax.plot(w, m, c="r", lw=0.5, alpha=1.0, label="model")
                ax.axhline(emdict[emname]["CONT"][i], c="r", lw=0.5, ls="--", alpha=1.0)
                for obs_em_wave in obs_em_waves:
                    ax.axvline(obs_em_wave, c="r", lw=0.5, ls="--", alpha=1.0)
                fs = 8
                if targetids is not None:
                    ax.text(0.05, 0.85, "{}".format(targetids[i]), fontsize=fs, transform=ax.transAxes)
                ax.text(0.05, 0.75, "Z={:.3f}".format(zspecs[i]), fontsize=fs, transform=ax.transAxes)
                ax.text(0.05, 0.05, "SNR({})={:.1f}".format(emname, emdict[emname]["FLUX"][i] * emdict[emname]["FLUX_IVAR"][i] ** 0.5), fontsize=fs, transform=ax.transAxes)
                if objtypes is not None:
                    ax.text(0.95, 0.85, "OBJTYPE={}".format(objtypes[i]), fontsize=fs, ha="right", transform=ax.transAxes)
                if spectypes is not None:
                    ax.text(0.95, 0.75, "SPECTYPE={}".format(spectypes[i]), fontsize=fs, ha="right", transform=ax.transAxes)
                if deltachi2s is not None:
                    ax.text(0.95, 0.05, "log10(DELTACHI2)={:.1f}".format(np.log10(deltachi2s[i])), fontsize=fs, ha="right", transform=ax.transAxes)
                ax.grid(True)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_ylim(ylim)
            if (ix % nrow == nrow - 1) | (i == ii[-1]):
                pdf.savefig(fig, bbox_inches="tight")
                plt.close()
