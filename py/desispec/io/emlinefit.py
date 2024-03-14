"""
desispec.io.emlinefit
=====================

IO routines for desi_emlinefit_afterburner.
"""

import os
from astropy.table import Table
import fitsio
import numpy as np
from desitarget.geomask import match_to
from desitarget.targets import main_cmx_or_sv
from desiutil.dust import ext_odonnell
from desiutil.dust import ebv as dust_ebv
from desiutil.log import get_logger
from desispec.emlinefit import get_rf_em_waves
from .util import checkgzip

def get_targetids(d, bitnames, log=None):
    """
    Returns the TARGETIDs passing the bitnames for the CMX_TARGET, SV{1,2,3}_DESI_TARGET, or DESI_TARGET mask.

    Args:
        d: structured array, typically a FIBERMAP catalog,
            with TARGETID and the CMX_TARGET, SV{1,2,3}_DESI_TARGET, or DESI_TARGET column.
        bitnames: comma-separated list of target bitnames to fit from the ``*DESI_TARGET`` mask (string)
        log (optional, defaults to get_logger()): Logger object

    Returns:
        targetids: list of TARGETIDs passing the {CMX,*DESI}_TARGET mask (np.array)

    Note:
        * If several bitnames are provided, selects the union of those.
        * Safer if d is already trimmed on unique TARGETIDs (see FIBERMAP format with ``zbest-*.fits`` files)
    """
    # AR log
    if log is None:
        log = get_logger()

    # AR get the *DESI_TARGET column + mask
    keys, masks, survey = main_cmx_or_sv(d)
    dtkey = keys[0]
    mask = masks[0]
    log.info("input catalog is identified as survey={}".format(survey))
    allowed_bitnames = mask.names()

    # AR read + select the targetids
    sel = np.zeros(len(d), dtype=bool)
    for bitname in bitnames.split(","):
        if bitname not in allowed_bitnames:
            msg = "{} not in allowed bitnames for {} ({})".format(
                    bitname, dtkey, allowed_bitnames
            )
            log.error(msg)
            raise ValueError(msg)
        sel |= (d[dtkey] & mask[bitname]) > 0
    targetids = d["TARGETID"][sel]
    log.info("selecting {} targets with {} in {}".format(sel.sum(), bitnames, dtkey))
    return targetids


def read_emlines_inputs(
    redrock,
    coadd,
    mwext_corr=True,
    rv=3.1,
    bitnames="ALL",
    targetids=None,
    rr_keys="TARGETID,Z,ZWARN,SPECTYPE,DELTACHI2",
    fm_keys="TARGET_RA,TARGET_DEC,OBJTYPE",
    ebvmax=2.,
    log=None,
):
    """
    Read the columns and spectra information (waves, fluxes, ivars) from the redrock and coadd files,
    which will go as input to get_emlines().

    Args:
        redrock: full path to a redrock/zbest file
        coadd: full path to a coadd file (everest-format)
        mwext_corr (optional, defaults to True): correct flux for foreground MW extinction? (boolean)
        rv (optional, defaults to 3.1): value of R_V, used if mwext_corr=True (float)
        bitnames (optional, defaults to "ALL", meaning fitting all fibers): comma-separated list of target bitnames to fit from the ``*DESI_TARGET`` mask (string)
        targetids (optional, defaults to None): list of TARGETIDs to restrict to (int, list, or numpy array)
        rr_keys (optional, defaults to "TARGETID,Z,ZWARN,SPECTYPE,DELTACHI2"): comma-separated list of columns from REDSHIFTS to propagate (string)
        fm_keys (optional, defaults to "TARGET_RA,TARGET_DEC,OBJTYPE"): comma-separated list of columns from FIBERMAP to propagate (string)
        ebvmax (optional, defaults to 2): spectra with ebv >= ebvmax will be masked (ivar=0 for all pixels) (float)
        log (optional, defaults to get_logger()): Logger object

    Returns:
        rr: structured array with the desired redrock columns for the desired TARGETIDs
        fm: structured array with the desired fibermap columns for the desired TARGETIDs
        waves: wavelengths (numpy array of shape (Nwave))
        fluxes: Galactic-extinction-corrected fluxes (numpy array of shape (Nspec, Nwave))
        ivars: inverse variances (numpy array of shape (Nspec, Nwave))

    Note:
        * We add TARGETID and Z to rr_keys if TARGETID not present in rr_keys nor in fm_keys.
        * If keys in rr_keys or fm_keys are not present in the redrock, those will be ignored.
        * If both bitnames and targetids are provided, we take the overlap of the two.
        * Mar. 2024: addition of ebvmax=2. argument, to avoid too long computing time
            on some backup tiles (https://github.com/desihub/desispec/issues/2186)
    """
    # AR log
    if log is None:
        log = get_logger()

    redrock = checkgzip(redrock)
    coadd = checkgzip(coadd)

    # AR targetids to np.array()
    if targetids is not None:
        if isinstance(targetids, list):
            targetids = np.array(targetids)
            log.info("convert targetids from list to np.array()")
        if isinstance(targetids, int):
            targetids = np.array([targetids])
            log.info("convert targetids from int to np.array()")

    # AR sanity checks
    for fn in [redrock, coadd]:
        if not os.path.isfile(fn):
            msg = "no {} file".format(fn)
            log.error(msg)
            raise FileNotFoundError(msg)
    keys = [key for key in rr_keys.split(",") if key in fm_keys.split(",")]
    if len(keys) > 0:
        msg = "the following columns are both present in rr_keys and fm_keys: {}".format(",".join(keys))
        log.error(msg)
        raise RuntimeError(msg)
    # AR grab TARGETID from the REDSHIFTS/ZBEST extension
    if "TARGETID" in fm_keys.split(","):
        log.info("removing TARGETID from fm_keys")
        fm_keys = ",".join([key for key in fm_keys.split(",") if key != "TARGETID"])
    for key in ["TARGETID", "Z"]:
        if key not in rr_keys.split(","):
            log.info("adding {} to rr_keys".format(key))
            rr_keys = "{},{}".format(key, rr_keys)

    # AR redrock: reading the correct extension
    with fitsio.FITS(redrock) as h:
        extnames = [h[i].get_extname() for i in range(len(h))]
        if "REDSHIFTS" in extnames:
            rr_extname = "REDSHIFTS"
        elif "ZBEST" in extnames:
            rr_extname = "ZBEST"
        else:
            msg = "{} has neither REDSHIFTS or ZBEST extension".format(redrock)
            log.error(msg)
            raise RuntimeError(msg)
        rr = Table(h[rr_extname].read())

    # AR coadd: reading fibermap + waves/fluxes/ivars
    # AR    fibermap has 500 rows (even in pre-everest)
    co = {}
    with fitsio.FITS(coadd) as h:
        fm = Table(h["FIBERMAP"].read())
        # AR available cameras
        cameras = []
        extnames = [h[i].get_extname() for i in range(len(h))]
        for camera in ["B", "R", "Z"]:
            if "{}_FLUX".format(camera) in extnames:
                cameras.append(camera)
        for camera in cameras:
            for key in [
                "{}_WAVELENGTH".format(camera),
                "{}_FLUX".format(camera),
                "{}_IVAR".format(camera),
            ]:
                co[key] = h[key].read()

    # AR selecting targetids
    if bitnames == "ALL":
        bit_targetids = rr["TARGETID"]
    else:
        bit_targetids = get_targetids(fm, bitnames, log=log)
    if targetids is None:
        targetids = bit_targetids
    else:
        targetids = targetids[np.in1d(targetids, bit_targetids)]
    nspec = len(targetids)
    log.info("Dealing with {} spectra".format(nspec))

    # AR targetids: cutting on redrock
    ii_rr = match_to(rr["TARGETID"], targetids)
    rr = rr[ii_rr]
    if len(rr) != nspec:
        msg = "{} TARGETIDs are not in {}".format(nspec - len(rr), redrock)
        log.error(msg)
        raise RuntimeError(msg)
    # AR targetids: cutting on fibermap + fluxes/ivars
    ii_co = match_to(fm["TARGETID"], targetids)
    if ii_co.size != nspec:
        msg = "{} TARGETIDs are not in {}".format(nspec - ii_co.size, coadd)
        log.error(msg)
        raise RuntimeError(msg)
    fm = fm[ii_co]
    for camera in cameras:
        for key in ["{}_FLUX".format(camera), "{}_IVAR".format(camera)]:
            co[key] = co[key][ii_co, :]
    # AR targetids: sanity check
    msg = None
    if len(rr) != len(fm):
        msg = "issue when selecting on targetids: len(rr) = {} != len(fm) = {}".format(
            len(rr), len(fm),
        )
    else:
        tmpn = (rr["TARGETID"] != fm["TARGETID"]).sum()
        if tmpn > 0:
            msg = "issue when selecting on targetids: {} mismatches between rr and fm".format(tmpn)
    if msg is not None:
        log.error(msg)
        raise RuntimeError(msg)

    # AR EBV
    if mwext_corr:
        ebvs = dust_ebv(fm["TARGET_RA"], fm["TARGET_DEC"])

    # AR rr_keys: restrict to existing ones
    rmv_rr_keys = [key for key in rr_keys.split(",") if key not in rr.dtype.names]
    if len(rmv_rr_keys) > 0:
        log.info("{} removed from rr_keys, as not present in {}".format(",".join(rmv_rr_keys), rr_extname))
    rr_keys = [key for key in rr_keys.split(",") if key not in rmv_rr_keys]
    rr = rr[rr_keys]
    # AR fm_keys: restrict to existing ones
    rmv_fm_keys = [key for key in fm_keys.split(",") if key not in fm.dtype.names]
    if len(rmv_fm_keys) > 0:
        log.info("{} removed from fm_keys, as not present in FIBERMAP".format(",".join(rmv_fm_keys)))
    fm_keys = [key for key in fm_keys.split(",") if key not in rmv_fm_keys]
    fm = fm[fm_keys]

    # AR b : 3600,5800  , r : 5760,7620 , z : 7520,9824
    # AR we keep all data points (i.e. including overlaps)
    # AR note that waves is not ordered, as a consequence
    waves = np.zeros(0)
    fluxes = np.zeros(0).reshape(nspec, 0)
    ivars = np.zeros(0).reshape(nspec, 0)
    for camera in cameras:
        tmpw = co["{}_WAVELENGTH".format(camera)]
        tmpfl = co["{}_FLUX".format(camera)]
        tmpiv = co["{}_IVAR".format(camera)]
        if mwext_corr:
            # AR TBD: use a smarter way with no loop on tids...
            # AR TBD: but as 500 rows at most, ~ok
            tmpexts = ext_odonnell(tmpw, Rv=rv)
            for i in range(nspec):
                tmp_mw_trans = 10 ** (-0.4 * ebvs[i] * rv * tmpexts)
                tmpfl[i, :] /= tmp_mw_trans
                tmpiv[i, :] *= tmp_mw_trans ** 2
        waves = np.append(waves, tmpw)
        fluxes = np.append(fluxes, tmpfl, axis=1)
        ivars = np.append(ivars, tmpiv, axis=1)

    # AR set ivar=0 for ebv >= ebvmax spectra
    sel = ebvs >= ebvmax
    ivars[sel, :] = 0
    for tid, ebv in zip(rr["TARGETID"][sel], ebvs[sel]):
        log.warning(
            "set ivar=0 for all pixels of TARGETID={} (EBV={:.2f} >= {})".format(
                tid, ebv, ebvmax,
            )
        )
    log.info(
        "we set ivar=0 for {}/{} spectra with EBV >= {}".format(
            sel.sum(), nspec, ebvmax
        )
    )

    return rr, fm, waves, fluxes, ivars


def write_emlines(
    output,
    emdict,
    rr=None,
    fm=None,
    redrock=None,
    coadd=None,
    rf_fit_hw=None,
    min_rf_fit_hw=None,
    rf_cont_w=None,
    rv=None,
    log=None,
):
    """
    Writes the emission line fitting to a file.

    Args:
        output: output fits file (string)
        emdict: dictionary output by get_emlines()
        rr (optional, defaults to None): redrock REDSHIFTS/ZBEST extension
        fm (optional, defaults to None): redrock FIBERMAP extension
        redrock (optional, defaults to None): used redrock/zbest file
        coadd (optional, defaults to None): used coadd file (everest-format)
        rf_fit_hw (optional, defaults to None): *rest-frame* wavelength width (in A) used for fitting on each side of the line (float)
        min_rf_fit_hw (optional, defaults to None): minimum requested *rest-frame* width (in A) on each side of the line to consider the fitting (float)
        rf_cont_w (optional, defaults to None): *rest-frame* wavelength extent (in A) to fit the continuum (float)
        rv (optional, defaults to None): value of R_V (float)
        log (optional, defaults to get_logger()): Logger object
    """
    # AR log
    if log is None:
        log = get_logger()

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
    if redrock is not None:
        hdr["RRFN"] = redrock
    if coadd is not None:
        hdr["COADDFN"] = coadd
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
    if os.path.isfile(output):
        log.info("Removing existing {}".format(output))
        os.remove(output)
    fd = fitsio.FITS(output, "rw")
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
        emnames (optional, defaults to ["OII", "HDELTA", "HGAMMA", "HBETA", "OIII", "HALPHA"]): list of plotted lines (list of string)
        targetids (optional, defaults to None): list of TARGETIDs (numpy array)
        objtypes (optional, defaults to None): list of fibermap OBJTYPEs (numpy array)
        spectypes (optional, defaults to None): list of redrock SPECTYPEs (numpy array)
        deltachi2s (optional, defaults to None): list of redrock DELTACHI2s (numpy array)
        ylim (optional, defaults to (None, None)): ylim for plotting (float doublet)
        nrow (optional, defaults to 10): number of rows, i.e. galaxy, per pdf page (int)
        rowsort_byz (optional, defaults to True): the spectra are displayed by increasing redshifts (boolean)

    Note:
        * For `emdict`:

            - Each emission line has its own subdictionary.
            - Each emission line dictionary should contain these keys:

                * CONT : continuum in ``1e-17 * erg/cm2/s/A``
                * FLUX, FLUX_IVAR: flux in ``1e-17 * erg/cm2/s/A``
                * waves, fluxes, ivars, models: data used for fitting + fitted model

        * If some emname is not present in emdict, it will be discarded.
        * Each spectra will be one row of plot, each emission line corresponds to a column.
        * If rowsort_byz=False, then the spectra will be displayed by order of appearance.
    """
    # AR plot imports
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

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
                fig = plt.figure(figsize=(25, 15))
                gs = gridspec.GridSpec(nrow, len(emnames), wspace=0.1, hspace=0.1)
            for i_emname, emname in enumerate(emnames):
                ax = plt.subplot(gs[ix % nrow, i_emname])
                w = emdict[emname]["waves"][i]
                d = emdict[emname]["fluxes"][i]
                iv = emdict[emname]["ivars"][i]
                m = emdict[emname]["models"][i]
                # # AR case of camera overlap
                jj = w.argsort()
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
