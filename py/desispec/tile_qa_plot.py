"""
desispec.tile_qa
============
Utility functions to generate the per-cumulative-tile QA png.
"""

import os
import sys
import subprocess
from pkg_resources import resource_filename
import yaml
from glob import glob
import tempfile
from desitarget.targetmask import desi_mask, bgs_mask
from desispec.maskbits import fibermask
from desispec.io import read_fibermap
from desispec.tsnr import tsnr2_to_efftime
from astropy.table import Table
from astropy.io import fits
import fitsio
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
import matplotlib.image as mpimg
from desiutil.dust import ebv as dust_ebv
from astropy import units
from astropy.coordinates import SkyCoord


# AR tile radius in degrees
tile_radius_deg = 1.628


def get_qa_config():
    """
    Reads the configuration file (data/qa/qa-params.yaml)

    Args:
        None.

    Returns:
        Content of the qa-params.yaml file
    """
    fn = resource_filename("desispec", "data/qa/qa-params.yaml")
    f = open(fn, "r")
    config = yaml.safe_load(f)
    f.close()
    return config


def get_qa_badmsks():
    """
    Returns the bitmask values for bad_qafstatus_mask and bad_petal_mask..

    Args:
        None

    Returns:
        The bitmask values for bad_qafstatus_mask and bad_petal_mask.
    """
    config = get_qa_config()
    badqa_names = config["exposure_qa"]["bad_qafstatus_mask"]
    badqa_val = fibermask.mask(badqa_names)
    badpet_names = config["exposure_qa"]["bad_petal_mask"]
    badpet_val = fibermask.mask(badpet_names)
    return badqa_val, badpet_val


def assert_tracer(tracer):
    """
    Asserts tracer in the list of allowed tracers.

    Args:
        tracer: tracer name (string, upper case)

    Note:
        Will exit with an error if not asserted.
    """
    config = get_qa_config()
    all_tracers = list(config["tile_qa_plot"]["tracers"].keys())
    if tracer not in all_tracers:
        sys.exit(
            "Wrong tracer ({}); should be from {}; exiting".format(
                tracer, ", ".join(all_tracers)
            )
        )


def get_tracer_zminmax(tracer):
    """
    Returns some fiducial redshift range per tracer to compute basic stats.

    Args:
        tracer: tracer name (see get_tracer_names()) (string)

    Returns:
        zmin: minimum redshift (float)
        zmax: minimum redshift (float)
    """
    assert_tracer(tracer)
    config = get_qa_config()
    zmin = config["tile_qa_plot"]["tracers"][tracer]["zmin"]
    zmax = config["tile_qa_plot"]["tracers"][tracer]["zmax"]
    return zmin, zmax


def get_zbins():
    """
    Redshift grid for the tile QA n(z)

    Args:
        None

    Returns:
        The redshift grid (float array)
    """
    config = get_qa_config()
    zmin = config["tile_qa_plot"]["bins"]["zmin"]
    zmax = config["tile_qa_plot"]["bins"]["zmax"]
    dz = config["tile_qa_plot"]["bins"]["dz"]
    return np.arange(zmin, zmax, dz).round(2)


def get_tracer(tracer, d, fstatus_key="QAFIBERSTATUS"):
    """
    For a given tracer, returns the selection used for the tile QA n(z):
        - (fstatus_key & bad_qafstatus_mask) == 0
        - desi_mask or bgs_mask
        - if ELG_LOP: additional cut on (d["DESI_TARGET"] & desi_mask["QSO"]) == 0

    Args:
        tracer: "BGS_BRIGHT", "BGS_FAINT", "LRG", "ELG_LOP", or "QSO" (string)
        d: structured array with at least FIBERSTATUS, DESI_TARGET, BGS_TARGET
        fstatus_key (optional, defaults to QAFIBERSTATUS): key to use as FIBERSTATUS (string)

    Returns:
        sel: selected tracer sample (boolean array)
    """
    assert_tracer(tracer)
    badqa_val, _ = get_qa_badmsks()
    sel = (d[fstatus_key] & badqa_val) == 0
    if tracer in ["BGS_BRIGHT", "BGS_FAINT"]:
        sel &= (d["BGS_TARGET"] & bgs_mask[tracer]) > 0
    else:
        sel &= (d["DESI_TARGET"] & desi_mask[tracer]) > 0
    # AR ELG_LOP : excluding ELG_LOP x QSO
    if tracer == "ELG_LOP":
        sel &= (d["DESI_TARGET"] & desi_mask["QSO"]) == 0
    return sel


def get_tracer_zok(tracer, dchi2_min, d, fstatus_key="QAFIBERSTATUS"):
    """
    For a given tracer, returns the spectro. valid sample for the tile QA n(z):
        - (fstatus_key & bad_qafstatus_mask) == 0
        - DELTACHI2 > dchi2_min
        - if QSO: additional cut on SPECTYPE="QSO"

    Args:
        tracer: "BGS_BRIGHT", "BGS_FAINT", "LRG", "ELG_LOP", or "QSO" (string)
        dchi2_min: minimum DELTACHI2 value (read from YAML file)
        d: structured array with at least DELTACHI2, SPECTYPE
        fstatus_key (optional, defaults to QAFIBERSTATUS): key to use as FIBERSTATUS (string)

    Returns:
        sel: spectro. valid sample (boolean array)
    """
    assert_tracer(tracer)
    badqa_val, _ = get_qa_badmsks()
    sel = (d[fstatus_key] & badqa_val) == 0
    sel &= d["DELTACHI2"] > dchi2_min
    if tracer == "QSO":
        sel &= d["SPECTYPE"] == "QSO"
    return sel


def get_zhists(
    tileids, tracer, dchi2_min, d, fstatus_key="QAFIBERSTATUS", tileid_key=None
):
    """
    Returns the fractional, per tileid, n(z) for a given tracer.

    Args:
        tileids: int or list or numpy array of int
        tracer: one of the tracers defined in get_tracer_names() (string)
        dchi2_min: DELTACHI2 threshold for a valid redshift
        d: structured array with at least FIBERSTATUS, DESI_TARGET, BGS_TARGET, DELTACHI2, SPECTYPE, Z
        fstatus_key (optional, defaults to QAFIBERSTATUS): key to use as FIBERSTATUS (string)
        tileid_key (optional, defaults to None): column name for TILEID (string)

    Returns:
        bins: the redshift bin grid (float array)
        zhists: fractional, per tileid, n(z) (numpy array of shape
                (nbin) if tileids is int or has length=1
                (nbin, len(tileids)) else.

    Notes:
        If tileid_key is not provided, assumes all spectra are from the same tile.
        If tileid_key is provided, will identify the tileids with np.unique(d[tileid_key]).
    """
    assert_tracer(tracer)

    # AR zbins
    bins = get_zbins()
    nbin = len(bins) - 1

    # AR restricting to valid fibers + selecting tracer
    istracer = get_tracer(tracer, d, fstatus_key=fstatus_key)

    # AR making hist for each TILEID
    if tileid_key is None:
        parent = istracer
        zok = (parent) & (get_tracer_zok(tracer, dchi2_min, d, fstatus_key=fstatus_key))
        hist = np.histogram(d["Z"][(parent) & (zok)], bins=bins, density=False)[0]
        zhists = hist / parent.sum()
    else:
        tileids = np.unique(d[tileid_key])
        zhists = np.zeros((nbin, len(tileids)), dtype=float)
        for i in range(len(tileids)):
            parent = (istracer) & (d[tileid_key] == tileids[i])
            zok = (parent) & (
                get_tracer_zok(tracer, dchi2_min, d, fstatus_key=fstatus_key)
            )
            hist = np.histogram(d["Z"][(parent) & (zok)], bins=bins, density=False)[0]
            zhists[:, i] = hist / parent.sum()

    return bins, zhists


def get_viewer_cutout(
    tileid,
    tilera,
    tiledec,
    tmpoutdir=tempfile.mkdtemp(),
    width_deg=4,
    pixscale=10,
    dr="dr9",
    timeout=15,
):
    """
    Downloads a cutout of the tile region from legacysurvey.org/viewer.

    Args:
        tileid: TILEID (int)
        tilera: tile center R.A. (float)
        tiledec: tile center Dec. (float)
        tmpoutdir (optional, defaults to a temporary directory): temporary directory where
        width_deg (optional, defaults to 4): width of the cutout in degrees (float)
        pixscale (optional, defaults to 10): pixel scale of the cutout
        dr (optional, default do "dr9"): imaging data release
        timeout (optional, defaults to 15): time (in seconds) after which we quit the wget call (int)

    Returns:
        img: output of mpimg.imread() reading of the cutout (np.array of floats)

    Notes:
        Duplicating fiberassign.fba_launch_io.get_viewer_cutout()
    """
    # AR cutout
    tmpfn = "{}tmp-{}.jpeg".format(tmpoutdir, tileid)
    size = int(width_deg * 3600.0 / pixscale)
    layer = "ls-{}".format(dr)
    tmpstr = 'timeout {} wget -q -o /dev/null -O {} "http://legacysurvey.org/viewer-dev/jpeg-cutout/?layer={}&ra={:.5f}&dec={:.5f}&pixscale={:.0f}&size={:.0f}"'.format(
        timeout, tmpfn, layer, tilera, tiledec, pixscale, size
    )
    try:
        subprocess.check_call(tmpstr, stderr=subprocess.DEVNULL, shell=True)
    except subprocess.CalledProcessError:
        print("no cutout from viewer after {}s, stopping the wget call".format(timeout))
    try:
        img = mpimg.imread(tmpfn)
    except:
        img = np.zeros((size, size, 3))
    if os.path.isfile(tmpfn):
        os.remove(tmpfn)
    return img


def deg2pix(dras, ddecs, width_deg, width_pix):
    """
    Converts (dras,ddecs) to (xs,ys) in cutout img pixels.

    Args:
        dras: projected distance (degrees) along R.A. to the center of the cutout (np.array of floats)
        ddecs: projected distance (degrees) along Dec. to the center of the cutout (np.array of floats)
        width_deg: width of the cutout in degrees (np.array of floats)
        width_pix: width of the cutout in pixels (np.array of floats)

    Returns:
        dxs: distance (pixels) along x to the center of the cutout (np.array of floats)
        dys: distance (pixels) along y to the center of the cutout (np.array of floats)

    Notes:
        not sure at the <1 pixel level...
        Duplicated from fiberassign.fba_launch_io.deg2pix()
    """
    dxs = width_pix * (0.5 - dras / width_deg)
    dys = width_pix * (0.5 + ddecs / width_deg)
    return dxs, dys


def plot_cutout(ax, tileid, tilera, tiledec, width_deg, petal_c="w", ebv_c="orange"):
    """
    Plots a ls-dr9 cutout, with overlaying the petals and the EBV contours.

    Args:
        ax: pyplot object
        tileid: TILEID (int)
        tilera: tile center R.A. (float)
        tiledec: tile center Dec. (float)
        width_deg: width of the cutout in degrees (np.array of floats)
        petal_c (optional, defaults to "w"): color used to display petals (string)
        ebv_c (optional, default to "y"): color used to display the EBV contours (string)

    Notes:
        Different than fiberassign.fba_launch_io.plot_cutout().
    """
    # AR get the cutout
    img = get_viewer_cutout(
        tileid, tilera, tiledec, width_deg=width_deg, pixscale=10, dr="dr9", timeout=15,
    )

    # AR display cutout
    width_pix = img.shape[0]
    ax.imshow(
        img,
        origin="upper",
        zorder=0,
        extent=[0, width_pix, 0, width_pix],
        aspect="equal",
    )

    # AR display petals
    for ang, p in zip(np.linspace(2 * np.pi, 0, 11), [7, 8, 9, 0, 1, 2, 3, 4, 5, 6]):
        dxs, dys = deg2pix(
            np.array([0, tile_radius_deg * np.cos(ang)]),
            np.array([0, tile_radius_deg * np.sin(ang)]),
            width_deg,
            width_pix,
        )
        ax.plot(
            dxs, dys, c=petal_c, lw=0.25, alpha=1.0, zorder=1,
        )
        anglab = ang + 0.1 * np.pi
        dxs, dys = deg2pix(
            1.1 * tile_radius_deg * np.cos(anglab),
            1.1 * tile_radius_deg * np.sin(anglab),
            width_deg,
            width_pix,
        )
        ax.text(
            dxs, dys, "{:.0f}".format(p), color=petal_c, va="center", ha="center",
        )

    # AR display outer edge
    ang = np.linspace(2 * np.pi, 0, 1000)
    dxs, dys = deg2pix(
        tile_radius_deg * np.cos(ang),
        tile_radius_deg * np.sin(ang),
        width_deg,
        width_pix,
    )
    ax.plot(
        dxs, dys, c=petal_c, lw=0.25, alpha=1.0, zorder=1,
    )

    # AR
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, width_pix + 0.5)
    ax.set_ylim(-0.5, width_pix + 0.5)
    ax.axis("off")

    # AR EBV contours
    # AR the cutout is centered on tile R.A., Dec.
    # AR    with width_deg
    # AR first make a regular grid in (R.A, Dec.)
    # AR note: would be nicer to start from a (dra, ddec) grid,
    # AR    and use cs.spherical_offsets_by, but this exists
    # AR    in astropy/4.3.1, and current version is astropy/4.0.1..
    npts = 100
    ramin = tilera - width_deg / 2. / np.cos(np.radians(tiledec))
    ramax = tilera + width_deg / 2. / np.cos(np.radians(tiledec))
    decmin = tiledec - width_deg / 2.
    decmax = tiledec + width_deg / 2.
    xs = np.linspace(ramin, ramax, npts)
    ys = np.linspace(decmin, decmax, npts)
    grid_ras, grid_decs = np.meshgrid(xs, ys)
    # AR get EBV
    grid_ebvs = dust_ebv(grid_ras, grid_decs)
    # AR get projected distance (in degree) to tile center
    cs = SkyCoord(grid_ras * units.degree, grid_decs * units.degree, frame="icrs")
    tile_cs = SkyCoord(tilera * units.degree, tiledec * units.degree, frame="icrs")
    dras, ddecs = cs.spherical_offsets_to(tile_cs)
    dras, ddecs = dras.to(units.degree).value, ddecs.to(units.degree).value
    # AR not exactly sure why this is needed for ddecs, but it works..
    dras, ddecs = -dras, -ddecs
    # AR convert to cutout pixel coordinates
    dxs, dys = deg2pix(dras, ddecs, width_deg, width_pix)
    # AR plot contours
    cnt = ax.contour(dxs, dys, grid_ebvs, levels=3, colors=ebv_c, linewidths=0.5, zorder=1)
    ax.clabel(cnt, inline=1, fontsize=10)
    ax.text(0.5, -0.05, "Contours = EBV map", ha="center", fontsize=10, transform=ax.transAxes)


def get_petalqa_props(key):
    """
    For a given key, returns specific properties, to display diagnoses on the QA plot.

    Args:
        key: column name present in the PETALQA extension of qa-tile-TILEID-NIGHT.fits (string)

    Returns:
        short: shortname of the key (string)
        precision: number of digits to report (int)
        okmin: minimum value to be considered as ok (float)
        okmax: maximum value to be considered as ok (float)
        combine: None, "sum", "mean", i.e. how to combine the stats for all petals (string)
    """
    config = get_qa_config()
    # AR set to None by default
    short, precision, okmin, okmax, combine = None, None, None, None, None
    # AR updating values
    if key == "PETAL_LOC":
        short, precision = "PETAL", 0
    if key == "WORSTREADNOISE":
        short, precision, okmax, combine = (
            "RDN",
            1,
            config["exposure_qa"]["max_readnoise"],
            "mean",
        )
    if key == "NGOODPOS":
        short, precision, okmin, combine = (
            "GOODPOS",
            0,
            500 * config["exposure_qa"]["max_frac_of_bad_positions_per_petal"],
            "sum",
        )
    if key == "NSTDSTAR":
        short, precision, okmin, combine = (
            "NSTD",
            0,
            config["exposure_qa"]["min_number_of_good_stdstars_per_petal"],
            "sum",
        )
    if key == "STARRMS":
        short, precision, okmax, combine = (
            "*RMS",
            3,
            config["exposure_qa"]["max_rms_of_rflux_ratio_of_stdstars"],
            "mean",
        )
    if key == "TSNR2FRA":
        short, precision, okmin, okmax, combine = (
            "TSNR2FRA",
            2,
            config["exposure_qa"]["tsnr2_petal_minfrac"],
            config["exposure_qa"]["tsnr2_petal_maxfrac"],
            "mean",
        )
    if key in ["BTHRUFRAC", "RTHRUFRAC", "ZTHRUFRAC", "THRUFRAC"]:
        short, precision, okmin, okmax = (
            key,
            2,
            config["tile_qa_plot"]["thrufrac_min"],
            config["tile_qa_plot"]["thrufrac_max"],
        )
        if key == "THRUFRAC":
            short = "THRUFRAC_X"
    if key in ["BSKYTHRURMS", "RSKYTHRURMS", "ZSKYTHRURMS"]:
        short, precision, okmax = key, 3, config["tile_qa_plot"]["skythrurms_max"]
    if key in ["BSKYCHI2PDF", "RSKYCHI2PDF", "ZSKYCHI2PDF"]:
        short, precision, okmax = key, 2, config["tile_qa_plot"]["skychi2pdf_max"]

    #if key == "BADPETAL" :
    #    config["tile_qa_plot"]

    # AR return
    return short, precision, okmin, okmax, combine


def print_petal_infos(ax, petalqa, fiberqa):
    """
    Print some diagnoses for each petal, and for the whole tile.

    Args:
        ax: pyplot object
        petalqa: the PETALQA extension data of the tile-qa-TILEID-NIGHT.fits file
    """

    show_bad_petal_cause=True

    # AR keys to compute stats
    keys = [
        "PETAL_LOC",
        "WORSTREADNOISE",
        "NGOODPOS",
        "NSTDSTAR",
        "STARRMS",
        "TSNR2FRA",
        "BTHRUFRAC",
        "RTHRUFRAC",
        "ZTHRUFRAC",
        "THRUFRAC",
        "BSKYTHRURMS",
        "RSKYTHRURMS",
        "ZSKYTHRURMS",
        "BSKYCHI2PDF",
        "RSKYCHI2PDF",
        "ZSKYCHI2PDF"
    ]
    # AR keys to display
    disp_keys = ["PETAL_LOC", "NGOODPOS", "NSTDSTAR", "STARRMS", "THRUFRAC"]

    # AR storing properties in a dictionary
    mydict = {}
    for key in keys:
        mydict[key] = {}
        (
            mydict[key]["short"],
            mydict[key]["precision"],
            mydict[key]["okmin"],
            mydict[key]["okmax"],
            mydict[key]["combine"],
        ) = get_petalqa_props(key)

    # AR to record {PETAL}-{KEY} where issue
    fails = []

    #
    y, dy = 0.95, -0.08
    x0, dx = 0.05, 0.20
    fs = 10
    x = x0
    # AR header
    for key in disp_keys:
        ax.text(
            x,
            y,
            mydict[key]["short"],
            fontsize=fs,
            ha="center",
            transform=ax.transAxes,
        )
        x += dx
    if show_bad_petal_cause :
        ax.text(x,y,"BAD")
        x += dx
    y += dy
    # AR stats per petal
    for i in range(10):
        x = x0
        for key in keys:
            # AR value for the considered petal
            # AR THRUFRAC: taking the value the furthest from 1
            if key == "THRUFRAC":
                tmpvals = np.array(
                    [
                        petalqa["BTHRUFRAC"][i],
                        petalqa["RTHRUFRAC"][i],
                        petalqa["ZTHRUFRAC"][i],
                    ]
                )
                val = tmpvals[np.argmax(np.abs(tmpvals - 1))]
            else:
                val = petalqa[key][i]
            # AR failure?
            fontweight, color = "normal", "k"
            isfail = False
            if mydict[key]["okmin"] is not None:
                if val < mydict[key]["okmin"]:
                    isfail = True
            if mydict[key]["okmax"] is not None:
                if val > mydict[key]["okmax"]:
                    isfail = True
            if isfail:
                fails.append(
                    "{}-{}={:.{}f}".format(petalqa["PETAL_LOC"][i], key, val, mydict[key]["precision"])
                )
                fontweight, color = "bold", "r"



            # AR print if in disp_keys
            if key in disp_keys:
                ax.text(
                    x,
                    y,
                    "{:.{}f}".format(val, mydict[key]["precision"]),
                    color=color,
                    fontsize=fs,
                    fontweight=fontweight,
                    ha="center",
                    transform=ax.transAxes,
                )
                x += dx

        if show_bad_petal_cause :
            val=""
            for cause in ["BADPETALPOS","BADPETALSKY","BADPETALSTDSTAR","BADPETALFLUXCAL","BADPETALSNR","BADREADNOISE"] :
                n1 = np.sum(fiberqa["PETAL_LOC"]==i)
                n2 = np.sum((fiberqa["PETAL_LOC"]==i)&(fiberqa["QAFIBERSTATUS"]&fibermask.mask(cause)>0))
                cause = cause.replace("BADPETAL","")
                if n2==n1 :
                    if val == "" :
                        val=cause
                    else :
                        val=val+"&"+cause
            ax.text(x,y,val,color="red",fontsize="small")
            x += dx

        y += dy

    # AR stats for all petals
    x = x0
    for key in disp_keys:
        txt = "-"
        if key == "PETAL_LOC":
            txt = "ALL"
        if mydict[key]["combine"] is not None:
            if mydict[key]["combine"] == "sum":
                txt = "{:.{}f}".format(petalqa[key].sum(), mydict[key]["precision"])
            if mydict[key]["combine"] == "mean":
                txt = "{:.{}f}".format(petalqa[key].mean(), mydict[key]["precision"])
        ax.text(
            x,
            y,
            txt,
            color="b",
            fontsize=fs,
            fontweight="normal",
            ha="center",
            transform=ax.transAxes,
        )
        x += dx
    y += 2 * dy
    # AR failed cases
    if len(fails)>0 :
        ax.text(
            x0,
            y,
            "Alert: {}".format(", ".join(fails)),
            color="r",
            fontsize=fs,
            #fontweight="bold",
            ha="left",
            transform=ax.transAxes,
        )

    ax.axis("off")


def set_mwd(ax, org=120):
    """
    Prepare a plot for a Mollweide map

    Args:
        ax: a plt.subplots() instance
        org: R.A. at the center of the plot (default=120) (float)
    """
    # org is the origin of the plot, 0 or a multiple of 30 degrees in [0,360).
    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels + 360 + org, 360)
    ax.set_xticklabels(tick_labels)  # we add the scale on the x axis
    ax.set_xlabel("R.A [deg]")
    ax.xaxis.label.set_fontsize(12)
    ax.set_ylabel("Dec. [deg]")
    ax.yaxis.label.set_fontsize(12)
    ax.grid(True)


def get_radec_mw(ras, decs, org):
    """
    Converts R.A. and Dec. coordinates to be used in a Mollweide projection.

    Args:
        ras: R.A. coordinates in degrees (float array)
        decs: Dec. coordinates in degrees (float array)

    Returns:
        Two float arrays with transformed R.A. and Dec.
    """
    ras = np.remainder(ras + 360 - org, 360)  # shift ra values
    ras[ras > 180] -= 360  # scale conversion to [-180, 180]
    ras = -ras  # reverse the scale: East to the left
    return np.radians(ras), np.radians(decs)


def plot_mw_skymap(fig, ax, tileid, tilera, tiledec, survey, program, org=120):
    """
    Plots the target density sky map with a Mollweide projection, with highlighting the tile position.

    Args:
        fig: pyplot figure object
        ax: pyplot object with projection="mollweide"
        tileid: TILEID (int)
        tilera: tile R.A. in degrees (float)
        tiledec: tile Dec. in degrees (float)
        survey: usually "main", "sv1", "sv3" (string)
        program: "dark" or "bright" (string)
        org: R.A. at the center of the plot (default=120) (float)

    Note:
        If survey is not "main" and program is not "bright" or "dark",
            will color-code EBV, not the target density.
    """
    # AR reading the pixweight file
    pixwfn = "{}/target/catalogs/dr9/1.1.1/pixweight/main/resolve/{}/pixweight-1-{}.fits".format(
        os.getenv("DESI_ROOT"), program, program
    )
    if not os.path.isfile(pixwfn) :
        print("use dark pixweight map")
        tprogram="dark"
        pixwfn = "{}/target/catalogs/dr9/1.1.1/pixweight/main/resolve/{}/pixweight-1-{}.fits".format(
            os.getenv("DESI_ROOT"), tprogram, tprogram
        )

    hdr = fits.getheader(pixwfn, 1)
    nside, nest = hdr["HPXNSIDE"], hdr["HPXNEST"]
    pixwd = fits.open(pixwfn)[1].data
    npix = hp.nside2npix(nside)
    thetas, phis = hp.pix2ang(nside, np.arange(npix), nest=nest)
    ras, decs = np.degrees(phis), 90.0 - np.degrees(thetas)
    # AR plotting skymap
    set_mwd(ax, org)
    # AR dr9
    ramws, decmws = get_radec_mw(ras, decs, org)
    sel = pixwd["FRACAREA"] > 0
    if (survey == "main") & (program in ["bright", "dark"]):
        dens_med = np.median(pixwd["ALL"][sel])
        clim = (0.5, 1.5)
        if program == "dark":
            clim = (0.75, 1.25)
        c = pixwd["ALL"] / dens_med
        clabel = "{} targets".format(program, dens_med)
    else:
        clim = (0, 0.1)
        c = pixwd["EBV"]
        clabel = "EBV"
    sc = ax.scatter(
        ramws[sel],
        decmws[sel],
        c=c[sel],
        s=1,
        alpha=0.5,
        zorder=0,
        cmap=matplotlib.cm.coolwarm,
        vmin=clim[0],
        vmax=clim[1],
        rasterized=True,
    )
    # AR cbar
    pos = ax.get_position().get_points().flatten()
    cax = fig.add_axes(
        [
            pos[0] + 0.1 * (pos[2] - pos[0]),
            pos[1] + 0.3 * (pos[3] - pos[1]),
            0.4 * (pos[2] - pos[0]),
            0.005,
        ]
    )
    cbar = plt.colorbar(
        sc, cax=cax, fraction=0.025, orientation="horizontal", extend="both"
    )
    cbar.mappable.set_clim(clim)
    cbar.set_label(clabel, fontweight="bold")
    # AR our tileid
    tmpra = np.remainder(tilera + 360 - org, 360)
    if tmpra > 180:
        tmpra -= 360
    if tmpra > 0:
        dra = -40
    else:
        dra = 40
    ramws, decmws = get_radec_mw(
        np.array([tilera, tilera + dra]), np.array([tiledec, tiledec - 30]), org
    )
    ax.scatter(
        ramws[0],
        decmws[0],
        edgecolors="k",
        facecolors="none",
        marker="o",
        s=50,
        zorder=1,
    )
    arrow_args = dict(color="k", width=1, headwidth=5, headlength=10)
    ax.annotate(
        "",
        xy=(ramws[0], decmws[0]),
        xytext=(ramws[1], decmws[1]),
        arrowprops=arrow_args,
    )


def get_expids_efftimes(tileqafits, prod):
    """
    Get the EFFTIME and EFFTIMEQA for the EXPIDs from the coadd.

    Args:
        tileqafits: path to the tile-qa-TILEID-NIGHT.fits file
        prod: full path to input reduction, e.g. /global/cfs/cdirs/desi/spectro/redux/daily (string)

    Returns:
        structured array with the following keys:
            EXPID, NIGHT, EFFTIME_SPEC, QA_EFFTIME_SPEC

    Notes:
        We work from the spectra-*fits files; if not present in the same folder
            as tileqafits, we look into the expected path using prod.
        As this is run *before* desi_tsnr_afterburner, we compute here the
            EFFTIME_SPEC values.
        If no GOALTYPE in tileqafits header, we default to dark.
        TBD: we purposely do not use TSNR2 keys from qa-params.yaml,
            as those do not handle the TSNR2_ELG->TSNR2_LRG change from
            2021 shutdown.
            We use:
            - dark before 20210901: TSNR2_ELG
            - dark after 20210901: TSNR2_LRG
            - bright: TSNR2_BGS
            - backup: TSNR2_BGS
            Method assessed against all Main exposures until 20211013 in daily tsnr-exposures.fits.
    """
    # AR GOALTYPE (defaulting to dark) + TSNR2 key
    goaltype = "dark"
    h = fits.open(tileqafits)
    hdr = fits.getheader(tileqafits, "FIBERQA")
    if "GOALTYPE" in [cards[0] for cards in hdr.cards]:
        goaltype = hdr["GOALTYPE"].lower()
    if goaltype in ["bright", "backup"]:
        tsnr2_key = "TSNR2_BGS"
    else:
        if hdr["LASTNITE"] < 20210921:
            tsnr2_key = "TSNR2_ELG"
        else:
            tsnr2_key = "TSNR2_LRG"

    # AR get list of exposures used for the tile
    # AR first try spectra*fits files in the same folder as tileqafits
    tmpstr = os.path.join(
        os.path.dirname(tileqafits),
        "spectra-*-{}-thru{}.fits".format(hdr["TILEID"], hdr["LASTNITE"]),
    )
    spectra_fns = sorted(glob(tmpstr))
    # AR then try based on prod
    if len(spectra_fns) == 0:
        tmpstr = os.path.join(
            prod,
            "tiles",
            "cumulative",
            "{}".format(hdr["TILEID"]),
            "{}".format(hdr["LASTNITE"]),
            "spectra-*-{}-thru{}.fits".format(hdr["TILEID"], hdr["LASTNITE"]),
        )
        spectra_fns = sorted(glob(tmpstr))
    if len(spectra_fns) > 0:
        fmap = read_fibermap(spectra_fns[0])
        expids, ii = np.unique(fmap["EXPID"], return_index=True)
        nights = fmap["NIGHT"][ii]
    # AR then try based on prod
    else:
        expids, nights = [], []
    nexp = len(expids)

    # AR looping on EXPIDS
    d = Table()
    d["EXPID"] = expids
    d["NIGHT"] = nights
    d["EFFTIME_SPEC"], d["QA_EFFTIME_SPEC"] = np.zeros(nexp), np.zeros(nexp)
    for i in range(nexp):
        # AR EFFTIME_SPEC, with looping on petals and cameras
        tsnr2_petals = np.zeros(10)
        for petal in range(10):
            for camera in ["b", "r", "z"]:
                tsnr2_key_cam = "{}_{}".format(tsnr2_key, camera.upper())
                fn = os.path.join(
                    prod,
                    "exposures",
                    "{}".format(nights[i]),
                    "{:08d}".format(expids[i]),
                    "cframe-{}{}-{:08d}.fits".format(camera, petal, expids[i]),
                )
                if os.path.isfile(fn):
                    vals = fitsio.read(fn, ext="SCORES", columns=[tsnr2_key_cam])[tsnr2_key_cam]
                    tsnr2_petals[petal] += np.median(vals[vals > 0])
        d["EFFTIME_SPEC"][i] = tsnr2_to_efftime(tsnr2_petals[tsnr2_petals > 0].mean(), tsnr2_key.split("_")[-1])
        # QA_EFFTIME_SPEC, reading exposure-qa*fits
        fn = os.path.join(
                            prod,
                            "exposures",
                            "{}".format(nights[i]),
                            "{:08d}".format(expids[i]),
                            "exposure-qa-{:08d}.fits".format(expids[i]),
        )
        if os.path.isfile(fn):
            d["QA_EFFTIME_SPEC"][i] = fits.getheader(fn, "FIBERQA")["EFFTIME"]

    return d


def make_tile_qa_plot(
    tileqafits,
    prod,
    pngoutfile=None,
    dchi2_min=None,
    tsnr2_key=None,
    refdir=resource_filename("desispec", "data/qa"),
):
    """
    Generate the per-cumulative tile QA png file.
    Will replace .fits by .png in tileqafits for the png filename.

    Args:
        tileqafits: path to the tile-qa-TILEID-NIGHT.fits file
        prod: full path to input reduction, e.g. /global/cfs/cdirs/desi/spectro/redux/daily (string)

    Options:
        pngoutfile: output filename; default to tileqafits .fits -> .png
        dchi2_min (optional, defaults to value in qa-params.yaml): minimum DELTACHI2 for a valid zspec (float)
        tsnr2_key (optional, defaults to value in qa-params.yaml): TSNR2 key used for plot (string)
        refdir (optional, defaults to "desispec","data/qa"): path to folder with reference measurements for the n(z) and the TSNR2 (string)

    Note:
        If hdr["SURVEY"] is not "main", will not plot the n(z).
        If hdr["FAPRGRM"].lower() is not "bright" or "dark", will not plot the TSNR2 plot nor the skymap.
    """
    # AR config
    config = get_qa_config()

    # AR default values
    if dchi2_min is None:
        dchi2_min = config["tile_qa_plot"]["dchi2_min"]
    if tsnr2_key is None:
        tsnr2_key = config["tile_qa_plot"]["tsnr2_key"]

    # SB derive output file name, handling case if ".fits" appears in path
    if pngoutfile is None:
        base = os.path.splitext(os.path.basename(tileqafits))[0]
        pngoutfile = os.path.join(os.path.dirname(tileqafits), base+'.png')

    # AR reading
    h = fits.open(tileqafits)
    hdr = h["FIBERQA"].header
    fiberqa = h["FIBERQA"].data
    petalqa = h["PETALQA"].data

    if not "SURVEY" in hdr :
        print("no SURVEY keyword in header, skip this tile")
        return

    # AR start plotting
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(6, 4, wspace=0.25, hspace=0.2)

    # AR exposures from that TILEID
    exps = get_expids_efftimes(tileqafits, prod)
    xs = (-0.2, 0.1, 0.4, 0.7)
    y, dy = 0.95, -0.10
    fs = 10
    ax = plt.subplot(gs[0, 1])
    ax.axis("off")
    txts = ["EXPID", "NIGHT", "EFFTIME", "QA_EFFTIME"]
    for x, txt in zip(xs, txts):
        ax.text(x, y, txt, fontsize=fs, fontweight="bold", transform=ax.transAxes)
    y += 2 * dy
    for i in range(len(exps)):
        txts = [
            "{:08d}".format(exps["EXPID"][i]),
            "{}".format(exps["NIGHT"][i]),
            "{:.0f}s".format(exps["EFFTIME_SPEC"][i]),
            "{:.0f}s".format(exps["QA_EFFTIME_SPEC"][i]),
        ]
        for x, txt in zip(xs, txts):
            ax.text(x, y, txt, fontsize=fs, transform=ax.transAxes)
        y += dy

    # AR cutout
    ax = plt.subplot(gs[2:4, 1])
    plot_cutout(ax, hdr["TILEID"], hdr["TILERA"], hdr["TILEDEC"], 4)

    # AR n(z)
    # AR n(z): plotting only if main survey
    if hdr["SURVEY"] == "main":

        # AR n(z): reference
        ref = Table.read(os.path.join(refdir, "qa-reference-nz.ecsv"))

        # AR n(z), for the tracers for that program
        tracers = [
            tracer
            for tracer in list(config["tile_qa_plot"]["tracers"].keys())
            if config["tile_qa_plot"]["tracers"][tracer]["program"]
            == hdr["FAPRGRM"].upper()
        ]
        cols = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(tracers)]

        # AR number of valid zspec in zmin, zmax
        n_valid, nref_valid = 0.0, 0.0

        # compare number of qsos from redrock and QuasarNP
        nqso_rr  = 0
        ### nqso_qnp = 0

        # AR plot
        ax = plt.subplot(gs[0:2, 2])
        for tracer, col in zip(tracers, cols):
            # AR considered tile
            bins, zhists = get_zhists(hdr["TILEID"], tracer, dchi2_min, fiberqa)
            cens = 0.5 * (bins[1:] + bins[:-1])
            ax.plot(cens, zhists, color=col, label=tracer)
            # AR number of valid zspec
            zmin, zmax = get_tracer_zminmax(tracer)
            istracer = get_tracer(tracer, fiberqa)
            sel = (bins[:-1] >= zmin) & (bins[1:] <= zmax)
            n_valid += zhists[sel].sum() * istracer.sum()

            if tracer=="QSO" :
                nqso_rr = int(zhists[sel].sum() * istracer.sum())
                ### nqso_qnp = np.sum((fiberqa['IS_QSO_QN']==1)\
                ###            &(fiberqa['Z_QN']>=zmin)&(fiberqa['Z_QN']<=zmax))

            # AR reference
            sel = ref["TRACER"] == tracer
            ax.fill_between(
                cens,
                ref["N_MEAN"][sel] - ref["N_MEAN_STD"][sel],
                ref["N_MEAN"][sel] + ref["N_MEAN_STD"][sel],
                color=col,
                alpha=0.3,
                label="{} reference".format(tracer),
            )
            # AR reference number of valid zspec
            sel &= (ref["ZMIN"] >= zmin) & (ref["ZMAX"] <= zmax)
            nref_valid += ref["N_MEAN"][sel].sum() * istracer.sum()
        ax.legend(ncol=2)
        ax.set_xlabel("Z")
        ax.set_ylabel("Per tile fractional count")
        if hdr["FAPRGRM"].lower() == "bright":
            ax.set_xlim(0, 1.5)
            ax.set_ylim(0, 0.4)
        else:
            ax.set_xlim(0, 6)
            ax.set_ylim(0, 0.2)
        ax.grid(True)
        # AR n(z) : ratio
        ratio_nz = n_valid / nref_valid
    # AR n(z): if not main, just put dummy -1
    else:
        ratio_nz = -1
        nqso_rr  = -1
        ### nqso_qnp = -1

    # AR Z vs. FIBER plot
    ax = plt.subplot(gs[0:2, 3])
    xlim, ylim = (-100, 5100), (-1.1, 1.1)
    yticks = np.array([0, 0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 6])
    # AR identifying non-assigned/sky/broken fibers
    # AR    (equivalent of OBJTYPE!="TGT" in fiberassign-TILEID.fits.gz)
    # AR    undirect way, as not all columns are here...
    # AR the DESI_TARGET column for sky should be present + correctly set
    # AR for all surveys (with same bits); SUPP_SKY will have SKY set too
    nontgt = np.zeros(len(fiberqa), dtype=bool)
    for msk in ["SKY", "BAD_SKY"]:
        nontgt |= (fiberqa["DESI_TARGET"] & desi_mask[msk]) > 0
    for msk in ["UNASSIGNED", "STUCKPOSITIONER", "BROKENFIBER"]:
        nontgt |= (fiberqa["QAFIBERSTATUS"] & fibermask[msk]) > 0
    sels = [
        (~nontgt) & (fiberqa["QAFIBERSTATUS"] == 0),
        (~nontgt) & (fiberqa["QAFIBERSTATUS"] > 0),
        nontgt
    ]
    labels = ["QAFIBERSTATUS = 0", "QAFIBERSTATUS > 0", "non-TGT"]
    cs = ["b", "r", "y"]
    zorders = [1, 1, 0]
    for sel, label, c, zorder in zip(sels, labels, cs, zorders):
        ax.scatter(fiberqa["FIBER"][sel], np.log10(0.1 + fiberqa["Z"][sel]), s=0.1, c=c, alpha=1.0, zorder=zorder, label="{} ({} fibers)".format(label, sel.sum()))
    for petal in range(10):
        if petal % 2 == 0:
            ax.axvspan(petal * 500, (petal + 1) * 500, color="k", alpha=0.05, zorder=0)
        ax.text(petal * 500 + 250, -1.09, str(petal), color="k", fontsize=10, ha="center")
    ax.set_xlabel("FIBER")
    ax.set_ylabel("Z")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_yticks(np.log10(0.1 + yticks))
    ax.set_yticklabels(yticks.astype(str))
    ax.grid(True)
    ax.legend(loc=2, markerscale=10, fontsize=7)

    show_efftime = True # else show TSNR

    if show_efftime :

        ax = plt.subplot(gs[2:4, 2])
        x = fiberqa["MEAN_FIBER_X"]
        y = fiberqa["MEAN_FIBER_Y"]
        fibers = fiberqa["FIBER"]
        efftime = fiberqa["EFFTIME_SPEC"]
        medefftime = np.median(efftime[efftime>0])
        vmin = 0.5*medefftime
        vmax = 1.5*medefftime

        sel = (efftime>0)
        sc = ax.scatter(
            x[sel],
            y[sel],
            c=efftime[sel],
            cmap=matplotlib.cm.viridis_r,
            vmin=vmin,
            vmax=vmax,
            s=5,
        )

        sel = ((fiberqa["QAFIBERSTATUS"] & fibermask.mask("LOWEFFTIME")) > 0)&(efftime>0)
        ax.scatter(x[sel],y[sel],
                   edgecolor="r", facecolors="none", s=5, alpha=0.5,
                   label="LOWEFFTIME")
        # plotting fibers discarded because of EBV=0
        sel = (fiberqa["QAFIBERSTATUS"] & fibermask.mask("LOWEFFTIME")) == 0
        sel &= (fiberqa["EBV"] == 0)
        ax.scatter(x[sel], y[sel],
                   marker="x", color="purple", s=10, lw=0.5, alpha=0.5,
                   label="EBV=0")

        ax.set_xlabel("FIBER_X [mm]")
        ax.set_ylabel("FIBER_Y [mm]")
        # AR 2*505 mm matches the 4 deg width of the cutout
        ax.set_xlim(-505, 505)
        ax.set_ylim(-505, 505)
        ax.grid(True)
        ax.set_aspect("equal")
        ax.legend(loc=3, ncol=2, markerscale=5)
        # cbar = plt.colorbar(sc, extend="both")
        p =  ax.get_position().get_points().flatten()
        cax = fig.add_axes([
            p[0] + 0.05 * (p[2] - p[0]),
            p[1] + 0.94 * (p[3]-p[1]),
            0.9 * (p[2] - p[0]),
            0.05 * (p[3]-p[1])
        ])
        cbar = plt.colorbar(sc, cax=cax, orientation="horizontal", ticklocation="bottom", pad=0, extend="both")
        #cbar.mappable.set_clim(clim)
        # cbar.set_label("EFFTIME (sec)")
        cbar.ax.text(0.5, 0.5, "EFFTIME (sec)", color="k", ha="center", va="center", transform=cbar.ax.transAxes)

        # AR ratio of the median TSNR2 w.r.t ref
        #sel = np.isfinite(ref["{}_{}".format(tsnr2_key, hdr["FAPRGRM"].upper())])
        #sel &= np.isfinite(tsnr2s)
        #ratio_tsnr2 = np.median(tsnr2s[sel]) / np.median(
        #    ref["{}_{}".format(tsnr2_key, hdr["FAPRGRM"].upper())][sel]


    else :

    # AR TSNR: reference
        ref = Table.read(os.path.join(refdir, "qa-reference-tsnr2.ecsv"))

        # AR TSNR2
        ax = plt.subplot(gs[2:4, 2])
        if hdr["FAPRGRM"].lower() in ["bright", "dark"]:
            clim = (0.5, 1.5)
            # AR TSNR2: ratio (discarding ebv=0 for now, as the TSNR2 is then biased)
            badqa_val, badpet_val = get_qa_badmsks()
            sel = (fiberqa["QAFIBERSTATUS"] & badqa_val) == 0
            sel &= fiberqa["EBV"] > 0
            tsnr2s = np.nan + np.zeros(5000)
            tsnr2s[fiberqa["FIBER"][sel]] = fiberqa[tsnr2_key][sel]
            # AR TSNR2: plot
            sc = ax.scatter(
                ref["FIBERASSIGN_X"],
                ref["FIBERASSIGN_Y"],
                c=tsnr2s / ref["{}_{}".format(tsnr2_key, hdr["FAPRGRM"].upper())],
                cmap=matplotlib.cm.coolwarm_r,
                vmin=clim[0],
                vmax=clim[1],
                s=5,
            )
            sel = ((fiberqa["QAFIBERSTATUS"] & fibermask.mask("LOWEFFTIME")) > 0)
            fibers = fiberqa["FIBER"][sel]
            ax.scatter(ref["FIBERASSIGN_X"][fibers], ref["FIBERASSIGN_Y"][fibers],
                edgecolor="k", facecolors="none", s=5,
                label="bad_petal_mask")
            # AR TSNR2: plotting fibers discarded because of EBV=0
            sel = (fiberqa["QAFIBERSTATUS"] & badqa_val) == 0
            sel &= fiberqa["EBV"] == 0
            fibers = fiberqa["FIBER"][sel]
            ax.scatter(ref["FIBERASSIGN_X"][fibers], ref["FIBERASSIGN_Y"][fibers],
                marker="x", color="k", s=10, lw=0.5,
                label="EBV=0")
            #
            ax.set_xlabel("FIBERASSIGN_X [mm]")
            ax.set_ylabel("FIBERASSIGN_Y [mm]")
            # AR 2*505 mm matches the 4 deg width of the cutout
            ax.set_xlim(-505, 505)
            ax.set_ylim(-505, 505)
            ax.grid(True)
            ax.set_aspect("equal")
            ax.legend(loc=2)
            cbar = plt.colorbar(sc, extend="both")
            cbar.mappable.set_clim(clim)
            cbar.set_label("{} / {}_REFERENCE".format(tsnr2_key, tsnr2_key))
            # AR ratio of the median TSNR2 w.r.t ref
            sel = np.isfinite(ref["{}_{}".format(tsnr2_key, hdr["FAPRGRM"].upper())])
            sel &= np.isfinite(tsnr2s)
            ratio_tsnr2 = np.median(tsnr2s[sel]) / np.median(
                ref["{}_{}".format(tsnr2_key, hdr["FAPRGRM"].upper())][sel]
            )
        # AR TSNR2: if not bright/dark, just put dummy -1
        else:
            ratio_tsnr2 = -1
    # AR TSNR2: display petal ids
    for ang, p in zip(np.linspace(2 * np.pi, 0, 11), [3, 2, 1, 0, 9, 8, 7, 6, 5, 4]):
        anglab = ang + 0.1 * np.pi
        ax.text(
            450 * np.cos(anglab),
            450 * np.sin(anglab),
            "{:.0f}".format(p),
            color="k",
            va="center",
            ha="center",
        )

    # AR positioners accuracy
    ax = plt.subplot(gs[2:4, 3])
    x = fiberqa["MEAN_FIBER_X"]
    y = fiberqa["MEAN_FIBER_Y"]
    fibers = fiberqa["FIBER"]
    c = np.sqrt(fiberqa["MEAN_DELTA_X"] ** 2 + fiberqa["MEAN_DELTA_Y"] ** 2)
    vmin = 0.
    vmax = 0.03

    sc = ax.scatter(
        x,
        y,
        c=c,
        cmap=matplotlib.cm.viridis,
        vmin=vmin,
        vmax=vmax,
        s=5,
    )

    ax.set_xlabel("FIBER_X [mm]")
    ax.set_ylabel("FIBER_Y [mm]")
    # AR 2*505 mm matches the 4 deg width of the cutout
    ax.set_xlim(-505, 505)
    ax.set_ylim(-505, 505)
    ax.grid(True)
    ax.set_aspect("equal")
    # cbar = plt.colorbar(sc, extend="both")
    p =  ax.get_position().get_points().flatten()
    cax = fig.add_axes([
        p[0] + 0.05 * (p[2] - p[0]),
        p[1] + 0.94 * (p[3]-p[1]),
        0.9 * (p[2] - p[0]),
        0.05 * (p[3]-p[1])
    ])
    cbar = plt.colorbar(sc, cax=cax, orientation="horizontal", ticklocation="bottom", pad=0, extend="max")
    cbar.set_ticks([0, 0.01, 0.02, 0.03])
    cbar.ax.text(0.5, 0.5, "DELTA_XY (mm)", color="k", ha="center", va="center", transform=cbar.ax.transAxes)

    # AR display petal ids
    for ang, p in zip(np.linspace(2 * np.pi, 0, 11), [3, 2, 1, 0, 9, 8, 7, 6, 5, 4]):
        anglab = ang + 0.1 * np.pi
        ax.text(
            450 * np.cos(anglab),
            450 * np.sin(anglab),
            "{:.0f}".format(p),
            color="k",
            va="center",
            ha="center",
        )

    # AR sky map
    ax = plt.subplot(gs[1, 1], projection="mollweide")
    plot_mw_skymap(
        fig,
        ax,
        hdr["TILEID"],
        hdr["TILERA"],
        hdr["TILEDEC"],
        hdr["SURVEY"],
        hdr["FAPRGRM"].lower(),
        org=120,
    )
    for k in ["RMSDIST","EFFTIME"] :
        if k not in hdr : hdr[k]=0
    # AR overall infos
    ax = plt.subplot(gs[0:2, 0])
    ax.axis("off")
    x0, x1, y, dy, fs = 0.45, 0.55, 0.95, -0.08, 10

    if hdr["VALID"] :
        ax.text(0.1,y,"TILE IS VALID",color="green",
                fontsize=int(fs*1.2),
                fontweight="bold",
                ha="left",
                transform=ax.transAxes,
        )
        y += dy
    else :
        ax.text(0.1,y,"TILE IS NOT VALID",color="red",
                fontsize=int(fs*1.2),
                fontweight="bold",
                ha="left",
                transform=ax.transAxes,
        )
        y += dy

    for txt in [
        ["TILEID-thruNIGHT", "{:06d}-{}".format(hdr["TILEID"], hdr["LASTNITE"])],
        ["SURVEY-PROGRAM", "{}-{}".format(hdr["SURVEY"], hdr["FAPRGRM"])],
        ["RA , DEC", "{:.3f} , {:.3f}".format(hdr["TILERA"], hdr["TILEDEC"])],
        ["EBVFAC", "{:.2f}".format(hdr["EBVFAC"])],
        ["", ""],
        ["efftime / goaltime", "{:.0f}/{:.0f}={:.2f}".format(exps["EFFTIME_SPEC"].sum(), hdr["GOALTIME"], exps["EFFTIME_SPEC"].sum() / hdr["GOALTIME"])],
        ["qa_efftime / goaltime", "{:.0f}/{:.0f}={:.2f}".format(hdr["EFFTIME"], hdr["GOALTIME"], hdr["EFFTIME"] / hdr["GOALTIME"])],
        ["n(z) / n_ref(z)", "{:.2f}".format(ratio_nz)],
        ### ["nqso(RR) , nqso(QNP)", "{} , {}".format(nqso_rr,nqso_qnp)],
        ["nqso(RR)", "{}".format(nqso_rr)],

        ["NGOODFIB", "{}".format(hdr["NGOODFIB"])],
        ["NGOODPET", "{}".format(hdr["NGOODPET"])],
        ["Fiber pos. RMS(2D)", "{:.3f} mm".format(hdr["RMSDIST"])],
    ]:
        fontweight, col = "normal", "k"
        if (
            (txt[0] == "efftime / goaltime") &
            (exps["EFFTIME_SPEC"].sum() / hdr["GOALTIME"] < hdr["MINTFRAC"])
        ) | (
                (txt[0] == "qa_efftime / goaltime") &
                (hdr["EFFTIME"] / hdr["GOALTIME"] < hdr["MINTFRAC"])
        ):
            fontweight, col = "bold", "r"
        if (txt[0] == "n(z) / n_ref(z)") & (ratio_nz < 0.8):
            fontweight, col = "bold", "r"
        #if (txt[0] == "tsnr2 / tsnr2_ref") & (ratio_tsnr2 < 0.8):
        #    fontweight, col = "bold", "r"
        ax.text(
            x0,
            y,
            txt[0],
            color=col,
            fontsize=fs,
            fontweight=fontweight,
            ha="right",
            transform=ax.transAxes,
        )
        ax.text(
            x1,
            y,
            txt[1],
            color=col,
            fontsize=fs,
            fontweight=fontweight,
            ha="left",
            transform=ax.transAxes,
        )
        y += dy


    # AR per petal diagnoses
    ax = plt.subplot(gs[2:4, 0])
    print_petal_infos(ax, petalqa,fiberqa)
    try :
        #  AR saving plot
        plt.savefig(pngoutfile, bbox_inches="tight")
    except ValueError as e :
        print("failed to save figure")
        print(e)
        pngoutfile=None

    plt.close()
    return pngoutfile
