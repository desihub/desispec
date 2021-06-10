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
import tempfile
from desitarget.targetmask import desi_mask, bgs_mask
from desispec.maskbits import fibermask
from astropy.table import Table
from astropy.io import fits
import fitsio
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
import matplotlib.image as mpimg

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


def get_qa_fstatus_badmsk():
    """
    Returns the bitmask values for bad fibers.

    Args:
        None

    Returns:
        The bitmask value.
    """
    config = get_qa_config()
    badmsk_names = config["exposure_qa"]["bad_qafstatus_mask"]
    return fibermask.mask(badmsk_names)


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
        - FIBERSTATUS=0
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
    sel = (d[fstatus_key] & get_qa_fstatus_badmsk()) == 0
    if tracer in ["BGS_BRIGHT", "BGS_FAINT"]:
        sel = (d["BGS_TARGET"] & bgs_mask[tracer]) > 0
    else:
        sel = (d["DESI_TARGET"] & desi_mask[tracer]) > 0
    # AR ELG_LOP : excluding ELG_LOP x QSO
    if tracer == "ELG_LOP":
        sel &= (d["DESI_TARGET"] & desi_mask["QSO"]) == 0
    return sel


def get_tracer_zok(tracer, dchi2_min, d, fstatus_key="QAFIBERSTATUS"):
    """
    For a given tracer, returns the spectro. valid sample for the tile QA n(z):
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
    sel = (d[fstatus_key] & get_qa_fstatus_badmsk()) == 0
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


def plot_cutout(ax, tileid, tilera, tiledec, width_deg, c="w"):
    """
    Plots a ls-dr9 cutout, with overlaying the petals.

    Args:
        ax: pyplot object
        tileid: TILEID (int)
        tilera: tile center R.A. (float)
        tiledec: tile center Dec. (float)
        width_deg: width of the cutout in degrees (np.array of floats)
        c (optional, defaults to "w"): color used to display targets (string)
    
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
            dxs, dys, c=c, lw=0.25, alpha=1.0, zorder=1,
        )
        anglab = ang + 0.1 * np.pi
        dxs, dys = deg2pix(
            1.1 * tile_radius_deg * np.cos(anglab),
            1.1 * tile_radius_deg * np.sin(anglab),
            width_deg,
            width_pix,
        )
        ax.text(
            dxs, dys, "{:.0f}".format(p), color=c, va="center", ha="center",
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
        dxs, dys, c=c, lw=0.25, alpha=1.0, zorder=1,
    )

    # AR
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, width_pix + 0.5)
    ax.set_ylim(-0.5, width_pix + 0.5)
    ax.axis("off")


def print_petal_infos(ax, petalqa):
    """
    Print some diagnoses for each petal, and for the whole tile.
    
    Args:
        ax: pyplot object
        petalqa: the PETALQA extension data of the tile-qa-TILEID-NIGHT.fits file
    """
    config = get_qa_config()

    # AR stats per petal
    mydict = {
        "PETAL_LOC": {
            "X": 0.05,
            "SHORT": "PETAL",
            "PRECISION": 0,
            "MIN": -1e99,
            "MAX": 1e99,
        },
        "WORSTREADNOISE": {
            "X": 0.25,
            "SHORT": "RDN",
            "PRECISION": 1,
            "MIN": -1e99,
            "MAX": config["exposure_qa"]["max_readnoise"],
        },
        "NGOODPOS": {
            "X": 0.45,
            "SHORT": "GOODPOS",
            "PRECISION": 0,
            "MIN": 500 * config["exposure_qa"]["max_frac_of_bad_positions_per_petal"],
            "MAX": 1e99,
        },
        "NSTDSTAR": {
            "X": 0.65,
            "SHORT": "NSTD",
            "PRECISION": 0,
            "MIN": config["exposure_qa"]["min_number_of_good_stdstars_per_petal"],
            "MAX": 1e99,
        },
        "STARRMS": {
            "X": 0.85,
            "SHORT": "*RMS",
            "PRECISION": 3,
            "MIN": -1e99,
            "MAX": config["exposure_qa"]["max_rms_of_rflux_ratio_of_stdstars"],
        },
        "TSNR2FRA": {
            "X": 1.05,
            "SHORT": "TSNR2FRA",
            "PRECISION": 1,
            "MIN": config["exposure_qa"]["tsnr2_petal_minfrac"],
            "MAX": config["exposure_qa"]["tsnr2_petal_maxfrac"],
        },
    }
    y, dy = 0.95, -0.1
    fs = 10
    for key in list(mydict.keys()):
        ax.text(
            mydict[key]["X"],
            y,
            mydict[key]["SHORT"],
            fontsize=fs,
            ha="center",
            transform=ax.transAxes,
        )
    y += dy
    for i in range(10):
        for key in list(mydict.keys()):
            #
            fontweight, color = "normal", "k"
            if (petalqa[key][i] < mydict[key]["MIN"]) | (
                petalqa[key][i] > mydict[key]["MAX"]
            ):
                fontweight, color = "bold", "r"
            #
            ax.text(
                mydict[key]["X"],
                y,
                "{:.{}f}".format(petalqa[key][i], mydict[key]["PRECISION"]),
                color=color,
                fontsize=fs,
                fontweight=fontweight,
                ha="center",
                transform=ax.transAxes,
            )
        y += dy
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
        clim = (0.75, 1.25)
        c = pixwd["ALL"] / dens_med
        clabel = "All {} targets / ({:.0f}/deg2)".format(program, dens_med)
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


def make_tile_qa_plot(
    tileqafits,
    dchi2_min=None,
    tsnr2_key=None,
    refdir=resource_filename("desispec", "data/qa"),
):
    """
    Generate the per-cumulative tile QA png file.
    Will replace .fits by .png in tileqafits for the png filename.

    Args:
        tileqafits: path to the tile-qa-TILEID-NIGHT.fits file
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

    # AR reading
    h = fits.open(tileqafits)
    hdr = h["FIBERQA"].header
    fiberqa = h["FIBERQA"].data
    petalqa = h["PETALQA"].data

    # AR start plotting
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 3, wspace=0.25, hspace=0.2)

    # AR cutout
    ax = plt.subplot(gs[1, 1])
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

        # AR plot
        ax = plt.subplot(gs[0, 2])
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
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 0.2)
        ax.grid(True)
        # AR n(z) : ratio
        ratio_nz = n_valid / nref_valid
    # AR n(z): if not main, just put dummy -1
    else:
        ratio_nz = -1

    # AR TSNR: reference
    ref = Table.read(os.path.join(refdir, "qa-reference-tsnr2.ecsv"))

    # AR TSNR2
    ax = plt.subplot(gs[1, 2])
    if hdr["FAPRGRM"].lower() in ["bright", "dark"]:
        clim = (0.5, 1.5)
        # AR TSNR2: ratio
        sel = (fiberqa["QAFIBERSTATUS"] & get_qa_fstatus_badmsk()) == 0
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
        ax.set_xlabel("FIBERASSIGN_X [mm]")
        ax.set_ylabel("FIBERASSIGN_Y [mm]")
        ax.grid(True)
        ax.set_aspect("equal")
        cbar = plt.colorbar(sc, extend="both")
        cbar.mappable.set_clim(clim)
        # AR as we are plotting the ratio of some TSNR2 quantity,
        # AR    it is the same as the ratio of EFFTIME_SPEC
        # AR    hence we report EFFTIME_SPEC to make it more intuitive
        cbar.set_label("EFFTIME_SPEC / EFFTIME_SPEC_REFERENCE")
        # AR ratio of the median TSNR2 w.r.t ref
        sel = np.isfinite(ref["{}_{}".format(tsnr2_key, hdr["FAPRGRM"].upper())])
        sel &= np.isfinite(tsnr2s)
        ratio_tsnr2 = np.median(tsnr2s[sel]) / np.median(
            ref["{}_{}".format(tsnr2_key, hdr["FAPRGRM"].upper())][sel]
        )
    # AR TSNR2: if not bright/dark, just put dummy -1
    else:
        ratio_tsnr2 = -1

    # AR sky map
    ax = plt.subplot(gs[0, 1], projection="mollweide")
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

    # AR overall infos
    ax = plt.subplot(gs[0, 0])
    ax.axis("off")
    x, y, dy, fs = 0.05, 0.95, -0.1, 15
    for t in [
        "TILEID, thruNIGHT = {:06d} , {}".format(hdr["TILEID"], hdr["NIGHT"]),
        "SURVEY , PROGRAM = {} , {}".format(hdr["SURVEY"], hdr["FAPRGRM"]),
        "RA , DEC = {:.3f} , {:.3f}".format(hdr["TILERA"], hdr["TILEDEC"]),
        "EBVFAC = {:.2f}".format(hdr["EBVFAC"]),
        "",
        "efftime / goaltime = {:.2f}".format(hdr["EFFTIME_SPEC"] / hdr["GOALTIME"]),
        "ratio n(z) / n_ref(z) = {:.2f}".format(ratio_nz),
        # AR as we are computing the ratio of some TSNR2 quantity,
        # AR    it is the same as the ratio of EFFTIME_SPEC
        # AR    hence we report EFFTIME_SPEC to make it more intuitive
        "ratio efftime / efftime_ref = {:.2f}".format(ratio_tsnr2),
    ]:
        fontweight, col = "normal", "k"
        # if (t[:18] == "efftime / goaltime") & (hdr["EFFTIME_SPEC"] / hdr["GOALTIME"] < hdr["MINTFRAC"]):
        if (t[:18] == "efftime / goaltime") & (hdr["EFFTIME_SPEC"] / hdr["GOALTIME"] < 0.85): # TBD: replace by MINTFRAC
            fontweight, col = "bold", "r"
        if (t[:10] == "ratio n(z)") & (ratio_nz < 0.8):
            fontweight, col = "bold", "r"
        if (t[:13] == "ratio efftime") & (ratio_tsnr2 < 0.8):
            fontweight, col = "bold", "r"
        ax.text(x, y, t.expandtabs(), color=col, fontsize=fs, fontweight=fontweight, transform=ax.transAxes)
        y += dy

    # AR per petal diagnoses
    ax = plt.subplot(gs[1, 0])
    print_petal_infos(ax, petalqa)

    #  AR saving plot
    plt.savefig(
        tileqafits.replace(".fits", ".png"), bbox_inches="tight",
    )
    plt.close()
