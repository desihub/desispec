#!/usr/bin/env python

# AR general
import sys
import os
from glob import glob
import tempfile
import textwrap
from desiutil.log import get_logger
# AR scientifical
import numpy as np
import fitsio
# AR astropy
from astropy.table import Table, vstack
from astropy.io import fits
# AR desitarget
from desitarget.targetmask import desi_mask, bgs_mask
from desitarget.targetmask import zwarn_mask as desitarget_zwarn_mask
from desitarget.targets import main_cmx_or_sv
from desitarget.targets import zcut as lya_zcut
# AR desispec
from desispec.fiberbitmasking import get_skysub_fiberbitmask_val
from desispec.io import findfile
# AR matplotlib
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator
# AR PIL (to create pdf from pngs)
from PIL import Image

log = get_logger()

def get_nightqa_outfns(outdir, night):
    """
    Utility function to get nightqa file names.

    Args:
        outdir: output folder name (string)
        night: night (int)

    Returns:
        dictionary with file names
    """
    return {
        "html" : os.path.join(outdir, "nightqa-{}.html".format(night)),
        "dark" : os.path.join(outdir, "dark-{}.pdf".format(night)),
        "badcol" : os.path.join(outdir, "badcol-{}.png".format(night)),
        "ctedet" : os.path.join(outdir, "ctedet-{}.pdf".format(night)),
        "sframesky" : os.path.join(outdir, "sframesky-{}.pdf".format(night)),
        "tileqa" : os.path.join(outdir, "tileqa-{}.pdf".format(night)),
        "skyzfiber" : os.path.join(outdir, "skyzfiber-{}.png".format(night)),
        "petalnz" : os.path.join(outdir, "petalnz-{}.pdf".format(night)),
    }



def get_surveys_night_expids(
    night,
    datadir = None):
    """
    List the (EXPIDs, TILEIDs) from a given night for a given survey.

    Args:
        night: night (int)
        surveys: comma-separated list of surveys to consider, in lower-cases, e.g. "sv1,sv2,sv3,main" (str)
        datadir (optional, defaults to $DESI_SPECTRO_DATA): full path where the {NIGHT}/desi-{EXPID}.fits.fz files are (str)

    Returns:
        expids: list of the EXPIDs (np.array())
        tileids: list of the TILEIDs (np.array())
        surveys: list of the SURVEYs (np.array())

    Notes:
        Based on:
        - parsing the OBSTYPE keywords from the SPEC extension header of the desi-{EXPID}.fits.fz files;
        - for OBSTYPE="SCIENCE", parsing the fiberassign-TILEID.fits* header
    """
    if datadir is None:
        datadir = os.getenv("DESI_SPECTRO_DATA")
    fns = sorted(
        glob(
            os.path.join(
                datadir,
                "{}".format(night),
                "????????",
                "desi-????????.fits.fz",
            )
        )
    )
    expids, tileids, surveys = [], [], []
    for i in range(len(fns)):
        hdr = fits.getheader(fns[i], "SPEC")
        if hdr["OBSTYPE"] == "SCIENCE":
            survey = "unknown"
            # AR look for the fiberassign file
            # AR - used wildcard, because early files (pre-SV1?) were not gzipped
            # AR - first check SURVEY keyword (should work for SV3 and later)
            # AR - if not present, take FA_SURV
            fafns = glob(os.path.join(os.path.dirname(fns[i]), "fiberassign-??????.fits*"))
            if len(fafns) > 0:
                fahdr = fits.getheader(fafns[0], 0)
                if "SURVEY" in fahdr:
                    survey = fahdr["SURVEY"]
                else:
                    survey = fahdr["FA_SURV"]
            if survey == "unknown":
                log.warning("SURVEY could not be identified for {}; setting to 'unknown'".format(fns[i]))
            # AR append
            expids.append(hdr["EXPID"])
            tileids.append(hdr["TILEID"])
            surveys.append(survey)
    expids, tileids, surveys = np.array(expids), np.array(tileids), np.array(surveys)
    per_surv = []
    for survey in np.unique(surveys):
        sel = surveys == survey
        per_surv.append(
            "{} exposures from {} tiles for SURVEY={}".format(
                sel.sum(), np.unique(tileids[sel]).size, survey,
            )
        )
    log.info("for NIGHT={} found {}".format(night, " and ".join(per_surv)))
    return expids, tileids, surveys


def get_dark_night_expid(night, prod):
    """
    Returns the EXPID of the 300s DARK exposure for a given night.

    Args:
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc (string)

    Returns:
        expid: EXPID (int)

    Notes:
        If nothing found, returns None.
        20220110 : new method, relying on processing_tables
    """
    #
    expid = None
    proctable_fn = os.path.join(
        prod,
        "processing_tables",
        "processing_table_{}-{}.csv".format(os.path.basename(prod), night),
    )
    log.info("proctable_fn = {}".format(proctable_fn))
    if not os.path.isfile(proctable_fn):
        log.warning("no {} found; returning None".format(proctable_fn))
    else:
        d = Table.read(proctable_fn)
        sel = (d["OBSTYPE"] == "dark") & (d["JOBDESC"] == "ccdcalib")
        if sel.sum() == 0:
            log.warning(
                "found zero exposures with OBSTYPE=dark and JOBDESC=ccdcalib in proctable_fn; returning None",
            )
        elif sel.sum() > 1:
            log.warning(
                "found {} > 1 exposures with OBSTYPE=dark and JOBDESC=ccdcalib in proctable_fn; returning None".format(
                    sel.sum(),
                )
            )
        else:
            expid = int(str(d["EXPID"][sel][0]).strip("|"))
            log.info(
                "found EXPID={} as the 300s DARK for NIGHT={}".format(
                    expid, night,
                )
            )
    return expid


def get_ctedet_night_expid(night, prod):
    """
    Returns the EXPID of the 1s FLAT exposure for a given night.
    If not present, takes the science exposure with the lowest sky counts.

    Args:
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc (string)

    Returns:
        expid: EXPID (int)

    Notes:
        If nothing found, returns None.
        We look for preproc files.
        As we are looking for a faint signal, we want the image with the less electrons,
            thus it could be picking a BRIGHT short exposure against a longer DARK exposure.
    """
    expids = np.array(
        [
            int(os.path.basename(fn))
            for fn in sorted(
                glob(
                    os.path.join(
                        prod,
                        "preproc",
                        "{}".format(night),
                        "*",
                    )
                )
            )
        ]
    )
    ctedet_expid = None
    # AR checking preproc-??-{EXPID}.fits
    for expid in expids:
        fns = sorted(
            glob(
                os.path.join(
                    prod,
                    "preproc",
                    "{}".format(night),
                    "{:08d}".format(expid),
                    "preproc-??-{:08d}.fits".format(expid),
                )
            )
        )
        # AR if some preproc files, just pick the first one
        if len(fns) > 0:
            hdr = fits.getheader(fns[0], "IMAGE")
            if (hdr["OBSTYPE"] == "FLAT") & (hdr["REQTIME"] == 1):
                ctedet_expid = hdr["EXPID"]
                break
    if ctedet_expid is not None:
        log.info(
            "found EXPID={} as the 1s FLAT for NIGHT={}".format(
                expid, night,
            )
        )
    # AR if no 1s FLAT, go for the SCIENCE exposure with the lowest sky counts
    # AR using the r-band sky
    else:
        log.warning(
            "no EXPID found as the 1s FLAT for NIGHT={}; going for SCIENCE exposures".format(night)
        )
        minsky = 1e10
        # AR checking sky-r?-{EXPID}.fits
        for expid in expids:
            fns = sorted(
                glob(
                    os.path.join(
                        prod,
                        "exposures",
                        "{}".format(night),
                        "{:08d}".format(expid),
                        "sky-r?-{:08d}.fits".format(expid),
                    )
                )
            )
            # AR if some sky files, just pick the first one
            if len(fns) > 0:
                hdr = fits.getheader(fns[0], "SKY")
                if hdr["OBSTYPE"] == "SCIENCE":
                    sky = np.median(fits.open(fns[0])["SKY"].data)
                    log.info("{} r-sky = {:.1f}".format(os.path.basename(fns[0]), sky))
                    if sky < minsky:
                        ctedet_expid, minsky = expid, sky
                        log.info("\t=> pick {}".format(expid))
        #
        if ctedet_expid is not None:
            log.info(
                "found EXPID={} as the sky image with the lowest counts for NIGHT={}".format(
                    ctedet_expid, night,
                )
            )
        else:
            log.warning(
                "no SCIENCE EXPID with sky-r?-*fits file found for NIGHT={}; returning None".format(night)
            )
    return ctedet_expid


def create_mp4(fns, outmp4, duration=15):
    """
    Create an animated .mp4 from a set of input files (usually pngs).

    Args:
        fns: list of input filenames, in the correct order (list of strings)
        outmp4: output .mp4 filename
        duration (optional, defaults to 15): video duration in seconds (float)

    Notes:
        Requires ffmpeg to be installed.
        At NERSC, run in the bash command line: "module load ffmpeg".
        The movie uses fns in the provided order.
    """
    # AR is ffmpeg installed
    if os.system("which ffmpeg") != 0:
        log.error("ffmpeg needs to be installed to create the mp4 movies; it can be installed at nersc with 'module load ffmpeg'")
        raise RuntimeError("ffmpeg needs to be installed to run create_mp4()")
    # AR deleting existing video mp4, if any
    if os.path.isfile(outmp4):
        log.info("deleting existing {}".format(outmp4))
        os.remove(outmp4)
    # AR temporary folder
    tmpoutdir = tempfile.mkdtemp()
    # AR copying files to tmpoutdir
    n_img = len(fns)
    for i in range(n_img):
        _ = os.system("cp {} {}/tmp-{:04d}.png".format(fns[i], tmpoutdir, i))
    print(fns)
    # AR ffmpeg settings
    default_fps = 25. # ffmpeg default fps 
    pts_fac = "{:.1f}".format(duration / (n_img / default_fps))
    # cmd = "ffmpeg -i {}/tmp-%04d.png -filter:v 'setpts={}*PTS' {}".format(tmpoutdir, pts_fac, outmp4)
    # AR following encoding so that mp4 are displayed in safari, firefox
    cmd = "ffmpeg -i {}/tmp-%04d.png -vf 'setpts={}*PTS,crop=trunc(iw/2)*2:trunc(ih/2)*2' -vcodec libx264 -pix_fmt yuv420p {}".format(tmpoutdir, pts_fac, outmp4)
    _ = os.system(cmd)
    # AR deleting temporary tmp*png files
    for i in range(n_img):
        os.remove("{}/tmp-{:04d}.png".format(tmpoutdir, i))


def create_dark_pdf(outpdf, night, prod, dark_expid, binning=4):
    """
    For a given night, create a pdf with the 300s binned dark.

    Args:
        outpdf: output pdf file (string)
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc (string)
        dark_expid: EXPID of the 300s DARK exposure to display (int)
        binning (optional, defaults to 4): binning of the image (which will be beforehand trimmed) (int)
    """
    cameras = ["b", "r", "z"]
    petals = np.arange(10, dtype=int)
    clim = (-5, 5)
    with PdfPages(outpdf) as pdf:
        for petal in petals:
            fig = plt.figure(figsize=(20, 10))
            gs = gridspec.GridSpec(1, len(cameras), wspace=0.1)
            for ic, camera in enumerate(cameras):
                ax = plt.subplot(gs[ic])
                fn = os.path.join(
                        prod,
                        "preproc",
                        "{}".format(night),
                        "{:08d}".format(dark_expid),
                        "preproc-{}{}-{:08d}.fits".format(camera, petal, dark_expid),
                )
                ax.set_title("EXPID={} {}{}".format(dark_expid, camera, petal))
                if os.path.isfile(fn):
                    log.info("reading {}".format(fn))
                    h = fits.open(fn)
                    image, ivar, mask = h["IMAGE"].data, h["IVAR"].data, h["MASK"].data
                    # AR setting to np.nan pixels with ivar = 0 or mask > 0
                    # AR hence, when binning, any binned pixel with a masked pixel
                    # AR will appear as np.nan (easy way to go)
                    d = image.copy()
                    sel = (ivar == 0) | (mask > 0)
                    d[sel] = np.nan
                    # AR trimming
                    shape_orig = d.shape
                    if shape_orig[0] % binning != 0:
                        d = d[shape_orig[0] % binning:, :]
                    if shape_orig[1] % binning != 0:
                        d = d[:, shape_orig[1] % binning:]
                        log.info(
                            "{} trimmed from ({}, {}) to ({}, {})".format(
                                fn, shape_orig[0], shape_orig[1], d.shape[0], d.shape[1],
                            )
                        )
                    d_reshape = d.reshape(
                        d.shape[0] // binning,
                        binning,
                        d.shape[1] // binning,
                        binning
                    )
                    d_bin = d_reshape.mean(axis=1).mean(axis=-1)
                    # AR displaying masked pixels (np.nan) in red
                    d_bin_msk = np.ma.masked_where(d_bin == np.nan, d_bin)
                    cmap = matplotlib.cm.Greys_r
                    cmap.set_bad(color="r")
                    im = ax.imshow(d_bin_msk, cmap=cmap, vmin=clim[0], vmax=clim[1])
                    if camera == cameras[-1]:
                        p =  ax.get_position().get_points().flatten()
                        cax = fig.add_axes([
                            p[0] + 1.05 * (p[2] - p[0]),
                            p[1],
                            0.05 * (p[2] - p[0]),
                            1.0 * (p[3]-p[1])
                        ])
                        cbar = plt.colorbar(im, cax=cax, orientation="vertical", ticklocation="right", pad=0, extend="both")
                        cbar.set_label("Units : ?")
                        cbar.mappable.set_clim(clim)
                else:
                    log.warning("missing {}".format(fn))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()


def create_badcol_png(outpng, night, prod, n_previous_nights=10):
    """
    For a given night, create a png file with displaying the number of bad columns per {camera}{petal}.

    Args:
        outpng: output png file (string)
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc (string)
        n_previous_nights (optional, defaults to 10): number of previous nights to plot (int)
    """
    cameras = ["b", "r", "z"]
    colors = ["b", "r", "k"]
    petals = np.arange(10, dtype=int)
    # AR grabbing the n_previous_nights previous nights
    nights = np.array(
        [int(os.path.basename(fn))
            for fn in sorted(
                glob(
                    os.path.join(
                        prod,
                        "exposures",
                        "*"
                    )
                )
            )
        ]
    )
    nights = nights[nights < night]
    nights = nights[-n_previous_nights:]
    all_nights = nights.tolist() + [night]
    # AR reading
    badcols = {}
    for nite in all_nights:
        badcols[nite] = {
            camera : np.nan + np.zeros(len(petals)) for camera in cameras
        }
        for camera in cameras:
            for petal in petals:
                fn = os.path.join(
                        prod,
                        "calibnight",
                        "{}".format(nite),
                        "badcolumns-{}{}-{}.csv".format(camera, petal, nite),
                )
                if os.path.isfile(fn):
                    log.info("reading {}".format(fn))
                    badcols[nite][camera][petal] = len(Table.read(fn))
    # AR plotting
    fig, ax = plt.subplots()
    for nite in all_nights:
        for camera, color in zip(cameras, colors):
            if nite == night:
                marker, alpha, lw, label = "-o", 1.0, 2.0, "{}-camera".format(camera)
            else:
                marker, alpha, lw, label = "-", 0.3, 0.8, None
            ax.plot(petals, badcols[nite][camera], marker, lw=lw, alpha=alpha, color=color, label=label)
    ax.legend(loc=2)
    if len(nights) > 0:
        ax.text(
            0.98, 0.95, "Thin: {} previous nights ({}-{})".format(
                n_previous_nights, nights.min(), nights.max(),
            ), color="k", fontsize=10, ha="right", transform=ax.transAxes,
        )
    ax.set_title("{}".format(night))
    ax.set_xlabel("PETAL_LOC")
    ax.set_xlim(petals[0] - 1, petals[-1] + 1)
    ax.set_ylabel("N(badcolumn)")
    ax.set_ylim(0, 50)
    ax.grid()
    plt.savefig(outpng, bbox_inches="tight")
    plt.close()


def create_ctedet_pdf(outpdf, night, prod, ctedet_expid, nrow=21, xmin=None, xmax=None, ylim=(-5, 10)):
    """
    For a given night, create a pdf with a CTE diagnosis (from preproc files).

    Args:
        outpdf: output pdf file (string)
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc (string)
        ctedet_expid: EXPID for the CTE diagnosis (1s FLAT, or darker science exposure) (int)
        nrow (optional, defaults to 21): number of rows to include in median (int)
        xmin (optional, defaults to None): minimum column to display (int)
        xmax (optional, defaults to None): maximum column to display (int)
        ylim (optional, default to (-5, 10)): ylim for the median plot (duplet)

    Notes:
        Credits to S. Bailey.
        Copied-pasted-adapted from /global/homes/s/sjbailey/desi/dev/ccd/plot-amp-cte.py
    """
    cameras = ["b", "r", "z"]
    petals = np.arange(10, dtype=int)
    clim = (-5, 5)
    with PdfPages(outpdf) as pdf:
        for petal in petals:
            for camera in cameras:
                petcam_xmin, petcam_xmax = xmin, xmax
                fig = plt.figure(figsize=(30, 5))
                gs = gridspec.GridSpec(2, 1, wspace=0.1, height_ratios = [1, 4])
                ax2d = plt.subplot(gs[0])
                ax1d = plt.subplot(gs[1])
                #
                fn = os.path.join(
                        prod,
                        "preproc",
                        "{}".format(night),
                        "{:08d}".format(ctedet_expid),
                        "preproc-{}{}-{:08d}.fits".format(camera, petal, ctedet_expid),
                )
                ax1d.set_title(
                    "{}\nMedian of {} rows above/below CCD amp boundary".format(
                        fn, nrow,
                    )
                )
                if os.path.isfile(fn):
                    # AR read
                    img = fits.open(fn)["IMAGE"].data
                    ny, nx = img.shape
                    if petcam_xmin is None:
                        petcam_xmin = 0
                    if petcam_xmax is None:
                        petcam_xmax = nx
                    above = np.median(img[ny // 2: ny // 2 + nrow, petcam_xmin : petcam_xmax], axis=0)
                    below = np.median(img[ny // 2 - nrow : ny // 2, petcam_xmin : petcam_xmax], axis=0)
                    xx = np.arange(petcam_xmin, petcam_xmax)
                    # AR plot 2d image
                    extent = [petcam_xmin - 0.5, petcam_xmax - 0.5, ny // 2 - nrow - 0.5, ny // 2 + nrow - 0.5]
                    vmax = {"b" : 20, "r" : 40, "z" : 60}[camera]
                    ax2d.imshow(img[ny // 2 - nrow : ny // 2 + nrow, petcam_xmin : petcam_xmax], vmin=-5, vmax=vmax, extent=extent)
                    ax2d.xaxis.tick_top()
                    # AR plot 1d median
                    ax1d.plot(xx, above, alpha=0.5, label="above (AMPC : x < {}; AMPD : x > {}".format(nx // 2 - 1, nx // 2 -1))
                    ax1d.plot(xx, below, alpha=0.5, label="below (AMPA : x < {}; AMPB : x > {}".format(nx // 2 - 1, nx // 2 -1))
                    ax1d.legend(loc=2)
                    # AR amplifier x-boundary
                    ax1d.axvline(nx // 2 - 1, color="k", ls="--")
                    ax1d.set_xlabel("CCD column")
                    ax1d.set_xlim(petcam_xmin, petcam_xmax)
                    ax1d.set_ylim(ylim)
                    ax1d.grid()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close()


def create_sframesky_pdf(outpdf, night, prod, expids):
    """
    For a given night, create a pdf from per-expid sframe for the sky fibers only.

    Args:
        outpdf: output pdf file (string)
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc (string)
        expids: expids to display (list or np.array)
    """
    #
    cameras = ["b", "r", "z"]
    petals = np.arange(10, dtype=int)
    #
    nightdir = os.path.join(prod, "exposures", "{}".format(night))
    # AR sorting the EXPIDs by increasing order
    expids = np.sort(expids)
    #
    with PdfPages(outpdf) as pdf:
        for expid in expids:
            tileid = None
            fns = sorted(
                glob(
                    os.path.join(
                        nightdir,
                        "{:08d}".format(expid),
                        "sframe-??-{:08d}.fits".format(expid),
                    )
                )
            )
            if len(fns) > 0:
                #
                mydict = {camera : {} for camera in cameras}
                for ic, camera in enumerate(cameras):
                    for petal in petals:
                        fn = os.path.join(
                            nightdir,
                            "{:08d}".format(expid),
                            "sframe-{}{}-{:08d}.fits".format(camera, petal, expid),
                        )
                        if os.path.isfile(fn):
                            h = fits.open(fn)
                            sel = h["FIBERMAP"].data["OBJTYPE"] == "SKY"
                            h["FLUX"].data = h["FLUX"].data[sel, :]
                            h["FIBERMAP"].data = h["FIBERMAP"].data[sel]
                            if "flux" not in mydict[camera]:
                                mydict[camera]["wave"] = h["WAVELENGTH"].data
                                nwave = len(mydict[camera]["wave"])
                                mydict[camera]["petals"] = np.zeros(0, dtype=int)
                                mydict[camera]["flux"] = np.zeros(0).reshape((0, nwave))
                                mydict[camera]["isflag"] = np.zeros(0, dtype=bool)
                            mydict[camera]["flux"] =  np.append(mydict[camera]["flux"], h["FLUX"].data, axis=0)
                            mydict[camera]["petals"] = np.append(mydict[camera]["petals"], petal + np.zeros(h["FLUX"].data.shape[0], dtype=int))
                            mydict[camera]["isflag"] = np.append(mydict[camera]["isflag"], (h["FIBERMAP"].data["FIBERSTATUS"] & get_skysub_fiberbitmask_val()) > 0)
                            if tileid is None:
                                tileid = h["FIBERMAP"].header["TILEID"]
                            print("\t", night, expid, camera+str(petal), ((h["FIBERMAP"].data["FIBERSTATUS"] & get_skysub_fiberbitmask_val()) > 0).sum(), "/", sel.sum())
                    print(night, expid, camera, mydict[camera]["isflag"].sum(), "/", mydict[camera]["isflag"].size)
                #
                fig = plt.figure(figsize=(20, 10))
                gs = gridspec.GridSpec(len(cameras), 1, hspace=0.025)
                clim = (-100, 100)
                xlim = (0, 3000)
                for ic, camera in enumerate(cameras):
                    ax = plt.subplot(gs[ic])
                    nsky = 0
                    if "flux" in mydict[camera]:
                        nsky = mydict[camera]["flux"].shape[0]
                        im = ax.imshow(mydict[camera]["flux"], cmap=matplotlib.cm.Greys_r, vmin=clim[0], vmax=clim[1], zorder=0)
                        for petal in petals:
                            ii = np.where(mydict[camera]["petals"] == petal)[0]
                            if len(ii) > 0:
                                ax.plot([0, mydict[camera]["flux"].shape[1]], [ii.min(), ii.min()], color="r", lw=1, zorder=1)
                                ax.text(10, ii.mean(), "{}".format(petal), color="r", fontsize=10, va="center")
                        ax.set_ylim(0, mydict[cameras[0]]["flux"].shape[0])
                        if ic == 1:
                            p =  ax.get_position().get_points().flatten()
                            cax = fig.add_axes([
                                p[0] + 0.85 * (p[2] - p[0]),
                                p[1],
                                0.01 * (p[2] - p[0]),
                                1.0 * (p[3]-p[1])
                            ])
                            cbar = plt.colorbar(im, cax=cax, orientation="vertical", ticklocation="right", pad=0, extend="both")
                            cbar.set_label("FLUX [{}]".format(h["FLUX"].header["BUNIT"]))
                            cbar.mappable.set_clim(clim)
                    ax.text(0.99, 0.92, "CAMERA={}".format(camera), color="k", fontsize=15, fontweight="bold", ha="right", transform=ax.transAxes)
                    if ic == 0:
                        ax.set_title("EXPID={:08d}  NIGHT={}  TILED={}  {} SKY fibers".format(
                            expid, night, tileid, nsky) 
                        )
                    ax.set_xlim(xlim)
                    if ic == 2:
                        ax.set_xlabel("WAVELENGTH direction")
                    ax.set_yticklabels([])
                    ax.set_ylabel("FIBER direction")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close()


def create_tileqa_pdf(outpdf, night, prod, expids, tileids, group='cumulative'):
    """
    For a given night, create a pdf from the tile-qa*png files, sorted by increasing EXPID.

    Args:
        outpdf: output pdf file (string)
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc (string)
        expids: expids of the tiles to display (list or np.array)
        tileids: tiles to display (list or np.array)

    Options:
        group (str): tile group "cumulative" or "pernight"
    """
    # AR exps, to sort by increasing EXPID for that night
    expids, tileids = np.array(expids), np.array(tileids)
    ii = expids.argsort()
    # AR protecting against the empty exposure list case
    if len(expids) > 0:
        expids, tileids = expids[ii], tileids[ii]
        ii = np.array([np.where(tileids == tileid)[0][0] for tileid in np.unique(tileids)])
        expids, tileids = expids[ii], tileids[ii]
        ii = expids.argsort()
        expids, tileids = expids[ii], tileids[ii]
    #
    fns = []
    for tileid in tileids:
        fn = findfile('tileqapng', night=night, tile=tileid, groupname=group, specprod_dir=prod)
        if os.path.isfile(fn):
            fns.append(fn)
        else:
            log.warning("no {}".format(fn))
    # AR creating pdf
    with PdfPages(outpdf) as pdf:
        for fn in fns:
            fig, ax = plt.subplots()
            img = Image.open(fn)
            ax.imshow(img, origin='upper')
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight", dpi=300)
            plt.close()


def create_skyzfiber_png(outpng, night, prod, tileids, dchi2_threshold=9, group='cumulative'):
    """
    For a given night, create a Z vs. FIBER plot for all SKY fibers.

    Args:
        outpdf: output pdf file (string)
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc (string)
        tileids: list of tileids to consider (list or numpy array)
        dchi2_threshold (optional, defaults to 9): DELTACHI2 value to split the sample (float)

    Options:
        group (str): tile group "cumulative" or "pernight"

    Notes:
        Work from the redrock*fits files.
    """
    # AR safe
    tileids = np.unique(tileids)
    # AR gather all infos from the redrock*fits files
    fibers, zs, dchi2s = [], [], []
    nfn = 0
    for tileid in tileids:
        tmp = findfile('redrock', night=night, tile=tileid, groupname=group, spectrograph=0, specprod_dir=prod)
        tiledir = os.path.dirname(tmp)
        fns = sorted(glob(os.path.join(tiledir, f'redrock-?-{tileid}-*{night}.fits')))
        nfn += len(fns)
        for fn in fns:
            fm = fitsio.read(fn, ext="FIBERMAP", columns=["OBJTYPE", "FIBER"])
            rr = fitsio.read(fn, ext="REDSHIFTS", columns=["Z", "DELTACHI2"])
            sel = fm["OBJTYPE"] == "SKY"
            log.info("selecting {} / {} SKY fibers in {}".format(sel.sum(), len(rr), fn))
            fibers += fm["FIBER"][sel].tolist()
            zs += rr["Z"][sel].tolist()
            dchi2s += rr["DELTACHI2"][sel].tolist()
    fibers, zs, dchi2s = np.array(fibers), np.array(zs), np.array(dchi2s)
    # AR plot
    fig, ax = plt.subplots()
    for sel, selname, color in zip(
        [
            dchi2s < dchi2_threshold,
            dchi2s > dchi2_threshold
        ],
        [
            "OBJTYPE=SKY and DELTACHI2<{}".format(dchi2_threshold),
            "OBJTYPE=SKY and DELTACHI2 > {}".format(dchi2_threshold),
        ],
        ["orange", "b"]
    ):
        ax.scatter(fibers[sel], zs[sel], c=color, s=1, alpha=0.1, label="{} ({} fibers)".format(selname, sel.sum()))
    ax.grid()
    ax.set_title("NIGHT = {} ({} fibers from {} redrock*fits files)".format(night, len(fibers), nfn))
    ax.set_xlabel("FIBER")
    ax.set_xlim(-100, 5100)
    ax.set_label("Z")
    ax.set_ylim(-0.1, 6.0)
    ax.legend(loc=2, markerscale=10)
    plt.savefig(outpng, bbox_inches="tight")
    plt.close()


def create_petalnz_pdf(outpdf, night, prod, tileids, surveys, dchi2_threshold=25, group='cumulative'):
    """
    For a given night, create a per-petal, per-tracer n(z) pdf file.

    Args:
        outpdf: output pdf file (string)
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc (string)
        tileids: list of tileids to consider (list or numpy array)
        surveys: list of the surveys for each tileid of tileids (list or numpy array)

    Options:
        dchi2_threshold (optional, defaults to 25): DELTACHI2 value to split the sample (float)
        group (str): tile group "cumulative" or "pernight"

    Notes:
        Only displays:
            - sv1, sv2, sv3, main, as otherwise the plotted tracers are not relevant;
            - FAPRGRM="bright" or "dark" tileids.
        If the tile-qa-TILEID-thruNIGHT.fits file is missing, that tileid is skipped.
        For the Lya, work from the zmtl*fits files, trying to mimick what is done in desitarget.mtl.make_mtl().
        The LRG, ELG, QSO, BGS_BRIGHT, BGS_FAINT bit are the same for sv1, sv2, sv3, main,
            so ok to simply use the bit mask values from the main.
        TBD : we query the FAPRGRM of the tile-qa-*fits header, not sure that properly works for
            surveys other than main..
    """
    petals = np.arange(10, dtype=int)
    # AR safe
    tileids, ii = np.unique(tileids, return_index=True)
    surveys = surveys[ii]
    # AR cutting on sv1, sv2, sv3, main
    sel = np.in1d(surveys, ["sv1", "sv2", "sv3", "main"])
    if sel.sum() > 0:
        log.info(
            "removing {}/{} tileids corresponding to surveys={}, different than sv1, sv2, sv3, main".format(
                (~sel).sum(), tileids.size, ",".join(np.unique(surveys[~sel]).astype(str)),
            )
        )
        tileids, surveys = tileids[sel], surveys[sel]
    #
    # AR gather all infos from the zmtl*fits files
    ds = {"bright" : [], "dark" : []}
    ntiles = {"bright" : 0, "dark" : 0}
    for tileid, survey in zip(tileids, surveys):
        # AR bright or dark?
        fn = findfile('tileqa', night=night, tile=tileid, groupname=group, specprod_dir=prod)
        # AR if no tile-qa*fits, we skip the tileid
        if not os.path.isfile(fn):
            log.warning("no {} file, proceeding to next tile".format(fn))
            continue
        hdr = fits.getheader(fn, "FIBERQA")
        if "FAPRGRM" not in hdr:
            log.warning("no FAPRGRM in {} header, proceeding to next tile".format(fn))
            continue
        faprgrm = hdr["FAPRGRM"].lower()
        if faprgrm not in ["bright", "dark"]:
            log.warning("{} : FAPRGRM={} not in bright, dark, proceeding to next tile".format(fn, faprgrm))
            continue
        # AR reading zmtl files
        istileid = False
        for petal in petals:
            fn = findfile('zmtl', night=night, tile=tileid, spectrograph=petal, groupname=group, specprod_dir=prod)
            if not os.path.isfile(fn):
                log.warning("{} : no file".format(fn))
            else:
                istileid = True
                d = Table.read(fn, hdu="ZMTL")
                # AR rename *DESI_TARGET and *BGS_TARGET to DESI_TARGET and BGS_TARGET
                keys, _, _ = main_cmx_or_sv(d)
                d.rename_column(keys[0], "DESI_TARGET")
                d.rename_column(keys[1], "BGS_TARGET")
                # AR cutting on columns
                d = d[
                    "TARGETID", "DESI_TARGET", "BGS_TARGET",
                    "Z", "ZWARN", "SPECTYPE", "DELTACHI2",
                    "Z_QN", "Z_QN_CONF", "IS_QSO_QN",
                ]
                d["SURVEY"] = np.array([survey for x in range(len(d))], dtype=object)
                d["TILEID"] = np.array([tileid for x in range(len(d))], dtype=int)
                d["PETAL_LOC"] = petal + np.zeros(len(d), dtype=int)
                sel = np.zeros(len(d), dtype=bool)
                if faprgrm == "bright":
                    for msk in ["BGS_BRIGHT", "BGS_FAINT"]:
                        sel |= (d["BGS_TARGET"] & bgs_mask[msk]) > 0
                if faprgrm == "dark":
                    for msk in ["LRG", "ELG", "QSO"]:
                        sel |= (d["DESI_TARGET"] & desi_mask[msk]) > 0
                log.info("selecting {} tracer targets from {}".format(sel.sum(), fn))
                d = d[sel]
                ds[faprgrm].append(d)
        if istileid:
            ntiles[faprgrm] += 1
    # AR stack
    faprgrms, tracers = [], []
    for faprgrm, faprgrm_tracers in zip(
        ["bright", "dark"],
        [["BGS_BRIGHT", "BGS_FAINT"], ["LRG", "ELG", "QSO"]],
    ):
        if len(ds[faprgrm]) > 0:
            ds[faprgrm] = vstack(ds[faprgrm])
            faprgrms += [faprgrm]
            tracers += faprgrm_tracers
    # AR define subsamples
    for faprgrm in faprgrms:
        # AR valid fiber
        valid = np.ones(len(ds[faprgrm]), dtype=bool)
        nodata = ds[faprgrm]["ZWARN"] & desitarget_zwarn_mask["NODATA"] != 0
        badqa = ds[faprgrm]["ZWARN"] & desitarget_zwarn_mask.mask("BAD_SPECQA|BAD_PETALQA") != 0
        ds[faprgrm]["VALID"] = (~nodata) & (~badqa)
        # AR DELTACHI2 above threshold
        ds[faprgrm]["ZOK"] = ds[faprgrm]["DELTACHI2"] > dchi2_threshold
        # AR Lya
        if faprgrm == "dark":
            ds[faprgrm]["LYA"] = (
                (ds[faprgrm]["Z"] >= lya_zcut)
                |
                ((ds[faprgrm]["Z_QN"] >= lya_zcut) & (ds[faprgrm]["IS_QSO_QN"] == 1))
            )
    # AR small internal plot utility function
    def get_tracer_props(tracer):
        if tracer in ["BGS_BRIGHT", "BGS_FAINT"]:
            faprgrm, mask, dtkey = "bright", bgs_mask, "BGS_TARGET"
            xlim, ylim = (-0.2, 1.5), (0, 5.0)
        else:
            faprgrm, mask, dtkey = "dark", desi_mask, "DESI_TARGET"
            if tracer == "LRG":
                xlim, ylim = (-0.2, 2), (0, 3.0)
            elif tracer == "ELG":
                xlim, ylim = (-0.2, 3), (0, 3.0)
            else: # AR QSO
                xlim, ylim = (-0.2, 6), (0, 3.0)
        return faprgrm, mask, dtkey, xlim, ylim
    # AR plot
    #
    # AR color for each tracer
    colors = {
        "BGS_BRIGHT" : "purple",
        "BGS_FAINT" : "c",
        "LRG" : "r",
        "ELG" : "b",
        "QSO" : "orange",
    }
    with PdfPages(outpdf) as pdf:
        # AR we need some tiles to plot!
        if ntiles["bright"] + ntiles["dark"] > 0:
            for survey in np.unique(surveys):
                ntiles_surv = {
                    faprgrm : np.unique(
                        ds[faprgrm]["TILEID"][ds[faprgrm]["SURVEY"] == survey]
                    ).size for faprgrm in faprgrms
                }
                # AR plotting only if some tiles
                if np.sum([ntiles_surv[faprgrm] for faprgrm in faprgrms]) == 0:
                    continue
                # AR three plots:
                # AR - fraction of VALID fibers, bright+dark together
                # AR - fraction of ZOK fibers, per tracer
                # AR - fraction of LYA candidates for QSOs
                fig = plt.figure(figsize=(40, 5))
                gs = gridspec.GridSpec(1, 3, wspace=0.5)
                title = "SURVEY={} : {} tiles from {}".format(
                    survey,
                    " and ".join(["{} {}".format(ntiles_surv[faprgrm], faprgrm.upper()) for faprgrm in faprgrms]),
                    night,
                )
                # AR fraction of ~VALID fibers, bright+dark together
                ax = plt.subplot(gs[0])
                ys = np.nan + np.zeros(len(petals))
                for petal in petals:
                    npet, nvalid = 0, 0
                    for faprgrm in faprgrms:
                        issurvpet = (ds[faprgrm]["SURVEY"] == survey) & (ds[faprgrm]["PETAL_LOC"] == petal)
                        npet += issurvpet.sum()
                        nvalid += ((issurvpet) & (ds[faprgrm]["VALID"])).sum()
                    ys[petal] = nvalid / npet
                ax.plot(petals, ys, "-o", color="k")
                ax.set_title(title)
                ax.set_xlabel("PETAL_LOC")
                ax.set_ylabel("fraction of VALID_fibers")
                ax.xaxis.set_major_locator(MultipleLocator(1))
                ax.set_ylim(0.5, 1.0)
                ax.grid()
                # AR - fraction of ZOK fibers, per tracer (VALID fibers only)
                ax = plt.subplot(gs[1])
                for tracer in tracers:
                    faprgrm, mask, dtkey, _, _ = get_tracer_props(tracer)
                    istracer = ds[faprgrm]["SURVEY"] == survey
                    istracer &= (ds[faprgrm][dtkey] & mask[tracer]) > 0
                    istracer &= ds[faprgrm]["VALID"]
                    ys = np.nan + np.zeros(len(petals))
                    for petal in petals:
                        ispetal = (istracer) & (ds[faprgrm]["PETAL_LOC"] == petal)
                        iszok = (ispetal) & (ds[faprgrm]["ZOK"])
                        ys[petal] = iszok.sum() / ispetal.sum()
                    ax.plot(petals, ys, "-o", color=colors[tracer], label=tracer)
                ax.set_title(title)
                ax.set_xlabel("PETAL_LOC")
                ax.set_ylabel("fraction of DELTACHI2 >_{}\n(VALID fibers only)".format(dchi2_threshold))
                ax.xaxis.set_major_locator(MultipleLocator(1))
                if survey == "main":
                    ax.set_ylim(0.7, 1.0)
                else:
                    ax.set_ylim(0.0, 1.0)
                ax.grid()
                ax.legend()
                # AR - fraction of LYA candidates for QSOs
                ax = plt.subplot(gs[2])
                if "dark" in faprgrms:
                    faprgrm = "dark"
                    ys = np.nan + np.zeros(len(petals))
                    for petal in petals:
                        ispetsurv = (ds[faprgrm]["SURVEY"] == survey) & (ds[faprgrm]["PETAL_LOC"] == petal) & (ds[faprgrm]["VALID"])
                        isqso = (ispetsurv) & ((ds[faprgrm][dtkey] & desi_mask["QSO"]) > 0)
                        islya = (isqso) & (ds[faprgrm]["LYA"])
                        ys[petal] = islya.sum() / isqso.sum()
                    ax.plot(petals, ys, "-o", color=colors["QSO"])
                ax.set_title(title)
                ax.set_xlabel("PETAL_LOC")
                ax.set_ylabel("fraction of LYA candidates\n(VALID QSO fibers only)")
                ax.xaxis.set_major_locator(MultipleLocator(1))
                ax.set_ylim(0, 1)
                ax.grid()
                #
                pdf.savefig(fig, bbox_inches="tight")
                plt.close()
                # AR per-petal, per-tracer n(z)
                for tracer in tracers:
                    faprgrm, mask, dtkey, xlim, ylim = get_tracer_props(tracer)
                    istracer = ds[faprgrm]["SURVEY"] == survey
                    istracer &= (ds[faprgrm][dtkey] & mask[tracer]) > 0
                    istracer &= ds[faprgrm]["VALID"]
                    istracer_zok = (istracer) & (ds[faprgrm]["ZOK"])
                    bins = np.arange(xlim[0], xlim[1] + 0.05, 0.05)
                    #
                    if ntiles_surv[faprgrm] > 0:
                        fig = plt.figure(figsize=(40, 5))
                        gs = gridspec.GridSpec(1, 10, wspace=0.3)
                        for petal in petals:
                            ax = plt.subplot(gs[petal])
                            _ = ax.hist(
                                ds[faprgrm]["Z"][istracer_zok],
                                bins=bins,
                                density=True,
                                histtype="stepfilled",
                                alpha=0.5,
                                color=colors[tracer],
                                label="{} All petals".format(tracer),
                            )
                            _ = ax.hist(
                                ds[faprgrm]["Z"][(istracer_zok) & (ds[faprgrm]["PETAL_LOC"] == petal)],
                                bins=bins,
                                density=True,
                                histtype="step",
                                alpha=1.0,
                                color="k",
                                label="{} PETAL_LOC = {}".format(tracer, petal),
                            )
                            ax.set_title(
                                "{} {}-{} tiles from {}".format(
                                    ntiles_surv[faprgrm],
                                    survey.upper(),
                                    faprgrm.upper(),
                                    night,
                                )
                            )
                            ax.set_xlabel("Z")
                            if petal == 0:
                                ax.set_ylabel("Normalized counts")
                            else:
                                ax.set_yticklabels([])
                            ax.set_xlim(xlim)
                            ax.set_ylim(ylim)
                            ax.grid()
                            ax.set_axisbelow(True)
                            ax.legend(loc=1)
                            ax.text(
                                0.97, 0.8,
                                "DELTACHI2 > {}".format(dchi2_threshold),
                                fontsize=10, fontweight="bold", color="k",
                                ha="right", transform=ax.transAxes,
                            )
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close()


def path_full2web(fn):
    """
    Convert full path to web path (needs DESI_ROOT to be defined).

    Args:
        fn: filename full path

    Returns:
        Web path
    """
    return fn.replace(os.getenv("DESI_ROOT"), "https://data.desi.lbl.gov/desi")


def _javastring():
    """
    Return a string that embeds a date in a webpage.

    Notes:
        Credits to ADM (desitarget/QA.py).
    """
    js = textwrap.dedent(
        """
    <SCRIPT LANGUAGE="JavaScript">
    var months = new Array(13);
    months[1] = "January";
    months[2] = "February";
    months[3] = "March";
    months[4] = "April";
    months[5] = "May";
    months[6] = "June";
    months[7] = "July";
    months[8] = "August";
    months[9] = "September";
    months[10] = "October";
    months[11] = "November";
    months[12] = "December";
    var dateObj = new Date(document.lastModified)
    var lmonth = months[dateObj.getMonth() + 1]
    var date = dateObj.getDate()
    var fyear = dateObj.getYear()
    if (fyear < 2000)
    fyear = fyear + 1900
    if (date == 1 || date == 21 || date == 31)
    document.write(" " + lmonth + " " + date + "st, " + fyear)
    else if (date == 2 || date == 22)
    document.write(" " + lmonth + " " + date + "nd, " + fyear)
    else if (date == 3 || date == 23)
    document.write(" " + lmonth + " " + date + "rd, " + fyear)
    else
    document.write(" " + lmonth + " " + date + "th, " + fyear)
    </SCRIPT>
    """
    )
    return js


def write_html_today(html):
    """
    Write in an html object today's date.

    Args:
        html: html file object.
    """
    html.write(
        "<p style='font-size:1vw; text-align:right'><i>Last updated: {}</p></i>\n".format(
            _javastring()
        )
    )


def write_html_collapse_script(html, classname):
    """
    Write the required lines to have the collapsing sections working.

    Args:
        html: an html file object.
        classname: "collapsible" only for now (string)
    """
    html.write("<script>\n")
    html.write("var coll = document.getElementsByClassName('{}');\n".format(classname))
    html.write("var i;\n")
    html.write("for (i = 0; i < coll.length; i++) {\n")
    html.write("\tcoll[i].addEventListener('click', function() {\n")
    html.write("\t\tthis.classList.toggle('{}');\n".format(classname.replace("collapsible", "active")))
    html.write("\t\tvar content = this.nextElementSibling;\n")
    html.write("\t\tif (content.style.display === 'block') {\n")
    html.write("\t\tcontent.style.display = 'none';\n")
    html.write("\t\t} else {\n")
    html.write("\t\t\tcontent.style.display = 'block';\n")
    html.write("\t\t}\n")
    html.write("\t});\n")
    html.write("}\n")
    html.write("</script>\n")



def write_nightqa_html(outfns, night, prod, css, surveys=None, nexp=None, ntile=None):
    """
    Write the nightqa-{NIGHT}.html page.

    Args:
        outfns: dictionary with filenames generated by desi_night_qa (output from get_nightqa_outfns)
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc (string)
        css: path to the nightqa.css file
        surveys (optional, defaults to None): considered surveys (string)
        nexp (optional, defaults to None): number of considered exposures (int)
        ntile (optional, defaults to None): number of considered tiles (int)
    """
    # ADM html preamble.
    html = open(outfns["html"], "w")
    html.write("<html><body>\n")
    html.write("<h1>Night QA page for {}</h1>\n".format(night))
    html.write("\n")

    #
    html.write("<head>\n")
    html.write("\t<meta charset='UTF-8'>\n")
    html.write("\t<meta http-equiv='X-UA-Compatible' content='IE=edge'>\n")
    html.write(
        "\t<meta name='viewport' content='width=device-width, initial-scale=1.0'>\n"
    )
    html.write("\t<link rel='stylesheet' href='{}'>\n".format(path_full2web(css)))
    html.write("</head>\n")
    html.write("\n")
    html.write("<body>\n")
    html.write("\n")
    #
    html.write("\t<p>For {}, {} exposures from {} {} tiles are analyzed.</p>\n".format(night, nexp, ntile, surveys))
    html.write("\t<p>Please click on each tab from top to bottom, and follow instructions.</p>\n")

    # AR night log
    # AR testing different possible names
    nightdir = os.path.join(os.getenv("DESI_ROOT"), "survey", "ops", "nightlogs", "{}".format(night))
    nightfn = None
    for basename in [
        "NightSummary{}.html".format(night),
        "nightlog_kpno.html",
        "nightlog_nersc.html",
        "nightlog.html",
    ]:
        if nightfn is None:
            if os.path.isfile(os.path.join(nightdir, basename)):
                nightfn = os.path.join(os.path.join(nightdir, basename))
    html.write(
        "<button type='button' class='collapsible'>\n\t<strong>{} Night summary</strong>\n</button>\n".format(
            night,
        )
    )
    html.write("<div class='content'>\n")
    html.write("\t<br>\n")
    if nightfn is not None:
        html.write("\t<p>Read the nightlog for {}: {}, displayed below.</p>\n".format(night, path_full2web(nightfn)))
        html.write("\t<p>And consider subscribing to the desi-nightlog mailing list!\n")
        html.write("\t</br>\n")
        html.write("\t<br>\n")
        html.write("\t<iframe src='{}' width=100% height=100%></iframe>\n".format(path_full2web(nightfn)))
        html.write("\t<p>And consider subscribing to the desi-nightlog mailing list!\n")
    else:
        html.write("\t<p>No found nightlog for in {}</p>\n".format(path_full2web(nightdir)))
    html.write("\t</br>\n")
    html.write("</div>\n")
    html.write("\n")

    # AR calibnight: flat and psf
    # AR color-coding:
    # AR - red : file does not exist
    # AR - blue : file exists, but is a symlink
    # AR - green : file exists
    cameras = ["b", "r", "z"]
    petals = np.arange(10, dtype=int)
    #
    html.write(
        "<button type='button' class='collapsible'>\n\t<strong>{} calibnight</strong>\n</button>\n".format(
            night,
        )
    )
    html.write("<div class='content'>\n")
    html.write("\t<br>\n")
    html.write("\t<p>Assess the presence of all psfnight and fiberflatnight files for {}.</p>\n".format(night))
    html.write("\t<p>If a file appears in <span style='color:green;'>green</span>, it means it is present.</p>\n")
    html.write("\t<p>If a file appears in <span style='color:blue;'>blue</span>, it means it is a symlink to another file, the name of which is reported.</p>\n")
    html.write("\t<p>If a file appears in <span style='color:red;'>red</span>, it means it is missing.</p>\n")
    html.write("\t</br>\n")
    html.write("<table>\n")
    for petal in petals:
        html.write("\t<tr>\n")
        for case in ["psfnight", "fiberflatnight"]:
            for camera in cameras:
                fn = os.path.join(
                    prod,
                    "calibnight",
                    "{}".format(night),
                    "{}-{}{}-{}.fits".format(case, camera, petal, night),
                )
                fnshort, color = os.path.basename(fn), "red"
                if os.path.isfile(fn):
                    if os.path.islink(fn):
                        fnshort, color = os.path.basename(os.readlink(fn)), "blue"
                    else:
                        color = "green"
                    html.write("\t\t<td> <span style='color:{};'>{}</span> </td>\n".format(color, fnshort))
                if camera != cameras[-1]:
                    html.write("\t\t<td> &emsp; </td>\n")
            if case == "psfnight":
                html.write("\t\t<td> &emsp; &emsp; &emsp; </td>\n")
        html.write("\t</tr>\n")
    html.write("</table>\n")
    html.write("</div>\n")
    html.write("\n")

    # AR various tabs:
    # AR - dark
    # AR - badcol
    # AR - ctedet
    # AR - sframesky
    # AR - tileqa
    # AR - skyzfiber
    # AR - petalnz
    for case, caselab, width, text in zip(
        ["dark", "badcol", "ctedet", "sframesky", "tileqa", "skyzfiber", "petalnz"],
        ["DARK", "bad columns", "CTE detector", "sframesky", "Tile QA", "SKY Z vs. FIBER", "Per-petal n(z)"],
        ["100%", "35%", "100%", "75%", "90%", "35%", "100%"],
        [
            "This pdf displays the 300s (binned) DARK (one page per spectrograph; non-valid pixels are displayed in red)\nWatch it and report unsual features (easy to say!)",
            "This plot displays the histograms of the bad columns.\nWatch it and report unsual features (easy to say!)",
            "This pdf displays a small diagnosis to detect CTE anormal behaviour (one petal-camera per page)\nWatch it and report unusual features (typically if the lower enveloppe of the blue or orange curve is systematically lower than the other one).",
            "This pdf displays the sframe image for the sky fibers for each Main exposure (one exposure per page).\nWatch it and report unsual features (easy to say!)",
            "This pdf displays the tile-qa-TILEID-thru{}.png files for the Main tiles (one tile per page).\nWatch it, in particular the Z vs. FIBER plot, and report unsual features (easy to say!)".format(night),
            "This plot displays all the SKY fibers for the {} night.\nWatch it and report unsual features (easy to say!)".format(night),
            "This pdf displays the per-tracer, per-petal n(z) for the {} night.\nWatch it and report unsual features (easy to say!)".format(night),
        ]
    ):
        html.write(
            "<button type='button' class='collapsible'>\n\t<strong>{} {}</strong>\n</button>\n".format(
                night, caselab,
            )
        )
        html.write("<div class='content'>\n")
        html.write("\t<br>\n")
        if os.path.isfile(outfns[case]):
            for text_split in text.split("\n"):
                html.write("\t<p>{}</p>\n".format(text_split))
            html.write("\t<tr>\n")
            if os.path.splitext(outfns[case])[-1] == ".png":
                outpng = path_full2web(outfns[case])
                html.write(
                    "\t<a href='{}'><img SRC='{}' width={} height=auto></a>\n".format(
                        outpng, outpng, width,
                    )
                )
            elif os.path.splitext(outfns[case])[-1] == ".pdf":
                outpdf = path_full2web(outfns[case])
                html.write("\t<iframe src='{}' width={} height=100%></iframe>\n".format(outpdf, width))
            else:
                log.error("Unexpected extension for {}".format(outfns[case]))
                raise RuntimeError("Unexpected extension for {}".format(outfns[case]))
        else:
            html.write("\t<p>No {}.</p>\n".format(path_full2web(outfns[case])))
        html.write("\t</br>\n")
        html.write("</div>\n")
        html.write("\n")

    # AR lines to make collapsing sections
    write_html_collapse_script(html, "collapsible")

    # ADM html postamble for main page.
    write_html_today(html)
    html.write("</html></body>\n")
    html.close()

