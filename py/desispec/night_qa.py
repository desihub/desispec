"""
desispec.night_qa
=================

"""
# AR general
import sys
import os
from glob import glob
import tempfile
import textwrap
from desiutil.log import get_logger
import multiprocessing
from datetime import datetime
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
from desitarget.geomask import match_to
# AR desispec
from desispec.fiberbitmasking import get_skysub_fiberbitmask_val
from desispec.io import findfile
from desispec.calibfinder import CalibFinder
from desispec.scripts import preproc
from desispec.tile_qa_plot import get_tilecov
# AR matplotlib
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator
# AR PIL (to create pdf from pngs)
from PIL import Image

log = get_logger()

cameras = ["b", "r", "z"]
petals = np.arange(10, dtype=int)

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
        "morningdark" : os.path.join(outdir, "morningdark-{}.pdf".format(night)),
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

    Note:
        Based on:
        - parsing the OBSTYPE keywords from the SPEC extension header of the desi-{EXPID}.fits.fz files;
        - for OBSTYPE="SCIENCE", parsing the ``fiberassign-TILEID.fits*`` header
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
        hdr = fitsio.read_header(fns[i], "SPEC")
        if hdr["OBSTYPE"] == "SCIENCE":
            survey = "unknown"
            # AR look for the fiberassign file
            # AR - used wildcard, because early files (pre-SV1?) were not gzipped
            # AR - first check SURVEY keyword (should work for SV3 and later)
            # AR - if not present, take FA_SURV
            fafns = glob(os.path.join(os.path.dirname(fns[i]), "fiberassign-??????.fits*"))
            if len(fafns) > 0:
                fahdr = fitsio.read_header(fafns[0], 0)
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

    Note:
        * If nothing found, returns None.
        * 20220110 : new method, relying on processing_tables
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


def get_morning_dark_night_expid(night, prod, exptime=1200):
    """
    Returns the EXPID of the latest 1200s DARK exposure for a given night.

    Args:
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc (string)
        exptime (optional, defaults to 1200): exposure time we are looking for (float)

    Returns:
        expid: EXPID (int)

    Note:
        * If nothing found, returns None.
        * As of now (20221220), those morning darks are not processed by the daily
          pipeline; hence we do not use the processing_tables, but the
          exposure_tables.
    """
    #
    expid = None
    exptable_fn = os.path.join(
        prod,
        "exposure_tables",
        str(night // 100),
        "exposure_table_{}.csv".format(night),
    )
    log.info("exptable_fn = {}".format(exptable_fn))
    if not os.path.isfile(exptable_fn):
        log.warning("no {} found; returning None".format(exptable_fn))
    else:
        d = Table.read(exptable_fn)
        sel = d["OBSTYPE"] == "dark"
        sel &= d["COMMENTS"] == "|" # AR we request an exposure with no known issue
        sel &= np.abs(d["EXPTIME"] - exptime) < 1
        if sel.sum() == 0:
            log.warning(
                "found zero exposures with OBSTYPE=dark and COMMENTS='|' and abs(EXPTIME-{})<1 in expable_fn; returning None".format(
                    exptime,
                )
            )
        else:
            d = d[sel]
            # AR pick the latest one
            d = d[d["MJD-OBS"].argsort()]
            expid = d["EXPID"][-1]
            log.info(
                "found EXPID={} as the latest {}s DARK for NIGHT={}".format(
                    expid, exptime, night,
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

    Note:
        * If nothing found, returns None.
        * We look for preproc files.
        * As we are looking for a faint signal, we want the image with the less electrons,
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
                    "preproc-??-{:08d}.fits*".format(expid),
                )
            )
        )
        # AR if some preproc files, just pick the first one
        if len(fns) > 0:
            hdr = fitsio.read_header(fns[0], "IMAGE")
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
                        "sky-r?-{:08d}.fits*".format(expid),
                    )
                )
            )
            # AR if some sky files, just pick the first one
            if len(fns) > 0:
                hdr = fitsio.read_header(fns[0], "SKY")
                if hdr["OBSTYPE"] == "SCIENCE":
                    sky = np.median(fitsio.read(fns[0], "SKY"))
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

    Note:
        * Requires ffmpeg to be installed.
        * At NERSC, run in the bash command line: "module load ffmpeg".
        * The movie uses fns in the provided order.
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


def _read_dark(fn, night, prod, dark_expid, petal, camera, binning=4):
    """
    Internal function used by create_dark_pdf(), to read and bin the 300s dark.

    Args:
        fn: full path to the dark image (string)
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc (string)
        dark_expid: EXPID of the 300s DARK exposure to display (int)
        petal: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (int)
        camera: "b", "r", or "z" (string)
        binning (optional, defaults to 4): binning of the image (which will be beforehand trimmed) (int)

    Returns:
        mydict: a dictionary with the binned+masked image, plus various infos,
        assuming a preproc file is available. Otherwise ``None``.
    """
    if os.path.isfile(fn):
        mydict = {}
        mydict["dark_expid"] = dark_expid
        mydict["petal"] = petal
        mydict["camera"] = camera
        log.info("reading {}".format(fn))
        with fitsio.FITS(fn) as h:
            image, ivar, mask = h["IMAGE"].read(), h["IVAR"].read(), h["MASK"].read()
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
        mydict["image"] = d_bin_msk
        return mydict
    else:
        log.warning("missing {}".format(fn))
        return None


def create_dark_pdf(outpdf, night, prod, dark_expid, nproc, binning=4):
    """
    For a given night, create a pdf with the 300s binned dark.

    Args:
        outpdf: output pdf file (string)
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc (string)
        dark_expid: EXPID of the 300s or 1200s DARK exposure to display (int)
        nproc: number of processes running at the same time (int)
        binning (optional, defaults to 4): binning of the image (which will be beforehand trimmed) (int)

    Note:
        If the identified dark image is not processed and if the raw data is there,
        we do process it (in a temporary folder), so that we can generate the plots.
    """
    # AR first check if we need to process this dark image
    proctable_fn = os.path.join(
        prod,
        "processing_tables",
        "processing_table_{}-{}.csv".format(os.path.basename(prod), night),
    )
    run_preproc = False
    if not os.path.isfile(proctable_fn):
        run_preproc = True
    else:
        d = Table.read(proctable_fn)
        sel = d["OBSTYPE"] == "dark"
        d = d[sel]
        proc_expids = [int(expid.strip("|")) for expid in d["EXPID"]]
        if dark_expid not in proc_expids:
            run_preproc = True
    # AR run preproc?
    if run_preproc:
        # AR does the raw exposure exist?
        rawfn = findfile("raw", night, dark_expid)
        if os.path.isfile(rawfn):
            specprod_dir = tempfile.mkdtemp()
            outdir = os.path.join(specprod_dir, "preproc", str(night), "{:08d}".format(dark_expid))
            os.makedirs(outdir, exist_ok=True)
            cmd = "desi_preproc -n {} -e {} --outdir {} --ncpu {}".format(
                night, dark_expid, outdir, nproc,
            )
            log.info("run: {}".format(cmd))

            # like os.system(cmd), but avoids system call for MPI compatibility
            preproc.main(cmd.split()[1:])

        # AR if we reached this stage, we expect the raw data to be there
        else:
            msg = "no raw image {} -> skipping".format(rawfn)
            log.error(msg)
            raise ValueError(msg)
    else:
        specprod_dir = prod
    #
    myargs = []
    for petal in petals:
        for camera in cameras:
            myargs.append(
                [
                    findfile("preproc", night, dark_expid, camera+str(petal), specprod_dir=specprod_dir),
                    night,
                    prod,
                    dark_expid,
                    petal,
                    camera,
                    binning,
                ]
            )
    # AR launching pool
    pool = multiprocessing.Pool(processes=nproc)
    with pool:
        mydicts = pool.starmap(_read_dark, myargs)
    # AR plotting
    # AR remarks:
    # AR - the (x,y) conversions for the side panels
    # AR    are a bit counter-intuitive, as ax.imshow()
    # AR    reverses the displayed axes...
    # AR - the panels positioning is not very elegant,
    # AR    probably could be coded in a nicer way!
    clim = (-5, 5)
    cmap = matplotlib.cm.Greys_r
    cmap.set_bad(color="r")
    width_ratios = 0.1 + np.zeros(2 * len(cameras))
    width_ratios[::2] = 1
    tmpcols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    with PdfPages(outpdf) as pdf:
        for petal in petals:
            fig = plt.figure(figsize=(20, 10))
            gs = gridspec.GridSpec(2, 2 * len(cameras), wspace=0.1, width_ratios = width_ratios, hspace=0.0, height_ratios = [0.05, 1])
            for ic, camera in enumerate(cameras):
                ax = fig.add_subplot(gs[1, 2 * ic])
                ax_y = fig.add_subplot(gs[0, 2 * ic])
                ax_x = fig.add_subplot(gs[1, 2 * ic + 1])
                ii = [i for i in range(len(myargs)) if myargs[i][4] == petal and myargs[i][5] == camera]
                assert(len(ii) == 1)
                mydict = mydicts[ii[0]]
                if mydict is not None:
                    assert(mydict["petal"] == petal)
                    assert(mydict["camera"] == camera)
                    img = mydict["image"]
                    im = ax.imshow(img, cmap=cmap, vmin=clim[0], vmax=clim[1])
                    pos = ax.get_position().bounds
                    # AR median profile along x, for each pair of amps
                    tmpxs = np.nanmedian(img[:, : img.shape[1] // 2], axis=1)
                    tmpys = np.arange(len(tmpxs))
                    ax_x.plot(tmpxs, tmpys, color=tmpcols[0], alpha=0.5, zorder=1)
                    tmpxs = np.nanmedian(img[:, img.shape[1] // 2 :], axis=1)
                    tmpys = np.arange(len(tmpxs))
                    ax_x.plot(tmpxs, tmpys, color=tmpcols[1], alpha=0.5, zorder=1)
                    ax_x.set_ylim(ax.get_xlim()[::-1])
                    ax_x.set_xlim(-0.5, 0.5)
                    ax_x.set_yticklabels([])
                    ax_x.grid()
                    pos_x =  list(ax_x.get_position().bounds)
                    pos_x[0], pos_x[1], pos_x[3] = pos[0] + pos[2], pos[1], pos[3]
                    ax_x.set_position(pos_x)
                    # AR median profile along y, for each pair of amps
                    tmpys = np.nanmedian(img[: img.shape[0] // 2, :], axis=0)
                    tmpxs = np.arange(len(tmpys))
                    ax_y.plot(tmpxs, tmpys, color=tmpcols[0], alpha=0.5, zorder=1)
                    tmpys = np.nanmedian(img[img.shape[0] // 2 :, :], axis=0)
                    tmpxs = np.arange(len(tmpys))
                    ax_y.plot(tmpxs, tmpys, color=tmpcols[1], alpha=0.5, zorder=1)
                    ax_y.set_title("EXPID={} {}{}".format(dark_expid, camera, petal))
                    ax_y.set_xlim(ax.get_ylim()[::-1])
                    ax_y.set_ylim(-0.5, 0.5)
                    ax_y.set_xticklabels([])
                    ax_y.grid()
                    pos_y =  list(ax_y.get_position().bounds)
                    pos_y[0], pos_y[1], pos_y[2] = pos[0], pos[1] + pos[3], pos[2]
                    ax_y.set_position(pos_y)
                    if camera == cameras[-1]:
                        p = ax_x.get_position().get_points().flatten()
                        cax = fig.add_axes([
                            p[0] + 1.5 * (p[2] - p[0]),
                            p[1],
                            0.5 * (p[2] - p[0]),
                            1.0 * (p[3] - p[1])
                        ])
                        cbar = plt.colorbar(im, cax=cax, orientation="vertical", ticklocation="right", pad=0, extend="both")
                        cbar.set_label("Units : ?")
                        cbar.mappable.set_clim(clim)
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
    colors = ["b", "r", "k"]
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


def _read_ctedet(night, prod, ctedet_expid, petal, camera):
    """
    Internal function used by create_ctedet_pdf(), reading the ctedet_expid preproc info.

    Args:
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc (string)
        ctedet_expid: EXPID for the CTE diagnosis (1s FLAT, or darker science exposure) (int)
        petal: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (int)
        camera: "b", "r", or "z" (string)

    Returns:
        mydict: a dictionary with the IMAGE data, plus various infos, assuming a preproc file is available, otherwise ``None``.
    """
    #
    fn = findfile("preproc", night, ctedet_expid, camera+str(petal),
            specprod_dir=prod)
    if os.path.isfile(fn):
        mydict = {}
        mydict["fn"] = fn
        # AR read
        with fitsio.FITS(fn) as fx:
            mydict["img"] = fx["IMAGE"].read()
        # AR check if we re displaying a 1s FLAT
        hdr = fitsio.read_header(fn, "IMAGE")
        if (hdr["OBSTYPE"] == "FLAT") & (hdr["REQTIME"] == 1):
            mydict["is_onesec_flat"] = True
        else:
            mydict["is_onesec_flat"] = False
        # AR grab columns with identified problem
        cfinder = CalibFinder([hdr])
        for key in ["OFFCOLSA", "OFFCOLSB", "OFFCOLSC", "OFFCOLSD"]:
            if cfinder.haskey(key):
                mydict[key] = cfinder.value(key)
            else:
                 mydict[key] = None
        return mydict
    else:
        return None


def create_ctedet_pdf(outpdf, night, prod, ctedet_expid, nproc, nrow=21, xmin=None, xmax=None, ylim=(-5, 10),
    yoffcols={"A" : -2.5, "B" : -3.5, "C" : -2.5, "D" : -3.5},
):
    """
    For a given night, create a pdf with a CTE diagnosis (from preproc files).

    Args:
        outpdf: output pdf file (string)
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc (string)
        ctedet_expid: EXPID for the CTE diagnosis (1s FLAT, or darker science exposure) (int)
        nproc: number of processes running at the same time (int)
        nrow (optional, defaults to 21): number of rows to include in median (int)
        xmin (optional, defaults to None): minimum column to display (float)
        xmax (optional, defaults to None): maximum column to display (float)
        ylim (optional, default to (-5, 10)): ylim for the median plot (duplet)
        yoffcols (optional, defaults to {"A" : -2.5, "B" : -3.5, "C" : -2.5, "D" : -3.5}):
            y-values to report the per-amplifier OFFCOLS info, if any (dictionnary of floats)

    Note:
        * Credits to S. Bailey.
        * Copied-pasted-adapted from /global/homes/s/sjbailey/desi/dev/ccd/plot-amp-cte.py
    """
    myargs = []
    for petal in petals:
        for camera in cameras:
            myargs.append(
                [
                    night,
                    prod,
                    ctedet_expid,
                    petal,
                    camera,
                ]
            )
    # AR launching pool
    pool = multiprocessing.Pool(processes=nproc)
    with pool:
        mydicts = pool.starmap(_read_ctedet, myargs)
    # AR plotting
    clim = (-5, 5)
    tmpcols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {"A" : tmpcols[1], "B" : tmpcols[1], "C" : tmpcols[0], "D" : tmpcols[0]}
    with PdfPages(outpdf) as pdf:
        for mydict in mydicts:
            petcam_xmin, petcam_xmax = xmin, xmax
            fig = plt.figure(figsize=(30, 5))
            gs = gridspec.GridSpec(2, 1, wspace=0.1, height_ratios = [1, 4])
            ax2d = plt.subplot(gs[0])
            ax1d = plt.subplot(gs[1])
            if mydict is not None:
                ax1d.set_title(
                "{}\nMedian of {} rows above/below CCD amp boundary".format(
                    mydict["fn"], nrow,
                )
            )
                # AR is it a 1s FLAT image?
                if not mydict["is_onesec_flat"]:
                    ax1d.text(
                        0.5, 0.7,
                        "WARNING: not displaying a 1s FLAT image",
                        color="k", fontsize=20, fontweight="bold", ha="center", va="center",
                        transform=ax1d.transAxes,
                    )
                img = mydict["img"]
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
                # AR known columns with offset?
                # AR     stored in format like "12:24,1700:1900"
                for amp in ["A", "B", "C", "D"]:
                    key = "OFFCOLS{}".format(amp)
                    if mydict[key] is not None:
                        for colrange in mydict[key].split(","):
                            colmin, colmax = int(colrange.split(":")[0]), int(colrange.split(":")[1])
                            ax1d.annotate("", xy=(colmax, yoffcols[amp]), xytext=(colmin, yoffcols[amp]), arrowprops=dict(arrowstyle="<->", lw="3", color=colors[amp]))
                            if amp in ["A", "C"]:
                                tmpy = yoffcols[amp] + 0.025 * (ylim[1] - ylim[0])
                            else:
                                tmpy = yoffcols[amp] - 0.030 * (ylim[1] - ylim[0])
                            ax1d.text(0.5 * (colmin + colmax), tmpy, "AMP{} known offset: {}".format(amp, colrange), ha="center", va="center")
                # AR plot 1d median
                ax1d.plot(xx, above, alpha=0.5, color=colors["C"], label="above (AMPC : x < {}; AMPD : x > {}".format(nx // 2 - 1, nx // 2 -1))
                ax1d.plot(xx, below, alpha=0.5, color=colors["A"], label="below (AMPA : x < {}; AMPB : x > {}".format(nx // 2 - 1, nx // 2 -1))
                ax1d.legend(loc=2)
                # AR amplifier x-boundary
                ax1d.axvline(nx // 2 - 1, color="k", ls="--")
                ax1d.set_xlabel("CCD column")
                ax1d.set_xlim(petcam_xmin, petcam_xmax)
                ax1d.set_ylim(ylim)
                ax1d.grid()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()


def _read_sframesky(night, prod, expid):
    """
    Internal function called by create_sframesky_pdf, which generates
    the per-expid sframesky plot.

    Args:
        outpng: output png file (string)
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc (string)
        expid: expid to display (int)

    Returns:
        mydict, dictionary with per-camera wave, flux, and various infos, assuming sframe files were available, otherwise ``None``.
    """
    #
    nightdir = os.path.join(prod, "exposures", "{}".format(night))
    #
    tileid = None
    fns = sorted(
        glob(
            os.path.join(
                nightdir,
                "{:08d}".format(expid),
                "sframe-??-{:08d}.fits*".format(expid),
            )
        )
    )
    if len(fns) > 0:
        # AR read the sframe sky fibers
        mydict = {camera : {} for camera in cameras}
        for ic, camera in enumerate(cameras):
            for petal in petals:
                fn, exists = findfile('sframe', night, expid, camera+str(petal),
                        specprod_dir=prod, return_exists=True)
                if exists:
                    with fitsio.FITS(fn) as h:
                        fibermap = h["FIBERMAP"].read()
                        sel = fibermap["OBJTYPE"] == "SKY"
                        fibermap = fibermap[sel]
                        flux = h["FLUX"].read()[sel]
                        ivar = h["IVAR"].read()[sel]
                        wave = h["WAVELENGTH"].read()
                        flux_header = h["FLUX"].read_header()
                        fibermap_header = h["FIBERMAP"].read_header()

                    if "flux" not in mydict[camera]:
                        mydict[camera]["wave"] = wave
                        nwave = len(mydict[camera]["wave"])
                        mydict[camera]["petals"] = np.zeros(0, dtype=int)
                        mydict[camera]["flux"] = np.zeros(0).reshape((0, nwave))
                        mydict[camera]["nullivar"] = np.zeros(0, dtype=bool).reshape((0, nwave))
                        mydict[camera]["isflag"] = np.zeros(0, dtype=bool)
                    mydict[camera]["flux"] =  np.append(mydict[camera]["flux"], flux, axis=0)
                    mydict[camera]["nullivar"] = np.append(mydict[camera]["nullivar"], ivar == 0, axis=0)
                    mydict[camera]["petals"] = np.append(mydict[camera]["petals"], petal + np.zeros(flux.shape[0], dtype=int))
                    band = camera.lower()[0]
                    mydict[camera]["isflag"] = np.append(mydict[camera]["isflag"], (fibermap["FIBERSTATUS"] & get_skysub_fiberbitmask_val(band=band)) > 0)
                    if tileid is None:
                        tileid = fibermap_header["TILEID"]
                    print("\t", night, expid, camera+str(petal), ((fibermap["FIBERSTATUS"] & get_skysub_fiberbitmask_val(band=band)) > 0).sum(), "/", sel.sum())
            print(night, expid, camera, mydict[camera]["isflag"].sum(), "/", mydict[camera]["isflag"].size)
        # AR various infos
        mydict["expid"] = expid
        mydict["night"] = night
        mydict["tileid"] = tileid
        mydict["flux_unit"] = flux_header["BUNIT"]
        #
        return mydict
    else:
        return None


def create_sframesky_pdf(outpdf, night, prod, expids, nproc):
    """
    For a given night, create a pdf from per-expid sframe for the sky fibers only.

    Args:
        outpdf: output pdf file (string)
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc (string)
        expids: expids to display (list or np.array)
        nproc: number of processes running at the same time (int)
    """
    #
    nightdir = os.path.join(prod, "exposures", "{}".format(night))
    # AR sorting the EXPIDs by increasing order
    myargs = []
    for expid in np.sort(expids):
        myargs.append(
            [
                night,
                prod,
                expid,
            ]
        )
    # AR launching pool
    pool = multiprocessing.Pool(processes=nproc)
    with pool:
        mydicts = pool.starmap(_read_sframesky, myargs)
    # AR creating pdf (+ removing temporary files)
    cmap = matplotlib.cm.Greys_r
    with PdfPages(outpdf) as pdf:
        for mydict in mydicts:
            if mydict is not None:
                fig = plt.figure(figsize=(20, 10))
                gs = gridspec.GridSpec(len(cameras), 1, hspace=0.025)
                clim = (-100, 100)
                xlim = (0, 3000)
                for ic, camera in enumerate(cameras):
                    ax = plt.subplot(gs[ic])
                    nsky = 0
                    if "flux" in mydict[camera]:
                        nsky = mydict[camera]["flux"].shape[0]
                        im = ax.imshow(mydict[camera]["flux"], cmap=cmap, vmin=clim[0], vmax=clim[1], zorder=0)
                        # AR overlay in transparent pixels with ivar=0
                        # AR a bit obscure why I need to add +1 in xs, ys...
                        # AR probably some indexing convention in imshow()
                        xys = np.argwhere(mydict[camera]["nullivar"])
                        xs, ys = 1 + xys[:, 0], 1 + xys[:, 1]
                        ax.scatter(ys, xs, c="g", s=0.1, alpha=0.1, zorder=1, rasterized=True)
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
                            cbar.set_label("FLUX [{}]".format(mydict["flux_unit"]))
                            cbar.mappable.set_clim(clim)
                    ax.text(0.99, 0.92, "CAMERA={}".format(camera), color="k", fontsize=15, fontweight="bold", ha="right", transform=ax.transAxes)
                    if ic == 0:
                        ax.set_title("EXPID={:08d}  NIGHT={}  TILEID={}  {} SKY fibers".format(
                            mydict["expid"], mydict["night"], mydict["tileid"], nsky)
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
        group (str, optional): tile group "cumulative" or "pernight"
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


def create_skyzfiber_png(outpng, night, prod, tileids, dchi2_threshold=9, group="cumulative"):
    """
    For a given night, create a Z vs. FIBER plot for all SKY fibers, and one for
        each of the main backup/bright/dark programs

    Args:
        outpdf: output pdf file (string)
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc (string)
        tileids: list of tileids to consider (list or numpy array)
        dchi2_threshold (optional, defaults to 9): DELTACHI2 value to split the sample (float)
        group (str, optional): tile group "cumulative" or "pernight"

    Note:
        Work from the ``redrock-*.fits`` files.
    """
    # AR safe
    tileids = np.unique(tileids)
    # AR gather all infos from the redrock*fits files
    fibers, zs, dchi2s, faflavors = [], [], [], []
    nfn = 0
    for tileid in tileids:
        # AR main backup/bright/dark ?
        faflavor = None
        fns = sorted(
            glob(
                os.path.join(
                    os.getenv("DESI_ROOT"),
                    "spectro",
                    "data",
                    "{}".format(night),
                    "*",
                    "fiberassign-{:06d}.fits*".format(tileid),
                )
            )
        )
        if len(fns) > 0:
            hdr = fitsio.read_header(fns[0], 0)
            if "FAFLAVOR" in hdr:
                faflavor = hdr["FAFLAVOR"]
        log.info("identified FAFLAVOR for {}: {}".format(tileid, faflavor))
        # AR
        fns = []
        for petal in petals:
            fn, exists = findfile(
                "redrock",
                night=night,
                tile=tileid,
                groupname=group,
                spectrograph=petal,
                specprod_dir=prod,
                return_exists=True,
            )
            if exists:
                fns.append(fn)
            else:
                log.warning("no {}".format(fn))
        nfn += len(fns)
        for fn in fns:
            fm = fitsio.read(fn, ext="FIBERMAP", columns=["OBJTYPE", "FIBER"])
            rr = fitsio.read(fn, ext="REDSHIFTS", columns=["Z", "DELTACHI2"])
            sel = fm["OBJTYPE"] == "SKY"
            log.info("selecting {} / {} SKY fibers in {}".format(sel.sum(), len(rr), fn))
            fibers += fm["FIBER"][sel].tolist()
            zs += rr["Z"][sel].tolist()
            dchi2s += rr["DELTACHI2"][sel].tolist()
            faflavors += [faflavor for x in range(sel.sum())]
    fibers, zs, dchi2s, faflavors = np.array(fibers), np.array(zs), np.array(dchi2s), np.array(faflavors, dtype=str)
    # AR plot
    plot_faflavors = ["all", "mainbackup", "mainbright", "maindark"]
    ylim = (-1.1, 1.1)
    yticks = np.array([0, 0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 6])
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, len(plot_faflavors), wspace=0.1)
    for ip, plot_faflavor in enumerate(plot_faflavors):
        ax = plt.subplot(gs[ip])
        if plot_faflavor == "all":
            faflavor_sel = np.ones(len(fibers), dtype=bool)
            title = "NIGHT = {}\nAll tiles ({} fibers)".format(night, len(fibers))
        else:
            faflavor_sel = faflavors == plot_faflavor
            title = "NIGHT = {}\nFAFLAVOR={} ({} fibers)".format(night, plot_faflavor, faflavor_sel.sum())
        if faflavor_sel.sum() < 5000:
            alpha = 0.3
        else:
            alpha = 0.1
        for sel, selname, color in zip(
            [
                (faflavor_sel) & (dchi2s < dchi2_threshold),
                (faflavor_sel) & (dchi2s > dchi2_threshold),
            ],
            [
                "OBJTYPE=SKY and DELTACHI2<{}".format(dchi2_threshold),
                "OBJTYPE=SKY and DELTACHI2>{}".format(dchi2_threshold),
            ],
            ["orange", "b"]
        ):
            ax.scatter(fibers[sel], np.log10(0.1 + zs[sel]), c=color, s=1, alpha=alpha, label="{} ({} fibers)".format(selname, sel.sum()))
        ax.grid()
        ax.set_title(title)
        ax.set_xlabel("FIBER")
        ax.set_xlim(-100, 5100)
        if ip == 0:
            ax.set_ylabel("Z")
        ax.set_ylim(ylim)
        ax.set_yticks(np.log10(0.1 + yticks))
        ax.set_yticklabels(yticks.astype(str))
        ax.legend(loc=2, markerscale=10)
    plt.savefig(outpng, bbox_inches="tight")
    plt.close()


def plot_newlya(
    ax,
    ntilecovs,
    nnewlyas,
    tileids=None,
    title=None,
    xlabel="Number of observed tiles",
    ylabel="(N_newlya_obs / N_newlya_expect) - 1",
    xlim=(0.5, 6.5),
    ylim=(-2.5, 2.5),
    nvalifiber_norm=3900,
):
    """
    Plot the ratio of the number of newly identified Ly-a to the expected number,
        vs. the tile observations coverage for a list of tiles.

    Args:
        ax: pyplot Axes object
        ntilecovs: average number of previously observed tiles for each number of pass (np.array(len(tiles), n_dark_passids))
        nnewlyas: list of nb of newly identified Ly-a (with a valid fiber), normalized to nvalifiber_norm (list or np.array() of ints)
        tileids (optional, defaults to None): tileids (to report outliers) (list or np.array() of ints)
        title (optional, defaults to None): plot title (str)
        xlabel (optional, defaults to "Number of observed tiles"): plot xlabel (str)
        ylabel (optional, defaults to "Number of newly identified LYA candidates"): plot ylabel(str)
        xlim (optional, defaults to (0.5, 6.5)): plot xlim (tuple)
        ylim (optional, defaults to (-2.5, 2.5): plot ylim (tuple)
        nvalifiber_norm (optional, defaults to 3900): number of valid fibers to normalize to, for the expected regions (int)

    Note:
        * ntilecovs[:, 0] lists the fractions of the tiles covered by 1 tile, etc.
        * The plotted y-values are: (N_newlya_observed / N_newlya_expected) - 1.
        * The expected numbers are based on all main dark tiles (from daily) up to May 26th 2022.
        * The 1-2-3-sigma regions reflect the approximate scatter of those data.
    """
    #
    n_dark_passids = 7
    if ntilecovs.shape[1] != n_dark_passids:
        msg = "ntilecovs.shape[1] = {} is different than n_dark_passids = {}".format(
            ntilecovs.shape[1], n_dark_passids,
        )
        log.error(msg)
        raise ValueError(msg)
    # AR overall mean number of pass coverage
    mean_ntilecovs = np.zeros(len(nnewlyas))
    for i in range(n_dark_passids):
        mean_ntilecovs += (i + 1) * ntilecovs[:, i]
    # AR expected number of newlya
    # AR based on main dark tiles (from daily) up to May 26th 2022
    def expect_newlyas(ntilecovs):
        return (
            ntilecovs[:, 0] * 287.7 +
            ntilecovs[:, 1] * 137.8 +
            ntilecovs[:, 2] * 58.1 +
            ntilecovs[:, 3] * 20.9 +
            ntilecovs[:, 4] * 10.5 +
            ntilecovs[:, 5] * 0.0 +
            ntilecovs[:, 6] * 0.6
        )
    # AR approximate expected regions:
    # AR - based on main dark tiles (from daily) up to May 26th 2022
    # AR - the ntilecov in expect_1sig() is the *average* pass coverage (i.e. mean_ntilecovs)
    # AR - blue, green, red: approximative 1-sigma, 2-sigma, 3-sigma
    # AR - special case for ntilecov=1 (a bit more scatter there, as this is more dependent on
    # AR    the parent qso density)
    def expect_1sig(ntilecovs):
        vals = np.exp( (ntilecovs - 7.5) / 2.35)
        vals[(ntilecovs > 0.95) & (ntilecovs <= 1)] = 0.08
        return vals
    tmpxs = np.linspace(0.95, xlim[1], 1000)
    tmpys = expect_1sig(tmpxs)
    for nsig, col in zip([1, 2, 3], ["b", "g", "r"]):
        if nsig == 1:
            ax.fill_between(tmpxs, -tmpys, tmpys, color=col, alpha=0.25)
        else:
            ax.fill_between(tmpxs, -nsig * tmpys, -(nsig-1) * tmpys, color=col, alpha=0.25)
            ax.fill_between(tmpxs, (nsig - 1) * tmpys, nsig * tmpys, color=col, alpha=0.25)
    ax.text(0.025, 0.15, r"Blue  : approx. expected 1$\sigma$", color="b", ha="left", transform=ax.transAxes)
    ax.text(0.025, 0.10, r"Green: approx. expected 2$\sigma$", color="g", ha="left", transform=ax.transAxes)
    ax.text(0.025, 0.05, r"Red   : approx. expected 3$\sigma$", color="r", ha="left", transform=ax.transAxes)
    # AR compute (N_newlya_observed / N_newlya_expected) - 1
    ys = (nnewlyas / expect_newlyas(ntilecovs)) - 1
    # AR clipping so that each point is visible on the plot
    ys = np.clip(ys, ylim[0], ylim[1])
    # AR Poisson error
    yes = np.sqrt(nnewlyas) / expect_newlyas(ntilecovs)
    ax.scatter(mean_ntilecovs, ys, color="k", s=30, alpha=0.8, zorder=1)
    ax.errorbar(mean_ntilecovs, ys, yes, color="none", ecolor="k", elinewidth=1, zorder=1)
    # AR 3-sigma outliers
    ii = np.where(np.abs(ys) > 3 * expect_1sig(mean_ntilecovs))[0]
    for i in ii:
        if tileids is not None:
            label = "TILEID={}".format(tileids[i])
        else:
            label = None
        ax.scatter(mean_ntilecovs[i], ys[i], marker="x", s=80, zorder=2, label=label)
    #
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.grid()
    if (ii.size > 0) & (tileids is not None):
        ax.legend(loc=1, ncol=2)


def create_petalnz_pdf(
    outpdf,
    night,
    prod,
    tileids,
    surveys,
    dchi2_threshold=25,
    group="cumulative",
    newlya_ecsv=None,
):
    """
    For a given night, create a per-petal, per-tracer n(z) pdf file.

    Args:
        outpdf: output pdf file (string)
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc (string)
        tileids: list of tileids to consider (list or numpy array)
        surveys: list of the surveys for each tileid of tileids (list or numpy array)
        dchi2_threshold (optional, defaults to 25): DELTACHI2 value to split the sample (float)
        group (optional, str): tile group "cumulative" or "pernight"
        newlya_ecsv (optional, defaults to None): if set, table saving the per-tile number of newly identified
            Ly-a and the tile coverage (if no dark tiles, no file will be saved).

    Note:
        * Only displays:

          - sv1, sv2, sv3, main, as otherwise the plotted tracers are not relevant;
          - FAPRGRM="bright" or "dark" tileids;
          - for the main survey, tiles with EFFTIME > MINTFRAC * GOALTIME.

        * If the tile-qa-TILEID-thruNIGHT.fits file is missing, that tileid is skipped.
        * For the Lya, work from the zmtl*fits files, trying to mimick what is done in desitarget.mtl.make_mtl().
        * The LRG, ELG, QSO, BGS_BRIGHT, BGS_FAINT bit are the same for sv1, sv2, sv3, main,
          so ok to simply use the bit mask values from the main.
        * TBD : we query the FAPRGRM of the ``tile-qa-*.fits`` header, not sure that properly works for
          surveys other than main..
    """
    petals = np.arange(10, dtype=int)
    n_dark_passids = 7
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
    # AR for main tiles, restrict to tiles with EFFTIME > MINTFRAC * GOALTIME
    # AR we also read the tile-qa*fits header below in the loop,
    # AR    but it is simpler/safer to do separate this first loop
    # AR    to modify the tileids, surveys, if need be.
    sel = np.ones(len(tileids), dtype=bool)
    for i in range(len(tileids)):
        if surveys[i] == "main":
            fn = findfile("tileqa", night=night, tile=tileids[i], groupname=group, specprod_dir=prod)
            if not os.path.isfile(fn):
                log.warning("no {} file, proceeding to next tile".format(fn))
                continue
            hdr = fitsio.read_header(fn, "FIBERQA")
            if hdr["EFFTIME"] < hdr["MINTFRAC"] * hdr["GOALTIME"]:
                sel[i] = False
                log.info(
                    "discarding main survey TILEID={}, as EFFTIME={:.1f} < MINTFRAC*GOALTIME={:.2f}*{:.0f}={:.1f}".format(
                        tileids[i], hdr["EFFTIME"], hdr["MINTFRAC"], hdr["GOALTIME"], hdr["MINTFRAC"] * hdr["GOALTIME"],
                    )
                )
    tileids, surveys = tileids[sel], surveys[sel]
    # AR gather all infos from the zmtl*fits files
    # AR and few extra infos for dark tiles for Ly-a:
    # AR - PRIORITY from the redrock*fits EXP_FIBERMAP
    # AR - nb of previously observed overlapping tiles
    ds = {"bright" : [], "dark" : []}
    ntiles = {"bright" : 0, "dark" : 0}
    for tileid, survey in zip(tileids, surveys):
        # AR bright or dark?
        fn = findfile('tileqa', night=night, tile=tileid, groupname=group, specprod_dir=prod)
        # AR if no tile-qa*fits, we skip the tileid
        if not os.path.isfile(fn):
            log.warning("no {} file, proceeding to next tile".format(fn))
            continue
        hdr = fitsio.read_header(fn, "FIBERQA")
        if "FAPRGRM" not in hdr:
            log.warning("no FAPRGRM in {} header, proceeding to next tile".format(fn))
            continue
        faprgrm = hdr["FAPRGRM"].lower()
        if faprgrm not in ["bright", "dark"]:
            log.warning("{} : FAPRGRM={} not in bright, dark, proceeding to next tile".format(fn, faprgrm))
            continue
        # AR reading zmtl files
        istileid = False
        pix_ntilecovs = None
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
                # AR further infos for dark tiles for Ly-a
                if faprgrm == "dark":
                    # AR PRIORITY from EXP_FIBERMAP
                    try:
                        rrfn = fn.replace("zmtl", "redrock")
                        rrd = Table.read(rrfn, hdu="EXP_FIBERMAP")
                    except FileNotFoundError:
                        rrfn = fn.replace("zmtl", "zbest")
                        rrd = Table.read(rrfn, hdu="FIBERMAP")
                    # AR taking unique TARGETIDs, in case of several exposures
                    # AR    (in which case, PRIORITY is the same for all exposures)
                    _, rrii = np.unique(rrd["TARGETID"], return_index=True)
                    rrd = rrd[rrii]
                    # AR matching to d
                    rrii = match_to(rrd["TARGETID"], d["TARGETID"])
                    rrd = rrd[rrii]
                    msg = None
                    if (len(rrd) != len(d)):
                        msg = "only {} / {} matched objects between {} and {}".format(
                            len(rrd), len(d), rrfn, fn,
                        )
                    else:
                        if ((rrd["TARGETID"] != d["TARGETID"]).sum() > 0):
                            msg = "{} mismatches between {} and {}".format(
                                (rrd["TARGETID"] != d["TARGETID"]).sum(), rrfn, fn,
                            )
                    if msg is not None:
                        log.error(msg)
                        raise ValueError(msg)
                    # AR add PRIORITY
                    d["PRIORITY"] = rrd["PRIORITY"].copy()
                    # AR add number of previous tiles observed
                    # AR pix_ntilecovs is the number of hp pixels covered by NTILE
                    # AR be careful as ntilecov=1 (i.e. covered by one tile) is
                    # AR    stored in the 0-index, etc.
                    if pix_ntilecovs is None:
                        _, pix_ntilecovs, _, _, _ = get_tilecov(tileid, surveys=survey, programs=faprgrm.upper(), lastnight=night)
                        d["NTILECOV"] = np.zeros(len(d) * n_dark_passids).reshape((len(d), n_dark_passids))
                        for ntilecov in range(n_dark_passids):
                            sel = pix_ntilecovs == 1 + ntilecov
                            if sel.sum() > 0:
                                d["NTILECOV"][:, ntilecov] = sel.mean()
                # AR append
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
                gs = gridspec.GridSpec(1, 4, wspace=0.5)
                title = "SURVEY={} : {} tiles from {}".format(
                    survey,
                    " and ".join(["{} {}".format(ntiles_surv[faprgrm], faprgrm.upper()) for faprgrm in faprgrms]),
                    night,
                )
                if "dark" in ntiles_surv:
                    tmpn = ntiles_surv["dark"]
                else:
                    tmpn = 0
                title_dark = "SURVEY={} : {} DARK tiles from {}".format(
                    survey, tmpn, night,
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
                ax.set_title(title_dark)
                ax.set_xlabel("PETAL_LOC")
                ax.set_ylabel("fraction of LYA candidates\n(VALID QSO fibers only)")
                ax.set_xlim(-0.5, 9.5)
                ax.xaxis.set_major_locator(MultipleLocator(1))
                ax.set_ylim(0, 1)
                ax.grid()
                # AR - newly identified Ly-a = f(ntilecov)
                ax = plt.subplot(gs[3])
                xlim, ylim = (0.5, 6.5), (-2.5, 2.5)
                nvalifiber_norm = 3900
                if "dark" in faprgrms:
                    faprgrm = "dark"
                    isnewlya = ds[faprgrm]["SURVEY"] == survey
                    isnewlya &= ds[faprgrm]["VALID"]
                    isnewlya &= (ds[faprgrm][dtkey] & desi_mask["QSO"]) > 0
                    isnewlya &= ds[faprgrm]["LYA"]
                    isnewlya &= ds[faprgrm]["PRIORITY"] == desi_mask["QSO"].priorities["UNOBS"]
                    #
                    dark_tileids = np.unique(ds[faprgrm]["TILEID"])
                    tmpd = Table()
                    for key in ["TILEID", "LASTNIGHT", "NVALIDFIBER", "NNEWLYA"]:
                        tmpd[key] = np.zeros(len(dark_tileids), dtype=int)
                    tmpd["NTILECOV"] = np.zeros(len(dark_tileids) * n_dark_passids).reshape((len(dark_tileids), n_dark_passids))
                    for i in range(len(dark_tileids)):
                        sel = ds[faprgrm]["TILEID"] == dark_tileids[i]
                        tmpd["TILEID"][i] = dark_tileids[i]
                        tmpd["LASTNIGHT"][i] = night
                        tmpd["NVALIDFIBER"][i] = ((sel) & (ds[faprgrm]["VALID"])).sum()
                        tmpd["NTILECOV"][i, :] = ds[faprgrm]["NTILECOV"][sel, :][0]
                        tmpd["NNEWLYA"][i] = ((sel) & (isnewlya)).sum()
                    plot_newlya(
                        ax,
                        tmpd["NTILECOV"],
                        tmpd["NNEWLYA"] / (tmpd["NVALIDFIBER"] / nvalifiber_norm),
                        tileids=tmpd["TILEID"],
                        xlim=xlim,
                        ylim=ylim,
                    )
                    if newlya_ecsv is not None:
                        tmpd.write(newlya_ecsv, overwrite=True)
                else:
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    ax.grid()
                ax.set_title(title_dark)
                ax.set_xlabel("Avg nb of previously observed overlapping tiles on {}".format(night))
                ax.set_ylabel("(N_newlya_obs / N_newlya_expect) - 1")
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

    Note:
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

    # AR calibnight: flat, psf and bias
    # AR color-coding:
    # AR - red : file does not exist
    # AR - blue : file exists, but is a symlink
    # AR - green : file exists
    html.write(
        "<button type='button' class='collapsible'>\n\t<strong>{} calibnight</strong>\n</button>\n".format(
            night,
        )
    )
    html.write("<div class='content'>\n")
    html.write("\t<br>\n")
    html.write("\t<p>Assess the presence of all psfnight, fiberflatnight, and biasnight files for {}.</p>\n".format(night))
    html.write("\t<p>If a file appears in <span style='color:green;'>green</span>, it means it is present.</p>\n")
    html.write("\t<p>If a file appears in <span style='color:blue;'>blue</span>, it means it is a symlink to another file, the name of which is reported.</p>\n")
    html.write("\t<p>If a file appears in <span style='color:red;'>red</span>, it means it is missing.</p>\n")
    html.write("\t<p>If a file appears in <span style='color:orange;'>orange</span>, it means it does not date from the corresponding night (likely done in the morning, in which case the pipeline uses some default files).</p>\n")
    html.write("\t</br>\n")
    html.write("<table>\n")
    for petal in petals:
        html.write("\t<tr>\n")
        for case in ["psfnight", "fiberflatnight", "biasnight"]:
            for camera in cameras:
                fn = findfile(case, night, camera=camera+str(petal),
                        specprod_dir=prod)
                fnshort, color = os.path.basename(fn).replace("-{}".format(night), ""), "red"
                if os.path.isfile(fn):
                    if os.path.islink(fn):
                        fnshort, color = os.path.basename(os.readlink(fn)), "blue"
                    else:
                        # AR check the timestamp "night" vs. the night
                        # AR if cals are done before observations, those should match
                        m_time = os.path.getmtime(fn)
                        if int(datetime.fromtimestamp(m_time).strftime("%Y%m%d")) != night:
                            color = "orange"
                        else:
                            color = "green"
                html.write("\t\t<td> <span style='color:{};'>{}</span> </td>\n".format(color, fnshort))
                if camera != cameras[-1]:
                    html.write("\t\t<td> &emsp; </td>\n")
            if case in ["psfnight", "fiberflatnight"]:
                html.write("\t\t<td> &emsp; &emsp; &emsp; </td>\n")
        html.write("\t</tr>\n")
    html.write("</table>\n")
    html.write("</div>\n")
    html.write("\n")

    # AR various tabs:
    # AR - dark
    # AR - morningdark
    # AR - badcol
    # AR - ctedet
    # AR - sframesky
    # AR - tileqa
    # AR - skyzfiber
    # AR - petalnz
    for case, caselab, width, text in zip(
        ["dark", "morningdark", "badcol", "ctedet", "sframesky", "tileqa", "skyzfiber", "petalnz"],
        ["DARK", "Morning DARK", "bad columns", "CTE detector", "sframesky", "Tile QA", "SKY Z vs. FIBER", "Per-petal n(z)"],
        ["100%", "100%", "35%", "100%", "75%", "90%", "90%", "100%"],
        [
            "This pdf displays the 300s (binned) DARK (one page per spectrograph; non-valid pixels are displayed in red)\nThe side panels report the median profiles for each pair of amps along each direction.\nWatch it and report unsual features (easy to say!)",
             "This pdf displays the 1200s (binned) morning DARK (one page per spectrograph; non-valid pixels are displayed in red)\nThe side panels report the median profiles for each pair of amps along each direction.\nWatch it and report unsual features (easy to say!)",
            "This plot displays the histograms of the bad columns.\nWatch it and report unsual features (easy to say!)",
            "This pdf displays a small diagnosis to detect CTE anormal behaviour (one petal-camera per page)\nWatch it and report unusual features (typically if the lower enveloppe of the blue or orange curve is systematically lower than the other one).",
            "This pdf displays the sframe image for the sky fibers for each Main exposure (one exposure per page).\nPixels with IVAR=0 are displayed in yellow.\nWatch it and report unsual features (easy to say!)",
            "This pdf displays the tile-qa-TILEID-thru{}.png files for the Main tiles (one tile per page).\nWatch it, in particular the Z vs. FIBER plot, and report unsual features (easy to say!)".format(night),
            "This plot displays all the SKY fibers for the {} night.\nWatch it and report unsual features (easy to say!)".format(night),
            "This pdf displays the per-tracer, per-petal n(z) for the {} night.\nWatch it and report unsual features (easy to say!)\nIn particular, if some maindark tiles have been observed, pay attention to the two Ly-a related plots on the first row.".format(night),
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
                outpng = os.path.basename(outfns[case])
                html.write(
                    "\t<a href='{}'><img SRC='{}' width={} height=auto></a>\n".format(
                        outpng, outpng, width,
                    )
                )
            elif os.path.splitext(outfns[case])[-1] == ".pdf":
                outpdf = os.path.basename(outfns[case])
                html.write("\t<iframe src='{}' width={} height=100%></iframe>\n".format(outpdf, width))
            else:
                log.error("Unexpected extension for {}".format(outfns[case]))
                raise RuntimeError("Unexpected extension for {}".format(outfns[case]))
        else:
            html.write("\t<p>No {}.</p>\n".format(os.path.basename(outfns[case])))
        html.write("\t</br>\n")
        html.write("</div>\n")
        html.write("\n")

    # AR lines to make collapsing sections
    write_html_collapse_script(html, "collapsible")

    # ADM html postamble for main page.
    write_html_today(html)
    html.write("</html></body>\n")
    html.close()
