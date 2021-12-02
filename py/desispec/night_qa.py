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
# AR astropy
from astropy.table import Table
from astropy.io import fits
# AR desispec
from desispec.fiberbitmasking import get_skysub_fiberbitmask_val
# AR matplotlib
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import gridspec
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
    }



def get_survey_night_expids(
    night,
    survey,
    datadir = None):
    """
    List the (EXPIDs, TILEIDs) from a given night for a given survey.

    Args:
        night: night (int)
        survey: "main", "sv3", "sv2", or "sv1" (str)
        datadir (optional, defaults to $DESI_SPECTRO_DATA): full path where the {NIGHT}/desi-{EXPID}.fits.fz files are (str)

    Returns:
        expids: list of the EXPIDs (np.array())
        tileids: list of the TILEIDs (np.array())

    Notes:
        Based on parsing the OBSTYPE and NTSSURVY keywords from the SPEC extension header of the desi-{EXPID}.fits.fz files.
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
    expids, tileids = [], []
    for i in range(len(fns)):
        hdr = fits.getheader(fns[i], "SPEC")
        if hdr["OBSTYPE"] == "SCIENCE":
            if hdr["NTSSURVY"] == survey:
                expids.append(hdr["EXPID"])
                tileids.append(hdr["TILEID"])
    log.info(
        "found {} exposures from {} tiles for SURVEY={} and NIGHT={}".format(
            len(expids), np.unique(tileids).size, survey, night,
        )
    )
    return np.array(expids), np.array(tileids)


def get_dark_night_expid(night, datadir = None):
    """
    Returns the EXPID of the 300s DARK exposure for a given night.

    Args:
        night: night (int)
        datadir (optional, defaults to $DESI_SPECTRO_DATA): full path where the {NIGHT}/desi-{EXPID}.fits.fz files are (str)

    Returns:
        expid: EXPID (int)

    Notes:
        If nothing found, returns None.
    """
    if datadir is None:
        datadir = os.getenv("DESI_SPECTRO_DATA")
    #
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
    expid = None
    for i in range(len(fns)):
        hdr = fits.getheader(fns[i], "SPEC")
        if (hdr["OBSTYPE"] == "DARK") & (hdr["REQTIME"] == 300):
            expid = hdr["EXPID"]
            break
    if expid is None:
        log.warning(
            "no EXPID found as the 300s DARK for NIGHT={}".format(night)
        )
    else:
        log.info(
            "found EXPID={} as the 300s DARK for NIGHT={}".format(
                expid, night,
            )
        )
    return expid


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
        log.error("ffmpeg needs to be installed to run create_mp4(); exiting")
        sys.exit(1)
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
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc/ (string)
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
                    d = fits.open(fn)["IMAGE"].data
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
                    im = ax.imshow(d, cmap=matplotlib.cm.Greys_r, vmin=clim[0], vmax=clim[1])
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


def create_badcol_png(outpng, night, prod, tmpdir=tempfile.mkdtemp()):
    """
    For a given night, create a png file with displaying the number of bad columns per {camera}{petal}.

    Args:
        outpng: output png file (string)
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc/ (string)
        tmpdir (optional, defaults to a temporary directory): temporary directory where individual images are created
    """
    cameras = ["b", "r", "z"]
    colors = ["b", "r", "k"]
    petals = np.arange(10, dtype=int)
    # AR reading
    badcols = {camera : np.nan + np.zeros(len(petals)) for camera in cameras}
    for camera in cameras:
        for petal in petals:
            fn = os.path.join(
                    prod,
                    "calibnight",
                    "{}".format(night),
                    "badcolumns-{}{}-{}.csv".format(camera, petal, night),
            )
            if os.path.isfile(fn):
                log.info("reading {}".format(fn))
                badcols[camera][petal] = len(Table.read(fn))
    # AR plotting
    fig, ax = plt.subplots()
    for camera, color in zip(cameras, colors):
        ax.plot(petals, badcols[camera], "-o", color=color, label="{}-camera".format(camera))
        ax.legend(loc=2)
        ax.set_xlabel("PETAL_LOC")
        ax.set_xlim(petals[0] - 1, petals[-1] + 1)
        ax.set_ylabel("N(badcolumn)")
        ax.set_ylim(0, 50)
        ax.grid()
    plt.savefig(outpng, bbox_inches="tight")
    plt.close()


def create_ctedet_pdf(outpdf, night, prod, tmpdir=tempfile.mkdtemp()):
    """
    TBD
    TBD copy-paste-adapt code from here /global/homes/s/sjbailey/desi/dev/ccd/plot-amp-cte.py
    """
    a = 0 # dummy line


def create_sframesky_pdf(outpdf, night, prod, expids):
    """
    For a given night, create a pdf from per-expid sframe for the sky fibers only.

    Args:
        outpdf: output pdf file (string)
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc/ (string)
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


def create_tileqa_pdf(outpdf, night, prod, expids, tileids):
    """
    For a given night, create a pdf from the tile-qa*png files, sorted by increasing EXPID.

    Args:
        outpdf: output pdf file (string)
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc/ (string)
        expids: expids of the tiles to display (list or np.array)
        tileids: tiles to display (list or np.array)
    """
    # AR exps, to sort by increasing EXPID for that night
    expids, tileids = np.array(expids), np.array(tileids)
    ii = expids.argsort()
    expids, tileids = expids[ii], tileids[ii]
    ii = np.array([np.where(tileids == tileid)[0][0] for tileid in np.unique(tileids)])
    expids, tileids = expids[ii], tileids[ii]
    ii = expids.argsort()
    expids, tileids = expids[ii], tileids[ii]
    #
    fns = []
    for tileid in tileids:
        fn = os.path.join(
            prod,
            "tiles",
            "cumulative",
            "{}".format(tileid),
            "{}".format(night),
            "tile-qa-{}-thru{}.png".format(tileid, night))
        if os.path.isfile(fn):
            fns.append(fn)
        else:
            log.warning("no {}".format(fn))
    # AR creating pdf
    with PdfPages(outpdf) as pdf:
        for fn in fns:
            fig, ax = plt.subplots()
            img = Image.open(fn)
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight", dpi=300)
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



def write_nightqa_html(outfns, night, prod, css):
    """
    Write the nightqa-{NIGHT}.html page.

    Args:
        outfns: dictionary with filenames generated by desi_night_qa (output from get_nightqa_outfns)
        night: night (int)
        prod: full path to prod folder, e.g. /global/cfs/cdirs/desi/spectro/redux/blanc/ (string)
        css: path to the nightqa.css file
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
    html.write("\t<p>Please click on each tab from top to bottom, and follow instructions.</p>\n")

    # AR night log
    nighthtml = "https://data.desi.lbl.gov/desi/survey/ops/nightlogs/{}/NightSummary{}.html".format(
        night, night,
    )
    html.write(
        "<button type='button' class='collapsible'>\n\t<strong>{} night summary</strong>\n</button>\n".format(
            night,
        )
    )
    html.write("<div class='content'>\n")
    html.write("\t<br>\n")
    html.write("\t<p>Read the nightlog for {}: {}, displayed below.</p>\n".format(night, nighthtml))
    html.write("\t<p>And consider subscribing to the desi-nightlog mailing list!\n")
    html.write("\t</br>\n")
    html.write("\t<br>\n")
    html.write("\t<iframe src='{}' width=100% height=100%></iframe>\n".format(nighthtml))
    html.write("\t<p>And consider subscribing to the desi-nightlog mailing list!\n")
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

    # AR DARK
    html.write(
        "<button type='button' class='collapsible'>\n\t<strong>{} dark</strong>\n</button>\n".format(
            night,
        )
    )
    html.write("<div class='content'>\n")
    html.write("\t<br>\n")
    html.write("\t<p>This pdf displays the 300s (binned) DARK (one page per spectrograph).</p>\n")
    html.write("\t<p>Watch it and report unsual features (easy to say!)</p>\n")
    html.write("\t<tr>\n")
    html.write("\t<iframe src='{}' width=100% height=100%></iframe>\n".format(path_full2web(outfns["dark"])))
    html.write("\t</br>\n")
    html.write("</div>\n")
    html.write("\n")

    # AR bad columns
    html.write(
        "<button type='button' class='collapsible'>\n\t<strong>{} bad columns</strong>\n</button>\n".format(
            night,
        )
    )
    html.write("<div class='content'>\n")
    html.write("\t<br>\n")
    html.write("\t<p>This plot displays the histograms of the bad columns.</p>\n")
    html.write("\t<p>Watch it and report unsual features (easy to say!)</p>\n")
    html.write("\t<tr>\n")
    html.write("\t<br>\n")
    outpng = path_full2web(outfns["badcol"])
    txt = "<a href='{}'><img SRC='{}' width=35% height=auto></a>".format(
        outpng, outpng
    )
    html.write("\t{}\n".format(txt))
    html.write("\t</br>\n")
    html.write("</div>\n")
    html.write("\n")

    # AR CTE detector
    html.write(
        "<button type='button' class='collapsible'>\n\t<strong>{} cte detector</strong>\n</button>\n".format(
            night,
        )
    )
    html.write("<div class='content'>\n")
    html.write("\t<br>\n")
    html.write("\t<p>TBD : add here plots/video here.</p>\n")
    html.write("\t<p>TBD : add instructions here</p>\n")
    html.write("\t<tr>\n")
    # html.write("\t<video width=90% height=auto controls autoplay loop>\n") 
    # html.write("\t\t<source src='{}' type='video/mp4'>\n".format(path_full2web(outfns["ctedet"])))
    # html.write("\t</video>\n")
    html.write("\t</br>\n")
    html.write("</div>\n")
    html.write("\n")

    # AR Per-night observing sframesky, tileqa
    for case, width, text in zip(
        ["sframesky", "tileqa"],
        ["75%", "90%"],
        [
            "This pdf displays the sframe image for the sky fibers for each Main exposure (one exposure per page).\nWatch it and report unsual features (easy to say!)",
            "This pdf displays the tile-qa-TILEID-thru{}.png files for the Main tiles (one tile per page).\nWatch it, in particular the Z vs. FIBER plot, and report unsual features (easy to say!)".format(night),
        ]
    ):
        html.write(
            "<button type='button' class='collapsible'>\n\t<strong>{} {}</strong>\n</button>\n".format(
                night, case,
            )
        )
        html.write("<div class='content'>\n")
        html.write("\t<br>\n")
        for text_split in text.split("\n"):
            html.write("\t<p>{}</p>\n".format(text_split))
        html.write("\t<tr>\n")
        html.write("\t<iframe src='{}' width={} height=100%></iframe>\n".format(path_full2web(outfns[case]), width))
        html.write("\t</br>\n")
        html.write("</div>\n")
        html.write("\n")

    # AR lines to make collapsing sections
    write_html_collapse_script(html, "collapsible")

    # ADM html postamble for main page.
    write_html_today(html)
    html.write("</html></body>\n")
    html.close()

