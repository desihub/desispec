"""
desispec.skymag
===============

Utility function to compute the sky magnitude per arcmin2 based from the measured sky model
of an exposure and a static model of the instrument throughput.
"""

import os,sys
import numpy as np
import fitsio
from astropy import units, constants
from astropy.table import Table

from desiutil.log import get_logger
from speclite import filters
from desispec.io import read_sky,findfile,specprod_root,read_average_flux_calibration
from desispec.calibfinder import findcalibfile


average_calibrations = dict()
decam_filters = None

# only read once per process
def _get_average_calibration(filename) :
    """
    Use a dictionnary referenced by a global variable
    to keep a copy of the calibration
    instead of reading it at each function call.
    """
    global average_calibrations
    if not filename in average_calibrations :
        average_calibrations[filename] = read_average_flux_calibration(filename)
    return average_calibrations[filename]

# only read once per process
def _get_decam_filters() :
    global decam_filters
    if decam_filters is None :
        log = get_logger()
        log.info("read decam filters")
        decam_filters = filters.load_filters("decam2014-g", "decam2014-r", "decam2014-z")
    return decam_filters

# AR grz-band sky mag / arcsec2 from sky-....fits files
# AR now using work-in-progress throughput
# AR still provides a better agreement with GFAs than previous method
def compute_skymag(night, expid, specprod_dir=None):
    """
    Computes the sky magnitude for a given exposure. Uses the sky model
    and apply a fixed calibration for which the fiber aperture loss
    is well understood.

    Args:
       night: int, YYYYMMDD
       expid: int, exposure id
       specprod_dir: str, optional, specify the production directory.
            default is $DESI_SPECTRO_REDUX/$SPECPROD

    Returns:
        (gmag,rmag,zmag) AB magnitudes per arcsec2, tuple with 3 float values.
        Returns (99., 99., 99.) if no valid petals are found.
        Delegates per-petal calibration to compute_skymag_per_petal() and
        returns the mean over all valid petals.
    """
    table = compute_skymag_per_petal(night, expid, specprod_dir)
    if table is None:
        return (99., 99., 99.)
    gmag = np.nanmean(table['SKY_MAG_G_SPEC'])
    rmag = np.nanmean(table['SKY_MAG_R_SPEC'])
    zmag = np.nanmean(table['SKY_MAG_Z_SPEC'])
    return (gmag, rmag, zmag)


def compute_skymag_per_petal(night, expid, specprod_dir=None):
    """Compute per-petal sky magnitudes for one exposure.

    Calibrates the sky spectrum for each spectrograph petal and integrates
    over the DECam g, r, z filter curves.  Requires all three cameras
    (b, r, z) for a petal to be present with at least one fiber with
    non-zero IVAR; petals that fail this check are skipped.

    This function contains the per-petal calibration logic shared with
    compute_skymag(), which delegates to this function.

    Args:
        night: int, YYYYMMDD.
        expid: int, exposure ID.
        specprod_dir: str, optional. Defaults to $DESI_SPECTRO_REDUX/$SPECPROD.

    Returns:
        astropy.table.Table with columns PETAL_LOC (int16),
        SKY_MAG_G_SPEC (float32), SKY_MAG_R_SPEC (float32),
        SKY_MAG_Z_SPEC (float32), one row per valid petal.
        Returns None if no valid petals are found.
    """
    log = get_logger()

    # AR/DK DESI spectra wavelengths
    wmin, wmax, wdelta = 3600, 9824, 0.8
    fullwave = np.round(np.arange(wmin, wmax + wdelta, wdelta), 1)

    # AR (wmin,wmax) to "stitch" all three cameras
    wstitch = {"b": (wmin, 5790), "r": (5790, 7570), "z": (7570, 9824)}
    istitch = {}
    for camera in ["b", "r", "z"]:
        ii = np.where((fullwave >= wstitch[camera][0]) & (fullwave < wstitch[camera][1]))[0]
        istitch[camera] = (ii[0], ii[-1]+1) # begin (included), end (excluded)

    if specprod_dir is None :
        specprod_dir = specprod_root()

    filts = _get_decam_filters()

    petal_locs = []
    gmags = []
    rmags = []
    zmags = []

    for spec in range(10):
        sky = np.zeros(fullwave.shape)
        ok = True
        for camera in ["b", "r", "z"]:
            camspec = "{}{}".format(camera, spec)
            filename = findfile("sky", night=night, expid=expid, camera=camspec, specprod_dir=specprod_dir, readonly=True)
            if not os.path.isfile(filename):
                log.warning("skipping {}-{:08d}-{} : missing {}".format(night, expid, spec, filename))
                ok = False
                break
            fiber = 0
            skyivar = fitsio.read(filename, "IVAR")[fiber]
            if np.all(skyivar == 0):
                log.warning("skipping {}-{:08d}-{} : ivar=0 for {}".format(night, expid, spec, filename))
                ok = False
                break
            skyflux = fitsio.read(filename, 0)[fiber]
            skywave = fitsio.read(filename, "WAVELENGTH")
            header = fitsio.read_header(filename)
            exptime = header["EXPTIME"]

            # use fixed calibrations
            if night < 20210318 : # before mirror cleaning
                cal_filename="{}/spec/fluxcalib/fluxcalibaverage-{}-20201214.fits".format(os.environ["DESI_SPECTRO_CALIB"],camera)
            else :
                cal_filename="{}/spec/fluxcalib/fluxcalibaverage-{}-20210318.fits".format(os.environ["DESI_SPECTRO_CALIB"],camera)

            acal = _get_average_calibration(cal_filename)
            begin, end = istitch[camera]
            flux = np.interp(fullwave[begin:end], skywave, skyflux)
            acal_val = acal.value()

            if acal.ffracflux_wave is not None :
                acal_val /= acal.ffracflux_wave
            else :
                default_ffracflux = 0.6 # see DESI-6043
                log.warning("use a default fiber acceptance correction = {}".format(default_ffracflux))
                acal_val /= default_ffracflux

            acal_val = np.interp(fullwave[begin:end], acal.wave, acal_val)

            mean_fiber_diameter_arcsec = 1.52 # see DESI-6043
            fiber_area_arcsec = np.pi*(mean_fiber_diameter_arcsec/2)**2

            sky[begin:end] = flux / exptime / acal_val / fiber_area_arcsec * 1e-17 # ergs/s/cm2/A/arcsec2

        if not ok:
            continue  # to next spectrograph

        # AR zero-padding spectrum so that it covers the DECam grz passbands
        # AR looping through filters while waiting issue to be solved (https://github.com/desihub/speclite/issues/64)
        sky_pad, fullwave_pad = sky.copy(), fullwave.copy()
        for i in range(len(filts)):
            sky_pad, fullwave_pad = filts[i].pad_spectrum(sky_pad, fullwave_pad, method="zero")
        petal_mags = filts.get_ab_magnitudes(
            sky_pad * units.erg / (units.cm ** 2 * units.s * units.angstrom),
            fullwave_pad * units.angstrom
        ).as_array()[0]

        petal_locs.append(spec)
        gmags.append(petal_mags[0])
        rmags.append(petal_mags[1])
        zmags.append(petal_mags[2])

    if len(petal_locs) == 0:
        return None

    table = Table()
    table['PETAL_LOC'] = np.array(petal_locs, dtype=np.int16)
    table['SKY_MAG_G_SPEC'] = np.array(gmags, dtype=np.float32)
    table['SKY_MAG_R_SPEC'] = np.array(rmags, dtype=np.float32)
    table['SKY_MAG_Z_SPEC'] = np.array(zmags, dtype=np.float32)
    return table
