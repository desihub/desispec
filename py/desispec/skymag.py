"""
desispec.skymag
============

Utility function to compute the sky magnitude per arcmin2 based from the measured sky model
of an exposure and a static model of the instrument throughput.
"""

import os,sys
import numpy as np
import fitsio
from astropy import units, constants

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
    returns (gmag,rmag,zmag) AB magnitudes per arcsec2, tuple with 3 float values
    """

    log=get_logger()

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

    # AR looking for a petal with brz sky and ivar>0
    sky_spectra = []
    for spec in range(10) :
        sky = np.zeros(fullwave.shape)
        ok  = True
        for camera in ["b","r","z"] :
            camspec="{}{}".format(camera,spec)
            filename = findfile("sky",night=night,expid=expid,camera=camspec,specprod_dir=specprod_dir)
            if not os.path.isfile(filename) :
                log.warning("skipping {}-{:08d}-{} : missing {}".format(night,expid,spec,filename))
                ok = False
                break
            fiber=0
            skyivar=fitsio.read(filename,"IVAR")[fiber]
            if np.all(skyivar==0) :
                log.warning("skipping {}-{:08d}-{} : ivar=0 for {}".format(night,expid,spec,filename))
                ok=False
                break
            skyflux=fitsio.read(filename,0)[fiber]
            skywave=fitsio.read(filename,"WAVELENGTH")
            header=fitsio.read_header(filename)
            exptime=header["EXPTIME"]

            # use fixed calibrations
            if night < 20210318 : # before mirror cleaning
                cal_filename="{}/spec/fluxcalib/fluxcalibaverage-{}-20201214.fits".format(os.environ["DESI_SPECTRO_CALIB"],camera)
            else :
                cal_filename="{}/spec/fluxcalib/fluxcalibaverage-{}-20210318.fits".format(os.environ["DESI_SPECTRO_CALIB"],camera)

            acal = _get_average_calibration(cal_filename)
            begin, end = istitch[camera]
            flux = np.interp(fullwave[begin:end], skywave, skyflux)
            acal_val = np.interp(fullwave[begin:end], acal.wave, acal.value())
            sky[begin:end] = flux / exptime / acal_val * fiber_acceptance_for_point_sources / fiber_area_arcsec * 1e-17 # ergs/s/cm2/A/arcsec2

        if not ok : continue # to next spectrograph
        sky_spectra.append(sky)

    if len(sky_spectra)==0 : return (99.,99.,99.)
    if len(sky_spectra)==1 :
        sky = sky_spectra[0]
    else :
        sky = np.mean(np.array(sky_spectra),axis=0) # mean over petals/spectrographs

    # AR integrate over the DECam grz-bands
    filts = _get_decam_filters()

    # AR zero-padding spectrum so that it covers the DECam grz passbands
    # AR looping through filters while waiting issue to be solved (https://github.com/desihub/speclite/issues/64)
    sky_pad, fullwave_pad = sky.copy(), fullwave.copy()
    for i in range(len(filts)):
        sky_pad, fullwave_pad = filts[i].pad_spectrum(sky_pad, fullwave_pad, method="zero")
    mags = filts.get_ab_magnitudes(sky_pad * units.erg / (units.cm ** 2 * units.s * units.angstrom),fullwave_pad * units.angstrom).as_array()[0]

    return mags # AB mags for flux per arcsec2
