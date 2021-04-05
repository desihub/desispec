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

# only read once
def get_average_calibration(filename) :
    global average_calibrations
    if not filename in average_calibrations :
        average_calibrations[filename] = read_average_flux_calibration(filename)
    return average_calibrations[filename]

# AR grz-band sky mag / arcsec2 from sky-....fits files
# AR now using work-in-progress throughput
# AR still provides a better agreement with GFAs than previous method
def compute_skymag(night, expid, specprod_dir=None):



    log=get_logger()




    # AR/DK DESI spectra wavelengths
    wmin, wmax, wdelta = 3600, 9824, 0.8
    fullwave = np.round(np.arange(wmin, wmax + wdelta, wdelta), 1)
    cslice = {"b": slice(0, 2751), "r": slice(2700, 5026), "z": slice(4900, 7781)}
    # AR (wmin,wmax) to "stich" all three cameras
    wstich = {"b": (wmin, 5780), "r": (5780, 7570), "z": (7570, 9824)}

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

            #cal_filename=findcalibfile([header],"FLUXCALIB")

            # for now we use a fixed calibration as used in DESI-6043 for which we know what was the fiber aperture loss
            cal_filename="{}/spec/fluxcalib/fluxcalibnight-{}-20201216.fits".format(os.environ["DESI_SPECTRO_CALIB"],camera)
            # apply the correction from
            fiber_acceptance_for_point_sources = 0.60 # see DESI-6043
            mean_fiber_diameter_arcsec = 1.52 # see DESI-6043
            fiber_area_arcsec = np.pi*(mean_fiber_diameter_arcsec/2)**2

            acal = get_average_calibration(cal_filename)
            flux = np.interp(fullwave[cslice[camera]], skywave, skyflux)
            sky[cslice[camera]] = flux / exptime / acal.value() * fiber_acceptance_for_point_sources / fiber_area_arcsec * 1e-17 # ergs/s/cm2/A/arcsec2

        if not ok : continue # to next spectrograph
        sky_spectra.append(sky)

    if len(sky_spectra)==0 : return (99.,99.,99.)
    if len(sky_spectra)==1 :
        sky = sky_spectra[0]
    else :
        sky = np.mean(np.array(sky_spectra),axis=0) # mean over petals/spectrographs


    # AR integrate over the DECam grz-bands
    # AR using the curves with no atmospheric extinction
    filts = filters.load_filters("decam2014-g", "decam2014-r", "decam2014-z")

    # AR zero-padding spectrum so that it covers the DECam grz passbands
    # AR looping through filters while waiting issue to be solved (https://github.com/desihub/speclite/issues/64)
    sky_pad, fullwave_pad = sky.copy(), fullwave.copy()
    for i in range(len(filts)):
        sky_pad, fullwave_pad = filts[i].pad_spectrum(sky_pad, fullwave_pad, method="zero")
    mags = filts.get_ab_magnitudes(sky_pad * units.erg / (units.cm ** 2 * units.s * units.angstrom),fullwave_pad * units.angstrom).as_array()[0]

    #import matplotlib.pyplot as plt
    #plt.plot(fullwave_pad,sky_pad)
    #plt.grid()
    #plt.show()

    return mags # AB mags for flux per arcsec2
