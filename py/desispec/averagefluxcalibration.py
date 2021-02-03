import numpy as np
import scipy.constants
from desimodel.io import load_desiparams

class AverageFluxCalib(object):
    def __init__(self, wave, average_calib, atmospheric_extinction, seeing_term, \
                 pivot_airmass, pivot_seeing,\
                 atmospheric_extinction_uncertainty = None, seeing_term_uncertainty = None):
        """Lightweight wrapper object for average flux calibration data

        Args:
            wave : 1D[nwave] input wavelength (angstroms)
            average_calib: 1D[nwave] average calibration vector at pivot airmass and seeing ((electrons)/(1e-17 erg/cm^2))
            atmospheric_extinction : 1D[nwave] extinction term, magnitude
            seeing_term : 1D[nwave], magnitude
            pivot_airmass : float, airmass value for average_calib
            pivot_seeing : float, seeing value for average_calib (same definition and unit as SEEING keyword in images)
            atmospheric_extinction_uncertainty : 1D[nwave] uncertainty on extinction term, magnitude
            seeing_term_uncertainty : 1D[nwave], uncertainty on seeing term magnitude

        All arguments become attributes,
        the calib vector should be in units of [electrons]/[1e-17 erg/cm^2].

        The model is
        calib = average_calib*10**(-0.4*((seeing-pivot_seeing)*seeing_term + (airmass-pivot_airmass)*atmospheric_extinction))

        """
        assert wave.ndim == 1
        assert average_calib.shape == wave.shape
        assert atmospheric_extinction.shape == wave.shape
        assert seeing_term.shape == wave.shape

        self.wave = wave
        self.average_calib = average_calib
        self.atmospheric_extinction = atmospheric_extinction
        self.seeing_term = seeing_term
        self.pivot_airmass = pivot_airmass
        self.pivot_seeing = pivot_seeing
        self.atmospheric_extinction_uncertainty = atmospheric_extinction_uncertainty
        self.seeing_term_uncertainty = seeing_term_uncertainty
        self.meta = dict(units='electrons/(1e-17 erg/cm^2)')
        self.desiparams = load_desiparams()

    def value(self,airmass=1,seeing=1.1) :
        """Returns calibration vector for this airmass and seeing, in units of [electrons]/[1e-17 erg/cm^2]

        Args:
           airmass : float, airmass, as defined in image headers
           seeing ; float , seeing in arcsec FWHM, as defined in image headers

        Returns:
           calibation vector : 1D[nwave] corresponding to this class wavelength array self.wave, units are [photons]/[1e-17 erg/cm^2]
        """

        return self.average_calib*10**(-0.4*( (seeing-self.pivot_seeing)*self.seeing_term + (airmass-self.pivot_airmass)*self.atmospheric_extinction ))


    def throughput(self,airmass=1,seeing=1.1) :

        """Returns throughput for this airmass and seeing

        Args:
           airmass : float, airmass, as defined in image headers
           seeing ; float , seeing in arcsec FWHM, as defined in image headers

        Returns:
           throughput (dimensionless, electron in CCD per photon above atmosphere that could hit the mirror)
        """

        # self.value is in (electron/s/A)/( 1e-17 ergs/s/cm2/A) = (1e17 electron*cm2/erg)

        cal_scale = np.ones(self.wave.shape)
        cal_scale *= 1e17 # scale*value in (electron*cm2/erg)
        # reference effective collection area, 8.659m2 in DESI-347 v16, but now use DESIMODEL
        area = self.desiparams["area"]["geometric_area"]*1e4 # m2 -> cm2
        cal_scale /= area #  scale*value in (electron/erg)
        hplanck = scipy.constants.h #J.s
        hplanck *= 1e7 # erg.s
        cspeed = scipy.constants.c # m/s
        cspeed *= 1e10 # A/s
        energy = hplanck*cspeed/self.wave # (erg/photon)
        cal_scale *= energy # scale*value in (electron/photon)
        return cal_scale * self.value(airmass,seeing)
