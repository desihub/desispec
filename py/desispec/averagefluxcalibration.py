import numpy as np

class AverageFluxCalib(object):
    def __init__(self, wave, average_calib, atmospheric_extinction, seeing_term, \
                 pivot_airmass, pivot_seeing,\
                 atmospheric_extinction_uncertainty = None, seeing_term_uncertainty = None):
        """Lightweight wrapper object for average flux calibration data

        Args:
            wave : 1D[nwave] input wavelength (Angstroms)
            average_calib: 1D[nwave] average calibration vector at pivot airmass and seeing
            atmospheric_extinction : 1D[nwave] extinction term, magnitude
            seeing_term : 1D[nwave], magnitude
            pivot_airmass : float, airmass value for average_calib
            pivot_seeing : float, seeing value for average_calib (same definition and unit as SEEING keyword in images)
            atmospheric_extinction_uncertainty : 1D[nwave] uncertainty on extinction term, magnitude
            seeing_term_uncertainty : 1D[nwave], uncertainty on seeing term magnitude

        All arguments become attributes,

        The calib vector should be such that

            [1e-17 erg/s/cm^2/A] = [photons/A] / calib

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
        self.meta = dict(units='photons/(erg/s/cm^2)')

    def value(self,airmass=1,seeing=1.1) :
        """Returns calibration vector for this airmass and seeing

        Args:
           airmass : float, airmass, as defined in image headers
           seeing ; float , seeing in arcsec FWHM, as defined in image headers

        Returns:
           calibation vector : 1D[nwave] corresponfing to this class wavelength array self.wave, units are [photons/A]/[1e-17 erg/s/cm^2/A]
        """

        return self.average_calib*10**(-0.4*( (seeing-self.pivot_seeing)*self.seeing_term + (airmass-self.pivot_airmass)*self.atmospheric_extinction ))


    def throughput(self,airmass=1,seeing=1.1) :

        """Returns throughput for this airmass and seeing

        Args:
           airmass : float, airmass, as defined in image headers
           seeing ; float , seeing in arcsec FWHM, as defined in image headers

        Returns:
           throughput (dimensionless, electron per photon hitting the mirror)
        """


        cal_scale = np.ones(self.wave.shape)
        cal_scale *= 1e17 # electrons/A  /  ( ergs/s/cm2/A)
        # reference effective collection area
        area = 8.678709421*1e4 # cm2
        # DESI-347-v15.2 has 0.7647 obscuration by cage and fins
        # and mirror = 3.797m diameter
        # np.pi*(3.797/2)**2*0.7647 = 8.65888886
        # ratio is 0.2% from 1 ... ok
        cal_scale /= area # electrons  /  ergs
        hplanck = 6.62606957e-34 #J.s
        hplanck *= 1e7 # erg.s
        cspeed = 2.99792458e8 # m/s
        cspeed *= 1e10 # A/s
        energy = hplanck*cspeed/self.wave # erg ( erg.s * A/s / A)
        cal_scale *= energy # electrons /photons

        return cal_scale * self.value(airmass,seeing)
