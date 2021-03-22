"""
desispec.magnitude
========================

Broadband flux and magnitudes
"""

import numpy as np

def compute_broadband_flux(spectrum_wave,spectrum_flux,transmission_wave,transmission_value) :
    """
    Computes broadband flux

    Args:

     spectrum_wave: 1D numpy array (Angstrom)
     spectrum_flux: 1D numpy array is some input density unit, same size as spectrum_wave
     transmission_wave: 1D numpy array (Angstrom)
     transmission_value: 1D numpy array , dimensionless, same size as transmission_wave

    Returns:

     integrated flux (unit= A x (input density unit)) , scalar
    """

    # same size
    assert(spectrum_wave.size==spectrum_flux.size)
    assert(transmission_wave.size==transmission_value.size)

    # sort arrays, just in case
    ii=np.argsort(spectrum_wave)
    jj=np.argsort(transmission_wave)

    # tranmission contained in spectrum
    assert(spectrum_wave[ii[0]]<=transmission_wave[jj[0]])
    assert(spectrum_wave[ii[-1]]>=transmission_wave[jj[-1]])

    kk=(spectrum_wave>=transmission_wave[jj[0]])&(spectrum_wave<=transmission_wave[jj[-1]])

    # wavelength grid combining both grids in transmission_wave region
    wave=np.unique(np.hstack([spectrum_wave[kk],transmission_wave]))
    # value is product of interpolated values
    val=np.interp(wave,spectrum_wave[ii],spectrum_flux[ii])*np.interp(wave,transmission_wave[jj],transmission_value[jj])

    trapeze_area = (val[1:]+val[:-1])*(wave[1:]-wave[:-1])/2.
    return np.sum(trapeze_area)

def ab_flux_in_ergs_s_cm2_A(wave) :
    """
    Args:

     wave: 1D numpy array (Angstrom)

    Returns:

      ab flux in units of ergs/s/cm2/A
    """

    #import astropy.units
    #default_wavelength_unit = astropy.units.Angstrom
    #default_flux_unit = astropy.units.erg / astropy.units.cm**2 / astropy.units.s / default_wavelength_unit
    #_ab_constant = 3631. * astropy.units.Jansky * astropy.constants.c).to(default_flux_unit * default_wavelength_unit**2)

    _ab_constant = 0.10885464 # Angstrom erg / (cm2 s)
    return _ab_constant / wave**2

def compute_ab_mag(spectrum_wave,spectrum_flux,transmission_wave,transmission_value) :
    """
    Computes ab mag

    Args:

     spectrum_wave: 1D numpy array (Angstrom)
     spectrum_flux: 1D numpy array (in units of 1e-17 ergs/s/cm2/A), same size as spectrum_wave
     transmission_wave: 1D numpy array (Angstrom)
     transmission_value: 1D numpy array , dimensionless, same size as transmission_wave

    Returns:

     mag (float scalar)
    """

    numerator   = 1e-17*compute_broadband_flux(spectrum_wave,spectrum_flux,transmission_wave,transmission_value)
    # use same wavelength grid for denominator to limit interpolation biases
    denominator = compute_broadband_flux(spectrum_wave,ab_flux_in_ergs_s_cm2_A(spectrum_wave),transmission_wave,transmission_value)

    # may return NaN
    return - 2.5 * np.log10(numerator/denominator)
