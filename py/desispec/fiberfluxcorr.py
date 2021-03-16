"""
desispec.fiberfluxcorr
========================

Routines to compute fiber flux corrections
based on the fiber location, the exposure seeing,
and the target morphology.
"""

import numpy as np

from desimodel.fastfiberacceptance import FastFiberAcceptance
from desimodel.io import load_platescale

def flat_to_psf_flux_correction(fibermap,exposure_seeing_fwhm=1.1) :
    """
    Multiplicative factor to apply to the flat-fielded spectroscopic flux of a fiber
    to calibrate the spectrum of a point source, given the current exposure seeing

    Args:
      fibermap: fibermap of frame, astropy.table.Table
      exposure_seeing_fwhm: seeing FWHM in arcsec

    Returns: 1D numpy array with correction factor to apply to fiber fielded fluxes, valid for point sources.
    """
    #- Compute point source flux correction and fiber flux correction
    fa = FastFiberAcceptance()
    x_mm = fibermap["FIBER_X"]
    y_mm = fibermap["FIBER_Y"]
    bad = np.isnan(x_mm)|np.isnan(y_mm)
    x_mm[bad]=0.
    y_mm[bad]=0.
    dx_mm = fibermap["DELTA_X"] # mm
    dy_mm = fibermap["DELTA_Y"] # mm
    bad = np.isnan(dx_mm)|np.isnan(dy_mm)
    dx_mm[bad]=0.
    dy_mm[bad]=0.

    ps = load_platescale()
    isotropic_platescale = np.interp(x_mm**2+y_mm**2,ps['radius']**2,np.sqrt(ps['radial_platescale']*ps['az_platescale'])) # um/arcsec
    sigmas_um  = exposure_seeing_fwhm/2.35 * isotropic_platescale # um
    offsets_um = np.sqrt(dx_mm**2+dy_mm**2)*1000. # um

    fiber_frac = fa.value("POINT",sigmas_um,offsets_um)
    # at large r,
    #  isotropic_platescale is larger
    #  fiber angular size is smaller
    #  fiber flat is smaller
    #  fiber flat correction is larger
    #  have to divide by isotropic_platescale^2
    ok = (fiber_frac>0.01)
    point_source_correction = np.zeros(x_mm.shape)
    point_source_correction[ok] = 1./fiber_frac[ok]/isotropic_platescale[ok]**2

    # normalize to one because this is a relative correction here
    point_source_correction[ok] /= np.mean(point_source_correction[ok])

    return point_source_correction

def psf_to_fiber_flux_correction(fibermap,exposure_seeing_fwhm=1.1) :
    """
    Multiplicative factor to apply to the psf flux of a fiber
    to obtain the fiber flux, given the current exposure seeing.
    The fiber flux is the flux one would collect for this object in a fiber of 1.5 arcsec diameter,
    for a 1 arcsec seeing, FWHM (same definition as for the Legacy Surveys).

    Args:
      fibermap: fibermap of frame, astropy.table.Table
      exposure_seeing_fwhm: seeing FWHM in arcsec

    Returns: 1D numpy array with correction factor to apply to fiber fielded fluxes, valid for any sources.
    """

    # compute the seeing and plate scale correction

    fa = FastFiberAcceptance()
    x_mm = fibermap["FIBER_X"]
    y_mm = fibermap["FIBER_Y"]
    bad = np.isnan(x_mm)|np.isnan(y_mm)
    x_mm[bad]=0.
    y_mm[bad]=0.
    dx_mm = fibermap["DELTA_X"] # mm
    dy_mm = fibermap["DELTA_Y"] # mm
    bad = np.isnan(dx_mm)|np.isnan(dy_mm)
    dx_mm[bad]=0.
    dy_mm[bad]=0.

    ps = load_platescale()
    isotropic_platescale = np.interp(x_mm**2+y_mm**2,ps['radius']**2,np.sqrt(ps['radial_platescale']*ps['az_platescale'])) # um/arcsec
    # we could include here a wavelength dependence on seeing
    sigmas_um  = exposure_seeing_fwhm/2.35 * isotropic_platescale # um
    offsets_um = np.sqrt(dx_mm**2+dy_mm**2)*1000. # um
    nfibers = len(fibermap)
    point_sources    = (fibermap["MORPHTYPE"]=="PSF")
    extended_sources = ~point_sources
    half_light_radius_arcsec = fibermap["SHAPE_R"]

    # for current seeing, fiber plate scale , fiber size ...
    current_fiber_frac_point_source  = fa.value("POINT",sigmas_um,offsets_um)
    current_fiber_frac = current_fiber_frac_point_source.copy()
    # for the moment use result for an exponential disk profile
    current_fiber_frac[extended_sources] = fa.value("DISK",sigmas_um[extended_sources],offsets_um[extended_sources],half_light_radius_arcsec[extended_sources])

    # for "nominal" fiber size of 1.5 arcsec, and seeing of 1.
    isotropic_platescale = 107/1.5 # um/arcsec
    sigmas_um   = 1.0/2.35 * isotropic_platescale*np.ones(nfibers) # um
    offsets_um  = np.zeros(nfibers) # um , no offset

    nominal_fiber_frac_point_source = fa.value("POINT",sigmas_um,offsets_um)
    nominal_fiber_frac = nominal_fiber_frac_point_source.copy()
    nominal_fiber_frac[extended_sources] = fa.value("DISK",sigmas_um[extended_sources],offsets_um[extended_sources],half_light_radius_arcsec[extended_sources])

    # legacy survey fiber frac
    selection = (fibermap["MORPHTYPE"]=="PSF")&(fibermap["FLUX_R"]>0)
    imaging_fiber_frac_for_point_source = np.sum(fibermap["FIBERFLUX_R"][selection]*fibermap["FLUX_R"][selection])/np.sum(fibermap["FLUX_R"][selection]**2)
    imaging_fiber_frac = imaging_fiber_frac_for_point_source*np.ones(nfibers) # default is value for point sources
    selection = (fibermap["FLUX_R"]>1)
    imaging_fiber_frac[selection] = fibermap["FIBERFLUX_R"][selection]/fibermap["FLUX_R"][selection]
    to_saturate = (imaging_fiber_frac[selection]>imaging_fiber_frac_for_point_source)
    if np.sum(to_saturate)>0 :
        imaging_fiber_frac[selection][to_saturate] = imaging_fiber_frac_for_point_source # max is point source value


    # uncalibrated flux     ~= current_fiber_frac * total_flux
    # psf calibrated flux   ~= current_fiber_frac * total_flux / current_fiber_frac_point_source
    # fiber flux            = nominal_fiber_frac * total_flux
    #
    # to the multiplicative factor to apply to the current psf calibrated flux is:
    #
    # correction_current = (fiber flux)/(psf calibrated flux) = nominal_fiber_frac / current_fiber_frac * current_fiber_frac_point_source
    #
    # if we were to observe in nominal conditions, we would have obtained :
    #
    # correction_nominal = (fiber flux, nominal conditions)/(psf calibrated flux, nominal_conditions) = nominal_fiber_frac_point_source
    #
    # now we have a better measurement of this correction in nominal conditions from the imaging
    #
    # correction_nominal_imaging = FIBER_FLUX_R/FLUX_R = imaging_fiber_frac
    #
    # we return the current correction calibrated with the imaging :
    #
    # correction_final = correction_current * ( correction_nominal_imaging / correction_nominal)
    # correction_final =  nominal_fiber_frac / current_fiber_frac * current_fiber_frac_point_source / nominal_fiber_frac_point_source * imaging_fiber_frac
    #
    corr=np.ones(nfibers)
    ok=(current_fiber_frac>0)&(nominal_fiber_frac>0)&(current_fiber_frac_point_source>0)&(nominal_fiber_frac_point_source>0)
    corr[ok]=(nominal_fiber_frac[ok]/current_fiber_frac[ok])*(current_fiber_frac_point_source[ok]/nominal_fiber_frac_point_source[ok])
    return imaging_fiber_frac * corr
