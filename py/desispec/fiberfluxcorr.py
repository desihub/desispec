"""
desispec.fiberfluxcorr
========================

Routines to compute fiber flux corrections
based on the fiber location, the exposure seeing,
and the target morphology.
"""

import numpy as np

from desiutil.log import get_logger
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

    log = get_logger()

    for k in ["FIBER_X","FIBER_Y"] :
        if k not in fibermap.dtype.names :
            log.warning("no column '{}' in fibermap, cannot do the flat_to_psf correction, returning 1")
            return np.ones(len(fibermap))


    #- Compute point source flux correction and fiber flux correction
    fa = FastFiberAcceptance()
    x_mm = fibermap["FIBER_X"]
    y_mm = fibermap["FIBER_Y"]
    bad = np.isnan(x_mm)|np.isnan(y_mm)
    x_mm[bad]=0.
    y_mm[bad]=0.

    if "DELTA_X" in fibermap.dtype.names :
        dx_mm = fibermap["DELTA_X"] # mm
    else :
        log.warning("no column 'DELTA_X' in fibermap, assume DELTA_X=0")
        dx_mm = np.zeros(len(fibermap))

    if "DELTA_Y" in fibermap.dtype.names :
        dy_mm = fibermap["DELTA_Y"] # mm
    else :
        log.warning("no column 'DELTA_Y' in fibermap, assume DELTA_Y=0")
        dy_mm = np.zeros(len(fibermap))

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

def psf_to_fiber_flux_correction(fibermap,exposure_seeing_fwhm=1.1):
    """
    Multiplicative factor to apply to the psf flux of a fiber
    to obtain the fiber flux, given the current exposure seeing.
    The fiber flux is the flux one would collect for this object in a fiber of 1.5 arcsec diameter,
    for a 1 arcsec seeing, FWHM (same definition as for the Legacy Surveys).

    Args:
      fibermap: fibermap of frame, astropy.table.Table
      exposure_seeing_fwhm: seeing FWHM in arcsec
      nominal_profiles:  if true, assume psf2fiber correction for a nominal source profile of given target type. 

    Returns: 1D numpy array with correction factor to apply to fiber fielded fluxes, valid for any sources.
    """

    log = get_logger()

    for k in ["FIBER_X","FIBER_Y"] :
        if k not in fibermap.dtype.names :
            log.warning("no column '{}' in fibermap, cannot do the flat_to_psf correction, returning 1".format(k))
            return np.ones(len(fibermap))

    # compute the seeing and plate scale correction
    fa = FastFiberAcceptance()
    x_mm = fibermap["FIBER_X"]
    y_mm = fibermap["FIBER_Y"]
    bad = np.isnan(x_mm)|np.isnan(y_mm)
    x_mm[bad]=0.
    y_mm[bad]=0.

    if "DELTA_X" in fibermap.dtype.names :
        dx_mm = fibermap["DELTA_X"] # mm
    else :
        log.warning("no column 'DELTA_X' in fibermap, assume = zero")
        dx_mm = np.zeros(len(fibermap))

    if "DELTA_Y" in fibermap.dtype.names :
        dy_mm = fibermap["DELTA_Y"] # mm
    else :
        log.warning("no column 'DELTA_Y' in fibermap, assume = zero")
        dy_mm = np.zeros(len(fibermap))

    bad = np.isnan(dx_mm)|np.isnan(dy_mm)
    dx_mm[bad]=0.
    dy_mm[bad]=0.

    ps = load_platescale()
    isotropic_platescale = np.interp(x_mm**2+y_mm**2,ps['radius']**2,np.sqrt(ps['radial_platescale']*ps['az_platescale'])) # um/arcsec

    # we could include here a wavelength dependence on seeing
    sigmas_um  = (exposure_seeing_fwhm/2.35) * isotropic_platescale # um
    offsets_um = np.sqrt(dx_mm**2+dy_mm**2)*1000. # um
    nfibers = len(fibermap)

    log.info('Median psf microns {:.6f} [{:.6f} to {:.6f}]'.format(np.median(sigmas_um), sigmas_um.min(), sigmas_um.max()))
    log.info('Median fiber offset {:.6f} [{:.6f} to {:.6f}]'.format(np.median(offsets_um), offsets_um.min(), offsets_um.max()))
    
    if "MORPHTYPE" in fibermap.dtype.names :
        point_sources = (fibermap["MORPHTYPE"]=="PSF")
    else :
        log.warning("no column 'MORPHTYPE' in fibermap, assume all point sources.")
        point_sources = np.repeat(True,len(fibermap))

    extended_sources = ~point_sources

    log.info('Calculating relative fiberloss for {} PSF and {} non-PSF targets.'.format(np.count_nonzero(point_sources), np.count_nonzero(extended_sources)))
    
    if "SHAPE_R" in fibermap.dtype.names :
        half_light_radius_arcsec = fibermap["SHAPE_R"]
    else :
        log.warning("no column 'SHAPE_R' in fibermap, assume = 0.00 arcseconds.")
        half_light_radius_arcsec = np.zeros(len(fibermap))

    # saturate half_light_radius_arcsec at 2 arcsec
    # larger values would have extrapolated fiberfrac
    # when in fact the ratio of fiberfrac for different seeing
    # or fiber angular size are similar.
    max_radius=2.0
    half_light_radius_arcsec[half_light_radius_arcsec>max_radius]=max_radius

    # for current seeing, fiber plate scale , fiber size ...                                                                                                                                                                                 
    current_fiber_frac_point_source  = fa.value("POINT",sigmas_um,offsets_um)
    current_fiber_frac = current_fiber_frac_point_source.copy()

    # for the moment use result for an exponential disk profile.                                                                                                                                                                             
    current_fiber_frac[extended_sources] = fa.value("DISK",sigmas_um[extended_sources],offsets_um[extended_sources],ext_half_light_radius_arcsec[extended_sources])
    
    # for "nominal" fiber size of 1.5 arcsec, and seeing of 1.
    nominal_isotropic_platescale = 107/1.5 # um/arcsec
    sigmas_um   = 1.0/2.35 * nominal_isotropic_platescale*np.ones(nfibers) # um
    offsets_um  = np.zeros(nfibers) # um , no offset

    nominal_fiber_frac_point_source = fa.value("POINT",sigmas_um,offsets_um)
    nominal_fiber_frac = nominal_fiber_frac_point_source.copy()
    nominal_fiber_frac[extended_sources] = fa.value("DISK",sigmas_um[extended_sources],offsets_um[extended_sources],ext_half_light_radius_arcsec)

    # legacy survey fiber frac
    #selection = (fibermap["MORPHTYPE"]=="PSF")&(fibermap["FLUX_R"]>0)
    #imaging_fiber_frac_for_point_source = np.sum(fibermap["FIBERFLUX_R"][selection]*fibermap["FLUX_R"][selection])/np.sum(fibermap["FLUX_R"][selection]**2)
    #imaging_fiber_frac = imaging_fiber_frac_for_point_source*np.ones(nfibers) # default is value for point sources
    #selection = (fibermap["FLUX_R"]>1)
    #imaging_fiber_frac[selection] = fibermap["FIBERFLUX_R"][selection]/fibermap["FLUX_R"][selection]
    #to_saturate = (imaging_fiber_frac[selection]>imaging_fiber_frac_for_point_source)
    #if np.sum(to_saturate)>0 :
    #    imaging_fiber_frac[selection][to_saturate] = imaging_fiber_frac_for_point_source # max is point source value

    """
    uncalibrated flux     ~= current_fiber_frac * total_flux
    psf calibrated flux   ~= current_fiber_frac * total_flux / current_fiber_frac_point_source
    fiber flux            = nominal_fiber_frac * total_flux

    the multiplicative factor to apply to the current psf calibrated flux is:
    correction_current = (fiber flux)/(psf calibrated flux) = nominal_fiber_frac / current_fiber_frac * current_fiber_frac_point_source

    multiply by normalization between the fast fiber acceptance computation (using moffat with beta=3.5) and the one
    done for the imaging surveys assuming a Gaussian seeing of sigma=1/2.35 arcsec and a fiber of 1.5 arcsec diameter

    """

    # compute normalization between the fast fiber acceptance computation and the one
    # done the imaging surveys assuming a Gaussian seeing of sigma=1/2.35 arcsec and a fiber of 1.5 arcsec diameter
    scale = 0.789 / np.mean(nominal_fiber_frac_point_source)
    nominal_fiber_frac *= scale

    corr = current_fiber_frac_point_source
    ok = (current_fiber_frac>0.01)
    corr[ok]  *= (nominal_fiber_frac[ok] / current_fiber_frac[ok])
    corr[~ok] *= 0.

    log.info("Computed fiber frac relative to legacy for a nominal seeing fwhm of: {:.6f} arcseconds.".format(exposure_seeing_fwhm))
    
    return corr
