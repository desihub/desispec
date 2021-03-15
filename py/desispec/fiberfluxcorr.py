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

def point_source_fiber_flux_correction(fibermap,exposure_seeing_fwhm=1.1) :
    """
    Multiplicative factor to apply to the flatfielded spectroscopic flux of a fiber
    for a point source, given the current exposure seeing
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
