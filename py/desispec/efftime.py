"""
desispec.efftime
========================

Effective exposure time formulae

"""

import numpy as np


# AR https://desi.lbl.gov/trac/wiki/SurveyOps/SurveySpeed
# AR 2021-03-05: GFA values are now computed for a fiber diameter of 1.52"
def compute_efftime(table,
                    kterm=0.114, # KERM
                    ebv_r_coeff=2.165,
                    fiber_diameter_arcsec=1.52,
                    correct=True):
    """Computes the effective exposure times using transparency and fiber acceptance from the GFAs
    offline analysis, and the sky magnitudes from the spectroscopy.
    Uses the formulae described in https://desi.lbl.gov/trac/wiki/SurveyOps/SurveySpeed
    for the dark, bright and backup programs.

    Args:
      table: astropy.table.Table with columns
         - EXPTIME exposure time in seconds
         - SKY_MAG_R_SPEC AB magnitude per arcsec2 of sky, in decam-r filter
         - EBV E(B-V) from SFD map
         - TRANSPARENCY_GFA transparency (number between 0 and 1, ~1 for photometric nights irrespectively of AIRMASS)
         - AIRMASS (airmass >=1)
         - FIBERFAC_GFA PMGSTARS forced photometry amount of light in 1.52 asec diameter aperture normalized to nominal, assuming a point source profile
         - FIBERFAC_ELG_GFA PMGSTARS forced photometry amount of light in 1.52 asec diameter aperture normalized to nominal, assuming an ELG-like profile (r_half = 0.45 asec exponential)
         - FIBERFAC_BGS_GFA PMGSTARS forced photometry amount of light in 1.52 asec diameter aperture normalized to nominal, assuming a BGS-like profile (r_half = 1.5 asec de Vaucouleurs)
         - FIBER_FRACFLUX_GFA fraction of light in fiber for point source
         - FIBER_FRACFLUX_ELG_GFA fraction of light in fiber for typical ELG source
         - FIBER_FRACFLUX_BGS_GFA fraction of light in fiber for typical BGS source
    """

    exptime  = table["EXPTIME"]
    skymag   = table["SKY_MAG_R_SPEC"]
    sky      = 10**(-0.4*(skymag-22.5)) # nMgy/arcsec**2
    ebv      = table["EBV"]
    transparency = table["TRANSPARENCY_GFA"]
    airmass      = table["AIRMASS"]
    fiberfac_psf = table["FIBERFAC_GFA"] # fiber_frac * transparency normalized to 1 for nominal conditions
    fiberfac_elg = table["FIBERFAC_ELG_GFA"] # fiber_frac * transparency normalized to 1 for nominal conditions
    fiberfac_bgs = table["FIBERFAC_BGS_GFA"] # fiber_frac * transparency normalized to 1 for nominal conditions

    fiber_fracflux_bgs = table["FIBER_FRACFLUX_BGS_GFA"] # fraction of light down fiber
    fiber_fracflux_psf = table["FIBER_FRACFLUX_GFA"]

    exptime_nom = 1000.0  # AR seconds
    sky_nom = 3.73  # AR nMgy/arcsec**2
    flux_bright_nom = 15.8  # nMgy (r=19.5 mag for de Vaucouleurs rhalf=1.5" BGS)
    flux_backup_nom = 27.5  # nMgy (r=18.9 mag star)

    # AR airmass term
    airfac = 10.0 ** (kterm * (airmass - 1.0) / 2.5)
    # AR ebv term
    ebvfac = 10.0 ** (ebv_r_coeff * ebv / 2.5)
    # AR sky readnoise
    sky_rdn = 0.932  # AR nMgy/arcsec**2

    # AR "limit" fiber flux
    fiber_area_arcsec2 = np.pi*(fiber_diameter_arcsec/2)**2

    # flux in fiber artificially divided by fiber_area_arcsec2  because the sky flux is per arcsec2
    fflux_bright = flux_bright_nom * fiber_fracflux_bgs / airfac / ebvfac / fiber_area_arcsec2
    fflux_backup = flux_backup_nom * fiber_fracflux_psf / airfac / ebvfac / fiber_area_arcsec2

    if correct:
        fflux_bright *= transparency
        fflux_backup *= transparency
    
    # AR effective sky
    effsky_dark = (sky + sky_rdn * exptime_nom / exptime) / (1.0 + sky_rdn / sky_nom)
    effsky_bright = (sky + sky_rdn * exptime_nom / exptime + fflux_bright) / (
        1.0 + sky_rdn / sky_nom + fflux_bright / sky_nom
    )
    effsky_backup = (sky + sky_rdn * exptime_nom / exptime + fflux_backup) / (
        1.0 + sky_rdn / sky_nom + fflux_backup / sky_nom
    )
    # AR effective exposure time
    efftime_dark = (
        exptime
        * (fiberfac_elg / airfac) ** 2
        * (sky_nom / effsky_dark)
        / ebvfac ** 2
    )
    efftime_bright = (
        exptime
        * (fiberfac_bgs / airfac) ** 2
        * (sky_nom / effsky_bright)
        / ebvfac ** 2
    )
    efftime_backup = (
        exptime
        * (fiberfac_psf / airfac) ** 2
        * (sky_nom / effsky_backup)
        / ebvfac ** 2
    )

    # set to -1 values with incorrect inputs
    bad=table["AIRMASS"]<0.99
    bad |=(table["FIBER_FRACFLUX_GFA"]==0)
    bad |=(table["TRANSPARENCY_GFA"]>2)
    efftime_dark[bad]=0.
    efftime_bright[bad]=0.
    efftime_backup[bad]=0.

    return efftime_dark , efftime_bright , efftime_backup
