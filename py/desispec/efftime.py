"""
desispec.efftime
========================

Effective exposure time formulae

"""

import numpy as np


# AR https://desi.lbl.gov/trac/wiki/SurveyOps/SurveySpeed
# AR 2021-03-05: GFA values are now computed for a fiber diameter of 1.52"
def compute_efftime(table,
                    ffrac_psf_nom=0.58176816, # FRACFLUX_NOMINAL_POINTSOURCE in Aaron's table
                    ffrac_elg_nom=0.42423388, # FRACFLUX_NOMINAL_ELG
                    ffrac_bgs_nom=0.19544029, # FRACFLUX_NOMINAL_BGS
                    kterm=0.114, # KERM
                    ebv_r_coeff=2.165,
                    fiber_diameter_arcsec=1.52):
    """Computes the effective exposure times using transparency and fiber acceptance from the GFAs
    offline analysis, and the sky magnitudes from the spectroscopy.
    Uses the formulae described in https://desi.lbl.gov/trac/wiki/SurveyOps/SurveySpeed
    for the dark,bright and backup programs.

    Args:
      table: astropy.table.Table with columns
         - EXPTIME exposure time in seconds
         - SKY_MAG_R_SPEC AB magnitude per arcsec2 of sky, in decam-r filter
         - EBV E(B-V) from SFD map
         - TRANSPARENCY_GFA transparency (number between 0 and 1, ~1 for photometric nights irrespectively of AIRMASS)
         - AIRMASS (airmass >=1)
         - FIBER_FRACFLUX_GFA fraction of point source flux in fiber aperture (number between 0 and 1)
         - FIBER_FRACFLUX_ELG_GFA fraction of flux for a typical ELG in fiber aperture (number between 0 and 1)
         - FIBER_FRACFLUX_BGS_GFA fraction of flux for a typical BGS target in fiber aperture (number between 0 and 1)
    """

    exptime  = table["EXPTIME"]
    skymag   = table["SKY_MAG_R_SPEC"]
    sky      = 10**(-0.4*(skymag-22.5)) # nMgy/arcsec**2
    ebv      = table["EBV"]
    transparency = table["TRANSPARENCY_GFA"]
    airmass      = table["AIRMASS"]
    ffrac_psf    = table["FIBER_FRACFLUX_GFA"] #
    ffrac_elg    = table["FIBER_FRACFLUX_ELG_GFA"] #
    ffrac_bgs    = table["FIBER_FRACFLUX_BGS_GFA"] #

    exptime_nom = 1000.0  # AR seconds
    sky_nom = 3.73  # AR nMgy/arcsec**2
    fflux_bright_nom = (
        15.8  # AR nMgy/arcsec**2 (r=19.5 mag for de Vaucouleurs rhalf=1.5" BGS)
    )
    fflux_backup_nom = 27.5  # AR nMgy/arcsec**2 (r=18.9 mag star)
    # AR airmass term
    airfac = 10.0 ** (kterm * (airmass - 1.0) / 2.5)
    # AR ebv term
    ebvfac = 10.0 ** (ebv_r_coeff * ebv / 2.5)
    # AR sky readnoise
    sky_rdn = 0.932  # AR nMgy/arcsec**2

    fiber_area_arcsec2 = np.pi*(fiber_diameter_arcsec/2.)**2

    # AR "limit" fiber flux
    fflux_bright = (
        ffrac_bgs * transparency / airfac * fflux_bright_nom / fiber_area_arcsec2
    )
    fflux_backup = (
        ffrac_psf * transparency / airfac * fflux_backup_nom / fiber_area_arcsec2
    )
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
        * (ffrac_elg / ffrac_elg_nom * transparency / airfac) ** 2
        * (sky_nom / effsky_dark)
        / ebvfac ** 2
    )
    efftime_bright = (
        exptime
        * (ffrac_bgs / ffrac_bgs_nom * transparency / airfac) ** 2
        * (sky_nom / effsky_bright)
        / ebvfac ** 2
    )
    efftime_backup = (
        exptime
        * (ffrac_psf / ffrac_psf_nom * transparency / airfac) ** 2
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
