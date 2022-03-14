# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.database.redshift
==========================

Code for loading spectroscopic pipeline results (specifically redshifts)
into a database.

Notes
-----
* Future devlopment:

  - Plan for how to support fuji+guadalupe combined analysis.  May need to look
    into cross-schema views, or daughter tables that inherit from both schemas.
  - Anticipating loading afterburners and VACs into the database.

"""
import os
import re
import glob
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table, MaskedColumn
from astropy.time import Time
from pytz import utc

from sqlalchemy import (create_engine, event, ForeignKey, Column, DDL,
                        BigInteger, Boolean, Integer, String, Float, DateTime,
                        SmallInteger, bindparam)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import scoped_session, sessionmaker, relationship
from sqlalchemy.schema import CreateSchema
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION, REAL

from desiutil.iers import freeze_iers
from desiutil.log import get_logger, DEBUG, INFO

from ..io.meta import specprod_root, faflavor2program
from ..io.util import checkgzip
from .util import convert_dateobs, parse_pgpass, cameraid

Base = declarative_base()
engine = None
dbSession = scoped_session(sessionmaker())
schemaname = None
log = None


class SchemaMixin(object):
    """Mixin class to allow schema name to be changed at runtime. Also
    automatically sets the table name.
    """

    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    @declared_attr
    def __table_args__(cls):
        return {'schema': schemaname}


# class Truth(SchemaMixin, Base):
#     """Representation of the truth table.
#     """
#
#     targetid = Column(BigInteger, primary_key=True, autoincrement=False)
#     mockid = Column(BigInteger, nullable=False)
#     truez = Column(Float, nullable=False)
#     truespectype = Column(String, nullable=False)
#     templatetype = Column(String, nullable=False)
#     templatesubtype = Column(String, nullable=False)
#     templateid = Column(Integer, nullable=False)
#     seed = Column(BigInteger, nullable=False)
#     mag = Column(Float, nullable=False)
#     magfilter = Column(String, nullable=False)
#     flux_g = Column(Float, nullable=False)
#     flux_r = Column(Float, nullable=False)
#     flux_z = Column(Float, nullable=False)
#     flux_w1 = Column(Float, nullable=False)
#     flux_w2 = Column(Float, nullable=False)
#     flux_w3 = Column(Float, nullable=False)
#     flux_w4 = Column(Float, nullable=False)
#     oiiflux = Column(Float, nullable=False, default=-9999.0)
#     hbetaflux = Column(Float, nullable=False, default=-9999.0)
#     ewoii = Column(Float, nullable=False, default=-9999.0)
#     ewhbeta = Column(Float, nullable=False, default=-9999.0)
#     d4000 = Column(Float, nullable=False, default=-9999.0)
#     vdisp = Column(Float, nullable=False, default=-9999.0)
#     oiidoublet = Column(Float, nullable=False, default=-9999.0)
#     oiiihbeta = Column(Float, nullable=False, default=-9999.0)
#     oiihbeta = Column(Float, nullable=False, default=-9999.0)
#     niihbeta = Column(Float, nullable=False, default=-9999.0)
#     siihbeta = Column(Float, nullable=False, default=-9999.0)
#     mabs_1450 = Column(Float, nullable=False, default=-9999.0)
#     bal_templateid = Column(Integer, nullable=False, default=-1)
#     truez_norsd = Column(Float, nullable=False, default=-9999.0)
#     teff = Column(Float, nullable=False, default=-9999.0)
#     logg = Column(Float, nullable=False, default=-9999.0)
#     feh = Column(Float, nullable=False, default=-9999.0)
#
#     def __repr__(self):
#         return "<Truth(targetid={0.targetid:d})>".format(self)


# class Tractor(SchemaMixin, Base):
#     """Representation of the TRACTORPHOT table in tractorphot files.
#
#     Notes
#     -----
#     The various ``APFLUX`` (aperture flux) and ``LC`` (light curve) columns,
#     which are vector-valued, are not yet implemented.
#     """
#
#     release = Column(SmallInteger, nullable=False)
#     brickid = Column(Integer, nullable=False)
#     brickname = Column(String(8), nullable=False)
#     objid = Column(Integer, nullable=False)
#     brick_primary = Column(Boolean, nullable=False)
#     maskbits = Column(SmallInteger, nullable=False)
#     fitbits = Column(SmallInteger, nullable=False)
#     morphtype = Column(String(3), nullable=False)
#     ra = Column(DOUBLE_PRECISION, nullable=False)
#     dec = Column(DOUBLE_PRECISION, nullable=False)
#     ra_ivar = Column(REAL, nullable=False)
#     dec_ivar = Column(REAL, nullable=False)
#     bx = Column(REAL, nullable=False)
#     by = Column(REAL, nullable=False)
#     dchisq_psf = Column(REAL, nullable=False)
#     dchisq_rex = Column(REAL, nullable=False)
#     dchisq_dev = Column(REAL, nullable=False)
#     dchisq_exp = Column(REAL, nullable=False)
#     dchisq_ser = Column(REAL, nullable=False)
#     ebv = Column(REAL, nullable=False)
#     mjd_min = Column(DOUBLE_PRECISION, nullable=False)
#     mjd_max = Column(DOUBLE_PRECISION, nullable=False)
#     ref_cat = Column(String(2), nullable=False)
#     ref_id = Column(BigInteger, nullable=False)
#     pmra = Column(REAL, nullable=False)
#     pmdec = Column(REAL, nullable=False)
#     parallax = Column(REAL, nullable=False)
#     pmra_ivar = Column(REAL, nullable=False)
#     pmdec_ivar = Column(REAL, nullable=False)
#     parallax_ivar = Column(REAL, nullable=False)
#     ref_epoch = Column(REAL, nullable=False)
#     gaia_phot_g_mean_mag = Column(REAL, nullable=False)
#     gaia_phot_g_mean_flux_over_error = Column(REAL, nullable=False)
#     gaia_phot_g_n_obs = Column(SmallInteger, nullable=False)
#     gaia_phot_bp_mean_mag = Column(REAL, nullable=False)
#     gaia_phot_bp_mean_flux_over_error = Column(REAL, nullable=False)
#     gaia_phot_bp_n_obs = Column(SmallInteger, nullable=False)
#     gaia_phot_rp_mean_mag = Column(REAL, nullable=False)
#     gaia_phot_rp_mean_flux_over_error = Column(REAL, nullable=False)
#     gaia_phot_rp_n_obs = Column(SmallInteger, nullable=False)
#     gaia_phot_variable_flag = Column(Boolean, nullable=False)
#     gaia_astrometric_excess_noise = Column(REAL, nullable=False)
#     gaia_astrometric_excess_noise_sig = Column(REAL, nullable=False)
#     gaia_astrometric_n_obs_al = Column(SmallInteger, nullable=False)
#     gaia_astrometric_n_good_obs_al = Column(SmallInteger, nullable=False)
#     gaia_astrometric_weight_al = Column(REAL, nullable=False)
#     gaia_duplicated_source = Column(Boolean, nullable=False)
#     gaia_a_g_val = Column(REAL, nullable=False)
#     gaia_e_bp_min_rp_val = Column(REAL, nullable=False)
#     gaia_phot_bp_rp_excess_factor = Column(REAL, nullable=False)
#     gaia_astrometric_sigma5d_max = Column(REAL, nullable=False)
#     gaia_astrometric_params_solved = Column(SmallInteger, nullable=False)
#     flux_g = Column(REAL, nullable=False)
#     flux_r = Column(REAL, nullable=False)
#     flux_z = Column(REAL, nullable=False)
#     flux_w1 = Column(REAL, nullable=False)
#     flux_w2 = Column(REAL, nullable=False)
#     flux_w3 = Column(REAL, nullable=False)
#     flux_w4 = Column(REAL, nullable=False)
#     flux_ivar_g = Column(REAL, nullable=False)
#     flux_ivar_r = Column(REAL, nullable=False)
#     flux_ivar_z = Column(REAL, nullable=False)
#     flux_ivar_w1 = Column(REAL, nullable=False)
#     flux_ivar_w2 = Column(REAL, nullable=False)
#     flux_ivar_w3 = Column(REAL, nullable=False)
#     flux_ivar_w4 = Column(REAL, nullable=False)
#     fiberflux_g = Column(REAL, nullable=False)
#     fiberflux_r = Column(REAL, nullable=False)
#     fiberflux_z = Column(REAL, nullable=False)
#     fibertotflux_g = Column(REAL, nullable=False)
#     fibertotflux_r = Column(REAL, nullable=False)
#     fibertotflux_z = Column(REAL, nullable=False)
#     # APFLUX...
#     mw_transmission_g = Column(REAL, nullable=False)
#     mw_transmission_r = Column(REAL, nullable=False)
#     mw_transmission_z = Column(REAL, nullable=False)
#     mw_transmission_w1 = Column(REAL, nullable=False)
#     mw_transmission_w2 = Column(REAL, nullable=False)
#     mw_transmission_w3 = Column(REAL, nullable=False)
#     mw_transmission_w4 = Column(REAL, nullable=False)
#     nobs_g = Column(SmallInteger, nullable=False)
#     nobs_r = Column(SmallInteger, nullable=False)
#     nobs_z = Column(SmallInteger, nullable=False)
#     nobs_w1 = Column(SmallInteger, nullable=False)
#     nobs_w2 = Column(SmallInteger, nullable=False)
#     nobs_w3 = Column(SmallInteger, nullable=False)
#     nobs_w4 = Column(SmallInteger, nullable=False)
#     fracflux_g = Column(REAL, nullable=False)
#     fracflux_r = Column(REAL, nullable=False)
#     fracflux_z = Column(REAL, nullable=False)
#     fracflux_w1 = Column(REAL, nullable=False)
#     fracflux_w2 = Column(REAL, nullable=False)
#     fracflux_w3 = Column(REAL, nullable=False)
#     fracflux_w4 = Column(REAL, nullable=False)
#     fracmasked_g = Column(REAL, nullable=False)
#     fracmasked_r = Column(REAL, nullable=False)
#     fracmasked_z = Column(REAL, nullable=False)
#     fracin_g = Column(REAL, nullable=False)
#     fracin_r = Column(REAL, nullable=False)
#     fracin_z = Column(REAL, nullable=False)
#     anymask_g = Column(SmallInteger, nullable=False)
#     anymask_r = Column(SmallInteger, nullable=False)
#     anymask_z = Column(SmallInteger, nullable=False)
#     allmask_g = Column(SmallInteger, nullable=False)
#     allmask_r = Column(SmallInteger, nullable=False)
#     allmask_z = Column(SmallInteger, nullable=False)
#     wisemask_w1 = Column(SmallInteger, nullable=False)
#     wisemask_w2 = Column(SmallInteger, nullable=False)
#     psfsize_g = Column(REAL, nullable=False)
#     psfsize_r = Column(REAL, nullable=False)
#     psfsize_z = Column(REAL, nullable=False)
#     psfdepth_g = Column(REAL, nullable=False)
#     psfdepth_r = Column(REAL, nullable=False)
#     psfdepth_z = Column(REAL, nullable=False)
#     galdepth_g = Column(REAL, nullable=False)
#     galdepth_r = Column(REAL, nullable=False)
#     galdepth_z = Column(REAL, nullable=False)
#     nea_g = Column(REAL, nullable=False)
#     nea_r = Column(REAL, nullable=False)
#     nea_z = Column(REAL, nullable=False)
#     blob_nea_g = Column(REAL, nullable=False)
#     blob_nea_r = Column(REAL, nullable=False)
#     blob_nea_z = Column(REAL, nullable=False)
#     psfdepth_w1 = Column(REAL, nullable=False)
#     psfdepth_w2 = Column(REAL, nullable=False)
#     psfdepth_w3 = Column(REAL, nullable=False)
#     psfdepth_w4 = Column(REAL, nullable=False)
#     wise_coadd_id = Column(String(8), nullable=False)
#     wise_x = Column(REAL, nullable=False)
#     wise_y = Column(REAL, nullable=False)
#     # LC FLUX...
#     sersic = Column(REAL, nullable=False)
#     sersic_ivar = Column(REAL, nullable=False)
#     shape_r = Column(REAL, nullable=False)
#     shape_r_ivar = Column(REAL, nullable=False)
#     shape_e1 = Column(REAL, nullable=False)
#     shape_e1_ivar = Column(REAL, nullable=False)
#     shape_e2 = Column(REAL, nullable=False)
#     shape_e2_ivar = Column(REAL, nullable=False)
#     photsys = Column(String(1), nullable=False)
#     targetid = Column(BigInteger, primary_key=True, autoincrement=False)
#
#     def __repr__(self):
#         return "Tractor(targetid={0.targetid})".format(self)


class Target(SchemaMixin, Base):
    """Representation of the ``TARGETPHOT`` table in the targetphot files.

    Notes
    -----
    The various ``LC`` (light curve) columns,
    which are vector-valued, are not yet implemented.
    """

    release = Column(SmallInteger, nullable=False)
    brickid = Column(Integer, nullable=False)
    brickname = Column(String(8), nullable=False)
    brick_objid = Column(Integer, nullable=False)
    morphtype = Column(String(4), nullable=False)
    ra = Column(DOUBLE_PRECISION, nullable=False)
    dec = Column(DOUBLE_PRECISION, nullable=False)
    ra_ivar = Column(REAL, nullable=False)
    dec_ivar = Column(REAL, nullable=False)
    dchisq_psf = Column(REAL, nullable=False)
    dchisq_rex = Column(REAL, nullable=False)
    dchisq_dev = Column(REAL, nullable=False)
    dchisq_exp = Column(REAL, nullable=False)
    dchisq_ser = Column(REAL, nullable=False)
    ebv = Column(REAL, nullable=False)
    flux_g = Column(REAL, nullable=False)
    flux_r = Column(REAL, nullable=False)
    flux_z = Column(REAL, nullable=False)
    flux_ivar_g = Column(REAL, nullable=False)
    flux_ivar_r = Column(REAL, nullable=False)
    flux_ivar_z = Column(REAL, nullable=False)
    mw_transmission_g = Column(REAL, nullable=False)
    mw_transmission_r = Column(REAL, nullable=False)
    mw_transmission_z = Column(REAL, nullable=False)
    fracflux_g = Column(REAL, nullable=False)
    fracflux_r = Column(REAL, nullable=False)
    fracflux_z = Column(REAL, nullable=False)
    fracmasked_g = Column(REAL, nullable=False)
    fracmasked_r = Column(REAL, nullable=False)
    fracmasked_z = Column(REAL, nullable=False)
    fracin_g = Column(REAL, nullable=False)
    fracin_r = Column(REAL, nullable=False)
    fracin_z = Column(REAL, nullable=False)
    nobs_g = Column(SmallInteger, nullable=False)
    nobs_r = Column(SmallInteger, nullable=False)
    nobs_z = Column(SmallInteger, nullable=False)
    psfdepth_g = Column(REAL, nullable=False)
    psfdepth_r = Column(REAL, nullable=False)
    psfdepth_z = Column(REAL, nullable=False)
    galdepth_g = Column(REAL, nullable=False)
    galdepth_r = Column(REAL, nullable=False)
    galdepth_z = Column(REAL, nullable=False)
    flux_w1 = Column(REAL, nullable=False)
    flux_w2 = Column(REAL, nullable=False)
    flux_w3 = Column(REAL, nullable=False)
    flux_w4 = Column(REAL, nullable=False)
    flux_ivar_w1 = Column(REAL, nullable=False)
    flux_ivar_w2 = Column(REAL, nullable=False)
    flux_ivar_w3 = Column(REAL, nullable=False)
    flux_ivar_w4 = Column(REAL, nullable=False)
    mw_transmission_w1 = Column(REAL, nullable=False)
    mw_transmission_w2 = Column(REAL, nullable=False)
    mw_transmission_w3 = Column(REAL, nullable=False)
    mw_transmission_w4 = Column(REAL, nullable=False)
    allmask_g = Column(SmallInteger, nullable=False)
    allmask_r = Column(SmallInteger, nullable=False)
    allmask_z = Column(SmallInteger, nullable=False)
    fiberflux_g = Column(REAL, nullable=False)
    fiberflux_r = Column(REAL, nullable=False)
    fiberflux_z = Column(REAL, nullable=False)
    fibertotflux_g = Column(REAL, nullable=False)
    fibertotflux_r = Column(REAL, nullable=False)
    fibertotflux_z = Column(REAL, nullable=False)
    ref_epoch = Column(REAL, nullable=False)
    wisemask_w1 = Column(SmallInteger, nullable=False)
    wisemask_w2 = Column(SmallInteger, nullable=False)
    maskbits = Column(SmallInteger, nullable=False)
    # LC_...
    shape_r = Column(REAL, nullable=False)
    shape_r_ivar = Column(REAL, nullable=False)
    shape_e1 = Column(REAL, nullable=False)
    shape_e1_ivar = Column(REAL, nullable=False)
    shape_e2 = Column(REAL, nullable=False)
    shape_e2_ivar = Column(REAL, nullable=False)
    sersic = Column(REAL, nullable=False)
    sersic_ivar = Column(REAL, nullable=False)
    ref_id = Column(BigInteger, nullable=False)
    ref_cat = Column(String(2), nullable=False)
    gaia_phot_g_mean_mag = Column(REAL, nullable=False)
    gaia_phot_g_mean_flux_over_error = Column(REAL, nullable=False)
    gaia_phot_bp_mean_mag = Column(REAL, nullable=False)
    gaia_phot_bp_mean_flux_over_error = Column(REAL, nullable=False)
    gaia_phot_rp_mean_mag = Column(REAL, nullable=False)
    gaia_phot_rp_mean_flux_over_error = Column(REAL, nullable=False)
    gaia_phot_bp_rp_excess_factor = Column(REAL, nullable=False)
    gaia_duplicated_source = Column(Boolean, nullable=False)
    gaia_astrometric_sigma5d_max = Column(REAL, nullable=False)
    gaia_astrometric_params_solved = Column(Boolean, nullable=False)
    parallax = Column(REAL, nullable=False)
    parallax_ivar = Column(REAL, nullable=False)
    pmra = Column(REAL, nullable=False)
    pmra_ivar = Column(REAL, nullable=False)
    pmdec = Column(REAL, nullable=False)
    pmdec_ivar = Column(REAL, nullable=False)
    photsys = Column(String(1), nullable=False)
    targetid = Column(BigInteger, primary_key=True, autoincrement=False)
    subpriority = Column(DOUBLE_PRECISION, nullable=False)
    obsconditions = Column(BigInteger, nullable=False)
    priority_init = Column(BigInteger, nullable=False)
    numobs_init = Column(BigInteger, nullable=False)
    hpxpixel = Column(BigInteger, nullable=False)
    cmx_target = Column(BigInteger, nullable=False)
    desi_target = Column(BigInteger, nullable=False)
    bgs_target = Column(BigInteger, nullable=False)
    mws_target = Column(BigInteger, nullable=False)
    sv1_desi_target = Column(BigInteger, nullable=False)
    sv1_bgs_target = Column(BigInteger, nullable=False)
    sv1_mws_target = Column(BigInteger, nullable=False)
    sv2_desi_target = Column(BigInteger, nullable=False)
    sv2_bgs_target = Column(BigInteger, nullable=False)
    sv2_mws_target = Column(BigInteger, nullable=False)
    sv3_desi_target = Column(BigInteger, nullable=False)
    sv3_bgs_target = Column(BigInteger, nullable=False)
    sv3_mws_target = Column(BigInteger, nullable=False)
    scnd_target = Column(BigInteger, nullable=False)
    sv1_scnd_target = Column(BigInteger, nullable=False)
    sv2_scnd_target = Column(BigInteger, nullable=False)
    sv3_scnd_target = Column(BigInteger, nullable=False)

    # fiberassign = relationship("Fiberassign", back_populates="target")
    # potential = relationship("Potential", back_populates="target")
    # zpix_redshifts = relationship("Zpix", back_populates="target")
    # ztile_redshifts = relationship("Ztile", back_populates="target")

    def __repr__(self):
        return "Target(targetid={0.targetid})".format(self)


class Tile(SchemaMixin, Base):
    """Representation of the tiles file.

    Notes
    -----
    Most of the data that are currently in the tiles file are derivable
    from the exposures table with much greater precision::

        CREATE VIEW f5.tile AS SELECT tileid,
            -- SURVEY, FAPRGRM, FAFLAVOR?
            COUNT(*) AS nexp, SUM(exptime) AS exptime,
            MIN(tilera) AS tilera, MIN(tiledec) AS tiledec,
            SUM(efftime_etc) AS efftime_etc, SUM(efftime_spec) AS efftime_spec,
            SUM(efftime_gfa) AS efftime_gfa, MIN(goaltime) AS goaltime,
            -- OBSSTATUS?
            SUM(lrg_efftime_dark) AS lrg_efftime_dark,
            SUM(elg_efftime_dark) AS elg_efftime_dark,
            SUM(bgs_efftime_bright) AS bgs_efftime_bright,
            SUM(lya_efftime_dark) AS lya_efftime_dark,
            -- GOALTYPE?
            MIN(mintfrac) AS mintfrac, MAX(night) AS lastnight
        FROM f5.exposure GROUP BY tileid;

    However because of some unresolved discrepancies, we'll just load the
    full tiles file for now.
    """

    tileid = Column(Integer, primary_key=True, autoincrement=False)
    survey = Column(String(20), nullable=False)
    program = Column(String(6), nullable=False)
    faprgrm = Column(String(20), nullable=False)
    faflavor = Column(String(20), nullable=False)
    nexp = Column(BigInteger, nullable=False)  # In principle this could be replaced by a count of exposures
    exptime = Column(DOUBLE_PRECISION, nullable=False)
    tilera = Column(DOUBLE_PRECISION, nullable=False)   #- Calib exposures don't have RA, dec
    tiledec = Column(DOUBLE_PRECISION, nullable=False)
    efftime_etc = Column(DOUBLE_PRECISION, nullable=False)
    efftime_spec = Column(DOUBLE_PRECISION, nullable=False)
    efftime_gfa = Column(DOUBLE_PRECISION, nullable=False)
    goaltime = Column(DOUBLE_PRECISION, nullable=False)
    obsstatus = Column(String(20), nullable=False)
    lrg_efftime_dark = Column(DOUBLE_PRECISION, nullable=False)
    elg_efftime_dark = Column(DOUBLE_PRECISION, nullable=False)
    bgs_efftime_bright = Column(DOUBLE_PRECISION, nullable=False)
    lya_efftime_dark = Column(DOUBLE_PRECISION, nullable=False)
    goaltype = Column(String(20), nullable=False)
    mintfrac = Column(DOUBLE_PRECISION, nullable=False)
    lastnight = Column(Integer, nullable=False) # In principle this could be replaced by MAX(night) grouped by exposures.

    exposures = relationship("Exposure", back_populates="tile")
    fiberassign = relationship("Fiberassign", back_populates="tile")
    potential = relationship("Potential", back_populates="tile")
    ztile_redshifts = relationship("Ztile", back_populates="tile")

    def __repr__(self):
        return "Tile(tileid={0.tileid:d})".format(self)


class Exposure(SchemaMixin, Base):
    """Representation of the EXPOSURES HDU in the exposures file.

    Notes
    -----
    The column ``program`` is filled in via :func:`~desispec.io.meta.faflavor2program`.
    """

    night = Column(Integer, nullable=False, index=True)
    expid = Column(Integer, primary_key=True, autoincrement=False)
    tileid = Column(Integer, ForeignKey('tile.tileid'), nullable=False, index=True)
    tilera = Column(DOUBLE_PRECISION, nullable=False)   #- Calib exposures don't have RA, dec
    tiledec = Column(DOUBLE_PRECISION, nullable=False)
    date_obs = Column(DateTime(True), nullable=False)
    mjd = Column(DOUBLE_PRECISION, nullable=False)
    survey = Column(String(7), nullable=False)
    program = Column(String(6), nullable=False)
    faprgrm = Column(String(16), nullable=False)
    faflavor = Column(String(19), nullable=False)
    exptime = Column(DOUBLE_PRECISION, nullable=False)
    efftime_spec = Column(DOUBLE_PRECISION, nullable=False)
    goaltime = Column(DOUBLE_PRECISION, nullable=False)
    goaltype = Column(String(6), nullable=False)
    mintfrac = Column(DOUBLE_PRECISION, nullable=False)
    airmass = Column(REAL, nullable=False)
    ebv = Column(DOUBLE_PRECISION, nullable=False)
    seeing_etc = Column(DOUBLE_PRECISION, nullable=False)
    efftime_etc = Column(REAL, nullable=False)
    tsnr2_elg = Column(REAL, nullable=False)
    tsnr2_qso = Column(REAL, nullable=False)
    tsnr2_lrg = Column(REAL, nullable=False)
    tsnr2_lya = Column(DOUBLE_PRECISION, nullable=False)
    tsnr2_bgs = Column(REAL, nullable=False)
    tsnr2_gpbdark = Column(REAL, nullable=False)
    tsnr2_gpbbright = Column(REAL, nullable=False)
    tsnr2_gpbbackup = Column(REAL, nullable=False)
    lrg_efftime_dark = Column(REAL, nullable=False)
    elg_efftime_dark = Column(REAL, nullable=False)
    bgs_efftime_bright = Column(REAL, nullable=False)
    lya_efftime_dark = Column(DOUBLE_PRECISION, nullable=False)
    gpb_efftime_dark = Column(REAL, nullable=False)
    gpb_efftime_bright = Column(REAL, nullable=False)
    gpb_efftime_backup = Column(REAL, nullable=False)
    transparency_gfa = Column(DOUBLE_PRECISION, nullable=False)
    seeing_gfa = Column(DOUBLE_PRECISION, nullable=False)
    fiber_fracflux_gfa = Column(DOUBLE_PRECISION, nullable=False)
    fiber_fracflux_elg_gfa = Column(DOUBLE_PRECISION, nullable=False)
    fiber_fracflux_bgs_gfa = Column(DOUBLE_PRECISION, nullable=False)
    fiberfac_gfa = Column(DOUBLE_PRECISION, nullable=False)
    fiberfac_elg_gfa = Column(DOUBLE_PRECISION, nullable=False)
    fiberfac_bgs_gfa = Column(DOUBLE_PRECISION, nullable=False)
    airmass_gfa = Column(DOUBLE_PRECISION, nullable=False)
    sky_mag_ab_gfa = Column(DOUBLE_PRECISION, nullable=False)
    sky_mag_g_spec = Column(DOUBLE_PRECISION, nullable=False)
    sky_mag_r_spec = Column(DOUBLE_PRECISION, nullable=False)
    sky_mag_z_spec = Column(DOUBLE_PRECISION, nullable=False)
    efftime_gfa = Column(DOUBLE_PRECISION, nullable=False)
    efftime_dark_gfa = Column(DOUBLE_PRECISION, nullable=False)
    efftime_bright_gfa = Column(DOUBLE_PRECISION, nullable=False)
    efftime_backup_gfa = Column(DOUBLE_PRECISION, nullable=False)

    tile = relationship("Tile", back_populates="exposures")
    frames = relationship("Frame", back_populates="exposure")

    def __repr__(self):
        return "Exposure(night={0.night:d}, expid={0.expid:d}, tileid={0.tileid:d})".format(self)


class Frame(SchemaMixin, Base):
    """Representation of the FRAMES HDU in the exposures file.

    Notes
    -----
    The column ``frameid`` is a combination of ``expid`` and the camera name::

        frameid = 100*expid + cameraid(camera)

    where ``cameraid()`` is :func:`desispec.database.util.cameraid`.
    """

    frameid = Column(Integer, primary_key=True, autoincrement=False)  # Arbitrary integer composed from expid + cameraid
    # frameid = Column(BigInteger, primary_key=True, autoincrement=True)
    night = Column(Integer, nullable=False, index=True)
    expid = Column(Integer, ForeignKey('exposure.expid'), nullable=False)
    tileid = Column(Integer, nullable=False, index=True)
    #  4 TILERA               D
    #  5 TILEDEC              D
    #  6 MJD                  D
    mjd = Column(DOUBLE_PRECISION, nullable=False)
    #  7 EXPTIME              E
    exptime = Column(REAL, nullable=False)
    #  8 AIRMASS              E
    #  9 EBV                  E
    ebv = Column(REAL, nullable=False)
    # 10 SEEING_ETC           D
    # 11 EFFTIME_ETC          E
    # 12 CAMERA               2A
    camera = Column(String(2), nullable=False)
    # 13 TSNR2_GPBDARK        E
    # 14 TSNR2_ELG            E
    # 15 TSNR2_GPBBRIGHT      E
    # 16 TSNR2_LYA            D
    # 17 TSNR2_BGS            E
    # 18 TSNR2_GPBBACKUP      E
    # 19 TSNR2_QSO            E
    # 20 TSNR2_LRG            E
    tsnr2_gpbdark = Column(REAL, nullable=False)
    tsnr2_elg = Column(REAL, nullable=False)
    tsnr2_gpbbright = Column(REAL, nullable=False)
    tsnr2_lya = Column(DOUBLE_PRECISION, nullable=False)
    tsnr2_bgs = Column(REAL, nullable=False)
    tsnr2_gpbbackup = Column(REAL, nullable=False)
    tsnr2_qso = Column(REAL, nullable=False)
    tsnr2_lrg = Column(REAL, nullable=False)
    # 21 SURVEY               7A
    # 22 GOALTYPE             6A
    # 23 FAPRGRM              15A
    # 24 FAFLAVOR             18A
    # 25 MINTFRAC             D
    # 26 GOALTIME             D

    exposure = relationship("Exposure", back_populates="frames")

    def __repr__(self):
        return "Frame(expid={0.expid:d}, camera='{0.camera}')".format(self)


class Fiberassign(SchemaMixin, Base):
    """Representation of the FIBERASSIGN table in a fiberassign file.

    Notes
    -----
    * Targets are assigned to a ``location``.  A ``location`` happens to
      correspond to a ``fiber``, but this correspondence could change over
      time, and therefore should not be assumed to be a rigid 1:1 mapping.
    * ``PLATE_RA``, ``PLATE_DEC`` are sometimes missing.  These can be
      copies of ``TARGET_RA``, ``TARGET_DEC``, but in principle they could
      be different if chromatic offsets in targeting positions were
      ever implemented.
    """

    tileid = Column(Integer, ForeignKey('tile.tileid'), primary_key=True, index=True)
    targetid = Column(BigInteger, primary_key=True, index=True)  # potential ForeignKey on Target
    petal_loc = Column(SmallInteger, nullable=False)
    device_loc = Column(Integer, nullable=False)
    location = Column(Integer, primary_key=True)
    fiber = Column(Integer, nullable=False)
    fiberstatus = Column(Integer, nullable=False)
    target_ra = Column(DOUBLE_PRECISION, nullable=False)
    target_dec = Column(DOUBLE_PRECISION, nullable=False)
    lambda_ref = Column(REAL, nullable=False)
    fa_target = Column(BigInteger, nullable=False)
    fa_type = Column(SmallInteger, nullable=False)
    fiberassign_x = Column(REAL, nullable=False)
    fiberassign_y = Column(REAL, nullable=False)
    priority = Column(Integer, nullable=False)
    plate_ra = Column(DOUBLE_PRECISION, nullable=False)
    plate_dec = Column(DOUBLE_PRECISION, nullable=False)

    tile = relationship("Tile", back_populates="fiberassign")
    # target = relationship("Target", back_populates="fiberassign")

    def __repr__(self):
        return "Fiberassign(tileid={0.tileid:d}, fiber={0.fiber:d})".format(self)


class Potential(SchemaMixin, Base):
    """Representation of the POTENTIAL_ASSIGNMENTS table in a fiberassign file.
    """

    tileid = Column(Integer, ForeignKey('tile.tileid'), primary_key=True, index=True)
    targetid = Column(BigInteger, primary_key=True, index=True)  # potential ForeignKey on Target
    fiber = Column(Integer, nullable=False)
    location = Column(Integer, primary_key=True)

    tile = relationship("Tile", back_populates="potential")
    # target = relationship("Target", back_populates="potenial")

    def __repr__(self):
        return "Potential(tileid={0.tileid:d}, targetid={0.targetid:d}, location={0.location:d})".format(self)


class Zpix(SchemaMixin, Base):
    """Representation of the ``ZCATALOG`` table in zpix files.
    """

    targetid = Column(BigInteger, primary_key=True, autoincrement=False)  # potential ForeignKey on Target
    survey = Column(String(7), primary_key=True, autoincrement=False)
    program = Column(String(6), primary_key=True, autoincrement=False)
    spgrp = Column(String(10), nullable=False)
    spgrpval = Column(Integer, nullable=False)
    healpix = Column(Integer, nullable=False)
    z = Column(DOUBLE_PRECISION, index=True, nullable=False)
    zerr = Column(DOUBLE_PRECISION, nullable=False)
    zwarn = Column(BigInteger, index=True, nullable=False)
    chi2 = Column(DOUBLE_PRECISION, nullable=False)
    coeff_0 = Column(DOUBLE_PRECISION, nullable=False)
    coeff_1 = Column(DOUBLE_PRECISION, nullable=False)
    coeff_2 = Column(DOUBLE_PRECISION, nullable=False)
    coeff_3 = Column(DOUBLE_PRECISION, nullable=False)
    coeff_4 = Column(DOUBLE_PRECISION, nullable=False)
    coeff_5 = Column(DOUBLE_PRECISION, nullable=False)
    coeff_6 = Column(DOUBLE_PRECISION, nullable=False)
    coeff_7 = Column(DOUBLE_PRECISION, nullable=False)
    coeff_8 = Column(DOUBLE_PRECISION, nullable=False)
    coeff_9 = Column(DOUBLE_PRECISION, nullable=False)
    npixels = Column(BigInteger, nullable=False)
    spectype = Column(String(6), index=True, nullable=False)
    subtype = Column(String(20), index=True, nullable=False)
    ncoeff = Column(BigInteger, nullable=False)
    deltachi2 = Column(DOUBLE_PRECISION, nullable=False)
    coadd_fiberstatus = Column(Integer, nullable=False)
    #
    # Skipping columns that are in other tables.
    #
    coadd_numexp = Column(SmallInteger, nullable=False)
    coadd_exptime = Column(REAL, nullable=False)
    coadd_numnight = Column(SmallInteger, nullable=False)
    coadd_numtile = Column(SmallInteger, nullable=False)
    mean_delta_x = Column(REAL, nullable=False)
    rms_delta_x = Column(REAL, nullable=False)
    mean_delta_y = Column(REAL, nullable=False)
    rms_delta_y = Column(REAL, nullable=False)
    mean_fiber_ra = Column(DOUBLE_PRECISION, nullable=False)
    std_fiber_ra = Column(REAL, nullable=False)
    mean_fiber_dec = Column(DOUBLE_PRECISION, nullable=False)
    std_fiber_dec = Column(REAL, nullable=False)
    mean_psf_to_fiber_specflux = Column(REAL, nullable=False)
    tsnr2_gpbdark_b = Column(REAL, nullable=False)
    tsnr2_elg_b = Column(REAL, nullable=False)
    tsnr2_gpbbright_b = Column(REAL, nullable=False)
    tsnr2_lya_b = Column(REAL, nullable=False)
    tsnr2_bgs_b = Column(REAL, nullable=False)
    tsnr2_gpbbackup_b = Column(REAL, nullable=False)
    tsnr2_qso_b = Column(REAL, nullable=False)
    tsnr2_lrg_b = Column(REAL, nullable=False)
    tsnr2_gpbdark_r = Column(REAL, nullable=False)
    tsnr2_elg_r = Column(REAL, nullable=False)
    tsnr2_gpbbright_r = Column(REAL, nullable=False)
    tsnr2_lya_r = Column(REAL, nullable=False)
    tsnr2_bgs_r = Column(REAL, nullable=False)
    tsnr2_gpbbackup_r = Column(REAL, nullable=False)
    tsnr2_qso_r = Column(REAL, nullable=False)
    tsnr2_lrg_r = Column(REAL, nullable=False)
    tsnr2_gpbdark_z = Column(REAL, nullable=False)
    tsnr2_elg_z = Column(REAL, nullable=False)
    tsnr2_gpbbright_z = Column(REAL, nullable=False)
    tsnr2_lya_z = Column(REAL, nullable=False)
    tsnr2_bgs_z = Column(REAL, nullable=False)
    tsnr2_gpbbackup_z = Column(REAL, nullable=False)
    tsnr2_qso_z = Column(REAL, nullable=False)
    tsnr2_lrg_z = Column(REAL, nullable=False)
    tsnr2_gpbdark = Column(REAL, nullable=False)
    tsnr2_elg = Column(REAL, nullable=False)
    tsnr2_gpbbright = Column(REAL, nullable=False)
    tsnr2_lya = Column(REAL, nullable=False)
    tsnr2_bgs = Column(REAL, nullable=False)
    tsnr2_gpbbackup = Column(REAL, nullable=False)
    tsnr2_qso = Column(REAL, nullable=False)
    tsnr2_lrg = Column(REAL, nullable=False)
    zcat_nspec = Column(SmallInteger, nullable=False)
    zcat_primary = Column(Boolean, nullable=False)

    # target = relationship("Target", back_populates="zpix_redshifts")

    def __repr__(self):
        return "Zpix(targetid={0.targetid:d}, survey='{0.survey}', program='{0.program}')".format(self)


class Ztile(SchemaMixin, Base):
    """Representation of the ``ZCATALOG`` table in ztile files.
    """

    targetid = Column(BigInteger, primary_key=True, autoincrement=False)  # potential ForeignKey on Target
    survey = Column(String(7), nullable=False)
    program = Column(String(6), nullable=False)
    spgrp = Column(String, primary_key=True, autoincrement=False)
    spgrpval = Column(Integer, primary_key=True, autoincrement=False)
    z = Column(DOUBLE_PRECISION, index=True, nullable=False)
    zerr = Column(DOUBLE_PRECISION, nullable=False)
    zwarn = Column(BigInteger, index=True, nullable=False)
    chi2 = Column(DOUBLE_PRECISION, nullable=False)
    coeff_0 = Column(DOUBLE_PRECISION, nullable=False)
    coeff_1 = Column(DOUBLE_PRECISION, nullable=False)
    coeff_2 = Column(DOUBLE_PRECISION, nullable=False)
    coeff_3 = Column(DOUBLE_PRECISION, nullable=False)
    coeff_4 = Column(DOUBLE_PRECISION, nullable=False)
    coeff_5 = Column(DOUBLE_PRECISION, nullable=False)
    coeff_6 = Column(DOUBLE_PRECISION, nullable=False)
    coeff_7 = Column(DOUBLE_PRECISION, nullable=False)
    coeff_8 = Column(DOUBLE_PRECISION, nullable=False)
    coeff_9 = Column(DOUBLE_PRECISION, nullable=False)
    npixels = Column(BigInteger, nullable=False)
    spectype = Column(String(6), index=True, nullable=False)
    subtype = Column(String(20), index=True, nullable=False)
    ncoeff = Column(BigInteger, nullable=False)
    deltachi2 = Column(DOUBLE_PRECISION, nullable=False)
    coadd_fiberstatus = Column(Integer, nullable=False)
    #
    # Skipping columns that are in other tables.
    #
    tileid = Column(Integer, ForeignKey("tile.tileid"), primary_key=True, autoincrement=False)
    coadd_numexp = Column(SmallInteger, nullable=False)
    coadd_exptime = Column(REAL, nullable=False)
    coadd_numnight = Column(SmallInteger, nullable=False)
    coadd_numtile = Column(SmallInteger, nullable=False)
    mean_delta_x = Column(REAL, nullable=False)
    rms_delta_x = Column(REAL, nullable=False)
    mean_delta_y = Column(REAL, nullable=False)
    rms_delta_y = Column(REAL, nullable=False)
    mean_fiber_ra = Column(DOUBLE_PRECISION, nullable=False)
    std_fiber_ra = Column(REAL, nullable=False)
    mean_fiber_dec = Column(DOUBLE_PRECISION, nullable=False)
    std_fiber_dec = Column(REAL, nullable=False)
    mean_psf_to_fiber_specflux = Column(REAL, nullable=False)
    tsnr2_gpbdark_b = Column(REAL, nullable=False)
    tsnr2_elg_b = Column(REAL, nullable=False)
    tsnr2_gpbbright_b = Column(REAL, nullable=False)
    tsnr2_lya_b = Column(REAL, nullable=False)
    tsnr2_bgs_b = Column(REAL, nullable=False)
    tsnr2_gpbbackup_b = Column(REAL, nullable=False)
    tsnr2_qso_b = Column(REAL, nullable=False)
    tsnr2_lrg_b = Column(REAL, nullable=False)
    tsnr2_gpbdark_r = Column(REAL, nullable=False)
    tsnr2_elg_r = Column(REAL, nullable=False)
    tsnr2_gpbbright_r = Column(REAL, nullable=False)
    tsnr2_lya_r = Column(REAL, nullable=False)
    tsnr2_bgs_r = Column(REAL, nullable=False)
    tsnr2_gpbbackup_r = Column(REAL, nullable=False)
    tsnr2_qso_r = Column(REAL, nullable=False)
    tsnr2_lrg_r = Column(REAL, nullable=False)
    tsnr2_gpbdark_z = Column(REAL, nullable=False)
    tsnr2_elg_z = Column(REAL, nullable=False)
    tsnr2_gpbbright_z = Column(REAL, nullable=False)
    tsnr2_lya_z = Column(REAL, nullable=False)
    tsnr2_bgs_z = Column(REAL, nullable=False)
    tsnr2_gpbbackup_z = Column(REAL, nullable=False)
    tsnr2_qso_z = Column(REAL, nullable=False)
    tsnr2_lrg_z = Column(REAL, nullable=False)
    tsnr2_gpbdark = Column(REAL, nullable=False)
    tsnr2_elg = Column(REAL, nullable=False)
    tsnr2_gpbbright = Column(REAL, nullable=False)
    tsnr2_lya = Column(REAL, nullable=False)
    tsnr2_bgs = Column(REAL, nullable=False)
    tsnr2_gpbbackup = Column(REAL, nullable=False)
    tsnr2_qso = Column(REAL, nullable=False)
    tsnr2_lrg = Column(REAL, nullable=False)
    zcat_nspec = Column(SmallInteger, nullable=False)
    zcat_primary = Column(Boolean, nullable=False)

    tile = relationship("Tile", back_populates="ztile_redshifts")
    # target = relationship("Target", back_populates="ztile_redshifts")

    def __repr__(self):
        return "Ztile(targetid={0.targetid:d}, tileid={0.tileid:d}, spgrp='{0.spgrp}', spgrpval={0.spgrpval:d})".format(self)


def _frameid(data):
    """Update the ``frameid`` column.

    Parameters
    ----------
    data : :class:`astropy.table.Table`
        The initial data read from the file.

    Returns
    -------
    :class:`astropy.table.Table`
        Updated data table.
    """
    frameid = 100*data['EXPID'] + np.array([cameraid(c) for c in data['CAMERA']], dtype=data['EXPID'].dtype)
    data.add_column(frameid, name='FRAMEID', index=0)
    return data


def _tileid(data):
    """Update the ``tileid`` column.  Also check for the presence of ``PLATE_RA``, ``PLATE_DEC``.

    Parameters
    ----------
    data : :class:`astropy.table.Table`
        The initial data read from the file.

    Returns
    -------
    :class:`astropy.table.Table`
        Updated data table.
    """
    try:
        tileid = data.meta['TILEID']*np.ones(len(data), dtype=np.int32)
    except KeyError:
        log.error("Could not find TILEID in metadata!")
        raise
    data.add_column(tileid, name='TILEID', index=0)
    if 'TARGET_RA' in data.colnames and 'PLATE_RA' not in data.colnames:
        log.debug("Adding PLATE_RA, PLATE_DEC.")
        data['PLATE_RA'] = data['TARGET_RA']
        data['PLATE_DEC'] = data['TARGET_DEC']
    return data


def _survey_program(data):
    """Add ``SURVEY``, ``PROGRAM``, ``SPGRP`` columns to zpix and ztile tables.

    Parameters
    ----------
    data : :class:`astropy.table.Table`
        The initial data read from the file.

    Returns
    -------
    :class:`astropy.table.Table`
        Updated data table.

    Raises
    ------
    KeyError
        If a necessary header could not be found.
    """
    for i, key in enumerate(('SURVEY', 'PROGRAM', 'SPGRP')):
        try:
            val = data.meta[key]
        except KeyError:
            log.error("Could not find %s in metadata!", key)
            raise
        log.debug("Adding %s column.", key)
        data.add_column(np.array([val]*len(data)), name=key, index=i+1)
    return data


def load_file(filepaths, tcls, hdu=1, preload=None, expand=None, insert=None, convert=None,
              index=None, rowfilter=None, q3c=None,
              chunksize=50000, maxrows=0):
    """Load data file into the database, assuming that column names map
    to database column names with no surprises.

    Parameters
    ----------
    filepaths : :class:`str` or :class:`list`
        Full path to the data file or set of data files.
    tcls : :class:`sqlalchemy.ext.declarative.api.DeclarativeMeta`
        The table to load, represented by its class.
    hdu : :class:`int` or :class:`str`, optional
        Read a data table from this HDU (default 1).
    preload : callable, optional
        A function that takes a :class:`~astropy.table.Table` as an argument.
        Use this for more complicated manipulation of the data before loading,
        for example a function that depends on multiple columns. The return
        value should be the updated Table.
    expand : :class:`dict`, optional
        If set, map FITS column names to one or more alternative column names.
    insert : :class:`dict`, optional
        If set, insert one or more columns, before an existing column. The
        existing column will be copied into the new column(s).
    convert : :class:`dict`, optional
        If set, convert the data for a named (database) column using the
        supplied function.
    index : :class:`str`, optional
        If set, add a column that just counts the number of rows.
    rowfilter : callable, optional
        If set, apply this filter to the rows to be loaded.  The function
        should return :class:`bool`, with ``True`` meaning a good row.
    q3c : :class:`str`, optional
        If set, create q3c index on the table, using the RA column
        named `q3c`.
    chunksize : :class:`int`, optional
        If set, load database `chunksize` rows at a time (default 50000).
    maxrows : :class:`int`, optional
        If set, stop loading after `maxrows` are loaded.  Alteratively,
        set `maxrows` to zero (0) to load all rows.

    Returns
    -------
    :class:`int`
        The grand total of rows loaded.
    """
    tn = tcls.__tablename__
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    log.info("Identified %d files for ingestion.", len(filepaths))
    loaded_rows = 0
    for filepath in filepaths:
        if filepath.endswith('.fits') or filepath.endswith('.fits.gz'):
            data = Table.read(filepath, hdu=hdu, format='fits')
            log.info("Read %d rows of data from %s HDU %s.", len(data), filepath, hdu)
        elif filepath.endswith('.ecsv'):
            data = Table.read(filepath, format='ascii.ecsv')
            log.info("Read %d rows of data from %s.", len(data), filepath)
        elif filepath.endswith('.csv'):
            data = Table.read(filepath, format='ascii.csv')
            log.info("Read %d rows of data from %s.", len(data), filepath)
        else:
            log.error("Unrecognized data file, %s!", filepath)
            return
        if maxrows == 0:
            mr = len(data)
        else:
            mr = maxrows
        if preload is not None:
            data = preload(data)
            log.info("Preload function complete on %s.", tn)
        try:
            colnames = data.names
        except AttributeError:
            colnames = data.colnames
        masked = dict()
        for col in colnames:
            if data[col].dtype.kind == 'f':
                if isinstance(data[col], MaskedColumn):
                    bad = np.isnan(data[col].data.data[0:mr])
                    masked[col] = True
                else:
                    bad = np.isnan(data[col][0:mr])
                if np.any(bad):
                    if bad.ndim == 1:
                        log.warning("%d rows of bad data detected in column " +
                                    "%s of %s.", bad.sum(), col, filepath)
                    elif bad.ndim == 2:
                        nbadrows = len(bad.sum(1).nonzero()[0])
                        nbaditems = bad.sum(1).sum()
                        log.warning("%d rows (%d items) of bad data detected in column " +
                                    "%s of %s.", nbadrows, nbaditems, col, filepath)
                    else:
                        log.warning("Bad data detected in high-dimensional column %s of %s.", col, filepath)
                    #
                    # TODO: is this replacement appropriate for all columns?
                    #
                    if col in masked:
                        data[col].data.data[0:mr][bad] = -9999.0
                    else:
                        data[col][0:mr][bad] = -9999.0
        log.info("Integrity check complete on %s.", tn)
        if rowfilter is None:
            good_rows = np.ones((mr,), dtype=np.bool)
        else:
            good_rows = rowfilter(data[0:mr])
        log.info("Row filter applied on %s; %d rows remain.", tn, good_rows.sum())
        data_list = list()
        for col in colnames:
            if col in masked:
                data_list.append(data[col].data.data[0:mr][good_rows].tolist())
            else:
                data_list.append(data[col][0:mr][good_rows].tolist())
        data_names = [col.lower() for col in colnames]
        finalrows = len(data_list[0])
        log.info("Initial column conversion complete on %s.", tn)
        if expand is not None:
            for col in expand:
                i = data_names.index(col.lower())
                if isinstance(expand[col], str):
                    #
                    # Just rename a column.
                    #
                    log.debug("Renaming column %s (at index %d) to %s.", data_names[i], i, expand[col])
                    data_names[i] = expand[col]
                else:
                    #
                    # Assume this is an expansion of an array-valued column
                    # into individual columns.
                    #
                    del data_names[i]
                    del data_list[i]
                    for j, n in enumerate(expand[col]):
                        log.debug("Expanding column %d of %s (at index %d) to %s.", j, col, i, n)
                        data_names.insert(i + j, n)
                        data_list.insert(i + j, data[col][:, j].tolist())
                    log.debug(data_names)
            log.info("Column expansion complete on %s.", tn)
        del data
        if insert is not None:
            for col in insert:
                i = data_names.index(col)
                for item in insert[col]:
                    data_names.insert(i, item)
                    data_list.insert(i, data_list[i].copy())  # Dummy values
            log.info("Column insertion complete on %s.", tn)
        if convert is not None:
            for col in convert:
                i = data_names.index(col)
                data_list[i] = [convert[col](x) for x in data_list[i]]
            log.info("Column conversion complete on %s.", tn)
        if index is not None:
            data_list.insert(0, list(range(1, finalrows+1)))
            data_names.insert(0, index)
            log.info("Added index column '%s'.", index)
        data_rows = list(zip(*data_list))
        del data_list
        log.info("Converted columns into rows on %s.", tn)
        for k in range(finalrows//chunksize + 1):
            data_chunk = [dict(zip(data_names, row))
                          for row in data_rows[k*chunksize:(k+1)*chunksize]]
            if len(data_chunk) > 0:
                loaded_rows += len(data_chunk)
                engine.execute(tcls.__table__.insert(), data_chunk)
                log.info("Inserted %d rows in %s.",
                         min((k+1)*chunksize, finalrows), tn)
            else:
                log.error("Detected empty data chunk in %s!", tn)
        # for k in range(finalrows//chunksize + 1):
        #     data_insert = [dict([(col, data_list[i].pop(0))
        #                          for i, col in enumerate(data_names)])
        #                    for j in range(chunksize)]
        #     session.bulk_insert_mappings(tcls, data_insert)
        #     log.info("Inserted %d rows in %s..",
        #              min((k+1)*chunksize, finalrows), tn)
        dbSession.commit()
    if q3c is not None:
        q3c_index(tn, ra=q3c)
    return loaded_rows


def update_truth(filepath, hdu=2, chunksize=50000, skip=('SLOPES', 'EMLINES')):
    """Add data from columns in other HDUs of the Truth table.

    Parameters
    ----------
    filepath : :class:`str`
        Full path to the data file.
    hdu : :class:`int` or :class:`str`, optional
        Read a data table from this HDU (default 2).
    chunksize : :class:`int`, optional
        If set, update database `chunksize` rows at a time (default 50000).
    skip : :func:`tuple`, optional
        Do not load columns with these names (default, ``('SLOPES', 'EMLINES')``)
    """
    tcls = Truth
    tn = tcls.__tablename__
    t = tcls.__table__
    if filepath.endswith( ('.fits', '.fits.gz') ):
        with fits.open(filepath) as hdulist:
            data = hdulist[hdu].data
    elif filepath.endswith('.ecsv'):
        data = Table.read(filepath, format='ascii.ecsv')
    else:
        log.error("Unrecognized data file, %s!", filepath)
        return
    log.info("Read data from %s HDU %s", filepath, hdu)
    try:
        colnames = data.names
    except AttributeError:
        colnames = data.colnames
    for col in colnames:
        if data[col].dtype.kind == 'f':
            bad = np.isnan(data[col])
            if np.any(bad):
                nbad = bad.sum()
                log.warning("%d rows of bad data detected in column " +
                            "%s of %s.", nbad, col, filepath)
    log.info("Integrity check complete on %s.", tn)
    # if rowfilter is None:
    #     good_rows = np.ones((maxrows,), dtype=bool)
    # else:
    #     good_rows = rowfilter(data[0:maxrows])
    # data_list = [data[col][0:maxrows][good_rows].tolist() for col in colnames]
    data_list = [data[col].tolist() for col in colnames if col not in skip]
    data_names = [col.lower() for col in colnames if col not in skip]
    data_names[0] = 'b_targetid'
    finalrows = len(data_list[0])
    log.info("Initial column conversion complete on %s.", tn)
    del data
    data_rows = list(zip(*data_list))
    del data_list
    log.info("Converted columns into rows on %s.", tn)
    for k in range(finalrows//chunksize + 1):
        data_chunk = [dict(zip(data_names, row))
                      for row in data_rows[k*chunksize:(k+1)*chunksize]]
        q = t.update().where(t.c.targetid == bindparam('b_targetid'))
        if len(data_chunk) > 0:
            engine.execute(q, data_chunk)
            log.info("Updated %d rows in %s.",
                     min((k+1)*chunksize, finalrows), tn)


def load_redrock(datapath=None, hdu='REDSHIFTS', q3c=False):
    """Load redrock files into the zcat table.

    This function is deprecated since there should now be a single
    redshift catalog file.

    Parameters
    ----------
    datapath : :class:`str`
        Full path to the directory containing redrock files.
    hdu : :class:`int` or :class:`str`, optional
        Read a data table from this HDU (default 'REDSHIFTS').
    q3c : :class:`bool`, optional
        If set, create q3c index on the table.
    """
    if datapath is None:
        datapath = specprod_root()
    redrockpath = os.path.join(datapath, 'spectra-64', '*', '*', 'redrock-64-*.fits')
    log.info("Using redrock file search path: %s.", redrockpath)
    redrock_files = glob.glob(redrockpath)
    if len(redrock_files) == 0:
        log.error("No redrock files found!")
        return
    log.info("Found %d redrock files.", len(redrock_files))
    #
    # Read the identified redrock files.
    #
    for f in redrock_files:
        brickname = os.path.basename(os.path.dirname(f))
        with fits.open(f) as hdulist:
            data = hdulist[hdu].data
        log.info("Read data from %s HDU %s.", f, hdu)
        good_targetids = ((data['TARGETID'] != 0) & (data['TARGETID'] != -1))
        #
        # If there are too many targetids, the in_ clause will blow up.
        # Disabling this test, and crossing fingers.
        #
        # q = dbSession.query(ZCat).filter(ZCat.targetid.in_(data['TARGETID'].tolist())).all()
        # if len(q) != 0:
        #     log.warning("Duplicate TARGETID found in %s.", f)
        #     for z in q:
        #         log.warning("Duplicate TARGETID = %d.", z.targetid)
        #         good_targetids = good_targetids & (data['TARGETID'] != z.targetid)
        data_list = [data[col][good_targetids].tolist()
                     for col in data.names]
        data_names = [col.lower() for col in data.names]
        log.info("Initial column conversion complete on brick = %s.", brickname)
        #
        # Expand COEFF
        #
        col = 'COEFF'
        expand = ('coeff_0', 'coeff_1', 'coeff_2', 'coeff_3', 'coeff_4',
                  'coeff_5', 'coeff_6', 'coeff_7', 'coeff_8', 'coeff_9',)
        i = data_names.index(col.lower())
        del data_names[i]
        del data_list[i]
        for j, n in enumerate(expand):
            log.debug("Expanding column %d of %s (at index %d) to %s.", j, col, i, n)
            data_names.insert(i + j, n)
            data_list.insert(i + j, data[col][:, j].tolist())
        log.debug(data_names)
        #
        # redrock files don't contain the same columns as zcatalog.
        #
        for col in ZCat.__table__.columns:
            if col.name not in data_names:
                data_names.append(col.name)
                data_list.append([0]*len(data_list[0]))
        data_rows = list(zip(*data_list))
        log.info("Converted columns into rows on brick = %s.", brickname)
        try:
            dbSession.bulk_insert_mappings(ZCat, [dict(zip(data_names, row))
                                                  for row in data_rows])
        except IntegrityError as e:
            log.error("Integrity Error detected!")
            log.error(e)
            dbSession.rollback()
        else:
            log.info("Inserted %d rows in %s for brick = %s.",
                     len(data_rows), ZCat.__tablename__, brickname)
            dbSession.commit()
    if q3c:
        q3c_index('zcat')
    return


def q3c_index(table, ra='ra'):
    """Create a q3c index on a table.

    Parameters
    ----------
    table : :class:`str`
        Name of the table to index.
    ra : :class:`str`, optional
        If the RA, Dec columns are called something besides "ra" and "dec",
        set its name.  For example, ``ra='target_ra'``.
    """
    q3c_sql = """CREATE INDEX ix_{table}_q3c_ang2ipix ON {schema}.{table} (q3c_ang2ipix({ra}, {dec}));
    CLUSTER {schema}.{table} USING ix_{table}_q3c_ang2ipix;
    ANALYZE {schema}.{table};
    """.format(ra=ra, dec=ra.lower().replace('ra', 'dec'),
               schema=schemaname, table=table)
    log.info("Creating q3c index on %s.%s.", schemaname, table)
    dbSession.execute(q3c_sql)
    log.info("Finished q3c index on %s.%s.", schemaname, table)
    dbSession.commit()
    return


def setup_db(options=None, **kwargs):
    """Initialize the database connection.

    Parameters
    ----------
    options : :class:`argpare.Namespace`
        Parsed command-line options.
    kwargs : keywords
        If present, use these instead of `options`.  This is more
        user-friendly than setting up a :class:`~argpare.Namespace`
        object in, *e.g.* a Jupyter Notebook.

    Returns
    -------
    :class:`bool`
        ``True`` if the configured database is a PostgreSQL database.
    """
    global engine, schemaname
    #
    # Schema creation
    #
    if options is None:
        if len(kwargs) > 0:
            try:
                schema = kwargs['schema']
            except KeyError:
                schema = None
            try:
                overwrite = kwargs['overwrite']
            except KeyError:
                overwrite = False
            try:
                hostname = kwargs['hostname']
            except KeyError:
                hostname = None
            try:
                username = kwargs['username']
            except KeyError:
                username = 'desidev_admin'
            try:
                dbfile = kwargs['dbfile']
            except KeyError:
                dbfile = 'redshift.db'
            try:
                datapath = kwargs['datapath']
            except KeyError:
                datapath = None
            try:
                verbose = kwargs['verbose']
            except KeyError:
                verbose = False
        else:
            raise ValueError("No options specified!")
    else:
        schema = options.schema
        overwrite = options.overwrite
        hostname = options.hostname
        username = options.username
        dbfile = options.dbfile
        datapath = options.datapath
        verbose = options.verbose
    if schema:
        schemaname = schema
        # event.listen(Base.metadata, 'before_create', CreateSchema(schemaname))
        # if overwrite:
        #     event.listen(Base.metadata, 'before_create',
        #                  DDL('DROP SCHEMA IF EXISTS {0} CASCADE'.format(schemaname)))
        event.listen(Base.metadata, 'before_create',
                     DDL('CREATE SCHEMA IF NOT EXISTS {0}'.format(schemaname)))
    #
    # Create the file.
    #
    postgresql = False
    if hostname:
        postgresql = True
        db_connection = parse_pgpass(hostname=hostname,
                                     username=username)
        if db_connection is None:
            log.critical("Could not load database information!")
            return 1
    else:
        if os.path.basename(dbfile) == dbfile:
            db_file = os.path.join(datapath, dbfile)
        else:
            db_file = dbfile
        if overwrite and os.path.exists(db_file):
            log.info("Removing file: %s.", db_file)
            os.remove(db_file)
        db_connection = 'sqlite:///'+db_file
    #
    # SQLAlchemy stuff.
    #
    engine = create_engine(db_connection, echo=verbose)
    dbSession.remove()
    dbSession.configure(bind=engine, autoflush=False, expire_on_commit=False)
    log.info("Begin creating tables.")
    for tab in Base.metadata.tables.values():
        tab.schema = schemaname
    if overwrite:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    log.info("Finished creating tables.")
    return postgresql


def get_options(*args):
    """Parse command-line options.

    Parameters
    ----------
    args : iterable
        If arguments are passed, use them instead of ``sys.argv``.

    Returns
    -------
    :class:`argparse.Namespace`
        The parsed options.
    """
    from sys import argv
    from argparse import ArgumentParser
    prsr = ArgumentParser(description=("Load a data challenge simulation into a " +
                                       "database."),
                          prog=os.path.basename(argv[0]))
    prsr.add_argument('-f', '--filename', action='store', dest='dbfile',
                      default='redshift.db', metavar='FILE',
                      help="Store data in FILE (default %(default)s).")
    prsr.add_argument('-H', '--hostname', action='store', dest='hostname',
                      metavar='HOSTNAME', default='nerscdb03.nersc.gov',
                      help='If specified, connect to a PostgreSQL database on HOSTNAME (default %(default)s).')
    prsr.add_argument('-m', '--max-rows', action='store', dest='maxrows',
                      type=int, default=0, metavar='M',
                      help="Load up to M rows in the tables (default is all rows).")
    prsr.add_argument('-o', '--overwrite', action='store_true', dest='overwrite',
                      help='Delete any existing file(s) before loading.')
    prsr.add_argument('-r', '--rows', action='store', dest='chunksize',
                      type=int, default=50000, metavar='N',
                      help="Load N rows at a time (default %(default)s).")
    prsr.add_argument('-s', '--schema', action='store', dest='schema',
                      metavar='SCHEMA',
                      help='Set the schema name in the PostgreSQL database.')
    prsr.add_argument('-t', '--tiles-path', action='store', dest='tilespath', metavar='PATH',
                      default=os.path.join(os.environ['DESI_TARGET'], 'fiberassign', 'tiles', 'trunk'),
                      help="Load fiberassign data from PATH (default %(default)s).")
    prsr.add_argument('-U', '--username', action='store', dest='username',
                      metavar='USERNAME', default='desidev_admin',
                      help="If specified, connect to a PostgreSQL database with USERNAME (default %(default)s).")
    prsr.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                      help='Print extra information.')
    prsr.add_argument('-z', '--redrock', action='store_true', dest='redrock',
                      help='Force loading of the zcat table from redrock files.')
    prsr.add_argument('datapath', metavar='DIR', help='Load the data in DIR.')
    if len(args) > 0:
        options = prsr.parse_args(args)
    else:
        options = prsr.parse_args()
    return options


def main():
    """Entry point for command-line script.

    Returns
    -------
    :class:`int`
        An integer suitable for passing to :func:`sys.exit`.
    """
    global log
    freeze_iers()
    #
    # command-line arguments
    #
    options = get_options()
    #
    # Logging
    #
    if options.verbose:
        log = get_logger(DEBUG, timestamp=True)
    else:
        log = get_logger(INFO, timestamp=True)
    #
    # Initialize DB
    #
    postgresql = setup_db(options)
    #
    # Load configuration
    #
    loader = [{'filepaths': [os.path.join('/global/cscratch1/sd/ioannis/photocatalog', os.environ['SPECPROD'], 'targetphot-{specprod}.fits'.format(specprod=os.environ['SPECPROD'])),],
                             # os.path.join('/global/cscratch1/sd/ioannis/photocatalog', os.environ['SPECPROD'], 'targetphot-potential-targets-{specprod}.fits'.format(specprod=os.environ['SPECPROD'])),
                             # os.path.join('/global/cscratch1/sd/ioannis/photocatalog', os.environ['SPECPROD'], 'targetphot-missing-{specprod}.fits'.format(specprod=os.environ['SPECPROD']))],
               'tcls': Target,
               'hdu': 'TARGETPHOT',
               'expand': {'DCHISQ': ('dchisq_psf', 'dchisq_rex', 'dchisq_dev', 'dchisq_exp', 'dchisq_ser',)},
               'q3c': 'ra',
               'chunksize': options.chunksize,
               'maxrows': options.maxrows
               },
              {'filepaths': os.path.join(os.environ['DESI_SPECTRO_REDUX'], os.environ['SPECPROD'], 'tiles-{specprod}.fits'.format(specprod=os.environ['SPECPROD'])),
               'tcls': Tile,
               'hdu': 'TILE_COMPLETENESS',
               'q3c': 'tilera',
               'chunksize': options.chunksize,
               'maxrows': options.maxrows
               },
              {'filepaths': os.path.join(os.environ['DESI_SPECTRO_REDUX'], os.environ['SPECPROD'], 'exposures-{specprod}.fits'.format(specprod=os.environ['SPECPROD'])),
               'tcls': Exposure,
               'hdu': 'EXPOSURES',
               'insert': {'mjd': ('date_obs',)},
               'convert': {'date_obs': lambda x: Time(x, format='mjd').to_value('datetime').replace(tzinfo=utc)},
               'q3c': 'tilera',
               'chunksize': options.chunksize,
               'maxrows': options.maxrows
               },
              {'filepaths': os.path.join(os.environ['DESI_SPECTRO_REDUX'], os.environ['SPECPROD'], 'exposures-{specprod}.fits'.format(specprod=os.environ['SPECPROD'])),
               'tcls': Frame,
               'hdu': 'FRAMES',
               'preload': _frameid,
               'chunksize': options.chunksize,
               'maxrows': options.maxrows
              },
              {'filepaths': glob.glob(os.path.join(os.environ['DESI_SPECTRO_REDUX'], os.environ['SPECPROD'], 'zcatalog', 'zpix-*.fits')),
               'tcls': Zpix,
               'hdu': 'ZCATALOG',
               'preload': _survey_program,
               'expand': {'COEFF': ('coeff_0', 'coeff_1', 'coeff_2', 'coeff_3', 'coeff_4',
                                    'coeff_5', 'coeff_6', 'coeff_7', 'coeff_8', 'coeff_9',)},
               'rowfilter': lambda x: x['TARGETID'] > 0,
               'chunksize': options.chunksize,
               'maxrows': options.maxrows
               },
              {'filepaths': glob.glob(os.path.join(os.environ['DESI_SPECTRO_REDUX'], os.environ['SPECPROD'], 'zcatalog', 'ztile-*.fits')),
               'tcls': Ztile,
               'hdu': 'ZCATALOG',
               'preload': _survey_program,
               'expand': {'COEFF': ('coeff_0', 'coeff_1', 'coeff_2', 'coeff_3', 'coeff_4',
                                    'coeff_5', 'coeff_6', 'coeff_7', 'coeff_8', 'coeff_9',)},
               'rowfilter': lambda x: x['TARGETID'] > 0,
               'chunksize': options.chunksize,
               'maxrows': options.maxrows
               }]

    #
    # Load the tables that correspond to a small set of files.
    #
    for l in loader:
        tn = l['tcls'].__tablename__
        #
        # Don't use .one().  It actually fetches *all* rows.
        #
        q = dbSession.query(l['tcls']).first()
        if q is None:
            log.info("Loading %s from %s.", tn, str(l['filepaths']))
            load_file(**l)
            log.info("Finished loading %s.", tn)
        else:
            log.info("%s table already loaded.", tn.title())
    #
    # Find the tiles that need to be loaded. Not all fiberassign files are compressed!
    #
    try:
        fiberassign_files = [checkgzip(os.path.join(options.tilespath, (f"{tileid[0]:06d}")[0:3], f"fiberassign-{tileid[0]:06d}.fits"))
                             for tileid in dbSession.query(Tile.tileid).order_by(Tile.tileid)]
    except FileNotFoundError:
        log.error("Some fiberassign files were not found!")
        return 1
    log.debug(fiberassign_files)
    loader = [{'filepaths': fiberassign_files,
               'tcls': Fiberassign,
               'hdu': 'FIBERASSIGN',
               'preload': _tileid,
               'rowfilter': lambda x: x['TARGETID'] > 0,
               'q3c': 'target_ra',
               'chunksize': options.chunksize,
               'maxrows': options.maxrows
              },
              {'filepaths': fiberassign_files,
               'tcls': Potential,
               'hdu': 'POTENTIAL_ASSIGNMENTS',
               'preload': _tileid,
               'rowfilter': lambda x: x['TARGETID'] > 0,
               'chunksize': options.chunksize,
               'maxrows': options.maxrows
              }]
    for l in loader:
        tn = l['tcls'].__tablename__
        #
        # Don't use .one().  It actually fetches *all* rows.
        #
        q = dbSession.query(l['tcls']).first()
        if q is None:
            log.info("Loading %s from %s.", tn, str(l['filepaths']))
            load_file(**l)
            log.info("Finished loading %s.", tn)
        else:
            log.info("%s table already loaded.", tn.title())
    return 0


if __name__ == '__main__':
    sys.exit(main())
