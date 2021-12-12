# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.database.redshift
==========================

Code for loading spectroscopic pipeline results (specifically redshifts)
into a database.
"""
from __future__ import absolute_import, division, print_function
import os
import re
import glob

import numpy as np
from astropy.io import fits
from astropy.table import Table
from pytz import utc

from sqlalchemy import (create_engine, event, ForeignKey, Column, DDL,
                        BigInteger, Boolean, Integer, String, Float, DateTime,
                        bindparam)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import scoped_session, sessionmaker, relationship
from sqlalchemy.schema import CreateSchema

from desiutil.log import log, DEBUG, INFO

from ..io.meta import specprod_root
from .util import convert_dateobs, parse_pgpass

Base = declarative_base()
engine = None
dbSession = scoped_session(sessionmaker())
schemaname = None


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


class Truth(SchemaMixin, Base):
    """Representation of the truth table.
    """

    targetid = Column(BigInteger, primary_key=True, autoincrement=False)
    mockid = Column(BigInteger, nullable=False)
    truez = Column(Float, nullable=False)
    truespectype = Column(String, nullable=False)
    templatetype = Column(String, nullable=False)
    templatesubtype = Column(String, nullable=False)
    templateid = Column(Integer, nullable=False)
    seed = Column(BigInteger, nullable=False)
    mag = Column(Float, nullable=False)
    magfilter = Column(String, nullable=False)
    flux_g = Column(Float, nullable=False)
    flux_r = Column(Float, nullable=False)
    flux_z = Column(Float, nullable=False)
    flux_w1 = Column(Float, nullable=False)
    flux_w2 = Column(Float, nullable=False)
    flux_w3 = Column(Float, nullable=False)
    flux_w4 = Column(Float, nullable=False)
    oiiflux = Column(Float, nullable=False, default=-9999.0)
    hbetaflux = Column(Float, nullable=False, default=-9999.0)
    ewoii = Column(Float, nullable=False, default=-9999.0)
    ewhbeta = Column(Float, nullable=False, default=-9999.0)
    d4000 = Column(Float, nullable=False, default=-9999.0)
    vdisp = Column(Float, nullable=False, default=-9999.0)
    oiidoublet = Column(Float, nullable=False, default=-9999.0)
    oiiihbeta = Column(Float, nullable=False, default=-9999.0)
    oiihbeta = Column(Float, nullable=False, default=-9999.0)
    niihbeta = Column(Float, nullable=False, default=-9999.0)
    siihbeta = Column(Float, nullable=False, default=-9999.0)
    mabs_1450 = Column(Float, nullable=False, default=-9999.0)
    bal_templateid = Column(Integer, nullable=False, default=-1)
    truez_norsd = Column(Float, nullable=False, default=-9999.0)
    teff = Column(Float, nullable=False, default=-9999.0)
    logg = Column(Float, nullable=False, default=-9999.0)
    feh = Column(Float, nullable=False, default=-9999.0)

    def __repr__(self):
        return "<Truth(targetid={0.targetid:d})>".format(self)


class Target(SchemaMixin, Base):
    """Representation of the target table.
    """

    release = Column(Integer, nullable=False)
    brickid = Column(Integer, nullable=False)
    brickname = Column(String, nullable=False)
    brick_objid = Column(Integer, nullable=False)
    morphtype = Column(String, nullable=False)
    ra = Column(Float, nullable=False)
    dec = Column(Float, nullable=False)
    ra_ivar = Column(Float, nullable=False)
    dec_ivar = Column(Float, nullable=False)
    dchisq_psf = Column(Float, nullable=False)
    dchisq_rex = Column(Float, nullable=False)
    dchisq_dev = Column(Float, nullable=False)
    dchisq_exp = Column(Float, nullable=False)
    dchisq_comp = Column(Float, nullable=False)
    flux_g = Column(Float, nullable=False)
    flux_r = Column(Float, nullable=False)
    flux_z = Column(Float, nullable=False)
    flux_w1 = Column(Float, nullable=False)
    flux_w2 = Column(Float, nullable=False)
    flux_w3 = Column(Float, nullable=False)
    flux_w4 = Column(Float, nullable=False)
    flux_ivar_g = Column(Float, nullable=False)
    flux_ivar_r = Column(Float, nullable=False)
    flux_ivar_z = Column(Float, nullable=False)
    flux_ivar_w1 = Column(Float, nullable=False)
    flux_ivar_w2 = Column(Float, nullable=False)
    flux_ivar_w3 = Column(Float, nullable=False)
    flux_ivar_w4 = Column(Float, nullable=False)
    mw_transmission_g = Column(Float, nullable=False)
    mw_transmission_r = Column(Float, nullable=False)
    mw_transmission_z = Column(Float, nullable=False)
    mw_transmission_w1 = Column(Float, nullable=False)
    mw_transmission_w2 = Column(Float, nullable=False)
    mw_transmission_w3 = Column(Float, nullable=False)
    mw_transmission_w4 = Column(Float, nullable=False)
    nobs_g = Column(Integer, nullable=False)
    nobs_r = Column(Integer, nullable=False)
    nobs_z = Column(Integer, nullable=False)
    fracflux_g = Column(Float, nullable=False)
    fracflux_r = Column(Float, nullable=False)
    fracflux_z = Column(Float, nullable=False)
    fracmasked_g = Column(Float, nullable=False)
    fracmasked_r = Column(Float, nullable=False)
    fracmasked_z = Column(Float, nullable=False)
    fracin_g = Column(Float, nullable=False)
    fracin_r = Column(Float, nullable=False)
    fracin_z = Column(Float, nullable=False)
    allmask_g = Column(Float, nullable=False)
    allmask_r = Column(Float, nullable=False)
    allmask_z = Column(Float, nullable=False)
    wisemask_w1 = Column(Integer, nullable=False)
    wisemask_w2 = Column(Integer, nullable=False)
    psfdepth_g = Column(Float, nullable=False)
    psfdepth_r = Column(Float, nullable=False)
    psfdepth_z = Column(Float, nullable=False)
    galdepth_g = Column(Float, nullable=False)
    galdepth_r = Column(Float, nullable=False)
    galdepth_z = Column(Float, nullable=False)
    fracdev = Column(Float, nullable=False)
    fracdev_ivar = Column(Float, nullable=False)
    shapedev_r = Column(Float, nullable=False)
    shapedev_r_ivar = Column(Float, nullable=False)
    shapedev_e1 = Column(Float, nullable=False)
    shapedev_e1_ivar = Column(Float, nullable=False)
    shapedev_e2 = Column(Float, nullable=False)
    shapedev_e2_ivar = Column(Float, nullable=False)
    shapeexp_r = Column(Float, nullable=False)
    shapeexp_r_ivar = Column(Float, nullable=False)
    shapeexp_e1 = Column(Float, nullable=False)
    shapeexp_e1_ivar = Column(Float, nullable=False)
    shapeexp_e2 = Column(Float, nullable=False)
    shapeexp_e2_ivar = Column(Float, nullable=False)
    fiberflux_g = Column(Float, nullable=False)
    fiberflux_r = Column(Float, nullable=False)
    fiberflux_z = Column(Float, nullable=False)
    fibertotflux_g = Column(Float, nullable=False)
    fibertotflux_r = Column(Float, nullable=False)
    fibertotflux_z = Column(Float, nullable=False)
    ref_cat = Column(String, nullable=False)
    ref_id = Column(BigInteger, nullable=False)
    gaia_phot_g_mean_mag = Column(Float, nullable=False)
    gaia_phot_g_mean_flux_over_error = Column(Float, nullable=False)
    gaia_phot_bp_mean_mag = Column(Float, nullable=False)
    gaia_phot_bp_mean_flux_over_error = Column(Float, nullable=False)
    gaia_phot_rp_mean_mag = Column(Float, nullable=False)
    gaia_phot_rp_mean_flux_over_error = Column(Float, nullable=False)
    gaia_phot_bp_rp_excess_factor = Column(Float, nullable=False)
    gaia_astrometric_sigma5d_max = Column(Float, nullable=False)
    gaia_astrometric_params_solved = Column(BigInteger, nullable=False)
    gaia_astrometric_excess_noise = Column(Float, nullable=False)
    gaia_duplicated_source = Column(Boolean, nullable=False)
    parallax = Column(Float, nullable=False)
    parallax_ivar = Column(Float, nullable=False)
    pmra = Column(Float, nullable=False)
    pmra_ivar = Column(Float, nullable=False)
    pmdec = Column(Float, nullable=False)
    pmdec_ivar = Column(Float, nullable=False)
    maskbits = Column(Integer, nullable=False)
    ebv = Column(Float, nullable=False)
    photsys = Column(String, nullable=False)
    targetid = Column(BigInteger, primary_key=True, autoincrement=False)
    desi_target = Column(BigInteger, nullable=False)
    bgs_target = Column(BigInteger, nullable=False)
    mws_target = Column(BigInteger, nullable=False)
    subpriority = Column(Float, nullable=False)
    obsconditions = Column(BigInteger, nullable=False)
    priority_init = Column(BigInteger, nullable=False)
    numobs_init = Column(BigInteger, nullable=False)
    hpxpixel = Column(BigInteger, nullable=False)

    def __repr__(self):
        return "<Target(targetid={0.targetid})>".format(self)


class ObsList(SchemaMixin, Base):
    """Representation of the obslist table.
    """

    expid = Column(Integer, primary_key=True, autoincrement=False)
    tileid = Column(Integer, nullable=False)
    passnum = Column(Integer, nullable=False)
    ra = Column(Float, nullable=True)   #- Calib exposures don't have RA, dec
    dec = Column(Float, nullable=True)
    ebmv = Column(Float, nullable=True)
    night = Column(String, nullable=False)
    mjd = Column(Float, nullable=False)
    exptime = Column(Float, nullable=False)
    seeing = Column(Float, nullable=True)
    transparency = Column(Float, nullable=True)
    airmass = Column(Float, nullable=True)
    moonfrac = Column(Float, nullable=True)
    moonalt = Column(Float, nullable=True)
    moonsep = Column(Float, nullable=True)
    program = Column(String, nullable=False)
    flavor = Column(String, nullable=False)
    # dateobs = Column(DateTime(timezone=True), nullable=False)

    def __repr__(self):
        return "<ObsList(expid={0.expid:d})>".format(self)


class ZCat(SchemaMixin, Base):
    """Representation of the zcat table.
    """

    targetid = Column(BigInteger, primary_key=True, autoincrement=False)
    chi2 = Column(Float, nullable=False)
    coeff_0 = Column(Float, nullable=False)
    coeff_1 = Column(Float, nullable=False)
    coeff_2 = Column(Float, nullable=False)
    coeff_3 = Column(Float, nullable=False)
    coeff_4 = Column(Float, nullable=False)
    coeff_5 = Column(Float, nullable=False)
    coeff_6 = Column(Float, nullable=False)
    coeff_7 = Column(Float, nullable=False)
    coeff_8 = Column(Float, nullable=False)
    coeff_9 = Column(Float, nullable=False)
    z = Column(Float, index=True, nullable=False)
    zerr = Column(Float, nullable=False)
    zwarn = Column(BigInteger, index=True, nullable=False)
    npixels = Column(BigInteger, nullable=False)
    spectype = Column(String, index=True, nullable=False)
    subtype = Column(String, index=True, nullable=False)
    ncoeff = Column(BigInteger, nullable=False)
    deltachi2 = Column(Float, nullable=False)
    brickname = Column(String, index=True, nullable=False)
    numexp = Column(Integer, nullable=False, default=-1)
    numtile = Column(Integer, nullable=False)
    #
    # Columns that are just copied from the target table.
    #
    # brickid = Column(Integer, nullable=False)
    # brick_objid = Column(Integer, nullable=False)
    # ra = Column(Float, nullable=False)
    # dec = Column(Float, nullable=False)
    # flux_g = Column(Float, nullable=False)
    # flux_r = Column(Float, nullable=False)
    # flux_z = Column(Float, nullable=False)
    # flux_w1 = Column(Float, nullable=False)
    # flux_w2 = Column(Float, nullable=False)
    # mw_transmission_g = Column(Float, nullable=False)
    # mw_transmission_r = Column(Float, nullable=False)
    # mw_transmission_z = Column(Float, nullable=False)
    # mw_transmission_w1 = Column(Float, nullable=False)
    # mw_transmission_w2 = Column(Float, nullable=False)
    # psfdepth_g = Column(Float, nullable=False)
    # psfdepth_r = Column(Float, nullable=False)
    # psfdepth_z = Column(Float, nullable=False)
    # galdepth_g = Column(Float, nullable=False)
    # galdepth_r = Column(Float, nullable=False)
    # galdepth_z = Column(Float, nullable=False)
    # shapedev_r = Column(Float, nullable=False)
    # shapedev_e1 = Column(Float, nullable=False)
    # shapedev_e2 = Column(Float, nullable=False)
    # shapeexp_r = Column(Float, nullable=False)
    # shapeexp_e1 = Column(Float, nullable=False)
    # shapeexp_e2 = Column(Float, nullable=False)
    # subpriority = Column(Float, nullable=False)
    # desi_target = Column(BigInteger, nullable=False)
    # bgs_target = Column(BigInteger, nullable=False)
    # mws_target = Column(BigInteger, nullable=False)
    # hpxpixel = Column(BigInteger, nullable=False)

    def __repr__(self):
        return "<ZCat(targetid={0.targetid:d})>".format(self)


class FiberAssign(SchemaMixin, Base):
    """Representation of the fiberassign table.
    """

    tileid = Column(Integer, index=True, primary_key=True)
    targetid = Column(BigInteger, index=True, nullable=False)
    petal_loc = Column(Integer, nullable=False)
    device_loc = Column(Integer, nullable=False)
    location = Column(Integer, nullable=False)
    fiber = Column(Integer, primary_key=True)
    fiberstatus = Column(Integer, nullable=False)
    target_ra = Column(Float, nullable=False)
    target_dec = Column(Float, nullable=False)
    pmra = Column(Float, nullable=False)
    pmdec = Column(Float, nullable=False)
    pmra_ivar = Column(Float, nullable=False)
    pmdec_ivar = Column(Float, nullable=False)
    ref_epoch = Column(Float, nullable=False)
    lambda_ref = Column(Float, nullable=False)
    fa_target = Column(BigInteger, nullable=False)
    fa_type = Column(Integer, nullable=False)
    objtype = Column(String, nullable=False)
    fiberassign_x = Column(Float, nullable=False)
    fiberassign_y = Column(Float, nullable=False)
    numtarget = Column(Integer, nullable=False)
    priority = Column(Integer, nullable=False)
    subpriority = Column(Float, nullable=False)
    obsconditions = Column(BigInteger, nullable=False)
    numobs_more = Column(Integer, nullable=False)

    def __repr__(self):
        return "<FiberAssign(tileid={0.tileid:d}, fiber={0.fiber:d})>".format(self)


def load_file(filepath, tcls, hdu=1, expand=None, convert=None, index=None,
              rowfilter=None, q3c=False, chunksize=50000, maxrows=0):
    """Load a data file into the database, assuming that column names map
    to database column names with no surprises.

    Parameters
    ----------
    filepath : :class:`str`
        Full path to the data file.
    tcls : :class:`sqlalchemy.ext.declarative.api.DeclarativeMeta`
        The table to load, represented by its class.
    hdu : :class:`int` or :class:`str`, optional
        Read a data table from this HDU (default 1).
    expand : :class:`dict`, optional
        If set, map FITS column names to one or more alternative column names.
    convert : :class:`dict`, optional
        If set, convert the data for a named (database) column using the
        supplied function.
    index : :class:`str`, optional
        If set, add a column that just counts the number of rows.
    rowfilter : callable, optional
        If set, apply this filter to the rows to be loaded.  The function
        should return :class:`bool`, with ``True`` meaning a good row.
    q3c : :class:`bool`, optional
        If set, create q3c index on the table.
    chunksize : :class:`int`, optional
        If set, load database `chunksize` rows at a time (default 50000).
    maxrows : :class:`int`, optional
        If set, stop loading after `maxrows` are loaded.  Alteratively,
        set `maxrows` to zero (0) to load all rows.
    """
    tn = tcls.__tablename__
    if filepath.endswith('.fits'):
        with fits.open(filepath) as hdulist:
            data = hdulist[hdu].data
    elif filepath.endswith('.ecsv'):
        data = Table.read(filepath, format='ascii.ecsv')
    else:
        log.error("Unrecognized data file, %s!", filepath)
        return
    if maxrows == 0:
        maxrows = len(data)
    log.info("Read data from %s HDU %s", filepath, hdu)
    try:
        colnames = data.names
    except AttributeError:
        colnames = data.colnames
    for col in colnames:
        if data[col].dtype.kind == 'f':
            bad = np.isnan(data[col][0:maxrows])
            if np.any(bad):
                nbad = bad.sum()
                log.warning("%d rows of bad data detected in column " +
                            "%s of %s.", nbad, col, filepath)
                #
                # Temporary workaround for bad flux values, see
                # https://github.com/desihub/desitarget/issues/397
                #
                if col in ('FLUX_R', 'FIBERFLUX_R', 'FIBERTOTFLUX_R'):
                    data[col][0:maxrows][bad] = -9999.0
    log.info("Integrity check complete on %s.", tn)
    if rowfilter is None:
        good_rows = np.ones((maxrows,), dtype=bool)
    else:
        good_rows = rowfilter(data[0:maxrows])
    data_list = [data[col][0:maxrows][good_rows].tolist() for col in colnames]
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
            engine.execute(tcls.__table__.insert(), data_chunk)
            log.info("Inserted %d rows in %s.",
                     min((k+1)*chunksize, finalrows), tn)
    # for k in range(finalrows//chunksize + 1):
    #     data_insert = [dict([(col, data_list[i].pop(0))
    #                          for i, col in enumerate(data_names)])
    #                    for j in range(chunksize)]
    #     session.bulk_insert_mappings(tcls, data_insert)
    #     log.info("Inserted %d rows in %s..",
    #              min((k+1)*chunksize, finalrows), tn)
    # session.commit()
    # dbSession.commit()
    if q3c:
        q3c_index(tn)
    return


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
    if filepath.endswith('.fits'):
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


def load_fiberassign(datapath, maxpass=4, hdu='FIBERASSIGN', q3c=False,
                     latest_epoch=False, last_column='NUMOBS_MORE'):
    """Load fiber assignment files into the fiberassign table.

    Tile files can appear in multiple epochs, so for a given tileid, load
    the tile file with the largest value of epoch.  In the "real world",
    a tile file appears in each epoch until it is observed, therefore
    the tile file corresponding to the actual observation is the one
    with the largest epoch.

    Parameters
    ----------
    datapath : :class:`str`
        Full path to the directory containing tile files.
    maxpass : :class:`int`, optional
        Search for pass numbers up to this value (default 4).
    hdu : :class:`int` or :class:`str`, optional
        Read a data table from this HDU (default 'FIBERASSIGN').
    q3c : :class:`bool`, optional
        If set, create q3c index on the table.
    latest_epoch : :class:`bool`, optional
        If set, search for the latest tile file among several epochs.
    last_column : :class:`str`, optional
        Do not load columns past this name (default 'NUMOBS_MORE').
    """
    fiberpath = os.path.join(datapath, 'fiberassign*.fits*')
    log.info("Using tile file search path: %s.", fiberpath)
    tile_files = glob.glob(fiberpath)
    if len(tile_files) == 0:
        log.error("No tile files found!")
        return
    log.info("Found %d tile files.", len(tile_files))
    #
    # Find the latest epoch for every tile file.
    #
    latest_tiles = dict()
    if latest_epoch:
        tileidre = re.compile(r'/(\d+)/fiberassign/fiberassign\-(\d+)\.fits')
        for f in tile_files:
            m = tileidre.search(f)
            if m is None:
                log.error("Could not match %s!", f)
                continue
            epoch, tileid = map(int, m.groups())
            if tileid in latest_tiles:
                if latest_tiles[tileid][0] < epoch:
                    latest_tiles[tileid] = (epoch, f)
            else:
                latest_tiles[tileid] = (epoch, f)
    else:
        for f in tile_files:
            # fiberassign-TILEID.fits
            tileid = int(re.match(r'fiberassign\-(\d+)\.fits',
                         os.path.basename(f))[1])
            latest_tiles[tileid] = (0, f)
    log.info("Identified %d tile files for loading.", len(latest_tiles))
    #
    # Read the identified tile files.
    #
    data_index = None
    for tileid in latest_tiles:
        epoch, f = latest_tiles[tileid]
        with fits.open(f) as hdulist:
            data = hdulist[hdu].data
        log.info("Read data from %s HDU %s", f, hdu)
        for col in data.names[:data_index]:
            if data[col].dtype.kind == 'f':
                bad = np.isnan(data[col])
                if np.any(bad):
                    nbad = bad.sum()
                    log.warning("%d rows of bad data detected in column " +
                                "%s of %s.", nbad, col, f)
                    #
                    # This replacement may be deprecated in the future.
                    #
                    if col in ('TARGET_RA', 'TARGET_DEC', 'FIBERASSIGN_X', 'FIBERASSIGN_Y'):
                        data[col][bad] = -9999.0
                assert not np.any(np.isnan(data[col]))
                assert np.all(np.isfinite(data[col]))
        n_rows = len(data)
        if data_index is None:
            data_index = data.names.index(last_column) + 1
        data_list = ([[tileid]*n_rows] +
                     [data[col].tolist() for col in data.names[:data_index]])
        data_names = ['tileid'] + [col.lower() for col in data.names[:data_index]]
        log.info("Initial column conversion complete on tileid = %d.", tileid)
        data_rows = list(zip(*data_list))
        log.info("Converted columns into rows on tileid = %d.", tileid)
        dbSession.bulk_insert_mappings(FiberAssign, [dict(zip(data_names, row))
                                                     for row in data_rows])
        log.info("Inserted %d rows in %s for tileid = %d.",
                 n_rows, FiberAssign.__tablename__, tileid)
        dbSession.commit()
    if q3c:
        q3c_index('fiberassign', ra='target_ra')
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
        if overwrite:
            event.listen(Base.metadata, 'before_create',
                         DDL('DROP SCHEMA IF EXISTS {0} CASCADE'.format(schemaname)))
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
                      help="Store data in FILE.")
    prsr.add_argument('-H', '--hostname', action='store', dest='hostname',
                      metavar='HOSTNAME',
                      help='If specified, connect to a PostgreSQL database on HOSTNAME.')
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
    prsr.add_argument('-U', '--username', action='store', dest='username',
                      metavar='USERNAME', default='desidev_admin',
                      help="If specified, connect to a PostgreSQL database with USERNAME.")
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
    # from pkg_resources import resource_filename
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
    loader = [{'filepath': os.path.join(options.datapath, 'targets', 'truth-dark.fits'),
               'tcls': Truth,
               'hdu': 'TRUTH',
               'expand': None,
               'convert': None,
               'index': None,
               'q3c': False,
               'chunksize': options.chunksize,
               'maxrows': options.maxrows},
              {'filepath': os.path.join(options.datapath, 'targets', 'targets-dark.fits'),
               'tcls': Target,
               'hdu': 'TARGETS',
               'expand': {'DCHISQ': ('dchisq_psf', 'dchisq_rex', 'dchisq_dev', 'dchisq_exp', 'dchisq_comp',)},
               'convert': None,
               'index': None,
               'q3c': postgresql,
               'chunksize': options.chunksize,
               'maxrows': options.maxrows},
              {'filepath': os.path.join(options.datapath, 'survey', 'exposures.fits'),
               'tcls': ObsList,
               'hdu': 'EXPOSURES',
               'expand': {'PASS': 'passnum'},
               # 'convert': {'dateobs': lambda x: convert_dateobs(x, tzinfo=utc)},
               'convert': None,
               'index': None,
               'q3c': postgresql,
               'chunksize': options.chunksize,
               'maxrows': options.maxrows},
              {'filepath': os.path.join(options.datapath, 'spectro', 'redux', 'mini', 'zcatalog-mini.fits'),
               'tcls': ZCat,
               'hdu': 'ZCATALOG',
               'expand': {'COEFF': ('coeff_0', 'coeff_1', 'coeff_2', 'coeff_3', 'coeff_4',
                                    'coeff_5', 'coeff_6', 'coeff_7', 'coeff_8', 'coeff_9',)},
               'convert': None,
               'rowfilter': lambda x: ((x['TARGETID'] != 0) & (x['TARGETID'] != -1)),
               'q3c': postgresql,
               'chunksize': options.chunksize,
               'maxrows': options.maxrows}]
    #
    # Load the tables that correspond to a single file.
    #
    for l in loader:
        tn = l['tcls'].__tablename__
        #
        # Don't use .one().  It actually fetches *all* rows.
        #
        q = dbSession.query(l['tcls']).first()
        if q is None:
            if options.redrock and tn == 'zcat':
                log.info("Loading %s from redrock files in %s.", tn, options.datapath)
                load_redrock(datapath=options.datapath, q3c=postgresql)
            else:
                log.info("Loading %s from %s.", tn, l['filepath'])
                load_file(**l)
            log.info("Finished loading %s.", tn)
        else:
            log.info("%s table already loaded.", tn.title())
    #
    # Update truth table.
    #
    for h in ('BGS', 'ELG', 'LRG', 'QSO', 'STAR', 'WD'):
        update_truth(os.path.join(options.datapath, 'targets', 'truth-dark.fits'),
                     'TRUTH_' + h)
    #
    # Load fiber assignment files.
    #
    q = dbSession.query(FiberAssign).first()
    if q is None:
        log.info("Loading FiberAssign from %s.", options.datapath)
        load_fiberassign(options.datapath, q3c=postgresql)
        log.info("Finished loading FiberAssign.")
    else:
        log.info("FiberAssign table already loaded.")
    return 0
