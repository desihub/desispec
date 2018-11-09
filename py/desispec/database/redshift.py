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
                        BigInteger, Integer, String, Float, DateTime,
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
    contam_target = Column(BigInteger, nullable=False)
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
        return ("<Truth(targetid={0.targetid:d}, " +
                "mockid={0.mockid:d}, " +
                "contam_target={0.contam_target:d}, " +
                "truez={0.truez:f}, " +
                "truespectype='{0.truespectype}', " +
                "templatetype='{0.templatetype}', " +
                "templatesubtype='{0.templatesubtype}', " +
                "templateid={0.templateid:d}, " +
                "seed={0.seed:d}, " +
                "mag={0.mag:f}, " +
                "flux_g={0.flux_g:f}, " +
                "flux_r={0.flux_r:f}, " +
                "flux_z={0.flux_z:f}, " +
                "flux_w1={0.flux_w1:f}, " +
                "flux_w2={0.flux_w2:f}, " +
                "oiiflux={0.oiiflux:f}, " +
                "hbetaflux={0.hbetaflux:f}, " +
                "ewoii={0.ewoii:f}, " +
                "ewhbeta={0.ewhbeta:f}, " +
                "d4000={0.d4000:f}, " +
                "vdisp={0.vdisp:f}, " +
                "oiidoublet={0.oiidoublet:f}, " +
                "oiiihbeta={0.oiiihbeta:f}, " +
                "oiihbeta={0.oiihbeta:f}, " +
                "niihbeta={0.niihbeta:f}, " +
                "siihbeta={0.siihbeta:f}, " +
                "mabs_1450={0.mabs_1450:f}, " +
                "bal_templateid={0.bal_templateid:d}, " +
                "truez_norsd={0.truez_norsd:f}, " +
                "teff={0.teff:f}, " +
                "logg={0.logg:f}, " +
                "feh={0.feh:f}" +
                ")>").format(self)


class Target(SchemaMixin, Base):
    """Representation of the target table.
    """

    brickid = Column(Integer, nullable=False)
    brickname = Column(String, nullable=False)
    brick_objid = Column(Integer, nullable=False)
    ra = Column(Float, nullable=False)
    dec = Column(Float, nullable=False)
    flux_g = Column(Float, nullable=False)
    flux_r = Column(Float, nullable=False)
    flux_z = Column(Float, nullable=False)
    flux_w1 = Column(Float, nullable=False)
    flux_w2 = Column(Float, nullable=False)
    shapeexp_r = Column(Float, nullable=False)
    shapeexp_e1 = Column(Float, nullable=False)
    shapeexp_e2 = Column(Float, nullable=False)
    shapedev_r = Column(Float, nullable=False)
    shapedev_e1 = Column(Float, nullable=False)
    shapedev_e2 = Column(Float, nullable=False)
    psfdepth_g = Column(Float, nullable=False)
    psfdepth_r = Column(Float, nullable=False)
    psfdepth_z = Column(Float, nullable=False)
    galdepth_g = Column(Float, nullable=False)
    galdepth_r = Column(Float, nullable=False)
    galdepth_z = Column(Float, nullable=False)
    mw_transmission_g = Column(Float, nullable=False)
    mw_transmission_r = Column(Float, nullable=False)
    mw_transmission_z = Column(Float, nullable=False)
    mw_transmission_w1 = Column(Float, nullable=False)
    mw_transmission_w2 = Column(Float, nullable=False)
    targetid = Column(BigInteger, primary_key=True, autoincrement=False)
    desi_target = Column(BigInteger, nullable=False)
    bgs_target = Column(BigInteger, nullable=False)
    mws_target = Column(BigInteger, nullable=False)
    hpxpixel = Column(BigInteger, nullable=False)
    subpriority = Column(Float, nullable=False)

    def __repr__(self):
        return ("<Target(brickid={0.brickid:d}, " +
                "brickname='{0.brickname}', " +
                "brick_objid={0.brick_objid:d}, " +
                "ra={0.ra:f}, dec={0.dec:f}, " +
                "flux_g={0.flux_g:f}, " +
                "flux_r={0.flux_r:f}, " +
                "flux_z={0.flux_z:f}, " +
                "flux_w1={0.flux_w1:f}, " +
                "flux_w2={0.flux_w2:f}, " +
                "shapeexp_r={0.shapeexp_r:f}," +
                "shapeexp_e1={0.shapeexp_e1:f}," +
                "shapeexp_e2={0.shapeexp_e2:f}," +
                "shapedev_r={0.shapedev_r:f}, " +
                "shapedev_e1={0.shapedev_e1:f}," +
                "shapedev_e2={0.shapedev_e2:f}," +
                "psfdepth_g={0.psfdepth_g:f}, " +
                "psfdepth_r={0.psfdepth_r:f}, " +
                "psfdepth_z={0.psfdepth_z:f}, " +
                "galdepth_g={0.galdepth_g:f}, " +
                "galdepth_r={0.galdepth_r:f}, " +
                "galdepth_z={0.galdepth_z:f}, " +
                "mw_transmission_g={0.mw_transmission_g:f}, " +
                "mw_transmission_r={0.mw_transmission_r:f}, " +
                "mw_transmission_z={0.mw_transmission_z:f}, " +
                "mw_transmission_w1={0.mw_transmission_w1:f}, " +
                "mw_transmission_w2={0.mw_transmission_w2:f}, " +
                "targetid={0.targetid:d}, " +
                "desi_target={0.desi_target:d}, bgs_target={0.bgs_target}, " +
                "mws_target={0.mws_target:d}, " +
                "hpxpixel={0.hpxpixel:d}, " +
                "subpriority={0.subpriority:f}" +
                ")>").format(self)


class ObsList(SchemaMixin, Base):
    """Representation of the obslist table.
    """

    expid = Column(Integer, primary_key=True, autoincrement=False)
    tileid = Column(Integer, nullable=False)
    passnum = Column(Integer, nullable=False)
    ra = Column(Float, nullable=False)
    dec = Column(Float, nullable=False)
    ebmv = Column(Float, nullable=False)
    night = Column(String, nullable=False)
    mjd = Column(Float, nullable=False)
    exptime = Column(Float, nullable=False)
    seeing = Column(Float, nullable=False)
    transparency = Column(Float, nullable=False)
    airmass = Column(Float, nullable=False)
    moonfrac = Column(Float, nullable=False)
    moonalt = Column(Float, nullable=False)
    moonsep = Column(Float, nullable=False)
    program = Column(String, nullable=False)
    flavor = Column(String, nullable=False)
    # dateobs = Column(DateTime(timezone=True), nullable=False)

    def __repr__(self):
        return ("<ObsList(" +
                "expid={0.expid:d}, " +
                "tileid={0.tileid:d}, " +
                "passnum={0.passnum:d}, " +
                "ra={0.ra:f}, dec={0.dec:f}, " +
                "ebmv={0.ebmv:f}, " +
                "night='{0.night}', " +
                "mjd={0.mjd:f}, " +
                "exptime={0.exptime:f}, " +
                "seeing={0.seeing:f}, " +
                "transparency={0.transparency:f}, " +
                "airmass={0.airmass:f}," +
                "moonfrac={0.moonfrac:f}, " +
                "moonalt={0.moonalt:f}, " +
                "moonsep={0.moonsep:f}" +
                "program='{0.program}'," +
                "flavor='{0.flavor}'" +
                ")>").format(self)


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
    brickid = Column(Integer, nullable=False)
    brick_objid = Column(Integer, nullable=False)
    ra = Column(Float, nullable=False)
    dec = Column(Float, nullable=False)
    flux_g = Column(Float, nullable=False)
    flux_r = Column(Float, nullable=False)
    flux_z = Column(Float, nullable=False)
    flux_w1 = Column(Float, nullable=False)
    flux_w2 = Column(Float, nullable=False)
    mw_transmission_g = Column(Float, nullable=False)
    mw_transmission_r = Column(Float, nullable=False)
    mw_transmission_z = Column(Float, nullable=False)
    mw_transmission_w1 = Column(Float, nullable=False)
    mw_transmission_w2 = Column(Float, nullable=False)
    psfdepth_g = Column(Float, nullable=False)
    psfdepth_r = Column(Float, nullable=False)
    psfdepth_z = Column(Float, nullable=False)
    galdepth_g = Column(Float, nullable=False)
    galdepth_r = Column(Float, nullable=False)
    galdepth_z = Column(Float, nullable=False)
    shapedev_r = Column(Float, nullable=False)
    shapedev_e1 = Column(Float, nullable=False)
    shapedev_e2 = Column(Float, nullable=False)
    shapeexp_r = Column(Float, nullable=False)
    shapeexp_e1 = Column(Float, nullable=False)
    shapeexp_e2 = Column(Float, nullable=False)
    subpriority = Column(Float, nullable=False)
    desi_target = Column(BigInteger, nullable=False)
    bgs_target = Column(BigInteger, nullable=False)
    mws_target = Column(BigInteger, nullable=False)
    hpxpixel = Column(BigInteger, nullable=False)

    def __repr__(self):
        return ("<ZCat(" +
                "targetid={0.targetid:d}, " +
                "chi2={0.chi2:f}, " +
                "coeff_0={0.coeff_0:f}, " +
                "coeff_1={0.coeff_1:f}, " +
                "coeff_2={0.coeff_2:f}, " +
                "coeff_3={0.coeff_3:f}, " +
                "coeff_4={0.coeff_4:f}, " +
                "coeff_5={0.coeff_5:f}, " +
                "coeff_6={0.coeff_6:f}, " +
                "coeff_7={0.coeff_7:f}, " +
                "coeff_8={0.coeff_8:f}, " +
                "coeff_9={0.coeff_9:f}, " +
                "z={0.z:f}, zerr={0.zerr:f}, zwarn={0.zwarn:d}, " +
                "npixels={0.npixels:d}, " +
                "spectype='{0.spectype}', " +
                "subtype='{0.subtype}', " +
                "ncoeff={0.ncoeff:d}, " +
                "deltachi2={0.deltachi2:f}, " +
                "brickname='{0.brickname}', " +
                "numexp={0.numexp:d}, " +
                "numtile={0.numtile:d}, " +
                "brickid={0.brickid:d}, " +
                "brick_objid={0.brick_objid:d}, " +
                "ra={0.ra:f}, dec={0.dec:f}, " +
                "flux_g={0.flux_g:f}, " +
                "flux_r={0.flux_r:f}, " +
                "flux_z={0.flux_z:f}, " +
                "flux_w1={0.flux_w1:f}, " +
                "flux_w2={0.flux_w2:f}, " +
                "mw_transmission_g={0.mw_transmission_g:f}, " +
                "mw_transmission_r={0.mw_transmission_r:f}, " +
                "mw_transmission_z={0.mw_transmission_z:f}, " +
                "mw_transmission_w1={0.mw_transmission_w1:f}, " +
                "mw_transmission_w2={0.mw_transmission_w2:f}, " +
                "psfdepth_g={0.psfdepth_g:f}, " +
                "psfdepth_r={0.psfdepth_r:f}, " +
                "psfdepth_z={0.psfdepth_z:f}, " +
                "galdepth_g={0.galdepth_g:f}, " +
                "galdepth_r={0.galdepth_r:f}, " +
                "galdepth_z={0.galdepth_z:f}, " +
                "shapedev_r={0.shapedev_r:f}, " +
                "shapedev_e1={0.shapedev_e1:f}," +
                "shapedev_e2={0.shapedev_e2:f}," +
                "shapeexp_r={0.shapeexp_r:f}," +
                "shapeexp_e1={0.shapeexp_e1:f}," +
                "shapeexp_e2={0.shapeexp_e2:f}," +
                "subpriority={0.subpriority:f}, " +
                "desi_target={0.desi_target:d}, bgs_target={0.bgs_target}, " +
                "mws_target={0.mws_target:d}, " +
                "hpxpixel={0.hpxpixel:d}" +
                ")>").format(self)


class FiberAssign(SchemaMixin, Base):
    """Representation of the fiberassign table.
    """

    tileid = Column(Integer, index=True, primary_key=True)
    fiber = Column(Integer, primary_key=True)
    location = Column(Integer, nullable=False)
    numtarget = Column(Integer, nullable=False)
    priority = Column(Integer, nullable=False)
    targetid = Column(BigInteger, index=True, nullable=False)
    desi_target = Column(BigInteger, nullable=False)
    bgs_target = Column(BigInteger, nullable=False)
    mws_target = Column(BigInteger, nullable=False)
    target_ra = Column(Float, nullable=False)
    target_dec = Column(Float, nullable=False)
    design_x = Column(Float, nullable=False)
    design_y = Column(Float, nullable=False)
    brickname = Column(String, index=True, nullable=False)

    def __repr__(self):
        return ("<FiberAssign(tileid={0.tileid:d}, " +
                "fiber={0.fiber:d}, " +
                "location={0.location:d}, " +
                "numtarget={0.numtarget:d}, " +
                "priority={0.priority:d}, " +
                "targetid={0.targetid:d}, " +
                "desi_target={0.desi_target:d}, bgs_target={0.bgs_target}, " +
                "mws_target={0.mws_target:d}, " +
                "target_ra={0.target_ra:f}, target_dec={0.target_dec:f}, " +
                "design_x={0.design_x:f}, " +
                "design_y={0.design_y:f}, " +
                "brickname='{0.brickname}')>").format(self)


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
    log.info("Read data from %s.", filepath)
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
    log.info("Integrity check complete on %s.", tn)
    if rowfilter is None:
        good_rows = np.ones((maxrows,), dtype=np.bool)
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
    log.info("Read data from %s.", filepath)
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
    #     good_rows = np.ones((maxrows,), dtype=np.bool)
    # else:
    #     good_rows = rowfilter(data[0:maxrows])
    # data_list = [data[col][0:maxrows][good_rows].tolist() for col in colnames]
    data_list = [data[col].tolist() for col in colnames if col != 'EMLINES']
    data_names = [col.lower() for col in colnames if col != 'EMLINES']
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


def load_zbest(datapath=None, hdu='ZBEST', q3c=False):
    """Load zbest files into the zcat table.

    This function is deprecated since there should now be a single
    redshift catalog file.

    Parameters
    ----------
    datapath : :class:`str`
        Full path to the directory containing zbest files.
    hdu : :class:`int` or :class:`str`, optional
        Read a data table from this HDU (default 'ZBEST').
    q3c : :class:`bool`, optional
        If set, create q3c index on the table.
    """
    if datapath is None:
        datapath = specprod_root()
    zbestpath = os.path.join(datapath, 'spectra-64', '*', '*', 'zbest-64-*.fits')
    log.info("Using zbest file search path: %s.", zbestpath)
    zbest_files = glob.glob(zbestpath)
    if len(zbest_files) == 0:
        log.error("No zbest files found!")
        return
    log.info("Found %d zbest files.", len(zbest_files))
    #
    # Read the identified zbest files.
    #
    for f in zbest_files:
        brickname = os.path.basename(os.path.dirname(f))
        with fits.open(f) as hdulist:
            data = hdulist[hdu].data
        log.info("Read data from %s.", f)
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
        # zbest files don't contain the same columns as zcatalog.
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
                     latest_epoch=False, last_column='BRICKNAME'):
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
        Do not load columns past this name (default 'BRICKNAME').
    """
    fiberpath = os.path.join(datapath, 'tile_*.fits')
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
        tileidre = re.compile(r'/(\d+)/fiberassign/tile_(\d+)\.fits$')
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
            tileid = int((os.path.basename(f).split('.')[0]).split('_')[1])
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
        log.info("Read data from %s.", f)
        for col in ('TARGET_RA', 'TARGET_DEC', 'DESIGN_X', 'DESIGN_Y'):
            data[col][np.isnan(data[col])] = -9999.0
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
    prsr.add_argument('-z', '--zbest', action='store_true', dest='zbest',
                      help='Force loading of the zcat table from zbest files.')
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
    loader = [{'filepath': os.path.join(options.datapath, 'targets', 'truth.fits'),
               'tcls': Truth,
               'hdu': 'TRUTH',
               'expand': None,
               'convert': None,
               'index': None,
               'q3c': False,
               'chunksize': options.chunksize,
               'maxrows': options.maxrows},
              {'filepath': os.path.join(options.datapath, 'targets', 'targets.fits'),
               'tcls': Target,
               'hdu': 'TARGETS',
               'expand': None,
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
            if options.zbest and tn == 'zcat':
                log.info("Loading %s from zbest files in %s.", tn, options.datapath)
                load_zbest(datapath=options.datapath, q3c=postgresql)
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
        update_truth(os.path.join(options.datapath, 'targets', 'truth.fits'),
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
