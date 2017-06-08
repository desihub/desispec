# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.database.spectra
=========================

Code for loading spectra results into a database.

Supports both simulated survey (quicksurvey) and pipeline data.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from sqlalchemy import (create_engine, ForeignKey, Column,
                        Integer, String, Float, DateTime)
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import scoped_session, sessionmaker, relationship


Base = declarative_base()
engine = None
dbSession = scoped_session(sessionmaker())
schemaname = 'quickex'

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

    targetid = Column(Integer, primary_key=True)
    ra = Column(Float, nullable=False)
    dec = Column(Float, nullable=False)
    truez = Column(Float, nullable=False)
    truetype = Column(String, nullable=False)
    sourcetype = Column(String, nullable=False)
    brickname = Column(String, nullable=False)
    oiiflux = Column(Float, nullable=False)

    # frames = relationship('Frame', secondary=frame2brick,
    #                       back_populates='bricks')

    def __repr__(self):
        return ("<Truth(targetid={0.targetid:d}, " +
                "ra={0.ra:f}, dec={0.dec:f}, truez={0.truez:f}, " +
                "truetype='{0.truetype}', sourcetype='{0.sourcetype}', " +
                "brickname='{0.brickname}', " +
                "oiiflux={0.oiiflux:f})>").format(self)


class Target(SchemaMixin, Base):
    """Representation of the target table.
    """

    targetid = Column(Integer, primary_key=True)
    ra = Column(Float, nullable=False)
    dec = Column(Float, nullable=False)
    desi_target = Column(Integer, nullable=False)
    bgs_target = Column(Integer, nullable=False)
    mws_target = Column(Integer, nullable=False)
    subpriority = Column(Float, nullable=False)
    obsconditions = Column(Integer, nullable=False)
    brickname = Column(String, nullable=False)
    decam_flux_u = Column(Float, nullable=False)
    decam_flux_g = Column(Float, nullable=False)
    decam_flux_r = Column(Float, nullable=False)
    decam_flux_i = Column(Float, nullable=False)
    decam_flux_z = Column(Float, nullable=False)
    decam_flux_Y = Column(Float, nullable=False)
    shapedev_r = Column(Float, nullable=False)
    shapeexp_r = Column(Float, nullable=False)
    depth_r = Column(Float, nullable=False)
    galdepth_r = Column(Float, nullable=False)

    def __repr__(self):
        return ("<Target(targetid={0.targetid:d}, " +
                "ra={0.ra:f}, dec={0.dec:f}, " +
                "desi_target={0.desi_target:d}, bgs_target={0.bgs_target}, " +
                "mws_target={0.mws_target:d}, " +
                "subpriority={0.subpriority:f}, " +
                "obsconditions={0.obsconditions:d}, " +
                "brickname='{0.brickname}', " +
                "decam_flux_u={0.decam_flux_u:f}, " +
                "decam_flux_g={0.decam_flux_g:f}, " +
                "decam_flux_r={0.decam_flux_r:f}, " +
                "decam_flux_i={0.decam_flux_i:f}, " +
                "decam_flux_z={0.decam_flux_z:f}, " +
                "decam_flux_Y={0.decam_flux_Y:f}, " +
                "shapedev_r={0.shapedev_r:f}, shapeexp_r={0.shapeexp_r:f}, "
                "depth_r={0.depth_r:f}, " +
                "galdepth_r={0.galdepth_r:f})>").format(self)


class ObsList(SchemaMixin, Base):
    """Representation of the obslist table.
    """

    tileid = Column(Integer, primary_key=True)
    ra = Column(Float, nullable=False)
    dec = Column(Float, nullable=False)
    program = Column(String, nullable=False)
    ebmv = Column(Float, nullable=False)
    maxlen = Column(Float, nullable=False)
    moonfrac = Column(Float, nullable=False)
    moondist = Column(Float, nullable=False)
    moonalt = Column(Float, nullable=False)
    seeing = Column(Float, nullable=False)
    lintrans = Column(Float, nullable=False)
    airmass = Column(Float, nullable=False)
    dessn2 = Column(Float, nullable=False)
    status = Column(Integer, nullable=False)
    exptime = Column(Float, nullable=False)
    obssn2 = Column(Float, nullable=False)
    dateobs = Column(DateTime(timezone=True), nullable=False)
    mjd = Column(Float, nullable=False)

    def __repr__(self):
        return ("<ObsList(tileid={0.tileid:d}, " +
                "ra={0.ra:f}, dec={0.dec:f}, " +
                "program='{0.program}', " +
                "ebmv={0.ebmv:f}, " +
                "maxlen={0.maxlen:f}, " +
                "moonfrac={0.moonfrac:f}, " +
                "moondist={0.moondist:f}, " +
                "moonalt={0.moonalt:f}, " +
                "seeing={0.seeing:f}, " +
                "lintrans={0.lintrans:f}, " +
                "airmass={0.airmass:f}, " +
                "dessn2={0.dessn2:f}, " +
                "status={0.status:d}, " +
                "exptime={0.exptime:f}, " +
                "obssn2={0.obssn2:f}, " +
                "dateobs='{0.dateobs}', " +
                "mjd={0.mjd:f})>").format(self)


class ZCat(SchemaMixin, Base):
    """Representation of the zcat table.
    """

    targetid = Column(Integer, primary_key=True)
    brickname = Column(String, index=True, nullable=False)
    spectype = Column(String, nullable=False)
    z = Column(Float, nullable=False)
    zerr = Column(Float, nullable=False)
    zwarn = Column(Integer, nullable=False)
    numobs = Column(Integer, nullable=False)

    def __repr__(self):
        return ("<ZCat(targetid={0.targetid:d}, " +
                "brickname='{0.brickname}', " +
                "spectype='{0.spectype}', " +
                "z={0.z:f}, zerr={0.zerr:f}, " +
                "zwarn={0.zwarn:d}, numobs={0.numobs:d})>").format(self)


class FiberAssign(SchemaMixin, Base):
    """Representation of the fiberassign table.
    """

    tileid = Column(Integer, index=True, primary_key=True)
    fiber = Column(Integer, primary_key=True)
    positioner = Column(Integer, nullable=False)
    numtarget = Column(Integer, nullable=False)
    priority = Column(Integer, nullable=False)
    targetid = Column(Integer, index=True, nullable=False)
    desi_target = Column(Integer, nullable=False)
    bgs_target = Column(Integer, nullable=False)
    mws_target = Column(Integer, nullable=False)
    ra = Column(Float, nullable=False)
    dec = Column(Float, nullable=False)
    xfocal_design = Column(Float, nullable=False)
    yfocal_design = Column(Float, nullable=False)
    brickname = Column(String, index=True, nullable=False)

    def __repr__(self):
        return ("<FiberAssign(faid={0.faid:d}, " +
                "tileid={0.tileid:d}, " +
                "fiber={0.fiber:d}, " +
                "positioner={0.positioner:d}, " +
                "numtarget={0.numtarget:d}, " +
                "priority={0.priority:d}, " +
                "targetid={0.targetid:d}, " +
                "desi_target={0.desi_target:d}, bgs_target={0.bgs_target}, " +
                "mws_target={0.mws_target:d}, " +
                "ra={0.ra:f}, dec={0.dec:f}, " +
                "xfocal_design={0.xfocal_design:f}, " +
                "yfocal_design={0.yfocal_design:f}, " +
                "brickname='{0.brickname}')>").format(self)


def load_file(filepath, tcls, expand=None, convert=None,
              chunksize=50000, maxrows=0):
    """Load a FITS file into the database, assuming that FITS column names map
    to database column names with no surprises.

    Parameters
    ----------
    filepath : :class:`str`
        Full path to the FITS file.
    session : :class:`sqlalchemy.orm.session.Session`
        Database connection.
    tcls : :class:`sqlalchemy.ext.declarative.api.DeclarativeMeta`
        The table to load, represented by its class.
    expand : :class:`dict`, optional
        If set, map FITS column names to one or more alternative column names.
    convert : :class:`dict`, optional
        If set, convert the data for a named (database) column using the
        supplied function.
    chunksize : :class:`int`, optional
        If set, load database `chunksize` rows at a time (default 50000).
    maxrows : :class:`int`, optional
        If set, stop loading after `maxrows` are loaded.  Alteratively,
        set `maxrows` to zero (0) to load all rows.
    """
    from astropy.io import fits
    from desiutil.log import get_logger
    log = get_logger()
    tn = tcls.__tablename__
    with fits.open(filepath) as hdulist:
        data = hdulist[1].data
    if maxrows == 0:
        maxrows = len(data)
    log.info("Read data from %s.", filepath)
    for col in data.names:
        if data[col].dtype.kind == 'f':
            bad = np.isnan(data[col][0:maxrows])
            if np.any(bad):
                nbad = bad.sum()
                log.warning("%d rows of bad data detected in column " +
                            "%s of %s.", nbad, col, filepath)
    log.info("Integrity check complete on %s.", tn)
    data_list = [data[col][0:maxrows].tolist() for col in data.names]
    data_names = [col.lower() for col in data.names]
    log.info("Initial column conversion complete on %s.", tn)
    if expand is not None:
        for col in expand:
            if isinstance(expand[col], str):
                #
                # Just rename a column.
                #
                data_names[data.names.index(col)] = expand[col]
            else:
                #
                # Assume this is an expansion of an array-valued column
                # into individual columns.
                #
                i = data.names.index(col)
                del data_names[i]
                del data_list[i]
                for j, n in enumerate(expand[col]):
                    data_names.insert(i + j, n)
                    data_list.insert(i + j, data[col][:, j].tolist())
    log.info("Column expansion complete on %s.", tn)
    del data
    if convert is not None:
        for col in convert:
            i = data_names.index(col)
            data_list[i] = [convert[col](x) for x in data_list[i]]
    log.info("Column conversion complete on %s.", tn)
    data_rows = list(zip(*data_list))
    del data_list
    log.info("Converted columns into rows on %s.", tn)
    for k in range(maxrows//chunksize + 1):
        data_chunk = [dict(zip(data_names, row))
                      for row in data_rows[k*chunksize:(k+1)*chunksize]]
        if len(data_chunk) > 0:
            engine.execute(tcls.__table__.insert(), data_chunk)
            log.info("Inserted %d rows in %s.",
                     min((k+1)*chunksize, maxrows), tn)
    # for k in range(maxrows//chunksize + 1):
    #     data_insert = [dict([(col, data_list[i].pop(0))
    #                          for i, col in enumerate(data_names)])
    #                    for j in range(chunksize)]
    #     session.bulk_insert_mappings(tcls, data_insert)
    #     log.info("Inserted %d rows in %s..",
    #              min((k+1)*chunksize, maxrows), tn)
    # session.commit()
    # dbSession.commit()
    return


def load_fiberassign(datapath, maxpass=4):
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
    session : :class:`sqlalchemy.orm.session.Session`
        Database connection.
    maxpass : :class:`int`, optional
        Search for pass numbers up to this value (default 4).
    """
    from os.path import join
    from re import compile
    from glob import glob
    from astropy.io import fits
    from desiutil.log import get_logger
    log = get_logger()
    fiberpath = join(datapath, 'output', 'dark',
                     '[0-{0:d}]'.format(maxpass),
                     'fiberassign', 'tile_*.fits')
    log.info("Using tile file search path: %s.", fiberpath)
    tile_files = glob(fiberpath)
    if len(tile_files) == 0:
        log.error("No tile files found!")
        return
    log.info("Found %d tile files.", len(tile_files))
    tileidre = compile(r'/(\d+)/fiberassign/tile_(\d+)\.fits$')
    #
    # Find the latest epoch for every tile file.
    #
    latest_tiles = dict()
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
    log.info("Identified %d tile files for loading.", len(latest_tiles))
    #
    # Read the identified tile files.
    #
    for tileid in latest_tiles:
        epoch, f = latest_tiles[tileid]
        with fits.open(f) as hdulist:
            data = hdulist[1].data
        log.info("Read data from %s.", f)
        for col in ('RA', 'DEC', 'XFOCAL_DESIGN', 'YFOCAL_DESIGN'):
            data[col][np.isnan(data[col])] = -9999.0
            assert not np.any(np.isnan(data[col]))
            assert np.all(np.isfinite(data[col]))
        n_rows = len(data)
        data_list = ([[tileid]*n_rows] +
                     [data[col].tolist() for col in data.names])
        data_names = ['tileid'] + [col.lower() for col in data.names]
        log.info("Initial column conversion complete on tileid = %d.", tileid)
        data_rows = list(zip(*data_list))
        log.info("Converted columns into rows on tileid = %d.", tileid)
        dbSession.bulk_insert_mappings(FiberAssign, [dict(zip(data_names, row))
                                                     for row in data_rows])
        log.info("Inserted %d rows in %s for tileid = %d.",
                 n_rows, FiberAssign.__tablename__, tileid)
        dbSession.commit()
    return


def convert_dateobs(timestamp, tzinfo=None):
    """Convert a string `timestamp` into a :class:`datetime.datetime` object.

    Parameters
    ----------
    timestamp : :class:`str`
        Timestamp in string format.
    tzinfo : :class:`datetime.tzinfo`, optional
        If set, add time zone to the timestamp.

    Returns
    -------
    :class:`datetime.datetime`
        The converted `timestamp`.
    """
    from datetime import datetime
    x = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%f')
    if tzinfo is not None:
        x = x.replace(tzinfo=tzinfo)
    return x


def parse_pgpass(hostname='scidb2.nersc.gov', username='desidev_admin'):
    """Read a ``~/.pgpass`` file.

    Parameters
    ----------
    hostname : :class:`str`, optional
        Database hostname.
    username : :class:`str`, optional
        Database username.

    Returns
    -------
    :class:`str`
        A string suitable for creating a SQLAlchemy database engine, or None
        if no matching data was found.
    """
    from os.path import expanduser
    fmt = "postgresql://{3}:{4}@{0}:{1}/{2}"
    try:
        with open(expanduser('~/.pgpass')) as p:
            lines = p.readlines()
    except FileNotFoundError:
        return None
    data = dict()
    for l in lines:
        d = l.strip().split(':')
        if d[0] in data:
            data[d[0]][d[3]] = fmt.format(*d)
        else:
            data[d[0]] = {d[3]: fmt.format(*d)}
    if hostname not in data:
        return None
    try:
        pgpass = data[hostname][username]
    except KeyError:
        return None
    return pgpass


def main():
    """Entry point for command-line script.

    Returns
    -------
    :class:`int`
        An integer suitable for passing to :func:`sys.exit`.
    """
    global engine, schemaname
    from os import remove
    from os.path import basename, exists, join
    from sys import argv
    from argparse import ArgumentParser
    from pkg_resources import resource_filename
    from pytz import utc
    from desiutil.log import get_logger, DEBUG, INFO
    #
    # command-line arguments
    #
    prsr = ArgumentParser(description=("Load quicksurvey simulation into a " +
                                       "database."),
                          prog=basename(argv[0]))
    prsr.add_argument('-c', '--clobber', action='store_true', dest='clobber',
                      help='Delete any existing file(s) before loading.')
    prsr.add_argument('-f', '--filename', action='store', dest='dbfile',
                      default='quicksurvey.db', metavar='FILE',
                      help="Store data in FILE.")
    prsr.add_argument('-H', '--hostname', action='store', dest='hostname',
                      metavar='HOSTNAME',
                      help='If specified, connect to a PostgreSQL database on HOSTNAME.')
    prsr.add_argument('-m', '--max-rows', action='store', dest='maxrows',
                      type=int, default=0, metavar='M',
                      help="Load up to M rows in the tables (default is all rows).")
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
    prsr.add_argument('datapath', metavar='DIR', help='Load the data in DIR.')
    options = prsr.parse_args()
    #
    # Logging
    #
    if options.verbose:
        log = get_logger(DEBUG, timestamp=True)
    else:
        log = get_logger(INFO, timestamp=True)
    #
    # Schema.
    #
    # if options.schema:
    #     schemaname = options.schema
    #
    # Create the file.
    #
    if options.hostname:
        db_connection = parse_pgpass(hostname=options.hostname,
                                     username=options.username)
        if db_connection is None:
            log.critical("Could not load database information!")
            return 1
    else:
        if basename(options.dbfile) == options.dbfile:
            db_file = join(options.datapath, options.dbfile)
        else:
            db_file = options.dbfile
        if options.clobber and exists(db_file):
            log.info("Removing file: %s.", db_file)
            remove(db_file)
        db_connection = 'sqlite:///'+db_file
    #
    # SQLAlchemy stuff.
    #
    engine = create_engine(db_connection, echo=options.verbose)
    dbSession.remove()
    dbSession.configure(bind=engine, autoflush=False, expire_on_commit=False)
    log.info("Begin creating schema.")
    Base.metadata.create_all(engine)
    log.info("Finished creating schema.")
    #
    # Load configuration
    #
    loader = [{'tcls': Truth,
               'path': ('input', 'dark', 'truth.fits'),
               'expand': None,
               'convert': None},
              {'tcls': Target,
               'path': ('input', 'dark', 'targets.fits'),
               'expand': {'DECAM_FLUX': ('decam_flux_u', 'decam_flux_g',
                                         'decam_flux_r', 'decam_flux_i',
                                         'decam_flux_z', 'decam_flux_Y')},
               'convert': None},
              {'tcls': ObsList,
               'path': ('input', 'obsconditions', 'Benchmark030_001', 'obslist_all.fits'),
               'expand': {'DATE-OBS': 'dateobs'},
               'convert': {'dateobs': lambda x: convert_dateobs(x, tzinfo=utc)}},
              {'tcls': ZCat,
               'path': ('output', 'dark', '4', 'zcat.fits'),
               'expand': None,
               'convert': None}]
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
            filepath = join(options.datapath, *l['path'])
            log.info("Loading %s from %s.", tn, filepath)
            load_file(filepath, l['tcls'], expand=l['expand'],
                      convert=l['convert'], chunksize=options.chunksize,
                      maxrows=options.maxrows)
            log.info("Finished loading %s.", tn)
        else:
            log.info("%s table already loaded.", tn.title())
    #
    # Load fiber assignment files.
    #
    q = dbSession.query(FiberAssign).first()
    if q is None:
        log.info("Loading FiberAssign from %s.", options.datapath)
        load_fiberassign(options.datapath)
        log.info("Finished loading FiberAssign.")
    else:
        log.info("FiberAssign table already loaded.")
    return 0
