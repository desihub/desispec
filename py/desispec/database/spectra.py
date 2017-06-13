# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.database.spectra
=========================

Code for loading spectra results into a database.

The *intention* of this module is to support both simulated survey
(quicksurvey) and pipeline data.  However, as of Summer 2017, the
data models are sufficiently different to require separate loaders.
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from sqlalchemy import (create_engine, event, ForeignKey, Column, DDL,
                        BigInteger, Integer, String, Float, DateTime)
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import scoped_session, sessionmaker, relationship
from sqlalchemy.schema import CreateSchema
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
    ra = Column(Float, nullable=False)
    dec = Column(Float, nullable=False)
    truez = Column(Float, nullable=False)
    truetype = Column(String, nullable=False)
    sourcetype = Column(String, nullable=False)
    brickname = Column(String, nullable=False)
    oiiflux = Column(Float, nullable=False)

    def __repr__(self):
        return ("<Truth(targetid={0.targetid:d}, " +
                "ra={0.ra:f}, dec={0.dec:f}, truez={0.truez:f}, " +
                "truetype='{0.truetype}', sourcetype='{0.sourcetype}', " +
                "brickname='{0.brickname}', " +
                "oiiflux={0.oiiflux:f})>").format(self)


class Target(SchemaMixin, Base):
    """Representation of the target table.
    """

    targetid = Column(BigInteger, primary_key=True, autoincrement=False)
    ra = Column(Float, nullable=False)
    dec = Column(Float, nullable=False)
    desi_target = Column(BigInteger, nullable=False)
    bgs_target = Column(BigInteger, nullable=False)
    mws_target = Column(BigInteger, nullable=False)
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

    tileid = Column(Integer, primary_key=True, autoincrement=False)
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

    targetid = Column(BigInteger, primary_key=True, autoincrement=False)
    brickname = Column(String, index=True, nullable=False)
    spectype = Column(String, index=True, nullable=False)
    z = Column(Float, index=True, nullable=False)
    zerr = Column(Float, nullable=False)
    zwarn = Column(Integer, index=True, nullable=False)
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
    location = Column(Integer, nullable=False)
    numtarget = Column(Integer, nullable=False)
    priority = Column(Integer, nullable=False)
    targetid = Column(BigInteger, index=True, nullable=False)
    desi_target = Column(BigInteger, nullable=False)
    bgs_target = Column(BigInteger, nullable=False)
    mws_target = Column(BigInteger, nullable=False)
    ra = Column(Float, nullable=False)
    dec = Column(Float, nullable=False)
    xfocal_design = Column(Float, nullable=False)
    yfocal_design = Column(Float, nullable=False)
    brickname = Column(String, index=True, nullable=False)

    def __repr__(self):
        return ("<FiberAssign(faid={0.faid:d}, " +
                "tileid={0.tileid:d}, " +
                "fiber={0.fiber:d}, " +
                "location={0.location:d}, " +
                "numtarget={0.numtarget:d}, " +
                "priority={0.priority:d}, " +
                "targetid={0.targetid:d}, " +
                "desi_target={0.desi_target:d}, bgs_target={0.bgs_target}, " +
                "mws_target={0.mws_target:d}, " +
                "ra={0.ra:f}, dec={0.dec:f}, " +
                "xfocal_design={0.xfocal_design:f}, " +
                "yfocal_design={0.yfocal_design:f}, " +
                "brickname='{0.brickname}')>").format(self)


def load_file(filepath, tcls, hdu=1, expand=None, convert=None, q3c=False,
              chunksize=50000, maxrows=0):
    """Load a data file into the database, assuming that column names map
    to database column names with no surprises.

    Parameters
    ----------
    filepath : :class:`str`
        Full path to the data file.
    tcls : :class:`sqlalchemy.ext.declarative.api.DeclarativeMeta`
        The table to load, represented by its class.
    hdu : :class:`int`, optional
        Read a data table from this HDU (default 1).
    expand : :class:`dict`, optional
        If set, map FITS column names to one or more alternative column names.
    convert : :class:`dict`, optional
        If set, convert the data for a named (database) column using the
        supplied function.
    q3c : :class:`bool`, optional
        If set, create q3c index on the table.
    chunksize : :class:`int`, optional
        If set, load database `chunksize` rows at a time (default 50000).
    maxrows : :class:`int`, optional
        If set, stop loading after `maxrows` are loaded.  Alteratively,
        set `maxrows` to zero (0) to load all rows.
    """
    from astropy.io import fits
    from astropy.table import Table
    from desiutil.log import get_logger
    log = get_logger()
    tn = tcls.__tablename__
    if filepath.endswith('.fits'):
        with fits.open(filepath) as hdulist:
            data = hdulist[1].data
    elif filepath.endswith('.ecsv'):
        data = Table.read(filepath, format='ascii.ecsv')
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
    if q3c:
        q3c_index(tn)
    return


def load_fiberassign(datapath, maxpass=4, q3c=False):
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
    q3c : :class:`bool`, optional
        If set, create q3c index on the table.
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
    if q3c:
        q3c_index('fiberassign')
    return


def q3c_index(table):
    """Create a q3c index on a table.

    Parameters
    ----------
    table : :class:`str`
        Name of the table to index.
    """
    from desiutil.log import get_logger
    log = get_logger()
    q3c_sql = """CREATE INDEX ix_{table}_q3c_ang2ipix ON {schema}.{table} (q3c_ang2ipix(ra, dec));
    CLUSTER {schema}.{table} USING ix_{table}_q3c_ang2ipix;
    ANALYZE {schema}.{table};
    """.format(schema=schemaname, table=table)
    log.info("Creating q3c index on %s.%s.", schemaname, table)
    dbSession.execute(q3c_sql)
    log.info("Finished q3c index on %s.%s.", schemaname, table)
    dbSession.commit()
    return


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
    if options.schema:
        schemaname = options.schema
        # event.listen(Base.metadata, 'before_create', CreateSchema(schemaname))
        if options.clobber:
            event.listen(Base.metadata, 'before_create',
                         DDL('DROP SCHEMA IF EXISTS {0} CASCADE'.format(schemaname)))
        event.listen(Base.metadata, 'before_create',
                     DDL('CREATE SCHEMA IF NOT EXISTS {0}'.format(schemaname)))
    #
    # Create the file.
    #
    postgresql = False
    if options.hostname:
        postgresql = True
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
    log.info("Begin creating tables.")
    for tab in Base.metadata.tables.values():
        tab.schema = schemaname
    Base.metadata.create_all(engine)
    log.info("Finished creating tables.")
    #
    # Load configuration
    #
    loader = [{'filepath': join(options.datapath, 'input', 'dark', 'truth.fits'),
               'tcls': Truth,
               'hdu': 1,
               'expand': None,
               'convert': None,
               'q3c': postgresql,
               'chunksize': options.chucksize,
               'maxrows': options.maxrows},
              {'filepath': join(options.datapath, 'input', 'dark', 'targets.fits'),
               'tcls': Target,
               'hdu': 1,
               'expand': {'DECAM_FLUX': ('decam_flux_u', 'decam_flux_g',
                                         'decam_flux_r', 'decam_flux_i',
                                         'decam_flux_z', 'decam_flux_Y')},
               'convert': None,
               'q3c': postgresql,
               'chunksize': options.chucksize,
               'maxrows': options.maxrows},
              {'filepath': join(options.datapath, 'input', 'obsconditions', 'Benchmark030_001', 'obslist_all.fits'),
               'tcls': ObsList,
               'hdu': 1,
               'expand': {'DATE-OBS': 'dateobs'},
               'convert': {'dateobs': lambda x: convert_dateobs(x, tzinfo=utc)},
               'q3c': postgresql,
               'chunksize': options.chucksize,
               'maxrows': options.maxrows},
              {'filepath': join(options.datapath, 'output', 'dark', '4', 'zcat.fits'),
               'tcls': ZCat,
               'hdu': 1,
               'expand': None,
               'convert': None,
               'q3c': False,
               'chunksize': options.chucksize,
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
            log.info("Loading %s from %s.", tn, l['filepath'])
            load_file(**l)
            log.info("Finished loading %s.", tn)
        else:
            log.info("%s table already loaded.", tn.title())
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
