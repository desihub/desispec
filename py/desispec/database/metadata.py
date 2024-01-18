# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.database.metadata
==========================

Code for interacting with the file metadatabase.
"""
from __future__ import absolute_import, division, print_function
import os
import re
from glob import glob
from datetime import datetime, timedelta
import numpy as np
from astropy.io import fits
from pytz import utc
from sqlalchemy import (create_engine, text, Table, ForeignKey, Column,
                        Integer, String, Float, DateTime)
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm import sessionmaker, relationship, reconstructor, declarative_base
from sqlalchemy.orm.exc import NoResultFound, MultipleResultsFound
from matplotlib.patches import Circle, Polygon, Wedge
from matplotlib.collections import PatchCollection
from desiutil.log import get_logger, DEBUG


Base = declarative_base()


frame2brick = Table('frame2brick', Base.metadata,
                    Column('frame_id', ForeignKey('frame.id'),
                           primary_key=True),
                    Column('brick_id', ForeignKey('brick.id'),
                           primary_key=True))


class Brick(Base):
    """Representation of a region of the sky.
    """
    __tablename__ = 'brick'

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    q = Column(Integer, nullable=False)
    row = Column(Integer, nullable=False)
    col = Column(Integer, nullable=False)
    ra = Column(Float, nullable=False)
    dec = Column(Float, nullable=False)
    ra1 = Column(Float, nullable=False)
    dec1 = Column(Float, nullable=False)
    ra2 = Column(Float, nullable=False)
    dec2 = Column(Float, nullable=False)
    area = Column(Float, nullable=False)

    frames = relationship('Frame', secondary=frame2brick,
                          back_populates='bricks')

    def __repr__(self):
        return ("<Brick(id={0.id:d}, name='{0.name}', q={0.q:d}, " +
                "row={0.row:d}, col={0.col:d}, " +
                "ra={0.ra:f}, dec={0.dec:f}, " +
                "ra1={0.ra1:f}, dec1={0.dec1:f}, " +
                "ra2={0.ra2:f}, dec2={0.dec2:f}, " +
                "area={0.area:f})>").format(self)


class Tile(Base):
    """Representation of a particular pointing of the telescope.
    """
    __tablename__ = 'tile'

    id = Column(Integer, primary_key=True)
    ra = Column(Float, nullable=False)
    dec = Column(Float, nullable=False)
    desi_pass = Column(Integer, nullable=False)
    in_desi = Column(Integer, nullable=False)
    ebv_med = Column(Float, nullable=False)
    airmass = Column(Float, nullable=False)
    star_density = Column(Float, nullable=False)
    exposefac = Column(Float, nullable=False)
    program = Column(String, nullable=False)
    obsconditions = Column(Integer, nullable=False)

    def _constants(self):
        """Define mathematical constants associated with a tile.
        """
        self._radius = 1.605  # degrees
        self._cos_radius = 0.9996076746114829  # cos(radius)
        self._area = 0.0024650531167640308  # steradians: 2*pi*(1-cos(radius))
        self._circum_square = None
        self._petal2brick = None
        self._brick_polygons = None

    def __init__(self, *args, **kwargs):
        self._constants()
        super(Tile, self).__init__(*args, **kwargs)

    def __repr__(self):
        return ("<Tile(id={0.id:d}, ra={0.ra:f}, dec={0.dec:f}, " +
                "desi_pass={0.desi_pass:d}, in_desi={0.in_desi:d}, " +
                "ebv_med={0.ebv_med:f}, airmass={0.airmass:f}, " +
                "star_density={0.star_density:f}, " +
                "exposefac={0.exposefac:f}, program='{0.program}', " +
                "obsconditions={0.obsconditions:d})>").format(self)

    @reconstructor
    def init_on_load(self):
        self._constants()

    @property
    def radius(self):
        """Radius of tile in degrees.
        """
        return self._radius

    @property
    def cos_radius(self):
        """Cosine of the radius, precomputed for speed.
        """
        return self._cos_radius

    @property
    def area(self):
        """Area of the tile in steradians.
        """
        return self._area

    @property
    def circum_square(self):
        """Defines a square-like region on the sphere which circumscribes
        the tile.
        """
        if self._circum_square is None:
            tile_ra = self.ra + self.offset()
            ra = [tile_ra - self.radius,
                  tile_ra + self.radius,
                  tile_ra + self.radius,
                  tile_ra - self.radius,
                  tile_ra - self.radius]
            dec = [self.dec - self.radius,
                   self.dec - self.radius,
                   self.dec + self.radius,
                   self.dec + self.radius,
                   self.dec - self.radius]
            self._circum_square = (ra, dec)
        return self._circum_square

    def offset(self, shift=10.0):
        """Provide an offset to move RA away from wrap-around.

        Parameters
        ----------
        shift : :class:`float`, optional
            Amount to offset in degrees.

        Returns
        -------
        :class:`float`
            An amount to offset in degrees.
        """
        if self.ra < shift:
            return shift
        if self.ra > 360.0 - shift:
            return -shift
        return 0.0

    def brick_offset(self, brick):
        """Offset a brick in the same way as a tile.

        Parameters
        ----------
        brick : :class:`~desispec.database.metadata.Brick`
            A brick.

        Returns
        -------
        :func:`tuple`
            A tuple containing the shifted ra1 and ra2.
        """
        brick_ra1 = brick.ra1 + self.offset()
        brick_ra2 = brick.ra2 + self.offset()
        if brick_ra1 < 0:
            brick_ra1 += 360.0
        if brick_ra1 > 360.0:
            brick_ra1 -= 360.0
        if brick_ra2 < 0:
            brick_ra2 += 360.0
        if brick_ra2 > 360.0:
            brick_ra2 -= 360.0
        return (brick_ra1, brick_ra2)

    def _coarse_overlapping_bricks(self, session):
        """Get the bricks that *may* overlap a tile.

        Parameters
        ----------
        session : :class:`sqlalchemy.orm.session.Session`
            Database connection.

        Returns
        -------
        :class:`list`
            A list of :class:`~desispec.database.metadata.Brick` objects.
        """
        candidate_bricks = session.query(Brick).filter(text("(:dec + :radius > brick.dec1) AND (:dec - :radius < brick.dec2)")).params(dec=self.dec, radius=self._radius).all()
        bricks = list()
        for b in candidate_bricks:
            if ((np.cos(np.radians(self.ra - b.ra1)) > self.cos_radius) or
                (np.cos(np.radians(self.ra - b.ra2)) > self.cos_radius)):
                bricks.append(b)
        return bricks

    def petals(self, Npetals=10):
        """Convert a tile into a set of :class:`~matplotlib.patches.Wedge`
        objects.

        Parameters
        ----------
        Npetals : :class:`int`, optional
            Number of petals (default 10).

        Returns
        -------
        :class:`list`
            A list of :class:`~matplotlib.patches.Wedge` objects.
        """
        petal_angle = 360.0/Npetals
        tile_ra = self.ra + self.offset()
        return [Wedge((tile_ra, self.dec), self.radius, petal_angle*k,
                      petal_angle*(k+1), facecolor='b')
                for k in range(Npetals)]

    def overlapping_bricks(self, session, map_petals=False):
        """Perform a geometric calculation to find bricks that overlap
        a tile.

        Parameters
        ----------
        session : :class:`sqlalchemy.orm.session.Session`
            Database connection.
        map_petals : bool, optional
            If ``True`` a map of petal number to a list of overlapping bricks
            is returned.

        Returns
        -------
        :class:`list`
            If `map_petals` is ``False``, a list of
            :class:`~matplotlib.patches.Polygon` objects. Otherwise, a
            :class:`dict` mapping petal number to the
            :class:`~desispec.database.metadata.Brick` objects that overlap that
            petal.
        """
        if self._brick_polygons is None and self._petal2brick is None:
            candidates = self._coarse_overlapping_bricks(session)
            petals = self.petals()
            self._petal2brick = dict()
            self._brick_polygons = list()
            for b in candidates:
                b_ra1, b_ra2 = self.brick_offset(b)
                brick_corners = np.array([[b_ra1, b.dec1],
                                          [b_ra2, b.dec1],
                                          [b_ra2, b.dec2],
                                          [b_ra1, b.dec2]])
                brick_poly = Polygon(brick_corners, closed=True, facecolor='r')
                for i, p in enumerate(petals):
                    if brick_poly.get_path().intersects_path(p.get_path()):
                        brick_poly.set_facecolor('g')
                        if i in self._petal2brick:
                            self._petal2brick[i].append(b)
                        else:
                            self._petal2brick[i] = [b]
                self._brick_polygons.append(brick_poly)
        if map_petals:
            return self._petal2brick
        return self._brick_polygons

    def simulate_frame(self, session, band, spectrograph,
                       flavor='science', exptime=1000.0):
        """Simulate a DESI frame given a Tile object.

        Parameters
        ----------
        session : :class:`sqlalchemy.orm.session.Session`
            Database connection.
        band : :class:`str`
            'b', 'r', 'z'
        spectrograph : :class:`int`
            Spectrograph number [0-9].
        flavor : :class:`str`, optional
            Exposure flavor (default 'science').
        exptime : :class:`float`, optional
            Exposure time in seconds (default 1000).

        Returns
        -------
        :class:`tuple`
            A tuple containing a :class:`~desispec.database.metadata.Frame` object
            ready for loading, and a list of bricks that overlap.
        """
        dateobs = (datetime(2017+self.desi_pass, 1, 1, 0, 0, 0, tzinfo=utc) +
                   timedelta(seconds=(exptime*(self.id%2140))))
        band_map = {'b': 10, 'r': 20, 'z': 30}
        band_id_offset = 10**8
        frame_data = {'id': ((band_map[band]+spectrograph)*band_id_offset +
                             self.id),
                      'name': "{0}{1:d}-{2:08d}".format(band, spectrograph,
                                                        self.id),
                      'band': band,
                      'spectrograph': spectrograph,
                      'expid': self.id,
                      'night': dateobs.strftime("%Y%m%d"),
                      'flavor': flavor,
                      'telra': self.ra,
                      'teldec': self.dec,
                      'tile_id': self.id,
                      'exptime': exptime,
                      'dateobs': dateobs,
                      'alt': self.ra,
                      'az': self.dec}
        petal2brick = self.overlapping_bricks(session, map_petals=True)
        return (Frame(**frame_data), petal2brick[spectrograph])


class Frame(Base):
    """Representation of a particular pointing, exposure, spectrograph and
    band (a.k.a. 'channel' or 'arm').
    """
    __tablename__ = 'frame'

    id = Column(Integer, primary_key=True)  # e.g. 1900012345
    name = Column(String(11), unique=True, nullable=False)  # e.g. b0-00012345
    band = Column(String(1), nullable=False)  # b, r, z
    spectrograph = Column(Integer, nullable=False)  # 0, 1, 2, ...
    expid = Column(Integer, nullable=False)  # exposure number
    night = Column(String, ForeignKey('night.night'), nullable=False)
    flavor = Column(String, ForeignKey('exposureflavor.flavor'),
                    nullable=False)
    telra = Column(Float, nullable=False)
    teldec = Column(Float, nullable=False)
    tile_id = Column(Integer, nullable=False, default=-1)
    exptime = Column(Float, nullable=False)
    dateobs = Column(DateTime(timezone=True), nullable=False)
    alt = Column(Float, nullable=False)
    az = Column(Float, nullable=False)

    bricks = relationship('Brick', secondary=frame2brick,
                          back_populates='frames')

    def __repr__(self):
        return ("<Frame(id='{0.id}', band='{0.band}', " +
                "spectrograph={0.spectrograph:d}, expid={0.expid:d}, " +
                "night='{0.night}', flavor='{0.flavor}', " +
                "telra={0.telra:f}, teldec={0.teldec:f}, " +
                "tileid={0.tileid:d}, exptime={0.exptime:f}, " +
                "dateobs='{0.dateobs}', " +
                "alt={0.alt:f}, az={0.az:f})>").format(self)


class Night(Base):
    """List of observation nights.  Used to constrain the possible values.
    """
    __tablename__ = 'night'

    night = Column(String(8), primary_key=True)

    def __repr__(self):
        return "<Night(night='{0.night}')>".format(self)


class ExposureFlavor(Base):
    """List of exposure flavors.  Used to constrain the possible values.
    """
    __tablename__ = 'exposureflavor'

    flavor = Column(String, primary_key=True)

    def __repr__(self):
        return "<ExposureFlavor(flavor='{0.flavor}')>".format(self)


class Status(Base):
    """List of possible processing statuses.
    """
    __tablename__ = 'status'

    status = Column(String, primary_key=True)

    def __repr__(self):
        return "<Status(status='{0.status}')>".format(self)


class FrameStatus(Base):
    """Representation of the status of a particular
    :class:`~desispec.database.metadata.Frame`.
    """
    __tablename__ = 'framestatus'

    id = Column(Integer, primary_key=True)
    frame_id = Column(String, ForeignKey('frame.id'), nullable=False)
    status = Column(String, ForeignKey('status.status'), nullable=False)
    stamp = Column(DateTime(timezone=True), nullable=False)

    def __repr__(self):
        return ("<FrameStatus(id={0.id:d}, frame_id={0.frame_id:d}, " +
                "status='{0.status}', stamp='{0.stamp}')>").format(self)

class BrickStatus(Base):
    """Representation of the status of a particular
    :class:`~desispec.database.metadata.Brick`.
    """
    __tablename__ = 'brickstatus'

    id = Column(Integer, primary_key=True)
    brick_id = Column(Integer, ForeignKey('brick.id'), nullable=False)
    status = Column(String, ForeignKey('status.status'), nullable=False)
    stamp = Column(DateTime(timezone=True), nullable=False)

    def __repr__(self):
        return ("<BrickStatus(id={0.id:d}, brick_id={0.brick_id:d}, " +
                "status='{0.status}', stamp='{0.stamp}')>").format(self)


def get_all_tiles(session, obs_pass=0, limit=0):
    """Get all tiles from the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.session.Session`
        Database connection.
    obs_pass : :class:`int`, optional
        Select only tiles from this pass.
    limit : :class:`int`, optional
        Limit the number of tiles returned

    Returns
    -------
    :class:`list`
        A list of Tiles.
    """
    q = session.query(Tile).filter_by(in_desi=1)
    if obs_pass > 0:
        q = q.filter_by(desi_pass=obs_pass)
    if limit > 0:
        q = q.limit(limit)
    return q.all()


def load_simulated_data(session, obs_pass=0):
    """Load simulated frame and brick data.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.session.Session`
        Database connection.
    obs_pass : :class:`int`, optional
        If set, only simulate one pass.
    """
    log = get_logger()
    tiles = get_all_tiles(session, obs_pass=obs_pass)
    status = 'succeeded'
    for t in tiles:
        for band in 'brz':
            for spectrograph in range(10):
                frame, bricks = t.simulate_frame(session, band, spectrograph)
                try:
                    q = session.query(Night).filter_by(night=frame.night).one()
                except NoResultFound:
                    session.add(Night(night=frame.night))
                try:
                    q = session.query(ExposureFlavor).filter_by(flavor=frame.flavor).one()
                except NoResultFound:
                    session.add(ExposureFlavor(flavor=frame.flavor))
                # try:
                #     q = session.query(Status).filter_by(status=status).one()
                # except NoResultFound:
                #     session.add(Status(status=status))
                session.add(frame)
                session.add(FrameStatus(frame_id=frame.id, status=status, stamp=frame.dateobs))
                for brick in bricks:
                    session.add(BrickStatus(brick_id=brick.id, status=status, stamp=frame.dateobs))
                frame.bricks = bricks
        session.commit()
        log.info("Completed insert of tileid = {0:d}.".format(t.id))
    return


def load_data(session, datapath):
    """Load a night or multiple nights into the frame table.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.session.Session`
        Database connection.
    datapath : :class:`str`
        Name of a data directory.

    Returns
    -------
    :class:`list`
        A list of the exposure numbers found.
    """
    log = get_logger()
    fibermaps = glob(os.path.join(datapath, 'fibermap*.fits'))
    if len(fibermaps) == 0:
        return []
    # fibermap_ids = self.load_file(fibermaps)
    fibermapre = re.compile(r'fibermap-([0-9]{8})\.fits')
    exposures = [ int(fibermapre.findall(f)[0]) for f in fibermaps ]
    frame_data = list()
    frame2brick_data = list()
    framestatus_data = list()
    brickstatus_data = list()
    status = 'succeeded'
    band_map = {'b': 10, 'r': 20, 'z': 30}
    band_id_offset = 10**8
    for k, f in enumerate(fibermaps):
        with fits.open(f) as hdulist:
            # fiberhdr = hdulist['FIBERMAP'].header
            # night = fiberhdr['NIGHT']
            # dateobs = datetime.strptime(fiberhdr['DATE-OBS'],
            #                             '%Y-%m-%dT%H:%M:%S')
            bricknames = list(set(hdulist['FIBERMAP'].data['BRICKNAME'].tolist()))
        bricks = session.query(Brick).filter(Brick.name.in_(bricknames))
        # datafiles = glob(os.path.join(datapath, 'desi-*-{0:08d}.fits'.format(exposures[k])))
        # if len(datafiles) == 0:
        datafiles = glob(os.path.join(datapath, 'pix-[brz][0-9]-{0:08d}.fits'.format(exposures[k])))
        log.info("Found datafiles: {0}.".format(", ".join(datafiles)))
        # datafile_ids = self.load_file(datafiles)
        for f in datafiles:
            with fits.open(f) as hdulist:
                camera = hdulist[0].header['CAMERA']
                expid = int(hdulist[0].header['EXPID'])
                night = hdulist[0].header['NIGHT']
                flavor = hdulist[0].header['FLAVOR']
                telra = hdulist[0].header['TELRA']
                teldec = hdulist[0].header['TELDEC']
                tile_id = hdulist[0].header['TILEID']
                exptime = hdulist[0].header['EXPTIME']
                dateobs = datetime.strptime(hdulist[0].header['DATE-OBS'], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=utc)
                try:
                    alt = hdulist[0].header['ALT']
                except KeyError:
                    alt = 0.0
                try:
                    az = hdulist[0].header['AZ']
                except KeyError:
                    az = 0.0
            band = camera[0]
            assert band in 'brz'
            spectrograph = int(camera[1])
            assert 0 <= spectrograph <= 9
            frame_data = {'id': (band_map[band]+spectrograph) * band_id_offset + expid,
                          'name': "{0}-{1:08d}".format(camera, expid),
                          'band': band,
                          'spectrograph': spectrograph,
                          'expid': expid,
                          'night': night,
                          'flavor': flavor,
                          'telra': telra,
                          'teldec': teldec,
                          'tile_id': tile_id,
                          'exptime': exptime,
                          'dateobs': dateobs,
                          'alt': alt,
                          'az': az}
            frame = Frame(**frame_data)
            try:
                q = session.query(Night).filter_by(night=frame.night).one()
            except NoResultFound:
                session.add(Night(night=frame.night))
            try:
                q = session.query(ExposureFlavor).filter_by(flavor=frame.flavor).one()
            except NoResultFound:
                session.add(ExposureFlavor(flavor=frame.flavor))
            # try:
            #     q = session.query(Status).filter_by(status=status).one()
            # except NoResultFound:
            #     session.add(Status(status=status))
            frame.bricks = bricks
            session.add(frame)
            session.add(FrameStatus(frame_id=frame.id, status=status, stamp=frame.dateobs))
            for brick in bricks:
                session.add(BrickStatus(brick_id=brick.id, status=status, stamp=frame.dateobs))
        session.commit()
        log.info("Completed insert of fibermap = {0}.".format(f))
    return exposures


def main():
    """Entry point for command-line script.

    Returns
    -------
    :class:`int`
        An integer suitable for passing to :func:`sys.exit`.
    """
    #
    # command-line arguments
    #
    from argparse import ArgumentParser
    prsr = ArgumentParser(description=("Create and load a DESI metadata "+
                                       "database."))
    # prsr.add_argument('-a', '--area', action='store_true', dest='fixarea',
    #                   help=('If area is not specified in the brick file, ' +
    #                         'recompute it.'))
    prsr.add_argument('-b', '--bricks', action='store', dest='brickfile',
                      default='bricks-0.50-2.fits', metavar='FILE',
                      help='Read brick data from FILE.')
    prsr.add_argument('-c', '--clobber', action='store_true', dest='clobber',
                      help='Delete any existing file(s) before loading.')
    prsr.add_argument('-d', '--data', action='store', dest='datapath',
                      default=os.path.join(os.environ['DESI_SPECTRO_SIM'],
                                           os.environ['SPECPROD']),
                      metavar='DIR', help='Load the data in DIR.')
    prsr.add_argument('-f', '--filename', action='store', dest='dbfile',
                      default='metadata.db', metavar='FILE',
                      help="Store data in FILE.")
    prsr.add_argument('-p', '--pass', action='store', dest='obs_pass',
                      default=0, type=int, metavar='PASS',
                      help="Only simulate frames associated with PASS.")
    prsr.add_argument('-s', '--simulate', action='store_true', dest='simulate',
                      help="Run a simulation using DESI tiles.")
    prsr.add_argument('-t', '--tiles', action='store', dest='tilefile',
                      default='desi-tiles.fits', metavar='FILE',
                      help='Read tile data from FILE.')
    prsr.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                      help='Print extra information.')
    options = prsr.parse_args()
    #
    # Logging
    #
    if options.verbose:
        log = get_logger(DEBUG)
    else:
        log = get_logger()
    #
    # Create the file.
    #
    db_file = os.path.join(options.datapath, options.dbfile)
    if options.clobber and os.path.exists(db_file):
        log.info("Removing file: {0}.".format(db_file))
        os.remove(db_file)
    engine = create_engine('sqlite:///'+db_file, echo=options.verbose)
    log.info("Begin creating schema.")
    Base.metadata.create_all(engine)
    log.info("Finished creating schema.")
    Session = sessionmaker()
    Session.configure(bind=engine)
    session = Session()
    try:
        q = session.query(Status).one()
    except MultipleResultsFound:
        log.info("Status table already loaded.")
    except NoResultFound:
        session.add_all([Status(status='not processed'),
                         Status(status='failed'),
                         Status(status='succeeded')])
        session.commit()
    try:
        q = session.query(ExposureFlavor).one()
    except MultipleResultsFound:
        log.info("ExposureFlavor table already loaded.")
    except NoResultFound:
        session.add_all([ExposureFlavor(flavor='science'),
                         ExposureFlavor(flavor='arc'),
                         ExposureFlavor(flavor='flat')])
        session.commit()
    try:
        q = session.query(Brick).one()
    except MultipleResultsFound:
        log.info("Brick table already loaded.")
    except NoResultFound:
        brick_file = os.path.join(options.datapath, options.brickfile)
        log.info("Loading bricks from {0}.".format(brick_file))
        with fits.open(brick_file) as hdulist:
            brick_data = hdulist[1].data
        brick_list = [brick_data[col].tolist() for col in brick_data.names]
        if 'area' not in brick_data.names:
            brick_area = ((np.radians(brick_data['ra2']) -
                           np.radians(brick_data['ra1'])) *
                          (np.sin(np.radians(brick_data['dec2'])) -
                           np.sin(np.radians(brick_data['dec1']))))
            brick_list.append(brick_area.tolist())
        brick_columns = ('name', 'id', 'q', 'row', 'col', 'ra', 'dec',
                         'ra1', 'ra2', 'dec1', 'dec2', 'area')
        session.add_all([Brick(**b) for b in [dict(zip(brick_columns, row))
                                              for row in zip(*brick_list)]])
        session.commit()
        log.info("Finished loading bricks.")
    try:
        q = session.query(Tile).one()
    except MultipleResultsFound:
        log.info("Tile table already loaded.")
    except NoResultFound:
        tile_file = os.path.join(options.datapath, options.tilefile)
        log.info("Loading tiles from {0}.".format(tile_file))
        with fits.open(tile_file) as hdulist:
            tile_data = hdulist[1].data
        tile_list = [tile_data[col].tolist() for col in tile_data.names]
        tile_columns = ('id', 'ra', 'dec', 'desi_pass', 'in_desi', 'ebv_med',
                        'airmass', 'star_density', 'exposefac', 'program',
                        'obsconditions')
        session.add_all([Tile(**t) for t in [dict(zip(tile_columns, row))
                                             for row in zip(*tile_list)]])
        session.commit()
        log.info("Finished loading bricks.")
    if options.simulate:
        try:
            q = session.query(Frame).one()
        except MultipleResultsFound:
            log.info("Frame table already loaded.")
        except NoResultFound:
            load_simulated_data(session, options.obs_pass)
        log.info("Finished loading frames.")
    else:
        log.info("Loading real data.")
        expaths = glob(os.path.join(options.datapath, '[0-9]'*8))
        exposures = list()
        for e in expaths:
            log.info("Loading exposures in {0}.".format(e))
            exposures += load_data(session, e)
        log.info("Loaded exposures: {0}.".format(', '.join(map(str, exposures))))
    session.close()
    return 0
