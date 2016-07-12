"""
desispec.io.database
====================

Code for interacting with the file metadatabase.
"""
from __future__ import absolute_import, division, print_function
import sqlite3
from astropy.io import fits
import numpy as np
from glob import glob
import os
import re
from datetime import datetime, timedelta
from .crc import cksum
from ..log import get_logger, DEBUG
from collections import namedtuple
from matplotlib.patches import Circle, Polygon, Wedge
from matplotlib.collections import PatchCollection


Brick = namedtuple('Brick', ['id', 'name', 'q', 'row', 'col', 'ra', 'dec',
                             'ra1', 'ra2', 'dec1', 'dec2', 'area'])


class Tile(object):
    """Simple object to store individual tile data.
    """
    radius = 1.605  # degrees

    def __init__(self, tileid, ra, dec, obs_pass, in_desi):
        self._id = tileid
        self._ra = ra
        self._dec = dec
        self._obs_pass = obs_pass
        self._in_desi = bool(in_desi)
        self._cos_radius = None
        self._area = None
        self._circum_square = None

    def __repr__(self):
        return ("Tile(tileid={0.id:d}, ra={0.ra:f}, dec={0.ra:f}, " +
                "obs_pass={0.obs_pass:d}, in_desi={0.in_desi})").format(self)

    @property
    def id(self):
        return self._id

    @property
    def ra(self):
        return self._ra

    @property
    def dec(self):
        return self._dec

    @property
    def obs_pass(self):
        return self._obs_pass

    @property
    def in_desi(self):
        return self._in_desi

    @property
    def cos_radius(self):
        if self._cos_radius is None:
            self._cos_radius = np.cos(np.radians(self.radius))
        return self._cos_radius

    @property
    def area(self):
        if self._area is None:
            self._area = 2.0*np.pi*(1.0 - self.cos_radius)  # steradians
        return self._area

    @property
    def circum_square(self):
        """Given a `tile`, return the square that circumscribes it.

        Returns
        -------
        :func:`tuple`
            A tuple of RA, Dec, suitable for plotting.
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
            Amount to offset.

        Returns
        -------
        :class:`float`
            An amount to offset.
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
        brick : Brick
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

    def petals(self, Npetals=10):
        """Convert a tile into a set of Wedge objects.

        Parameters
        ----------
        Npetals : :class:`int`, optional
            Number of petals.

        Returns
        -------
        :class:`list`
            A list of Wedge objects.
        """
        p = list()
        petal_angle = 360.0/Npetals
        tile_ra = self.ra + self.offset()
        for k in range(Npetals):
            petal = Wedge((tile_ra, self.dec), self.radius, petal_angle*k,
                          petal_angle*(k+1), facecolor='b')
            p.append(petal)
        return p

    def overlapping_bricks(self, candidates, map_petals=False):
        """Convert a list of potentially overlapping bricks into actual overlaps.

        Parameters
        ----------
        candidates : :class:`list`
            A list of candidate bricks.
        map_petals : bool, optional
            If ``True`` a map of petal number to a list of overlapping bricks
            is returned.

        Returns
        -------
        :class:`list`
            A list of Polygon objects.
        """
        petals = self.petals()
        petal2brick = dict()
        bricks = list()
        for b in candidates:
            b_ra1, b_ra2 = self.brick_offset(b)
            brick_corners = np.array([[b_ra1, b.dec1],
                                      [b_ra2, b.dec1],
                                      [b_ra2, b.dec2],
                                      [b_ra1, b.dec2]])
            brick = Polygon(brick_corners, closed=True, facecolor='r')
            for i, p in enumerate(petals):
                if brick.get_path().intersects_path(p.get_path()):
                    brick.set_facecolor('g')
                    if i in petal2brick:
                        petal2brick[i].append(b.id)
                    else:
                        petal2brick[i] = [b.id]
            bricks.append(brick)
        if map_petals:
            return petal2brick
        return bricks

    def to_frame(self, band, spectrograph, flavor='science', exptime=1000.0):
        """Simulate a DESI frame given a Tile object.

        Parameters
        ----------
        band : :class:`str`
            'b', 'r', 'z'
        spectrograph : :class:`int`
            Spectrograph number [0-9].
        flavor : :class:`str`, optional
            Exposure flavor.
        exptime : :class:`float`, optional
            Exposure time.

        Returns
        -------
        :func:`tuple`
            A tuple suitable for loading into the frame table.
        """
        dateobs = (datetime(2017+self.obs_pass, 1, 1, 0, 0, 0) +
                   timedelta(seconds=(exptime*(self.id%2140))))
        frameid = "{0}{1:d}-{2:08d}".format(band, spectrograph, self.id)
        night = dateobs.strftime("%Y%m%d")
        return (frameid, band, spectrograph, self.id, night, flavor,
                self.ra, self.dec, self.id, exptime, dateobs,
                self.ra, self.dec)


class RawDataCursor(sqlite3.Cursor):
    """Allow simple object-oriented interaction with raw data database.
    """
    insert_brick = """INSERT INTO brick
        (brickname, brickid, brickq, brickrow, brickcol,
        ra, dec, ra1, ra2, dec1, dec2, area)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?);"""
    insert_tile = """INSERT INTO tile (tileid, ra, dec, pass, in_desi)
        VALUES (?, ?, ?, ?, ?);"""
    select_tile = "SELECT * FROM tile WHERE tileid = ?;"
    insert_tile2brick = "INSERT INTO tile2brick (tileid, petalid, brickid) VALUES (?, ?, ?);"
    select_tile2brick = "SELECT * from tile2brick WHERE tileid = ?;"
    insert_frame = """INSERT INTO frame
        (frameid, band, spectrograph, expid, night, flavor, telra, teldec, tileid, exptime, dateobs, alt, az)
        VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"""
    insert_frame2brick = "INSERT INTO frame2brick (frameid, brickid) VALUES (?, ?);"
    insert_framestatus = "INSERT INTO framestatus (frameid, status, stamp) VALUES (?, ?, ?);"
    insert_brickstatus = "INSERT INTO brickstatus (brickid, status, stamp) VALUES (?, ?, ?);"
    insert_night = "INSERT INTO night (night) VALUES (?);"
    select_night = "SELECT night FROM night WHERE night = ?;"
    insert_status = "INSERT INTO status (status) VALUES (?);"
    select_status = "SELECT status FROM status WHERE status = ?;"
    insert_flavor = "INSERT INTO exposureflavor (flavor) VALUES (?);"
    select_flavor = "SELECT flavor FROM exposureflavor WHERE flavor = ?;"

    def __init__(self, *args, **kwargs):
        super(RawDataCursor, self).__init__(*args, **kwargs)
        return

    def load_brick(self, fitsfile, fix_area=False):
        """Load a bricks FITS file into the database.

        Parameters
        ----------
        fitsfile : :class:`str`
            The name of a bricks file.
        fix_area : :class:`bool`, optional
            If ``True``, deal with missing area column.
        """
        with fits.open(fitsfile) as f:
            brickdata = f[1].data
        bricklist = [ brickdata[col].tolist() for col in brickdata.names ]
        if fix_area:
            #
            # This formula computes area in *steradians*.
            #
            area = ((np.radians(brickdata['ra2']) -
                     np.radians(brickdata['ra1'])) *
                    (np.sin(np.radians(brickdata['dec2'])) -
                     np.sin(np.radians(brickdata['dec1']))))
            bricklist.append(area.tolist())
        self.executemany(self.insert_brick, zip(*bricklist))
        return

    def load_tile(self, tilefile):
        """Load tile FITS file into the database.

        Parameters
        ----------
        tilefile : :class:`str`
            The name of a tile file.
        """
        with fits.open(tilefile) as f:
            tile_data = f[1].data
        tile_list = [tile_data['TILEID'].tolist(), tile_data['RA'].tolist(),
                     tile_data['DEC'].tolist(), tile_data['PASS'].tolist(),
                     tile_data['IN_DESI'].tolist()]
        self.executemany(self.insert_tile, zip(*tile_list))
        return

    def is_night(self, night):
        """Returns ``True`` if the night is in the night table.

        Parameters
        ----------
        night : :class:`str`
            Night name.

        Returns
        -------
        :class:`bool`
            ``True`` if the night is in the night table.
        """
        n = (night,)
        self.execute(self.select_night, n)
        rows = self.fetchall()
        return len(rows) == 1

    def load_night(self, nights):
        """Load a night or multiple nights into the night table.

        Parameters
        ----------
        nights : :class:`str` or :class:`list`
            A single night or list of nights.
        """
        if isinstance(nights, str):
            my_nights = [nights]
        else:
            my_nights = nights
        self.executemany(self.insert_night, zip(my_nights))
        return

    def is_status(self, status):
        """Returns ``True`` if `status` is in the status table.

        Parameters
        ----------
        status : :class:`str`
            Status name.

        Returns
        -------
        :class:`bool`
            ``True`` if `status` is in the status table.
        """
        s = (status, )
        self.execute(self.select_status, s)
        rows = self.fetchall()
        return len(rows) == 1

    def load_status(self, statuses):
        """Load a status or multiple statuses into the status table.

        Parameters
        ----------
        statuses : :class:`str` or :class:`list`
            A single night or list of nights.
        """
        if isinstance(statuses, str):
            my_statuses = [statuses]
        else:
            my_statuses = statuses
        self.executemany(self.insert_status, zip(my_statuses))
        return

    def is_flavor(self, flavor):
        """Returns ``True`` if the flavor is in the exposureflavor table.

        Parameters
        ----------
        flavor : :class:`str`
            A flavor name.

        Returns
        -------
        :class:`bool`
            ``True`` if the flavor is in the flavor table.
        """
        f = (flavor,)
        self.execute(self.select_flavor, f)
        rows = self.fetchall()
        return len(rows) == 1

    def load_flavor(self, flavors):
        """Load a flavor or multiple flavors into the exposureflavor table.

        Parameters
        ----------
        flavors : :class:`list` or :class:`str`
            One or more flavor names.
        """
        if isinstance(flavors, str):
            my_flavors = [flavors]
        else:
            my_flavors = flavors
        self.executemany(self.insert_flavor, zip(my_flavors))
        return

    def get_bricks(self, tile):
        """Get the bricks that overlap a tile.

        Parameters
        ----------
        tile : :class:`Tile`
            A Tile object.

        Returns
        -------
        :class:`list`
            A list of Brick objects that overlap `tile`.
        """
        #
        # RA wrap around can be handled by the requirements:
        # cos(tile.ra - ra1) > cos(tile_radius) or
        # cos(tile.ra - ra2) > cos(tile_radius)
        #
        # However sqlite3 doesn't have trig functions, so we do that "offboard".
        #
        q = """SELECT * FROM brick AS b
               WHERE (? + {0:f} > b.dec1)
               AND   (? - {0:f} < b.dec2)
               ORDER BY dec, ra;""".format(tile.radius)
        self.execute(q, (tile.dec, tile.dec))
        bricks = list()
        for b in map(Brick._make, self.fetchall()):
            if ((np.cos(np.radians(tile.ra - b.ra1)) > tile.cos_radius) or
                (np.cos(np.radians(tile.ra - b.ra2)) > tile.cos_radius)):
                bricks.append(b)
        return bricks

    def get_bricks_by_name(self, bricknames):
        """Search for and return brick data given the brick names.

        Parameters
        ----------
        bricknames : :class:`list` or :class:`str`
            Look up one or more brick names.

        Returns
        -------
        :class:`list`
            A list of Brick objects.
        """
        if isinstance(bricknames, str):
            b = [bricknames]
        else:
            b = bricknames
        q = "SELECT * FROM brick WHERE brickname IN ({})".format(','.join(['?']*len(b)))
        self.execute(q, b)
        bricks = list()
        for b in map(Brick._make, self.fetchall()):
            bricks.append(b)
        return bricks

    def get_brickid_by_name(self, bricknames):
        """Return the brickids that correspond to a set of bricknames.

        Parameters
        ----------
        bricknames : :class:`list` or :class:`str`
            Look up one or more brick names.

        Returns
        -------
        :class:`dict`
            A mapping of brick name to brick id.
        """
        bid = dict()
        bricks = self.get_bricks_by_name(bricknames)
        for b in bricks:
            bid[b.name] = b.id
        return bid

    def get_tile(self, tileid, N_tiles=28810):
        """Get the tile specified by `tileid` or a random tile.

        Parameters
        ----------
        tileid : :class:`int`
            Tile ID number.  Set to a non-positive integer to return a
            random tile.
        N_tiles : :class:`int`, optional
            Override the number of tiles.

        Returns
        -------
        :class:`Tile`
            A tile object.
        """
        #
        # tileid is 1-indexed.
        #
        if tileid < 1:
            i = np.random.randint(1, N_tiles+1)
        else:
            i = tileid
        self.execute(self.select_tile, (i,))
        rows = self.fetchall()
        return Tile(*(rows[0]))

    def get_all_tiles(self, obs_pass=0, limit=0):
        """Get all tiles from the database.

        Parameters
        ----------
        obs_pass : :class:`int`, optional
            Select only tiles from this pass.
        limit : :class:`int`, optional
            Limit the number of tiles returned

        Returns
        -------
        :class:`list`
            A list of Tiles.
        """
        q = "SELECT * FROM tile WHERE in_desi = ?"
        params = (1, )
        if obs_pass > 0:
            q += " AND pass = ?"
            params = (1, obs_pass)
        if limit > 0:
            q += " LIMIT {0:d}".format(limit)
        q += ';'
        self.execute(q, params)
        tiles = list()
        for row in self.fetchall():
            tiles.append(Tile(*row))
        return tiles

    def get_tile_bricks(self, tile):
        """Get the bricks that overlap `tile`.

        Parameters
        ----------
        tile : :class:`Tile`
            A Tile object.

        Returns
        -------
        :class:`dict`
            The overlapping bricks in a mapping from petal number to brickid.
        """
        self.execute(self.select_tile2brick, (tile.id,))
        rows = self.fetchall()
        petal2brick = dict()
        for r in rows:
            try:
                petal2brick[r[1]].append(r[2])
            except KeyError:
                petal2brick[r[1]] = [r[2]]
        return petal2brick

    def load_tile2brick(self, obs_pass=0):
        """Load the tile2brick table using simulated tiles.

        Parameters
        ----------
        obs_pass : :class:`int`, optional
            Select only tiles from this pass.
        """
        tiles = self.get_all_tiles(obs_pass=obs_pass)
        for tile in tiles:
            # petal2brick[tile.id] = dict()
            candidate_bricks = self.get_bricks(tile)
            petal2brick = tile.overlapping_bricks(candidate_bricks, map_petals=True)
            for p in petal2brick:
                nb = len(petal2brick[p])
                self.executemany(self.insert_tile2brick, zip([tile.id]*nb, [p]*nb, petal2brick[p]))
        return

    def load_simulated_data(self, obs_pass=0):
        """Load simulated frame and brick data.

        Parameters
        ----------
        obs_pass : :class:`int`, optional
            If set, only simulate one pass.
        """
        log = get_logger()
        tiles = self.get_all_tiles(obs_pass=obs_pass)
        status = 'succeeded'
        for t in tiles:
            petal2brick = self.get_tile_bricks(t)
            frame_data = list()
            frame2brick_data = list()
            framestatus_data = list()
            brickstatus_data = list()
            for band in 'brz':
                for spectrograph in range(10):
                    f = t.to_frame(band, spectrograph)
                    if not self.is_night(f[4]):
                        self.load_night(f[4])
                    if not self.is_flavor(f[5]):
                        self.load_flavor(f[5])
                    if not self.is_status(status):
                        self.load_status(status)
                    frame_data.append(f)
                    framestatus_data.append((f[0], status, f[10]))
                    for brick in petal2brick[spectrograph]:
                        frame2brick_data.append((f[0], brick))
                        brickstatus_data.append((brick, status, f[10]))
            #
            #
            #
            self.insert_frame_data(frame_data, frame2brick_data,
                                   framestatus_data, brickstatus_data)
            log.info("Completed insert of tileid = {0:d}.".format(t.id))
        return

    def load_data(self, datapath):
        """Load a night or multiple nights into the night table.

        Parameters
        ----------
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
        for k, f in enumerate(fibermaps):
            with fits.open(f) as hdulist:
                # fiberhdr = hdulist['FIBERMAP'].header
                # night = fiberhdr['NIGHT']
                # dateobs = datetime.strptime(fiberhdr['DATE-OBS'],
                #                             '%Y-%m-%dT%H:%M:%S')
                bricknames = list(set(hdulist['FIBERMAP'].data['BRICKNAME'].tolist()))
            brickids = self.get_brickid_by_name(bricknames)
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
                    tileid = hdulist[0].header['TILEID']
                    exptime = hdulist[0].header['EXPTIME']
                    dateobs = datetime.strptime(hdulist[0].header['DATE-OBS'], '%Y-%m-%dT%H:%M:%S')
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
                frameid = "{0}-{1:08d}".format(camera, expid)
                if not self.is_night(night):
                    self.load_night(night)
                if not self.is_flavor(flavor):
                    self.load_flavor(flavor)
                if not self.is_status(status):
                    self.load_status(status)
                frame_data.append((
                    frameid, # frameid, e.g. b0-00012345
                    band, # b, r, z
                    spectrograph, # 0-9
                    expid, # expid
                    night, # night
                    flavor, # flavor
                    telra, # telra
                    teldec, # teldec
                    tileid, # tileid
                    exptime, # exptime
                    dateobs, # dateobs
                    alt, # alt
                    az)) # az
                framestatus_data.append( (frameid, status, dateobs) )
                for i in brickids:
                    frame2brick_data.append( (frameid, brickids[i]) )
                    brickstatus_data.append( (brickids[i], status, dateobs) )
        #
        #
        #
        self.insert_frame_data(frame_data, frame2brick_data,
                               framestatus_data, brickstatus_data)
        log.info("Completed insert of frame data.")
        return exposures

    def insert_frame_data(self, frame, frame2brick, framestatus, brickstatus):
        """Actually insert the data loaded from raw data files or simulations.

        Parameters
        ----------
        frame : :class:`list`
            Data to be inserted into the ``frame`` table.
        frame : :class:`list`
            Data to be inserted into the ``frame`` table.
        frame : :class:`list`
            Data to be inserted into the ``frame`` table.
        frame : :class:`list`
            Data to be inserted into the ``frame`` table.
        """
        self.executemany(self.insert_frame, frame)
        self.executemany(self.insert_frame2brick, frame2brick)
        self.executemany(self.insert_framestatus, framestatus)
        self.executemany(self.insert_brickstatus, brickstatus)
        return


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
    from pkg_resources import resource_filename
    parser = ArgumentParser(description=("Create and load a DESI metadata "+
                                         "database."))
    parser.add_argument('-a', '--area', action='store_true', dest='fixarea',
        help='If area is not specified in the brick file, recompute it.')
    parser.add_argument('-b', '--bricks', action='store', dest='brickfile',
        default='bricks-0.50-2.fits', metavar='FILE',
        help='Read brick data from FILE.')
    parser.add_argument('-c', '--clobber', action='store_true', dest='clobber',
        help='Delete any existing file before loading.')
    parser.add_argument('-d', '--data', action='store', dest='datapath',
        default=os.path.join(os.environ['DESI_SPECTRO_SIM'],
                             os.environ['SPECPROD']),
        metavar='DIR', help='Load the data in DIR.')
    parser.add_argument('-f', '--filename', action='store', dest='dbfile',
        default='metadata.db', metavar='FILE',
        help="Store data in FILE.")
    parser.add_argument('-p', '--pass', action='store', dest='obs_pass',
        default=0, type=int, metavar='PASS',
        help="Only simulate frames associated with PASS.")
    parser.add_argument('-s', '--simulate', action='store_true',
        dest='simulate', help="Run a simulation using DESI tiles.")
    parser.add_argument('-t', '--tiles', action='store', dest='tilefile',
        default='desi-tiles.fits', metavar='FILE',
        help='Read tile data from FILE.')
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose',
        help='Print extra information.')
    options = parser.parse_args()
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
    dbfile = os.path.join(options.datapath, options.dbfile)
    if options.clobber and os.path.exists(dbfile):
        log.info("Removing file: {0}.".format(dbfile))
        os.remove(dbfile)
    if os.path.exists(dbfile):
        script = None
    else:
        schema = resource_filename('desispec', 'data/db/raw_data.sql')
        log.info("Reading schema from {0}.".format(schema))
        with open(schema) as sql:
            script = sql.read()
    conn = sqlite3.connect(dbfile)
    c = conn.cursor(RawDataCursor)
    if script is not None:
        c.executescript(script)
        c.connection.commit()
        log.info("Created schema.")
        brickfile = os.path.join(options.datapath, options.brickfile)
        c.load_brick(brickfile, fix_area=options.fixarea)
        c.connection.commit()
        log.info("Loaded bricks from {0}.".format(brickfile))
    tilefile = os.path.join(options.datapath, options.tilefile)
    if os.path.exists(tilefile):
        c.execute("SELECT COUNT(*) FROM tile;")
        rows = c.fetchall()
        if rows[0][0] == 0:
            c.load_tile(tilefile)
            log.info("Loaded tiles from {0}.".format(tilefile))
            c.connection.commit()
    if options.simulate:
        c.execute("SELECT COUNT(*) FROM tile2brick;")
        rows = c.fetchall()
        if rows[0][0] == 0:
            c.load_tile2brick(obs_pass=options.obs_pass)
            log.info("Completed tile2brick mapping.")
            c.connection.commit()
        c.load_simulated_data(obs_pass=options.obs_pass)
    else:
        exposurepaths = glob(os.path.join(options.datapath,
                                          '[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]'))
        exposures = list()
        for e in exposurepaths:
            log.info("Loading exposures in {0}.".format(e))
            exposures += c.load_data(e)
        log.info("Loaded exposures: {0}".format(', '.join(map(str,exposures))))
    c.connection.commit()
    c.connection.close()
    return 0
