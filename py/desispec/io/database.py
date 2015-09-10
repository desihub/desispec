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
from datetime import datetime
from .crc import cksum
from ..log import get_logger, DEBUG
#
#
#
def load_brick(fitsfile,dbfile,fix_area=False):
    """Load a bricks FITS file into the database.

    Args:
        fitsfile: string containing the name of a bricks file.
        dbfile: string containing the name of a SQLite database file.
        fix_area: (optional) If ``True``, deal with missing area column.

    Returns:
        None
    """
    with fits.open(fitsfile) as f:
        brickdata = f[1].data
    bricklist = [ brickdata[col].tolist() for col in brickdata.names ]
    if fix_area:
        area = np.degrees(
            (brickdata['ra2'] - brickdata['ra1'])
            * (np.sin(np.radians(brickdata['dec2'])) - np.sin(np.radians(brickdata['dec1']))))
        bricklist.append(area.tolist())
    conn = sqlite3.connect(dbfile)
    c = conn.cursor()
    insert = """INSERT INTO brick
        (brickname, brickid, brickq, brickrow, brickcol, ra, dec, ra1, ra2, dec1, dec2, area)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?);"""
    c.executemany(insert,zip(*bricklist))
    conn.commit()
    conn.close()
    return
#
#
#
def is_night(night,dbfile):
    """Returns ``True`` if the night is in the night table.
    """
    if isinstance(night,str):
        n = (int(night),)
    else:
        n = (night,)
    conn = sqlite3.connect(dbfile)
    c = conn.cursor()
    q = "SELECT night FROM night WHERE night = ?;"
    c.execute(q,n)
    rows = c.fetchall()
    conn.commit()
    conn.close()
    return len(rows) == 1
#
#
#
def load_night(nights,dbfile):
    """Load a night or multiple nights into the night table.

    Args:
        nights: integer, string or list of nights
        dbfile: string containing the name of a SQLite database file.

    Returns:
        None
    """
    if isinstance(nights,str):
        my_nights = [int(nights)]
    elif isinstance(nights,int):
        my_nights = [nights]
    else:
        my_nights = [int(n) for n in nights]
    conn = sqlite3.connect(dbfile)
    c = conn.cursor()
    insert = """INSERT INTO night (night)
        VALUES (?);"""
    c.executemany(insert,zip(my_nights))
    conn.commit()
    conn.close()
    return
#
#
#
def is_flavor(flavor,dbfile):
    """Returns ``True`` if the flavor is in the exposureflavor table.
    """
    f = (flavor,)
    conn = sqlite3.connect(dbfile)
    c = conn.cursor()
    q = "SELECT flavor FROM exposureflavor WHERE flavor = ?;"
    c.execute(q,f)
    rows = c.fetchall()
    conn.commit()
    conn.close()
    return len(rows) == 1
#
#
#
def load_flavor(flavors,dbfile):
    """Load a flavor or multiple flavors into the exposureflavor table.

    Args:
        flavors: string or list of flavors
        dbfile: string containing the name of a SQLite database file.

    Returns:
        None
    """
    if isinstance(flavors,str):
        my_flavors = [flavors]
    else:
        my_flavors = flavors
    conn = sqlite3.connect(dbfile)
    c = conn.cursor()
    insert = """INSERT INTO exposureflavor (flavor)
        VALUES (?);"""
    c.executemany(insert,zip(my_flavors))
    conn.commit()
    conn.close()
    return
#
#
#
def is_filetype(filetype,dbfile):
    """Returns ``True`` if the filetype is in the exposureflavor table.
    """
    f = (filetype,)
    conn = sqlite3.connect(dbfile)
    c = conn.cursor()
    q = "SELECT type FROM filetype WHERE type = ?;"
    c.execute(q,f)
    rows = c.fetchall()
    conn.commit()
    conn.close()
    return len(rows) == 1
#
#
#
def load_filetype(filetype,dbfile):
    """Load a filetype or multiple filetypes into the filetype table.

    Args:
        filetype: string or list of filetypes
        dbfile: string containing the name of a SQLite database file.

    Returns:
        None
    """
    if isinstance(filetype,str):
        my_types = [filetype]
    else:
        my_types = filetype
    conn = sqlite3.connect(dbfile)
    c = conn.cursor()
    insert = """INSERT INTO filetype (type)
        VALUES (?);"""
    c.executemany(insert,zip(my_types))
    conn.commit()
    conn.close()
    return
#
#
#
def get_bricks_by_name(bricknames,dbfile):
    """Search for and return brick data given the brick names.
    """
    if isinstance(bricknames,str):
        b = [bricknames]
    else:
        b = bricknames
    conn = sqlite3.connect(dbfile)
    c = conn.cursor()
    q = "SELECT * FROM brick WHERE brickname IN ({})".format(','.join(['?']*len(b)))
    c.execute(q,b)
    bricks = c.fetchall()
    conn.commit()
    conn.close()
    return bricks
#
#
#
def get_brickid_by_name(bricknames,dbfile):
    """Return the brickids that correspond to a set of bricknames.
    """
    bid = dict()
    bricks = get_bricks_by_name(bricknames,dbfile)
    for row in bricks:
        bid[row[1]] = row[0]
    return bid
#
#
#
def load_file(files,dbfile):
    """Load a file or list of files into the file table.

    Args:
        files: string or list containing filenames.
        dbfile: string containing the name of a SQLite database file.

    Returns:
        load_file: a list of the file ids.
    """
    if isinstance(files,str):
        my_files = [files]
    else:
        my_files = files
    ids = [cksum(f,hashname='sha1') for f in my_files]
    filenames = [os.path.basename(f) for f in my_files]
    directories = [os.path.dirname(f) for f in my_files]
    prodnames = [os.environ['PRODNAME']]*len(my_files)
    filetypes =[os.path.basename(f).split('-')[0] for f in my_files]
    for t in set(filetypes):
        if not is_filetype(t,dbfile):
            load_filetype(t,dbfile)
    conn = sqlite3.connect(dbfile)
    c = conn.cursor()
    insert = """INSERT INTO file
        (id, filename, directory, prodname, filetype)
        VALUES (?,?,?,?,?);"""
    c.executemany(insert,zip(ids,filenames,directories,prodnames,filetypes))
    conn.commit()
    conn.close()
    return ids
#
#
#
def load_data(datapath,dbfile):
    """Load a night or multiple nights into the night table.

    Args:
        datapath: string containing a data directory.
        dbfile: string containing the name of a SQLite database file.

    Returns:
        load_data: a list of the exposure numbers found.
    """
    from ..log import desi_logger
    fibermaps = glob(os.path.join(datapath,'fibermap*.fits'))
    if len(fibermaps) == 0:
        return []
    fibermap_ids = load_file(fibermaps,dbfile)
    fibermapre = re.compile(r'fibermap-([0-9]{8})\.fits')
    exposures = [ int(fibermapre.findall(f)[0]) for f in fibermaps ]
    exposure_data = list()
    exposure2brick_data = list()
    file2exposure_data = list(zip(fibermap_ids,exposures))
    for k,f in enumerate(fibermaps):
        with fits.open(f) as hdulist:
            night = int(hdulist['FIBERMAP'].header['NIGHT'])
            dateobs = datetime.strptime(hdulist['FIBERMAP'].header['DATE-OBS'],'%Y-%m-%dT%H:%M:%S')
            bricknames = list(set(hdulist['FIBERMAP'].data['BRICKNAME'].tolist()))
        datafiles = glob(os.path.join(datapath,'desi-*-{0:08d}.fits'.format(exposures[k])))
        if len(datafiles) == 0:
            datafiles = glob(os.path.join(datapath,'pix-*-{0:08d}.fits'.format(exposures[k])))
        desi_logger.debug("Found datafiles: {0}.".format(", ".join(datafiles)))
        datafile_ids = load_file(datafiles,dbfile)
        file2exposure_data += list(zip(datafile_ids, [exposures[k]]*len(datafile_ids)))
        with fits.open(datafiles[0]) as hdulist:
            exptime = hdulist[0].header['EXPTIME']
            flavor = hdulist[0].header['FLAVOR']
        if not is_night(night,dbfile):
            load_night(night,dbfile)
        if not is_flavor(flavor,dbfile):
            load_flavor(flavor,dbfile)
        exposure_data.append((
            exposures[k], # expid
            night, # night
            flavor, # flavor
            0.0, # telra
            0.0, # teldec
            -1, # tileid
            exptime, # exptime
            dateobs, # dateobs
            0.0, # alt
            0.0)) # az
        brickids = get_brickid_by_name(bricknames,dbfile)
        for i in brickids:
            exposure2brick_data.append( (exposures[k], brickids[i]) )
    conn = sqlite3.connect(dbfile)
    c = conn.cursor()
    insert = """INSERT INTO exposure
        (expid, night, flavor, telra, teldec, tileid, exptime, dateobs, alt, az)
        VALUES (?,?,?,?,?,?,?,?,?,?);"""
    c.executemany(insert,exposure_data)
    insert = """INSERT INTO exposure2brick
        (expid,brickid) VALUES (?,?);"""
    c.executemany(insert,exposure2brick_data)
    insert = """INSERT INTO file2exposure
        (fileid,expid) VALUES (?,?);"""
    c.executemany(insert,file2exposure_data)
    conn.commit()
    conn.close()
    return exposures
#
#
#
def main():
    """Call this function from a command-line script.
    """
    #
    # command-line arguments
    #
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Create and load a DESI metadata database.")
    parser.add_argument('-a', '--area', action='store_true', dest='fixarea',
        help='If area is not specified in the brick file, recompute it.')
    parser.add_argument('-b', '--bricks', action='store', dest='brickfile',
        default='bricks-0.50-2.fits', metavar='FILE',
        help='Read brick data from FILE.')
    parser.add_argument('-c', '--clobber', action='store_true', dest='clobber',
        help='Delete any existing file before loading.')
    parser.add_argument('-d', '--data', action='store', dest='datapath',
        default=os.path.join(os.environ['DESI_SPECTRO_SIM'],os.environ['PRODNAME']), metavar='DIR',
        help='Load the data in DIR.')
    parser.add_argument('-f', '--filename', action='store', dest='dbfile',
        default='metadata.db', metavar='FILE',
        help="Store data in FILE.")
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
    dbfile = os.path.join(options.datapath,'etc',options.dbfile)
    if options.clobber and os.path.exists(dbfile):
        log.info("Removing file: {0}.".format(dbfile))
        os.remove(dbfile)
    if not os.path.exists(dbfile):
        schema = os.path.join(os.environ['DESISPEC'],'etc','file_db.sql')
        log.info("Reading schema from {0}.".format(schema))
        with open(schema) as sql:
            script = sql.read()
        conn = sqlite3.connect(dbfile)
        c = conn.cursor()
        c.executescript(script)
        conn.commit()
        conn.close()
        log.info("Created schema.")
        brickfile = os.path.join(options.datapath,options.brickfile)
        load_brick(brickfile,dbfile,fix_area=options.fixarea)
        log.info("Loaded bricks from {0}.".format(brickfile))
    exposurepaths = glob(os.path.join(options.datapath,'[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]'))
    exposures = list()
    for e in exposurepaths:
        log.info("Loading exposures in {0}.".format(e))
        exposures += load_data(e,dbfile)
    log.info("Loaded exposures: {0}".format(', '.join(map(str,exposures))))
    return 0
#
# TODO
#
# Load file information; relative directory path
# Which files get entries in the file2brick table?  There could be a lot of
# duplication if every file goes in.
# How to determine file dependencies?
