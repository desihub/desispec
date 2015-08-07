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
            * (np.sin(np.radians(brickdata['dec2'])) - np.sin(np.radians(brickdata['dec2']))))
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
def load_data(datapath,dbfile):
    """Load a night or multiple nights into the night table.

    Args:
        datapath: string containing a data directory.
        dbfile: string containing the name of a SQLite database file.

    Returns:
        load_data: a list of the exposure numbers found.
    """
    fibermaps = glob(os.path.join(datapath,'fibermap*.fits'))
    if len(fibermaps) == 0:
        return []
    fibermapre = re.compile(r'fibermap-([0-9]{8})\.fits')
    exposures = [ int(fibermapre.findall(f)[0]) for f in fibermaps ]
    exposure_data = list()
    for k,f in enumerate(fibermaps):
        with fits.open(f) as hdulist:
            night = int(hdulist['FIBERMAP'].header['NIGHT'])
            dateobs = datetime.strptime(hdulist['FIBERMAP'].header['DATE-OBS'],'%Y-%m-%dT%H:%M:%S')
            bricks = set(hdulist['FIBERMAP'].data['BRICKNAME'].tolist())
        datafiles = glob(os.path.join(datapath,'desi-*-{0:08d}.fits'.format(exposures[k)))
        if len(datafiles) == 0:
            datafiles = glob(os.path.join(datapath,'pix-*-{0:08d}.fits'.format(exposures[k)))
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
    conn = sqlite3.connect(dbfile)
    c = conn.cursor()
    insert = """INSERT INTO exposure
        (expid, night, flavor, telra, teldec, tileid, exptime, dateobs, alt, az)
        VALUES (?,?,?,?,?,?,?,?,?,?);"""
    c.executemany(insert,exposure_data)
    conn.commit()
    conn.close()            
    return exposures
