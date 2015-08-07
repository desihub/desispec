"""
desispec.io.database
====================

Code for interacting with the file metadatabase.
"""
from __future__ import absolute_import, division, print_function
import sqlite3
from astropy.io import fits
import numpy as np
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
