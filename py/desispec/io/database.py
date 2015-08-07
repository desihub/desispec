"""
desispec.io.database
====================

Code for interacting with the file metadatabase.
"""
from __future__ import absolute_import, division, print_function
import sqlite3
from astropy.io import fits
#
#
#
def load_brick(fitsfile,dbfile):
    """Load a bricks FITS file into the database.

    Args:
        fitsfile: string containing the name of a bricks file.
        dbfile: string containing the name of a SQLite database file.

    Returns:
        None
    """
    with fits.open(fitsfile) as f:
        brickdata = f[1].data
    bricklist = [ brickdata[col].tolist() for col in brickdata.names ]
    conn = sqlite3.connect(dbfile)
    c = conn.cursor()
    insert = """INSERT INTO brick
        (brickname, brickid, brickq, brickrow, brickcol, ra, dec, ra1, ra2, dec1, dec2, area)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?);"""
    c.executemany(insert,zip(*bricklist))
    conn.commit()
    conn.close()
    return
