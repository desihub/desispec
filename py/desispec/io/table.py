"""
desispec.io.table
=================

Utility functions for reading FITS tables
"""

import fitsio
from astropy.table import Table
from .util import addkeys, checkgzip

def read_table(filename, ext=None, rows=None, columns=None):
    """
    Reads a FITS table into an astropy Table, avoiding masked columns

    Args:
        filename (str): full path to input fits file

    Options:
        ext (str or int): EXTNAME or extension number to read
        rows: array/list of rows to read
        columns: array/list of column names to read

    Context: astropy 5.0 Table.read converts NaN and blank strings to
    masked values, which is a pain.  This function reads the file with
    fitsio and then converts to a Table.
    """

    filename = checkgzip(filename)
    data, header = fitsio.read(filename, ext=ext, header=True, rows=rows, columns=columns)
    table = Table(data)
    if 'EXTNAME' in header:
        table.meta['EXTNAME'] = header['EXTNAME']

    # add header keywords while not propagating TFORM, BITPIX, etc
    addkeys(table.meta, header)

    return table

