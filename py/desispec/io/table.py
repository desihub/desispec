"""
desispec.io.table
=================

Utility functions for reading FITS tables 
"""

import fitsio
from astropy.table import Table
from .util import addkeys

def read_table(filename, ext=None):
    """
    Reads a FITS table into an astropy Table, avoiding masked columns
   
    Args:
        filename (str): full path to input fits file

    Options:
        ext (str or int): EXTNAME or extension number to read

    Context: astropy 5.0 Table.read converts NaN and blank strings to
    masked values, which is a pain.  This function reads the file with
    fitsio and then converts to a Table.
    """

    data, header = fitsio.read(filename, ext=ext, header=True)
    table = Table(data)
    if 'EXTNAME' in header:
        table.meta['EXTNAME'] = header['EXTNAME']
    # table.meta.update(header)
    addkeys(table.meta, header)
    return table

