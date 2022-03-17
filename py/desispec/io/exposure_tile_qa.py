"""
desispec.io.exposure_qa
=================

I/O routines for exposure and tile qa
"""

import os
import astropy.io.fits as pyfits
from astropy.table import Table

from desiutil.depend import add_dependencies

def write_exposure_qa(filename,fiber_qa_table,petal_qa_table=None) :
    """Writes an exposure-qa fits file.

    Args:
        filename: full path to file
        fiber_qa_table: astropy.table.Table object with one row per target
        petal_qa_table: astropy.table.Table object with one row per petal
    """
    hdus=pyfits.HDUList()
    fiber_qa_table.meta['EXTNAME']='FIBERQA'
    add_dependencies(fiber_qa_table.meta)
    hdus.append(pyfits.convenience.table_to_hdu(fiber_qa_table) )
    if petal_qa_table is not None :
        petal_qa_table.meta['EXTNAME']='PETALQA'
        hdus.append(pyfits.convenience.table_to_hdu(petal_qa_table) )
    outdir = os.path.dirname(filename)
    if not os.path.isdir(outdir) :
        os.makedirs(outdir)

    tmpfile = filename+'.tmp'
    hdus.writeto(tmpfile, overwrite=True, checksum=True)
    os.rename(tmpfile, filename)

def read_exposure_qa(filename) :
    """Reads an exposure-qa fits file.

    Args:
        filename: full path to file
    Returns : fiber_qa_table, petal_qa_table
        two astropy.table.Table objects
    """

    hdus=pyfits.open(filename)
    fiber_qa_table = Table.read(filename,'FIBERQA')
    if 'PETALQA' in hdus :
        petal_qa_table = Table.read(filename,'PETALQA')
    else :
        petal_qa_table = None
    hdus.close()
    return fiber_qa_table , petal_qa_table


def write_tile_qa(filename,fiber_qa_table,petal_qa_table=None) :
    """Writes an tile-qa fits file.

    Args:
        filename: full path to file
        fiber_qa_table: astropy.table.Table object with one row per target
        petal_qa_table: astropy.table.Table object with one row per petal
    """
    hdus=pyfits.HDUList()
    fiber_qa_table.meta['EXTNAME']='FIBERQA'
    add_dependencies(fiber_qa_table.meta)
    hdus.append(pyfits.convenience.table_to_hdu(fiber_qa_table) )
    if petal_qa_table is not None :
        petal_qa_table.meta['EXTNAME']='PETALQA'
        hdus.append(pyfits.convenience.table_to_hdu(petal_qa_table) )
    outdir = os.path.dirname(filename)
    if not os.path.isdir(outdir) :
        os.makedirs(outdir)

    tmpfile = filename+'.tmp'
    hdus.writeto(tmpfile, overwrite=True, checksum=True)
    os.rename(tmpfile, filename)

def read_tile_qa(filename) :
    """Reads an tile-qa fits file.

    Args:
        filename: full path to file
    Returns : fiber_qa_table, petal_qa_table
        two astropy.table.Table objects
    """
    hdus=pyfits.open(filename)
    fiber_qa_table = Table.read(filename,'FIBERQA')
    if 'PETALQA' in hdus :
        petal_qa_table = Table.read(filename,'PETALQA')
    else :
        petal_qa_table = None
    hdus.close()
    return fiber_qa_table , petal_qa_table
