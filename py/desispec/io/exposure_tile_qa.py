"""
desispec.io.exposure_qa
=================

I/O routines for exposure and tile qa
"""

import os
import astropy.io.fits as pyfits



def write_exposure_qa(filename,fiber_qa_table,petal_qa_table=None) :
    hdus=pyfits.HDUList()
    fiber_qa_table.meta['EXTNAME']='FIBERQA'
    hdus.append(pyfits.convenience.table_to_hdu(fiber_qa_table) )
    if petal_qa_table is not None :
        petal_qa_table.meta['EXTNAME']='PETALQA'
        hdus.append(pyfits.convenience.table_to_hdu(petal_qa_table) )
    outdir = os.path.dirname(filename)
    if not os.path.isdir(outdir) :
        os.makedirs(outdir)
    hdus.writeto(filename, overwrite=True, checksum=True)

def read_exposure_qa(filename) :
    hdus=pyfits.open(filename)

    fiber_qa_table = hdus['FIBERQA'].data
    if 'PETALQA' in hdus :
        petal_qa_table = hdus['PETALQA'].data
    else :
        petal_qa_table = None
    return fiber_qa_table , petal_qa_table


def write_tile_qa(filename,fiber_qa_table,petal_qa_table=None) :
    hdus=pyfits.HDUList()
    fiber_qa_table.meta['EXTNAME']='FIBERQA'
    hdus.append(pyfits.convenience.table_to_hdu(fiber_qa_table) )
    if petal_qa_table is not None :
        petal_qa_table.meta['EXTNAME']='PETALQA'
        hdus.append(pyfits.convenience.table_to_hdu(petal_qa_table) )
    outdir = os.path.dirname(filename)
    if not os.path.isdir(outdir) :
        os.makedirs(outdir)
    hdus.writeto(filename, overwrite=True, checksum=True)

def read_tile_qa(filename) :
    hdus=pyfits.open(filename)

    fiber_qa_table = hdus['FIBERQA'].data
    if 'PETALQA' in hdus :
        petal_qa_table = hdus['PETALQA'].data
    else :
        petal_qa_table = None
    return fiber_qa_table , petal_qa_table
