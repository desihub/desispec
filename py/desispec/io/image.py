"""
desispec.io.image
=================

I/O routines for Image objects
"""

import os
import time
import warnings
import numpy as np

from desispec.image import Image
from desispec.io.util import fitsheader, native_endian, makepath
from . import iotime
from .util import checkgzip, get_tempfilename, write_hdulist
from .fibermap import annotate_fibermap
from astropy.io import fits
from desiutil.depend import add_dependencies
from desiutil.log import get_logger
from astropy.table import Table

def write_image(outfile, image, meta=None, ncpu=1, cpus=None):
    """Writes image object to outfile

    Args:
        outfile (str): output file string
        image (desispec.image.Image): image to write, or any object with 2D
            array attributes image, ivar, mask
        meta (dict-like, optional): metadata key/values (e.g. FITS header)
        ncpu (int, optional): number of threads for parallel gzip compression
        cpus (list, optional): CPU ids for parallel gzip compression, one thread
            per unique CPU id, overriding ncpu if non-empty (a single CPU id
            means single-threaded gzip), e.g. CPUs of other MPI ranks from
            ``desispec.parallel.gather_cpu_affinity(comm)``.  Those ranks must
            be idle while writing, e.g. in
            :func:`desispec.parallel.polling_barrier` rather than a blocking
            ``comm.barrier()``, which may busy-poll their CPUs.

    Notes:
        parallel compression only if ncpu>1 (or cpus has >1 unique CPU id), ``pigz`` is installed,
        and outfile ends with .gz; the number of pigz threads is capped at
        desispec.io.util.MAX_PIGZ_THREADS.
        If pigz fails, an error is logged and the file is written with single-threaded gzip.
        See :func:`desispec.io.util.write_hdulist` for details.
    """

    log = get_logger()
    if meta is not None:
        hdr = fitsheader(meta)
    else:
        hdr = fitsheader(image.meta)

    add_dependencies(hdr)

    #- Work around fitsio>1.0 writing blank keywords, e.g. on 20191212
    for key in hdr.keys():
        if type(hdr[key]) == fits.card.Undefined:
            log.warning('Setting blank keyword {} to None'.format(key))
            hdr[key] = None

    outdir = os.path.dirname(os.path.abspath(outfile))
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    hx = fits.HDUList()
    hdu = fits.ImageHDU(image.pix.astype(np.float32), name='IMAGE', header=hdr)
    if 'CAMERA' not in hdu.header:
        hdu.header.append( ('CAMERA', image.camera.lower(), 'Spectrograph Camera') )

    if 'RDNOISE' not in hdu.header and np.isscalar(image.readnoise):
        hdu.header.append( ('RDNOISE', image.readnoise, 'Read noise [RMS electrons/pixel]'))

    hx.append(hdu)
    hx.append(fits.ImageHDU(image.ivar.astype(np.float32), name='IVAR'))
    hx.append(fits.CompImageHDU(image.mask.astype(np.int16), name='MASK'))
    if not np.isscalar(image.readnoise):
        hx.append(fits.ImageHDU(image.readnoise.astype(np.float32), name='READNOISE'))

    if hasattr(image, 'fibermap'):
        if isinstance(image.fibermap, Table):
            with warnings.catch_warnings():
                #- nanomaggies aren't an official IAU unit but don't complain
                warnings.filterwarnings('ignore', ".*nanomaggies.*")
                warnings.filterwarnings('ignore', ".*nmgy.*")
                fmhdu = fits.convenience.table_to_hdu(image.fibermap)
            fmhdu.name = 'FIBERMAP'
        else:
            fmhdu = fits.BinTableHDU(image.fibermap, name='FIBERMAP')
        fmhdu = annotate_fibermap(fmhdu)
        hx.append(fmhdu)

    t0 = time.time()
    write_hdulist(outfile, hx, ncpu=ncpu, cpus=cpus)
    duration = time.time() - t0
    log.info(iotime.format('write', outfile, duration))

    return outfile

def read_image(filename, skip=None):
    """
    Returns desispec.image.Image object from input file

    Options:
        skip: iterable of case-insensitive HDU names to skip, e.g. ('ivar', 'mask', 'readnoise')

    Returns: desispec.image.Image object
    """
    log = get_logger()
    filename = checkgzip(filename)
    skip = [s.upper() for s in skip] if skip is not None else []
    t0 = time.time()
    with fits.open(filename, uint=True, memmap=False) as fx:
        image = native_endian(fx['IMAGE'].data).astype(np.float64)
        ivar = native_endian(fx['IVAR'].data).astype(np.float64) if 'IVAR' not in skip else None
        mask = native_endian(fx['MASK'].data).astype(np.uint16) if 'MASK' not in skip else None
        camera = fx['IMAGE'].header['CAMERA'].lower()
        meta = fx['IMAGE'].header

        if 'READNOISE' in skip:
            readnoise = None
        else:
            if 'READNOISE' in fx:
                readnoise = native_endian(fx['READNOISE'].data).astype(np.float64)
            else:
                readnoise = fx['IMAGE'].header['RDNOISE']

    duration = time.time() - t0
    log.info(iotime.format('read', filename, duration))

    return Image(image, ivar, mask=mask, readnoise=readnoise,
                 camera=camera, meta=meta)
