"""
desispec.io.image
=================

I/O routines for Image objects
"""

import os
import gzip
import shutil
import subprocess
import time
import warnings
import numpy as np

from desispec.image import Image
from desispec.io.util import fitsheader, native_endian, makepath
from . import iotime
from .util import checkgzip, get_tempfilename
from .fibermap import annotate_fibermap
from astropy.io import fits
from desiutil.depend import add_dependencies
from desiutil.log import get_logger
from astropy.table import Table

def write_image(outfile, image, meta=None, ncpu=1):
    """Writes image object to outfile

    Args:
        outfile : output file string
        image : desispec.image.Image (or any object with 2D array attributes image, ivar, mask)

    Optional:
        meta : dict-like object with metadata key/values (e.g. FITS header)
        ncpu : number of threads for parallel gzip compression

    Notes:
        parallel compression only if ncpu>1, `pigz` is installed, and outfile ends with .gz
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
    tmpfile = get_tempfilename(outfile)
    if tmpfile.endswith('.gz') and ncpu > 1 and shutil.which('pigz') is not None:
        #- stream FITS directly into multithreaded pigz without an uncompressed
        #- intermediate file; use level 1 to match the single-core gzip path below
        cmd = ['pigz', '-1', '-c', '-p', str(ncpu)]
        try:
            with open(tmpfile, 'wb') as fx:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=fx)
                write_error = None
                try:
                    hx.writeto(proc.stdin, checksum=True)
                except BaseException as err:
                    #- if pigz exited early, astropy may report the broken pipe
                    #- as a different exception type, so check pigz exit code below
                    write_error = err
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
                returncode = proc.wait()

            if returncode != 0:
                msg = f'{" ".join(cmd)} failed with exit code {returncode} writing {tmpfile}'
                log.error(msg)
                raise RuntimeError(msg) from write_error
            elif write_error is not None:
                raise write_error
        except BaseException:
            #- don't leave a partial file behind
            if os.path.exists(tmpfile):
                os.remove(tmpfile)
            raise
    elif tmpfile.endswith('.gz'):
        #- astropy gzips at zlib level 9, which is ~2x slower than level 1
        #- for <1% larger files of noisy float32 data
        with gzip.GzipFile(tmpfile, 'wb', compresslevel=1) as fx:
            hx.writeto(fx, overwrite=True, checksum=True)
    else:
        hx.writeto(tmpfile, overwrite=True, checksum=True)
    os.rename(tmpfile, outfile)
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
