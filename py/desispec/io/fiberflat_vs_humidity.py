import numpy as np
import fitsio
import astropy.io.fits as fits
from desiutil.log import get_logger
from .meta import findfile
from .util import native_endian

def get_humidity(night,expid,camera) :
    log=get_logger()
    raw_filename=findfile("raw",night=night,expid=expid)
    table=fitsio.read(raw_filename,"SPECTCONS")
    keyword="{}HUMID".format(camera[0].upper())
    if keyword not in table.dtype.names :
        log.warning(f"no column {keyword} in SPECTCONS table of file {raw_filename}, returning nan")
        return np.nan

    unit=int(camera[1])
    selection=(table["unit"]==unit)
    if np.sum(selection)==0 :
        log.warning("no unit '{}' in '{}'".format(unit,raw_filename))
        return np.nan
    humidity=float(table[keyword][selection][0])
    log.debug(f"NIGHT={night} EXPID={expid} CAM={camera} HUMIDITY={humidity}")
    if humidity <= 0 or humidity >= 100:
        log.warning(f"Unphysical humidity value={humidity}, in column {keyword} in SPECTCONS table of file {raw_filename} for unit={unit}")
    return humidity

def read_fiberflat_vs_humidity(filename):
    """Read fiberflat vs humidity from filename

    Args:
        filename (str): path to fiberflat_vs_humidity file

    Returns: fiberflat , humidity , wave
        fiberflat is 3D [nhumid, nspec, nwave]
        humidity is 1D [nhumid] (and in percent)
        wave is 1D [nwave] (and in Angstrom)
        header (fits header)
    """

    with fits.open(filename, uint=True, memmap=False) as fx:
        header = fx[0].header
        wave  = native_endian(fx["WAVELENGTH"].data.astype('f8'))
        fiberflat = list()
        humidity  = list()
        for index in range(100) :
             hdu="HUM{:02d}".format(index)
             if hdu not in fx : continue
             fiberflat.append(native_endian(fx[hdu].data.astype('f8')))
             humidity.append(fx[hdu].header["MEDHUM"])

    humidity  = np.array(humidity)
    fiberflat = np.array(fiberflat)
    assert(fiberflat.shape[0] == humidity.size)
    assert(fiberflat.shape[2] == wave.size)
    return fiberflat , humidity , wave, header
