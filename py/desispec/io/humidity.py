import numpy as np
import fitsio
from desiutil.log import get_logger
from .meta import findfile

def get_humidity(night,expid,camera) :
    log=get_logger()
    raw_filename=findfile("raw",night=night,expid=expid)
    table=fitsio.read(raw_filename,"SPECTCONS")
    keyword="{}HUMID".format(camera[0].upper())
    unit=int(camera[1])
    selection=(table["unit"]==unit)
    if np.sum(selection)==0 :
        log.warning("no unit '{}' in '{}'".format(unit,raw_filename))
        return np.nan
    humidity=float(table[keyword][selection][0])
    log.debug(f"NIGHT={night} EXPID={expid} CAM={camera} HUMIDITY={humidity}")
    return humidity
