'''
desispec.io.ctecorr
===============

I/O for CTE corrections
'''

import numpy as np
import os.path
from desiutil.log import get_logger
from desispec.io import findfile
from astropy.table import Table

def get_cte_corr(night,camera,amplifier,sector) :
    log = get_logger()
    filename = findfile('ctecorrnight', night=night, camera=camera)
    log.debug(f"Looking for file {filename}")
    if not os.path.isfile(filename) :
        log.error(f"No CTE file {filename}")
        return None
    table=Table.read(filename)
    log.debug(str(table))
    selection=(table["NIGHT"]==night)&(table["CAMERA"]==camera)&(table["AMPLIFIER"]==amplifier)&(table["SECTOR"]==sector)
    if np.sum(selection)==0 :
        log.warning(f"No CTE correction for {night},{camera},{amplifier},{sector}")
        return None
    i=np.where(selection)[0][0]
    res={}
    for k in ["AMPLITUDE","FRACLEAK"] :
        res[k]=table[k][i]
    log.info(f"CTE correction params={res}")
    return res
