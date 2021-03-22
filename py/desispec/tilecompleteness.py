"""
desispec.tilecompleteness.py
========================

Routines to determine the survey progress
and tiles completion.
"""

import os,sys
import numpy as np
import yaml
import glob
from astropy.table import Table,vstack

from desiutil.log import get_logger


def compute_tile_completeness_table(exposure_table,specprod_dir,auxiliary_table_filenames) :
    """ Computes a summary table of the observed tiles

    Args:
      exposure_table: astropy.table.Table with exposure summary table from a prod (output of desi_tsnr_afterburner)
      specprod_dir: str, production directory name
    Returns: astropy.table.Table with one row per TILEID, with completeness and exposure time.
    """

    log = get_logger()

    default_goaltime = 1000. # objective effective time in seconds





    tiles=np.unique(exposure_table["TILEID"])
    ntiles=tiles.size
    res=Table()

    res["TILEID"]=tiles
    res["EXPTIME"]=np.zeros(ntiles)
    res["NEXP"]=np.zeros(ntiles,dtype=int)
    res["ELG_EFFTIME_DARK"]=np.zeros(ntiles)
    res["BGS_EFFTIME_BRIGHT"]=np.zeros(ntiles)
    res["OBSSTATUS"] = np.array(np.repeat("UNKNOWN",ntiles),dtype='<U16')
    res["ZSTATUS"]   = np.array(np.repeat("NONE",ntiles),dtype='<U16')
    res["SURVEY"]    = np.array(np.repeat("UNKNOWN",ntiles),dtype='<U16')
    res["GOALTYP"]   = np.array(np.repeat("UNKNOWN",ntiles),dtype='<U16')
    res["TARGETS"]   = np.array(np.repeat("UNKNOWN",ntiles),dtype='<U16')
    res["FAFLAVOR"]   = np.array(np.repeat("UNKNOWN",ntiles),dtype='<U16')
    res["GOALTIME"]  = np.zeros(ntiles)

    # case is /global/cfs/cdirs/desi/survey/observations/SV1/sv1-tiles.fits
    if auxiliary_table_filenames is not None :
        for filename in auxiliary_table_filenames :

            if filename.find("sv1-tiles")>=0 :
                log.info("Use SV1 tiles information from {}".format(filename))
                table=Table.read(filename)
                ii=[]
                jj=[]
                tid2i={tid:i for i,tid in enumerate(table["TILEID"])}
                for j,tid in enumerate(res["TILEID"]) :
                    if tid in tid2i :
                        ii.append(tid2i[tid])
                        jj.append(j)

                res["SURVEY"][jj]="SV1"
                res["TARGETS"][jj]=table["TARGETS"][ii]

                is_dark   = [(targets.find("ELG")>=0)|(targets.find("LRG")>=0)|(targets.find("QSO")>=0) for targets in res["TARGETS"]]
                is_bright = [(targets.find("BGS")>=0)|(targets.find("MWS")>=0) for targets in res["TARGETS"]]
                is_backup = [(targets.find("BACKUP")>=0) for targets in res["TARGETS"]]

                res["GOALTYP"][is_dark]   = "DARK"
                res["GOALTYP"][is_bright] = "BRIGHT"
                res["GOALTYP"][is_backup] = "BACKUP"

                # 4 times nominal exposure time for DARK and BRIGHT
                res["GOALTIME"][res["GOALTYP"]=="DARK"]   = 4*1000.
                res["GOALTIME"][res["GOALTYP"]=="BRIGHT"] = 4*150.
                res["GOALTIME"][res["GOALTYP"]=="BACKUP"] = 30.

            else :
                log.warning("Sorry I don't know what to do with {}".format(filename))

    # test default
    res["GOALTIME"][res["GOALTIME"]==0] = default_goaltime

    for i,tile in enumerate(tiles) :
        jj=(exposure_table["TILEID"]==tile)
        res["NEXP"][i]=np.sum(jj)
        for k in ["EXPTIME","ELG_EFFTIME_DARK","BGS_EFFTIME_BRIGHT"] :
            res[k][i] = np.sum(exposure_table[k][jj])

        # copy the following from the exposure table if it exists
        for k in ["SURVEY","GOALTYP","FAFLAVOR"] :
            if k in exposure_table.dtype.names :
                val = exposure_table[k][jj][0]
                if val != "UNKNOWN" :
                    res[k][i] = val # force consistency
        k = "GOALTIME"
        if k in exposure_table.dtype.names :
            val = exposure_table[k][jj][0]
            if val > 0. :
                res[k][i] = val # force consistency

    # truncate number of digits for exposure times to 0.1 sec
    for k in res.dtype.names :
        if k.find("EXPTIME")>=0 or k.find("EFFTIME")>=0 :
            res[k] = np.around(res[k],1)

    # trivial completeness for now (all of this work for this?)
    efftime_keyword_per_goaltyp = {}
    efftime_keyword_per_goaltyp["DARK"]="ELG_EFFTIME_DARK"
    efftime_keyword_per_goaltyp["BRIGHT"]="BGS_EFFTIME_BRIGHT"
    efftime_keyword_per_goaltyp["BACKUP"]="BGS_EFFTIME_BRIGHT"
    efftime_keyword_per_goaltyp["UNKNOWN"]="ELG_EFFTIME_DARK"

    for program in efftime_keyword_per_goaltyp :
        selection=(res["GOALTYP"]==program)
        if np.sum(selection)==0 : continue
        efftime_keyword=efftime_keyword_per_goaltyp[program]
        efftime=res[efftime_keyword]
        done=selection&(efftime>res["GOALTIME"])
        res["OBSSTATUS"][done]="OBSDONE"
        partial=selection&(efftime<res["GOALTIME"])
        res["OBSSTATUS"][partial]="OBSSTART"

    return res

def merge_tile_completeness_table(previous_table,new_table) :
    """ Merges tile summary tables. Entries with tiles previously marked as ZDONE are not modified.

    Args:
      previous_table: astropy.table.Table
      new_table: astropy.table.Table
    Returns: astropy.table.Table with merged entries.
    """
    # do not change the status of a ZDONE tile ; it's too late

    keep_from_previous = (previous_table["ZSTATUS"]=="ZDONE") | (~np.in1d(previous_table["TILEID"],new_table["TILEID"]))
    exclude_from_new   = np.in1d(new_table["TILEID"],previous_table["TILEID"][keep_from_previous])
    add_from_new = ~exclude_from_new
    log = get_logger()
    if np.sum(exclude_from_new)>0 :
        log.info("do not change the status of {} completed tiles".format(np.sum(exclude_from_new)))
    if np.sum(add_from_new)>0 :
        log.info("add or change the status of {} tiles".format(np.sum(add_from_new)))
        return vstack( [ previous_table[keep_from_previous] , new_table[add_from_new] ] )
    else :
        return previous_table
