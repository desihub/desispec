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
    res["EFFTIME_ETC"]=np.zeros(ntiles)
    res["EFFTIME_SPEC"]=np.zeros(ntiles)
    res["ELG_EFFTIME_DARK"]=np.zeros(ntiles)
    res["BGS_EFFTIME_BRIGHT"]=np.zeros(ntiles)
    res["LYA_EFFTIME_DARK"]=np.zeros(ntiles)
    res["OBSSTATUS"] = np.array(np.repeat("unknown",ntiles),dtype='<U16')
    res["ZSTATUS"]   = np.array(np.repeat("none",ntiles),dtype='<U16')
    res["SURVEY"]    = np.array(np.repeat("unknown",ntiles),dtype='<U16')
    res["GOALTYPE"]   = np.array(np.repeat("unknown",ntiles),dtype='<U16')
    res["MINTFRAC"]   = np.array(np.repeat(0.9,ntiles),dtype=float)
    res["FAFLAVOR"]   = np.array(np.repeat("unknown",ntiles),dtype='<U16')
    res["GOALTIME"]  = np.zeros(ntiles)

    # case is /global/cfs/cdirs/desi/survey/observations/SV1/sv1-tiles.fits
    if auxiliary_table_filenames is not None :
        for filename in auxiliary_table_filenames :

            if filename.find("sv1-tiles")>=0 :

                targets = np.array(np.repeat("unknown",ntiles),dtype='<U16')

                log.info("Use SV1 tiles information from {}".format(filename))
                table=Table.read(filename)
                ii=[]
                jj=[]
                tid2i={tid:i for i,tid in enumerate(table["TILEID"])}
                for j,tid in enumerate(res["TILEID"]) :
                    if tid in tid2i :
                        ii.append(tid2i[tid])
                        jj.append(j)

                res["SURVEY"][jj]="sv1"
                targets[jj]=table["TARGETS"][ii]

                is_dark   = [(t.lower().find("elg")>=0)|(t.lower().find("lrg")>=0)|(t.lower().find("qso")>=0) for t in targets]
                is_bright = [(t.lower().find("bgs")>=0)|(t.lower().find("mws")>=0) for t in targets]
                is_backup = [(t.lower().find("backup")>=0) for t in targets]

                res["GOALTYPE"][is_dark]   = "dark"
                res["GOALTYPE"][is_bright] = "bright"
                res["GOALTYPE"][is_backup] = "backup"

                # 4 times nominal exposure time for DARK and BRIGHT
                res["GOALTIME"][res["GOALTYPE"]=="dark"]   = 4*1000.
                res["GOALTIME"][res["GOALTYPE"]=="bright"] = 4*150.
                res["GOALTIME"][res["GOALTYPE"]=="backup"] = 30.

            else :
                log.warning("Sorry I don't know what to do with {}".format(filename))

    # test default
    res["GOALTIME"][res["GOALTIME"]==0] = default_goaltime

    for i,tile in enumerate(tiles) :
        jj=(exposure_table["TILEID"]==tile)
        res["NEXP"][i]=np.sum(jj)
        for k in ["EXPTIME","ELG_EFFTIME_DARK","BGS_EFFTIME_BRIGHT","LYA_EFFTIME_DARK","EFFTIME_ETC"] :
            if k in exposure_table.dtype.names :
                res[k][i] = np.sum(exposure_table[k][jj])
                if k == "EFFTIME_ETC" :
                    if np.any(exposure_table[k][jj]==0) : res[k][i]=0 # because we are missing data

        # copy the following from the exposure table if it exists
        for k in ["SURVEY","GOALTYPE","FAFLAVOR"] :
            if k in exposure_table.dtype.names :
                val = exposure_table[k][jj][0]
                if val != "unknown" :
                    res[k][i] = val # force consistency
        for k in ["GOALTIME","MINTFRAC"] :
            if k in exposure_table.dtype.names :
                val = exposure_table[k][jj][0]
                if val > 0. :
                    res[k][i] = val # force consistency

    # truncate number of digits for exposure times to 0.1 sec
    for k in res.dtype.names :
        if k.find("EXPTIME")>=0 or k.find("EFFTIME")>=0 :
            res[k] = np.around(res[k],1)

    # default efftime is ELG_EFFTIME_DARK
    res["EFFTIME_SPEC"]=res["ELG_EFFTIME_DARK"]

    # trivial completeness for now (all of this work for this?)
    efftime_keyword_per_goaltype = {}
    efftime_keyword_per_goaltype["bright"]="BGS_EFFTIME_BRIGHT"
    efftime_keyword_per_goaltype["backup"]="BGS_EFFTIME_BRIGHT"

    ii=((res["GOALTYPE"]=="bright")|(res["GOALTYPE"]=="backup"))
    res["EFFTIME_SPEC"][ii]=res["BGS_EFFTIME_BRIGHT"][ii]

    done=(res["EFFTIME_SPEC"]>res["MINTFRAC"]*res["GOALTIME"])
    res["OBSSTATUS"][done]="obsend"
    partial=(res["EFFTIME_SPEC"]>0.)&(res["EFFTIME_SPEC"]<=res["MINTFRAC"]*res["GOALTIME"])
    res["OBSSTATUS"][partial]="obsstart"

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
