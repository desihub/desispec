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


def compute_tile_completeness_table(exposure_table,specprod_dir) :
    """ Computes a summary table of the observed tiles

    Args:
      exposure_table: astropy.table.Table with exposure summary table from a prod (output of desi_tsnr_afterburner)
      specprod_dir: str, production directory name
    Returns: astropy.table.Table with one row per TILEID, with completeness and exposure time.
    """

    log = get_logger()

    if not "DESI_SURVEYOPS" in os.environ :
        message="need DESI_SURVEYOPS env. variable set"
        log.error(message)
        raise RuntimeError(message)

    survey_ops_dir=os.environ["DESI_SURVEYOPS"]



    default_objective_effective_exptime = 1000.

    search_path="{}/ops/tiles-*.ecsv".format(survey_ops_dir)
    log.info("Searching tiles in {}".format(search_path))
    tiles_filenames = sorted(glob.glob(search_path))
    if len(tiles_filenames)==0 :
        log.warning("No survey config found??")

    input_tiles_tables = []

    for tiles_filename in tiles_filenames :
        log.info("Collecting infos from {}".format(tiles_filename))
        survey=os.path.basename(tiles_filename).split(".")[0].split("-")[-1].lower()
        config_filename="{}/ops/config-{}.yaml".format(survey_ops_dir,survey)
        log.info("Will use config. file {}".format(config_filename))
        input_tiles_table=Table.read(tiles_filename)
        selection=(input_tiles_table["TILEID"]>=0)
        input_tiles_table=input_tiles_table[selection]
        ntiles=len(input_tiles_table)

        if ntiles==0 : continue

        with open(config_filename) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        input_tiles_table["EFFTIME_GOAL"]=np.repeat(default_objective_effective_exptime,ntiles)

        for program in np.unique(input_tiles_table["PROGRAM"]) :
            selection=(input_tiles_table["PROGRAM"]==program)
            print("do something for program",program)

        input_tiles_table["SURVEY"]=np.repeat(survey,ntiles)

        input_tiles_tables.append(input_tiles_table)

    if len(input_tiles_tables)==1 :
        input_tiles_table=input_tiles_tables[0]
    elif len(input_tiles_tables)>1 :
        input_tiles_table=vstack(input_tiles_tables)
    else :
        input_tiles_table=None

    tiles=np.unique(exposure_table["TILEID"])
    ntiles=tiles.size
    res=Table()

    res["TILEID"]=tiles
    res["EXPTIME"]=np.zeros(ntiles)
    res["NEXP"]=np.zeros(ntiles,dtype=int)
    res["ELG_EFFTIME_DARK"]=np.zeros(ntiles)
    res["BGS_EFFTIME_BRIGHT"]=np.zeros(ntiles)
    res["COMPLETENESS"]=np.repeat("UNKNOWN",ntiles)
    res["SURVEY"]=np.repeat("UNKNOWN",ntiles)
    res["PROGRAM"]=np.repeat("UNKNOWN",ntiles)
    res["OBJECTIVE_EFFTIME"]=np.repeat(default_objective_effective_exptime,ntiles)

    if input_tiles_table is not None :
        ii=[]
        jj=[]
        tid2i={tid:i for i,tid in enumerate(input_tiles_table["TILEID"])}
        for j,tid in enumerate(res["TILEID"]) :
            if tid in tid2i :
                ii.append(tid2i[tid])
                jj.append(j)
        for k in ["SURVEY","PROGRAM","OBJECTIVE_EFFTIME"] :
            res[k][jj] = input_tiles_table[k][ii]

    for i,tile in enumerate(tiles) :
        jj=(exposure_table["TILEID"]==tile)
        res["NEXP"][i]=np.sum(jj)
        for k in ["EXPTIME","ELG_EFFTIME_DARK","BGS_EFFTIME_BRIGHT"] :
            res[k][i] = np.sum(exposure_table[k][jj])


    # trivial completeness for now (all of the work for this?)

    efftime_keyword_per_program = {}
    efftime_keyword_per_program["DARK"]="ELG_EFFTIME_DARK"
    efftime_keyword_per_program["BRIGHT"]="BGS_EFFTIME_BRIGHT"
    efftime_keyword_per_program["BACKUP"]="BGS_EFFTIME_BACKUP"
    efftime_keyword_per_program["UNKNOWN"]="ELG_EFFTIME_DARK"

    for program in efftime_keyword_per_program :
        selection=(res["PROGRAM"]==program)
        if np.sum(selection)==0 : continue
        efftime_keyword=efftime_keyword_per_program[program]
        efftime=res[efftime_keyword]
        done=selection&(efftime>res["OBJECTIVE_EFFTIME"])
        res["COMPLETENESS"][done]="DONE"
        partial=selection&(efftime<res["OBJECTIVE_EFFTIME"])
        res["COMPLETENESS"][partial]="PARTIAL"

    return res

def merge_tile_completeness_table(previous_table,new_table) :
    """ Merges tile summary tables. Entries with tiles previously marked as DONE are not modified.

    Args:
      previous_table: astropy.table.Table
      new_table: astropy.table.Table
    Returns: astropy.table.Table with merged entries.
    """
    # do not change the status of a DONE tile ; it's too late

    keep_from_previous = (previous_table["COMPLETENESS"]=="DONE") | (~np.in1d(previous_table["TILEID"],new_table["TILEID"]))
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
