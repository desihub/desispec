"""
desispec.exposure_qa
============

Utility functions to compute an exposure QA scores.
"""

import os,sys
import numpy as np
from astropy.table import Table,vstack
import fitsio
import yaml
from pkg_resources import resource_filename
import glob

from desiutil.log import get_logger

from desispec.exposure_qa import compute_exposure_qa
from desispec.io import read_fibermap,findfile,read_exposure_qa,write_exposure_qa
from desispec.maskbits import fibermask

"""
# only read it once per process
_qa_params = None
def get_qa_params() :
    global _qa_params
    if _qa_params is None :
        param_filename =resource_filename('desispec', 'data/qa/qa-params.yaml')
        with open(param_filename) as f:
            _qa_params = yaml.safe_load(f)
    return _qa_params
"""
def mystack(tables) :

    if len(tables)==1 :
        return tables[0]

    log = get_logger()
    stack=Table()
    length=None
    for key in tables[0].dtype.names :
        vals=[]
        for table in tables :
            vals.append(table[key])
        vals=np.hstack(vals)
        if length is not None :
            if vals.size != length :
                log.warning("skip column {}".format(key))
                continue
        stack[key]=vals
        length=vals.size
    return stack

def compute_tile_qa(night, tileid, specprod_dir, exposure_qa_dir=None):
    """
    Computes the exposure_qa
    Args:
       night: int, YYYYMMDD
       tileid: int, tile id
       specprod_dir: str, specify the production directory.
                     default is $DESI_SPECTRO_REDUX/$SPECPROD
       exposure_qa_dir: str, optional, directory where the exposure qa are saved
    returns an astropy.table.Table with one row per target and at least a TARGETID column
    """

    log=get_logger()


    #qa_params=get_qa_params()["tile_qa"]

    # get list of exposures used for the tile
    tiledir=f"{specprod_dir}/tiles/cumulative/{tileid:d}/{night}"
    spectra_files=sorted(glob.glob(f"{tiledir}/spectra-*-{tileid:d}-thru{night}.fits"))
    if len(spectra_files)==0 :
        log.error("no spectra files in "+tileid)
        return None, None

    fmap=read_fibermap(spectra_files[0])
    expids=np.unique(fmap["EXPID"])
    lexpids=list(expids)
    log.info(f"for tile={tileid} night={night} expids={lexpids}")

    if exposure_qa_dir is None :
        exposure_qa_dir = specprod_dir

    exposure_fiberqa_tables = []
    exposure_petalqa_tables = []
    for expid in expids :
        exposure_night = (fmap["NIGHT"][fmap["EXPID"]==expid][0])
        filename=findfile("exposureqa",night=exposure_night,expid=expid,specprod_dir=exposure_qa_dir)
        if not os.path.isfile(filename) :
            log.info("running missing exposure qa")
            exposure_fiberqa_table , exposure_petalqa_table = compute_exposure_qa(night, expid, specprod_dir)
            if exposure_fiberqa_table is not None :
                write_exposure_qa(filename, exposure_fiberqa_table , exposure_petalqa_table)
            else :
                log.warning("failed to compute exposure qa")
                continue
        else :
            log.info(f"reading {filename}")
            exposure_fiberqa_table , exposure_petalqa_table = read_exposure_qa(filename)
        exposure_fiberqa_tables.append(exposure_fiberqa_table)
        exposure_petalqa_tables.append(exposure_petalqa_table)

    if len(exposure_fiberqa_tables)==0 :
        log.error(f"no exposure qa data for tile {tile}")
        return None, None

    # stack qa tables
    if len(expids) > 1 :
        exposure_fiberqa_tables = mystack(exposure_fiberqa_tables)
        exposure_petalqa_tables = mystack(exposure_petalqa_tables)
    else :
        exposure_fiberqa_tables = exposure_fiberqa_tables[0]
        exposure_petalqa_tables = exposure_petalqa_tables[0]

    # collect fibermaps and scores of all coadds
    coadd_files=sorted(glob.glob(f"{tiledir}/coadd-*-{tileid:d}-thru{night}.fits"))
    zbest_files=sorted(glob.glob(f"{tiledir}/coadd-*-{tileid:d}-thru{night}.fits"))
    fibermaps=[]
    scores=[]
    zbests=[]
    for coadd_file in coadd_files :
        log.info("reading {}".format(coadd_file))
        fibermaps.append(Table.read(coadd_file,"FIBERMAP"))
        scores.append(Table.read(coadd_file,"SCORES"))
        zbest_file = coadd_file.replace("coadd","zbest")
        log.info("reading {}".format(zbest_file))
        zbests.append(Table.read(zbest_file,"ZBEST"))
    log.info("stacking")
    fibermap=mystack(fibermaps)
    scores=mystack(scores)
    zbests=mystack(zbests)
    targetids=fibermap["TARGETID"]

    tile_fiberqa_table = Table()
    for k in ['TARGETID','PETAL_LOC','DEVICE_LOC', 'LOCATION', 'FIBER', 'TARGET_RA', 'TARGET_DEC', 'FIBER_X', 'FIBER_Y', 'DELTA_X', 'DELTA_Y','DESI_TARGET', 'BGS_TARGET'] :
        if k in fibermap.dtype.names :
            tile_fiberqa_table[k]=fibermap[k]

    # add TSNR info
    scores_tid_to_index = {tid:index for index,tid in enumerate(scores["TARGETID"])}
    tsnr2_key="TSNR2_LRG"
    if tsnr2_key in scores.dtype.names :
        tile_fiberqa_table[tsnr2_key] = np.zeros(targetids.size)
        for i,tid in enumerate(targetids) :
            if tid in scores_tid_to_index :
                tile_fiberqa_table[tsnr2_key][i] = scores[tsnr2_key][scores_tid_to_index[tid]]

    # add ZBEST info
    zbest_tid_to_index = {tid:index for index,tid in enumerate(zbests["TARGETID"])}
    keys=["Z","SPECTYPE","DELTACHI2"]
    for k in keys :
        tile_fiberqa_table[k] = np.zeros(targetids.size,dtype=zbests[k].dtype)
    zbest_ii=[]
    fiberqa_ii=[]
    for i,tid in enumerate(targetids) :
        if tid in zbest_tid_to_index :
            zbest_ii.append(zbest_tid_to_index[tid])
            fiberqa_ii.append(i)
    for k in keys :
        tile_fiberqa_table[k][fiberqa_ii] = zbests[k][zbest_ii]



    # QAFIBERSTATUS is OR of input exposures
    tile_fiberqa_table["QAFIBERSTATUS"]=np.zeros(targetids.size,dtype=exposure_fiberqa_tables["QAFIBERSTATUS"].dtype)
    for i,tid in enumerate(targetids) :
        jj = (exposure_fiberqa_tables["TARGETID"]==tid)
        tile_fiberqa_table["QAFIBERSTATUS"][i] = np.bitwise_or.reduce(exposure_fiberqa_tables['QAFIBERSTATUS'][jj])

    bad_fibers_mask=fibermask.mask("STUCKPOSITIONER|BROKENFIBER|RESTRICTED|MISSINGPOSITION|BADPOSITION|POORPOSITION")
    good_fibers = np.where((tile_fiberqa_table['QAFIBERSTATUS']&bad_fibers_mask)==0)[0]
    good_petals = np.unique(tile_fiberqa_table['PETAL_LOC'][good_fibers])

    tile_petalqa_table = Table()
    petals=np.unique(exposure_fiberqa_tables["PETAL_LOC"])
    tile_petalqa_table["PETAL_LOC"]=petals

    # add meta info
    tile_fiberqa_table.meta["TILEID"]=tileid
    tile_fiberqa_table.meta["NIGHT"]=night
    tile_fiberqa_table.meta["NGOODFIBERS"]=good_fibers.size
    tile_fiberqa_table.meta["NGOODPETALS"]=good_petals.size



    return tile_fiberqa_table ,tile_petalqa_table
