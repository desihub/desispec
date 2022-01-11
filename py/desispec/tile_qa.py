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

from desispec.exposure_qa import compute_exposure_qa,get_qa_params
from desispec.io import read_fibermap,findfile,read_exposure_qa,write_exposure_qa
from desispec.io.util import replace_prefix
from desispec.maskbits import fibermask


def _rm_meta_keywords(table) :

    if table.meta is not None :
        for k in ['CHECKSUM','DATASUM'] :
            if k in table.meta : table.meta.pop(k) # otherwise WARNING: MergeConflictWarning
    return table

def compute_tile_qa(night, tileid, specprod_dir, exposure_qa_dir=None, group='cumulative'):
    """
    Computes the exposure_qa
    Args:
       night: int, YYYYMMDD
       tileid: int, tile id
       specprod_dir: str, specify the production directory.
                     default is $DESI_SPECTRO_REDUX/$SPECPROD
       exposure_qa_dir: str, optional, directory where the exposure qa are saved
       group: str, "cumulative" or "pernight" tile group
    returns two tables (astropy.table.Table), fiberqa (with one row per target and at least a TARGETID column)
            and petalqa (with one row per petal and at least a PETAL_LOC column)
    """

    log=get_logger()

    # get list of exposures used for the tile
    tiledir=f"{specprod_dir}/tiles/{group}/{tileid:d}/{night}"
    spectra_files=sorted(glob.glob(f"{tiledir}/spectra-*-{tileid:d}-*{night}.fits"))
    if len(spectra_files)==0 :
        log.error(f"no spectra files in {tiledir}")
        return None, None

    fmap=read_fibermap(spectra_files[0])
    expids=np.unique(fmap["EXPID"])
    lexpids=list(expids)
    log.info(f"for tile={tileid} night={night} expids={lexpids}")

    if exposure_qa_dir is None :
        exposure_qa_dir = specprod_dir

    exposure_qa_meta = None

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
                log.info("wrote {}".format(filename))
            else :
                log.warning("failed to compute exposure qa")
                continue
        else :
            log.info(f"reading {filename}")
            exposure_fiberqa_table , exposure_petalqa_table = read_exposure_qa(filename)
        if exposure_qa_meta is None :
            exposure_qa_meta = exposure_fiberqa_table.meta
        else :
            exposure_fiberqa_table.meta=None # otherwise MergeConflictWarning
            exposure_petalqa_table.meta=None # otherwise MergeConflictWarning
        exposure_fiberqa_tables.append(_rm_meta_keywords(exposure_fiberqa_table))
        exposure_petalqa_tables.append(_rm_meta_keywords(exposure_petalqa_table))

    if len(exposure_fiberqa_tables)==0 :
        log.error(f"no exposure qa data for tile {tile}")
        return None, None

    # stack qa tables
    if len(expids) > 1 :
        exposure_fiberqa_tables = vstack(exposure_fiberqa_tables)
        exposure_petalqa_tables = vstack(exposure_petalqa_tables)
    else :
        exposure_fiberqa_tables = exposure_fiberqa_tables[0]
        exposure_petalqa_tables = exposure_petalqa_tables[0]

    # collect fibermaps and scores of all coadds
    coadd_files=sorted(glob.glob(f"{tiledir}/coadd-*-{tileid:d}-*{night}.fits"))

    fibermaps=[]
    scores=[]
    redshifts=[]
    ### zqsos=[]
    for coadd_file in coadd_files :
        log.info("reading {}".format(coadd_file))
        fibermaps.append(_rm_meta_keywords(Table.read(coadd_file,"FIBERMAP")))
        scores.append(_rm_meta_keywords(Table.read(coadd_file,"SCORES")))

        redrock_file = replace_prefix(coadd_file, "coadd", "redrock")
        extname="REDSHIFTS"
        if not os.path.isfile(redrock_file) :
            zbest_file = replace_prefix(coadd_file, "coadd", "zbest")
            if os.path.isfile(zbest_file) :
                log.warning("switch to zbest file {}".format(zbest_file))
                redrock_file = zbest_file
                extname="ZBEST"
        log.info("reading {}".format(redrock_file))
        zz=Table.read(redrock_file,extname)
        zz.remove_column("COEFF") # 1D array per entry, not needed
        redshifts.append(_rm_meta_keywords(zz))

        #- SB: commenting out zqso code since these have been replaced by
        #- zmtl, but that has to run after QA to be able to update zwarn.
        #- Something like zqso could be revived in QA after the QN afterburner
        #- is finished, but data model details will likely be different.

        ### zqso_file = coadd_file.replace("coadd","zqso")
        ### if not os.path.isfile(zqso_file) :
        ###     log.warning("missing {}".format(zqso_file))
        ### else :
        ###     log.info("reading {}".format(zqso_file))
        ###     zqso=Table.read(zqso_file,"ZQSO")
        ###     zqsos.append(_rm_meta_keywords(zqso))

    log.info("stacking")
    fibermap=vstack(fibermaps)
    scores=vstack(scores)
    redshifts=vstack(redshifts)
    ### if len(zqsos)>0 :
    ###     zqsos=vstack(zqsos)
    ### else :
    ###     zqsos=None
    targetids=fibermap["TARGETID"]

    tile_fiberqa_table = Table()
    for k in ['TARGETID','PETAL_LOC','DEVICE_LOC', 'LOCATION', 'FIBER', 'TARGET_RA', 'TARGET_DEC', 'MEAN_FIBER_X', 'MEAN_FIBER_Y', 'MEAN_DELTA_X', 'MEAN_DELTA_Y', 'RMS_DELTA_X', 'RMS_DELTA_Y','DESI_TARGET', 'BGS_TARGET', 'EBV'] :
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

    # add REDSHIFTS info
    redshifts_tid_to_index = {tid:index for index,tid in enumerate(redshifts["TARGETID"])}
    keys=["Z","SPECTYPE","DELTACHI2"]
    for k in keys :
        tile_fiberqa_table[k] = np.zeros(targetids.size,dtype=redshifts[k].dtype)
    redshifts_ii=[]
    fiberqa_ii=[]
    for i,tid in enumerate(targetids) :
        if tid in redshifts_tid_to_index :
            redshifts_ii.append(redshifts_tid_to_index[tid])
            fiberqa_ii.append(i)
    for k in keys :
        tile_fiberqa_table[k][fiberqa_ii] = redshifts[k][redshifts_ii]

    # add ZQSO info
    ### keys=["Z_QN","Z_QN_CONF","IS_QSO_QN"]
    ### if zqsos is not None :
    ###     zqso_tid_to_index = {tid:index for index,tid in enumerate(zqsos["TARGETID"])}
    ###     for k in keys :
    ###         tile_fiberqa_table[k] = np.zeros(targetids.size,dtype=zqsos[k].dtype)
    ###     zqso_ii=[]
    ###     fiberqa_ii=[]
    ###     for i,tid in enumerate(targetids) :
    ###         if tid in zqso_tid_to_index :
    ###             zqso_ii.append(zqso_tid_to_index[tid])
    ###             fiberqa_ii.append(i)
    ###     for k in keys :
    ###         tile_fiberqa_table[k][fiberqa_ii] = zqsos[k][zqso_ii]
    ### else :
    ###     for k in keys :
    ###         if k == "IS_QSO_QN" :
    ###             tile_fiberqa_table[k] = np.zeros(len(tile_fiberqa_table),dtype=int)
    ###         else :
    ###             tile_fiberqa_table[k] = np.zeros(len(tile_fiberqa_table),dtype=float)

    # AND and OR of exposures QAFIBERSTATUS
    and_qafiberstatus = np.zeros(targetids.size,dtype=exposure_fiberqa_tables["QAFIBERSTATUS"].dtype)
    or_qafiberstatus  = np.zeros(targetids.size,dtype=exposure_fiberqa_tables["QAFIBERSTATUS"].dtype)
    for i,tid in enumerate(targetids) :
        jj = (exposure_fiberqa_tables["TARGETID"]==tid)
        and_qafiberstatus[i] = np.bitwise_and.reduce(exposure_fiberqa_tables['QAFIBERSTATUS'][jj])
        or_qafiberstatus[i] = np.bitwise_or.reduce(exposure_fiberqa_tables['QAFIBERSTATUS'][jj])

    # tile QAFIBERSTATUS is AND of input exposures (+ LOWEFFTIME, see below)
    tile_fiberqa_table["QAFIBERSTATUS"]= and_qafiberstatus

    qa_params=get_qa_params()
    bad_fibers_mask=fibermask.mask(qa_params["exposure_qa"]["bad_qafstatus_mask"])

    # fiber EFFTIME is only counted for good data (per exposure and fiber)
    tile_fiberqa_table["EFFTIME_SPEC"]=np.zeros(targetids.size,dtype=exposure_fiberqa_tables["EFFTIME_SPEC"].dtype)
    for i,tid in enumerate(targetids) :
        jj = (exposure_fiberqa_tables["TARGETID"]==tid)
        tile_fiberqa_table["EFFTIME_SPEC"][i] = np.sum(exposure_fiberqa_tables['EFFTIME_SPEC'][jj]*((exposure_fiberqa_tables['QAFIBERSTATUS'][jj]&bad_fibers_mask)==0))

    # set bit of LOWEFFTIME per fiber
    minimal_efftime = qa_params["tile_qa"]["fiber_rel_mintfrac"]*exposure_qa_meta["MINTFRAC"]*exposure_qa_meta["GOALTIME"]
    low_efftime_fibers = np.where(tile_fiberqa_table["EFFTIME_SPEC"]<minimal_efftime)[0]
    tile_fiberqa_table['QAFIBERSTATUS'][low_efftime_fibers] |= fibermask.mask("LOWEFFTIME")

    # good fibers are the fibers with efftime above the threshold
    good_fibers = np.where((tile_fiberqa_table['QAFIBERSTATUS']&fibermask.mask("LOWEFFTIME"))==0)[0]


    # set some scores per petal, only to make plots
    good_petals = np.unique(tile_fiberqa_table['PETAL_LOC'][good_fibers])

    npetal=10
    tile_petalqa_table = Table()
    petals=np.unique(exposure_fiberqa_tables["PETAL_LOC"])
    tile_petalqa_table["PETAL_LOC"]=np.arange(npetal,dtype=np.int16)
    # all of these will be means of inputs, so get float32 output dtype
    keys=['WORSTREADNOISE', 'NGOODPOS', 'NSTDSTAR', 'STARRMS', 'TSNR2FRA', 'NCFRAME',
          'BSKYTHRURMS', 'BSKYCHI2PDF', 'RSKYTHRURMS', 'RSKYCHI2PDF', 'ZSKYTHRURMS', 'ZSKYCHI2PDF',
          'BTHRUFRAC', 'RTHRUFRAC', 'ZTHRUFRAC']
    for k in keys :
        tile_petalqa_table[k]=np.zeros(npetal, dtype=np.float32)

    for petal in petals :
        ii=(exposure_petalqa_tables["PETAL_LOC"]==petal)
        for k in keys :
            vals=exposure_petalqa_tables[k][ii]
            nonnull=(vals!=0)
            if np.sum(nonnull)>0 :
                tile_petalqa_table[k][petal]=np.mean(vals[nonnull])

    # Petal EFFTIME
    tile_petalqa_table["EFFTIME_SPEC"]=np.zeros(npetal, dtype=np.float32)
    for petal in petals :
        entries=(tile_fiberqa_table['PETAL_LOC'] == petal)
        if np.any(entries):
            tile_petalqa_table['EFFTIME_SPEC'][petal]=np.median(tile_fiberqa_table["EFFTIME_SPEC"][entries])
        else:
            tile_petalqa_table['EFFTIME_SPEC'][petal] = 0.0


    # tile EFFTIME is median of efftime of fibers that are good in all exposures
    always_good_fibers = ((or_qafiberstatus&bad_fibers_mask)==0)
    if np.any(always_good_fibers):
        tile_efftime = np.median(tile_fiberqa_table["EFFTIME_SPEC"][always_good_fibers])
    else:
        tile_efftime = 0.0

    # A tile is valid if efftime > mintfrac*goaltime AND number of good fibers > threshold
    required_tile_efftime = exposure_qa_meta["MINTFRAC"]*exposure_qa_meta["GOALTIME"]
    tile_is_valid = \
            (tile_efftime > required_tile_efftime) & \
            (good_fibers.size > qa_params["tile_qa"]["min_number_of_good_fibers"])

    log.info("Tile {} EFFTIME_SPEC = {:.1f} sec (thres={}), NGOODFIB = {} , valid = {}".format(tileid,tile_efftime,int(exposure_qa_meta["MINTFRAC"]*exposure_qa_meta["GOALTIME"]),good_fibers.size,tile_is_valid))

    # add meta info
    tile_fiberqa_table.meta["TILEID"]=tileid
    tile_fiberqa_table.meta["LASTNITE"]=night
    tile_fiberqa_table.meta["NGOODFIB"]=good_fibers.size
    tile_fiberqa_table.meta["NGOODPET"]=good_petals.size
    tile_fiberqa_table.meta["EFFTIME"]=tile_efftime
    tile_fiberqa_table.meta["VALID"]=tile_is_valid

    # rms dist of good fibers
    dist2 = (tile_fiberqa_table["MEAN_DELTA_X"]**2+tile_fiberqa_table["RMS_DELTA_X"]**2+tile_fiberqa_table["MEAN_DELTA_Y"]**2+tile_fiberqa_table["RMS_DELTA_Y"]**2)
    if len(good_fibers)>0 :
        dist2=dist2[good_fibers]
        ii=np.where((~np.isnan(dist2)))[0]
        if ii.size>0 :
            mdist2=np.mean(dist2[ii])
            if mdist2<0 : mdist2=0
            rmsdist = np.sqrt(mdist2)
        else :
            rmsdist = 0.
    else :
        rmsdist = 0.
    tile_fiberqa_table.meta["RMSDIST"]=rmsdist # mm


    keys = ["TILEID","TILERA","TILEDEC","GOALTIME","GOALTYPE","FAPRGRM","SURVEY","EBVFAC","MINTFRAC"]
    for k in keys :
        if k in exposure_qa_meta :
            tile_fiberqa_table.meta[k] = exposure_qa_meta[k]

    return tile_fiberqa_table ,tile_petalqa_table
