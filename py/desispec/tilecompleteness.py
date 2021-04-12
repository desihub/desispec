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


def compute_tile_completeness_table(exposure_table,specprod_dir,auxiliary_table_filenames,min_number_of_petals=8) :
    """ Computes a summary table of the observed tiles

    Args:
      exposure_table: astropy.table.Table with exposure summary table from a prod (output of desi_tsnr_afterburner)
      specprod_dir: str, production directory name
      auxiliary_table_filenames: list(str), list of auxiliary table names, optional
      min_number_of_petals: int, minimum number of petals to declare a tile done
    Returns: astropy.table.Table with one row per TILEID, with completeness and exposure time.
    """

    log = get_logger()

    default_goaltime = 1000. # objective effective time in seconds





    tiles=np.unique(exposure_table["TILEID"])
    ntiles=tiles.size
    res=Table()

    res["TILEID"]=tiles
    res["SURVEY"]=np.array(np.repeat("unknown",ntiles),dtype='<U20')
    res["FAPRGRM"]=np.array(np.repeat("unknown",ntiles),dtype='<U20')
    res["FAFLAVOR"]=np.array(np.repeat("unknown",ntiles),dtype='<U20')
    res["NEXP"]=np.zeros(ntiles,dtype=int)
    res["EXPTIME"]=np.zeros(ntiles)
    res["EFFTIME_ETC"]=np.zeros(ntiles)
    res["EFFTIME_SPEC"]=np.zeros(ntiles)
    res["EFFTIME_GFA"]=np.zeros(ntiles)
    res["GOALTIME"]  = np.zeros(ntiles)
    res["OBSSTATUS"] = np.array(np.repeat("unknown",ntiles),dtype='<U20')
    res["ZDONE"]   = np.array(np.repeat("false",ntiles),dtype='<U20')
    res["ELG_EFFTIME_DARK"]=np.zeros(ntiles)
    res["BGS_EFFTIME_BRIGHT"]=np.zeros(ntiles)
    res["LYA_EFFTIME_DARK"]=np.zeros(ntiles)
    res["GOALTYPE"]   = np.array(np.repeat("unknown",ntiles),dtype='<U20')
    res["MINTFRAC"]   = np.array(np.repeat(0.9,ntiles),dtype=float)
    res["LASTNIGHT"] = np.zeros(ntiles, dtype=np.int32)

    # case is /global/cfs/cdirs/desi/survey/observations/SV1/sv1-tiles.fits
    if auxiliary_table_filenames is not None :
        for filename in auxiliary_table_filenames :

            if filename.find("sv1-tiles")>=0 :

                targets = np.array(np.repeat("unknown",ntiles))

                log.info("Use SV1 tiles information from {}".format(filename))
                table=Table.read(filename, 1)
                ii=[]
                jj=[]
                tid2i={tid:i for i,tid in enumerate(table["TILEID"])}
                for j,tid in enumerate(res["TILEID"]) :
                    if tid in tid2i :
                        ii.append(tid2i[tid])
                        jj.append(j)

                for i,j in zip(ii,jj) :
                    res["SURVEY"][j]=str(table["PROGRAM"][i]).lower()
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
        for k in ["EXPTIME","ELG_EFFTIME_DARK","BGS_EFFTIME_BRIGHT","LYA_EFFTIME_DARK","EFFTIME_ETC","EFFTIME_GFA"] :
            if k in exposure_table.dtype.names :
                res[k][i] = np.sum(exposure_table[k][jj])
                if k == "EFFTIME_ETC" or k == "EFFTIME_GFA" :
                    if np.any((exposure_table[k][jj]==0)&(exposure_table["EFFTIME_SPEC"][jj]>0)) : res[k][i]=0 # because we are missing data

        # copy the following from the exposure table if it exists and not already set as sv1
        for k in ["SURVEY","GOALTYPE","FAPRGRM","FAFLAVOR"] :
            if k in exposure_table.dtype.names :
                for val in  exposure_table[k][jj] : # loop in case one of them is not unknown
                    if val != "unknown" :
                        if k != "SURVEY" or res[k][i]!= "sv1" :
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

    # what was the last night on which each tile was observed,
    # for looking up the cumulative redshift file
    goodexp = (exposure_table['EFFTIME_SPEC'] > 0)
    for i, tileid in enumerate(res['TILEID']):
        thistile = (exposure_table['TILEID'] == tileid)
        if np.any(thistile & goodexp):
            lastnight = np.max(exposure_table['NIGHT'][thistile & goodexp])
        else:
            #- no exposures for this tile with EFFTIME_SPEC>0,
            #- so just use last one that appears at all
            lastnight = np.max(exposure_table['NIGHT'][thistile])

        res['LASTNIGHT'][i] = lastnight

    assert np.all(res['LASTNIGHT'] > 0)

    for i in np.where((res["OBSSTATUS"]=="obsend")&(res["ZDONE"]=="false"))[0] :
        tileid=res["TILEID"][i]
        night=res["LASTNIGHT"][i]
        log.info(f"checking redshifts for tile {tileid} on night {night}")
        nok = number_of_good_zbest(tileid=tileid,night=night,specprod_dir=specprod_dir)
        if nok >= min_number_of_petals :
            res["ZDONE"][i]="true"
        elif nok > 0 :
            log.warning("keep ZDONE=false for tile {} because only {} good petals (requirement is >={})".format(tileid,nok,min_number_of_petals))
    partial=(res["EFFTIME_SPEC"]>0.)&(res["EFFTIME_SPEC"]<=res["MINTFRAC"]*res["GOALTIME"])
    res["OBSSTATUS"][partial]="obsstart"

    res = reorder_columns(res)

    # reorder rows
    ii  = np.argsort(res["LASTNIGHT"])
    res = res[ii]

    return res

def reorder_columns(table) :
    neworder=['TILEID','SURVEY','FAPRGRM','FAFLAVOR','NEXP','EXPTIME','EFFTIME_ETC','EFFTIME_SPEC','EFFTIME_GFA','GOALTIME','OBSSTATUS','ZDONE','ELG_EFFTIME_DARK','BGS_EFFTIME_BRIGHT','LYA_EFFTIME_DARK','GOALTYPE','MINTFRAC','LASTNIGHT']

    if not np.all(np.in1d(neworder,table.dtype.names)) or not np.all(np.in1d(table.dtype.names,neworder)) :
        print("error, mismatch of some keys")
        print(sorted(neworder))
        print(sorted(table.dtype.names))
        sys.exit(12)

    if np.all(np.array(neworder)==np.array(table.dtype.names)) : # same
        return table

    newtable=Table()
    newtable.meta=table.meta
    for k in neworder :
       newtable[k]=table[k]

    return newtable

def is_same_table_rows(table1,index1,table2,index2) :

    #if table1[index1] == table2[index2] : return True

    if sorted(table1.dtype.names) != sorted(table2.dtype.names) :
        print("not same keys??")
        return False
    for k in table1.dtype.names :
        if table1[k][index1] != table2[k][index2] :
            return False
    return True


def merge_tile_completeness_table(previous_table,new_table) :
    """ Merges tile summary tables. Entries with tiles previously marked as ZDONE are not modified.

    Args:
      previous_table: astropy.table.Table
      new_table: astropy.table.Table
    Returns: astropy.table.Table with merged entries.
    """

    log = get_logger()

    # check whether there is any difference for the new ones
    t2i={t:i for i,t in enumerate(previous_table["TILEID"])}

    nadd=0
    nmod=0
    nsame=0
    nforcekeep=0
    keep_from_previous = []
    add_from_new = []
    for j,t in enumerate(new_table["TILEID"]) :
        if t not in t2i :
            nadd += 1
            add_from_new.append(j)
            continue
        i=t2i[t]
        if is_same_table_rows(previous_table,i,new_table,j) :
            nsame += 1
            keep_from_previous.append(i)
            continue
        if previous_table["ZDONE"][i]=="true" and new_table["ZDONE"][j]!="true" :
            if previous_table["OBSSTATUS"][i]==new_table["OBSSTATUS"][j] :
                nforcekeep += 1
                nsame += 1
                keep_from_previous.append(i)
                log.warning("For tile {} zdone=true for lastnight={} but zdone={} for lastnight={}".format(t,previous_table["LASTNIGHT"][i],new_table["ZDONE"][j],new_table["LASTNIGHT"][j]))
                log.warning("previously: {}".format(previous_table[i]))
                log.warning("newly: {}".format(new_table[j]))
                continue
            elif previous_table["GOALTIME"][i] != new_table["GOALTIME"][j] :
                log.warning("For tile {} GOALTIME changed from {} to {} ???".format(t,previous_table["GOALTIME"][i],new_table["GOALTIME"][j]))
            else :
                log.warning("For tile {} OBSTATUS changed from {} to {} ???".format(t,previous_table["OBSSTATUS"][i],new_table["OBSSTATUS"][j]))

        nmod += 1
        add_from_new.append(j)


    log.info("{} tiles unchanged".format(nsame))
    log.info("{} tiles modified".format(nmod))
    log.info("{} tiles added".format(nadd))

    if len(add_from_new)>0 :
        res = vstack( [ previous_table[keep_from_previous] , new_table[add_from_new] ] )
    else :
        res = previous_table

    res = reorder_columns(res)
    # reorder rows
    ii  = np.argsort(res["LASTNIGHT"])
    res = res[ii]

    return res

def number_of_good_zbest(tileid,night,specprod_dir,warn=True) :

    log=get_logger()
    nok=0
    for spectro in range(10) :

        coadd_filename = os.path.join(specprod_dir,"tiles/cumulative/{}/{}/coadd-{}-{}-thru{}.fits".format(tileid,night,spectro,tileid,night))
        if not os.path.isfile(coadd_filename) :
            if warn : log.warning("missing {}".format(coadd_filename))
            continue
        zbest_filename = os.path.join(specprod_dir,"tiles/cumulative/{}/{}/zbest-{}-{}-thru{}.fits".format(tileid,night,spectro,tileid,night))
        if not os.path.isfile(zbest_filename) :
            if warn : log.warning("missing {}".format(zbest_filename))
            continue

        # do more tests

        nok+=1

    return nok
