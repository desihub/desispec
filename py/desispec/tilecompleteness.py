"""
desispec.tilecompleteness
=========================

Routines to determine the survey progress
and tiles completion.
"""

import os
import numpy as np
import glob
from astropy.table import Table, vstack

from desispec.io import read_table

from desiutil.log import get_logger


def read_gfa_data(gfa_proc_dir):
    """Find the most recent GFA summary file in `gfa_proc_dir`.

    In addition, this function will read an "SV1" file to get earlier EXPIDs.
    The "SV1" file will be concatenated with the most recent summary to produce
    the returned table. There is some duplication of rows in this process, but
    because survey phases did overlap, it's not really possible to separate the
    summaries into distinct phases anyway.

    See also documentation here: https://desi.lbl.gov/trac/wiki/SurveyValidation/SV1/conditions/summary_files.

    Parameters
    ----------
    gfa_proc_dir : :class:`str`
        The GFA directory. Usually this will be ``/global/cfs/cdirs/desi/survey/GFA/``.

    Returns
    -------
    :class:`~astropy.table.Table`
        The summary data read from HDU2 of the files.
    """
    log = get_logger()
    #
    # First read the SV1 file, there should be only one at this point.
    #
    filenames = glob.glob(os.path.join(gfa_proc_dir, 'offline_matched_coadd_ccds_SV1-thru_????????.fits'))
    if len(filenames) == 1:
        log.info("Reading SV1 file: %s", filenames[0])
        tables = [read_table(filenames[0], 2)]
        log.info('%d GFA SV1 table entries', len(tables[0]))
    else:
        log.warning("Could not find a SV1 file, skipping!")
        tables = []
    #
    # Find the most recent file, independently of label.
    #
    filenames = glob.glob(os.path.join(gfa_proc_dir, 'offline_matched_coadd_ccds_*-thru_????????.fits'))
    if len(filenames) == 0:
        mess = "did not find any file offline_matched_coadd_ccds_*-thru_????????.fits in %s"
        log.critical(mess, gfa_proc_dir)
        raise RuntimeError(mess % gfa_proc_dir)
    #
    # Sort the filenames by night, independently of any other label.
    #
    night_filename = dict([(os.path.basename(f).split('_')[-1].split('.')[0], f) for f in filenames])
    last_night = sorted(night_filename.keys())[-1]
    filename = night_filename[last_night]
    log.info("Reading most recent file: %s", filename)
    tables.append(read_table(filename, 2))  # HDU2 is average over frames during spectro exposure and median across CCDs.
    log.info('%d GFA table entries', len(tables[1]))
    #
    # Merge the tables.
    #
    table = vstack(tables)
    log.info('%d GFA merged table entries', len(table))
    return table


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

    tiles, ii = np.unique(exposure_table["TILEID"], return_index=True)
    ntiles=tiles.size
    res=Table()

    res["TILEID"]=tiles
    res["TILERA"]=exposure_table['TILERA'][ii]
    res["TILEDEC"]=exposure_table['TILEDEC'][ii]
    res["SURVEY"]=np.array(np.repeat("unknown",ntiles),dtype='<U20')
    res["PROGRAM"]=exposure_table['PROGRAM'][ii]
    res["FAPRGRM"]=np.array(np.repeat("unknown",ntiles),dtype='<U20')
    res["FAFLAVOR"]=np.array(np.repeat("unknown",ntiles),dtype='<U20')
    res["NEXP"]=np.zeros(ntiles,dtype=int)
    res["EXPTIME"]=np.zeros(ntiles)
    res["EFFTIME_ETC"]=np.zeros(ntiles)
    res["EFFTIME_SPEC"]=np.zeros(ntiles)
    res["EFFTIME_GFA"]=np.zeros(ntiles)
    res["GOALTIME"]  = np.zeros(ntiles)
    res["OBSSTATUS"] = np.array(np.repeat("unknown",ntiles),dtype='<U20')
    res["LRG_EFFTIME_DARK"]=np.zeros(ntiles)
    res["ELG_EFFTIME_DARK"]=np.zeros(ntiles)
    res["BGS_EFFTIME_BRIGHT"]=np.zeros(ntiles)
    res["LYA_EFFTIME_DARK"]=np.zeros(ntiles)
    res["GOALTYPE"]   = np.array(np.repeat("unknown",ntiles),dtype='<U20')
    res["MINTFRAC"]   = np.array(np.repeat(0.9,ntiles),dtype=float)
    res["LASTNIGHT"] = np.zeros(ntiles, dtype=np.int32)
    res.meta['EXTNAME'] = 'TILE_COMPLETENESS'

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

    for i, tile in enumerate(tiles):
        jj = (exposure_table["TILEID"] == tile)
        res["NEXP"][i] = np.sum(jj)
        for k in ("EXPTIME", "LRG_EFFTIME_DARK", "ELG_EFFTIME_DARK", "BGS_EFFTIME_BRIGHT",
                  "LYA_EFFTIME_DARK", "EFFTIME_SPEC", "EFFTIME_ETC", "EFFTIME_GFA"):
            if k in exposure_table.dtype.names:
                #
                # If exposure_table comes from a Table.read(), it may contain masked values.
                # If exposure_table comes from desispec.io.read_table(), it may contain NaN,
                # in particular for a small number of exposures in the GFA summary file.
                # In some cases the NaN values will already have been replaced by zero, but
                # we can and should be careful.
                #
                if not np.isfinite(exposure_table[k][jj]).all():
                    log.warning("exposure_table column '%s' contains NaN or other non-finite values, a sum will be performed over valid values.", k)
                    good_values = exposure_table[k][jj].copy()
                    good_values[~np.isfinite(good_values)] = 0
                    res[k][i] = np.sum(good_values)
                else:
                    if hasattr(exposure_table[k], 'mask'):
                        log.warning("exposure_table column '%s' contains masked values, a sum will be performed over unmasked values.", k)
                    res[k][i] = np.sum(exposure_table[k][jj])
                if k == "EFFTIME_ETC" or k == "EFFTIME_GFA":
                    if np.any((exposure_table[k][jj] == 0) & (exposure_table["EFFTIME_SPEC"][jj] > 0)):
                        res[k][i] = 0  # because we are missing data

        # copy the following from the exposure table if it exists and not already set as sv1

        # look for first exposure with SURVEY set (the others are exposures that were not processed)
        jj2=(exposure_table["TILEID"]==tile)&(exposure_table["SURVEY"]!="unknown")
        if np.sum(jj2)>0 :
            j=np.where(jj2)[0][0]
        else :
            j=np.where(jj)[0][0]

        for k in ["SURVEY","GOALTYPE","FAPRGRM","FAFLAVOR"] :
            if k in exposure_table.dtype.names :
                val = exposure_table[k][j]
                if val != "unknown" :
                    if k != "SURVEY" or res[k][i]!= "sv1" :
                        res[k][i] = val # force consistency

        for k in ["GOALTIME","MINTFRAC"] :
            if k in exposure_table.dtype.names :
                val = exposure_table[k][j]
                if val > 0. :
                    res[k][i] = val # force consistency

    # truncate number of digits for exposure times to 0.1 sec
    for k in res.dtype.names :
        if k.find("EXPTIME")>=0 or k.find("EFFTIME")>=0 :
            res[k] = np.around(res[k],1)

    # efftime_spec set per exposure and here we simpy use the sum
    # default efftime is LRG_EFFTIME_DARK
    #res["EFFTIME_SPEC"]=res["LRG_EFFTIME_DARK"]
    #ii=((res["GOALTYPE"]=="bright")|(res["GOALTYPE"]=="backup"))
    #res["EFFTIME_SPEC"][ii]=res["BGS_EFFTIME_BRIGHT"][ii]

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

    partial=(res["EFFTIME_SPEC"]>0.)&(res["EFFTIME_SPEC"]<=res["MINTFRAC"]*res["GOALTIME"])
    res["OBSSTATUS"][partial]="obsstart"

    # special cases that are in list but have efftime_spec=0.0,
    # e.g. dither tiles or tiles where all exp so far are bad
    other = (res['EFFTIME_SPEC'] == 0.0)
    res['OBSSTATUS'][other] = 'other'

    res = reorder_columns(res)

    # reorder rows
    ii  = np.argsort(res["LASTNIGHT"])
    res = res[ii]

    return res

def reorder_columns(table) :
    neworder=['TILEID','SURVEY','PROGRAM','FAPRGRM','FAFLAVOR','NEXP','EXPTIME','TILERA','TILEDEC','EFFTIME_ETC','EFFTIME_SPEC','EFFTIME_GFA','GOALTIME','OBSSTATUS','LRG_EFFTIME_DARK','ELG_EFFTIME_DARK','BGS_EFFTIME_BRIGHT','LYA_EFFTIME_DARK','GOALTYPE','MINTFRAC','LASTNIGHT']

    if not np.all(np.in1d(neworder,table.dtype.names)) or not np.all(np.in1d(table.dtype.names,neworder)) :
        log = get_logger()
        log.critical("error, mismatch of some keys")
        log.critical("new: {}".format(sorted(neworder)))
        log.critical("input: {}".format(sorted(table.dtype.names)))
        raise ValueError('mismatch of input and reordered columns')

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
        message="not same columns in the two tables {} != {}".format(sorted(table1.dtype.names),sorted(table2.dtype.names))
        log.error(message)
        raise KeyError(message)
    for k in table1.dtype.names :
        v1=table1[k][index1]
        v2=table2[k][index2]
        if np.isreal(v1) :
            if np.isnan(v1) and np.isnan(v2) : continue
        if v1 != v2 :
            return False
    return True


def merge_tile_completeness_table(previous_table,new_table) :
    """ Merges tile summary tables.

    Args:
      previous_table: astropy.table.Table
      new_table: astropy.table.Table
    Returns: astropy.table.Table with merged entries.
    """

    log = get_logger()

    # first check columns and add in previous if missing
    for k in new_table.dtype.names :
        if not k in previous_table.dtype.names :
            log.info("New column {}".format(k))
            previous_table[k] = np.zeros(len(previous_table),dtype=new_table[k].dtype)

    # check whether there is any difference for the new ones
    t2i={t:i for i,t in enumerate(previous_table["TILEID"])}

    nadd=0
    nmod=0
    nforcekeep=0

    # keep all tiles that are not in the new table
    keep_from_previous = list(np.where(~np.in1d(previous_table["TILEID"],new_table["TILEID"]))[0])
    nsame = len(keep_from_previous)

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

        # do some sanity check
        any_change=False
        for k in ["SURVEY","GOALTYPE"] :
            if new_table[k][j] == "unknown" and previous_table[k][i] != "unknown" :
                log.warning("IGNORE change for tile {} of {}: {} -> {}".format(t,k,previous_table[k][i],new_table[k][j]))
                new_table[k][j] = previous_table[k][i]
                any_change=True

        survey = new_table["SURVEY"][j]
        if survey in ["cmx","sv1","sv2","sv3"]:
            for k in ["GOALTIME","OBSSTATUS"] :
                if new_table[k][j] != previous_table[k][i] :
                    log.warning("IGNORE change for tile {} of {}: {} -> {}".format(t,k,previous_table[k][i],new_table[k][j]))
                    new_table[k][j] = previous_table[k][i]
                    any_change=True

        if any_change : # recheck if still different
            if is_same_table_rows(previous_table,i,new_table,j) :
                nsame += 1
                keep_from_previous.append(i)
                continue

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
