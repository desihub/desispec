"""
desispec.scripts.purge_tilenight
================================

"""
import argparse
from desispec.io.meta import findfile
from desispec.workflow.exptable import get_exposure_table_pathname
from desispec.workflow.proctable import get_processing_table_pathname
from desispec.workflow.tableio import load_table, write_table

import os
import glob
import shutil
import sys
import numpy as np
import time

def get_parser():
    """
    Creates an arguments parser for the desi_purge_tilenight script
    """
    parser = argparse.ArgumentParser(usage = "{prog} [options]")
    parser.add_argument("-n", "--night", type=int, required=True,
            help="Night that the tile was observed.")
    parser.add_argument("-t", "--tiles", type=str, required=True,
            help="Tiles to remove from current prod. (comma separated)")
    parser.add_argument("--not-dry-run", action="store_true",
            help="set to actually perform action rather than print actions")
    return parser

def remove_directory(dirname, dry_run=True):
    """
    Remove the given directory from the file system

    Args:
        dirname, str. Full pathname to the directory you want to remove
        dru_run, bool. True if you want to print actions instead of performing them.
                       False to actually perform them.
    """
    if os.path.exists(dirname):
        print(f"Identified directory {dirname} as existing.")
        print(f"Dir has contents: {os.listdir(dirname)}")
        if dry_run:
            print(f"Dry_run set, so not performing action.")
        else:
            print(f"Removing: {dirname}")
            shutil.rmtree(dirname)
    else:
        print(f"Directory {dirname} doesn't exist, so no action required.")

def purge_tilenight(tiles, night, dry_run=True):
    """
    Removes all files assosciated with tiles on a given night.

    Removes preproc files, exposures files including frames, redrock files
    for perexp and pernight, and cumulative redshifts for nights on or
    after the night in question. Only exposures associated with the tile
    on the given night are removed, but all future cumulative redshift jobs
    are also removed.

    Args:
        tiles, list of int. Tile to remove from current prod.
        night, int. Night that tiles were observed.
        dry_run, bool. If True, only prints actions it would take

    Note: does not yet remove healpix redshifts touching this tile
    """
    if night is None:
        raise ValueError("Must specify night.")
    if tiles is None:
        raise ValueError("Must specify list of tiles.")

    epathname = get_exposure_table_pathname(night=str(night), usespecprod=True)
    etable = load_table(tablename=epathname, tabletype='exptable')

    print(f'Purging night {night} tiles {tiles}')
    future_cumulatives = {}
    for tile in tiles:
        print(f'Purging tile {tile}')
        exptable = etable[etable['TILEID'] == tile]

        ## Per exposure: remove preproc, exposure, and perexp redshift dirs
        for row in exptable:
            expid = int(row['EXPID'])

            for ftype in ['preproc', 'frame']:
                dirname = os.path.dirname(findfile(filetype=ftype, night=night,
                                                   expid=expid, camera='b0',
                                                   spectrograph=0, tile=tile))
                remove_directory(dirname, dry_run)

            groupname = 'perexp'
            ftype = 'redrock'
            dirname = os.path.dirname(findfile(filetype=ftype, night=night,
                                               expid=expid, camera='b0',
                                               spectrograph=0, tile=tile,
                                               groupname=groupname))
            remove_directory(dirname, dry_run)

        ## Remove the pernight redshift directory if it exists
        groupname = 'pernight'
        ftype = 'redrock'
        dirname = os.path.dirname(findfile(filetype=ftype, night=night,
                                           camera='b0', spectrograph=0,
                                           tile=tile, groupname=groupname))
        remove_directory(dirname, dry_run)

        ## Look at all cumulative redshifts and remove any that would include the
        ## give tile-night data (any THRUNIGHT on or after the night given)
        groupname = 'cumulative'
        ftype = 'redrock'
        tiledirname = os.path.dirname(os.path.dirname(
            findfile(filetype=ftype, night=night, camera='b0', spectrograph=0,
                     tile=tile, groupname=groupname)))
        if os.path.exists(tiledirname):
            thrunights = os.listdir(tiledirname)
            for thrunight in thrunights:
                thrunight_int = int(thrunight)
                if thrunight_int >= night:
                    dirname = os.path.join(tiledirname,thrunight)
                    remove_directory(dirname, dry_run)
                    if thrunight_int > night:
                        if thrunight_int in future_cumulatives:
                            future_cumulatives[thrunight_int].append(tile)
                        else:
                            future_cumulatives[thrunight_int] = [tile]

    ## Finally, remove any dashboard caches for the impacted nights
    allnights = sorted([night] + list(future_cumulatives.keys()))
    for dashnight in allnights:
        for cachefiletype in ['expinfo', 'zinfo']:
            dashcache = findfile(cachefiletype, night=dashnight)
            if os.path.exists(dashcache):
                if dry_run:
                    print(f"Dry_run set, so not removing {dashcache}.")
                else:
                    print(f"Removing: {dashcache}.")
                    os.remove(dashcache)
            else:
                print(f"Couldn't find {cachefiletype} file: {dashcache}")

    ## Load old processing table
    timestamp = time.strftime('%Y%m%d_%Hh%Mm')
    ppathname = findfile('processing_table', night=night)
    ptable = load_table(tablename=ppathname, tabletype='proctable')

    ## Now let's remove the tiles from the processing table
    keep = np.isin(ptable['TILEID'], tiles, invert=True)
    print(f'Removing {len(keep) - np.sum(keep)}/{len(keep)} processing '
          + f'table entries for {night=}')
    ptable = ptable[keep]

    if dry_run:
        print(f'dry_run: not changing {ppathname}')
    else:
        print(f'Archiving old processing table for {night=} with '
              + f'timestamp {timestamp} and saving trimmed one')
        ## move old processing table out of the way
        os.rename(ppathname,ppathname.replace('.csv',f".csv.{timestamp}"))
        ## save new trimmed processing table
        write_table(ptable,tablename=ppathname)

    ## Now archive and modify future processing tables
    for futurenight, futuretiles in future_cumulatives.items():
        ppathname = findfile('processing_table', night=futurenight)
        ptable = load_table(tablename=ppathname, tabletype='proctable')

        ## Now let's remove the tiles from the processing table
        nokeep = ptable['JOBNAME'] == 'cumulative'
        nokeep &= np.isin(ptable['TILEID'], futuretiles)
        keep = np.bitwise_not(nokeep)
        print(f'Removing {len(keep) - np.sum(keep)}/{len(keep)} processing '
              + f'table entries for night={futurenight}')
        ptable = ptable[keep]

        if dry_run:
            print(f'dry_run: not changing {ppathname}')
        else:
            print(f'Archiving old processing table for night={futurenight} with '
                  + f'timestamp {timestamp} and saving trimmed one')
            ## move old processing table out of the way
            os.rename(ppathname, ppathname.replace('.csv', f".csv.{timestamp}"))
            ## save new trimmed processing table
            write_table(ptable, tablename=ppathname)