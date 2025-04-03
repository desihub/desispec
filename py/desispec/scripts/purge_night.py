"""
desispec.scripts.purge_night
================================

"""
import argparse
from desispec.io.meta import findfile
from desiutil.log import get_logger
from desispec.scripts.purge_tilenight import purge_tilenight, remove_directory
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
    parser = argparse.ArgumentParser(
        description='Purges a night from a production, intended ' +
                    'for providing a fresh start before resubmitting that night ' +
                    'from the beginning with desi_submit_night. ' +
                    'CAVEAT: this does not purge healpix redshifts, ' +
                    'perexp redshifts, or cumulative redshifts after this night; ' +
                    'i.e. it is intended for cleanup when the failures occured ' +
                    'earlier in the processing.'
    )
    parser.add_argument("-n", "--night", type=int, required=True,
                        help="Night to remove")
    parser.add_argument("--not-dry-run", action="store_true",
                        help="Actually remove files and directories instead of just logging what would be done")
    parser.add_argument("--no-attic", action="store_true",
                        help="delete files directly and do not copy them to attic")

    return parser

def purge_night(night, dry_run=True, no_attic=False):
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
        no_attic, bool. If True, delete files directly and do not copy them to attic

    Note: does not yet remove healpix redshifts touching this tile
    """
    if night is None:
        raise ValueError("Must specify night.")

    specprod = os.environ['SPECPROD']
    epathname = findfile('exposure_table', night=night)
    tiles = None
    if os.path.exists(epathname):
        etable = load_table(tablename=epathname, tabletype='exptable')

        ## select tiles for which future redshift jobs would depend, LASTSTEP==skysub
        ## will be removed with the night-level directory removal
        tile_sel = ((etable['OBSTYPE']=='science') & (etable['LASTSTEP']=='all'))
        tiles = np.asarray(etable['TILEID'][tile_sel])

    log = get_logger()
    log.info(f'Purging night {night}')

    ## Now proceed with removing fill night-level directories and files
    ## specific to the specified night
    reduxdir = os.path.join(os.environ['DESI_SPECTRO_REDUX'], specprod)
    log.info(f'Purging {night} from {reduxdir}')
    os.chdir(reduxdir)

    #- Night and tile directories
    nightdirs = [
                    f'calibnight/{night}',
                    f'exposures/{night}',
                    f'nightqa/{night}',
                    f'preproc/{night}',
                    f'run/scripts/night/{night}',
                ]
    nightdirs += sorted(glob.glob(f'tiles/cumulative/*/{night}'))
    nightdirs += sorted(glob.glob(f'tiles/pernight/*/{night}'))
    nightdirs += sorted(glob.glob(f'run/scripts/tiles/cumulative/*/{night}'))
    nightdirs += sorted(glob.glob(f'run/scripts/tiles/pernight/*/{night}'))

    for dirpath in nightdirs:
        remove_directory(dirpath, dry_run=dry_run, no_attic=no_attic)

    #- Individual files
    processing_table = findfile('processing_table', night=night, specprod=specprod)
    dashboard_exp = findfile('expinfo', night=night, specprod=specprod)
    dashboard_z = findfile('zinfo', night=night, specprod=specprod)

    for filename in [processing_table, dashboard_exp, dashboard_z]:
        if os.path.exists(filename):
            if dry_run:
                log.info(f'dry_run: would remove {filename}')
            else:
                log.info(f'Removing {filename}')
                os.remove(filename)
        else:
            log.info(f'already gone: {filename}')

    ## Next perform the purge of the individual tiles, should skip over
    ## all the individual exposure directories already removed above and then
    ## remove the future redshifts that used the data purged here
    if tiles is not None:
        log.info(f'Future redshifts from {tiles=} will also be removed.')
        purge_tilenight(tiles, night, dry_run=dry_run)

    ## These should now be taken care of by per-tile based removal
    # log.warning("Not attempting to find and purge perexp redshifts")
    # log.warning("Not attempting to find and purge healpix redshifts")

    log.info(f"Done purging {specprod} night {night}")

    if dry_run:
        log.warning('That was a dry run with no files removed; rerun '
                    + 'with --not-dry-run to actually remove files')