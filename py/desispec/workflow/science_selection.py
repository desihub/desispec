"""
desispec.workflow.science_selection
===================================

"""
import sys, os, glob
import json
from astropy.io import fits
from astropy.table import Table, join
import numpy as np

import time, datetime
from collections import OrderedDict
import subprocess
from copy import deepcopy

from desispec.scripts.tile_redshifts import generate_tile_redshift_scripts
from desispec.workflow.redshifts import get_ztile_script_pathname, \
    get_ztile_relpath, \
    get_ztile_script_suffix, read_minimal_exptables_columns
from desispec.workflow.queue import get_resubmission_states, update_from_queue, queue_info_from_qids
from desispec.workflow.timing import what_night_is_it
from desispec.workflow.desi_proc_funcs import get_desi_proc_batch_file_pathname, \
                                              create_desi_proc_batch_script, \
                                              get_desi_proc_batch_file_path, \
                                              get_desi_proc_tilenight_batch_file_pathname, \
                                              create_desi_proc_tilenight_batch_script
from desispec.workflow.utils import pathjoin, sleep_and_report
from desispec.workflow.tableio import write_table
from desispec.workflow.proctable import table_row_to_dict
from desiutil.log import get_logger

from desispec.io import findfile, specprod_root
from desispec.io.util import decode_camword, create_camword, \
    difference_camwords, \
    camword_to_spectros, camword_union, camword_intersection, parse_badamps


#################################################
############## Misc Functions ###################
#################################################


import numpy as np
from astropy.table import Table, vstack


def determine_science_to_proc(etable, tiles, surveys, laststeps,
                              processed_tiles=None,
                              all_tiles=True,
                              ignore_last_tile=False,
                              complete_tiles_thrunight=None,
                              specstatus_path=None):
    """
     Selects the science exposures that should be processed from a populated
     exposure table given the details and flags given as inputs.

    Args:
        etable (astropy.table.Table): A DESI exposure_table
        tiles (array-like, optional): Only submit jobs for these TILEIDs.
        surveys (array-like, optional): Only submit science jobs for these
            surveys (lowercase)
        laststeps (array-like, optional): Only submit jobs for exposures with
            LASTSTEP in these science_laststeps (lowercase)
        processed_tiles (array-like, optional): TILEIDs that have already
            been processed
        all_tiles (bool, optional): Default is True. Set to NOT restrict to
            completed tiles as defined by the table pointed to by specstatus_path.
        ignore_last_tile (bool): Default is False. Whether to ignore the last
            observed tile. Generally used with daily operations when expecting
            more data to come in the future.
        complete_tiles_thrunight (int, optional): Default is None. Only
            tiles completed on or before the supplied YYYYMMDD are considered
            completed and will be processed. All complete tiles are submitted
            if None or all_tiles is True.
        specstatus_path (str, optional): Location of the surveyops specstatus
            table.Default is $DESI_SURVEYOPS/ops/tiles-specstatus.ecsv.

    Returns:
        astropy.table.Table: A DESI exposure_table only containing the science
            exposures that should be processed.
        list: A list of the tiles that should be processed, in the order they
            first appear in the input exposure_table.
     """
    log = get_logger()
    ## divide into calibration and science etables
    full_etable = etable.copy()
    sci_etable = etable[etable['OBSTYPE'] == 'science']

    ## Cut on exposure time
    if len(sci_etable) > 0:
        sci_etable = sci_etable[sci_etable['EXPTIME'] >= 60]

    ## Remove any exposure related to the last tile when in daily mode
    ## and during the nightly processing
    if ignore_last_tile and len(sci_etable) > 0:
        last_ind = np.argmax(full_etable['EXPID'])
        if full_etable['OBSTYPE'][last_ind] == 'science':
            last_tile = full_etable['TILEID'][last_ind]
            log.info(f"Ignoring exposures associated with tile {last_tile} since it"
                    + f" was the last exposure observed and {ignore_last_tile=}")
            sci_etable = sci_etable[sci_etable['TILEID'] != last_tile]

    ## Cut on LASTSTEP
    if len(sci_etable) > 0:
        good_exps = np.isin(np.array(sci_etable['LASTSTEP']).astype(str), laststeps)
        sci_etable = sci_etable[good_exps]

    ## Identify tiles that have already been processed and remove them
    if len(sci_etable) > 0:
        keep = np.bitwise_not(np.isin(sci_etable['TILEID'], processed_tiles))
        sci_etable = sci_etable[keep]

    ## filter by TILEID if requested
    if tiles is not None and len(sci_etable) > 0:
        log.info(f'Filtering by tiles={tiles}')
        keep = np.isin(sci_etable['TILEID'], tiles)
        sci_etable = sci_etable[keep]

    ## filter by SURVEY if requested
    if surveys is not None and len(sci_etable) > 0:
        log.info(f'Filtering by surveys={surveys}')
        if 'SURVEY' not in etable.dtype.names:
            raise ValueError(f'surveys={surveys} filter requested, but no '
                             + f'SURVEY column in exposure_table')

        keep = np.zero(len(sci_etable), dtype=bool)
        # np.isin doesn't work with bytes vs. str from Tables but direct
        # comparison does, so loop
        for survey in surveys:
            keep |= sci_etable['SURVEY'] == survey

        sci_etable = sci_etable[keep]

    ## If asked to do so, only process tiles deemed complete by the specstatus file
    if not all_tiles and len(sci_etable) > 0 and complete_tiles_thrunight is not None:
        all_completed_tiles = get_completed_tiles(specstatus_path,
                                    complete_tiles_thrunight=complete_tiles_thrunight)
        keep = np.isin(sci_etable['TILEID'], all_completed_tiles)
        sci_tiles = np.unique(sci_etable['TILEID'][keep])
        log.info(f"Processing completed science tiles: "
                 + f"{', '.join(sci_tiles.astype(str))}")
        log.info(f"Filtering by completed tiles retained "
                 + f"{len(sci_tiles)}/{sum(np.unique(sci_etable['TILEID'])>0)} science tiles")
        log.info(f"Filtering by completed tiles retained "
                 + f"{sum(keep)}/{sum(sci_etable['TILEID']>0)} science exposures")
        sci_etable = sci_etable[keep]

    ## Identify tiles to be processed, in chronological order
    tiles_to_proc = []
    for tile in sci_etable['TILEID']:
        if tile in tiles_to_proc:
            continue
        else:
            tiles_to_proc.append(tile)

    return sci_etable, tiles_to_proc


def get_completed_tiles(specstatus_path=None, complete_tiles_thrunight=None):
    """
    Uses a tiles-specstatus.ecsv file and selection criteria to determine
    what tiles have beeen completed. Takes an optional argument to point
    to a custom specstatus file. Returns an array of TILEID's.

    Args:
        specstatus_path, str, optional. Location of the surveyops specstatus
            table. Default is $DESI_SURVEYOPS/ops/tiles-specstatus.ecsv.
        complete_tiles_thrunight, int, optional. Default is None. Only
            tiles completed on or before the supplied YYYYMMDD are considered
            completed and will be processed. All complete
            tiles are submitted if None.

    Returns:
        array-like. The tiles from the specstatus file determined by the
        selection criteria to be completed.
    """
    log = get_logger()
    if specstatus_path is None:
        if 'DESI_SURVEYOPS' not in os.environ:
            raise ValueError("DESI_SURVEYOPS is not defined in your environment. " +
                             "You must set it or specify --specstatus-path explicitly.")
        specstatus_path = os.path.join(os.environ['DESI_SURVEYOPS'], 'ops',
                                       'tiles-specstatus.ecsv')
        log.info(f"specstatus_path not defined, setting default to {specstatus_path}.")
    if not os.path.exists(specstatus_path):
        raise IOError(f"Couldn't find {specstatus_path}.")
    specstatus = Table.read(specstatus_path)

    ## good tile selection
    iszdone = (specstatus['ZDONE'] == 'true')
    isnotmain = (specstatus['SURVEY'] != 'main')
    enoughfraction = 0.1  # 10% rather than specstatus['MINTFRAC']
    isenoughtime = (specstatus['EFFTIME_SPEC'] >
                    specstatus['GOALTIME'] * enoughfraction)
    ## only take the approved QA tiles in main
    goodtiles = iszdone
    ## not all special and cmx/SV tiles have zdone set, so also pass those with enough time
    goodtiles |= (isenoughtime & isnotmain)
    ## main backup also don't have zdone set, so also pass those with enough time
    goodtiles |= (isenoughtime & (specstatus['FAPRGRM'] == 'backup'))

    if complete_tiles_thrunight is not None:
        goodtiles &= (specstatus['LASTNIGHT'] <= complete_tiles_thrunight)

    return np.array(specstatus['TILEID'][goodtiles])


def get_tiles_cumulative(sci_etable, z_submit_types, all_cumulatives, night):
    """
    Takes an exposure table, list of redshift types to submit, and a boolean
    defining whether to return all cumulatives or not, and returns the list
    of tiles for which cumulative redshifts should be performed based on whether
    it is the last known night in which that tiles was observed.

    Args:
        sci_etable, Table. An exposure table with column TILEID.
        z_submit_types, list or None. List of strings identifying the
            redshift types to run.
        all_cumulatives, bool. If True all tile id's in the sci_etable are
            returned, otherwise only those who were observed last on the given
            night are returned for cumulative redshifts
        night, int. The night in question, in YYYYMMDD format.

    Returns:
        tiles_cumulative, list. List of tile id's that should get cumulative
            redshifts.

    """
    log = get_logger()
    tiles_cumulative = list()
    if z_submit_types is not None and 'cumulative' in z_submit_types:
        tiles_this_night = np.unique(np.asarray(sci_etable['TILEID']))
        # select only science tiles, not calibs
        tiles_this_night = tiles_this_night[tiles_this_night > 0]
        if all_cumulatives:
            tiles_cumulative = list(tiles_this_night)
            log.info(f'Submitting cumulative redshifts for all tiles: {tiles_cumulative}')
        else:
            allexp = read_minimal_exptables_columns(tileids=tiles_this_night)
            for tileid in tiles_this_night:
                nights_with_tile = allexp['NIGHT'][allexp['TILEID'] == tileid]
                if len(nights_with_tile) > 0 and night == np.max(nights_with_tile):
                    tiles_cumulative.append(tileid)
            log.info(f'Submitting cumulative redshifts for {len(tiles_cumulative)}'
                     + f'/{len(tiles_this_night)} tiles for '
                     + f'which {night} is the last night: {tiles_cumulative}')

    return tiles_cumulative
