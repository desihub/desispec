"""
desispec.workflow.calibration_selection
=======================================

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
from desispec.workflow.processing import generate_calibration_dict
from desispec.workflow.redshifts import get_ztile_script_pathname, \
                                        get_ztile_relpath, \
                                        get_ztile_script_suffix
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


def determine_calibrations_to_proc(full_etable, do_cte_flats=True,
                                   still_acquiring=False):
    """
     Selects the calibration exposures that should be processed from a
     populated exposure table.

    Args:
        full_etable (astropy.table.Table): A DESI exposure_table.
        do_cte_flats (bool, optional): Default is True. If True, cte flats
            are used if available to correct for cte effects.
        still_acquiring (bool): Whether data is still being acquired or the
            provided table is the complete set that will exist for the night.

    Returns:
        astropy.table.Table: A DESI exposure_table only containing the
            calibration exposures that should be processed.
     """
    log = get_logger()
    etable = full_etable.copy()

    if len(etable) > 0:
        etable = etable[etable['OBSTYPE'] != 'science']

    if len(etable) == 0:
        return etable[[]]

    valid_etable, exptypes = select_valid_calib_exposures(etable)
    zeros = valid_etable[exptypes=='zero']
    best_arcflat_set = None
    if np.sum(exptypes=='dark') >= 1 and np.sum(exptypes=='arc') >= 5 \
            and np.sum(exptypes=='flat') >= 12:
        if np.sum(exptypes=='cteflat')>=1 or not do_cte_flats:
            log.info(f"Found at least one possible calibration set to test.")
        elif still_acquiring:
            log.info(f"Only found {np.sum(exptypes=='cteflat')} cteflats "
                     f"but still acquiring new data, so stopping here until "
                     f"more information is known.")
            return etable[[]]
        else:
            log.info(f"Only found {np.sum(exptypes=='cteflat')} cteflats "
                     f"but no longer acquiring new data, so proceeding "
                     f"to check for complete calibration sets.")
        best_arcflat_set = find_best_arc_flat_sets(etable)
    elif still_acquiring:
        log.info(f"Couldn't find a complete set of cals to test for validity "
                 f"but still acquiring new data, so stopping here until "
                 f"more information is known.")
        return etable[[]]
    else:
        log.error(f"Couldn't find a complete set of cals to test for validity "
                 f"and no longer acquiring new data, so this will likely be "
                 f"a fatal issue.")

    out_table = vstack([zeros, etable[exptypes == 'dark'][:1]])
    if best_arcflat_set is not None:
        out_table = vstack([out_table, best_arcflat_set])
    if do_cte_flats:
        out_table = vstack([out_table, etable[exptypes == 'cteflat'][:3]])

    ## Since the arc set can include LASTSTEP=='ignore', cut those out now
    if len(out_table) > 1:
        out_table = out_table[out_table['LASTSTEP']=='all']
    return out_table



def select_valid_calib_exposures(etable):
    """
     Selects the calibration exposures from a populated exposure table
     that pass consistency requirements on exposure time and have
     LASTSTEP=='all', except for arcs which keep all LASTSTEPS for the time
     being do to how the arc sets are selected later on.

    Args:
        etable (astropy.table.Table): A DESI exposure_table

    Returns:
        astropy.table.Table: A DESI exposure_table only containing the
            calibration exposures that pass exposure time cuts and are
            therefore candidates for valid calibration exposures.
        np.array: An array of derived observation types where each index
            corresponds to the row of the returned exposure_table. The
            obseration types include CTE information and aren't redundant with
            column 'OBSTYPE'.
     """
    etable = etable[((etable['OBSTYPE'] == 'arc') or (etable['LASTSTEP'] == 'all'))]
    good_exptimes, exptype = [], []
    for erow in etable:
        ## Zero should have 0 exptime
        if erow['OBSTYPE'] == 'zero' \
                and matches_exptime(erow['EXPTIME'], 0.):
            good_exptimes.append(True)
            exptype.append('zero')
        ## Any 300s dark is valid
        if erow['OBSTYPE'] == 'dark' \
                and matches_exptime(erow['EXPTIME'], 300.):
            good_exptimes.append(True)
            exptype.append('dark')
        ## only 5s arcs labeled "short Arcs all" have correct lamps
        ## for correct duration
        elif erow['OBSTYPE'] == 'arc' \
                and matches_exptime(erow['EXPTIME'], 5.) \
                and erow['PROGRAM'] == 'CALIB short Arcs all':
            good_exptimes.append(True)
            exptype.append('arc')
        ## Only 120s flats labeled 'DESI-CALIB-0' are correct for flat fielding
        elif erow['OBSTYPE'] == 'flat' \
                and matches_exptime(erow['EXPTIME'], 120.) \
                and 'DESI-CALIB-0' in erow['PROGRAM']:
            good_exptimes.append(True)
            exptype.append('flat')
        ## CTE flats come in 1s, 3s, and 10s varieties
        elif erow['OBSTYPE'] == 'flat' and 'CTE' in erow['PROGRAM']:
            if matches_exptime(erow['EXPTIME'],1.) \
                    or matches_exptime(erow['EXPTIME'],3.) \
                    or matches_exptime(erow['EXPTIME'],10.):
                good_exptimes.append(True)
                exptype.append('cteflat')
            else:
                good_exptimes.append(False)
                exptype.append('other')
        ## Everything else is invalid (note zeros are handled separately)
        else:
            good_exptimes.append(False)
            exptype.append('other')

    outtable = etable[np.array(good_exptimes)]
    exptype = np.array(exptype)
    assert len(outtable) == len(exptype), ("output table and exposure types "
                                           + "should be the same length")
    return outtable, exptype

def matches_exptime(val_or_array, exptime, exptime_tolerance=1.):
    """
    For a scalar or array it does an elementwise comparison of that value or
    values with the given exposure time to test whether it is within a given
    tolerance.

    Args:
        val_or_array (int, float, np.array): Exposure time(s) of exposure(s)
            to test against exptime.
        exptime (int, float): the reference exposure time. Returns true if
            val_or_array is within exptime_tolerance of exptime.
        exptime_tolerance (float): the tolerance within which the exposure
            times need to match.

    Returns:
        bool: True if val_or_array elements are within exptime_tolerance
            of exptime inclusive.
    """
    exptime, exptime_tolerance = float(exptime), float(exptime_tolerance)
    return np.abs(val_or_array - exptime) <= exptime_tolerance




def print_row_message(message, dictlike, keys=None, print_func=print):
    """
    Helper function that prints a subset of values from a dict or astropy table
    in a user friendly way with the option to use print or desi logger.

    Args:
        message (str): Message to be printed verbatim before the key:val pairs
        dictlike (dict or astropy.Table.table): the object holding the values
            for the keys provided in keys, or the defaults given in the
            function.
        keys (list or np.array): iterable of strings that when passed to
            the dictlike object return values.
        print_func (function): the function used to return or display the
            information.
    """
    if keys is None:
        keys = ['EXPID', 'SEQNUM', 'SEQTOT', 'LASTSTEP', 'BADCAMWORD',
                   'BADAMPS', 'PROGRAM', 'OBSTYPE']
    string = message
    for col in keys:
        string += f'  {col}:{dictlike[col]}'
    print_func(string)


def find_best_arc_flat_sets(exptable, ngoodarcthreshold=3, narcsequence=5,
                       nflatsequence=3, nflatlamps=4,
                       arcflatexpdiff=15, flatflatexpdiff=3):
    """
     Selects the calibration exposures from a populated exposure table
     that pass consistency requirements on exposure time and have
     LASTSTEP=='all', except for arcs which keep all LASTSTEPS for the time
     being do to how the arc sets are selected later on.

    Args:
        etable (astropy.table.Table): A DESI exposure_table.
        ngoodarcthreshold (int): Number of good arcs in a single sequence
            necessary to be considered a complete set.
        narcsequence (int): Number of arcs in a single arc sequence.
        nflatsequence (int): Number of flats in a single flat sequence for
            a single lamp configuration.
        nflatlamps (int): Number of lamp configurations in a single
            set for flat calibrations.
        arcflatexpdiff (int): Numeric difference in exposure ID between
            the final arc exposure and the first flat exposure in a standard
            calibration sequence.
        flatflatexpdiff (int): Numeric difference in exposure ID between
            the final flat exposure of one lamp and the first flat exposure
            of the next lamp in a standard calibration sequence.

    Returns:
        None or astropy.table.Table: A DESI exposure_table containing only
            the exposures corresponding to the 'best' set of available
            arc+flat calibrations given the input table. Can return None
            if no set is available or just arcs if no valid flat set is
            available.
     """
    exptable = select_valid_calib_exposures(etable=exptable)
    log = get_logger()
    exptable.sort(['EXPID'])
    arc_sequence_sum = int(narcsequence*(1+narcsequence)/2)
    flat_sequence_sum = int(nflatsequence*(1+nflatsequence)/2)
    arcs, flats = [], {0:[], 1:[], 2:[], 3:[]}
    complete_arc_sets, complete_sets = [], []
    log.info(f"Looping over {len(exptable)} rows")
    for erow in exptable:
        if erow['OBSTYPE'] == 'arc':
            if erow['SEQNUM'] == 1:
                print_row_message(f"Identified the start of a new arc sequence:",
                                  erow, exptable.colnames, log.debug)
                arcs, flats = [erow], {0:[], 1:[], 2:[], 3:[]}
            elif len(arcs) > 0 and erow['EXPID'] == arcs[-1]['EXPID']+1 and erow['SEQNUM'] == arcs[-1]['SEQNUM']+1:
                print_row_message(f"Identified the start of a new arc sequence:",
                                  erow, exptable.colnames, log.debug)
                arcs.append(erow)
                if len(arcs) == narcsequence and erow['SEQNUM'] == erow['SEQTOT']:
                    if np.sum([erow['SEQNUM'] for erow in arcs]) == arc_sequence_sum:
                        log.info(f"Identified a complete set of {narcsequence} arcs")
                        complete_arc_set = {'table':vstack(arcs), 'ngood': 0, 'meanbadcams': 0}
                        complete_arc_set['calib_arcs'] = np.all(['calib short arcs all' in arc['PROGRAM'][0].lower() for arc in arcs])
                        nbadcams = 0
                        for arc in arcs:
                            if arc['LASTSTEP'][0] == 'all':
                                complete_arc_set['ngood'] += 1
                                nbadcams += len(decode_camword(arc['BADCAMWORD'][0]))
                                nbadcams += len(parse_badamps(arc['BADAMPS'][0]))
                        complete_arc_set['meanbadcams'] = float(nbadcams)/float(complete_arc_set['ngood'])
                        if complete_arc_set['ngood'] >= ngoodarcthreshold:
                            if complete_arc_set['calib_arcs'] and complete_arc_set['meanbadcams'] == 0 \
                                    and complete_arc_set['ngood'] == narcsequence:
                                log.info(f"Found ideal arc set.")
                            else:
                                log.info(f"Arc set has at least one issue..")
                            complete_arc_sets.append(complete_arc_set)
                        else:
                            arcs, flats = [], {0: [], 1: [], 2: [], 3: []}
            else:
                log.info("Arc wasn't the first in a sequence or next sequentially so " \
                        + "the sequence wasn't complete. Removing any exposures in "
                        + "current exposure list, ignoring current exposure, and moving on.")
                arcs, flats = [], {0:[], 1:[], 2:[], 3:[]}
        elif erow['OBSTYPE'] == 'flat':
            try:
                calibnum = int(str(erow['PROGRAM']).split('desi-calib-')[1][:2])
            except IndexError:
                calibnum = None
            if calibnum is None:
                arcs, flats = [], {0: [], 1: [], 2: [], 3: []}
                continue
            elif erow['SEQNUM'] == 1:
                if calibnum == 0:
                    if len(arcs) != narcsequence \
                            or erow['EXPID'] - arcs[-1]['EXPID'] > arcflatexpdiff:
                        print_row_message(f"Identified the start of a new flat sequence:" \
                                          + "but no valid arcs",
                                           erow, exptable.colnames, log.debug)
                        arcs, flats = [], {0:[], 1:[], 2:[], 3:[]}
                    else:
                        print_row_message(f"Identified the start of a new flat sequence:",
                                          erow, exptable.colnames, log.debug)
                        flats = {0: [erow], 1: [], 2: [], 3: []}
                else:
                    if len(flats[calibnum-1]) != nflatsequence \
                            or erow['EXPID'] - flats[calibnum-1][-1]['EXPID'] > flatflatexpdiff:
                        print_row_message(f"Identified the start of a new flat lamp sequence:" \
                                          + "but no valid previous lamps",
                                          erow, exptable.colnames, log.debug)
                        arcs, flats = [], {0:[], 1:[], 2:[], 3:[]}
                    else:
                        print_row_message(f"Identified the start of a new flat lamp sequence:",
                                          erow, exptable.colnames, log.debug)
                        flats[calibnum] = [erow]
            else:
                if len(flats[calibnum]) == erow['SEQNUM']-1   \
                       and flats[calibnum][-1]['SEQNUM'] == erow['SEQNUM']-1  \
                       and flats[calibnum][-1]['EXPID'] == erow['EXPID']-1:
                    flats[calibnum].append(erow)
                    if calibnum == 3 and len(flats[calibnum]) == nflatsequence:
                        print_row_message(f"Found a complete set",
                                          erow, exptable.colnames, log.info)
                        callist = arcs.copy()
                        ngoodarcs = np.sum([arc['LASTSTEP'][0]=='all' for arc in arcs])
                        ngoodflats = 0
                        for calflatlist in flats.values():
                            ngoodflats += np.sum([flat['LASTSTEP']=='all' for flat in calflatlist])
                            callist.extend(calflatlist)
                        caltable = vstack(callist)
                        del callist

                        complete_set = {'table': caltable.copy(),
                                        'ngoodarcs': ngoodarcs,
                                        'ngoodflats': ngoodflats,
                                        'meanbadcams': 0}
                        caltable = caltable[caltable['LASTSTEP'] == 'all']
                        ## make sure all flat are valid, otherwise don't save set
                        if ngoodflats != nflatsequence*nflatlamps:
                            log.debug(f"At least one bad flat found: {caltable}")
                        else:
                            complete_set['calib_arcs'] = np.all(
                                ['calib' in arc['PROGRAM'][0] for arc in arcs])
                            nbadcams = 0
                            for cal in caltable:
                                nbadcams += len(decode_camword(cal['BADCAMWORD']))
                                nbadcams += len(parse_badamps(cal['BADAMPS']))
                            complete_set['meanbadcams'] = float(nbadcams) \
                                                          / float(len(caltable))
                            if complete_set['ngoodarcs'] >= ngoodarcthreshold:
                                if complete_set['calib_arcs'] \
                                        and complete_set['ngoodarcs'] == narcsequence \
                                        and complete_set['meanbadcams'] == 0:
                                    log.info(f"Found ideal arc-flat set.")
                                else:
                                    log.info(f"Set has at least one issue so continuing the search.")
                                complete_sets.append(complete_set)
                        arcs, flats = [], {0: [], 1: [], 2: [], 3: []}
                else:
                    arcs, flats = [], {0:[], 1:[], 2:[], 3:[]}
        else:
            print_row_message(f"OBSTYPE wasn't arc or flat:",
                              erow, exptable.colnames, log.debug)
            arcs, flats = [], {0:[], 1:[], 2:[], 3:[]}

    log.debug(complete_sets)
    log.debug(len(complete_arc_sets))
    log.debug(len(complete_sets))

    setlist = complete_sets
    if len(complete_sets) == 0:
        if len(complete_arc_sets) == 0:
            return None
        else:
            setlist = complete_arc_sets

    bestset = setlist[0]
    if len(setlist) > 1:
        for calset in setlist[1:]:
            if calset['meanbadcams'] < bestset['meanbadcams']:
                log.info(f"Found arcset with {calset['meanbadcams']} mean bad cameras" \
                        + f" which is less than previous best {bestset['meanbadcams']}")
                bestset = calset
            elif calset['meanbadcams'] == bestset['meanbadcams'] \
                    and calset['ngoodarcs'] > bestset['ngoodarcs']:
                log.info(f"Found arcset with same {calset['meanbadcams']} mean bad cameras" \
                        + f" but more good exposures than previous best " \
                        + f"{calset['ngoodarcs']} > {bestset['meanbadcams']}")
                bestset = calset

    return bestset['table']
