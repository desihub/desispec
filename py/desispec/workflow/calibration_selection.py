"""
desispec.workflow.calibration_selection
=======================================

"""
import numpy as np
from astropy.table import Table, vstack
from collections import Counter

from desiutil.log import get_logger
from desispec.io.util import decode_camword, parse_badamps, all_impacted_cameras


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

    ## Selecting cals, so remove science exposures
    if len(etable) > 0:
        etable = etable[etable['OBSTYPE'] != 'science']

    ## If no rows, stop here
    if len(etable) == 0:
        return etable[[]]

    ## Use OBSTYPE, PROGRAM, and EXPTIME to select exposures that match
    ## calibration exposures for those fields
    ## Note even arcs with LASTSTEP='ignore' are retained here
    valid_etable, exptypes = select_valid_calib_exposures(etable)

    ## If 1 dark, 5 arcs, 12 flats, and 3 ctes then we have a candidate set,
    ## so return that it. Otherwise if no new data is coming in we should try
    ## to calibrate with what we have. If still taking data and no complete set,
    ## return nothing so that we swiftly exit and wait for more data.
    if np.sum(exptypes == 'dark') >= 1 and np.sum(exptypes == 'arc') >= 5 \
            and np.sum(exptypes == 'flat') >= 12 \
            and (np.sum(exptypes == 'cteflat') >= 3 or not do_cte_flats):
        log.info(f"Found at least one possible calibration set to test.")
    elif still_acquiring:
        log.info(f"Only found {Counter(exptypes)} calibrations "
                 + f"but still acquiring new data, so stopping here until "
                 + f"more information is known.")
        return etable[[]]
    else:
        log.error(f"Only found {Counter(exptypes)} calibrations "
                  + "and not acquiring new data, so this may be fatal. "
                  + "You may want to consider using an override night.")

    ## Run a more detailed algorithm to ensure we have a complete set of
    ## arcs and a complete set of 3 flats for each of 4 lamps
    best_arcflat_set = find_best_arc_flat_sets(valid_etable)

    ## Create the output table with all zeros, the first valid dark,
    ## the best set of arcs and flats, and all cte flats
    zeros = valid_etable[exptypes=='zero']
    out_table = vstack([zeros, valid_etable[exptypes == 'dark'][:1]])
    if best_arcflat_set is not None:
        out_table = vstack([out_table, best_arcflat_set])
    if do_cte_flats:
        out_table = vstack([out_table, valid_etable[exptypes == 'cteflat'][:3]])

    ## Since the arc set can include LASTSTEP=='ignore', cut those out now
    if len(out_table) > 0:
        out_table = out_table[out_table['LASTSTEP']=='all']

    return out_table



def select_valid_calib_exposures(etable):
    """
     Selects the calibration exposures from a populated exposure table
     that pass consistency requirements on exposure time and have
     LASTSTEP=='all', EXCEPT for arcs which keep all LASTSTEPS for the time
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
    ## Select only good exposures, except for arcs where we keep anything for now
    etable = etable[((etable['OBSTYPE'] == 'arc') | (etable['LASTSTEP'] == 'all'))]

    ## For each exposure, determine if the exptime and obstype are consistent
    ## with a calibration exposure. For arcs and flats also check PROGRAM
    good_exptimes, exptype = [], []
    for erow in etable:
        ## Zero should have 0 exptime
        if erow['OBSTYPE'] == 'zero' \
                and matches_exptime(erow['EXPTIME'], 0.):
            good_exptimes.append(True)
            exptype.append('zero')
        ## Any 300s dark is valid
        elif erow['OBSTYPE'] == 'dark' \
                and matches_exptime(erow['EXPTIME'], 300.):
            good_exptimes.append(True)
            exptype.append('dark')
        ## only 5s arcs labeled "short Arcs all" have correct lamps
        ## for correct duration
        elif erow['OBSTYPE'] == 'arc' \
                and matches_exptime(erow['EXPTIME'], 5.) \
                and erow['PROGRAM'] == 'calib short arcs all':
            good_exptimes.append(True)
            exptype.append('arc')
        ## Only 120s flats labeled 'DESI-CALIB-0*' are correct for flat fielding
        elif erow['OBSTYPE'] == 'flat' \
                and matches_exptime(erow['EXPTIME'], 120.) \
                and 'desi-calib-0' in erow['PROGRAM']:
            good_exptimes.append(True)
            exptype.append('flat')
        ## CTE flats come in 1s, 3s, and 10s varieties
        elif erow['OBSTYPE'] == 'flat' and 'cte' in erow['PROGRAM']:
            if matches_exptime(erow['EXPTIME'],1.) \
                    or matches_exptime(erow['EXPTIME'],3.) \
                    or matches_exptime(erow['EXPTIME'],10.):
                good_exptimes.append(True)
                exptype.append('cteflat')
            else:
                good_exptimes.append(False)
        ## Everything else is invalid (note zeros are handled separately)
        else:
            good_exptimes.append(False)

    ## Only keep those flagged as being consistent with cals
    outtable = etable[np.array(good_exptimes)]
    exptype = np.array(exptype)

    ## Make sure the list creations above are at consistent
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

def format_row_message(message, dictlike, keys=None):
    """
    Standardize formatting for messages about rows of tables.

    Args:
        message (str): Message to be printed verbatim before the key:val pairs
        dictlike (dict or astropy.Table.table): the object holding the values
            for the keys provided in keys, or the defaults given in the
            function.
        keys (list or np.array): iterable of strings that when passed to
            the dictlike object return values.

    Returns formatted string to pass to print, log.debug, log.info, etc.
    """
    ## If no keys given, use these
    if keys is None:
        keys = ['EXPID', 'SEQNUM', 'SEQTOT', 'LASTSTEP', 'BADCAMWORD',
                   'BADAMPS', 'PROGRAM', 'OBSTYPE']
    string = message
    ## Append the pairs specified by keys with values from dictlike to the
    ## input message
    for col in keys:
        string += f'  {col}:{dictlike[col]}'

    return string


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
    log = get_logger()

    ## Verify only exposures that pass sanity checks are checked
    exptable, exptypes = select_valid_calib_exposures(etable=exptable)

    ## Make sure they are in chronologial order
    exptable.sort(['EXPID'])

    ## Calculate sum of an arc and flat sequence for quick-checking that all
    ## sequence numbers are present
    arc_sequence_sum = int(narcsequence*(1+narcsequence)/2)
    flat_sequence_sum = int(nflatsequence*(1+nflatsequence)/2)

    ## Initialize variables to save our sets
    arcs, flats = [], {0:[], 1:[], 2:[], 3:[]}
    complete_arc_sets, complete_sets = [], []

    ## Loop over exposures and if an arc or flat, check to see if it fits into
    ## the pattern for a valid calibration sequence of:
    ## 5 short arcs, 5 long arcs, 4 sets of 3 flats each with different lamp
    log.info(f"Looping over {len(exptable)} rows")
    for erow in exptable:
        log.debug(format_row_message("Processing erow", erow))
        if erow['OBSTYPE'] == 'arc':
            if erow['SEQNUM'] == 1:
                ## if the first arc then we are at the start of a new sequence
                ## remove anything saved and register this as the first arc
                log.debug(format_row_message(f"Identified the start of a new arc sequence:",
                                             erow, exptable.colnames))
                arcs, flats = [erow], {0:[], 1:[], 2:[], 3:[]}
            elif len(arcs) > 0 and erow['EXPID'] == arcs[-1]['EXPID']+1 and erow['SEQNUM'] == arcs[-1]['SEQNUM']+1:
                ## if not the first arc, make sure this arc is compatible with
                ## the last arc. If so, add it
                log.debug(format_row_message(f"Identified additional arc in sequence:",
                                             erow, exptable.colnames))
                arcs.append(erow)
                if len(arcs) == narcsequence and erow['SEQNUM'] == erow['SEQTOT']:
                    if np.sum([erow['SEQNUM'] for erow in arcs]) == arc_sequence_sum:
                        ## if the last arc in the sequence and all exps in the
                        ## sequence are present, do more processing to verify
                        ## this is a good set
                        log.info(f"Identified a complete set of {narcsequence} arcs")

                        ## vstack separate list of rows to avoid astropy #16119 bug modifying inputs.
                        ## This keeps arcs as a list of Rows instead of becoming a list of Tables
                        complete_arc_set = {'table':vstack(list(arcs)), 'ngood': 0, 'meanbadcams': 0}
                        ## calib_arcs True if all PROGRAM's indicate that they
                        ## are calib short arcs
                        complete_arc_set['calib_arcs'] = np.all(['calib short arcs all'
                                                                 in arc['PROGRAM'].lower()
                                                                 for arc in arcs])
                        ## count the number of bad cameras in the set
                        nbadcams = 0
                        for arc in arcs:
                            if arc['LASTSTEP'] == 'all':
                                complete_arc_set['ngood'] += 1
                                badcams = all_impacted_cameras(arc['BADCAMWORD'],
                                                               arc['BADAMPS'])
                                nbadcams += len(badcams)
                        ## find average number of bad cameras only among good
                        ## exposures in the set
                        if complete_arc_set['ngood'] >= ngoodarcthreshold:
                            ## If the number of good exposures is above threshold
                            ## then save the current set as a valid option
                            complete_arc_set['meanbadcams'] = float(nbadcams)/float(complete_arc_set['ngood'])
                            if complete_arc_set['calib_arcs'] and complete_arc_set['meanbadcams'] == 0 \
                                    and complete_arc_set['ngood'] == narcsequence:
                                log.info(f"Found ideal arc set.")
                            else:
                                log.info(f"Arc set has at least one issue..")
                            complete_arc_sets.append(complete_arc_set)
                        else:
                            ## If there aren't enough good arcs then it isn't
                            ## a valid set so negate it and move on
                            log.info(f"Skipping arc set ngood={complete_arc_set['ngood']} < {ngoodarcthreshold};")
                            arcs, flats = [], {0: [], 1: [], 2: [], 3: []}
            else:
                log.info("Arc wasn't the first in a sequence or next sequentially so " \
                        + "the sequence wasn't complete. Removing any exposures in "
                        + "current exposure list, ignoring current exposure, and moving on.")
                arcs, flats = [], {0:[], 1:[], 2:[], 3:[]}
        elif erow['OBSTYPE'] == 'flat':
            ## If it's a flat try to parse the PROGRAM name to identify
            ## the lamp used
            try:
                calibnum = int(str(erow['PROGRAM']).split('desi-calib-')[1][:2])
            except IndexError:
                calibnum = None
            ## if lamp not listed then it isn't a calibration flat, so
            ## restart the search
            if calibnum is None:
                arcs, flats = [], {0: [], 1: [], 2: [], 3: []}
                continue
            elif erow['SEQNUM'] == 1:
                ## If first flat in sequence then check things
                ## about number of exposure ID's it is away from the last
                ## exposure. If first lamp it should be arcflatexpdiff
                ## away from the last arc, if another lamp it will
                ## be flatflatexpdiff away from previous lamp
                if calibnum == 0:
                    if len(arcs) != narcsequence:
                        log.debug(format_row_message(f"Identified the start of a new flat sequence:" \
                                                   + f"but wrong number of arcs {len(arcs)} != {narcsequence}",
                                                     erow, exptable.colnames))
                        arcs, flats = [], {0:[], 1:[], 2:[], 3:[]}
                    elif erow['EXPID'] - arcs[-1]['EXPID'] > arcflatexpdiff:
                        expid_arc = int(arcs[-1]['EXPID'])
                        expid_flat = int(erow['EXPID'])
                        expid_diff = expid_flat - expid_arc
                        log.debug(format_row_message(f"Identified the start of a new flat sequence:" \
                                                   + f"but {expid_diff=} > {arcflatexpdiff}",
                                                      erow, exptable.colnames))
                        dt = 24*60*(erow['MJD-OBS'] - arcs[-1]['MJD-OBS'])
                        log.warning(f'arc {expid_arc} to flat {expid_flat} -> {dt:.1f} minutes')
                        arcs, flats = [], {0:[], 1:[], 2:[], 3:[]}
                    else:
                        log.debug(format_row_message(f"Identified the start of a new flat sequence:",
                                                     erow, exptable.colnames))
                        flats = {0: [erow], 1: [], 2: [], 3: []}
                else:
                    if len(flats[calibnum-1]) != nflatsequence \
                            or erow['EXPID'] - flats[calibnum-1][-1]['EXPID'] > flatflatexpdiff:
                        log.debug(format_row_message(f"Identified the start of a new flat lamp sequence:" \
                                                    + "but no valid previous lamps",
                                                     erow, exptable.colnames))
                        arcs, flats = [], {0:[], 1:[], 2:[], 3:[]}
                    else:
                        log.debug(format_row_message(f"Identified the start of a new flat lamp sequence:",
                                                     erow, exptable.colnames))
                        flats[calibnum] = [erow]
            else:
                ## If not the first flat in a sequence then it should be the next
                ## in the sequence and one exp id away from the last expoure.
                if len(flats[calibnum]) == erow['SEQNUM']-1   \
                       and flats[calibnum][-1]['SEQNUM'] == erow['SEQNUM']-1  \
                       and flats[calibnum][-1]['EXPID'] == erow['EXPID']-1:
                    flats[calibnum].append(erow)
                    ## If the 4th calibration lamp and the last flat in the
                    ## sequence, check the accumulated set to see if it is
                    ## usable or not
                    if calibnum == 3 and \
                            np.sum([erow['SEQNUM'] for erow in flats[calibnum]]) == flat_sequence_sum:
                        log.info(format_row_message(f"Found a complete flat set",
                                                    erow, exptable.colnames))
                        callist = arcs.copy()
                        ngoodarcs = np.sum([arc['LASTSTEP']=='all' for arc in arcs])
                        ngoodflats = 0
                        ## for each lamp, find the number of good flats and
                        ## add the flats to the list of expoures
                        for calflatlist in flats.values():
                            ngoodflats += np.sum([flat['LASTSTEP']=='all' for flat in calflatlist])
                            callist.extend(calflatlist)
                        caltable = vstack(list(callist))  # list wrapper in case astropy changes inputs
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
                            ## if the appropriate number of flats, calculate
                            ## even more quantities including the mean number
                            ## of bad cameras on valid exposures
                            complete_set['calib_arcs'] = np.all(
                                ['calib' in arc['PROGRAM'] for arc in arcs])
                            nbadcams = 0
                            for cal in caltable:
                                badcams = all_impacted_cameras(cal['BADCAMWORD'],
                                                               cal['BADAMPS'])
                                nbadcams += len(badcams)
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
                        ## whether valid or not, this was the last in a sequence
                        ## so reset the sets. If valid it was added to
                        ## complete_sets above
                        arcs, flats = [], {0: [], 1: [], 2: [], 3: []}
                else:
                    ## if flat isn't next in the sequence, then reset the sets
                    arcs, flats = [], {0:[], 1:[], 2:[], 3:[]}
        else:
            ## if obstype isn't arc or flat, ignore it and reset the sets
            log.debug(format_row_message(f"OBSTYPE wasn't arc or flat:",
                                         erow, exptable.colnames))
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
