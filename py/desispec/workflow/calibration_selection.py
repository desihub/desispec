"""
desispec.workflow.calibration_selection
=======================================

"""
import numpy as np
from astropy.table import Table, vstack
from collections import Counter

from desiutil.log import get_logger
from desispec.io.util import decode_camword, parse_badamps, all_impacted_cameras, erow_to_goodcamword


def select_calib_darks(etable):
    """
    Returns subset of etable row(s) with darks to use for badcolumn calibration

    Args:
        etable (astropy.table.Table): a DESI exposure_table

    Returns:
        dark_etable (astropy.table.Table): table of darks to use

    Currently this returns a table of length-1 with a single dark, but in the
    future it could return more than one dark if we found that useful.

    If no good darks are found, return a length-0 table with the same columns
    so that it can still be vstacked with other etable entries.
    """
    # copy input so that we can sort without modifying original
    etable = etable.copy()

    keep = np.where((etable['OBSTYPE']=='dark') & (etable['LASTSTEP'] != 'ignore'))[0]
    etable = etable[keep]

    if len(etable) == 0:
        log = get_logger()
        log.warning('No good dark exposures found in etable')
        return etable

    # count good cameras per row
    num_goodcam = np.zeros(len(etable))
    for i in range(len(etable)):
        cameras = decode_camword(erow_to_goodcamword(etable[i]))
        num_goodcam[i] = len(cameras)

    # sort by number of missing cameras $while preserving EXPID order for ties
    num_missing_cam = np.max(num_goodcam) - num_goodcam
    sorted_indices = np.argsort(num_missing_cam, kind='stable')
    i = sorted_indices[0]

    return etable[i:i+1]  #- Table of length 1, not Row object


def determine_calibrations_to_proc(etable, do_cte_flats=True,
                                   still_acquiring=False):
    """
     Selects the calibration exposures that should be processed from a
     populated exposure table.

    Args:
        etable (astropy.table.Table): A DESI exposure_table.
        do_cte_flats (bool, optional): Default is True. If True, cte flats
            are used if available to correct for cte effects.
        still_acquiring (bool): Whether data is still being acquired or the
            provided table is the complete set that will exist for the night.

    Returns:
        astropy.table.Table: A DESI exposure_table only containing the
            calibration exposures that should be processed.
     """
    log = get_logger()
    full_etable = etable.copy()

    ## If no rows, stop here
    if len(full_etable) == 0:
        return full_etable[[]]
    
    ## Selecting cals, so remove science exposures
    cal_etable = full_etable[full_etable['OBSTYPE'] != 'science']

    ## If no rows, stop here
    if len(cal_etable) == 0:
        return full_etable[[]]

    ## Use OBSTYPE, PROGRAM, and EXPTIME to select exposures that match
    ## calibration exposures for those fields
    ## Note even arcs with LASTSTEP='ignore' are retained here
    valid_etable, exptypes = select_valid_calib_exposures(cal_etable)

    ## If no rows, stop here
    if len(valid_etable) == 0:
        return full_etable[[]]

    ## Find if a valid 1s CTE flat
    is_cte_1s = ((exptypes=='cteflat')
                 & matches_exptime(valid_etable['EXPTIME'], exptime=1.))

    ## If 1 dark, 5 arcs, 12 flats, and 3 ctes then we have a candidate set,
    ## so return that it. Otherwise if no new data is coming in we should try
    ## to calibrate with what we have. If still taking data and no complete set,
    ## return nothing so that we swiftly exit and wait for more data.
    if np.sum(exptypes == 'dark') >= 1 and np.sum(exptypes == 'arc') >= 5 \
            and np.sum(exptypes == 'flat') >= 12 \
            and ((np.sum(is_cte_1s) >= 1) or not do_cte_flats):
        log.info(f"Found at least one possible calibration set to test.")
    elif still_acquiring:
        log.info(f"Only found {Counter(exptypes)} calibrations "
                 + f"but still acquiring new data, so stopping here until "
                 + f"more information is known.")
        return etable[[]]
    else:
        log.warning(f"Only found {Counter(exptypes)} calibrations "
                  + "and not acquiring new data, so this may be fatal "
                  + "if you aren't using an override file.")

    ## Run a more detailed algorithm to ensure we have a complete set of
    ## arcs and a complete set of 3 flats for each of 4 lamps
    ## Note this gets all calibrations without exposure_time cut, as it
    ## does it's own selection
    best_arcflat_set = find_best_arc_flat_sets(cal_etable)

    ## Create the output table with all zeros, the best selected dark,
    ## the best set of arcs and flats, and all cte flats
    zeros = valid_etable[exptypes=='zero']
    darks = select_calib_darks(valid_etable)
    out_table = vstack([zeros, darks, best_arcflat_set])

    ## If doing cte flats, select one of each exptime based on proximity to the
    ## last 120s flat
    ## Since we require a 120s flat, only proceed if we have a valid 120s flat
    if do_cte_flats and len(best_arcflat_set) > 0 \
            and np.sum(best_arcflat_set['OBSTYPE']=='flat')>0:
        ## identify the time of the last 120s flat
        lastflattime = np.max(best_arcflat_set['MJD-OBS'][best_arcflat_set['OBSTYPE']=='flat'])
        ## select the cte exposures
        ctes = valid_etable[exptypes == 'cteflat']
        ## loop over exposure times and if any ctes of that time exist,
        ## add the nearest exposure to the last flat
        selected_ctes = []
        for exptime in [10., 3., 1.]:
            ## boolean array, True if exposure time matches
            match_exptime = matches_exptime(ctes['EXPTIME'], exptime=exptime)
            ## if any matches, proceed
            if np.any(match_exptime):
                ## take matches and find the closest one in time to the 120s flat
                matched_ctes = ctes[match_exptime]
                closest = np.argmin(np.abs(matched_ctes['MJD-OBS']-lastflattime))
                selected_ctes.append(matched_ctes[closest])
        ## if we found any ctes, stack them into the table (note selected_ctes
        ## is a list of rows, so need to unpack into the vstack list)
        if len(selected_ctes) > 0:
            out_table = vstack([out_table, *selected_ctes])

    return out_table



def select_valid_calib_exposures(etable, allow_any_laststep=None):
    """
     Selects the calibration exposures from a populated exposure table
     that pass consistency requirements on exposure time and have
     LASTSTEP=='all', EXCEPT for arcs which keep all LASTSTEPS for the time
     being do to how the arc sets are selected later on.

    Args:
        etable (astropy.table.Table): A DESI exposure_table
        allow_any_laststep (list or np.array): List of obstype strings for
            which to allow exposures of any LASTSTEP.

    Returns:
        astropy.table.Table: A DESI exposure_table only containing the
            calibration exposures that pass exposure time cuts and are
            therefore candidates for valid calibration exposures.
        np.array: An array of derived observation types where each index
            corresponds to the row of the returned exposure_table. The
            obseration types include CTE information and aren't redundant with
            column 'OBSTYPE'.
     """
    ## Select only good exposures, except for obstypes in allow_any_laststep
    laststep_mask = (etable['LASTSTEP'] == 'all')
    if allow_any_laststep is not None:
        for obstype in allow_any_laststep:
            laststep_mask |= (etable['OBSTYPE'] == obstype)
    etable = etable[laststep_mask]

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

def matches_exptime(val_or_array, exptime, exptime_tolerance=0.5):
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

def is_complete_set(explist):
    """
     Function that takes a list of exposure rows (or etable) and identifies
     whether they are consistent with a complete set of calibrations of
     a particular type (either arcs or a single flat lamp) based on SEQNUM
     and SEQTOT

    Args:
        explist (list): A list of DESI exposure_table rows for what is expected
            to be a single calibration set with SEQNUM's ranging from 1 to
            SEQTOT in any order.

    Returns:
        (bool): True if the explist has all SEQNUM's up to SEQTOT and all
            SEQTOT's are consistent and the list is the appropriate length.
    """

    if len(explist) == 0:
        return False
    else:
        seqnums, seqtots = [], []
        ## Get all seqnums and seqtots
        for erow in explist:
            seqnums.append(erow['SEQNUM'])
            seqtots.append(erow['SEQTOT'])
        ## this is equal to sum of 1+2+...+SEQTOT
        expected_sum = int(seqtots[0]*(1+seqtots[0])/2)
        ## A full sequence should add up to the expected total, have consistent
        ## SEQTOT, have length of SEQTOT, and have max SEQNUM of SEQTOT
        return (np.all(np.array(seqtots)==seqtots[0])
                and len(explist) == seqtots[0]
                and np.max(seqnums) == seqtots[0]
                and np.sum(seqnums) == expected_sum)

def find_best_arc_flat_sets(exptable, ngoodarcthreshold=3, nflatlamps=4,
                            arcflattimediff=80.):
    """
     Selects the calibration exposures from a populated exposure table.

    Args:
        etable (astropy.table.Table): A DESI exposure_table.
        ngoodarcthreshold (int): Number of good arcs in a single sequence
            necessary to be considered a complete set.
        nflatlamps (int): Number of lamp configurations in a single
            set for flat calibrations.
        arcflattimediff (float): Time difference in minutes between
            the final arc exposure and the first flat exposure or the final
            flat exposure and the first arc exposure

    Returns:
        None or astropy.table.Table: A DESI exposure_table containing only
            the exposures corresponding to the 'best' set of available
            arc+flat calibrations given the input table. Can return empty table
            if no set is available or just arcs if no valid flat set is
            available.
    """
    log = get_logger()

    ## Verify only exposures that pass sanity checks are checked
    exptable, exptypes = select_valid_calib_exposures(etable=exptable,
                                                      allow_any_laststep=['arc'])

    ## Make sure they are in chronologial order
    exptable.sort(['EXPID'])

    ## Initialize variables to save our sets
    arcs, flats = [], {lamp:[] for lamp in range(nflatlamps)}
    complete_arc_sets, complete_flat_sets = [], []

    ## Loop over exposures and if an arc or flat, check to see if it fits into
    ## the pattern for a valid calibration sequence of:
    ## Sequential SEQNUMs leading up to SEQTOT for arcs with
    ## PROGRAM='calib short arcs all' and nflatlamps
    ## independent sets of SEQNUMs leading up to SEQTOT for flats
    ## differentiated by PROGRAM="calib desi-calib-0? leds only"
    log.info(f"Looping over {len(exptable)} rows")
    for erow in exptable:
        obstype, program = str(erow['OBSTYPE']).lower(), str(erow['PROGRAM']).lower()
        seqnum, seqtot = int(erow['SEQNUM']), int(erow['SEQTOT'])
        log.debug(format_row_message("Processing erow", erow))
        if obstype == 'arc' and program == 'calib short arcs all':
            flats = {lamp:[] for lamp in range(nflatlamps)}
            if seqnum == 1:
                ## if the first arc then we are at the start of a new sequence
                ## remove anything saved and register this as the first arc
                log.debug(format_row_message(f"Identified the start of a new arc sequence:",
                                             erow, exptable.colnames))
                arcs = [erow]
            elif len(arcs) > 0 and seqnum == arcs[-1]['SEQNUM']+1:
                ## if not the first arc, make sure this arc is compatible with
                ## the last arc. If so, add it
                log.debug(format_row_message(f"Identified additional arc in sequence:",
                                             erow, exptable.colnames))
                arcs.append(erow)
                if seqnum == seqtot and is_complete_set(arcs):
                    ## if the last arc in the sequence and all exps in the
                    ## sequence are present, do more processing to verify
                    ## this is a good set
                    log.info(f"Identified a complete set of {seqtot} arcs")

                    ## vstack separate list of rows to avoid astropy #16119 bug modifying inputs.
                    ## This keeps arcs as a list of Rows instead of becoming a list of Tables
                    arctable = vstack(list(arcs))

                    ## count the number of good exposures
                    arctable = arctable[arctable['LASTSTEP']=='all']

                    ## If the number of good exposures is above threshold
                    ## then save the current set as a valid option
                    if len(arctable) >= ngoodarcthreshold:
                        ## find average number of bad cameras only among good
                        ## exposures in the set
                        narcbadcams = []
                        for arc in arctable:
                            badcams = all_impacted_cameras(arc['BADCAMWORD'],
                                                           arc['BADAMPS'])
                            narcbadcams.append(len(badcams))

                        ## Create bundled "arcset" with useful comparative
                        ## information
                        arcset = {'table': arctable,
                                  'nbadcams': narcbadcams,
                                  'meanbadcams': np.mean(narcbadcams)}
                        complete_arc_sets.append(arcset)
                    else:
                        ## If there aren't enough good arcs then it isn't
                        ## a valid set so negate it and move on
                        log.info(f"Skipping arc set ngood="
                                 + f"{len(arctable)} < {ngoodarcthreshold}")
                    ## After saving arcset, reset arc and flat builders
                    arcs = []
                elif seqnum == seqtot:
                    log.info("Arc was last in sequence but the set wasn't valid." \
                             + " Removing any exposures in current exposure list,"
                             + " ignoring current exposure, and moving on.")
                    arcs = []
            else:
                log.info("Arc wasn't the first in a sequence or next sequentially so " \
                        + "the sequence wasn't complete. Removing any exposures in "
                        + "current exposure list, ignoring current exposure, and moving on.")
                arcs = []
        elif obstype == 'flat' and program.startswith('calib desi-calib-') \
                and program.endswith(' leds only'):
            arcs = []
            ## If it's a flat try to parse the PROGRAM name to identify
            ## the lamp used
            try:
                lampnum = int(program.split('desi-calib-')[1][:2])
            except IndexError:
                lampnum = None
            ## if lamp not listed then it isn't a calibration flat, so
            ## restart the builders
            if lampnum is None:
                ## if obstype isn't arc or flat, ignore it and reset the builders
                log.debug(format_row_message(f"PROGRAM wasn't correct:",
                                             erow, exptable.colnames))
                flats = {lamp:[] for lamp in range(nflatlamps)}
            elif seqnum != len(flats[lampnum])+1 \
                    or ( len(flats[lampnum]) > 0 and seqnum != flats[lampnum][-1]['SEQNUM']+1 ):
                ## If current seqnum isn't compatible with what is already
                ## in the flat structure for that lamp number, then reset
                ## the entire flat builder
                log.debug(format_row_message(f"flat {seqnum=} but exposures "
                                             + f"already present for "
                                             + f"{lampnum=}: {flats[lampnum]}. "
                                             + f"Resetting flat sequence",
                                             erow, exptable.colnames))
                flats = {lamp:[] for lamp in range(nflatlamps)}
                ## If first in the sequence, start add it and keep searching
                ## If in the middle of the sequence don't add it since it's
                ## clearly not the start of a sequence
                if seqnum == 1:
                    log.debug(format_row_message(f"Identified the start of a "
                                                 + f"new flat lamp sequence:",
                                                 erow, exptable.colnames))
                    flats[lampnum].append(erow)
            else:
                ## If not the first flat in a sequence then it should be the next
                ## in the sequence and one exp id away from the last expoure.
                flats[lampnum].append(erow)

                ## If all lamps have the appropriate number of exposures and
                ## the seqnums matched, then save as a valid series of flats
                is_complete = False
                if seqnum == seqtot:
                    is_complete = np.all([is_complete_set(explist) for explist
                                          in flats.values()])

                if is_complete:
                    log.info(format_row_message(f"Found a complete flat set",
                                                erow, exptable.colnames))
                    callist = []
                    ## for each lamp, find the number of good flats and
                    ## add the flats to the list of expoures
                    for calflatlist in flats.values():
                        callist.extend(calflatlist)
                    flattable = vstack(callist)  # list wrapper in case astropy changes inputs
                    del callist

                    ## make sure all flat are valid, otherwise don't save set
                    ## Note this shouldn't as of 20240301 as we already cut
                    ## on LASTSTEP for flats before this point
                    if not np.all(flattable['LASTSTEP'] == 'all'):
                        log.debug(f"At least one bad flat found: {flattable}")
                    else:
                        ## if the appropriate number of flats,
                        ## add as complete flat set
                        flattable = flattable[flattable['LASTSTEP'] == 'all']
                        complete_flat_sets.append(flattable)
                    ## whether valid or not, this was the last in a sequence
                    ## so reset the sets. If valid it was added to
                    ## complete_sets above
                    flats = {lamp:[] for lamp in range(nflatlamps)}
        else:
            log.debug("Either wasn't an arc or flat or the programs were not " \
                    + "consistent with a calibration exposure. Resetting the "
                    + "arc and flat builders.")
            arcs = []
            flats = {lamp:[] for lamp in range(nflatlamps)}

    log.debug(len(complete_arc_sets))
    log.debug(len(complete_flat_sets))

    ## Loop over arc sets and flat sets and see which ones are compatible in
    ## time to be paired together
    complete_sets = []
    max_mjd_dt = float(arcflattimediff) / (24.*60.)
    for arcset in complete_arc_sets:
        arctable = arcset['table']
        arcstart, arcend = np.min(arctable['MJD-OBS']), np.max(arctable['MJD-OBS'])
        narcbadcams = arcset['nbadcams']
        for flattable in complete_flat_sets:
            flatstart, flatend = np.min(flattable['MJD-OBS']), np.max(flattable['MJD-OBS'])
            ## Only proceed with the arc-flat set if the time difference is less
            ## than the specified difference
            if np.abs(arcstart-flatend) < max_mjd_dt or np.abs(flatstart-arcend) < max_mjd_dt:
                ## If time difference is valid, it is a valid set so
                ## continue to produce info about set
                ## start based on the existing list of badcam numbers from arcs
                ## and add number of bad cameras for the flats
                nbadcams = narcbadcams.copy()
                for flat in flattable:
                    badcams = all_impacted_cameras(flat['BADCAMWORD'],
                                                   flat['BADAMPS'])
                    nbadcams.append(len(badcams))

                if arcend < flatstart:
                    mjd_dt = flatstart - arcend
                elif flatend < arcstart:
                    mjd_dt = arcstart - flatend
                else:
                    ## arcs and flats overlap in time, which is surprising but not necessarily fatal
                    ## could happend if multiple partial sets were taken but combined they are good
                    arc_expids = sorted(arctable['EXPID'])
                    flat_expids = sorted(flattable['EXPID'])
                    log.error(f'Overlapping arc/flat sequences? {arc_expids=} {flat_expids=}')
                    mjd_dt = 0.0

                complete_set = {'table': vstack([arctable, flattable]),
                                'meanbadcams': np.mean(nbadcams),
                                'mjd_dt': mjd_dt}
                complete_sets.append(complete_set)
                arcs = complete_set['table'][complete_set['table']['OBSTYPE'] == 'arc']
                if complete_set['meanbadcams'] == 0 \
                        and len(arcs) == arcs['SEQTOT'][0]:
                    log.info(f"Found ideal arc-flat set.")
                else:
                    log.info(f"Found an arc-flat set but with at least one issue.")

    ## If there are no complete_sets, then fall back to just arc sets
    ## if there are no arc sets either, then immediately return empty table
    setlist = complete_sets
    if len(complete_sets) == 0:
        if len(complete_arc_sets) == 0:
            return exptable[[]]
        else:
            setlist = complete_arc_sets

    ## Select the best set by selecting the one with the fewest meanbadcams
    ## If two tie in meanbadcams, then select the one with more
    ## LASTSTEP='all' arc exposures. If a tie, pick set with smallest
    ## timediff between arcs and flats. If still a tie, choose the first set.
    bestset = setlist[0]
    if len(setlist) > 1:
        for calset in setlist[1:]:
            if calset['meanbadcams'] < bestset['meanbadcams']:
                log.info(f"Found calset with {calset['meanbadcams']} mean bad cameras" \
                        + f" which is less than previous best {bestset['meanbadcams']}")
                bestset = calset
            elif calset['meanbadcams'] == bestset['meanbadcams'] \
                    and len(calset['table']) > len(bestset['table']):
                log.info(f"Found set with same {calset['meanbadcams']} mean bad " \
                        + f"cameras but more good exposures than previous best " \
                        + f"{len(calset['table'])} > {len(bestset['table'])}")
                bestset = calset
            ## last tie-breaker is time difference, which only applies to complete sets not
            ## arc sets, so make sure dict has 'mjd_dt'
            elif ('mjd_dt' in calset) and ('mjd_dt' in bestset) and (calset['mjd_dt'] < bestset['mjd_dt']):
                bestset = calset

    ## return the exposure table with the best selection of arcs and flats
    return bestset['table']
