"""
desispec.scripts.submit_night
=============================

"""
from desiutil.log import get_logger
import numpy as np
import os
import sys
import time
import re
from astropy.table import Table, vstack
## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.scripts.update_exptable import update_exposure_table
from desispec.workflow.tableio import load_tables, write_table
from desispec.workflow.utils import pathjoin, sleep_and_report
from desispec.workflow.timing import what_night_is_it, during_operating_hours
from desispec.workflow.exptable import get_exposure_table_path, \
    get_exposure_table_name, get_last_step_options
from desispec.workflow.proctable import default_obstypes_for_proctable, get_processing_table_path, \
                                        get_processing_table_name, erow_to_prow, table_row_to_dict, \
                                        default_prow
from desispec.workflow.procfuncs import parse_previous_tables, \
    get_type_and_tile, \
    define_and_assign_dependency, create_and_submit, \
    checkfor_and_submit_joint_job, submit_tilenight_and_redshifts, \
    night_to_starting_iid, generate_calibration_dict_and_iid
from desispec.workflow.queue import update_from_queue, any_jobs_not_complete
from desispec.workflow.desi_proc_funcs import get_desi_proc_batch_file_path
from desispec.workflow.redshifts import read_minimal_exptables_columns
from desispec.io.util import decode_camword, difference_camwords, \
    create_camword, parse_badamps

def proc_night(night=None, proc_obstypes=None, z_submit_types=None,
               queue='realtime', reservation=None, system_name=None,
               exp_table_path=None, proc_table_path=None, tab_filetype='csv',
               dry_run_level=0, dry_run=False, no_redshifts=False, error_if_not_available=True,
               append_to_proc_table=False, ignore_proc_table_failures = False,
               dont_check_job_outputs=False, dont_resubmit_partial_jobs=False,
               tiles=None, surveys=None, laststeps=None,
               all_tiles=False, specstatus_path=None, use_specter=False,
               do_cte_flat=False, complete_tiles_thrunight=None,
               all_cumulatives=False,

               daily=False, specprod=None, path_to_data=None, exp_obstypes=None,
               camword=None, badcamword=None, badamps=None,
               exps_to_ignore=None, exp_cadence_time=2, verbose=False):
    """
    Creates a processing table and an unprocessed table from a fully populated exposure table and submits those
    jobs for processing (unless dry_run is set).

    Args:
        night (int): The night of data to be processed. Exposure table must exist.
        proc_obstypes (list or np.array, optional): A list of exposure OBSTYPE's that should be processed (and therefore
            added to the processing table).
        z_submit_types (list of str or comma-separated list of str, optional): The "group" types of redshifts that should be
            submitted with each exposure. If not specified, default for daily processing is
            ['cumulative', 'pernight-v0']. If false, 'false', or [], then no redshifts are submitted.
        queue (str, optional): The name of the queue to submit the jobs to. Default is "realtime".
        reservation (str, optional): The reservation to submit jobs to. If None, it is not submitted to a reservation.
        system_name (str): batch system name, e.g. cori-haswell, cori-knl, perlmutter-gpu
        exp_table_path (str): Full path to where to exposure tables are stored, WITHOUT the monthly directory included.
        proc_table_path (str): Full path to where to processing tables to be written.
        tab_filetype (str, optional): The file extension (without the '.') of the exposure and processing tables.
        dry_run_level (int, optional): If nonzero, this is a simulated run. If dry_run=1 the scripts will be written but not submitted.
            If dry_run=2, the scripts will not be written nor submitted. Logging will remain the same
            for testing as though scripts are being submitted. Default is 0 (false).
        dry_run (bool, optional): When to run without submitting scripts or not. If dry_run_level is defined, then it over-rides
            this flag. dry_run_level not set and dry_run=True, dry_run_level is set to 2 (no scripts
            generated or run). Default for dry_run is False.
        no_redshifts (bool, optional): Whether to submit redshifts or not. If True, redshifts are not submitted.
        error_if_not_available (bool, optional): Default is True. Raise as error if the required exposure table doesn't exist,
            otherwise prints an error and returns.
        append_to_proc_table (bool, optional): True if you want to submit jobs even if a processing table already exists.
            Otherwise jobs will be appended to it. Default is False
        ignore_proc_table_failures (bool, optional): True if you want to submit other jobs even the loaded
            processing table has incomplete jobs in it. Use with caution. Default is False.
        dont_check_job_outputs (bool, optional): Default is False. If False, the code checks for the existence of the expected final
            data products for the script being submitted. If all files exist and this is False,
            then the script will not be submitted. If some files exist and this is False, only the
            subset of the cameras without the final data products will be generated and submitted.
        dont_resubmit_partial_jobs (bool, optional): Default is False. Must be used with dont_check_job_outputs=False. If this flag is
            False, jobs with some prior data are pruned using PROCCAMWORD to only process the
            remaining cameras not found to exist.
        tiles (array-like, optional): Only submit jobs for these TILEIDs.
        surveys (array-like, optional): Only submit science jobs for these surveys (lowercase)
        laststeps (array-like, optional): Only submit jobs for exposures with LASTSTEP in these laststeps (lowercase)
        all_tiles (bool, optional): Default is False. Set to NOT restrict to completed tiles as defined by
            the table pointed to by specstatus_path.
        specstatus_path (str, optional): Default is $DESI_SURVEYOPS/ops/tiles-specstatus.ecsv.
            Location of the surveyops specstatus table.
        use_specter (bool, optional): Default is False. If True, use specter, otherwise use gpu_specter by default.
        do_cte_flat (bool, optional): Default is False. If True, one second flat exposures are processed for cte identification.
        complete_tiles_thrunight (int, optional): Default is None. Only tiles completed
            on or before the supplied YYYYMMDD are considered
            completed and will be processed. All complete
            tiles are submitted if None or all_tiles is True.
        all_cumulatives (bool, optional): Default is False. Set to run cumulative redshifts for all tiles
            even if the tile has observations on a later night.
                specprod: str. The name of the current production. If used, this will overwrite the SPECPROD environment variable.

        daily: bool. Flag that sets other flags for running this script for the daily pipeline.
        specprod: str. The specprod in which you wish to run the processing.
        path_to_data: str. Path to the raw data.
        exp_obstypes: str or comma separated list of strings. The exposure OBSTYPE's that you want to include in the exposure table.
        camword: str. Camword that, if set, alters the set of cameras that will be set for processing.
                      Examples: a0123456789, a1, a2b3r3, a2b3r4z3.
        badcamword: str. Camword that, if set, will be removed from the camword defined in camword if given, or the camword
                         inferred from the data if camword is not given.
        badamps: str. Comma seperated list of bad amplifiers that should not be processed. Should be of the
                      form "{camera}{petal}{amp}", i.e. "[brz][0-9][ABCD]". Example: 'b7D,z8A'
        exp_cadence_time: int. Wait time in seconds between loops over each science exposure. Default 2.
        verbose: bool. True if you want more verbose output, false otherwise. Current not propagated to lower code,
                       so it is only used in the main daily_processing script itself.
    """
    log = get_logger()

    ## Set a flag to determine whether to process the last tile in the exposure table
    ## or not. This is used in daily mode when processing and exiting mid-night.
    ignore_last_tile = False

    ## If running in daily mode, change a bunch of defaults
    if daily:
        ## What night are we running on?
        true_night = what_night_is_it()
        if night is not None:
            night = int(night)
            if true_night != night:
                log.info(f"True night is {true_night}, but running for {night=}")
        else:
            night = true_night

        if during_operating_hours(dry_run=dry_run) and (true_night == night):
            ignore_last_tile = True

        update_exptable = True
        error_if_not_available = False
        do_cte_flat = True
        append_to_proc_table = True
        all_cumulatives = True
        all_tiles = True
        complete_tiles_thrunight = None

    if night is None:
        err = "Must specify night unless running in daily=True mode"
        log.error(err)
        raise ValueError(err)
    else:
        print(f"Processing {night=}")

    ## Recast booleans from double negative
    check_for_outputs = (not dont_check_job_outputs)
    resubmit_partial_complete = (not dont_resubmit_partial_jobs)

    ## Determine where the exposure table will be written
    if exp_table_path is None:
        exp_table_path = get_exposure_table_path(night=night, usespecprod=True)
    name = get_exposure_table_name(night=night, extension=tab_filetype)
    exp_table_pathname = pathjoin(exp_table_path, name)
    if not os.path.exists(exp_table_pathname) and error_if_not_available:
        raise IOError(f"Exposure table: {exp_table_pathname} not found. Exiting this night.")

    ## Determine where the processing table will be written
    if proc_table_path is None:
        proc_table_path = get_processing_table_path()
    os.makedirs(proc_table_path, exist_ok=True)
    name = get_processing_table_name(prodmod=night, extension=tab_filetype)
    proc_table_pathname = pathjoin(proc_table_path, name)

    ## Define the group types of redshifts you want to generate for each tile
    if no_redshifts:
        z_submit_types = None
    else:
        if z_submit_types is None:
            pass
        elif isinstance(z_submit_types, str):
            if z_submit_types.lower() == 'false':
                z_submit_types = None
            elif z_submit_types.lower() == 'none':
                z_submit_types = None
            else:
                z_submit_types = [ztype.strip().lower() for ztype in z_submit_types.split(',')]
                for ztype in z_submit_types:
                    if ztype not in ['cumulative', 'pernight-v0', 'pernight', 'perexp']:
                        raise ValueError(f"Couldn't understand ztype={ztype} in z_submit_types={z_submit_types}.")
        else:
            raise ValueError(f"Couldn't understand z_submit_types={z_submit_types}, type={type(z_submit_types)}.")

    if z_submit_types is None:
        print("Not submitting scripts for redshift fitting")
    else:
        print(f"Redshift fitting with redshift group types: {z_submit_types}")

    ## Reconcile the dry_run and dry_run_level
    if dry_run and dry_run_level == 0:
        dry_run_level = 2
    elif dry_run_level > 0:
        dry_run = True

    ## Check if night has already been submitted and don't submit if it has, unless told to with ignore_existing
    if os.path.exists(proc_table_pathname):
        if not append_to_proc_table:
            print(f"ERROR: Processing table: {proc_table_pathname} already exists and not "+
                  "given flag --append-to-proc-table. Exiting this night.")
            return

    ## Determine where the unprocessed data table will be written
    unproc_table_pathname = pathjoin(proc_table_path, name.replace('processing', 'unprocessed'))

    ## If running in daily mode, or requested, then update the exposure table
    if update_exptable:
        update_exposure_table(night=night, specprod=specprod, exp_table_path=exp_table_path,
                              path_to_data=path_to_data, exp_obstypes=exp_obstypes, camword=camword,
                              badcamword=badcamword, badamps=badamps, tab_filetype='csv',
                              exps_to_ignore=exps_to_ignore, exp_cadence_time=exp_cadence_time,
                              dry_run_level=dry_run_level, verbose=verbose)

    ## Combine the table names and types for easier passing to io functions
    table_pathnames = [exp_table_pathname, proc_table_pathname]
    table_types = ['exptable', 'proctable']

    ## Load in the files defined above
    etable, ptable = load_tables(tablenames=table_pathnames, tabletypes=table_types)
    full_etable = etable.copy()
    orig_ptable = ptable.copy()

    ## Cut on OBSTYPES
    if proc_obstypes is None:
        proc_obstypes = default_obstypes_for_proctable()
    print(f"Processing the following obstypes: {proc_obstypes}")

    good_types = np.isin(np.array(etable['OBSTYPE']).astype(str), proc_obstypes)
    etable = etable[good_types]

    ## divide into calibration and science etables
    issci = (etable['OBSTYPE'] == 'science')
    cal_etable = etable[~issci]
    sci_etable = etable[issci]

    ## Remove any exposure related to the last tile when in daily mode
    ## and during the nightly processing
    if ignore_last_tile and len(sci_etable) > 0:
        last_tile = sci_etable['TILEID'][np.argmax(sci_etable['EXPID'])]
        sci_etable = sci_etable[sci_etable['TILEID'] != last_tile]

    ## filter by TILEID and SURVEY if requested
    if tiles is not None and len(sci_etable) > 0:
        log.info(f'Filtering by tiles={tiles}')
        keep = np.isin(sci_etable['TILEID'], tiles)
        sci_etable = sci_etable[keep]

    if surveys is not None and len(sci_etable) > 0:
        log.info(f'Filtering by surveys={surveys}')
        if 'SURVEY' not in etable.dtype.names:
            raise ValueError(f'surveys={surveys} filter requested, but no SURVEY column in {exp_table_pathname}')

        keep = np.zero(len(sci_etable), dtype=bool)
        # np.isin doesn't work with bytes vs. str from Tables but direct comparison does, so loop
        for survey in surveys:
            keep |= sci_etable['SURVEY'] == survey

        sci_etable = sci_etable[keep]

    ## If asked to do so, only process tiles deemed complete by the specstatus file
    if not all_tiles and len(sci_etable) > 0:
        all_completed_tiles = get_completed_tiles(specstatus_path,
                                              complete_tiles_thrunight=complete_tiles_thrunight)
        keep = np.isin(sci_etable['TILEID'], all_completed_tiles)
        sci_tiles = np.unique(sci_etable['TILEID'][keep])
        log.info(f"Processing completed science tiles: {', '.join(sci_tiles.astype(str))}")
        log.info(f"Filtering by completed tiles retained {len(sci_tiles)}/{sum(np.unique(sci_etable['TILEID'])>0)} science tiles")
        log.info(f"Filtering by completed tiles retained {sum(keep)}/{sum(sci_etable['TILEID']>0)} science exposures")
        sci_etable = sci_etable[keep]

    ## Cut on LASTSTEP
    ## If laststeps not defined, default is only LASTSTEP=='all' exposures for non-tilenight runs
    if laststeps is None:
        laststeps = ['all',]
    else:
        laststep_options = get_last_step_options()
        for laststep in laststeps:
            if laststep not in laststep_options:
                raise ValueError(f"Couldn't understand laststep={laststep} in laststeps={laststeps}.")
    print(f"Processing exposures with the following LASTSTEP's: {laststeps}")
    tilenight_laststeps = laststeps

    good_exps = np.isin(np.array(sci_etable['LASTSTEP']).astype(str), laststeps)
    sci_etable = sci_etable[good_exps]

    ## For cumulative redshifts, identify tiles for which this is the last night that they were observed
    tiles_cumulative = get_tiles_cumulative(sci_etable, z_submit_types, all_cumulatives, night)

    ## Get relevant data from the tables
    calibjobs, internal_id = generate_calibration_dict_and_iid(ptable, night)

    tableng = len(ptable)
    if tableng > 0:
        ptable = update_from_queue(ptable, dry_run=0)
        if dry_run_level < 3:
            write_table(ptable, tablename=proc_table_pathname)
        if any_jobs_not_complete(ptable['STATUS']):
            if not ignore_proc_table_failures:
                print("ERROR: Some jobs have an incomplete job status. This script "
                      + "will not fix them. You should remedy those first. "
                      + "To proceed anyway use '--ignore-proc-table-failures'. Exiting.")
                return
            else:
                print("Warning: Some jobs have an incomplete job status, but "
                      + "you entered '--ignore-proc-table-failures'. This "
                      + "script will not fix them. "
                      + "You should have fixed those first. Proceeding...")
        ptable_expids = np.unique(np.concatenate(ptable['EXPID']))
        if len(set(etable['EXPID']).difference(set(ptable_expids))) == 0:
            print("All EXPID's already present in processing table, nothing to run. Exiting")
            return ptable
    else:
        ptable_expids = np.array([], dtype=int)

    ## Now figure out everything that isn't in the final list, which we'll
    ## Write out to the unproccessed table
    toproc_exps = np.append([cal_etable['EXPID'], sci_etable['EXPID']])
    toprocess = np.isin(full_etable['EXPID'], toproc_exps)
    processed = np.isin(full_etable['EXPID'], ptable_expids)
    unproc_table = full_etable[~(toprocess|processed)]

    ## Done determining what not to process, so write out unproc file
    if dry_run_level < 3:
        write_table(unproc_table, tablename=unproc_table_pathname)

    ## If just starting out and no dark, do the nightlybias
    do_bias = ('bias' in proc_obstypes or 'dark' in proc_obstypes) and num_zeros>0
    if tableng == 0 and np.sum(isdark) == 0 and do_bias:
        print("\nNo dark found. Submitting nightlybias before processing exposures.\n")
        prow = default_prow()
        prow['INTID'] = internal_id
        prow['OBSTYPE'] = 'zero'
        internal_id += 1
        prow['JOBDESC'] = 'nightlybias'
        prow['NIGHT'] = night
        prow['CALIBRATOR'] = 1
        cams = set(decode_camword('a0123456789'))
        for row in unproc_table:
            if row['OBSTYPE'] == 'zero' and 'calib' in row['PROGRAM']:
                proccamword = difference_camwords(row['CAMWORD'], row['BADCAMWORD'])
                cams = cams.intersection(set(decode_camword(proccamword)))
        prow['PROCCAMWORD'] = create_camword(list(cams))
        prow = create_and_submit(prow, dry_run=dry_run_level, queue=queue,
                                 reservation=reservation,
                                 strictly_successful=True,
                                 check_for_outputs=check_for_outputs,
                                 resubmit_partial_complete=resubmit_partial_complete,
                                 system_name=system_name)
        calibjobs['nightlybias'] = prow.copy()
        ## Add the processing row to the processing table
        ptable.add_row(prow)
        ## Write out the processing table
        if dry_run_level < 3:
            write_table(ptable, tablename=proc_table_pathname)
            sleep_and_report(2, message_suffix=f"after nightlybias",
                             dry_run=dry_run)

    ## Identify tiles that still need to be processed.
    ## Since we want to process in chronological order we can't due the
    ## efficient method of differencing sets
    processed_tiles = set(ptable['TILEID'])
    unprocessed_tiles = []
    for tile in sci_etable['TILEID']:
        ## If we haven't prcessed it already and we don't already have it in the
        ## list, then add the tile to the list
        if tile not in processed_tiles and tile not in unprocessed_tiles:
            unprocessed_tiles.append(tile)

    ## Loop over new tiles and process them
    for ii, tile in enumerate(unprocessed_tiles):
        print(f'\n\n##################### {tile} #########################')

        ## Identify the science exposures for the given tile
        sciences = etable[etable['TILEID'] == tile]

        # don't submit cumulative redshifts for lasttile if it isn't in tiles_cumulative
        cur_z_submit_types = z_submit_types.copy()
        if ((z_submit_types is not None) and ('cumulative' in z_submit_types)
            and (tile not in tiles_cumulative)):
            cur_z_submit_types.remove('cumulative')

        ptable, internal_id \
            = submit_tilenight_and_redshifts(ptable, sciences, calibjobs, internal_id,
                                            dry_run=dry_run_level,
                                            queue=queue,
                                            reservation=reservation,
                                            strictly_successful=True,
                                            check_for_outputs=check_for_outputs,
                                            resubmit_partial_complete=resubmit_partial_complete,
                                            z_submit_types=cur_z_submit_types,
                                            system_name=system_name, use_specter=use_specter,
                                            laststeps=tilenight_laststeps)

        if len(ptable) > 0 and dry_run_level < 3:
            write_table(ptable, tablename=proc_table_pathname)

        sleep_and_report(1, message_suffix=f"to slow down the queue submission rate",
                         dry_run=dry_run)

        ## Flush the outputs
        sys.stdout.flush()
        sys.stderr.flush()

    if len(ptable) > 0:
        ## All jobs now submitted, update information from job queue and save
        ptable = update_from_queue(ptable, dry_run=dry_run_level)
        if dry_run_level < 3:
            write_table(ptable, tablename=proc_table_pathname)

    print(f"Completed submission of exposures for night {night}.", '\n\n\n')
    return ptable



# for erow in etable:
#     if erow['OBSTYPE'] == 'science' and erow['EXPTIME'] < 60:
#         good_exptimes.append(False)
#     elif erow['OBSTYPE'] == 'arc' and erow['EXPTIME'] > 8.:
#         good_exptimes.append(False)
#     elif erow['OBSTYPE'] == 'dark' and np.abs(float(erow['EXPTIME']) - 300.) > 1:
#         good_exptimes.append(False)
#     elif erow['OBSTYPE'] == 'flat' and np.abs(float(erow['EXPTIME']) - 120.) > 1:
#         if do_cte_flat and not already_found_cte_flat \
#            and np.abs(float(erow['EXPTIME']) - 1.) < 0.5:
#             good_exptimes.append(True)
#             already_found_cte_flat = True
#         else:
#             good_exptimes.append(False)
#     else:
#         good_exptimes.append(True)



def select_arcs(exptable):
    """
    Select the set of arcs to use for calibrations in a night

    Args:
        exptable, astropy.table.Table. Should contain at least columns: OBSTYPE, EXPTIME,
             LASTSTEP, PROGRAM, SEQNUM, SEQTOT

    Returns:
        arcs, astropy.table.Table. A table containing a subset of the rows of the
             input table, subselected to the best set of arcs to be used for
             calibrations given those available in the input exposure table.
    """
    ## If no valid arcs, return None immediately
    if np.sum(((exptable['OBSTYPE']=='arc') & (exptable['LASTSTEP']=='all'))) < 3:
        return None

    ## Subselect only arcs
    exptable = exptable[exptable['OBSTYPE']=='arc']
    ## Next subselect to short arcs
    exptable = exptable[np.where(np.abs(exptable['EXPTIME']-5.)< 1.)[0]]

    ## If we have designated calibration exposures, search those first
    calibprogram = np.array(['calib' in erow['PROGRAM'] for erow in exptable])
    if np.any(calibprogram):
        calibtable = exptable[calibprogram]
        calibtable.sort(['EXPID'])



def test_select_arcs():
    from astropy.table import Table
    j = Table()
    j["EXPID"] = list(np.arange(0,5)) + list(np.arange(10,15)) \
                 + list(np.arange(20,25)) + list(np.arange(30,35))
    j['SEQNUM'] = np.array([1 + (i % 5) for i in np.arange(20)])
    j['SEQTOT'] = np.ones(20, dtype=int) * 5
    j['LASTSTEP'] = np.array(['ignore']*20)
    j['LASTSTEP'][:7] = 'all'
    j['LASTSTEP'][8:19] = 'all'
    j['PROGRAM'] = np.array(['calib short arcs all']*20)
    j['PROGRAM'][10:] = 'other arc'






def get_completed_tiles(specstatus_path=None, complete_tiles_thrunight=None):
    """
    Uses a tiles-specstatus.ecsv file and selection criteria to determine
    what tiles have beeen completed. Takes an optional argument to point
    to a custom specstatus file. Returns an array of TILEID's.

    Args:
        specstatus_path, str, optional. Default is $DESI_SURVEYOPS/ops/tiles-specstatus.ecsv.
            Location of the surveyops specstatus table.
        complete_tiles_thrunight, int, optional. Default is None. Only tiles completed
            on or before the supplied YYYYMMDD are considered
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

def filter_by_tiles(etable, tiles):
    log = get_logger()
    if tiles is not None:
        log.info(f'Filtering by tiles={tiles}')
        if len(etable) > 0:
            keep = np.isin(etable['TILEID'], tiles)
            etable = etable[keep]
    return etable

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