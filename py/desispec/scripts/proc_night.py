"""
desispec.scripts.proc_night
=============================

"""
from desispec.io import findfile
from desispec.scripts.link_calibnight import derive_include_exclude
from desispec.workflow.calibration_selection import \
    determine_calibrations_to_proc
from desispec.workflow.science_selection import determine_science_to_proc, \
    get_tiles_cumulative
from desiutil.log import get_logger
import numpy as np
import os
import sys
import time
import re
from socket import gethostname
from astropy.table import Table, vstack

from desispec.scripts.update_exptable import update_exposure_table
from desispec.workflow.tableio import load_table, load_tables, write_table
from desispec.workflow.utils import sleep_and_report, \
    verify_variable_with_environment, load_override_file
from desispec.workflow.timing import what_night_is_it, during_operating_hours
from desispec.workflow.exptable import get_last_step_options, \
    read_minimal_science_exptab_cols
from desispec.workflow.proctable import default_obstypes_for_proctable, \
    erow_to_prow, default_prow, read_minimal_tilenight_proctab_cols
from desispec.workflow.submission import submit_linkcal_jobs, \
    submit_necessary_biasnights_and_preproc_darks
from desispec.workflow.processing import define_and_assign_dependency, \
    create_and_submit, \
    submit_tilenight_and_redshifts, \
    generate_calibration_dict, \
    night_to_starting_iid, make_joint_prow, \
    set_calibrator_flag, make_exposure_prow, \
    all_calibs_submitted, \
    update_and_recursively_submit, update_accounted_for_with_linking, \
    submit_redshifts, check_darknight_deps_and_update_prow
from desispec.workflow.queue import update_from_queue, any_jobs_need_resubmission, \
    get_resubmission_states
from desispec.io.util import decode_camword, difference_camwords, \
    create_camword, replace_prefix, erow_to_goodcamword, camword_union


def proc_night(night=None, proc_obstypes=None, z_submit_types=None,
               queue=None, reservation=None, system_name=None,
               exp_table_pathname=None, proc_table_pathname=None,
               override_pathname=None, update_exptable=False,
               dry_run_level=0, n_nights_darks=None, dry_run=False, no_redshifts=False,
               ignore_proc_table_failures = False,
               dont_check_job_outputs=False, dont_resubmit_partial_jobs=False,
               tiles=None, surveys=None, science_laststeps=None,
               all_tiles=False, specstatus_path=None, use_specter=False,
               no_cte_flats=False, no_darknight=False, complete_tiles_thrunight=None,
               all_cumulatives=False, daily=False, specprod=None,
               path_to_data=None, exp_obstypes=None, camword=None,
               badcamword=None, badamps=None, exps_to_ignore=None,
               sub_wait_time=0.1, verbose=False, dont_require_cals=False,
               psf_linking_without_fflat=False, no_resub_failed=False,
               no_resub_any=False, still_acquiring=False):
    """
    Process some or all exposures on a night. Can be used to process an entire
    night, or used to process data currently available on a given night using
    the '--daily' flag.

    Args:
        night (int): The night of data to be processed. Exposure table must exist.
        proc_obstypes (list or np.array, optional): A list of exposure OBSTYPE's
            that should be processed (and therefore added to the processing table).
        z_submit_types (list of str):
            The "group" types of redshifts that should be submitted with each
            exposure. If not specified, default for daily processing is
            ['cumulative', 'pernight-v0']. If false, 'false', or [], then no
            redshifts are submitted.
        queue (str, optional): The name of the queue to submit the jobs to.
            Default is "realtime".
        reservation (str, optional): The reservation to submit jobs to.
            If None, it is not submitted to a reservation.
        system_name (str): batch system name, e.g. cori-haswell, cori-knl,
            perlmutter-gpu
        exp_table_pathname (str): Full path to where to exposure tables are stored,
            including file name.
        proc_table_pathname (str): Full path to where to processing tables to be
            written, including file name
        override_pathname (str): Full path to the override file.
        update_exptable (bool): If true then the exposure table is updated.
            The default is False.
        dry_run_level (int, optional): If nonzero, this is a simulated run. Default is 0.
            0 which runs the code normally.
            1 writes all files but doesn't submit any jobs to Slurm.
            2 writes tables but doesn't write scripts or submit anything.
            3 Doesn't write or submit anything but queries Slurm normally for job status.
            4 Doesn't write, submit jobs, or query Slurm.
            5 Doesn't write, submit jobs, or query Slurm; instead it makes up the status of the jobs.
        n_nights_darks (int, optional): Number of nights of darks to process into a darknight.
            Default is None, which uses the function's default.
        dry_run (bool, optional): When to run without submitting scripts or
            not. If dry_run_level is defined, then it over-rides this flag.
            dry_run_level not set and dry_run=True, dry_run_level is set to 3
            (no scripts generated or run). Default for dry_run is False.
        no_redshifts (bool, optional): Whether to submit redshifts or not.
            If True, redshifts are not submitted.
        ignore_proc_table_failures (bool, optional): True if you want to submit
            other jobs even the loaded processing table has incomplete jobs in
            it. Use with caution. Default is False.
        dont_check_job_outputs (bool, optional): Default is False. If False,
            the code checks for the existence of the expected final data
            products for the script being submitted. If all files exist and
            this is False, then the script will not be submitted. If some
            files exist and this is False, only the subset of the cameras
            without the final data products will be generated and submitted.
        dont_resubmit_partial_jobs (bool, optional): Default is False. Must be
            used with dont_check_job_outputs=False. If this flag is False, jobs
            with some prior data are pruned using PROCCAMWORD to only process
            the remaining cameras not found to exist.
        tiles (array-like, optional): Only submit jobs for these TILEIDs.
        surveys (array-like, optional): Only submit science jobs for these
            surveys (lowercase)
        science_laststeps (array-like, optional): Only submit jobs for exposures
            with LASTSTEP in these science_laststeps (lowercase)
        all_tiles (bool, optional): Default is False. Set to NOT restrict to
            completed tiles as defined by the table pointed to by specstatus_path.
        specstatus_path (str, optional): Default is
            $DESI_SURVEYOPS/ops/tiles-specstatus.ecsv. Location of the
            surveyops specstatus table.
        use_specter (bool, optional): Default is False. If True, use specter,
            otherwise use gpu_specter by default.
        no_cte_flats (bool, optional): Default is False. If False, cte flats
            are used if available to correct for cte effects.
        no_darknight (bool, optional): Default is False. If True, do not
            submit darknight job even if darks are available.
        complete_tiles_thrunight (int, optional): Default is None. Only tiles
            completed on or before the supplied YYYYMMDD are considered
            completed and will be processed. All complete tiles are submitted
            if None or all_tiles is True.
        all_cumulatives (bool, optional): Default is False. Set to run
            cumulative redshifts for all tiles even if the tile has observations
            on a later night.
        specprod: str. The name of the current production. If used, this will
            overwrite the SPECPROD environment variable.
        daily: bool. Flag that sets other flags for running this script for the
            daily pipeline.
        path_to_data: str. Path to the raw data.
        exp_obstypes: str or comma separated list of strings. The exposure
            OBSTYPE's that you want to include in the exposure table.
        camword: str. Camword that, if set, alters the set of cameras that will
            be set for processing. Examples: a0123456789, a1, a2b3r3,
            a2b3r4z3. Note this is only true for new exposures being
            added to the exposure_table in 'daily' mode.
        badcamword: str. Camword that, if set, will be removed from the camword
            defined in camword if given, or the camword inferred from
            the data if camword is not given. Note this is only true
            for new exposures being added to the exposure_table
            in 'daily' mode.
        badamps: str. Comma seperated list of bad amplifiers that should not
            be processed. Should be of the form "{camera}{petal}{amp}",
            i.e. "[brz][0-9][ABCD]". Example: 'b7D,z8A'. Note this is
            only true for new exposures being added to the
            exposure_table in 'daily' mode.
        sub_wait_time: int. Wait time in seconds between submission loops.
            Default 0.1 seconds.
        verbose: bool. True if you want more verbose output, false otherwise.
            Current not propagated to lower code, so it is only used in the
            main daily_processing script itself.
        dont_require_cals: bool. Default False. If set then the code doesn't
            require either a valid set of calibrations or a valid override file
            to link to calibrations in order to proceed with science processing.
        psf_linking_without_fflat: bool. Default False. If set then the code
            will NOT raise an error if asked to link psfnight calibrations
            without fiberflatnight calibrations.
        no_resub_failed: bool. Set to True if you do NOT want to resubmit
            jobs with Slurm status 'FAILED' by default. Default is False.
        no_resub_any: bool. Set to True if you do NOT want to resubmit
            jobs. Default is False.
        still_acquiring: bool. If True, assume more data might be coming, e.g.
            wait for additional exposures of latest tile.  If False, auto-derive
            True/False based upon night and current time. Primarily for testing.
    """
    ## Get logger
    log = get_logger()
    log.info(f'----- Processing {night} at {time.asctime()} -----')
    log.info(f"SLURM_JOB_ID={os.getenv('SLURM_JOB_ID')} on {gethostname()}")

    ## Inform user of how some parameters will be used
    if camword is not None:
        log.info(f"Note custom {camword=} will only be used for new exposures"
                 f" being entered into the exposure_table, not all exposures"
                 f" to be processed.")
    if badcamword is not None:
        log.info(f"Note custom {badcamword=} will only be used for new exposures"
                 f" being entered into the exposure_table, not all exposures"
                 f" to be processed.")
    if badamps is not None:
        log.info(f"Note custom {badamps=} will only be used for new exposures"
                 f" being entered into the exposure_table, not all exposures"
                 f" to be processed.")

    ## Reconcile the dry_run and dry_run_level
    if dry_run and dry_run_level == 0:
        dry_run_level = 3
    elif dry_run_level > 0:
        dry_run = True

    ## If running in daily mode, change a bunch of defaults
    if daily:
        ## What night are we running on?
        true_night = what_night_is_it()
        if night is not None:
            night = int(night)
            if true_night != night:
                log.info(f"True night is {true_night}, but running daily for {night=}")
        else:
            night = true_night

        if science_laststeps is None:
            science_laststeps = ['all', 'skysub', 'fluxcal']

        if z_submit_types is None and not no_redshifts:
            z_submit_types = ['cumulative']

        ## still_acquiring is flag to determine whether to process the last tile in the exposure table
        ## or not. This is used in daily mode when processing and exiting mid-night.
        ## override still_acquiring==False if daily mode during observing hours
        if during_operating_hours(dry_run=dry_run) and (true_night == night):
            if not still_acquiring:
                log.info(f'Daily mode during observing hours on current night, so assuming that more data might arrive and setting still_acquiring=True')
            still_acquiring = True

        update_exptable = True
        append_to_proc_table = True
        all_cumulatives = True
        all_tiles = True
        complete_tiles_thrunight = None
        ## Default for nightly processing is realtime queue
        if queue is None:
            queue = 'realtime'

    ## Default for normal processing is regular queue
    if queue is None:
        queue = 'regular'
    log.info(f"Submitting to the {queue} queue.")

    ## Set night
    if night is None:
        err = "Must specify night unless running in daily=True mode"
        log.error(err)
        raise ValueError(err)
    else:
        log.info(f"Processing {night=}")

    ## Recast booleans from double negative
    check_for_outputs = (not dont_check_job_outputs)
    resubmit_partial_complete = (not dont_resubmit_partial_jobs)
    require_cals = (not dont_require_cals)
    do_cte_flats = (not no_cte_flats)
    do_darknight = (not no_darknight)

    ## False if not submitting or simulating
    update_proctable = (dry_run_level == 0 or dry_run_level > 3)

    ## cte flats weren't available before 20211130 so hardcode that in
    if do_cte_flats and night < 20211130:
        log.info("Asked to do cte flat correction but before 20211130 no "
                    + "no cte flats are available to do the correction. "
                    + "Code will NOT perform cte flat corrections.")
        do_cte_flats = False
    ## Only do darknight if date is after 20240509 (see desispec issue #2571)
    ## darks weren't linear before 20240510 so hardcode that in
    if do_darknight and night < 20240510:
        log.info("Asked to do darknight, but before 20240510 "
                    + "some cameras weren't linear in their darks. "
                    + "Code will NOT create a darknight.")
        do_darknight = False

    ###################
    ## Set filenames ##
    ###################
    ## Ensure specprod is set in the environment and that it matches user
    ## specified value if given
    specprod = verify_variable_with_environment(specprod, var_name='specprod',
                                                env_name='SPECPROD')

    ## Determine where the exposure table will be written
    if exp_table_pathname is None:
        exp_table_pathname = findfile('exposure_table', night=night)
    if not os.path.exists(exp_table_pathname) and not update_exptable:
        raise IOError(f"Exposure table: {exp_table_pathname} not found. Exiting this night.")

    ## Determine where the processing table will be written
    if proc_table_pathname is None:
        proc_table_pathname = findfile('processing_table', night=night)
    proc_table_path = os.path.dirname(proc_table_pathname)
    if dry_run_level < 3:
        os.makedirs(proc_table_path, exist_ok=True)

    ## Determine where the unprocessed data table will be written
    unproc_table_pathname = replace_prefix(proc_table_pathname, 'processing', 'unprocessed')

    ## Require cal_override to exist if explcitly specified
    if override_pathname is None:
        override_pathname = findfile('override', night=night, readonly=True)
    elif not os.path.exists(override_pathname):
        raise IOError(f"Specified override file: "
                      f"{override_pathname} not found. Exiting this night.")

    #######################################
    ## Define parameters based on inputs ##
    #######################################
    ## If science_laststeps not defined, default is only LASTSTEP=='all' exposures
    if science_laststeps is None:
        science_laststeps = ['all']
    else:
        laststep_options = get_last_step_options()
        for laststep in science_laststeps:
            if laststep not in laststep_options:
                raise ValueError(f"Couldn't understand laststep={laststep} "
                                 + f"in science_laststeps={science_laststeps}.")
    log.info(f"Processing exposures with the following LASTSTEP's: {science_laststeps}")

    ## Define the group types of redshifts you want to generate for each tile
    if no_redshifts:
        log.info(f"no_redshifts set, so ignoring {z_submit_types=}")
        z_submit_types = None

    if z_submit_types is None:
        log.info("Not submitting scripts for redshift fitting")
    else:
        for ztype in z_submit_types:
            if ztype not in ['cumulative', 'pernight-v0', 'pernight', 'perexp']:
                raise ValueError(f"Couldn't understand ztype={ztype} "
                                 + f"in z_submit_types={z_submit_types}.")
        log.info(f"Redshift fitting with redshift group types: {z_submit_types}")

    ## Identify OBSTYPES to process
    if proc_obstypes is None:
        proc_obstypes = default_obstypes_for_proctable()

    #############################
    ## Start the Actual Script ##
    #############################
    ## If running in daily mode, or requested, then update the exposure table
    ## This reads in and writes out the exposure table to disk
    if update_exptable:
        log.info("Running update_exposure_table.")
        update_exposure_table(night=night, specprod=specprod,
                              exp_table_pathname=exp_table_pathname,
                              path_to_data=path_to_data, exp_obstypes=exp_obstypes,
                              camword=camword, badcamword=badcamword, badamps=badamps,
                              exps_to_ignore=exps_to_ignore,
                              dry_run_level=dry_run_level, verbose=verbose)
        log.info("Done with update_exposure_table.\n\n")

    ## Combine the table names and types for easier passing to io functions
    table_pathnames = [exp_table_pathname, proc_table_pathname]
    table_types = ['exptable', 'proctable']

    ## Load in the files defined above
    etable, init_ptable = load_tables(tablenames=table_pathnames, tabletypes=table_types)
    full_etable = etable.copy()

    ## Set default camword for linkcal and biaspdark jobs that don't rely on specific exposures
    if camword is None:
        if len(etable) > 0:
            camword = camword_union(etable['CAMWORD'])
        else:
            camword = 'a0123456789'

    ## Now that the exposure table is updated, check if we need to run biases and/or preproc darks
    if n_nights_darks is None:
        n_nights_after_darks = None
        n_nights_before_darks = None
    else:
        n_nights_after_darks = int((n_nights_darks-1) // 2)
        n_nights_before_darks = n_nights_darks - 1 - n_nights_after_darks
    proc_biasdark_obstypes = ('zero' in proc_obstypes or 'dark' in proc_obstypes)
    no_bias_job = (len(init_ptable) == 0
               or ('biasnight' not in init_ptable['JOBDESC'] and 'biaspdark' not in init_ptable['JOBDESC']) )
    should_submit = ((not still_acquiring)
                     or (len(etable) > 2 and np.sum(etable['OBSTYPE']=='dark') > 0
                         and np.sum(etable['OBSTYPE']=='arc') > 0 and np.sum(etable['OBSTYPE']=='zero') > 9)
                    )
    returned_ptable = None
    if proc_biasdark_obstypes and no_bias_job and should_submit:
        ## This will populate the processing table with the biases and preproc dark job if
        ## it needed to submit them. It will do it for future and past nights relevant for
        ## the current night's dark nights.
        log.info("Running submit_necessary_biasnights_and_preproc_darks")
        if n_nights_before_darks is None:
            kwargs = {}
        else:
            kwargs = {'n_nights_before': n_nights_before_darks, 'n_nights_after': n_nights_after_darks}
        returned_ptable = submit_necessary_biasnights_and_preproc_darks(reference_night=night, proc_obstypes=proc_obstypes,
                                                                        camword=camword, badcamword=badcamword, badamps=badamps,
                                                                        exp_table_pathname=exp_table_pathname,
                                                                        proc_table_pathname=proc_table_pathname,
                                                                        specprod=specprod, path_to_data=path_to_data,
                                                                        sub_wait_time=sub_wait_time, dry_run_level=dry_run_level,
                                                                        queue=queue, system_name=system_name,
                                                                        psf_linking_without_fflat=psf_linking_without_fflat,
                                                                        **kwargs)
        log.info("Done with submit_necessary_biasnights_and_preproc_darks.\n\n")
    elif proc_biasdark_obstypes and (not still_acquiring) and (not no_bias_job):
        log.info("Running submit_necessary_biasnights_and_preproc_darks to process any additional darks")
        kwargs = {'n_nights_before': 0,  'n_nights_after': 0}
        returned_ptable = submit_necessary_biasnights_and_preproc_darks(reference_night=night, proc_obstypes=proc_obstypes,
                                                                        camword=camword, badcamword=badcamword, badamps=badamps,
                                                                        exp_table_pathname=exp_table_pathname,
                                                                        proc_table_pathname=proc_table_pathname,
                                                                        specprod=specprod, path_to_data=path_to_data,
                                                                        sub_wait_time=sub_wait_time, dry_run_level=dry_run_level,
                                                                        queue=queue, system_name=system_name,
                                                                        psf_linking_without_fflat=psf_linking_without_fflat,
                                                                        **kwargs)

    ## Load in the updated processing table if saved to disk, otherwise use what is in memory
    if dry_run_level > 2:
        if returned_ptable is None:
            ptable = init_ptable
        else:
            ptable = returned_ptable
    else:
        ptable = load_table(tablename=proc_table_pathname, tabletype='proctable')

    ## Quickly exit if we haven't processed the biasprdark job yet and we should have
    jobtypes_requested = ('zero' in proc_obstypes or 'dark' in proc_obstypes)
    job_exists = np.any(np.isin(np.array([b'biasnight', b'biaspdark', b'linkcal']), ptable['JOBDESC'].data))
    # ## ptables not saved for levels 3 and over, so if still acquiring, assume not yet available
    # ## otherwise assume it is available
    #expect_job_exist = ( dry_run_level<3 or (still_acquiring and dry_run_level>=3) )
    if require_cals and jobtypes_requested and not job_exists:
        log.critical("Bias and preproc dark job not found in processing table. "
                    + "We will need to wait for darks to be processed. "
                    + f"Exiting {night=}.")
        ## If still acquiring new data in daily mode, don't exit with error code
        ## But do exit
        log.info(f'Stopping at {time.asctime()}\n')
        if still_acquiring:
            if len(ptable) > 0:
                processed = np.isin(full_etable['EXPID'],
                                    np.unique(np.concatenate(ptable['EXPID'])))
                unproc_table = full_etable[~processed]
            else:
                unproc_table = full_etable

            return ptable, unproc_table
        else:
            sys.exit(1)

    ## For I/O efficiency, pre-populate exposure table and processing table caches
    ## of all nights if doing cross-night redshifts so that future per-night "reads"
    ## will use the cache.
    if z_submit_types is not None and 'cumulative' in z_submit_types:
        ## this shouldn't need to change since we've already updated the exptab
        read_minimal_science_exptab_cols()
        ## this would become out of date for the current night except
        ## write_table will keep it up to date
        read_minimal_tilenight_proctab_cols()

    ## Cut on OBSTYPES
    log.info(f"Processing the following obstypes: {proc_obstypes}")
    good_types = np.isin(np.array(etable['OBSTYPE']).astype(str), proc_obstypes)
    etable = etable[good_types]

    ## Update processing table
    tableng = len(ptable)
    if tableng > 0:
        if update_proctable:
            ptable = update_from_queue(ptable, dry_run_level=dry_run_level)
        if dry_run_level < 3:
            write_table(ptable, tablename=proc_table_pathname, tabletype='proctable')
        if any_jobs_need_resubmission(ptable['STATUS']) and not no_resub_any:
            ## Try up to two times to resubmit failures, afterwards give up
            ## unless explicitly told to proceed with the failures
            ## Note after 2 resubmissions, the code won't resubmit anymore even
            ## if given ignore_proc_table_failures
            log.info("Job failures were detected. Resubmitting those jobs "
                     + "before continuing with new submissions.")

            ptable, nsubmits, nbad = update_and_recursively_submit(ptable,
                                                             no_resub_failed=no_resub_failed,
                                                             max_resubs=2,
                                                             ptab_name=proc_table_pathname,
                                                             dry_run_level=dry_run_level,
                                                             reservation=reservation)

            if any_jobs_need_resubmission(ptable['STATUS']):
                if not ignore_proc_table_failures:
                    err = "Some jobs have an incomplete job status. This script " \
                          + "will not fix them. You should remedy those first. "
                    log.error(err)
                    ## if the failures are in calibrations, then crash since
                    ## we need them for any new jobs
                    if any_jobs_need_resubmission(ptable['STATUS'][ptable['CALIBRATOR'] > 0]):
                        err += "To proceed anyway use "
                        err += "'--ignore-proc-table-failures'. Exiting."
                        raise AssertionError(err)
                else:
                    log.warning("Some jobs have an incomplete job status, but "
                          + "you entered '--ignore-proc-table-failures'. This "
                          + "script will not fix them. "
                          + "You should have fixed those first. Proceeding...")
        ## Short cut to exit faster if all science exposures have been processed
        ## but only if we have successfully processed the calibrations
        good_etab = etable[etable['LASTSTEP']=='all']
        terminal_cal_reached = False
        if 'nightlyflat' in ptable['JOBDESC']:
            terminal_cal_reached = True
        elif np.sum(good_etab['OBSTYPE']=='flat') < 12 and not still_acquiring \
                and 'psfnight' in ptable['JOBDESC']:
            terminal_cal_reached = True
        # scisel = ptable['OBSTYPE'] == 'science'
        # if np.sum(scisel) > 0:
        #     ptable_expids = set(np.concatenate(ptable['EXPID'][scisel]))
        # else:
        #     ptable_expids = set()
        etable_expids = set(etable['EXPID'][etable['OBSTYPE'] == 'science'])
        if terminal_cal_reached:
            if len(etable_expids) == 0:
                log.info(f"No science exposures yet. Exiting at {time.asctime()}.")
                return ptable, None
            # elif len(etable_expids.difference(ptable_expids)) == 0:
            #     log.info("All science EXPID's already present in processing table, "
            #              + f"nothing to run. Exiting at {time.asctime()}.")
            #     return ptable, None

        int_id = np.max(ptable['INTID'])+1
    else:
        int_id = night_to_starting_iid(night=night)

    ################### Determine What to Process ###################
    ## Load calibration_override_file
    overrides = load_override_file(filepathname=override_pathname)
    cal_override = {}
    if 'calibration' in overrides:
        cal_override = overrides['calibration']

    ## Determine calibrations that will be linked
    files_to_link = None
    if 'linkcal' in cal_override and (len(ptable) == 0 or 'linkcal' not in ptable['JOBDESC']):
        proccamword = difference_camwords(camword, badcamword)
        ptable, files_to_link = submit_linkcal_jobs(night, ptable, cal_override=cal_override,
                        psf_linking_without_fflat=psf_linking_without_fflat, proccamword=proccamword,
                        dry_run_level=dry_run_level, queue=queue, reservation=reservation,
                        check_outputs=check_for_outputs, system_name=system_name)
        if len(ptable) > 0 and dry_run_level < 3:
            write_table(ptable, tablename=proc_table_pathname, tabletype='proctable')
            sleep_and_report(sub_wait_time,
                             message_suffix=f"to slow down the queue submission rate",
                             dry_run=dry_run, logfunc=log.info)

    ## Identify what calibrations have been done
    calibjobs = generate_calibration_dict(ptable, files_to_link)

    ## Determine the appropriate set of calibrations
    ## Only run if we haven't already linked or done fiberflatnight's
    cal_etable = etable[[]]
    if not all_calibs_submitted(calibjobs['accounted_for'], do_cte_flats, do_darknight):
        cal_etable = determine_calibrations_to_proc(etable,
                                                    do_cte_flats=do_cte_flats,
                                                    still_acquiring=still_acquiring)

    ## Determine the appropriate science exposures
    sci_etable, tiles_to_proc = determine_science_to_proc(
                                        etable=etable, tiles=tiles,
                                        surveys=surveys, laststeps=science_laststeps,
                                        all_tiles=all_tiles,
                                        ignore_last_tile=still_acquiring,
                                        complete_tiles_thrunight=complete_tiles_thrunight,
                                        specstatus_path=specstatus_path)

    ## For cumulative redshifts, identify tiles for which this is the last
    ## night that they were observed
    tiles_cumulative = get_tiles_cumulative(sci_etable, z_submit_types,
                                            all_cumulatives, night)

    ################### Process the data ###################
    ## Process Calibrations
    ## For now assume that a linkcal job links all files and we therefore
    ## don't need to submit anything more.
    def create_submit_add_and_save(prow, proctable, check_outputs=check_for_outputs,
                                   extra_job_args=None):
        log.info(f"\nProcessing: {prow}\n")
        prow = create_and_submit(prow, dry_run=dry_run_level, queue=queue,
                                 reservation=reservation,
                                 strictly_successful=True,
                                 check_for_outputs=check_outputs,
                                 resubmit_partial_complete=resubmit_partial_complete,
                                 system_name=system_name,
                                 use_specter=use_specter,
                                 extra_job_args=extra_job_args)
        ## Add the processing row to the processing table
        proctable.add_row(prow)
        if len(proctable) > 0 and dry_run_level < 3:
            write_table(proctable, tablename=proc_table_pathname, tabletype='proctable')
        sleep_and_report(sub_wait_time,
                         message_suffix=f"to slow down the queue submission rate",
                         dry_run=dry_run, logfunc=log.info)
        return prow, proctable

    ## Actually process the calibrations
    ## Only run if we haven't already linked or done fiberflatnight's
    if not all_calibs_submitted(calibjobs['accounted_for'], do_cte_flats, do_darknight):
        ptable, calibjobs, int_id = submit_calibrations(cal_etable, ptable,
                                                cal_override, calibjobs,
                                                int_id, night, files_to_link,
                                                create_submit_add_and_save,
                                                n_nights_before_darks=n_nights_before_darks,
                                                n_nights_after_darks=n_nights_after_darks,
                                                proc_table_path=os.path.dirname(proc_table_pathname),
                                                do_cte_flats=do_cte_flats, do_darknight=do_darknight)

    ## Require some minimal level of calibrations to process science exposures
    if require_cals and not all_calibs_submitted(calibjobs['accounted_for'], do_cte_flats, do_darknight):
        err = (f"Exiting because not all calibration files accounted for "
               + f"with links or submissions and require_cals is True.")
        log.error(err)
        ## If still acquiring new data in daily mode, don't exit with error code
        ## But do exit
        log.info(f'Stopping at {time.asctime()}\n')
        if still_acquiring:
            if len(ptable) > 0:
                processed = np.isin(full_etable['EXPID'],
                                    np.unique(np.concatenate(ptable['EXPID'])))
                unproc_table = full_etable[~processed]
            else:
                unproc_table = full_etable

            return ptable, unproc_table
        else:
            sys.exit(1)

    ## Process Sciences
    ## Loop over new tiles and process them
    unique_ptab_tiles = np.unique(ptable['TILEID'])
    for tile in tiles_to_proc:
        # don't submit cumulative redshifts for lasttile if it isn't in tiles_cumulative
        if z_submit_types is None:
            cur_z_submit_types = []
        else:
            cur_z_submit_types = z_submit_types.copy()
        ## Check if tile has already been processed. If it has, see if all
        ## steps have been submitted
        tnight = None
        if tile in unique_ptab_tiles:
            tile_prows = ptable[ptable['TILEID']==tile]
            ## old proctables have poststdstar, check for that or tilenight
            if 'tilenight' in tile_prows['JOBDESC']:
                tnight = tile_prows[tile_prows['JOBDESC']=='tilenight'][0]
            elif 'poststdstar' in tile_prows['JOBDESC']:
                poststdstars = tile_prows[tile_prows['JOBDESC']=='poststdstar']
                tnight = poststdstars[-1]
                ## Try to gather all the expids, but if that fails just move on
                ## with the EXPID's from the last entry. This only happens in daily
                ## for old proctabs and doesn't matter for cumulative redshifts
                if len(poststdstars) > 1:
                    try:
                        ## If more than one poststdstar, combine all EXPIDs for fake tilenight job
                        tnight['EXPID'] = np.sort(np.concatenate(poststdstars['EXPID']))
                    except:
                        ## Log a warning but don't do anything since this only
                        ## impacts documentation and not data reduction
                        log.warning(f"Tried and failed to populate full EXPIDs for: {dict(tnight)}")

            ## if spectra processed, check for redshifts and remove any found
            if tnight is not None:
                for cur_ztype in cur_z_submit_types.copy():
                    if cur_ztype in tile_prows['JOBDESC']:
                        cur_z_submit_types.remove(cur_ztype)
                ## If the spectra have been processed and all requested redshifts
                ## are done, move on to the next tile
                if len(cur_z_submit_types) == 0:
                    continue

        log.info(f'\n\n################# Submitting {tile} #####################')

        ## Identify the science exposures for the given tile
        tile_etable = sci_etable[sci_etable['TILEID'] == tile]

        ## Should change submit_tilenight_and_redshifts to take erows
        ## but for now will remain backward compatible and use prows
        ## Create list of prows from selected etable rows
        sciences = []
        for erow in tile_etable:
            prow = erow_to_prow(erow)
            prow['INTID'] = int_id
            int_id += 1
            prow['JOBDESC'] = prow['OBSTYPE']
            prow = define_and_assign_dependency(prow, calibjobs)
            sciences.append(prow)

        if 'cumulative' in cur_z_submit_types and tile not in tiles_cumulative:
            cur_z_submit_types.remove('cumulative')

        if len(cur_z_submit_types) == 0:
            cur_z_submit_types = None

        ## if not tilenight, do tilenight and redshifts, otherwise just do redshifts
        if tnight is None:
            ## Process tilenight and redshifts
            ## No longer need to return sciences since this is always the
            ## full set of exposures, but will keep for now for backward
            ## compatibility
            extra_job_args = {}
            if 'science' in overrides and 'tilenight' in overrides['science']:
                extra_job_args = overrides['science']['tilenight']
            else:
                extra_job_args = {}

            extra_job_args['z_submit_types'] = cur_z_submit_types
            extra_job_args['laststeps'] = science_laststeps

            ptable, sciences, int_id = submit_tilenight_and_redshifts(
                                        ptable, sciences, calibjobs, int_id,
                                        dry_run=dry_run_level, queue=queue,
                                        reservation=reservation,
                                        strictly_successful=True,
                                        check_for_outputs=check_for_outputs,
                                        resubmit_partial_complete=resubmit_partial_complete,
                                        system_name=system_name,
                                        use_specter=use_specter,
                                        extra_job_args=extra_job_args)
        elif cur_z_submit_types is not None:
            ## Just process redshifts
            ptable, int_id = submit_redshifts(ptable, sciences, tnight, int_id,
                                              queue=queue, reservation=reservation,
                                              dry_run=dry_run_level,
                                              strictly_successful=True,
                                              check_for_outputs=check_for_outputs,
                                              resubmit_partial_complete=resubmit_partial_complete,
                                              z_submit_types=cur_z_submit_types,
                                              system_name=system_name)

        if len(ptable) > 0 and dry_run_level < 3:
            write_table(ptable, tablename=proc_table_pathname, tabletype='proctable')

        sleep_and_report(sub_wait_time,
                         message_suffix=f"to slow down the queue submission rate",
                         dry_run=dry_run, logfunc=log.info)

        ## Flush the outputs
        sys.stdout.flush()
        sys.stderr.flush()

    ################### Wrap things up ###################
    unproc_table = None
    if len(ptable) > 0:
        ## All jobs now submitted, update information from job queue
        ## If dry_run_level > 3, then Slurm isn't queried
        if update_proctable:
            ptable = update_from_queue(ptable, dry_run_level=dry_run_level)
        if dry_run_level < 3:
            write_table(ptable, tablename=proc_table_pathname, tabletype='proctable')
            ## Now that processing is complete, lets identify what we didn't process
            if len(ptable) > 0:
                processed = np.isin(full_etable['EXPID'], np.unique(np.concatenate(ptable['EXPID'])))
                unproc_table = full_etable[~processed]
            else:
                unproc_table = full_etable
            write_table(unproc_table, tablename=unproc_table_pathname)
    elif dry_run_level < 3 and len(full_etable) > 0:
        ## Done determining what not to process, so write out unproc file
        unproc_table = full_etable
        write_table(unproc_table, tablename=unproc_table_pathname)

    if dry_run_level >= 3:
        log.info(f"{dry_run_level=} so not saving outputs.")
        log.info(f"\n{full_etable=}")
        log.info(f"\nn{ptable=}")
        log.info(f"\n{unproc_table=}")

    if still_acquiring:
        log.info(f"Current submission of exposures "
                 + f"for {night=} are complete except for last tile at {time.asctime()}.\n\n\n\n")
    else:
        log.info(f"All done: Completed submission of exposures for night {night} at {time.asctime()}.\n")

    return ptable, unproc_table


def submit_calibrations(cal_etable, ptable, cal_override, calibjobs, int_id,
                        curnight, files_to_link, create_submit_add_and_save,
                        n_nights_before_darks=None, n_nights_after_darks=None,
                        proc_table_path=None, do_cte_flats=True, do_darknight=True):
    log = get_logger()

    if len(cal_etable) == 0:
        return ptable, calibjobs, int_id

    if len(ptable) > 0:
        ## we use this to check for individual jobs rather than combination
        ## jobs, so only check for scalar jobs where JOBDESC == OBSTYPE
        ## ex. dark, zero, arc, and flat
        explists = ptable['EXPID'][ptable['JOBDESC']==ptable['OBSTYPE']]
        if len(explists) == 0:
            processed_cal_expids = np.array([]).astype(int)
        elif len(explists) == 1:
            processed_cal_expids = np.unique(explists[0]).astype(int)
        else:
            processed_cal_expids = np.unique(np.concatenate(explists).astype(int))
    else:
        processed_cal_expids = np.array([]).astype(int)

    ## Otherwise proceed with submitting the calibrations
    ## Define objects to process
    darks, flats, ctes, cte1s = list(), list(), list(), list()
    zeros = cal_etable[cal_etable['OBSTYPE']=='zero']
    arcs = cal_etable[cal_etable['OBSTYPE']=='arc']
    if 'dark' in cal_etable['OBSTYPE']:
        darks = cal_etable[cal_etable['OBSTYPE']=='dark']
    if 'flat' in cal_etable['OBSTYPE']:
        allflats = cal_etable[cal_etable['OBSTYPE']=='flat']
        is_cte = np.array(['cte' in prog.lower() for prog in allflats['PROGRAM']])
        flats = allflats[~is_cte]
        ctes = allflats[is_cte]

    do_darknight = do_darknight and not calibjobs['accounted_for']['darknight']
    do_badcol = len(darks) > 0 and not calibjobs['accounted_for']['badcolumns']
    have_flats_for_cte = len(ctes) > 0 and len(flats) > 0
    do_cte = do_cte_flats and have_flats_for_cte and not calibjobs['accounted_for']['ctecorrnight']

    ## if do badcol or cte, then submit a ccdcalib job, otherwise submit a
    ## nightlybias job
    if do_darknight or do_badcol or do_cte:
        ######## Submit ccdcalib ########
        ## process dark for bad columns even if we don't have zeros for nightlybias
        ## ccdcalib = darknight(darks) + badcol(dark) + cte correction
        jobdesc = 'ccdcalib'

        if calibjobs[jobdesc] is None:
            ## Define which erow to use to create the processing table row
            all_expids = []
            if do_badcol or not do_cte:
                ## use first exposure if do_badcol or if (do_darknight and not do_cte)
                job_erow = darks[0]
                dark_expid = job_erow['EXPID']
                all_expids.append(dark_expid)
            else:
                job_erow = ctes[-1]
            ## if doing cte correction, create expid list of last 120s flat
            ## and all ctes provided by the calibration selection function
            if do_cte:
                cte_expids = np.array([flats[-1]['EXPID'], *ctes['EXPID']])
                all_expids.extend(cte_expids)
            else:
                cte_expids = None

            prow, int_id = make_exposure_prow(job_erow, int_id,
                                              calibjobs, jobdesc=jobdesc)

            if len(all_expids) > 1:
                prow['EXPID'] = np.array(all_expids)

            prow['CALIBRATOR'] = 1

            if do_darknight:
                prow = check_darknight_deps_and_update_prow(prow, n_nights_before=n_nights_before_darks,
                                                                          n_nights_after=n_nights_after_darks,
                                                                          proc_table_path=proc_table_path)
                #if not enough_darks:
                #    log.critical("Not enough darks for every camera. Stopping submission of calibrations "
                #                 + "until this criteria is met.")
                #    return ptable, calibjobs, int_id


            extra_job_args = {'steps': []}
            if do_darknight:
                extra_job_args['steps'].append('darknight')
                extra_job_args['n_nights_before'] = n_nights_before_darks
                extra_job_args['n_nights_after'] = n_nights_after_darks
            if do_badcol:
                extra_job_args['steps'].append('badcolumn')
                extra_job_args['dark_expid'] = dark_expid
            if do_cte:
                extra_job_args['steps'].append('ctecorr')
                extra_job_args['cte_expids'] = cte_expids
            prow, ptable = create_submit_add_and_save(prow, ptable,
                                                      extra_job_args=extra_job_args)
            calibjobs[prow['JOBDESC']] = prow.copy()
            log.info(f"Submitted ccdcalib job with {do_darknight=}, "
                     + f"{do_badcol=}, {do_cte=}")

    if do_darknight:
        calibjobs['accounted_for']['darknight'] = True
    if do_badcol:
        calibjobs['accounted_for']['badcolumns'] = True
    if do_cte:
        calibjobs['accounted_for']['ctecorrnight'] = True

    ######## Submit arcs and psfnight ########
    if len(arcs)>0 and not calibjobs['accounted_for']['psfnight']:
        arc_prows = []
        for arc_erow in arcs:
            if arc_erow['EXPID'] in processed_cal_expids:
                matches = np.where([arc_erow['EXPID'] in itterprow['EXPID']
                                    for itterprow in ptable])[0]
                if len(matches) == 1:
                    prow = ptable[matches[0]]
                    log.info("Found existing arc prow in ptable, "
                             + f"including it for psfnight job: {list(prow)}")
                    arc_prows.append(prow)
                continue
            prow, int_id = make_exposure_prow(arc_erow, int_id, calibjobs)
            prow, ptable = create_submit_add_and_save(prow, ptable)
            arc_prows.append(prow)

        joint_prow, int_id = make_joint_prow(arc_prows, descriptor='psfnight',
                                             internal_id=int_id)
        ptable = set_calibrator_flag(arc_prows, ptable)
        joint_prow, ptable = create_submit_add_and_save(joint_prow, ptable)
        calibjobs[joint_prow['JOBDESC']] = joint_prow.copy()
        calibjobs['accounted_for']['psfnight'] = True

    ######## Submit flats and nightlyflat ########
    ## If nightlyflat defined we don't need to process more normal flats
    if len(flats) > 0 and not calibjobs['accounted_for']['fiberflatnight']:
        flat_prows = []
        for flat_erow in flats:
            if flat_erow['EXPID'] in processed_cal_expids:
                matches = np.where([flat_erow['EXPID'] in itterprow['EXPID']
                                    for itterprow in ptable])[0]
                if len(matches) == 1:
                    prow = ptable[matches[0]]
                    log.info("Found existing flat prow in ptable, "
                             + f"including it for nightlyflat job: {list(prow)}")
                    flat_prows.append(prow)
                continue

            jobdesc = 'flat'
            prow, int_id = make_exposure_prow(flat_erow, int_id, calibjobs,
                                              jobdesc=jobdesc)
            prow, ptable = create_submit_add_and_save(prow, ptable)
            flat_prows.append(prow)

        joint_prow, int_id = make_joint_prow(flat_prows, descriptor='nightlyflat',
                                             internal_id=int_id)
        ptable = set_calibrator_flag(flat_prows, ptable)
        if 'nightlyflat' in cal_override:
            extra_args = cal_override['nightlyflat']
        else:
            extra_args = None
        joint_prow, ptable = create_submit_add_and_save(joint_prow, ptable,
                                                        extra_job_args=extra_args)
        calibjobs[joint_prow['JOBDESC']] = joint_prow.copy()
        calibjobs['accounted_for']['fiberflatnight'] = True

    ######## Submit cte flats ########
    jobdesc = 'flat'
    for cte_erow in ctes:
        if cte_erow['EXPID'] in processed_cal_expids:
            continue
        prow, int_id = make_exposure_prow(cte_erow, int_id, calibjobs,
                                      jobdesc=jobdesc)
        prow, ptable = create_submit_add_and_save(prow, ptable)

    return ptable, calibjobs, int_id
