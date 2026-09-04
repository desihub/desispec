"""
desispec.scripts.submit_prod
============================

"""
import yaml
import numpy as np
import os
import sys
import time
import re
import glob
from astropy.table import Table

from desispec.parallel import stdouterr_redirected
from desiutil.log import get_logger
from desispec.io import findfile
from desispec.scripts.proc_night import proc_night
from desispec.scripts.link_calibnight import derive_include_exclude
## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.utils import verify_variable_with_environment, listpath, \
    remove_slurm_environment_variables, load_override_file
from desispec.workflow.exptable import read_minimal_science_exptab_cols
from desispec.workflow.proctable import default_obstypes_for_proctable, \
    get_err_qid
from desispec.workflow.submission import submit_necessary_biasnights_and_preproc_darks
from desispec.scripts.submit_night import submit_night
from desispec.workflow.queue import check_queue_count, get_resubmission_states
import desispec.workflow.proctable
from desispec.util import wrap_long_logs


def get_nights_in_date_range(first_night, last_night):
    """
    Returns a full list of all nights that have an exposure table
    exposure

    Args:
        first_night, int. First night to include (inclusive).
        last_night, int. Last night to include (inclusive).

    Returns:
        nights, list. A list of nights on or after Jan 1 2020 in which data exists at NERSC.
    """
    etab_path = findfile('exptable', night='99999999', readonly=True)
    glob_path = etab_path.replace('99999999', '202?????').replace('999999', '202???')
    etab_files = sorted(glob.glob(glob_path))
    nights = []
    for n in etab_files:
        # - nights are 20YYMMDD
        if re.match(r'^20\d{6}$', n):
            nights.append(int(n))

    nights = np.array(nights)
    nights = nights[((nights >= first_night) & (nights <= last_night))]
    return nights

def get_all_valid_nights(first_night, last_night):
    """
    Returns a full list of all nights that have at least one valid science
    exposure

    Args:
        first_night, int. First night to include (inclusive).
        last_night, int. Last night to include (inclusive).

    Returns:
        nights, list. A list of nights on or after Jan 1 2020 in which data exists at NERSC.
    """
    fulletab = read_minimal_science_exptab_cols()
    nights = np.unique(fulletab['NIGHT'])
    nights = nights[((nights>=first_night)&(nights<=last_night))]
    return nights

def get_all_science_nights_for_prod(production_yaml, verbose=False):
    """
    Derives all the nights with valid science exposures that should be processed
    based on a production yaml file and returns a list of int nights. The yaml
    file must contain either NIGHTS or FIRST_NIGHT and LAST_NIGHT.

    Args:
        production_yaml (str or dict): Production yaml or pathname of the
            yaml file that defines the production.
        verbose (bool): Whether to be verbose in log outputs.

    Returns:
        nights, list. A list of nights on or after Jan 1 2020 in which data exists at NERSC.
    """
    log = get_logger()
    ## If production_yaml not loaded, load the file
    if isinstance(production_yaml, str):
        if not os.path.exists(production_yaml):
            raise IOError(f"Prod yaml file doesn't exist: {production_yaml} not found.")
        with open(production_yaml, 'rb') as yamlfile:
            config = yaml.safe_load(yamlfile)
    else:
        config = production_yaml

    all_nights, first_night = None, None
    if 'NIGHTS' in config and 'LAST_NIGHT' in config:
        log.error(f"Both NIGHTS and LAST_NIGHT specified. Using NIGHTS "
                  + f"and ignoring LAST_NIGHT.")
    if 'NIGHTS' in config:
        all_nights = np.array(list(config['NIGHTS'])).astype(int)
        if verbose:
            log.info(f"Setting all_nights to NIGHTS: {all_nights}")
            log.info("Setting first_night to earliest night in NIGHTS:"
                     + f" {np.min(all_nights)}")
        first_night = np.min(all_nights)
        if verbose:
            log.info("Setting last_night to latest night in NIGHTS: "
                     + f"{np.max(all_nights)}")
        last_night = np.max(all_nights)
    elif 'LAST_NIGHT' in config:
        last_night = int(config['LAST_NIGHT'])
        if verbose:
            log.info(f"Setting last_night to LATEST_NIGHT: {last_night}")
    else:
        raise ValueError("Either NIGHT or LAST_NIGHT required in yaml "
                         + f"file {production_yaml}")

    if first_night is None:
        if 'FIRST_NIGHT' in config:
            first_night = int(config['FIRST_NIGHT'])
            if verbose:
                log.info(f"Setting first_night to FIRST_NIGHT: {first_night}")
        else:
            if verbose:
                log.info("Setting first_night to earliest in a normal prod: 20201214")
            first_night = 20201214

    if all_nights is None:
        # all_nights = get_nights_in_date_range(first_night, last_night)
        if verbose:
            log.info("Populating all_nights with all of the nights with valid science "
                     + f"exposures between {first_night} and {last_night} inclusive")
        all_nights = get_all_valid_nights(first_night, last_night)

    all_nights = np.sort(all_nights).tolist()
    log.info(wrap_long_logs(f"All nights in production: {all_nights}"))

    return all_nights

def get_nights_to_process(production_yaml, verbose=False):
    """
    Derives the nights that need to be processed based on a production yaml file and
    processing tables that exist. The yaml file must contain either NIGHTS or
    FIRST_NIGHT and LAST_NIGHT.

    Args:
        production_yaml (str or dict): Production yaml or pathname of the
            yaml file that defines the production.
        verbose (bool): Whether to be verbose in log outputs.

    Returns:
        nights, list. A list of nights on or after Jan 1 2020 in which data exists at NERSC.
    """
    log = get_logger()
    all_nights = get_all_science_nights_for_prod(production_yaml=production_yaml, verbose=verbose)

    log.info(f"Assuming nights with science jobs in proctable are complete and removing from the list of nights to process.")
    nights_to_process, nights_with_proctable = [], dict()
    for night in all_nights[::-1]:
        ## If proctable exists, file it for further testing about whether the night is completed or not
        pfile = findfile('proctable', night=night, readonly=True)
        if os.path.exists(pfile):
            nights_with_proctable[night] = pfile
        else:
            nights_to_process.append(night)

    ## Because of the reverse order in the loop above, this dict is in reverse chronological order
    ## Since we submit science nights in chronological order, we want to check the proctables starting
    ## with the latest night and stop at the first one that has science jobs, as we expect all of the
    ## earlier nights to also include science exposures (ie be complete).
    ## However, instead of exiting, we keep looping but add the earlier complete nights to skipped_nights
    ## since they've already been processed and we want to report them as skipped instead of just silently skipping them.
    skipped_nights = []
    need_to_check = True
    for night, pfile in nights_with_proctable.items():
        ## don't need to open file if need_to_check is False
        ## and also don't need to use desispec.workflow.tableio.load_table here
        ## since that brings extra overhead and only matters for multi-value
        ## columns we don't care about
        ## Note the test is specifically for science jobs and not merely for the
        ## existence of the proctable. Reference nights that are linked to by an
        ## earlier night have their calibrations submitted ahead of time by
        ## submit_early_refnight_calibrations(), which leaves behind a proctable
        ## with calibration jobs but no science jobs. Those nights still need to
        ## be submitted for science here, so don't simplify this to os.path.exists.
        if need_to_check and 'science' not in Table.read(pfile)['OBSTYPE']:
            nights_to_process.append(night)
        else:
            skipped_nights.append(night)
            need_to_check = False

    log.info(wrap_long_logs(f"Skipped the following nights that already had a proctable with science jobs: {sorted(skipped_nights)}"))
    return sorted(nights_to_process)


def get_linkcal_refnight(night):
    """
    Return the reference night that the given night links calibrations from,
    based on that night's override file, if any.

    Args:
        night (int): The night to inspect, in YYYYMMDD format.

    Returns:
        tuple: (refnight, files_to_link) where refnight is an int night or None
            if the night has no override file or no linkcal refnight, and
            files_to_link is the set of calibration filename prefixes that would
            be linked (an empty set when refnight is None).

    Raises:
        Exception: If the override file exists but its linkcal entry can't be
            interpreted. The pathname is logged before the error propagates.
    """
    log = get_logger()

    override_pathname = findfile('override', night=night, readonly=True)
    if not os.path.exists(override_pathname):
        return None, set()

    overrides = load_override_file(filepathname=override_pathname)
    if not overrides or 'calibration' not in overrides:
        return None, set()

    cal_override = overrides['calibration']
    if 'linkcal' not in cal_override or 'refnight' not in cal_override['linkcal']:
        return None, set()

    linkcal = cal_override['linkcal']
    ## Resolve include/exclude exactly as submit_linkcal_jobs() does. A
    ## malformed override file is fatal, but this inspects the override file of
    ## every night in the production, so name the offending file rather than
    ## leaving a bare error from deep inside derive_include_exclude.
    try:
        files_to_link, _ = derive_include_exclude(linkcal.get('include', None),
                                                 linkcal.get('exclude', None))
        refnight = int(linkcal['refnight'])
    except Exception as err:
        log.critical(f"Could not interpret the linkcal entry in {override_pathname}")
        raise
    return refnight, files_to_link


def get_refnights_needing_early_calibration(nights, verbose=False):
    """
    Identify reference nights that must have their calibrations submitted before
    the nights that link to them are submitted.

    Normally a night links its calibrations from an earlier night, which the
    chronological submission order handles for free. Occasionally an override
    file points *forward* in time, e.g. when a night's flats are bad but the
    following night's are good. In that case the reference night's calibrations
    have to be submitted first so that the earlier night's linkcal job can be
    given a cross-night dependency on them.

    Args:
        nights (list of int): The nights that will be submitted for processing.
        verbose (bool): Whether to be verbose in log outputs.

    Returns:
        list: A list of (night, needs_bias_first) tuples in the order they
            should be submitted, i.e. any night that another entry depends on
            comes first. needs_bias_first is True when something links
            'biasnight' from that night, in which case its biasnight must be
            submitted on its own before the rest of its calibrations.
    """
    log = get_logger()

    ## needs_bias_first[refnight] is True if any night links biasnight from it
    needs_bias_first = dict()

    def note_edge(refnight, files_to_link):
        """Record that some night links files_to_link from refnight"""
        if 'biasnight' in files_to_link:
            needs_bias_first[refnight] = True

    ## Seed with the reference nights that are later than the night linking to them
    seeds = []
    for night in sorted(nights):
        refnight, files_to_link = get_linkcal_refnight(night)
        if refnight is None or refnight <= night:
            continue
        log.info(f"Night {night} links calibrations {sorted(files_to_link)} from "
                 + f"the later night {refnight}, so {refnight} must have its "
                 + "calibrations submitted first.")
        note_edge(refnight, files_to_link)
        if refnight not in seeds:
            seeds.append(refnight)

    if len(seeds) == 0:
        if verbose:
            log.info("No override files link calibrations from a later night, "
                     + "so no calibrations need to be submitted out of order.")
        return []

    ## Walk the chain of linkcal references from each seed so that anything a
    ## seed itself links from is submitted before the seed. Nights are appended
    ## in post-order, i.e. dependencies first.
    ordered, visited, in_progress = [], set(), set()

    def walk(night):
        if night in visited:
            return
        if night in in_progress:
            ## A cycle cannot be topologically ordered, so there is no correct
            ## submission order to fall back on. Proceeding anyway would submit
            ## one side of the cycle before the calibrations it links from,
            ## which is the very failure this pre-pass exists to prevent, so
            ## stop before anything is submitted.
            msg = (f"Circular linkcal reference detected involving night "
                   + f"{night}: its override chain returns to itself. A cyclic "
                   + "linkcal configuration cannot be ordered, so no "
                   + "calibrations can be submitted. Fix the override files.")
            log.critical(msg)
            raise ValueError(msg)
        in_progress.add(night)
        refnight, files_to_link = get_linkcal_refnight(night)
        if refnight is not None:
            note_edge(refnight, files_to_link)
            walk(refnight)
        in_progress.discard(night)
        visited.add(night)
        if not os.path.exists(findfile('exposure_table', night=night, readonly=True)):
            log.error(f"No exposure table for {night=}, so it can't be submitted "
                      + "for early calibration processing. Skipping it.")
            return
        ordered.append(night)

    for seed in seeds:
        walk(seed)

    return [(night, needs_bias_first.get(night, False)) for night in ordered]


def bias_dependency_available(refnight_ptable, night):
    """
    Test whether a reference night has a job that provides a biasnight.

    An earlier night that links 'biasnight' from this reference night needs
    something on the reference night to depend on that actually produces or
    provides that biasnight. A row merely being present is not enough:

    * submit_biasnight_and_preproc_darks() returns the existing processing table
      untouched when a bias row is already there, so a row left in any state
      needing resubmission (FAILED, MAX_RESUB, DEP_NOT_SUBD, TIMEOUT, ...) by an
      earlier run reaches here with a real queue id.
    * a job whose submission failed is left with LATEST_QID set to get_err_qid().
    * a linkcal row only supplies a biasnight if this night's own override links
      biasnight rather than some other calibration.

    Args:
        refnight_ptable (Table or None): The reference night's processing table,
            as returned by submit_necessary_biasnights_and_preproc_darks().
        night (int): The reference night, used to resolve its own override file
            when judging whether a linkcal row supplies a biasnight.

    Returns:
        bool: True if a usable bias-providing job is present.
    """
    log = get_logger()

    if refnight_ptable is None or len(refnight_ptable) == 0:
        return False

    err_qid = get_err_qid()
    ## Any state the pipeline would resubmit is a state we cannot depend on.
    ## This list already includes 'UNSUBMITTED'.
    unusable_states = set(get_resubmission_states())
    own_links = None
    for row in refnight_ptable:
        jobdesc = str(row['JOBDESC'])
        if jobdesc not in ('biasnight', 'biaspdark', 'linkcal'):
            continue

        ## A job that did not run provides nothing to depend on. Note
        ## LATEST_QID == get_default_qid() with STATUS 'COMPLETED' is fine: that
        ## means the outputs already existed so no job needed submitting.
        status = str(row['STATUS']).upper()
        if status in unusable_states:
            log.warning(f"Ignoring {jobdesc} job on {night} because its state "
                        + f"{status} means it did not successfully run.")
            continue
        if int(row['LATEST_QID']) == err_qid:
            log.warning(f"Ignoring {jobdesc} job on {night} because its "
                        + f"LATEST_QID is the error value {err_qid}.")
            continue

        if jobdesc in ('biasnight', 'biaspdark'):
            return True

        ## A linkcal only helps if this night links biasnight from elsewhere
        if own_links is None:
            own_links = get_linkcal_refnight(night)[1]
        if 'biasnight' in own_links:
            return True
        log.warning(f"Ignoring linkcal job on {night} because that override "
                    + f"links {sorted(own_links)}, which does not include "
                    + "biasnight.")

    return False


def submit_early_refnight_calibrations(nights, logpath, specprod=None,
                                       queue=None, reservation=None,
                                       dry_run_level=0, refnights=None):
    """
    Submit the calibrations for any reference night that an earlier night links
    its calibrations from, so that those calibrations exist before the earlier
    night is submitted.

    Args:
        nights (list of int): The nights that will be submitted for processing.
        logpath (str): Directory in which to write the per-night logs.
        specprod (str, optional): Name of the spectroscopic production.
        queue (str, optional): The Slurm queue to submit the jobs to.
        reservation (str, optional): The reservation to submit jobs to.
        dry_run_level (int, optional): Default is 0. Passed to proc_night.
        refnights (list, optional): The (night, needs_bias_first) tuples from
            get_refnights_needing_early_calibration(). Pass the caller's already
            computed list so that the caller and this function cannot disagree
            about which nights need early calibration. Derived here if None.

    Returns:
        list: The nights whose calibrations were submitted, in submission order.
    """
    log = get_logger()

    if refnights is None:
        refnights = get_refnights_needing_early_calibration(nights,
                                                            verbose=True)
    if len(refnights) == 0:
        return []

    log.info(wrap_long_logs("Submitting calibrations ahead of the normal "
                            + f"chronological order for {refnights=}"))

    ## Calibration-only processing, so that no science jobs are submitted out of
    ## chronological order. determine_calibrations_to_proc() drops science
    ## exposures itself, so removing them here doesn't change which calibrations
    ## get selected, it only keeps determine_science_to_proc() from seeing them.
    calib_obstypes = [obstype for obstype in default_obstypes_for_proctable()
                      if obstype != 'science']

    ## Each reference night is submitted in two stages, and the nights are
    ## visited in the dependency order that discovery returned, so anything a
    ## night links from has already been submitted by the time it is reached.
    ##
    ## Stage A, for a night that an earlier night links 'biasnight' from, submits
    ## that night's biasnight on its own. It has to precede stage B for the same
    ## night, because the darknight generation in stage B spans nights: it calls
    ## submit_necessary_biasnights_and_preproc_darks(), which loops over the
    ## surrounding nights and can reach the earlier linking night, submitting
    ## that night's linkcal job. That job can only pick up a cross-night
    ## dependency if this night's bias already exists.
    ##
    ## Stage B submits the remaining calibrations for the night.
    ##
    ## Passing only 'zero' in stage A keeps
    ## submit_necessary_biasnights_and_preproc_darks() from looking at any night
    ## other than the reference night.
    submitted_nights = []
    for night, needs_bias_first in refnights:
        if needs_bias_first:
            log.info(f"Submitting the biasnight for {night=} on its own before "
                     + "the rest of its calibrations, since an earlier night "
                     + "links biasnight from it.")
            if dry_run_level >= 4:
                log.info(f"{dry_run_level=} so not submitting the biasnight. "
                         + f"Would have submitted it for {night=}")
            else:
                logfile = os.path.join(logpath, f'night-{night}-biasnight.log')
                with stdouterr_redirected(logfile):
                    refnight_ptable = submit_necessary_biasnights_and_preproc_darks(
                        reference_night=night, proc_obstypes=['zero'],
                        ## camword/badcamword are re-derived from the exposure
                        ## table, these are only the fallback for a night with
                        ## no exposures
                        camword='a0123456789', badcamword=None,
                        exp_table_pathname=findfile('exposure_table', night=night),
                        proc_table_pathname=findfile('processing_table', night=night),
                        specprod=specprod, dry_run_level=dry_run_level,
                        queue=queue, reservation=reservation)

                ## Confirm a usable bias dependency actually landed. It won't if
                ## the reference night has no zeros, in which case there is
                ## nothing for the earlier night to link biasnight from and its
                ## linkcal would be submitted with no dependency, linking
                ## against a biasnight that never gets made.
                ## Check the returned table rather than re-reading it from disk:
                ## the readonly path is a separate read-only mount that can lag
                ## behind a write that just happened, and this table was only
                ## written moments ago.
                if bias_dependency_available(refnight_ptable, night):
                    log.info(f"Completed the biasnight submission for {night=}.")
                else:
                    log.critical(f"No bias job was submitted for reference "
                                 + f"{night=}, so the nights linking biasnight "
                                 + "from it have nothing to depend on. Check "
                                 + "that it has valid zeros.")
                    raise RuntimeError("Failed to submit the biasnight for "
                                       + f"reference {night=} that an earlier "
                                       + "night links biasnight from")

        log.info(f"Submitting calibrations for reference {night=}")
        if dry_run_level >= 4:
            log.info(f"{dry_run_level=} so not running desi_proc_night. "
                     + f"Would have run calibrations for {night=}")
            submitted_nights.append(night)
            continue

        ## Belt-and-suspenders: reset the processing table cache to force a re-read.
        desispec.workflow.proctable.reset_tilenight_ptab_cache()

        time.sleep(2)  # Sleep to ensure any file system changes have time to propagate
        logfile = os.path.join(logpath, f'night-{night}-calibsonly.log')
        with stdouterr_redirected(logfile):
            proc_night(night=night, proc_obstypes=calib_obstypes,
                       z_submit_types=None, no_redshifts=True,
                       dry_run_level=dry_run_level,
                       queue=queue, reservation=reservation)
        submitted_nights.append(night)
        log.info(f"Completed the calibration submission for {night=}.")

    return submitted_nights


def submit_production(production_yaml, queue_threshold=4500, dry_run_level=False):
    """
    Interprets a production_yaml file and submits the respective nights for processing
    within the defined production. The yaml file must contain SPECPROD and either NIGHTS or FIRST_NIGHT and LAST_NIGHT.

    Args:
        production_yaml (str): Pathname of the yaml file that defines the production.
        queue_threshold (int): The number of jobs for the current user in the queue
            at which the script stops submitting new jobs.
        dry_run_level (int, optional): Default is 0. Should the jobs written to the processing table actually be submitted
            for processing. This is passed directly to desi_proc_night.

    Returns:
        None.
    """
    log = get_logger()
    ## Load the yaml file
    if not os.path.exists(production_yaml):
        raise IOError(f"Prod yaml file doesn't exist: {production_yaml} not found.")
    with open(production_yaml, 'rb') as yamlfile:
        conf = yaml.safe_load(yamlfile)

    ## Unset Slurm environment variables set when running in scrontab
    remove_slurm_environment_variables()

    ## Make sure the specprod matches, if not set it to that in the file
    if 'SPECPROD' not in conf:
        raise ValueError(f"SPECPROD required in yaml file {production_yaml}")
    specprod = str(conf['SPECPROD']).lower()
    specprod = verify_variable_with_environment(var=specprod, var_name='specprod',
                                                env_name='SPECPROD')

    ## Define the user
    user = os.environ['USER']

    ## Look for sentinal
    sentinel_file = os.path.join(os.environ['DESI_SPECTRO_REDUX'],
                                 os.environ['SPECPROD'], 'run',
                                 'prod_submission_complete.txt')
    if os.path.exists(sentinel_file):
        log.info(f"Sentinel file {sentinel_file} exists, therefore all "
                 + f"nights already submitted.")
        return 0

    ## Load the nights to process
    all_nights = get_nights_to_process(production_yaml=conf, verbose=True)

    ## Load the other parameters for running desi_proc_night
    if 'THRU_NIGHT' in conf:
        thru_night = int(conf['THRU_NIGHT'])
        log.info(f"Setting thru_night to THRU_NIGHT: {thru_night}")
    else:
        thru_night = np.max(all_nights)
        log.warning(f"Setting thru_night to last night: {thru_night}")

    ## If not specified, run "cumulative" redshifts, otherwise do
    ## as directed
    no_redshifts = False
    if 'Z_SUBMIT_TYPES' in conf:
        z_submit_types_str = str(conf['Z_SUBMIT_TYPES'])
        if z_submit_types_str.lower() in ['false', 'none']:
            z_submit_types = None
            no_redshifts = True
        else:
            z_submit_types = [ztype.strip().lower() for ztype in
                                   z_submit_types_str.split(',')]
    else:
        z_submit_types = ['cumulative']

    if 'SURVEYS' in conf:
        surveys_str = str(conf['SURVEYS'])
        if surveys_str.lower() in ['false', 'none']:
            surveys = None
        else:
            surveys = [survey.strip().lower() for survey in
                       surveys_str.split(',')]
    else:
        surveys = None

    ## Bring in the queue and reservation information, if any
    if 'QUEUE' in conf:
        queue = conf['QUEUE']
    else:
        queue = 'regular'

    if 'RESERVATION' in conf:
        reservation = str(conf['RESERVATION'])
        if reservation.lower() == 'none':
            reservation = None
    else:
        reservation = None

    ## Let user know what was defined
    if z_submit_types is not None:
        log.info(f'Using z_submit_types: {z_submit_types}')
    if surveys is not None:
        log.info(f'Using surveys: {surveys}')
    log.info(f'Using queue: {queue}')
    if reservation is not None:
        log.info(f'Using reservation: {reservation}')

    ## Define log location
    logpath = os.path.join(os.environ['DESI_SPECTRO_REDUX'],
                          os.environ['SPECPROD'], 'run', 'logs')
    if dry_run_level < 4:
        os.makedirs(logpath, exist_ok=True)
    else:
        log.info(f"{dry_run_level=} so not creating {logpath}")

    ## Get the number of jobs in the queue before submitting anything. In dryrun
    ## mode this is the only time it is checked, to properly simulate stopping if
    ## the queue is too full without rechecking for jobs we never submitted.
    ## Otherwise the main loop below rechecks it before each night.
    num_in_queue = check_queue_count(user=user, include_scron=False,
                                     dry_run_level=dry_run_level)

    ## Submit the calibrations for any reference night that an earlier night
    ## links its calibrations from, so that they exist by the time that earlier
    ## night is submitted below. These nights aren't removed from all_nights;
    ## only their calibrations are submitted here, and they are submitted again
    ## in the loop below for their science exposures, if they have any.
    ##
    ## Discover before testing the queue, so that a production with no
    ## forward-pointing override is never held up by queue pressure it doesn't
    ## care about.
    refnights_needed = get_refnights_needing_early_calibration(all_nights,
                                                               verbose=True)
    if len(refnights_needed) > 0 and num_in_queue > queue_threshold:
        ## These calibrations are a prerequisite for the chronological loop
        ## below, and that loop re-checks the queue independently. If it were
        ## allowed to run after this pre-pass was skipped, a queue that drained
        ## in between would let an override night be submitted with no
        ## reference night calibrations to link. Stop instead, leaving the
        ## sentinel unwritten so a later invocation redoes this in order.
        log.warning(wrap_long_logs(
            f"{num_in_queue} jobs in the queue > {queue_threshold}, so the "
            + "reference night calibrations that must be submitted before the "
            + f"rest of the production ({refnights_needed}) cannot be "
            + "submitted now. Stopping so that a later invocation can submit "
            + "them in the correct order."))
        return 0

    early_calib_nights = submit_early_refnight_calibrations(
        nights=all_nights, logpath=logpath, specprod=specprod,
        queue=queue, reservation=reservation, dry_run_level=dry_run_level,
        refnights=refnights_needed)

    ## Do the main processing
    finished = False
    processed_nights = []
    log.info(wrap_long_logs(f"Processing {all_nights=}"))
    for night in sorted(all_nights):
        ## If the queue is too full, stop submitting nights
        ## don't keep checking if in dry run mode since we're not submitting new jobs
        if dry_run_level < 1:
            num_in_queue = check_queue_count(user=user, include_scron=False,
                                            dry_run_level=dry_run_level)
        else:
            log.info(f"{dry_run_level=} so not checking queue count each iteration. "
                     + f"Would have checked for user {user}.")

        ## In Jura the largest night had 115 jobs, to be conservative we submit
        ## up to 4500 jobs (out of a 5000 limit) by default
        if num_in_queue > queue_threshold:
            log.info(f"{num_in_queue} jobs in the queue > {queue_threshold},"
                     + " so stopping the job submissions.")
            break

        ## We don't expect exposure tables to change during code execution here
        ## but we do expect processing tables to evolve, so clear that cache
        log.info(f"Processing {night=}")

        ## Belt-and-suspenders: reset the processing table cache to force a re-read.
        ## This shouldn't be necessary, but resetting the cache is conservative.
        desispec.workflow.proctable.reset_tilenight_ptab_cache()

        if dry_run_level < 4:
            time.sleep(2)  # Sleep to ensure any file system changes have time to propagate
            logfile = os.path.join(logpath, f'night-{night}.log')
            with stdouterr_redirected(logfile):
                proc_night(night=night, z_submit_types=z_submit_types,
                           no_redshifts=no_redshifts,
                           complete_tiles_thrunight=thru_night,
                           surveys=surveys, dry_run_level=dry_run_level,
                           queue=queue, reservation=reservation)
        else:
            log.info(f"{dry_run_level=} so not running desi_proc_night. "
                     + f"Would have run for {night=}")

        processed_nights.append(night)
        # proc_night(night=None, proc_obstypes=None, z_submit_types=None,
        #            queue=None, reservation=None, system_name=None,
        #            exp_table_pathname=None, proc_table_pathname=None,
        #            override_pathname=None, update_exptable=False,
        #            dry_run_level=0, dry_run=False, no_redshifts=False,
        #            ignore_proc_table_failures=False,
        #            dont_check_job_outputs=False,
        #            dont_resubmit_partial_jobs=False,
        #            tiles=None, surveys=None, science_laststeps=None,
        #            all_tiles=False, specstatus_path=None, use_specter=False,
        #            no_cte_flats=False, complete_tiles_thrunight=None,
        #            all_cumulatives=False, daily=False, specprod=None,
        #            path_to_data=None, exp_obstypes=None, camword=None,
        #            badcamword=None, badamps=None, exps_to_ignore=None,
        #            sub_wait_time=0.1, verbose=False,
        #            dont_require_cals=False,
        #            psf_linking_without_fflat=False,
        #            still_acquiring=False)
        log.info(f"Completed {night=}.")
    else:
        ## I.e. if the above loop didn't "break" because of exceeding the queue
        ## and all nights finished
        finished = True
        # write the sentinel
        if dry_run_level < 4:
            with open(sentinel_file, 'w') as sentinel:
                all_prod_nights = get_all_science_nights_for_prod(production_yaml=production_yaml,
                                                                  verbose=False)
                sentinel.write(f"All done with processing for {production_yaml}\n")
                sentinel.write(f"Nights processed: {all_prod_nights}\n")
        else:
            log.info(f"{dry_run_level=} so not creating {sentinel_file}")

    if len(early_calib_nights) > 0:
        log.info(wrap_long_logs("Submitted calibrations ahead of the normal order"
                                + f" for the following nights: {early_calib_nights}"))
    log.info(wrap_long_logs(f"Processed the following nights: {processed_nights}"))
    if finished:
        log.info('\n\n\n')
        log.info("All nights submitted")
