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

from desiutil.log import get_logger
from desispec.io import findfile
from desispec.scripts.proc_night import proc_night
## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.utils import verify_variable_with_environment, listpath
from desispec.workflow.redshifts import read_minimal_exptables_columns
from desispec.scripts.submit_night import submit_night
from desispec.workflow.queue import check_queue_count


def get_all_nights(first_night, last_night):
    """
    Returns a full list of all nights that have an exposure table
    exposure

    Inputs:
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
        # - nights are 202YMMDD
        if re.match('^202\d{5}$', n):
            nights.append(int(n))

    nights = np.array(nights)
    nights = nights[((nights >= first_night) & (nights <= last_night))]
    return nights

def get_all_valid_nights(first_night, last_night):
    """
    Returns a full list of all nights that have at least one valid science
    exposure

    Inputs:
        first_night, int. First night to include (inclusive).
        last_night, int. Last night to include (inclusive).

    Returns:
        nights, list. A list of nights on or after Jan 1 2020 in which data exists at NERSC.
    """
    # TODO when merged into branch with crossnight dependencies this function changes name
    fulletab = read_minimal_exptables_columns()
    nights = np.unique(fulletab['NIGHT'])
    nights = nights[((nights>=first_night)&(nights<=last_night))]
    return nights


def submit_production(production_yaml, dry_run_level=False):
    """
    Interprets a production_yaml file and submits the respective nights for processing
    within the defined production.

    Args:
        production_yaml (str): Pathname of the yaml file that defines the production.
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

    ## Make sure the specprod matches, if not set it to that in the file
    if 'SPECPROD' not in conf:
        raise ValueError(f"SPECPROD required in yaml file {production_yaml}")
    specprod = str(conf['SPECPROD']).lower()
    specprod = verify_variable_with_environment(var=specprod, var_name='specprod',
                                                env_name='SPECPROD')
    user = os.environ['USER']

    ## Look for sentinal
    sentinel_file = os.path.join(os.environ['DESI_SPECTRO_REDUX'],
                                 os.environ['SPECPROD'], 'run', 'sentinel.txt')
    if os.path.exists(sentinel_file):
        log.info(f"Sentinel file {sentinel_file} exists, therefore all "
                 + f"nights already submitted.")
        return 0

    ## Load the nights to process
    all_nights, first_night = None, None
    if 'NIGHTS' in conf and 'LAST_NIGHT' in conf:
        log.error(f"Both NIGHTS and LAST_NIGHT specified. Using NIGHTS "
                  + f"and ignoring LAST_NIGHT.")
    if 'NIGHTS' in conf:
        all_nights = np.array(list(conf['NIGHTS'])).astype(int)
        log.info(f"Setting all_nights to NIGHTS: {all_nights}")
        log.info("Setting first_night to earliest night in NIGHTS:"
                 + f" {np.min(all_nights)}")
        first_night = np.min(all_nights)
        log.info("Setting last_night to latest night in NIGHTS: "
                 + f"{np.max(all_nights)}")
        last_night = np.max(all_nights)
    elif 'LAST_NIGHT' in conf:
        last_night = int(conf['LAST_NIGHT'])
        log.info(f"Setting last_night to LATEST_NIGHT: {last_night}")
    else:
        raise ValueError("Either NIGHT or LAST_NIGHT required in yaml "
                         + f"file {production_yaml}")

    if first_night is None:
        if 'FIRST_NIGHT' in conf:
            first_night = int(conf['FIRST_NIGHT'])
            log.info(f"Setting first_night to FIRST_NIGHT: {first_night}")
        else:
            log.info("Setting first_night to earliest in a normal prod: 20201214")
            first_night = 20201214

    if all_nights is None:
        # all_nights = get_all_nights(first_night, last_night)
        log.info("Populating all_nights with all of the nights with valid science "
                 + f"exposures between {first_night} and {last_night} inclusive")
        all_nights = get_all_valid_nights(first_night, last_night)

    ## Load the other parameters for running desi_proc_night
    if 'THRU_NIGHT' in conf:
        thru_night = int(conf['THRU_NIGHT'])
        log.info(f"Setting thru_night to THRU_NIGHT: {thru_night}")
    else:
        thru_night = last_night
        log.warning(f"Setting thru_night to last_night: {thru_night}")

    ## note that None defaults to "cumulative" when handed to desi_proc_night
    ## unless no_redshifts is True
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
        z_submit_types = None

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

    ## Do the main processing
    finished = False
    processed_nights, skipped_nights = [], []
    all_nights = sorted(all_nights)
    log.info(f"Processing {all_nights=}")
    for night in sorted(all_nights):
        num_in_queue = check_queue_count(user=user, include_scron=False,
                                         dry_run_level=dry_run_level)
        ## In Jura the largest night had 115 jobs, to be conservative say 200
        if num_in_queue > 4800:
            log.info(f"{num_in_queue} jobs in the queue, so stopping the job submissions.")
            break
        if os.path.exists(findfile('proctable', night=night, readonly=True)):
            skipped_nights.append(night)
            log.info(f"{night=} already has a proctable, skipping.")
            continue
        ## We don't expect exposure tables to change during code execution here
        ## but we do expect processing tables to evolve, so clear that cache
        log.info(f"Processing {night=}")
        # TODO uncomment when merged into branch with crossnight dependencies
        #desispec.workflow.proctable.reset_tilenight_ptab_cache()
        if dry_run_level < 4:
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
        with open(sentinel_file, 'w') as sentinel:
            sentinel.write(f"All done with processing for {production_yaml}")
            sentinel.write(f"Nights processed: {all_nights}")

    log.info("Skipped the following nights that already had a processing table:")
    log.info(skipped_nights)
    log.info("Processed the following nights:")
    log.info(processed_nights)
    if finished:
        log.info('\n\n\n')
        log.info("All nights submitted")
