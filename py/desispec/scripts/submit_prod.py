"""
desispec.scripts.submit_prod
============================

Please add module-level documentation.
"""
import yaml
import numpy as np
import os
import sys
import time
import re
## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.utils import verify_variable_with_environment, listpath
from desispec.scripts.submit_night import submit_night


def assign_survey(night, conf):
    """
    Takes a desi production configuration (yaml) dictionary and determines
    the survey corresponding to a given night based on the contents of the conf
    dictionary, if psosible. Otherwise returns None.

    Args:
        night, int. The night you want to know the survey it corresponds to.
        conf, dict. Dictionary that returned when the configuration yaml file was read in.

    Returns:
        survey, str. The survey the night was taken under, according to the conf file.
    """
    for survey in conf['DateRanges']:
        first, last = conf['DateRanges'][survey]
        if night >= first and night <= last:
            return survey
    else:
        return None


def get_all_nights():
    """
    Returns a full list of all nights availabel in the DESI Raw data directory.

    Returns:
        nights, list. A list of nights on or after Jan 1 2020 in which data exists at NERSC.
    """
    nights = list()
    for n in listpath(os.getenv('DESI_SPECTRO_DATA')):
        # - nights are 202YMMDD
        if re.match('^202\d{5}$', n):
            nights.append(int(n))
    return nights


def submit_production(production_yaml, dry_run=False, error_if_not_available=False):
    """
    Interprets a production_yaml file and submits the respective nights for processing
    within the defined production.

    Args:
        production_yaml, str. Pathname of the yaml file that defines the production.
        dry_run, bool. Default is False. Should the jobs written to the processing table actually be submitted
                                             for processing.
        error_if_not_available, bool. Default is True. Raise as error if the required exposure table doesn't exist,
                                      otherwise prints an error and returns.
    Returns:
        None.
    """
    if not os.path.exists(production_yaml):
        raise IOError(f"Prod Yaml file doesn't exist: {production_yaml} not found. Exiting.")
    conf = yaml.safe_load(open(production_yaml, 'rb'))
    specprod = str(conf['name']).lower()
    specprod = verify_variable_with_environment(var=specprod, var_name='specprod', env_name='SPECPROD')
    if 'reservation' in conf:
        reservation = str(conf['reservation'])
        if reservation.lower() == 'none':
            reservation = None
    else:
        reservation = None
    if 'queue' in conf:
        queue = conf['queue']
    else:
        queue = 'realtime'

    if 'OVERWRITEEXISTING' in conf:
        overwrite_existing = conf['OVERWRITEEXISTING']
    else:
        overwrite_existing = False

    print(f'Using queue: {queue}')
    if reservation is not None:
        print(f'Using reservation: {reservation}')
    if overwrite_existing:
        print("Ignoring the fact that files exists and submitting those nights anyway")

    all_nights = get_all_nights()
    non_survey_nights = []
    for night in all_nights:
        survey = assign_survey(night, conf)
        if survey is None:
            non_survey_nights.append(night)
            continue
        elif survey in conf['ProcessData'] and conf['ProcessData'][survey] is False:
            print(f'Asked not to process survey: {survey}, Not processing night={night}.', '\n\n\n')
            continue
        elif survey in conf['SkipNights'] and night in conf['SkipNights'][survey]:
            print(f'Asked to skip night={night} (in survey: {survey}). Skipping.', '\n\n\n')
            continue
        else:
            print(f'Processing {survey} night: {night}')
            submit_night(night, proc_obstypes=None, dry_run=dry_run, queue=queue, reservation=reservation,
                         overwrite_existing=overwrite_existing, error_if_not_available=error_if_not_available)
            print(f"Completed {night}. Sleeping for 30s")
            time.sleep(30)

    print("Skipped the following nights that were not assigned to a survey:")
    print(non_survey_nights, '\n\n\n')
    print("All nights submitted")
