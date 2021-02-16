
import yaml
from desiutil.log import get_logger
import numpy as np
import os
import sys
import time
import re
from astropy.table import Table
## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.tableio import load_tables, write_tables, write_table
from desispec.workflow.utils import verify_variable_with_environment, pathjoin, listpath, get_printable_banner
from desispec.workflow.timing import during_operating_hours, what_night_is_it, nersc_start_time, nersc_end_time
from desispec.workflow.exptable import default_exptypes_for_exptable, get_exposure_table_column_defs, validate_badamps, \
                                       get_exposure_table_path, get_exposure_table_name, summarize_exposure
from desispec.workflow.proctable import default_exptypes_for_proctable, get_processing_table_path, get_processing_table_name, erow_to_prow
from desispec.workflow.procfuncs import parse_previous_tables, flat_joint_fit, arc_joint_fit, get_type_and_tile, \
                                        science_joint_fit, define_and_assign_dependency, create_and_submit, \
                                        update_and_recurvsively_submit, checkfor_and_submit_joint_job
from desispec.workflow.queue import update_from_queue, any_jobs_not_complete





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
    Returns a full list of all nights availabel in the DESI Raw data directory
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
    conf=yaml.safe_load(open(production_yaml,'rb'))
    specprod = str(conf['name']).lower()
    specprod = verify_variable_with_environment(var=specprod,var_name='specprod',env_name='SPECPROD')

    all_nights = get_all_nights()
    non_survey_nights = []
    for night in all_nights:
        survey = assign_survey(night,conf)
        if survey is None:
            non_survey_nights.append(night)
            continue
        elif survey in conf['ProcessData'] and conf['ProcessData'][survey] is False:
            print(f'Asked not to process survey: {survey}, Not processing night={night}.','\n\n\n')
            continue
        elif survey in conf['SkipNights'] and night in conf['SkipNights'][survey]:
            print(f'Asked to skip night={night} (in survey: {survey}). Skipping.','\n\n\n')
            continue
        else:
            print(f'Processing {survey} night: {night}')
            submit_night(night, procobstypes=None, dry_run=dry_run, queue='realtime',
                         error_if_not_available=error_if_not_available)
            print(f"Completed {night}. Sleeping for 30s")
            time.sleep(30)
            
    print("Skipped the following nights that were not assigned to a survey:")
    print(non_survey_nights, '\n\n\n')
    print("All nights submitted")


def submit_night(night, procobstypes=None, dry_run=False, queue='realtime',
                 exp_table_path=None, proc_table_path=None, tab_filetype='csv',
                 error_if_not_available=True):
    """
    Creates a processing table and an unprocessed table from a fully populated exposure table and submits those
    jobs for processing (unless dry_run is set).

    Args:
        night, int. The night of data to be processed. Exposure table must exist.
        procobstypes, list or np.array. Optional. A list of exposure OBSTYPE's that should be processed (and therefore
                                              added to the processing table).
        dry_run, bool. Default is False. Should the jobs written to the processing table actually be submitted
                                             for processing.
        exp_table_path: str. Full path to where to exposure tables are stored, WITHOUT the monthly directory included.
        proc_table_path: str. Full path to where to processing tables to be written.
        queue: str. The name of the queue to submit the jobs to. Default is "realtime".
        tab_filetype: str. The file extension (without the '.') of the exposure and processing tables.
        error_if_not_available, bool. Default is True. Raise as error if the required exposure table doesn't exist,
                                      otherwise prints an error and returns.
    Returns:
        processing_table, Table. The output processing table. Each row corresponds with an exposure that should be
                                 processed.
        unprocessed_table, Table. The output unprocessed table. Each row is an exposure that should not be processed.
    """
    log = get_logger()

    if procobstypes is None:
        procobstypes = default_exptypes_for_proctable()

    ## Determine where the exposure table will be written
    if exp_table_path is None:
        exp_table_path = get_exposure_table_path(night=night, usespecprod=True)
    name = get_exposure_table_name(night=night, extension=tab_filetype)
    exp_table_pathname = pathjoin(exp_table_path, name)
    if not os.path.exists(exp_table_pathname):
        if error_if_not_available:
            raise IOError(f"Exposure table: {exp_table_pathname} not found. Exiting this night.")
        else:
            print(f"ERROR: Exposure table: {exp_table_pathname} not found. Exiting this night.")
            return

    ## Determine where the processing table will be written
    if proc_table_path is None:
        proc_table_path = get_processing_table_path()
    os.makedirs(proc_table_path, exist_ok=True)
    name = get_processing_table_name(prodmod=night, extension=tab_filetype)
    proc_table_pathname = pathjoin(proc_table_path, name)

    ## Determine where the unprocessed data table will be written
    unproc_table_pathname = pathjoin(proc_table_path, name.replace('processing', 'unprocessed'))

    ## Combine the table names and types for easier passing to io functions
    table_pathnames = [exp_table_pathname, proc_table_pathname]
    table_types = ['exptable', 'proctable']

    ## Load in the files defined above
    etable, ptable = load_tables(tablenames=table_pathnames, tabletypes=table_types)

    ## Get context specific variable values
    true_night = what_night_is_it()
    nersc_start = nersc_start_time(night=true_night)
    nersc_end = nersc_end_time(night=true_night)

    good_exps = np.array([col.lower() != 'ignore' for col in etable['LASTSTEP']]).astype(bool)
    good_types = np.array([val in procobstypes for val in etable['OBSTYPE']]).astype(bool)
    good_exptimes = []
    for erow in etable:
        if erow['OBSTYPE'] == 'science' and erow['EXPTIME'] < 60:
            good_exptimes.append(False)
        elif erow['OBSTYPE'] == 'arc' and erow['EXPTIME'] > 8.:
            good_exptimes.append(False)
        else:
            good_exptimes.append(True)
    good_exptimes = np.array(good_exptimes)
    good = (good_exps & good_types & good_exptimes)
    unproc_table = etable[~good]
    etable = etable[good]

    write_table(unproc_table, tablename=unproc_table_pathname)
    ## Get relevant data from the tables
    arcs, flats, sciences, arcjob, flatjob, \
    curtype, lasttype, curtile, lasttile, internal_id, last_not_dither = parse_previous_tables(etable, ptable, night)

    ## Loop over new exposures and process them as relevant to that type
    for ii,erow in enumerate(etable):
        exp = int(erow['EXPID'])
        print(f'\n\n##################### {exp} #########################')

        print(f"\nFound: {erow}")

        curtype, curtile = get_type_and_tile(erow)

        if (curtype != lasttype) or (curtile != lasttile) and lasttype is not None:
            ptable, arcjob, flatjob, sciences, internal_id = checkfor_and_submit_joint_job(ptable, arcs, flats,
                                                                                           sciences, arcjob,
                                                                                           flatjob, lasttype,
                                                                                           last_not_dither,
                                                                                           internal_id,
                                                                                           dry_run=dry_run,
                                                                                           queue=queue,
                                                                                           strictly_successful=True)

        prow = erow_to_prow(erow)
        prow['INTID'] = internal_id
        internal_id += 1
        prow = define_and_assign_dependency(prow, arcjob, flatjob)
        print(f"\nProcessing: {prow}\n")
        prow = create_and_submit(prow, dry_run=dry_run, queue=queue, strictly_successful=True)
        ptable.add_row(prow)

        ## Note: Assumption here on number of flats
        if curtype == 'flat' and flatjob is None and int(erow['SEQTOT']) < 5:
            flats.append(prow)
        elif curtype == 'arc' and arcjob is None:
            arcs.append(prow)
        elif curtype == 'science' and prow['LASTSTEP'] != 'skysub':
            sciences.append(prow)

        lasttile = curtile
        lasttype = curtype
        last_not_dither = (prow['OBSDESC'] != 'dither')

        if not dry_run:
            time.sleep(1)

        tableng = len(ptable)
        if tableng > 0 and ii % 10 == 0:
            write_table(ptable, tablename=proc_table_pathname)
            if not dry_run:
                print("\n","Sleeping 10s to slow down the queue submission rate")
                time.sleep(10)

        ## Flush the outputs
        sys.stdout.flush()
        sys.stderr.flush()

    if tableng > 0:
        ## No more data coming in, so do bottleneck steps if any apply
        ptable, arcjob, flatjob, sciences, internal_id = checkfor_and_submit_joint_job(ptable, arcs, flats, sciences, \
                                                                                       arcjob, flatjob, lasttype,
                                                                                       last_not_dither, \
                                                                                       internal_id, dry_run=dry_run,
                                                                                       queue=queue, strictly_successful=True)
        ## All jobs now submitted, update information from job queue and save
        ptable = update_from_queue(ptable, start_time=nersc_start, end_time=nersc_end, dry_run=dry_run)
        write_table(ptable, tablename=proc_table_pathname)

    print(f"Completed submission of exposures for night {night}.",'\n\n\n')

