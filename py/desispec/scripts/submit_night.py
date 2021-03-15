from desiutil.log import get_logger
import numpy as np
import os
import sys
import time
import re
from astropy.table import Table
## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.tableio import load_tables, write_table
from desispec.workflow.utils import pathjoin
from desispec.workflow.timing import what_night_is_it, nersc_start_time, nersc_end_time
from desispec.workflow.exptable import get_exposure_table_path, get_exposure_table_name
from desispec.workflow.proctable import default_exptypes_for_proctable, get_processing_table_path, \
                                        get_processing_table_name, erow_to_prow
from desispec.workflow.procfuncs import parse_previous_tables, get_type_and_tile, \
                                        define_and_assign_dependency, create_and_submit, checkfor_and_submit_joint_job
from desispec.workflow.queue import update_from_queue
from desispec.workflow.desi_proc_funcs import get_desi_proc_batch_file_path

def submit_night(night, proc_obstypes=None, dry_run=False, queue='realtime', reservation=None,
                 exp_table_path=None, proc_table_path=None, tab_filetype='csv',
                 error_if_not_available=True, overwrite_existing=False):
    """
    Creates a processing table and an unprocessed table from a fully populated exposure table and submits those
    jobs for processing (unless dry_run is set).

    Args:
        night, int. The night of data to be processed. Exposure table must exist.
        proc_obstypes, list or np.array. Optional. A list of exposure OBSTYPE's that should be processed (and therefore
                                              added to the processing table).
        dry_run, bool. Default is False. Should the jobs written to the processing table actually be submitted
                                             for processing.
        exp_table_path: str. Full path to where to exposure tables are stored, WITHOUT the monthly directory included.
        proc_table_path: str. Full path to where to processing tables to be written.
        queue: str. The name of the queue to submit the jobs to. Default is "realtime".
        reservation: str. The reservation to submit jobs to. If None, it is not submitted to a reservation.
        tab_filetype: str. The file extension (without the '.') of the exposure and processing tables.
        error_if_not_available: bool. Default is True. Raise as error if the required exposure table doesn't exist,
                                      otherwise prints an error and returns.
        overwrite_existing: bool. True if you want to submit jobs even if scripts already exist.
    Returns:
        None.
    """
    log = get_logger()

    if proc_obstypes is None:
        proc_obstypes = default_exptypes_for_proctable()

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

    ## Check if night has already been submitted and don't submit if it has, unless told to with ignore_existing
    batchdir = get_desi_proc_batch_file_path(night=night)
    if not overwrite_existing:
        if os.path.exists(batchdir) and len(os.listdir(batchdir)) > 0:
            print(f"ERROR: Batch jobs already exist for night {night} and not given flag "+
                  "overwrite_existing. Exiting this night.")
            return
        elif os.path.exists(proc_table_pathname):
            print(f"ERROR: Processing table: {proc_table_pathname} already exists and and not "+
                  "given flag overwrite_existing. Exiting this night.")
            return

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
    good_types = np.array([val in proc_obstypes for val in etable['OBSTYPE']]).astype(bool)
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
    for ii, erow in enumerate(etable):
        exp = int(erow['EXPID'])
        print(f'\n\n##################### {exp} #########################')

        print(f"\nFound: {erow}")

        curtype, curtile = get_type_and_tile(erow)

        if lasttype is not None and ((curtype != lasttype) or (curtile != lasttile)):
            ptable, arcjob, flatjob, sciences, internal_id = checkfor_and_submit_joint_job(ptable, arcs, flats,
                                                                                           sciences, arcjob,
                                                                                           flatjob, lasttype,
                                                                                           internal_id,
                                                                                           dry_run=dry_run,
                                                                                           queue=queue,
                                                                                           reservation=reservation,
                                                                                           strictly_successful=True)

        prow = erow_to_prow(erow)
        prow['INTID'] = internal_id
        internal_id += 1
        prow = define_and_assign_dependency(prow, arcjob, flatjob)
        print(f"\nProcessing: {prow}\n")
        prow = create_and_submit(prow, dry_run=dry_run, queue=queue, reservation=reservation, strictly_successful=True)
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
                print("\n", "Sleeping 10s to slow down the queue submission rate")
                time.sleep(10)

        ## Flush the outputs
        sys.stdout.flush()
        sys.stderr.flush()

    if tableng > 0:
        ## No more data coming in, so do bottleneck steps if any apply
        ptable, arcjob, flatjob, sciences, internal_id = checkfor_and_submit_joint_job(ptable, arcs, flats, sciences,
                                                                                       arcjob, flatjob, lasttype,
                                                                                       internal_id, dry_run=dry_run,
                                                                                       queue=queue,
                                                                                       reservation=reservation,
                                                                                       strictly_successful=True)
        ## All jobs now submitted, update information from job queue and save
        ptable = update_from_queue(ptable, start_time=nersc_start, end_time=nersc_end, dry_run=dry_run)
        write_table(ptable, tablename=proc_table_pathname)

    print(f"Completed submission of exposures for night {night}.", '\n\n\n')
