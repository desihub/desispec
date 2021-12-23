from desiutil.log import get_logger
import numpy as np
import os
import sys
import time
import re
from astropy.table import Table, vstack
## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.tableio import load_tables, write_table
from desispec.workflow.utils import pathjoin, sleep_and_report
from desispec.workflow.timing import what_night_is_it
from desispec.workflow.exptable import get_exposure_table_path, get_exposure_table_name
from desispec.workflow.proctable import default_obstypes_for_proctable, get_processing_table_path, \
                                        get_processing_table_name, erow_to_prow, table_row_to_dict, \
                                        default_prow
from desispec.workflow.procfuncs import parse_previous_tables, get_type_and_tile, \
                                        define_and_assign_dependency, create_and_submit, checkfor_and_submit_joint_job
from desispec.workflow.queue import update_from_queue, any_jobs_not_complete
from desispec.workflow.desi_proc_funcs import get_desi_proc_batch_file_path
from desispec.io.util import decode_camword, difference_camwords, create_camword

def submit_night(night, proc_obstypes=None, z_submit_types=None, queue='realtime', reservation=None, system_name=None,
                 exp_table_path=None, proc_table_path=None, tab_filetype='csv',
                 dry_run_level=0, dry_run=False, no_redshifts=False, error_if_not_available=True,
                 append_to_proc_table=False, ignore_proc_table_failures = False,
                 dont_check_job_outputs=False, dont_resubmit_partial_jobs=False,
                 tiles=None, surveys=None):
    """
    Creates a processing table and an unprocessed table from a fully populated exposure table and submits those
    jobs for processing (unless dry_run is set).

    Args:
        night, int. The night of data to be processed. Exposure table must exist.
        proc_obstypes, list or np.array. Optional. A list of exposure OBSTYPE's that should be processed (and therefore
                                              added to the processing table).
        z_submit_types: list of str's or comma separated list of string. The "group" types of redshifts that should be
                                       submitted with each exposure. If not specified, default for daily processing is
                                       ['cumulative', 'pernight-v0']. If false, 'false', or [], then no redshifts are submitted.
        exp_table_path: str. Full path to where to exposure tables are stored, WITHOUT the monthly directory included.
        proc_table_path: str. Full path to where to processing tables to be written.
        queue: str. The name of the queue to submit the jobs to. Default is "realtime".
        reservation: str. The reservation to submit jobs to. If None, it is not submitted to a reservation.
        system_name: batch system name, e.g. cori-haswell, cori-knl, perlmutter-gpu
        dry_run_level, int, If nonzero, this is a simulated run. If dry_run=1 the scripts will be written but not submitted.
                      If dry_run=2, the scripts will not be written nor submitted. Logging will remain the same
                      for testing as though scripts are being submitted. Default is 0 (false).
        dry_run, bool. When to run without submitting scripts or not. If dry_run_level is defined, then it over-rides
                       this flag. dry_run_level not set and dry_run=True, dry_run_level is set to 2 (no scripts
                       generated or run). Default for dry_run is False.
        no_redshifts, bool. Whether to submit redshifts or not. If True, redshifts are not submitted.
        tab_filetype: str. The file extension (without the '.') of the exposure and processing tables.
        error_if_not_available: bool. Default is True. Raise as error if the required exposure table doesn't exist,
                                      otherwise prints an error and returns.
        append_to_proc_table: bool. True if you want to submit jobs even if a processing table already exists.
                                         Otherwise jobs will be appended to it. Default is False
        ignore_proc_table_failures: bool. True if you want to submit other jobs even the loaded
                                        processing table has incomplete jobs in it. Use with caution. Default is False.
        dont_check_job_outputs, bool. Default is False. If False, the code checks for the existence of the expected final
                                 data products for the script being submitted. If all files exist and this is False,
                                 then the script will not be submitted. If some files exist and this is False, only the
                                 subset of the cameras without the final data products will be generated and submitted.
        dont_resubmit_partial_jobs, bool. Default is False. Must be used with dont_check_job_outputs=False. If this flag is
                                          False, jobs with some prior data are pruned using PROCCAMWORD to only process the
                                          remaining cameras not found to exist.
        tiles, array-like, optional. Only submit jobs for these TILEIDs.
        surveys, array-like, optional. Only submit science jobs for these surveys (lowercase)

    Returns:
        None.
    """
    log = get_logger()

    ## Recast booleans from double negative
    check_for_outputs = (not dont_check_job_outputs)
    resubmit_partial_complete = (not dont_resubmit_partial_jobs)

    if proc_obstypes is None:
        proc_obstypes = default_obstypes_for_proctable()

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

    ## Define the group types of redshifts you want to generate for each tile
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

    ## Combine the table names and types for easier passing to io functions
    table_pathnames = [exp_table_pathname, proc_table_pathname]
    table_types = ['exptable', 'proctable']

    ## Load in the files defined above
    etable, ptable = load_tables(tablenames=table_pathnames, tabletypes=table_types)

    ## filter by TILEID if requested
    if tiles is not None:
        log.info(f'Filtering by tiles={tiles}')
        if etable is not None:
            keep = np.isin(etable['TILEID'], tiles)
            etable = etable[keep]
        if ptable is not None:
            keep = np.isin(ptable['TILEID'], tiles)
            ptable = ptable[keep]

    if surveys is not None:
        log.info(f'Filtering by surveys={surveys}')
        if etable is not None:
            if 'SURVEY' not in etable.dtype.names:
                raise ValueError(f'surveys={surveys} filter requested, but no SURVEY column in {exp_table_pathname}')

            # only apply survey filter to OBSTYPE=science exposures, i.e. auto-keep non-science
            keep = (etable['OBSTYPE'] != 'science')

            # np.isin doesn't work with bytes vs. str from Tables but direct comparison does, so loop
            for survey in surveys:
                keep |= etable['SURVEY'] == survey

            etable = etable[keep]
        if ptable is not None:
            # ptable doesn't have "SURVEY", so filter by the TILEIDs we just kept
            keep = np.isin(ptable['TILEID'], etable['TILEID'])
            ptable = ptable[keep]

    good_exps = np.array([col.lower() != 'ignore' for col in etable['LASTSTEP']]).astype(bool)
    good_types = np.array([val in proc_obstypes for val in etable['OBSTYPE']]).astype(bool)
    good_exptimes = []
    for erow in etable:
        if erow['OBSTYPE'] == 'science' and erow['EXPTIME'] < 60:
            good_exptimes.append(False)
        elif erow['OBSTYPE'] == 'arc' and erow['EXPTIME'] > 8.:
            good_exptimes.append(False)
        elif erow['OBSTYPE'] == 'dark' and np.abs(float(erow['EXPTIME'])-300.) > 1:
            good_exptimes.append(False)
        else:
            good_exptimes.append(True)

    good_exptimes = np.array(good_exptimes)
    good = (good_exps & good_types & good_exptimes)
    unproc_table = etable[~good]
    etable = etable[good]


    ## Simple table organization to ensure cals processed first
    ## To be eventually replaced by more sophisticated cal selection
    ## Get one dark first
    isdark = (etable['OBSTYPE'] == 'dark')
    if np.sum(isdark)>0:
        wheredark = np.where(isdark)[0]
        if len(wheredark) > 1:
            unproc_table = vstack([unproc_table, etable[wheredark[1:]]])
            unproc_table.sort('EXPID')
        etable = vstack([etable[wheredark[0]], etable[~isdark]])

    ## Then get rest of the cals above scis
    issci = (etable['OBSTYPE'] == 'science')
    etable = vstack([etable[~issci], etable[issci]])

    ## Done determining what not to process, so write out unproc file
    if dry_run_level < 3:
        write_table(unproc_table, tablename=unproc_table_pathname)

    ## Get relevant data from the tables
    arcs, flats, sciences, calibjobs, curtype, lasttype, \
    curtile, lasttile, internal_id = parse_previous_tables(etable, ptable, night)
    if len(ptable) > 0:
        ptable = update_from_queue(ptable, dry_run=0)
        if dry_run_level < 3:
            write_table(ptable, tablename=proc_table_pathname)
        if any_jobs_not_complete(ptable['STATUS']) and not ignore_proc_table_failures:
            print("ERROR: Some jobs have an incomplete job status. This script will "+
                  "not fix them. You should remedy those first. Exiting")
            return
        ptable_expids = np.unique(np.concatenate(ptable['EXPID']))
        if len(set(etable['EXPID']).difference(set(ptable_expids))) == 0:
            print("ERROR: All EXPID's already present in processing table. Exiting")
            return
    else:
        ptable_expids = np.array([], dtype=int)

    ## Loop over new exposures and process them as relevant to that type
    tableng = len(ptable)

    if tableng == 0 and np.sum(isdark) == 0:
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

    for ii, erow in enumerate(etable):
        if erow['EXPID'] in ptable_expids:
            continue
        erow = table_row_to_dict(erow)
        exp = int(erow['EXPID'])
        print(f'\n\n##################### {exp} #########################')

        print(f"\nFound: {erow}")

        curtype, curtile = get_type_and_tile(erow)

        if lasttype is not None and ((curtype != lasttype) or (curtile != lasttile)):
            ptable, calibjobs, sciences, internal_id \
                = checkfor_and_submit_joint_job(ptable, arcs, flats, sciences,
                                                calibjobs,
                                                lasttype, internal_id,
                                                dry_run=dry_run_level,
                                                queue=queue,
                                                reservation=reservation,
                                                strictly_successful=True,
                                                check_for_outputs=check_for_outputs,
                                                resubmit_partial_complete=resubmit_partial_complete,
                                                z_submit_types=z_submit_types,
                                                system_name=system_name)

        prow = erow_to_prow(erow)
        prow['INTID'] = internal_id
        internal_id += 1
        if prow['OBSTYPE'] == 'dark':
            prow['JOBDESC'] = 'ccdcalib'
        else:
            prow['JOBDESC'] = prow['OBSTYPE']
        prow = define_and_assign_dependency(prow, calibjobs)
        print(f"\nProcessing: {prow}\n")
        prow = create_and_submit(prow, dry_run=dry_run_level, queue=queue,
                                 reservation=reservation, strictly_successful=True,
                                 check_for_outputs=check_for_outputs,
                                 resubmit_partial_complete=resubmit_partial_complete,
                                 system_name=system_name)

        ## If processed a dark, assign that to the dark job
        if curtype == 'dark':
            prow['CALIBRATOR'] = 1
            calibjobs['ccdcalib'] = prow.copy()

        ## Add the processing row to the processing table
        ptable.add_row(prow)
        #ptable_expids = np.append(ptable_expids, erow['EXPID'])

        ## Note: Assumption here on number of flats
        if curtype == 'flat' and calibjobs['nightlyflat'] is None \
                and int(erow['SEQTOT']) < 5 and float(erow['EXPTIME']) > 100.:
            flats.append(prow)
        elif curtype == 'arc' and calibjobs['psfnight'] is None:
            arcs.append(prow)
        elif curtype == 'science' and prow['LASTSTEP'] != 'skysub':
            sciences.append(prow)

        lasttile = curtile
        lasttype = curtype

        tableng = len(ptable)
        if tableng > 0 and ii % 10 == 0 and dry_run_level < 3:
            write_table(ptable, tablename=proc_table_pathname)

        sleep_and_report(1, message_suffix=f"to slow down the queue submission rate",
                         dry_run=dry_run)

        ## Flush the outputs
        sys.stdout.flush()
        sys.stderr.flush()

    if tableng > 0:
        ## No more data coming in, so do bottleneck steps if any apply
        ptable, calibjobs, sciences, internal_id \
            = checkfor_and_submit_joint_job(ptable, arcs, flats, sciences, calibjobs,
                                            lasttype, internal_id, dry_run=dry_run_level,
                                            queue=queue, reservation=reservation,
                                            strictly_successful=True,
                                            check_for_outputs=check_for_outputs,
                                            resubmit_partial_complete=resubmit_partial_complete,
                                            z_submit_types=z_submit_types,
                                            system_name=system_name)
        ## All jobs now submitted, update information from job queue and save
        ptable = update_from_queue(ptable, dry_run=dry_run_level)
        if dry_run_level < 3:
            write_table(ptable, tablename=proc_table_pathname)

    print(f"Completed submission of exposures for night {night}.", '\n\n\n')
