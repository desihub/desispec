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
from desispec.workflow.exptable import get_exposure_table_path, \
    get_exposure_table_name, get_last_step_options
from desispec.workflow.proctable import default_obstypes_for_proctable, get_processing_table_path, \
                                        get_processing_table_name, erow_to_prow, table_row_to_dict, \
                                        default_prow
from desispec.workflow.procfuncs import parse_previous_tables, get_type_and_tile, \
                                        define_and_assign_dependency, create_and_submit, \
                                        checkfor_and_submit_joint_job, submit_tilenight_and_redshifts
from desispec.workflow.queue import update_from_queue, any_jobs_not_complete
from desispec.workflow.desi_proc_funcs import get_desi_proc_batch_file_path
from desispec.io.util import decode_camword, difference_camwords, create_camword

def submit_night(night, proc_obstypes=None, z_submit_types=None, queue='realtime',
                 reservation=None, system_name=None,
                 exp_table_path=None, proc_table_path=None, tab_filetype='csv',
                 dry_run_level=0, dry_run=False, no_redshifts=False, error_if_not_available=True,
                 append_to_proc_table=False, ignore_proc_table_failures = False,
                 dont_check_job_outputs=False, dont_resubmit_partial_jobs=False,
                 tiles=None, surveys=None, laststeps=None, use_tilenight=False,
                 all_tiles=False, specstatus_path=None, use_specter=False, do_cte_flat=False):
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
        laststeps, array-like, optional. Only submit jobs for exposures with LASTSTEP in these laststeps (lowercase)
        use_tilenight, bool, optional. Default is False. If True, use desi_proc_tilenight for prestdstar, stdstar,
                             and poststdstar steps for science exposures.
        all_tiles, bool, optional. Default is False. Set to NOT restrict to completed tiles as defined by
                                              the table pointed to by specstatus_path.
        specstatus_path, str, optional. Default is $DESI_SURVEYOPS/ops/tiles-specstatus.ecsv.
                                        Location of the surveyops specstatus table.
        use_specter, bool, optional. Default is False. If True, use specter, otherwise use gpu_specter by default.
        do_cte_flat, bool, optional. Default is False. If True, one second flat exposures are processed for cte identification.

    Returns:
        None.
    """
    log = get_logger()

    ## Recast booleans from double negative
    check_for_outputs = (not dont_check_job_outputs)
    resubmit_partial_complete = (not dont_resubmit_partial_jobs)

    if proc_obstypes is None:
        proc_obstypes = default_obstypes_for_proctable()
    print(f"Processing the following obstypes: {proc_obstypes}")

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

    ## If laststeps not defined, default is only LASTSTEP=='all' exposures for non-tilenight runs
    if laststeps is None:
        laststeps = ['all',]
    else:
        laststep_options = get_last_step_options()
        for laststep in laststeps:
            if laststep not in laststep_options:
                raise ValueError(f"Couldn't understand laststep={laststep} in laststeps={laststeps}.")
    print(f"Processing exposures with the following LASTSTEP's: {laststeps}")

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
    full_etable = etable.copy()

    ## Sort science exposures by TILEID
    sciexps = (etable['OBSTYPE']=='science')
    scisrtd = etable[sciexps].argsort(['TILEID','EXPID'])
    etable[sciexps] = etable[sciexps][scisrtd]

    ## filter by TILEID if requested
    if tiles is not None:
        log.info(f'Filtering by tiles={tiles}')
        if etable is not None:
            keep = np.isin(etable['TILEID'], tiles)
            etable = etable[keep]
        #if ptable is not None:
        #    keep = np.isin(ptable['TILEID'], tiles)
        #    ptable = ptable[keep]

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
        #if ptable is not None:
        #    # ptable doesn't have "SURVEY", so filter by the TILEIDs we just kept
        #    keep = np.isin(ptable['TILEID'], etable['TILEID'])
        #    ptable = ptable[keep]

    ## If asked to do so, only process tiles deemed complete by the specstatus file
    if not all_tiles:
        completed_tiles = get_completed_tiles(specstatus_path)

        ## Add -99 to keep calibration exposures
        completed_tiles = np.append([-99], completed_tiles)
        if etable is not None:
            keep = np.isin(etable['TILEID'], completed_tiles)
            log.info(f'Filtering by completed tiles retained {sum(keep)}/{len(etable)} exposures')
            etable = etable[keep]

    ## Cut on LASTSTEP
    good_exps = np.isin(np.array(etable['LASTSTEP']).astype(str), laststeps)
    etable = etable[good_exps]

    ## Count zeros before trimming by OBSTYPE since they are used for
    ## nightly bias even if they aren't processed individually
    num_zeros = np.sum([erow['OBSTYPE'] == 'zero' and
                       (erow['PROGRAM'].startswith('calib zeros') or erow['PROGRAM'].startswith('zeros for dark'))
                       for erow in etable])

    ## Cut on OBSTYPES
    good_types = np.isin(np.array(etable['OBSTYPE']).astype(str), proc_obstypes)
    etable = etable[good_types]

    ## Cut on EXPTIME
    good_exptimes = []
    for erow in etable:
        if erow['OBSTYPE'] == 'science' and erow['EXPTIME'] < 60:
            good_exptimes.append(False)
        elif erow['OBSTYPE'] == 'arc' and erow['EXPTIME'] > 8.:
            good_exptimes.append(False)
        elif erow['OBSTYPE'] == 'dark' and np.abs(float(erow['EXPTIME']) - 300.) > 1:
            good_exptimes.append(False)
        elif erow['OBSTYPE'] == 'flat' and np.abs(float(erow['EXPTIME']) - 120.) > 1:
            if not do_cte_flat or np.abs(float(erow['EXPTIME']) - 1.) > 0.5:
                good_exptimes.append(False)
        else:
            good_exptimes.append(True)
    etable = etable[np.array(good_exptimes)]

    ## Simple table organization to ensure cals processed first
    ## To be eventually replaced by more sophisticated cal selection
    ## Get one dark first
    isdarkcal = np.array([(erow['OBSTYPE'] == 'dark' and 'calib' in
                          erow['PROGRAM']) for erow in etable])
    isdark = np.array([(erow['OBSTYPE'] == 'dark') for erow in etable])

    ## If a cal, want to select that but ignore all other darks
    ## elif only a dark sequence, use that
    if np.sum(isdarkcal)>0:
        wheredark = np.where(isdarkcal)[0]
        ## note this is ~isdark because want to get rid of all other darks
        etable = vstack([etable[wheredark[0]], etable[~isdark]])
    elif np.sum(isdark)>0:
        wheredark = np.where(isdark)[0]
        etable = vstack([etable[wheredark[0]], etable[~isdark]])

    ## Then get rest of the cals above scis
    issci = (etable['OBSTYPE'] == 'science')
    etable = vstack([etable[~issci], etable[issci]])

    ## Get relevant data from the tables
    arcs, flats, sciences, calibjobs, curtype, lasttype, \
    curtile, lasttile, internal_id = parse_previous_tables(etable, ptable, night)
    if len(ptable) > 0:
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
            return
    else:
        ptable_expids = np.array([], dtype=int)

    tableng = len(ptable)

    ## Now figure out everything that isn't in the final list, which we'll
    ## Write out to the unproccessed table
    toprocess = np.isin(full_etable['EXPID'], etable['EXPID'])
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

    ## Loop over new exposures and process them as relevant to that type
    for ii, erow in enumerate(etable):
        if erow['EXPID'] in ptable_expids:
            continue
        erow = table_row_to_dict(erow)
        exp = int(erow['EXPID'])
        print(f'\n\n##################### {exp} #########################')

        print(f"\nFound: {erow}")

        curtype, curtile = get_type_and_tile(erow)

        if lasttype is not None and ((curtype != lasttype) or (curtile != lasttile)):
            # If done with science exposures for a tile and use_tilenight==True, use
            # submit_tilenight_and_redshifts, otherwise use checkfor_and_submit_joint_job
            if use_tilenight and lasttype == 'science' and len(sciences)>0:
                ptable, sciences, internal_id \
                    = submit_tilenight_and_redshifts(ptable, sciences, calibjobs, lasttype, internal_id,
                                                    dry_run=dry_run_level,
                                                    queue=queue,
                                                    reservation=reservation,
                                                    strictly_successful=True,
                                                    check_for_outputs=check_for_outputs,
                                                    resubmit_partial_complete=resubmit_partial_complete,
                                                    z_submit_types=z_submit_types,
                                                    system_name=system_name,use_specter=use_specter)
            else:
                cur_z_submit_types = z_submit_types
                ## If running redshifts and there is a future exposure of the same tile
                ## then only run per exposure redshifts until then
                if lasttype == 'science' and z_submit_types is not None and not use_tilenight:
                    tile_exps = etable['EXPID'][((etable['TILEID'] == lasttile) &
                                                 (etable['LASTSTEP'] == 'all'))]
                    unprocd_exps = [exp not in ptable_expids for exp in tile_exps]
                    if np.any(unprocd_exps):
                        print(f"Identified that tile {lasttile} has future exposures"
                            + f" for this night. Not submitting full night "
                            + f"redshift jobs.")
                        if 'perexp' in z_submit_types:
                            print("Still submitting perexp redshifts")
                            cur_z_submit_types = ['perexp']
                        else:
                            cur_z_submit_types = None
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
                                                z_submit_types=cur_z_submit_types,
                                                system_name=system_name)

        prow = erow_to_prow(erow)
        prow['INTID'] = internal_id
        internal_id += 1
        if prow['OBSTYPE'] == 'dark':
            if num_zeros == 0:
                prow['JOBDESC'] = 'badcol'   # process dark for bad columns even if we don't have zeros for nightlybias
            else:
                prow['JOBDESC'] = 'ccdcalib' # ccdcalib = nightlybias(zeros) + badcol(dark)
        else:
            prow['JOBDESC'] = prow['OBSTYPE']
        prow = define_and_assign_dependency(prow, calibjobs)
        if (not use_tilenight) or erow['OBSTYPE'] != 'science':
            print(f"\nProcessing: {prow}\n")
            prow = create_and_submit(prow, dry_run=dry_run_level, queue=queue,
                                 reservation=reservation, strictly_successful=True,
                                 check_for_outputs=check_for_outputs,
                                 resubmit_partial_complete=resubmit_partial_complete,
                                 system_name=system_name,use_specter=use_specter)

            ## If processed a dark, assign that to the dark job
            if curtype == 'dark':
                prow['CALIBRATOR'] = 1
                calibjobs[prow['JOBDESC']] = prow.copy()

            ## Add the processing row to the processing table
            ptable.add_row(prow)

        ptable_expids = np.append(ptable_expids, erow['EXPID'])

        ## Note: Assumption here on number of flats
        if curtype == 'flat' and calibjobs['nightlyflat'] is None \
                and int(erow['SEQTOT']) < 5 \
                and np.abs(float(erow['EXPTIME'])-120.) < 1.:
            flats.append(prow)
        elif curtype == 'arc' and calibjobs['psfnight'] is None:
            arcs.append(prow)
        elif curtype == 'science' and (prow['LASTSTEP'] != 'skysub' or use_tilenight):
            sciences.append(prow)

        lasttile = curtile
        lasttype = curtype

        tableng = len(ptable)
        if tableng > 0 and ii % 1 == 0 and dry_run_level < 3:
            write_table(ptable, tablename=proc_table_pathname)

        sleep_and_report(1, message_suffix=f"to slow down the queue submission rate",
                         dry_run=dry_run)

        ## Flush the outputs
        sys.stdout.flush()
        sys.stderr.flush()

    if tableng > 0:
        ## No more data coming in, so do bottleneck steps if any apply
        if use_tilenight and len(sciences)>0:
            ptable, sciences, internal_id \
                = submit_tilenight_and_redshifts(ptable, sciences, calibjobs, lasttype, internal_id,
                                                dry_run=dry_run_level,
                                                queue=queue,
                                                reservation=reservation,
                                                strictly_successful=True,
                                                check_for_outputs=check_for_outputs,
                                                resubmit_partial_complete=resubmit_partial_complete,
                                                z_submit_types=z_submit_types,
                                                system_name=system_name,use_specter=use_specter)
        else:
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


def get_completed_tiles(specstatus_path=None):
    """
    Uses a tiles-specstatus.ecsv file and selection criteria to determine
    what tiles have beeen completed. Takes an optional argument to point
    to a custom specstatus file. Returns an array of TILEID's.
    Args:
        specstatus_path, str, optional. Default is $DESI_SURVEYOPS/ops/tiles-specstatus.ecsv.
                                        Location of the surveyops specstatus table.

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

    return np.array(specstatus['TILEID'][goodtiles])
