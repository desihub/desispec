#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os
import sys
import time
from astropy.table import Table


## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.tableio import load_tables, write_tables, write_table
from desispec.workflow.utils import verify_variable_with_environment, pathjoin, listpath
from desispec.workflow.timing import during_operating_hours, what_night_is_it, nersc_start_time, nersc_end_time
from desispec.workflow.exptable import default_exptypes_for_exptable, get_surveynum, get_exposure_table_column_defs, \
                                       get_exposure_table_path, get_exposure_table_name, summarize_exposure
from desispec.workflow.proctable import default_exptypes_for_proctable, get_processing_table_path, get_processing_table_name, erow_to_prow
from desispec.workflow.procfuncs import parse_previous_tables, flat_joint_fit, arc_joint_fit, get_type_and_tile, \
                                        science_joint_fit, define_and_assign_dependency, create_and_submit, \
                                        update_and_recurvsively_submit, checkfor_and_submit_joint_job
from desispec.workflow.queue import update_from_queue, any_jobs_not_complete

def daily_processing_manager(specprod=None, exp_table_path=None, proc_table_path=None, path_to_data=None,
                             expobstypes=None, procobstypes=None, camword=None, override_night=None, tab_filetype='csv', queue='realtime',
                             data_cadence_time=30, queue_cadence_time=1800, dry_run=False,continue_looping_debug=False):
    """
    Generates processing tables for the nights requested. Requires exposure tables to exist on disk.

    Args:
        specprod: str. The name of the current production. If used, this will overwrite the SPECPROD environment variable.
        exp_table_path: str. Full path to where to exposure tables are stored, WITHOUT the monthly directory included.
        proc_table_path: str. Full path to where to processing tables to be written.
        path_to_data: str. Path to the raw data.
        expobstypes: str or comma separated list of strings. The exposure OBSTYPE's that you want to include in the exposure table.
        procobstypes: str or comma separated list of strings. The exposure OBSTYPE's that you want to include in the processing table.
        camword: str. Camword that, if set, overwrites the list of cameras found in the files and only runs on those given/
                      Examples: a0123456789, a1, a2b3r3, a2b3r4z3.
        override_night: str or int. 8 digit night, e.g. 20200314, of data to run on. If None, it runs on the current night.
        tab_filetype: str. The file extension (without the '.') of the exposure and processing tables.
        queue: str. The name of the queue to submit the jobs to. Default is "realtime".
        data_cadence_time: int. Wait time in seconds between loops in looking for new data. Default is 30 seconds.
        queue_cadence_time: int. Wait time in seconds between loops in checking queue statuses and resubmitting failures. Default is 1800s.
        dry_run: boolean. If true, no scripts are written and no scripts are submitted. The tables are still generated
                 and written, however. The timing is accelerated. This option is most useful for testing and simulating a run.
        continue_looping_debug: bool. FOR DEBUG purposes only. Will continue looping in search of new data until the process
                                 is terminated. Default is False.

    Returns: Nothing

    Notes:
        Generates both exposure table and processing tables 'on the fly' and saves them at various checkpoints. These
        should be capable of being reloaded in case of interuption or accidental termination of the manager's process.
    """
    ## If not being done during operating hours, and we're not simulating data or running a catchup run, exit.
    if not during_operating_hours(dry_run=dry_run) and override_night is None:
        print("Not during operating hours, and not asked to perform a dry run or run on historic data. Exiting.")
        sys.exit(0)

    ## What night are we running on?
    true_night = what_night_is_it()
    if override_night is not None:
        night = int(override_night)
        print(f"True night is {true_night}, but running for night={night}")
    else:
        night = true_night

    if continue_looping_debug:
        print("continue_looping_debug is set. Will continue looking for new data and need to be terminated by the user.")

    ## Get default values for input variables
    if procobstypes is None:
        procobstypes = default_exptypes_for_proctable()
    if expobstypes is None:
        ## Define the obstypes to save information for in the exposure table
        expobstypes = default_exptypes_for_exptable()

    ## Must contain all the types used in processing
    for typ in procobstypes:
        if typ not in expobstypes:
            expobstypes.append(typ)

    if camword is not None:
        print(f"Overriding camword in data with user provided value: {camword}")

    ## Adjust wait times if simulating things
    speed_modifier = 1
    if dry_run:
        speed_modifier = 0.1

    ## Get context specific variable values
    surveynum = get_surveynum(night)
    nersc_start = nersc_start_time(night=true_night)
    nersc_end = nersc_end_time(night=true_night)
    colnames, coltypes, coldefaults = get_exposure_table_column_defs(return_default_values=True)

    ## Define where to find the data
    path_to_data = verify_variable_with_environment(var=path_to_data,var_name='path_to_data', env_name='DESI_SPECTRO_DATA')

    specprod = verify_variable_with_environment(var=specprod,var_name='specprod',env_name='SPECPROD')

    ## Determine where the exposure table will be written
    if exp_table_path is None:
        exp_table_path = get_exposure_table_path(night=night)
    os.makedirs(exp_table_path,exist_ok=True)
    name = get_exposure_table_name(night=night, extension=tab_filetype)
    exp_table_pathname = pathjoin(exp_table_path, name)

    ## Determine where the processing table will be written
    if proc_table_path is None:
        proc_table_path = get_processing_table_path()
    os.makedirs(proc_table_path, exist_ok=True)
    name = get_processing_table_name(prodmod=night, extension=tab_filetype)
    proc_table_pathname = pathjoin(proc_table_path, name)

    ## Determine where the unprocessed data table will be written
    unproc_table_pathname = pathjoin(proc_table_path,name.replace('processing', 'unprocessed'))

    ## Combine the table names and types for easier passing to io functions
    table_pathnames = [exp_table_pathname, proc_table_pathname, unproc_table_pathname]
    table_types = ['etable','ptable','unproc_table']

    ## Load in the files defined above
    etable, ptable, unproc_table = load_tables(tablenames=table_pathnames, \
                                               tabletypes=table_types)

    ## Get relevant data from the tables
    all_exps = set(etable['EXPID'])
    arcs,flats,sciences, arcjob,flatjob, \
    curtype,lasttype, curtile,lasttile, internal_id, last_not_dither = parse_previous_tables(etable, ptable, night)

    ## While running on the proper night and during night hours,
    ## or doing a dry_run or override_night, keep looping
    while ( (night == what_night_is_it()) and during_operating_hours(dry_run=dry_run) ) or ( override_night is not None ):
        ## Get a list of new exposures that have been found
        print(f"\n\n\nPreviously known exposures: {all_exps}")
        located_exps = set( np.array(listpath(path_to_data,str(night))).astype(int) )
        new_exps = located_exps.difference(all_exps)
        all_exps = located_exps # i.e. new_exps.union(all_exps)
        print(f"New exposures: {new_exps}")

        ## If there aren't any new exps and there won't be more because we're running on an old night or simulating things, exit
        if (not continue_looping_debug) and ( override_night is not None ) and ( len(list(new_exps))==0 ):
            print("Terminating the search for new exposures because no new exposures are present and you have" + \
                  " override_night set without continue_looping_debug")
            break

        ## Loop over new exposures and process them as relevant to that type
        for exp in sorted(list(new_exps)):
            ## Open relevant raw data files to understand what we're dealing with
            erow = summarize_exposure(path_to_data,night,exp,expobstypes,surveynum,colnames,coldefaults,verbosely=False)
            print(f"\nFound: {erow}")

            ## If there was an issue, continue. If it's a string summarizing the end of some sequence, use that info.
            ## If the exposure is assosciated with data, process that data.
            if erow is None:
                continue
            elif type(erow) is str:
                if 'short' in erow and flatjob is None:
                    flats = []
                elif 'long' in erow and flatjob is None:
                    ptable, flatjob = flat_joint_fit(ptable, flats, internal_id, dry_run=dry_run, queue=queue)
                    internal_id += 1
                elif 'arc' in erow and arcjob is None:
                    ptable, arcjob = arc_joint_fit(ptable, arcs, internal_id, dry_run=dry_run, queue=queue)
                    internal_id += 1
                else:
                    continue
            else:
                etable.add_row(erow)

                if erow['OBSTYPE'] not in procobstypes:
                    unproc_table.add_row(erow)
                    continue
                elif 'system test' in erow['PROGRAM'].lower():
                    unproc_table.add_row(erow)
                    continue

                curtype,curtile = get_type_and_tile(erow)

                if (curtype != lasttype) or (curtile != lasttile):
                    ptable, arcjob, flatjob, internal_id = checkfor_and_submit_joint_job(ptable, arcs, flats, sciences,
                                                                                         arcjob, flatjob, lasttype,
                                                                                         last_not_dither, internal_id,
                                                                                         dry_run=dry_run, queue=queue)

                prow = erow_to_prow(erow)

                if camword is not None:
                    prow['CAMWORD'] = camword
                prow['INTID'] = internal_id
                internal_id += 1
                prow = define_and_assign_dependency(prow, arcjob, flatjob)
                print(f"Processing: {prow}")
                prow = create_and_submit(prow, dry_run=dry_run, queue=queue)
                ptable.add_row(prow)

                if curtype == 'flat' and flatjob is None and int(erow['SEQTOT']) < 5:
                    flats.append(prow)
                elif curtype == 'arc' and arcjob is None:
                    arcs.append(prow)
                elif curtype == 'science' and last_not_dither:
                    sciences.append(prow)

                lasttile = curtile
                lasttype = curtype
                last_not_dither = (prow['OBSDESC'] != 'dither')

            time.sleep(10)
            write_tables([etable, ptable, unproc_table],
                         tablenames=[exp_table_pathname, proc_table_pathname, unproc_table_pathname])

        print("\nReached the end of curent iteration of new exposures.")
        print("Waiting {}s before looking for more new data".format(data_cadence_time*speed_modifier))
        time.sleep(data_cadence_time*speed_modifier)

        if len(ptable) > 0:
            ptable = update_from_queue(ptable, start_time=nersc_start, end_time=nersc_end, dry_run=dry_run)
            # ptable, nsubmits = update_and_recurvsively_submit(ptable,start_time=nersc_start,end_time=nersc_end,
            #                                                   ptab_name=proc_table_pathname, dry_run=dry_run)

            ## Exposure table doesn't change in the interim, so no need to re-write it to disk
            write_table(ptable, tablename=proc_table_pathname)
            time.sleep(30*speed_modifier)
        
    ## No more data coming in, so do bottleneck steps if any apply
    ptable, arcjob, flatjob, internal_id = checkfor_and_submit_joint_job(ptable, arcs, flats, sciences, \
                                                                         arcjob, flatjob, lasttype, last_not_dither,\
                                                                         internal_id, dry_run=dry_run, queue=queue)


    ## All jobs now submitted, update information from job queue and save
    ptable = update_from_queue(ptable,start_time=nersc_start,end_time=nersc_end, dry_run=dry_run)
    write_table(ptable, tablename=proc_table_pathname)

    print(f"Completed submission of exposures for night {night}.")

    # #######################################
    # ########## Queue Cleanup ##############
    # #######################################
    # print("Now resolving job failures.")
    #
    # ## Now we resubmit failed jobs and their dependencies until all jobs have un-submittable end state
    # ## e.g. they either succeeded or failed with a code-related issue
    # ii,nsubmits = 0, 0
    # while ii < 4 and any_jobs_not_complete(ptable['STATUS']):
    #     print(f"Starting iteration {ii} of queue updating and resubmissions of failures.")
    #     ptable, nsubmits = update_and_recurvsively_submit(ptable, submits=nsubmits, start_time=nersc_start,end_time=nersc_end,
    #                                                       ptab_name=proc_table_pathname, dry_run=dry_run)
    #     write_table(ptable, tablename=proc_table_pathname)
    #     if any_jobs_not_complete(ptable['STATUS']):
    #         time.sleep(queue_cadence_time*speed_modifier)
    #
    #     ptable = update_from_queue(ptable,start_time=nersc_start,end_time=nersc_end)
    #     write_table(ptable, tablename=proc_table_pathname)
    #     ii += 1
    #
    # print("No job failures left.")
    print("Exiting")
