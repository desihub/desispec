#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os
import time
from astropy.table import Table


## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.helper_funcs import get_logger, what_night_is_it, during_operating_hours, opj, listpath
from desispec.workflow.helper_funcs import get_surveynum, nersc_start_time, nersc_end_time, night_to_starting_iid
from desispec.workflow.helper_funcs import load_tables, load_table, write_table, write_tables, get_type_and_tile
from desispec.workflow.helper_funcs import create_and_submit_exposure, update_and_recurvsively_submit, continue_looping, update_from_queue
from desispec.workflow.helper_funcs import science_joint_fit, arc_joint_fit, flat_joint_fit, define_and_assign_dependency,verify_variable_with_environment
from desispec.workflow.create_exposure_tables import create_exposure_table, summarize_exposure, default_exptypes_for_exptable
from desispec.workflow.create_processing_table import create_processing_table, erow_to_irow
from desispec.workflow.create_exposure_tables import get_exposure_table_path, get_exposure_table_pathname, get_exposure_table_name
from desispec.workflow.create_processing_table import get_processing_table_path, get_processing_table_pathname, get_processing_table_name

log = get_logger()






def daily_processing_manager(specprod=None, exp_table_path=None, proc_table_path=None, path_to_data=None, scitypes=None,
                             dry_run=False, tab_filetype='csv'):
    ## Check this in running script. This allows the function to be used manually in interactive mode
    # if not during_operating_hours(mock=mock) :
    #     exit(0)

    night = what_night_is_it()
    surveynum = get_surveynum(night)
    nersc_start = nersc_start_time(obsnight=night)
    nersc_end = nersc_end_time(obsnight=night)

    if scitypes is None:
        scitypes = default_exptypes_for_exptable()

    ## Define where to find the data
    path_to_data = verify_variable_with_environment(var=path_to_data,var_name='path_to_data',
                                                    env_name='DESI_SPECTRO_DATA', output_mechanism=log.warning)

    # night = verify_variable_with_environment(var=night,var_name='night',env_name='PROD_NIGHT',
    #                                             output_mechanism=log.warning)

    specprod = verify_variable_with_environment(var=specprod,var_name='specprod',env_name='SPECPROD',
                                                output_mechanism=log.warning)

    if exp_table_path is None:
        exp_table_path = get_exposure_table_path(night=night)

    os.makedirs(exp_table_path,exist_ok=True)
    name = get_exposure_table_name(night=night, extension=tab_filetype)
    exp_table_pathname = opj(exp_table_path, name)

    if proc_table_path is None:
        proc_table_path = get_processing_table_path()

    os.makedirs(proc_table_path, exist_ok=True)
    name = get_processing_table_name(prodmod=night, extension=tab_filetype)
    proc_table_pathname = opj(proc_table_path, name)

    unproc_table_pathname = opj(proc_table_path,name.replace('processing', 'unprocessed'))


    arcs, flats, sciences = [], [], []
    arcjob, flatjob = None, None
    curtype,lasttype = None,None
    curtile,lasttile = None,None
    if os.path.isfile(proc_table_pathname):
        etable,itable,unproc_table = load_tables([exp_table_pathname,proc_table_pathname,unproc_table_pathname])
        if len(itable) > 0:
            irow = itable[-1]
            internal_id = int(irow['INTID'])+1
            lasttype,lasttile = get_type_and_tile(itable[-1])
            last_not_dither = (irow['OBSDESC'] != 'dither')
            jobtypes = itable['JOBDESC']

            if 'psfnight' in jobtypes:
                arcjob = itable[jobtypes=='psfnight']
            elif lasttype == 'arc':
                arcs = []
                seqnum = 10
                for row in itable[::-1]:
                    if row['OBSTYPE'].lower() == 'arc' and int(row['SEQNUM'])<seqnum:
                        arcs.append(row)
                        seqnum = int(row['SEQNUM'])

            if 'nightlyflat' in jobtypes:
                flatjob = itable[jobtypes=='nightlyflat']
            elif lasttype == 'flat' and int(itable['SEQTOT'][-1]) < 5:
                flats = []
                for row in itable[::-1]:
                    if row['OBSTYPE'].lower() == 'flat' and int(row['SEQTOT'])<5:
                        flats.append(row)

            if lasttype.lower() == 'science':
                for row in itable[::-1]:
                    if row['OBSTYPE'].lower() == 'science' and row['TILEID'] == lasttile and \
                               row['JOBDESC'] == 'prestd' and row['OBSDESC'] != 'dither':
                        sciences.append(row)
        else:
            internal_id = night_to_starting_iid(night)
            last_not_dither = True

    else:
        etable = create_exposure_table()
        unproc_table = etable.copy()
        itable = create_processing_table()
        internal_id = night_to_starting_iid(night)
        last_not_dither = True

    all_exps = set(etable['EXPID'])
    while what_night_is_it() == night and during_operating_hours(dry_run=dry_run) :
        located_exps = set( np.array(listpath(path_to_data,str(night))).astype(int) )
        newexps = located_exps.difference(all_exps)
        for exp in newexps:
            erow = summarize_exposure(path_to_data,night,exp,scitypes,surveynum,etable.colnames,verbosely=False)
            if erow is None:
                continue
            elif type(erow) is str:
                if 'short' in erow and flatjob is None:
                    flats = []
                elif 'long' in erow and flatjob is None:
                    etable, itable, flatjob = flat_joint_fit(etable, itable, flats, internal_id)
                    internal_id += 1
                elif 'arc' in erow and arcjob is None:
                    etable, itable, arcjob = arc_joint_fit(etable, itable, arcs, internal_id)
                    internal_id += 1
            else:
                etable.add_row(erow)

                if erow['OBSTYPE'] not in scitypes:
                    unproc_table.add_row(erow)
                    continue

                curtype,curtile = get_type_and_tile(erow)

                if (curtype != lasttype) or (curtile != lasttile):
                    if lasttype == 'science' and last_not_dither:
                        etable, itable, tilejob = science_joint_fit(etable, itable, sciences, internal_id)
                        internal_id += 1
                        sciences = []
                    elif lasttype == 'flat' and flatjob is None and len(flats) > 10:
                        etable, itable, flatjob = flat_joint_fit(etable, itable, flats, internal_id)
                        internal_id += 1
                    elif curtype == 'arc' and arcjob is None and len(arcs) > 4:
                        etable, itable, arcjob = arc_joint_fit(etable, itable, arcs, internal_id)
                        internal_id += 1

                irow = erow_to_irow(erow)
                irow['INTID'] = internal_id
                internal_id += 1
                irow = define_and_assign_dependency(irow, arcjob, flatjob)

                irow = create_and_submit_exposure(irow)
                itable.add_row(irow)

                if curtype == 'flat' and flatjob is None and int(erow['SEQTOT']) < 5:
                    flats.append(irow)
                elif curtype == 'arc' and arcjob is None:
                    arcs.append(irow)
                elif curtype == 'science' and last_not_dither:
                    sciences.append(irow)

                lasttile = curtile
                lasttype = curtype
                last_not_dither = (irow['OBSDESC'] != 'dither')

            time.sleep(10)
            write_tables([etable, itable, unproc_table],
                         fullpathnames=[exp_table_pathname,proc_table_pathname,unproc_table_pathname])

        time.sleep(300)

        itable = update_and_recurvsively_submit(itable,start_time=nersc_start,end_time=nersc_end,
                                                itab_name=proc_table_pathname)

        ## Exposure table doesn't change in the interim, so no need to re-write it to disk
        write_table(itable, tablename=proc_table_pathname)
        time.sleep(300)

    ## No more data coming in, so do bottleneck steps if any apply
    if lasttype == 'science' and last_not_dither:
        etable, itable, tilejob = science_joint_fit(etable, itable, sciences, internal_id)
        internal_id += 1
    elif lasttype == 'flat' and flatjob is None and len(flats) > 5:
        etable, itable, flatjob = flat_joint_fit(etable, itable, flats, internal_id)
        internal_id += 1
    elif curtype == 'arc' and arcjob is None and len(arcs) > 4:
        etable, itable, arcjob = arc_joint_fit(etable, itable, arcs, internal_id)
        internal_id += 1

    ## All jobs now submitted, update information from job queue and save
    itable = update_from_queue(itable,start_time=nersc_start,end_time=nersc_end, dry_run=dry_run)
    write_table(itable, tablename=proc_table_pathname)

    ## Now we resubmit failed jobs and their dependencies until all jobs have un-submittable end state
    ## e.g. they either succeeded or failed with a code-related issue
    ii = 0
    while ii < 4 and continue_looping(itable['STATUS']):
        print(f"Starting iteration {ii}")
        itable, nsubmits = update_and_recurvsively_submit(itable,start_time=nersc_start,end_time=nersc_end,
                                                          itab_name=proc_table_pathname, dry_run=dry_run)
        write_table(itable, tablename=proc_table_pathname)
        if not dry_run and continue_looping(itable['STATUS']):
            time.sleep(1800)

        itable = update_from_queue(itable,start_time=nersc_start,end_time=nersc_end)
        write_table(itable, tablename=proc_table_pathname)
        ii += 1




if __name__ == '__main__':
    daily_processing_manager(dry_run=True)
