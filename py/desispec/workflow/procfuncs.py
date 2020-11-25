import os
import glob
import json
from astropy.io import fits
from astropy.table import Table, join
import numpy as np
# import numpy as np

import argparse
import re
import time, datetime
import psutil
from os import listdir
from collections import OrderedDict
import subprocess
import sys
from copy import deepcopy



from desispec.workflow.queue import get_resubmission_states, update_from_queue
from desispec.workflow.timing import what_night_is_it
from desispec.workflow.desi_proc_funcs import get_desi_proc_batch_file_pathname, create_desi_proc_batch_script, \
    get_desi_proc_batch_file_path
from desispec.workflow.utils import pathjoin
from desispec.workflow.tableio import write_table


#################################################
############## Misc Functions ###################
#################################################
def night_to_starting_iid(night=None):
    if night is None:
        night = what_night_is_it()
    night = int(night)
    internal_id = (night - 20000000) * 1000
    return internal_id



#################################################
############ Script Functions ###################
#################################################
def batch_script_name(prow):
    pathname = get_desi_proc_batch_file_pathname(night = prow['NIGHT'], exp=prow['EXPID'], \
                                             jobdesc=prow['JOBDESC'], cameras=prow['CAMWORD'])
    scriptfile =  pathname + '.slurm'
    return scriptfile

def create_and_submit(prow, queue='realtime', dry_run=False, joint=False):
    prow = create_batch_script(prow, queue=queue, dry_run=dry_run, joint=joint)
    prow = submit_batch_script(prow, dry_run=dry_run)
    return prow

def desi_proc_command(prow, queue=None):
    if prow is None:
        import pdb
        pdb.set_trace()
    cmd = 'desi_proc'
    cmd += ' --batch'
    cmd += ' --nosubmit'
    cmd += ' --traceshift'
    if queue is not None:
        cmd += f' -q {queue}'
    if prow['OBSTYPE'].lower() == 'science':
        if prow['JOBDESC'] == 'prestdstar':
            cmd += ' --nostdstarfit --nofluxcalib'
        elif prow['JOBDESC'] == 'poststdstar':
            cmd += ' --noprestdstarfit --nostdstarfit'
    specs = ','.join(str(prow['CAMWORD'])[1:])
    cmd += ' --cameras={} -n {} -e {}'.format(specs, prow['NIGHT'], prow['EXPID'][0])
    return cmd

def desi_proc_joint_fit_command(prow, queue=None):
    cmd = 'desi_proc_joint_fit'
    cmd += ' --batch'
    cmd += ' --nosubmit'
    cmd += ' --traceshift'
    if queue is not None:
        cmd += f' -q {queue}'

    descriptor = prow['OBSTYPE'].lower()
        
    night = prow['NIGHT']
    specs = ','.join(str(prow['CAMWORD'])[1:])
    expids = prow['EXPID']
    expid_str = ','.join([str(eid) for eid in expids])

    cmd += f' --obstype {descriptor}'
    cmd += ' --cameras={} -n {} -e {}'.format(specs, night, expid_str)
    return cmd

def create_batch_script(prow, queue='realtime', dry_run=False, joint=False):
    if joint:
        cmd = desi_proc_joint_fit_command(prow, queue=queue)
    else:
        cmd = desi_proc_command(prow, queue=queue)

    #print(cmd)

    scriptpathname = batch_script_name(prow)
    if dry_run:
        print("Output file would have been: {}".format(scriptpathname))
        print("Command to be run: {}".format(cmd.split()))
    else:
        print("Running: {}".format(cmd.split()))
        scriptpathname = create_desi_proc_batch_script(night=prow['NIGHT'], exp=prow['EXPID'], \
                                                       cameras=prow['CAMWORD'], jobdesc=prow['JOBDESC'], \
                                                       queue=queue, cmdline=cmd)
        print("Outfile is: ".format(scriptpathname))

    prow['SCRIPTNAME'] = os.path.basename(scriptpathname)
    return prow


def submit_batch_script(submission, dry_run=False, strictly_successful=False):
    jobname = batch_script_name(submission)
    dependencies = submission['LATEST_DEP_QID']
    dep_list, dep_str = '', ''
    if dependencies is not None:
        jobtype = submission['JOBDESC']
        if strictly_successful:
            depcond = 'afterok'
        elif jobtype in ['flat','nightlyflat','poststdstar']:
            depcond = 'afterok'
        else:
            ## if arc, psfnight, prestdstar, or stdstarfit, any inputs is fine
            ## (though psfnight and stdstarfit will require some inputs otherwise they'll go up in flames)
            depcond = 'afterany'

        dep_str = f'--dependency={depcond}:'

        if np.isscalar(dependencies):
            dep_list = str(dependencies).strip(' \t')
            if dep_list == '':
                dep_str = ''
            else:
                dep_str += dep_list
        else:
            if len(dependencies)>1:
                dep_list = ':'.join(np.array(dependencies).astype(str))
                dep_str += dep_list
            elif len(dependencies) == 1 and dependencies[0] not in [None, 0]:
                dep_str += str(dependencies[0])
            else:
                dep_str = ''

    ## True function will actually submit to SLURM
    if dry_run:
        current_qid = int(time.time() - 1.6e9)
    else:
        # script = f'{jobname}.slurm'
        # script_path = pathjoin(batchdir, script)
        batchdir = get_desi_proc_batch_file_path(night=submission['NIGHT'])
        script_path = pathjoin(batchdir, jobname)
        if dep_str == '':
            current_qid = subprocess.check_output(['sbatch', '--parsable', f'{script_path}'],
                                                  stderr=subprocess.STDOUT, text=True)
        else:
            current_qid = subprocess.check_output(['sbatch', '--parsable',f'{dep_str}',f'{script_path}'],
                                                  stderr=subprocess.STDOUT, text=True)
        current_qid = int(current_qid.strip(' \t\n'))

    print(f'Submitted {jobname}   with dependencies {dep_str}. Returned qid: {current_qid}')

    submission['LATEST_QID'] = current_qid
    submission['ALL_QIDS'] = np.append(submission['ALL_QIDS'],current_qid)
    submission['STATUS'] = 'SU'
    submission['SUBMIT_DATE'] = int(time.time())
    
    return submission


#########################################
########     Joint fit     ##############
#########################################
def joint_fit(ptable, prows, internal_id, queue, descriptor, dry_run=False):
    if descriptor is None:
        return ptable, None
    elif descriptor == 'science':
        descriptor = 'stdstarfit'
    elif descriptor == 'arc':
        descriptor = 'psfnight'
    elif descriptor == 'flat':
        descriptor = 'nightlyflat'

    if descriptor not in ['stdstarfit', 'psfnight', 'nightlyflat']:
        return ptable, None

    joint_prow = make_joint_prow(prows, descriptor=descriptor, initid=internal_id)
    joint_prow = create_and_submit(joint_prow, queue=queue, joint=True, dry_run=dry_run)
    ptable.add_row(joint_prow)

    if descriptor == 'stdstarfit':
        for row in prows:
            row['JOBDESC'] = 'poststdstar'
            row['INTID'] = internal_id
            row['ALL_QIDS'] = np.ndarray(shape=0).astype(int)
            internal_id += 1
            row = assign_dependency(row, joint_prow)
            row = create_and_submit(row, dry_run=dry_run)
            ptable.add_row(row)
    else:
        ptable = set_calibrator_flag(prows, ptable)

    return ptable, joint_prow


## wrapper functions for joint fitting
def science_joint_fit(ptable, sciences, internal_id, queue='realtime', dry_run=False):
    return joint_fit(ptable=ptable, prows=sciences, internal_id=internal_id, queue=queue, descriptor='stdstarfit', dry_run=dry_run)

def flat_joint_fit(ptable, flats, internal_id, queue='realtime', dry_run=False):
    return joint_fit(ptable=ptable, prows=flats, internal_id=internal_id, queue=queue, descriptor='nightlyflat', dry_run=dry_run)

def arc_joint_fit(ptable, arcs, internal_id, queue='realtime', dry_run=False):
    return joint_fit(ptable=ptable, prows=arcs, internal_id=internal_id, queue=queue, descriptor='psfnight', dry_run=dry_run)

def make_joint_prow(prows, descriptor, initid):
    if type(prows[0]) in [dict, OrderedDict]:
        prow = prows[0].copy()
    else:
        prow = OrderedDict()
        for nam in prows[0].colnames:
            prow[nam] = prows[0][nam]

    prow['INTID'] = initid
    prow['JOBDESC'] = descriptor
    prow['ALL_QIDS'] = np.ndarray(shape=0).astype(int)
    if type(prows) in [list, np.array]:
        ids, qids, expids = [], [], []
        for currow in prows:
            ids.append(currow['INTID'])
            qids.append(currow['LATEST_QID'])
            expids.append(currow['EXPID'][0])
        prow['INT_DEP_IDS'] = np.array(ids)
        prow['LATEST_DEP_QID'] = np.array(qids)
        prow['EXPID'] = np.array(expids)
    else:
        prow['INT_DEP_IDS'] = np.array([prows['INTID']])
        prow['LATEST_DEP_QID'] = np.array([prows['LATEST_QID']])
        prow['EXPID'] = prows['EXPID']

    return prow

def checkfor_and_submit_joint_job(ptable, arcs, flats, sciences, flatjob, arcjob, \
                                  lasttype, last_not_dither, internal_id, dry_run=False, queue='realtime'):
    if lasttype == 'science' and last_not_dither:
        ptable, tilejob = science_joint_fit(ptable, sciences, internal_id, dry_run=dry_run, queue=queue)
        internal_id += 1
    elif lasttype == 'flat' and flatjob is None and len(flats) > 10:
        ptable, flatjob = flat_joint_fit(ptable, flats, internal_id, dry_run=dry_run, queue=queue)
        internal_id += 1
    elif lasttype == 'arc' and arcjob is None and len(arcs) > 4:
        ptable, arcjob = arc_joint_fit(ptable, arcs, internal_id, dry_run=dry_run, queue=queue)
        internal_id += 1
    return ptable, arcjob, flatjob, internal_id

#############################################
###### Various Table / row manipulators #####
#############################################
def parse_previous_tables(etable, ptable, night):
    arcs, flats, sciences = [], [], []
    arcjob, flatjob = None, None
    curtype,lasttype = None,None
    curtile,lasttile = None,None

    if len(ptable) > 0:
        prow = ptable[-1]
        internal_id = int(prow['INTID'])+1
        lasttype,lasttile = get_type_and_tile(ptable[-1])
        last_not_dither = (prow['OBSDESC'] != 'dither')
        jobtypes = ptable['JOBDESC']

        if 'psfnight' in jobtypes:
            arcjob = ptable[jobtypes=='psfnight'][0]
        elif lasttype == 'arc':
            arcs = []
            seqnum = 10
            for row in ptable[::-1]:
                erow = etable[etable['EXPID']==row['EXPID'][0]]
                if row['OBSTYPE'].lower() == 'arc' and int(erow['SEQNUM'])<seqnum:
                    arcs.append(row)
                    seqnum = int(erow['SEQNUM'])
                else:
                    break

        if 'nightlyflat' in jobtypes:
            flatjob = ptable[jobtypes=='nightlyflat'][0]
        elif lasttype == 'flat':
            flats = []
            for row in ptable[::-1]:
                erow = etable[etable['EXPID']==row['EXPID'][0]]
                if row['OBSTYPE'].lower() == 'flat' and int(erow['SEQTOT'])<5:
                    flats.append(row)
                else:
                    break

        if lasttype.lower() == 'science':
            for row in ptable[::-1]:
                if row['OBSTYPE'].lower() == 'science' and row['TILEID'] == lasttile and \
                   row['JOBDESC'] == 'prestdstar' and row['OBSDESC'] != 'dither':
                    sciences.append(row)
                else:
                    break
    else:
        internal_id = night_to_starting_iid(night)
        last_not_dither = True

    return arcs,flats,sciences, \
           arcjob, flatjob, \
           curtype, lasttype, \
           curtile, lasttile,\
           internal_id, last_not_dither

def set_calibrator_flag(matchrows, ptable):
    for prow in matchrows:
        ptable['CALIBRATOR'][ptable['INTID'] == prow['INTID']] = 1
    return ptable


def define_and_assign_dependency(prow, arcjob, flatjob):
    prow['JOBDESC'] = prow['OBSTYPE']
    if prow['OBSTYPE'] in ['science', 'twiflat']:
        dependency = flatjob
        prow['JOBDESC'] = 'prestdstar'
    elif prow['OBSTYPE'] == 'flat':
        dependency = arcjob
    else:
        dependency = None

    prow = assign_dependency(prow, dependency)

    return prow


def assign_dependency(prow, dependency):
    if dependency is not None:
        if type(dependency) in [list, np.array]:
            ids, qids = [], []
            for curdep in dependency:
                ids.append(curdep['INTID'])
                qids.append(curdep['LATEST_QID'])
            prow['INT_DEP_IDS'] = np.array(ids)
            prow['LATEST_DEP_QID'] = np.array(qids)
        else:
            prow['INT_DEP_IDS'] = np.array([dependency['INTID']])
            prow['LATEST_DEP_QID'] = np.array([dependency['LATEST_QID']])
    return prow


def get_type_and_tile(erow):
    return str(erow['OBSTYPE']).lower(), erow['TILEID']

def recursive_submit_failed(rown, proc_table, submits, id_to_row_map, ptab_name=None,
                            resubmission_states=None, dry_run=False):
    if resubmission_states is None:
        resubmission_states = get_resubmission_states()
    ideps = proc_table['INT_DEP_IDS'][rown]
    if ideps is None:
        proc_table['LATEST_DEP_QID'][rown] = None
    else:
        qdeps = []
        for idep in np.sort(np.atleast_1d(ideps)):
            if proc_table['STATUS'][id_to_row_map[idep]] in resubmission_states:
                proc_table, submits = recursive_submit_failed(id_to_row_map[idep], \
                                                              proc_table, submits, id_to_row_map)
            qdeps.append(proc_table['LATEST_QID'][id_to_row_map[idep]])

        qdeps = np.atleast_1d(qdeps)
        if len(qdeps) > 0:
            proc_table['LATEST_DEP_QID'][rown] = qdeps
        else:
            print("Error: number of qdeps should be 1 or more")
            print(f'Rown {rown}, ideps {ideps}')

    proc_table[rown] = submit_batch_script(proc_table[rown], dry_run=dry_run)
    submits += 1

    if dry_run:
        pass
    else:
        time.sleep(2)
        if submits % 10 == 0:
            if ptab_name is None:
                write_table(proc_table, table_type='processing', overwrite=True)
            else:
                write_table(proc_table, tablename=ptab_name, overwrite=True)
            time.sleep(60)
        if submits % 100 == 0:
            time.sleep(540)
            proc_table = update_from_queue(proc_table)
            if ptab_name is None:
                write_table(proc_table, table_type='processing', overwrite=True)
            else:
                write_table(proc_table, tablename=ptab_name, overwrite=True)

    return proc_table, submits

def update_and_recurvsively_submit(proc_table, submits=0, resub_states=None, start_time=None, end_time=None,
                                   ptab_name=None, dry_run=False):
    if resub_states is None:
        resub_states = get_resubmission_states()
    proc_table = update_from_queue(proc_table, start_time=start_time, end_time=end_time)
    id_to_row_map = {row['INTID']: rown for rown, row in enumerate(proc_table)}
    for rown in range(len(proc_table)):
        if proc_table['STATUS'][rown] in resub_states:
            proc_table, submits = recursive_submit_failed(rown, proc_table, \
                                                                   submits, id_to_row_map, ptab_name,
                                                                   resub_states, dry_run)
    return proc_table, submits


