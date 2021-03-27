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
from desispec.workflow.proctable import table_row_to_dict
from desiutil.log import get_logger

from desispec.io import findfile
from desispec.io.util import decode_camword, create_camword, difference_camwords, camword_to_spectros

#################################################
############## Misc Functions ###################
#################################################
def night_to_starting_iid(night=None):
    """
    Creates an internal ID for a given night. The resulting integer is an 8 digit number.
    The digits are YYMMDDxxx where YY is the years since 2000, MM and DD are the month and day. xxx are 000,
    and are incremented for up to 1000 unique job ID's for a given night.

    Args:
        night, str or int. YYYYMMDD of the night to get the starting internal ID for.

    Returns:
        internal_id, int. 9 digit number consisting of YYMMDD000. YY is years after 2000, MMDD is month and day.
                          000 being the starting job number (0).
    """
    if night is None:
        night = what_night_is_it()
    night = int(night)
    internal_id = (night - 20000000) * 1000
    return internal_id



#################################################
############ Script Functions ###################
#################################################
def batch_script_name(prow):
    """
    Wrapper script that takes a processing table row (or dictionary with NIGHT, EXPID, JOBDESC, PROCCAMWORD defined)
    and determines the script file pathname as defined by desi_proc's helper functions.

    Args:
        prow, Table.Row or dict. Must include keyword accessible definitions for 'NIGHT', 'EXPID', 'JOBDESC', and 'PROCCAMWORD'.

    Returns:
        scriptfile, str. The complete pathname to the script file, as it is defined within the desi_proc ecosystem.
    """
    pathname = get_desi_proc_batch_file_pathname(night = prow['NIGHT'], exp=prow['EXPID'], \
                                             jobdesc=prow['JOBDESC'], cameras=prow['PROCCAMWORD'])
    scriptfile =  pathname + '.slurm'
    return scriptfile

def check_for_outputs_on_disk(prow, resubmit_partial_complete=True):
    """
    Args:
        prow, Table.Row or dict. Must include keyword accessible definitions for processing_table columns found in
                                 desispect.workflow.proctable.get_processing_table_column_defs()
        resubmit_partial_complete, bool. Default is True. Must be used with check_for_outputs=True. If this flag is True,
                                         jobs with some prior data are pruned using PROCCAMWORD to only process the
                                         remaining cameras not found to exist.

    Returns:
        prow, Table.Row or dict. The same prow type and keywords as input except with modified values updated to reflect
                                 the change in job status after creating and submitting the job for processing.
    """
    log = get_logger()

    job_to_file_map = {'prestdstar': 'sframe', 'stdstarfit': 'stdstars', 'poststdstar': 'cframe',
                       'arc': 'psf', 'flat': 'fiberflat', 'psfnight': 'psfnight', 'nightlyflat': 'fiberflatnight',
                       'spectra': 'spectra_tile', 'coadds': 'coadds_tile', 'redshift': 'zbest_tile'}

    night = prow['NIGHT']
    filetype = job_to_file_map[prow['JOBDESC']]
    orig_camword = prow['PROCCAMWORD']

    ## if spectro based, look for spectros, else look for cameras
    if prow['JOBDESC'] in ['stdstarfit','spectra','coadds','redshift']:
        ## Spectrograph based
        spectros = camword_to_spectros(prow['PROCCAMWORD'])
        n_desired = len(spectros)
        ## Suppress outputs about using tile based files in findfile if only looking for stdstarfits
        if prow['JOBDESC'] == 'stdstarfit':
            tileid = None
        else:
            tileid = prow['TILEID']
        existing_spectros = []
        for spectro in spectros:
            expid = prow['EXPID'][0]
            if os.path.exists(findfile(filetype=filetype, night=night, expid=expid, spectrograph=spectro, tile=tileid)):
                existing_spectros.append(spectro)
        completed = (len(existing_spectros)==len(spectros))
        if not completed and resubmit_partial_complete and len(existing_spectros) > 0:
            existing_camword = 'a' + ''.join([str(spec) for spec in sorted(existing_spectros)])
            prow['PROCCAMWORD'] = difference_camwords(prow['PROCCAMWORD'],existing_camword)
    else:
        ## Otheriwse camera based
        cameras = decode_camword(prow['PROCCAMWORD'])
        n_desired = len(cameras)
        missing_cameras = []
        for cam in cameras:
            expid = prow['EXPID'][0]
            if not os.path.exists(findfile(filetype=filetype, night=night, expid=expid,camera=cam)):
                missing_cameras.append(cameras)
        completed = (len(missing_cameras) == 0)
        if not completed and resubmit_partial_complete and len(missing_cameras) < n_desired:
            prow['PROCCAMWORD'] = create_camword(missing_cameras)

    if completed:
        prow['STATUS'] = 'COMPLETED'
        log.info(f"{prow['JOBDESC']} job with exposure(s) {prow['EXPID']} already has " +
                 f"the desired {n_desired} {filetype}'s. Not submitting this job.")
    elif resubmit_partial_complete and orig_camword != prow['PROCCAMWORD']:
        log.info(f"{prow['JOBDESC']} job with exposure(s) {prow['EXPID']} already has " +
                 f"some {filetype}'s. Submitting smaller camword={prow['PROCCAMWORD']}.")
    return completed, prow


def create_and_submit(prow, queue='realtime', reservation=None, dry_run=False, joint=False,
                      strictly_successful=False, check_for_outputs=True, resubmit_partial_complete=True):
    """
    Wrapper script that takes a processing table row and three modifier keywords, creates a submission script for the
    compute nodes, and then submits that script to the Slurm scheduler with appropriate dependencies.

    Args:
        prow, Table.Row or dict. Must include keyword accessible definitions for processing_table columns found in
                                 desispect.workflow.proctable.get_processing_table_column_defs()
        queue, str. The name of the NERSC Slurm queue to submit to. Default is the realtime queue.
        reservation: str. The reservation to submit jobs to. If None, it is not submitted to a reservation.
        dry_run, bool. If true, this is a simulated run and the scripts will not be written or submitted. Output will
                       relevant for testing will be printed as though scripts are being submitted. Default is False.
        joint, bool. Whether this is a joint fitting job (the job involves multiple exposures) and therefore needs to be
                     run with desi_proc_joint_fit. Default is False.
        strictly_successful, bool. Whether all jobs require all inputs to have succeeded. For daily processing, this is
                                   less desirable because e.g. the sciences can run with SVN default calibrations rather
                                   than failing completely from failed calibrations. Default is False.
        check_for_outputs, bool. Default is True. If True, the code checks for the existence of the expected final
                                 data products for the script being submitted. If all files exist and this is True,
                                 then the script will not be submitted. If some files exist and this is True, only the
                                 subset of the cameras without the final data products will be generated and submitted.
        resubmit_partial_complete, bool. Default is True. Must be used with check_for_outputs=True. If this flag is True,
                                         jobs with some prior data are pruned using PROCCAMWORD to only process the
                                         remaining cameras not found to exist.

    Returns:
        prow, Table.Row or dict. The same prow type and keywords as input except with modified values updated to reflect
                                 the change in job status after creating and submitting the job for processing.

    Note:
        This modifies the input. Though Table.Row objects are generally copied on modification, so the change to the
        input object in memory may or may not be changed. As of writing, a row from a table given to this function will
        not change during the execution of this function (but can be overwritten explicitly with the returned row if desired).
    """
    orig_prow = prow.copy()
    if check_for_outputs:
        already_complete, prow = check_for_outputs_on_disk(prow, resubmit_partial_complete)
        if already_complete:
            return prow
    prow = create_batch_script(prow, queue=queue, dry_run=dry_run, joint=joint)
    prow = submit_batch_script(prow, reservation=reservation, dry_run=dry_run, strictly_successful=strictly_successful)
    ## If resubmitted partial, the PROCCAMWORD and SCRIPTNAME will correspond to the pruned values. But we want to
    ## retain the full job's value, so get those from the old job.
    if resubmit_partial_complete:
        prow['PROCCAMWORD'] = orig_prow['PROCCAMWORD']
        prow['SCRIPTNAME'] = orig_prow['SCRIPTNAME']
    return prow

def desi_proc_command(prow, queue=None):
    """
    Wrapper script that takes a processing table row (or dictionary with NIGHT, EXPID, OBSTYPE, JOBDESC, PROCCAMWORD defined)
    and determines the proper command line call to process the data defined by the input row/dict.

    Args:
        prow, Table.Row or dict. Must include keyword accessible definitions for 'NIGHT', 'EXPID', 'JOBDESC', and 'PROCCAMWORD'.
        queue, str. The name of the NERSC Slurm queue to submit to. Default is None (which leaves it to the desi_proc default).

    Returns:
        cmd, str. The proper command to be submitted to desi_proc to process the job defined by the prow values.
    """
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
    specs = str(prow['PROCCAMWORD'])
    cmd += ' --cameras={} -n {} -e {}'.format(specs, prow['NIGHT'], prow['EXPID'][0])
    if prow['BADAMPS'] != '':
        cmd += ' --badamps={}'.format(prow['BADAMPS'])
    return cmd

def desi_proc_joint_fit_command(prow, queue=None):
    """
    Wrapper script that takes a processing table row (or dictionary with NIGHT, EXPID, OBSTYPE, PROCCAMWORD defined)
    and determines the proper command line call to process the data defined by the input row/dict.

    Args:
        prow, Table.Row or dict. Must include keyword accessible definitions for 'NIGHT', 'EXPID', 'JOBDESC', and 'PROCCAMWORD'.
        queue, str. The name of the NERSC Slurm queue to submit to. Default is None (which leaves it to the desi_proc default).

    Returns:
        cmd, str. The proper command to be submitted to desi_proc_joint_fit to process the job defined by the prow values.
    """
    cmd = 'desi_proc_joint_fit'
    cmd += ' --batch'
    cmd += ' --nosubmit'
    cmd += ' --traceshift'
    if queue is not None:
        cmd += f' -q {queue}'

    descriptor = prow['OBSTYPE'].lower()
        
    night = prow['NIGHT']
    specs = str(prow['PROCCAMWORD'])
    expids = prow['EXPID']
    expid_str = ','.join([str(eid) for eid in expids])

    cmd += f' --obstype {descriptor}'
    cmd += ' --cameras={} -n {} -e {}'.format(specs, night, expid_str)
    return cmd

def create_batch_script(prow, queue='realtime', dry_run=False, joint=False):
    """
    Wrapper script that takes a processing table row and three modifier keywords and creates a submission script for the
    compute nodes.

    Args:
        prow, Table.Row or dict. Must include keyword accessible definitions for processing_table columns found in
                                 desispect.workflow.proctable.get_processing_table_column_defs()
        queue, str. The name of the NERSC Slurm queue to submit to. Default is the realtime queue.
        dry_run, bool. If true, this is a simulated run and the scripts will not be written or submitted. Output will
                       relevant for testing will be printed as though scripts are being submitted. Default is False.
        joint, bool. Whether this is a joint fitting job (the job involves multiple exposures) and therefore needs to be
                     run with desi_proc_joint_fit. Default is False.

    Returns:
        prow, Table.Row or dict. The same prow type and keywords as input except with modified values updated values for
                                 scriptname.

    Note:
        This modifies the input. Though Table.Row objects are generally copied on modification, so the change to the
        input object in memory may or may not be changed. As of writing, a row from a table given to this function will
        not change during the execution of this function (but can be overwritten explicitly with the returned row if desired).
    """
    log = get_logger()
    if joint:
        cmd = desi_proc_joint_fit_command(prow, queue=queue)
    else:
        cmd = desi_proc_command(prow, queue=queue)

    #log.debug(cmd)

    scriptpathname = batch_script_name(prow)
    if dry_run:
        log.info("Output file would have been: {}".format(scriptpathname))
        log.info("Command to be run: {}".format(cmd.split()))
    else:
        log.info("Running: {}".format(cmd.split()))
        scriptpathname = create_desi_proc_batch_script(night=prow['NIGHT'], exp=prow['EXPID'], \
                                                       cameras=prow['PROCCAMWORD'], jobdesc=prow['JOBDESC'], \
                                                       queue=queue, cmdline=cmd)
        log.info("Outfile is: {}".format(scriptpathname))

    prow['SCRIPTNAME'] = os.path.basename(scriptpathname)
    return prow


def submit_batch_script(prow, dry_run=False, reservation=None, strictly_successful=False):
    """
    Wrapper script that takes a processing table row and three modifier keywords and submits the scripts to the Slurm
    scheduler.

    Args:
        prow, Table.Row or dict. Must include keyword accessible definitions for processing_table columns found in
                                 desispect.workflow.proctable.get_processing_table_column_defs()
        dry_run, bool. If true, this is a simulated run and the scripts will not be written or submitted. Output will
                       relevant for testing will be printed as though scripts are being submitted. Default is False.
        reservation: str. The reservation to submit jobs to. If None, it is not submitted to a reservation.
        strictly_successful, bool. Whether all jobs require all inputs to have succeeded. For daily processing, this is
                                   less desirable because e.g. the sciences can run with SVN default calibrations rather
                                   than failing completely from failed calibrations. Default is False.

    Returns:
        prow, Table.Row or dict. The same prow type and keywords as input except with modified values updated values for
                                 scriptname.

    Note:
        This modifies the input. Though Table.Row objects are generally copied on modification, so the change to the
        input object in memory may or may not be changed. As of writing, a row from a table given to this function will
        not change during the execution of this function (but can be overwritten explicitly with the returned row if desired).
    """
    log = get_logger()
    jobname = batch_script_name(prow)
    dep_qids = prow['LATEST_DEP_QID']
    dep_list, dep_str = '', ''

    if len(dep_qids) > 0:
        jobtype = prow['JOBDESC']
        if strictly_successful:
            depcond = 'afterok'
        elif jobtype in ['flat','nightlyflat','poststdstar']:
            depcond = 'afterok'
        else:
            ## if arc, psfnight, prestdstar, or stdstarfit, any inputs is fine
            ## (though psfnight and stdstarfit will require some inputs otherwise they'll go up in flames)
            depcond = 'afterany'

        dep_str = f'--dependency={depcond}:'

        if np.isscalar(dep_qids):
            dep_list = str(dep_qids).strip(' \t')
            if dep_list == '':
                dep_str = ''
            else:
                dep_str += dep_list
        else:
            if len(dep_qids)>1:
                dep_list = ':'.join(np.array(dep_qids).astype(str))
                dep_str += dep_list
            elif len(dep_qids) == 1 and dep_qids[0] not in [None, 0]:
                dep_str += str(dep_qids[0])
            else:
                dep_str = ''

    # script = f'{jobname}.slurm'
    # script_path = pathjoin(batchdir, script)
    batchdir = get_desi_proc_batch_file_path(night=prow['NIGHT'])
    script_path = pathjoin(batchdir, jobname)
    batch_params = ['sbatch', '--parsable']
    if dep_str != '':
        batch_params.append(f'{dep_str}')
    if reservation is not None:
        batch_params.append(f'--reservation={reservation}')
    batch_params.append(f'{script_path}')

    if dry_run:
        current_qid = int(time.time() - 1.6e9)
    else:
        current_qid = subprocess.check_output(batch_params, stderr=subprocess.STDOUT, text=True)
        current_qid = int(current_qid.strip(' \t\n'))

    log.info(batch_params)
    log.info(f'Submitted {jobname} with dependencies {dep_str}  and reservation={reservation}. Returned qid: {current_qid}')

    prow['LATEST_QID'] = current_qid
    prow['ALL_QIDS'] = np.append(prow['ALL_QIDS'],current_qid)
    prow['STATUS'] = 'SUBMITTED'
    prow['SUBMIT_DATE'] = int(time.time())
    
    return prow


#############################################
##########   Row Manipulations   ############
#############################################
def define_and_assign_dependency(prow, arcjob, flatjob):
    """
    Given input processing row and possible arcjob (processing row for psfnight) and flatjob (processing row for
    nightlyflat), this defines the JOBDESC keyword and assigns the dependency appropriate for the job type of prow.

    Args:
        prow, Table.Row or dict. Must include keyword accessible definitions for 'OBSTYPE'. A row must have column names for
                                 'JOBDESC', 'INT_DEP_IDS', and 'LATEST_DEP_ID'.
        arcjob, Table.Row, dict, or NoneType. Processing row corresponding to psfnight for the night of the data in prow.
                                              This must contain keyword accessible values for 'INTID', and 'LATEST_QID'.
                                              If None, it assumes the dependency doesn't exist and no dependency is assigned.
        flatjob, Table.Row, dict, or NoneType. Processing row corresponding to nightlyflat for the night of the data in prow.
                                               This must contain keyword accessible values for 'INTID', and 'LATEST_QID'.
                                               If None, it assumes the dependency doesn't exist and no dependency is assigned.

    Returns:
        prow, Table.Row or dict. The same prow type and keywords as input except with modified values updated values for
                                 'JOBDESC', 'INT_DEP_IDS'. and 'LATEST_DEP_ID'.

    Note:
        This modifies the input. Though Table.Row objects are generally copied on modification, so the change to the
        input object in memory may or may not be changed. As of writing, a row from a table given to this function will
        not change during the execution of this function (but can be overwritten explicitly with the returned row if desired).
    """
    if prow['OBSTYPE'] in ['science', 'twiflat']:
        if flatjob is None:
            dependency = arcjob
        else:
            dependency = flatjob
        prow['JOBDESC'] = 'prestdstar'
    elif prow['OBSTYPE'] == 'flat':
        dependency = arcjob
    else:
        dependency = None

    prow = assign_dependency(prow, dependency)

    return prow


def assign_dependency(prow, dependency):
    """
    Given input processing row and possible arcjob (processing row for psfnight) and flatjob (processing row for
    nightlyflat), this defines the JOBDESC keyword and assigns the dependency appropriate for the job type of prow.

    Args:
        prow, Table.Row or dict. Must include keyword accessible definitions for 'OBSTYPE'. A row must have column names for
                                 'JOBDESC', 'INT_DEP_IDS', and 'LATEST_DEP_ID'.
        dependency, NoneType or scalar/list/array of Table.Row, dict. Processing row corresponding to the required input
                                                                      for the job in prow. This must contain keyword
                                                                      accessible values for 'INTID', and 'LATEST_QID'.
                                                                      If None, it assumes the dependency doesn't exist
                                                                      and no dependency is assigned.

    Returns:
        prow, Table.Row or dict. The same prow type and keywords as input except with modified values updated values for
                                 'JOBDESC', 'INT_DEP_IDS'. and 'LATEST_DEP_ID'.

    Note:
        This modifies the input. Though Table.Row objects are generally copied on modification, so the change to the
        input object in memory may or may not be changed. As of writing, a row from a table given to this function will
        not change during the execution of this function (but can be overwritten explicitly with the returned row if desired).
    """
    prow['INT_DEP_IDS'] = np.ndarray(shape=0).astype(int)
    prow['LATEST_DEP_QID'] = np.ndarray(shape=0).astype(int)
    if dependency is not None:
        if type(dependency) in [list, np.array]:
            ids, qids = [], []
            for curdep in dependency:
                if still_a_dependency(curdep):
                    ids.append(curdep['INTID'])
                    qids.append(curdep['LATEST_QID'])
            prow['INT_DEP_IDS'] = np.array(ids, dtype=int)
            prow['LATEST_DEP_QID'] = np.array(qids, dtype=int)
        elif type(dependency) in [dict, OrderedDict, Table.Row] and still_a_dependency(dependency):
            prow['INT_DEP_IDS'] = np.array([dependency['INTID']], dtype=int)
            prow['LATEST_DEP_QID'] = np.array([dependency['LATEST_QID']], dtype=int)
    return prow

def still_a_dependency(dependency):
    """
    Defines the criteria for which a dependency is deemed complete (and therefore no longer a dependency).

     Args:
        dependency, Table.Row or dict. Processing row corresponding to the required input for the job in prow.
                                     This must contain keyword accessible values for 'STATUS', and 'LATEST_QID'.

    Returns:
        bool. False if the criteria indicate that the dependency is completed and no longer a blocking factor (ie no longer
              a genuine dependency). Returns True if the dependency is still a blocking factor such that the slurm
              scheduler needs to be aware of the pending job.

    """
    return dependency['LATEST_QID'] > 0 and dependency['STATUS'] != 'COMPLETED'

def get_type_and_tile(erow):
    """
    Trivial function to return the OBSTYPE and the TILEID from an exposure table row

    Args:
        erow, Table.Row or dict. Must contain 'OBSTYPE' and 'TILEID' as keywords.

    Returns:
        tuple (str, str), corresponding to the OBSTYPE and TILEID values of the input erow.
    """
    return str(erow['OBSTYPE']).lower(), erow['TILEID']


#############################################
#########   Table manipulators   ############
#############################################
def parse_previous_tables(etable, ptable, night):
    """
    This takes in the exposure and processing tables and regenerates all the working memory variables needed for the
    daily processing script.

    Used by the daily processing to define most of its state-ful variables into working memory.
    If the processing table is empty, these are simply declared and returned for use.
    If the code had previously run and exited (or crashed), however, this will all the code to
    re-establish itself by redefining these values.

    Args:
        etable, Table, Exposure table of all exposures that have been dealt with thus far.
        ptable, Table, Processing table of all exposures that have been processed.
        night, str or int, the night the data was taken.

    Returns:
        arcs, list of dicts, list of the individual arc jobs used for the psfnight (NOT all
                                   the arcs, if multiple sets existed)
        flats, list of dicts, list of the individual flat jobs used for the nightlyflat (NOT
                                    all the flats, if multiple sets existed)
        sciences, list of dicts, list of the most recent individual prestdstar science exposures
                                       (if currently processing that tile)
        arcjob, dict or None, the psfnight job row if it exists. Otherwise None.
        flatjob, dict or None, the nightlyflat job row if it exists. Otherwise None.
        curtype, None, the obstype of the current job being run. Always None as first new job will define this.
        lasttype, str or None, the obstype of the last individual exposure row to be processed.
        curtile, None, the tileid of the current job (if science). Otherwise None. Always None as first
                       new job will define this.
        lasttile, str or None, the tileid of the last job (if science). Otherwise None.
        internal_id, int, an internal identifier unique to each job. Increments with each new job. This
                          is the latest unassigned value.
    """
    log = get_logger()
    arcs, flats, sciences = [], [], []
    arcjob, flatjob = None, None
    curtype,lasttype = None,None
    curtile,lasttile = None,None

    if len(ptable) > 0:
        prow = ptable[-1]
        internal_id = int(prow['INTID'])+1
        lasttype,lasttile = get_type_and_tile(ptable[-1])
        jobtypes = ptable['JOBDESC']

        if 'psfnight' in jobtypes:
            arcjob = table_row_to_dict(ptable[jobtypes=='psfnight'][0])
            log.info("Located joint fit arc job in exposure table: {}".format(arcjob))
        elif lasttype == 'arc':
            seqnum = 10
            for row in ptable[::-1]:
                erow = etable[etable['EXPID']==row['EXPID'][0]]
                if row['OBSTYPE'].lower() == 'arc' and int(erow['SEQNUM'])<seqnum:
                    arcs.append(table_row_to_dict(row))
                    seqnum = int(erow['SEQNUM'])
                else:
                    break
            ## Because we work backword to fill in, we need to reverse them to get chronological order back
            arcs = arcs[::-1]

        if 'nightlyflat' in jobtypes:
            flatjob = table_row_to_dict(ptable[jobtypes=='nightlyflat'][0])
            log.info("Located joint fit flat job in exposure table: {}".format(flatjob))
        elif lasttype == 'flat':
            for row in ptable[::-1]:
                erow = etable[etable['EXPID']==row['EXPID'][0]]
                if row['OBSTYPE'].lower() == 'flat' and int(erow['SEQTOT'])<5:
                    flats.append(table_row_to_dict(row))
                else:
                    break
            flats = flats[::-1]

        if lasttype.lower() == 'science':
            for row in ptable[::-1]:
                if row['OBSTYPE'].lower() == 'science' and row['TILEID'] == lasttile and \
                   row['JOBDESC'] == 'prestdstar' and row['LASTSTEP'] != 'skysub':
                    sciences.append(table_row_to_dict(row))
                else:
                    break
            sciences = sciences[::-1]
    else:
        internal_id = night_to_starting_iid(night)

    return arcs,flats,sciences, \
           arcjob, flatjob, \
           curtype, lasttype, \
           curtile, lasttile,\
           internal_id


def update_and_recurvsively_submit(proc_table, submits=0, resubmission_states=None, start_time=None, end_time=None,
                                   ptab_name=None, dry_run=False,reservation=None):
    """
    Given an processing table, this loops over job rows and resubmits failed jobs (as defined by resubmission_states).
    Before submitting a job, it checks the dependencies for failures. If a dependency needs to be resubmitted, it recursively
    follows dependencies until it finds the first job without a failed dependency and resubmits that. Then resubmits the
    other jobs with the new Slurm jobID's for proper dependency coordination within Slurm.

    Args:
        proc_table, Table, the processing table with a row per job.
        submits, int, the number of submissions made to the queue. Used for saving files and in not overloading the scheduler.
        resubmission_states, list or array of strings, each element should be a capitalized string corresponding to a
                                                       possible Slurm scheduler state, where you wish for jobs with that
                                                       outcome to be resubmitted
        start_time, str, datetime string in the format understood by NERSC Slurm scheduler. This should defined the earliest
                       date and time that you expected to have a job run in the queue. Used to narrow the window of jobs
                       to request information on.
        end_time, str, datetime string in the format understood by NERSC Slurm scheduler. This should defined the latest
                       date and time that you expected to have a job run in the queue. Used to narrow the window of jobs
                       to request information on.
        ptab_name, str, the full pathname where the processing table should be saved.
        dry_run, bool, whether this is a simulated run or not. If True, jobs are not actually submitted but relevant
                       information is printed to help with testing.
        reservation: str. The reservation to submit jobs to. If None, it is not submitted to a reservation.
    Returns:
        proc_table: Table, a table with the same rows as the input except that Slurm and jobid relevant columns have
                           been updated for those jobs that needed to be resubmitted.
        submits: int, the number of submissions made to the queue. This is incremented from the input submits, so it is
                      the number of submissions made from this function call plus the input submits value.

    Note:
        This modifies the inputs of both proc_table and submits and returns them.
    """
    if resubmission_states is None:
        resubmission_states = get_resubmission_states()
    proc_table = update_from_queue(proc_table, start_time=start_time, end_time=end_time)
    id_to_row_map = {row['INTID']: rown for rown, row in enumerate(proc_table)}
    for rown in range(len(proc_table)):
        if proc_table['STATUS'][rown] in resubmission_states:
            proc_table, submits = recursive_submit_failed(rown, proc_table, submits, id_to_row_map, ptab_name,
                                                          resubmission_states, reservation, dry_run)
    return proc_table, submits

def recursive_submit_failed(rown, proc_table, submits, id_to_row_map, ptab_name=None,
                            resubmission_states=None, reservation=None, dry_run=False):
    """
    Given a row of a processing table and the full processing table, this resubmits the given job.
    Before submitting a job, it checks the dependencies for failures in the processing table. If a dependency needs to
    be resubmitted, it recursively follows dependencies until it finds the first job without a failed dependency and
    resubmits that. Then resubmits the other jobs with the new Slurm jobID's for proper dependency coordination within Slurm.

    Args:
        rown, Table.Row, the row of the processing table that you want to resubmit.
        proc_table, Table, the processing table with a row per job.
        submits, int, the number of submissions made to the queue. Used for saving files and in not overloading the scheduler.
        id_to_row_map, dict, lookup dictionary where the keys are internal ids (INTID's) and the values are the row position
                             in the processing table.
        ptab_name, str, the full pathname where the processing table should be saved.
        resubmission_states, list or array of strings, each element should be a capitalized string corresponding to a
                                                       possible Slurm scheduler state, where you wish for jobs with that
                                                       outcome to be resubmitted
        reservation: str. The reservation to submit jobs to. If None, it is not submitted to a reservation.
        dry_run, bool, whether this is a simulated run or not. If True, jobs are not actually submitted but relevant
                       information is printed to help with testing.
    Returns:
        proc_table: Table, a table with the same rows as the input except that Slurm and jobid relevant columns have
                           been updated for those jobs that needed to be resubmitted.
        submits: int, the number of submissions made to the queue. This is incremented from the input submits, so it is
                      the number of submissions made from this function call plus the input submits value.

    Note:
        This modifies the inputs of both proc_table and submits and returns them.
    """
    log = get_logger()
    if resubmission_states is None:
        resubmission_states = get_resubmission_states()
    ideps = proc_table['INT_DEP_IDS'][rown]
    if ideps is None:
        proc_table['LATEST_DEP_QID'][rown] = np.ndarray(shape=0).astype(int)
    else:
        qdeps = []
        for idep in np.sort(np.atleast_1d(ideps)):
            if proc_table['STATUS'][id_to_row_map[idep]] in resubmission_states:
                proc_table, submits = recursive_submit_failed(id_to_row_map[idep], proc_table, submits,
                                                              id_to_row_map, reservation=reservation, dry_run=dry_run)
            qdeps.append(proc_table['LATEST_QID'][id_to_row_map[idep]])

        qdeps = np.atleast_1d(qdeps)
        if len(qdeps) > 0:
            proc_table['LATEST_DEP_QID'][rown] = qdeps
        else:
            log.error(f"number of qdeps should be 1 or more: Rown {rown}, ideps {ideps}")

    proc_table[rown] = submit_batch_script(proc_table[rown], reservation=reservation, dry_run=dry_run)
    submits += 1

    if not dry_run:
        time.sleep(2)
        if submits % 10 == 0:
            if ptab_name is None:
                write_table(proc_table, tabletype='processing', overwrite=True)
            else:
                write_table(proc_table, tablename=ptab_name, overwrite=True)
            time.sleep(60)
        if submits % 100 == 0:
            time.sleep(540)
            proc_table = update_from_queue(proc_table)
            if ptab_name is None:
                write_table(proc_table, tabletype='processing', overwrite=True)
            else:
                write_table(proc_table, tablename=ptab_name, overwrite=True)

    return proc_table, submits


#########################################
########     Joint fit     ##############
#########################################
def joint_fit(ptable, prows, internal_id, queue, reservation, descriptor,
              dry_run=False, strictly_successful=False, check_for_outputs=True, resubmit_partial_complete=True):
    """
    Given a set of prows, this generates a processing table row, creates a batch script, and submits the appropriate
    joint fitting job given by descriptor. If the joint fitting job is standard star fitting, the post standard star fits
    for all the individual exposures also created and submitted. The returned ptable has all of these rows added to the
    table given as input.

    Args:
        ptable, Table. The processing table where each row is a processed job.
        prows, list or array of dicts. The rows corresponding to the individual exposure jobs that are
                                                     inputs to the joint fit.
        internal_id, int, the next internal id to be used for assignment (already incremented up from the last used id number used).
        queue, str. The name of the queue to submit the jobs to. If None is given the current desi_proc default is used.
        reservation: str. The reservation to submit jobs to. If None, it is not submitted to a reservation.
        descriptor, str. Description of the joint fitting job. Can either be 'science' or 'stdstarfit', 'arc' or 'psfnight',
                         or 'flat' or 'nightlyflat'.
        dry_run, bool, whether this is a simulated run or not. If True, jobs are not actually submitted but relevant
                       information is printed to help with testing.
        strictly_successful, bool. Whether all jobs require all inputs to have succeeded. For daily processing, this is
                                   less desirable because e.g. the sciences can run with SVN default calibrations rather
                                   than failing completely from failed calibrations. Default is False.
        check_for_outputs, bool. Default is True. If True, the code checks for the existence of the expected final
                                 data products for the script being submitted. If all files exist and this is True,
                                 then the script will not be submitted. If some files exist and this is True, only the
                                 subset of the cameras without the final data products will be generated and submitted.
        resubmit_partial_complete, bool. Default is True. Must be used with check_for_outputs=True. If this flag is True,
                                         jobs with some prior data are pruned using PROCCAMWORD to only process the
                                         remaining cameras not found to exist.

    Returns:
        ptable, Table. The same processing table as input except with added rows for the joint fit job and, in the case
                       of a stdstarfit, the poststdstar science exposure jobs.
        joint_prow, dict. Row of a processing table corresponding to the joint fit job.
        internal_id, int, the next internal id to be used for assignment (already incremented up from the last used id number used).
    """
    log = get_logger()
    if len(prows) < 1:
        return ptable, None, internal_id

    if descriptor is None:
        return ptable, None
    elif descriptor == 'science':
        descriptor = 'stdstarfit'
    elif descriptor == 'arc':
        descriptor = 'psfnight'
    elif descriptor == 'flat':
        descriptor = 'nightlyflat'

    if descriptor not in ['stdstarfit', 'psfnight', 'nightlyflat']:
        return ptable, None, internal_id

    log.info(" ")
    log.info(f"Joint fit criteria found. Running {descriptor}.\n")

    joint_prow = make_joint_prow(prows, descriptor=descriptor, internal_id=internal_id)
    internal_id += 1
    joint_prow = create_and_submit(joint_prow, queue=queue, reservation=reservation, joint=True, dry_run=dry_run,
                                   strictly_successful=strictly_successful, check_for_outputs=check_for_outputs,
                                   resubmit_partial_complete=resubmit_partial_complete)
    ptable.add_row(joint_prow)

    if descriptor == 'stdstarfit':
        log.info(" ")
        log.info(f"Submitting individual science exposures now that joint fitting of standard stars is submitted.\n")
        for row in prows:
            if row['LASTSTEP'] == 'stdstarfit':
                continue
            ## in dry_run, mock Slurm ID's are generated using CPU seconds. Wait one second so we have unique ID's
            if dry_run:
                time.sleep(1)
            row['JOBDESC'] = 'poststdstar'
            row['INTID'] = internal_id
            internal_id += 1
            row['ALL_QIDS'] = np.ndarray(shape=0).astype(int)
            row = assign_dependency(row, joint_prow)
            row = create_and_submit(row, queue=queue, reservation=reservation, dry_run=dry_run,
                                    strictly_successful=strictly_successful)
            ptable.add_row(row)
    else:
        ## in dry_run, mock Slurm ID's are generated using CPU seconds. Wait one second so we have unique ID's
        if dry_run:
            time.sleep(1)
        log.info(f"Setting the calibration exposures as calibrators in the processing table.\n")
        ptable = set_calibrator_flag(prows, ptable)

    return ptable, joint_prow, internal_id


## wrapper functions for joint fitting
def science_joint_fit(ptable, sciences, internal_id, queue='realtime',
                      reservation=None, dry_run=False, strictly_successful=False,
                      check_for_outputs=True, resubmit_partial_complete=True):
    """
    Wrapper function for desiproc.workflow.procfuns.joint_fit specific to the stdstarfit joint fit.

    All variables are the same except:
        Arg 'sciences' is mapped to the prows argument of joint_fit.
        The joint_fit argument descriptor is pre-defined as 'stdstarfit'.
    """
    return joint_fit(ptable=ptable, prows=sciences, internal_id=internal_id, queue=queue, reservation=reservation,
                     descriptor='stdstarfit', dry_run=dry_run, strictly_successful=strictly_successful,
                     check_for_outputs=check_for_outputs, resubmit_partial_complete=resubmit_partial_complete)


def flat_joint_fit(ptable, flats, internal_id, queue='realtime',
                   reservation=None, dry_run=False, strictly_successful=False,
                   check_for_outputs=True, resubmit_partial_complete=True):
    """
    Wrapper function for desiproc.workflow.procfuns.joint_fit specific to the nightlyflat joint fit.

    All variables are the same except:
        Arg 'flats' is mapped to the prows argument of joint_fit.
        The joint_fit argument descriptor is pre-defined as 'nightlyflat'.
    """
    return joint_fit(ptable=ptable, prows=flats, internal_id=internal_id, queue=queue, reservation=reservation,
                     descriptor='nightlyflat', dry_run=dry_run, strictly_successful=strictly_successful,
                     check_for_outputs=check_for_outputs, resubmit_partial_complete=resubmit_partial_complete)


def arc_joint_fit(ptable, arcs, internal_id, queue='realtime',
                  reservation=None, dry_run=False, strictly_successful=False,
                  check_for_outputs=True, resubmit_partial_complete=True):
    """
    Wrapper function for desiproc.workflow.procfuns.joint_fit specific to the psfnight joint fit.

    All variables are the same except:
        Arg 'arcs' is mapped to the prows argument of joint_fit.
        The joint_fit argument descriptor is pre-defined as 'psfnight'.
    """
    return joint_fit(ptable=ptable, prows=arcs, internal_id=internal_id, queue=queue, reservation=reservation,
                     descriptor='psfnight', dry_run=dry_run, strictly_successful=strictly_successful,
                     check_for_outputs=check_for_outputs, resubmit_partial_complete=resubmit_partial_complete)


def make_joint_prow(prows, descriptor, internal_id):
    """
    Given an input list or array of processing table rows and a descriptor, this creates a joint fit processing job row.
    It starts by copying the first input row, overwrites relevant columns, and defines the new dependencies (based on the
    input prows).

    Args:
        prows, list or array of dicts. The rows corresponding to the individual exposure jobs that are
                                                     inputs to the joint fit.
        descriptor, str. Description of the joint fitting job. Can either be 'stdstarfit', 'psfnight', or 'nightlyflat'.
        internal_id, int, the next internal id to be used for assignment (already incremented up from the last used id number used).

    Returns:
        joint_prow, dict. Row of a processing table corresponding to the joint fit job.
    """
    first_row = prows[0]
    joint_prow = first_row.copy()

    joint_prow['INTID'] = internal_id
    joint_prow['JOBDESC'] = descriptor
    joint_prow['ALL_QIDS'] = np.ndarray(shape=0).astype(int)
    joint_prow['EXPID'] = np.array([ currow['EXPID'][0] for currow in prows ], dtype=int)
    joint_prow = assign_dependency(joint_prow,dependency=prows)
    return joint_prow

def checkfor_and_submit_joint_job(ptable, arcs, flats, sciences, arcjob, flatjob,
                                  lasttype, internal_id, dry_run=False,
                                  queue='realtime', reservation=None, strictly_successful=False,
                                  check_for_outputs=True, resubmit_partial_complete=True):
    """
    Takes all the state-ful data from daily processing and determines whether a joint fit needs to be submitted. Places
    the decision criteria into a single function for easier maintainability over time. These are separate from the
    new standard manifest*.json method of indicating a calibration sequence is complete. That is checked independently
    elsewhere and doesn't interact with this.

    Args:
        ptable, Table, Processing table of all exposures that have been processed.
        arcs, list of dicts, list of the individual arc jobs to be used for the psfnight (NOT all
                                   the arcs, if multiple sets existed). May be empty if none identified yet.
        flats, list of dicts, list of the individual flat jobs to be used for the nightlyflat (NOT
                                    all the flats, if multiple sets existed). May be empty if none identified yet.
        sciences, list of dicts, list of the most recent individual prestdstar science exposures
                                       (if currently processing that tile). May be empty if none identified yet.
        arcjob, dict or None, the psfnight job row if it exists. Otherwise None.
        flatjob, dict or None, the nightlyflat job row if it exists. Otherwise None.
        lasttype, str or None, the obstype of the last individual exposure row to be processed.
        internal_id, int, an internal identifier unique to each job. Increments with each new job. This
                          is the smallest unassigned value.
        dry_run, bool, whether this is a simulated run or not. If True, jobs are not actually submitted but relevant
                       information is printed to help with testing.
        queue, str. The name of the queue to submit the jobs to. If None is given the current desi_proc default is used.
        reservation: str. The reservation to submit jobs to. If None, it is not submitted to a reservation.
        strictly_successful, bool. Whether all jobs require all inputs to have succeeded. For daily processing, this is
                                   less desirable because e.g. the sciences can run with SVN default calibrations rather
                                   than failing completely from failed calibrations. Default is False.
        check_for_outputs, bool. Default is True. If True, the code checks for the existence of the expected final
                                 data products for the script being submitted. If all files exist and this is True,
                                 then the script will not be submitted. If some files exist and this is True, only the
                                 subset of the cameras without the final data products will be generated and submitted.
        resubmit_partial_complete, bool. Default is True. Must be used with check_for_outputs=True. If this flag is True,
                                         jobs with some prior data are pruned using PROCCAMWORD to only process the
                                         remaining cameras not found to exist.
    Returns:
        ptable, Table, Processing table of all exposures that have been processed.
        arcjob, dictor None, the psfnight job row if it exists. Otherwise None.
        flatjob, dict or None, the nightlyflat job row if it exists. Otherwise None.
        sciences, list of dicts, list of the most recent individual prestdstar science exposures
                                       (if currently processing that tile). May be empty if none identified yet or
                                       we just submitted them for processing.
        internal_id, int, if no job is submitted, this is the same as the input, otherwise it is incremented upward from
                          from the input such that it represents the smallest unused ID.
    """
    if lasttype == 'science' and len(sciences) > 0:
        log = get_logger()
        skysubonly = np.array([sci['LASTSTEP'] == 'skysub' for sci in sciences])
        if np.all(skysubonly):
            log.error("Identified all exposures in joint fitting request as skysub-only. Not submitting")
            sciences = []
            return ptable, arcjob, flatjob, sciences, internal_id

        if np.any(skysubonly):
            log.error("Identified skysub-only exposures in joint fitting request")
            log.info("Expid's: {}".format([row['EXPID'] for row in sciences]))
            log.info("LASTSTEP's: {}".format([row['LASTSTEP'] for row in sciences]))
            sciences = (np.array(sciences,dtype=object)[~skysubonly]).tolist()
            log.info("Removed skysub only exposures in joint fitting:")
            log.info("Expid's: {}".format([row['EXPID'] for row in sciences]))
            log.info("LASTSTEP's: {}".format([row['LASTSTEP'] for row in sciences]))

        from collections import Counter
        tiles = np.array([sci['TILEID'] for sci in sciences])
        counts = Counter(tiles)
        if len(counts.most_common()) > 1:
            log.error("Identified more than one tile in a joint fitting request")
            log.info("Expid's: {}".format([row['EXPID'] for row in sciences]))
            log.info("Tileid's: {}".format(tiles))
            log.info("Returning without joint fitting any of these exposures.")
            # most_common, nmost_common = counts.most_common()[0]
            # if most_common == -99:
            #     most_common, nmost_common = counts.most_common()[1]
            # log.warning(f"Given multiple tiles to jointly fit: {counts}. "+
            #             "Only processing the most common non-default " +
            #             f"tile: {most_common} with {nmost_common} exposures")
            # sciences = (np.array(sciences,dtype=object)[tiles == most_common]).tolist()
            # log.info("Tiles and exposure id's being submitted for joint fitting:")
            # log.info("Expid's: {}".format([row['EXPID'] for row in sciences]))
            # log.info("Tileid's: {}".format([row['TILEID'] for row in sciences]))
            sciences = []
            return ptable, arcjob, flatjob, sciences, internal_id

        ptable, tilejob, internal_id = science_joint_fit(ptable, sciences, internal_id, dry_run=dry_run, queue=queue,
                                                         reservation=reservation, strictly_successful=strictly_successful,
                                                         check_for_outputs=check_for_outputs,
                                                         resubmit_partial_complete=resubmit_partial_complete)
        if tilejob is not None:
            sciences = []

    elif lasttype == 'flat' and flatjob is None and len(flats)>11:
        ## Note here we have an assumption about the number of expected flats being greater than 11
        ptable, flatjob, internal_id = flat_joint_fit(ptable, flats, internal_id, dry_run=dry_run, queue=queue,
                                                      reservation=reservation, strictly_successful=strictly_successful,
                                                      check_for_outputs=check_for_outputs,
                                                      resubmit_partial_complete=resubmit_partial_complete
                                                      )

    elif lasttype == 'arc' and arcjob is None and len(arcs) > 4:
        ## Note here we have an assumption about the number of expected arcs being greater than 4
        ptable, arcjob, internal_id = arc_joint_fit(ptable, arcs, internal_id, dry_run=dry_run, queue=queue,
                                                    reservation=reservation, strictly_successful=strictly_successful,
                                                    check_for_outputs=check_for_outputs,
                                                    resubmit_partial_complete=resubmit_partial_complete
                                                    )
    return ptable, arcjob, flatjob, sciences, internal_id


def set_calibrator_flag(prows, ptable):
    """
    Sets the "CALIBRATOR" column of a procesing table row to 1 (integer representation of True)
     for all input rows. Used within joint fitting code to flag the exposures that were input
     to the psfnight or nightlyflat for later reference.

    Args:
        prows, list or array of Table.Rows or dicts. The rows corresponding to the individual exposure jobs that are
                                                     inputs to the joint fit.
        ptable, Table. The processing table where each row is a processed job.

    Returns:
        ptable, Table. The same processing table as input except with added rows for the joint fit job and, in the case
                       of a stdstarfit, the poststdstar science exposure jobs.
    """
    for prow in prows:
        ptable['CALIBRATOR'][ptable['INTID'] == prow['INTID']] = 1
    return ptable

