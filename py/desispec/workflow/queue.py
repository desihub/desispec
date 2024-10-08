"""
desispec.workflow.queue
=======================

"""
import os
import re
import numpy as np
from astropy.table import Table, vstack
import subprocess

from desispec.workflow.proctable import get_default_qid
from desiutil.log import get_logger
import time, datetime

global _cached_slurm_states
_cached_slurm_states = dict()

def get_resubmission_states(no_resub_failed=False):
    """
    Defines what Slurm job failure modes should be resubmitted in the hopes of the job succeeding the next time.

    Possible values that Slurm returns are::

        CA or ca or CANCELLED for cancelled jobs will only show currently running jobs in queue unless times are explicitly given
        BF BOOT_FAIL   Job terminated due to launch failure
        CA CANCELLED Job was explicitly cancelled by the user or system administrator. The job may or may not have been initiated.
        CD COMPLETED Job has terminated all processes on all nodes with an exit code of zero.
        DL DEADLINE Job terminated on deadline.
        F FAILED Job terminated with non-zero exit code or other failure condition.
        NF NODE_FAIL Job terminated due to failure of one or more allocated nodes.
        OOM OUT_OF_MEMORY Job experienced out of memory error.
        PD PENDING Job is awaiting resource allocation.
        PR PREEMPTED Job terminated due to preemption.
        R RUNNING Job currently has an allocation.
        RQ REQUEUED Job was requeued.
        RS RESIZING Job is about to change size.
        RV REVOKED Sibling was removed from cluster due to other cluster starting the job.
        S SUSPENDED Job has an allocation, but execution has been suspended and CPUs have been released for other jobs.
        TO TIMEOUT Job terminated upon reaching its time limit.

    Args:
        no_resub_failed: bool. Set to True if you do NOT want to resubmit
            jobs with Slurm status 'FAILED' by default. Default is False.

    Returns:
        list. A list of strings outlining the job states that should be resubmitted.
    """
    ## 'UNSUBMITTED' is default pipeline state for things not yet submitted
    ## 'DEP_NOT_SUBD' is set when resubmission can't proceed because a
    ## dependency has failed
    resub_states = ['UNSUBMITTED', 'DEP_NOT_SUBD', 'BOOT_FAIL', 'DEADLINE', 'NODE_FAIL',
                    'OUT_OF_MEMORY', 'PREEMPTED', 'TIMEOUT', 'CANCELLED']
    if not no_resub_failed:
        resub_states.append('FAILED')
    return resub_states


def get_termination_states():
    """
    Defines what Slurm job states that are final and aren't in question about needing resubmission.

    Possible values that Slurm returns are::

        CA or ca or CANCELLED for cancelled jobs will only show currently running jobs in queue unless times are explicitly given
        BF BOOT_FAIL   Job terminated due to launch failure
        CA CANCELLED Job was explicitly cancelled by the user or system administrator. The job may or may not have been initiated.
        CD COMPLETED Job has terminated all processes on all nodes with an exit code of zero.
        DL DEADLINE Job terminated on deadline.
        F FAILED Job terminated with non-zero exit code or other failure condition.
        NF NODE_FAIL Job terminated due to failure of one or more allocated nodes.
        OOM OUT_OF_MEMORY Job experienced out of memory error.
        PD PENDING Job is awaiting resource allocation.
        PR PREEMPTED Job terminated due to preemption.
        R RUNNING Job currently has an allocation.
        RQ REQUEUED Job was requeued.
        RS RESIZING Job is about to change size.
        RV REVOKED Sibling was removed from cluster due to other cluster starting the job.
        S SUSPENDED Job has an allocation, but execution has been suspended and CPUs have been released for other jobs.
        TO TIMEOUT Job terminated upon reaching its time limit.

    Returns:
        list. A list of strings outlining the job states that are considered final (without human investigation/intervention)
    """
    return ['COMPLETED', 'CANCELLED', 'FAILED']

def get_failed_states():
    """ 
    Defines what Slurm job states should be considered failed or problematic

    All possible values that Slurm returns are:
        BF BOOT_FAIL Job terminated due to launch failure, typically due to a hardware failure (e.g. unable to boot the node or block and the job can not be requeued).
        CA CANCELLED Job was explicitly cancelled by the user or system administrator. The job may or may not have been initiated.
        CD COMPLETED Job has terminated all processes on all nodes with an exit code of zero.
        CF CONFIGURING Job has been allocated resources, but are waiting for them to become ready for use (e.g. booting).
        CG COMPLETING Job is in the process of completing. Some processes on some nodes may still be active.
        DL DEADLINE Job terminated on deadline.
        F FAILED Job terminated with non-zero exit code or other failure condition.
        NF NODE_FAIL Job terminated due to failure of one or more allocated nodes.
        OOM OUT_OF_MEMORY Job experienced out of memory error.
        PD PENDING Job is awaiting resource allocation.
        PR PREEMPTED Job terminated due to preemption.
        R RUNNING Job currently has an allocation.
        RD RESV_DEL_HOLD Job is being held after requested reservation was deleted.
        RF REQUEUE_FED Job is being requeued by a federation.
        RH REQUEUE_HOLD Held job is being requeued.
        RQ REQUEUED Completing job is being requeued.
        RS RESIZING Job is about to change size.
        RV REVOKED Sibling was removed from cluster due to other cluster starting the job.
        SI SIGNALING Job is being signaled.
        SE SPECIAL_EXIT The job was requeued in a special state. This state can be set by users, typically in EpilogSlurmctld, if the job has terminated with a particular exit value.
        SO STAGE_OUT Job is staging out files.
        ST STOPPED Job has an allocation, but execution has been stopped with SIGSTOP signal. CPUS have been retained by this job.
        S SUSPENDED Job has an allocation, but execution has been suspended and CPUs have been released for other jobs.
        TO TIMEOUT Job terminated upon reaching its time limit.
    
    Returns:
        list. A list of strings outlining the job states that are considered to be
            failed or problematic.
    """
    return ['BOOT_FAIL', 'CANCELLED', 'DEADLINE', 'FAILED', 'NODE_FAIL',
            'OUT_OF_MEMORY', 'PREEMPTED', 'REVOKED', 'SUSPENDED', 'TIMEOUT']


def get_non_final_states():
    """
    Defines what Slurm job states that are not final and therefore indicate the
    job hasn't finished running.

    Possible values that Slurm returns are:

        CA or ca or CANCELLED for cancelled jobs will only show currently running jobs in queue unless times are explicitly given
        BF BOOT_FAIL   Job terminated due to launch failure
        CA CANCELLED Job was explicitly cancelled by the user or system administrator. The job may or may not have been initiated.
        CD COMPLETED Job has terminated all processes on all nodes with an exit code of zero.
        DL DEADLINE Job terminated on deadline.
        F FAILED Job terminated with non-zero exit code or other failure condition.
        NF NODE_FAIL Job terminated due to failure of one or more allocated nodes.
        OOM OUT_OF_MEMORY Job experienced out of memory error.
        PD PENDING Job is awaiting resource allocation.
        PR PREEMPTED Job terminated due to preemption.
        R RUNNING Job currently has an allocation.
        RQ REQUEUED Job was requeued.
        RS RESIZING Job is about to change size.
        RV REVOKED Sibling was removed from cluster due to other cluster starting the job.
        S SUSPENDED Job has an allocation, but execution has been suspended and CPUs have been released for other jobs.
        TO TIMEOUT Job terminated upon reaching its time limit.

    Returns:
        list. A list of strings outlining the job states that are considered final (without human investigation/intervention)
    """
    return ['PENDING', 'RUNNING', 'REQUEUED', 'RESIZING']

def get_mock_slurm_data():
    """
    Returns a string of output that mimics what Slurm would return from
    sacct -X --parsable2 --delimiter=, \
    --format=JobID,JobName,Partition,Submit,Eligible,Start,End,Elapsed,State,ExitCode -j <qid_str>

    Returns
    -------
    str
        Mock Slurm data csv format.
    """
    string = 'JobID,JobName,Partition,Submit,Eligible,Start,End,Elapsed,State,ExitCode\n'
    string += '49482394,arc-20211102-00107062-a0123456789,realtime,2021-11-02' \
              + 'T18:31:14,2021-11-02T18:36:33,2021-11-02T18:36:33,2021-11-02T' \
              + '18:48:32,00:11:59,COMPLETED,0:0' + '\n'
    string += '49482395,arc-20211102-00107063-a0123456789,realtime,2021-11-02' \
              + 'T18:31:16,2021-11-02T18:36:33,2021-11-02T18:48:34,2021-11-02T' \
              + '18:57:02,00:11:59,COMPLETED,0:0' + '\n'
    string += '49482397,arc-20211102-00107064-a0123456789,realtime,2021-11-02' \
              + 'T18:31:19,2021-11-02T18:36:33,2021-11-02T18:57:05,2021-11-02T' \
              + '19:06:17,00:11:59,COMPLETED,0:0' + '\n'
    string += '49482398,arc-20211102-00107065-a0123456789,realtime,2021-11-02' \
              + 'T18:31:24,2021-11-02T18:36:33,2021-11-02T19:06:18,2021-11-02T' \
              + '19:13:59,00:11:59,COMPLETED,0:0' + '\n'
    string += '49482399,arc-20211102-00107066-a0123456789,realtime,2021-11-02' \
              + 'T18:31:27,2021-11-02T18:36:33,2021-11-02T19:14:00,2021-11-02T' \
              + '19:24:49,00:11:59,COMPLETED,0:0'
    return string


def queue_info_from_time_window(start_time=None, end_time=None, user=None, \
                             columns='jobid,jobname,partition,submit,eligible,'+
                                     'start,end,elapsed,state,exitcode',
                             dry_run_level=0):
    """
    Queries the NERSC Slurm database using sacct with appropriate flags to get information within a specified time
    window of all jobs submitted or executed during that time.

    Parameters
    ----------
    start_time : str
        String of the form YYYY-mm-ddTHH:MM:SS. Based on the given night and the earliest hour you
        want to see queue information about.
    end_time : str
        String of the form YYYY-mm-ddTHH:MM:SS. Based on the given night and the latest hour you
        want to see queue information about.
    user : str
        The username at NERSC that you want job information about. The default is an the environment name if
        if exists, otherwise 'desi'.
    columns : str
        Comma seperated string of valid sacct column names, in lower case. To be useful for the workflow,
        it should have MUST have columns "JOBID" and "STATE". Other columns available that aren't included
        in the default list are: jobid,jobname,partition,submit,eligible,start,end,elapsed,state,exitcode.
        Other options include: suspended,derivedexitcode,reason,priority,jobname.
    dry_run_level : int
        If nonzero, this is a simulated run. Default is 0.
        0 which runs the code normally.
        1 writes all files but doesn't submit any jobs to Slurm.
        2 writes tables but doesn't write scripts or submit anything.
        3 Doesn't write or submit anything but queries Slurm normally for job status.
        4 Doesn't write, submit jobs, or query Slurm.
        5 Doesn't write, submit jobs, or query Slurm; instead it makes up the status of the jobs.

    Returns
    -------
    astropy.table.Table
        Table with the columns defined by the input variable 'columns' and information relating
        to all jobs submitted by the specified user in the specified time frame.
    """
    # global queue_info_table
    if dry_run_level > 4:
        string = get_mock_slurm_data()
        cmd_as_list = ['echo', string]
    elif dry_run_level > 3:
        cmd_as_list = ['echo', 'JobID,JobName,Partition,Submit,Eligible,Start,End,Elapsed,State,ExitCode']
    else:
        if user is None:
            if 'USER' in os.environ:
                user = os.environ['USER']
            else:
                user = 'desi'
        if start_time is None:
            start_time = '2020-04-26T00:00'
        if end_time is None:
            end_time = '2020-05-01T00:00'
        cmd_as_list = ['sacct', '-X', '--parsable2', '--delimiter=,', \
                       '-S', start_time, \
                       '-E', end_time, \
                       '-u', user, \
                       f'--format={columns}']

    table_as_string = subprocess.check_output(cmd_as_list, text=True,
                                              stderr=subprocess.STDOUT)
    queue_info_table = Table.read(table_as_string, format='ascii.csv')

    for col in queue_info_table.colnames:
        queue_info_table.rename_column(col, col.upper())

    ## Update the cached states of these jobids if we have that info to update
    update_queue_state_cache_from_table(queue_info_table)

    return queue_info_table

def queue_info_from_qids(qids, columns='jobid,jobname,partition,submit,'+
                         'eligible,start,end,elapsed,state,exitcode', dry_run_level=0):
    """
    Queries the NERSC Slurm database using sacct with appropriate flags to get
    information about specific jobs based on their jobids.

    Parameters
    ----------
    jobids : list or array of ints
        Slurm QID's at NERSC that you want to return information about.
    columns : str
        Comma seperated string of valid sacct column names, in lower case. To be useful for the workflow,
        it should have MUST have columns "JOBID" and "STATE". Other columns available that aren't included
        in the default list are: jobid,jobname,partition,submit,eligible,start,end,elapsed,state,exitcode.
        Other options include: suspended,derivedexitcode,reason,priority,jobname.
    dry_run_level : int
        If nonzero, this is a simulated run. Default is 0.
        0 which runs the code normally.
        1 writes all files but doesn't submit any jobs to Slurm.
        2 writes tables but doesn't write scripts or submit anything.
        3 Doesn't write or submit anything but queries Slurm normally for job status.
        4 Doesn't write, submit jobs, or query Slurm.
        5 Doesn't write, submit jobs, or query Slurm; instead it makes up the status of the jobs.

    Returns
    -------
    astropy.table.Table
        Table with the columns defined by the input variable 'columns' and information relating
        to all jobs submitted by the specified user in the specified time frame.
    """
    qids = np.atleast_1d(qids).astype(int)
    log = get_logger()

    ## If qids is too long, recursively call self and stack tables; otherwise sacct hangs
    nmax = 100
    if len(qids) > nmax:
        results = list()
        for i in range(0, len(qids), nmax):
            results.append(queue_info_from_qids(qids[i:i+nmax], columns=columns,
                                                dry_run_level=dry_run_level))
        results = vstack(results)
        return results
    elif len(qids) == 0:
        return Table(names=columns.upper().split(','))

    ## Turn the queue id's into a list
    ## this should work with str or int type also, though not officially supported
    qid_str = ','.join(np.atleast_1d(qids).astype(str)).replace(' ','')

    cmd_as_list = ['sacct', '-X', '--parsable2', '--delimiter=,',
                   f'--format={columns}', '-j', qid_str]
    if dry_run_level > 4:
        log.info("Dry run, would have otherwise queried Slurm with the"
                 +f" following: {' '.join(cmd_as_list)}")
        ### Set a random 5% of jobs as TIMEOUT, set seed for reproducibility
        # np.random.seed(qids[0])
        states = np.array(['COMPLETED'] * len(qids))
        #states[np.random.random(len(qids)) < 0.05] = 'TIMEOUT'
        ## Try two different column configurations, otherwise give up trying to simulate
        string = 'JobID,JobName,Partition,Submit,Eligible,Start,End,Elapsed,State,ExitCode'
        if columns.lower() == string.lower():
            for jobid, expid, state in zip(qids, 100000+np.arange(len(qids)), states):
                string += f'\n{jobid},arc-20211102-{expid:08d}-a0123456789,realtime,2021-11-02'\
                      +'T18:31:14,2021-11-02T18:36:33,2021-11-02T18:36:33,2021-11-02T'\
                      +f'18:48:32,00:11:59,{state},0:0'
        elif columns.lower() == 'jobid,state':
            string = 'JobID,State'
            for jobid, state in zip(qids, states):
                string += f'\n{jobid},{state}'
        # create command to run to exercise subprocess -> stdout parsing
        cmd_as_list = ['echo', string]
    elif dry_run_level > 3:
        cmd_as_list = ['echo', columns.lower()]
    else:
        log.info(f"Querying Slurm with the following: {' '.join(cmd_as_list)}")

    #- sacct sometimes fails; try several times before giving up
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            table_as_string = subprocess.check_output(cmd_as_list, text=True,
                                          stderr=subprocess.STDOUT)
            break
        except subprocess.CalledProcessError as err:
            log.error(f'{qid_str} job query via sacct failure at {datetime.datetime.now()}')
            log.error(f'{qid_str} {cmd_as_list}')
            log.error(f'{qid_str} {err.output=}')
    else:  #- for/else happens if loop doesn't succeed
        msg = f'{qid_str} job query via sacct failed {max_attempts} times; exiting'
        log.critical(msg)
        raise RuntimeError(msg)

    queue_info_table = Table.read(table_as_string, format='ascii.csv')
    for col in queue_info_table.colnames:
        queue_info_table.rename_column(col, col.upper())

    ## Update the cached states of these jobids if we have that info to update
    update_queue_state_cache_from_table(queue_info_table)

    return queue_info_table

def get_queue_states_from_qids(qids, dry_run_level=0, use_cache=False):
    """
    Queries the NERSC Slurm database using sacct with appropriate flags to get
    information on the job STATE. If use_cache is set and all qids have cached
    values from a previous query, those cached states will be returned instead.

    Parameters
    ----------
    jobids : list or array of ints
        Slurm QID's at NERSC that you want to return information about.
    dry_run_level : int
        If nonzero, this is a simulated run. Default is 0.
        0 which runs the code normally.
        1 writes all files but doesn't submit any jobs to Slurm.
        2 writes tables but doesn't write scripts or submit anything.
        3 Doesn't write or submit anything but queries Slurm normally for job status.
        4 Doesn't write, submit jobs, or query Slurm.
        5 Doesn't write, submit jobs, or query Slurm; instead it makes up the status of the jobs.
    use_cache : bool
        If True the code first looks for a cached status
        for the qid. If unavailable, then it queries Slurm. Default is False.

    Returns
    -------
    Dict
        Dictionary with the keys as jobids and values as the slurm state of the job.
    """
    def_qid = get_default_qid()
    global _cached_slurm_states
    qids = np.atleast_1d(qids).astype(int)
    log = get_logger()

    ## Only use cached values if all are cahced, since the time is dominated
    ## by the call itself rather than the number of jobids, so we may as well
    ## get updated information from all of them if we're submitting a query anyway
    outdict = dict()
    if use_cache and np.all(np.isin(qids, list(_cached_slurm_states.keys()))):
        log.info(f"All Slurm {qids=} are cached. Using cached values.")
        for qid in qids:
            outdict[qid] = _cached_slurm_states[qid]
    else:
        outtable = queue_info_from_qids(qids, columns='jobid,state',
                                        dry_run_level=dry_run_level)
        for row in outtable:
            if int(row['JOBID']) != def_qid:
                outdict[int(row['JOBID'])] = row['STATE']
    return outdict

def update_queue_state_cache_from_table(queue_info_table):
    """
    Takes a Slurm jobid and updates the queue id cache with the supplied state

    Parameters
    ----------
    queue_info_table : astropy.table.Table
        Table returned by an sacct query. Should contain at least JOBID and STATE
        columns

    Returns
    -------
    Nothing

    """
    ## Update the cached states of these jobids if we have that info to update
    if 'JOBID' in queue_info_table.colnames and 'STATE' in queue_info_table.colnames:
        for row in queue_info_table:
            update_queue_state_cache(qid=row['JOBID'], state=row['STATE'])

def update_queue_state_cache(qid, state):
    """
    Takes a Slurm jobid and updates the queue id cache with the supplied state

    Parameters
    ----------
    qid : int
        Slurm QID at NERSC
    state: str
        The current job status of the Slurm jobid

    Returns
    -------
    Nothing

    """
    global _cached_slurm_states
    if int(qid) != get_default_qid():
        _cached_slurm_states[int(qid)] = state

def clear_queue_state_cache():
    """
    Remove all entries from the queue state cache
    """
    global _cached_slurm_states
    _cached_slurm_states.clear()


def update_from_queue(ptable, qtable=None, dry_run_level=0, ignore_scriptnames=False,
                      check_complete_jobs=False):
    """
    Given an input prcessing table (ptable) and query table from the Slurm queue (qtable) it cross matches the
    Slurm job ID's and updates the 'state' in the table using the current state in the Slurm scheduler system.

    Parameters
    ----------
    ptable : astropy.table.Table
        Processing table that contains the jobs you want updated with the most recent queue table. Must
        have at least columnns 'LATEST_QID' and 'STATUS'.
    qtable : astropy.table.Table
        Table with the columns defined by the input variable 'columns' and information relating
        to all jobs submitted by the specified user in the specified time frame.
    ignore_scriptnames : bool
        Default is False. Set to true if you do not
        want to check whether the scriptname matches the jobname
        return by the slurm scheduler.
    check_complete_jobs: bool
        Default is False. Set to true if you want to
        also check QID's that currently have a STATUS "COMPLETED".
        in the ptable.
    dry_run_level : int
        If nonzero, this is a simulated run. Default is 0.
        0 which runs the code normally.
        1 writes all files but doesn't submit any jobs to Slurm.
        2 writes tables but doesn't write scripts or submit anything.
        3 Doesn't write or submit anything but queries Slurm normally for job status.
        4 Doesn't write, submit jobs, or query Slurm.
        5 Doesn't write, submit jobs, or query Slurm; instead it makes up the status of the jobs.

    Returns
    -------
    ptab : astropy.table.Table
        A opy of the same processing table as the input except that the "STATUS" column in ptable for all jobs is
        updated based on the 'STATE' in the qtable (as matched by "LATEST_QID" in the ptable
        and "JOBID" in the qtable).
    """
    log = get_logger()
    ptab = ptable.copy()
    if qtable is None:
        log.info("qtable not provided, querying Slurm using ptab's LATEST_QID set")
        ## Avoid null valued QID's (set to 2)
        sel = ptab['LATEST_QID'] > 2
        ## Only submit incomplete jobs unless explicitly told to check them
        ## completed jobs shouldn't change status
        if not check_complete_jobs:
            sel &= (ptab['STATUS'] != 'COMPLETED')
        log.info(f"Querying Slurm for {np.sum(sel)} QIDs from table of length {len(ptab)}.")
        qids = np.array(ptab['LATEST_QID'][sel])
        ## If you provide empty jobids Slurm gives you the three most recent jobs,
        ## which we don't want here
        if len(qids) == 0:
            log.info(f"No QIDs left to query. Returning the original table.")
            return ptab
        qtable = queue_info_from_qids(qids, dry_run_level=dry_run_level)

    log.info(f"Slurm returned information on {len(qtable)} jobs out of "
             +f"{len(ptab)} jobs in the ptab. Updating those now.")

    check_scriptname = ('JOBNAME' in qtable.colnames
                        and 'SCRIPTNAME' in ptab.colnames
                        and not ignore_scriptnames)
    if check_scriptname:
        log.info("Will be verifying that the file names are consistent")

    for row in qtable:
        if int(row['JOBID']) == get_default_qid():
            continue
        match = (int(row['JOBID']) == ptab['LATEST_QID'])
        if np.any(match):
            ind = np.where(match)[0][0]
            if check_scriptname and ptab['SCRIPTNAME'][ind] not in row['JOBNAME']:
                log.warning(f"For job with expids:{ptab['EXPID'][ind]}"
                            + f" the scriptname is {ptab['SCRIPTNAME'][ind]}"
                            + f" but the jobname in the queue was "
                            + f"{row['JOBNAME']}.")
            state = str(row['STATE']).split(' ')[0]
            ## Since dry run 1 and 2 save proc tables, don't alter the
            ## states for these when simulating
            ptab['STATUS'][ind] = state

    return ptab

def any_jobs_not_complete(statuses, termination_states=None):
    """
    Returns True if any of the job statuses in the input column of the processing table, statuses, are not complete
    (as based on the list of acceptable final states, termination_states, given as an argument. These should be states
    that are viewed as final, as opposed to job states that require resubmission.

    Parameters
    ----------
    statuses : Table.Column or list or np.array
        The statuses in the processing table "STATUS". Each element should
        be a string.
    termination_states : list or np.array
        Each element should be a string signifying a state that is returned
        by the Slurm scheduler that should be deemed terminal state.

    Returns
    -------
    bool
        True if any of the statuses of the jobs given in statuses are NOT a member of the termination states.
        Otherwise returns False.
    """
    if termination_states is None:
        termination_states = get_termination_states()
    return np.any([status not in termination_states for status in statuses])

def any_jobs_failed(statuses, failed_states=None):
    """
    Returns True if any of the job statuses in the input column of the
    processing table, statuses, are not complete (as based on the list of
    acceptable final states, termination_states, given as an argument. These
    should be states that are viewed as final, as opposed to job states
    that require resubmission.

    Parameters
    ----------
    statuses : Table.Column or list or np.array
        The statuses in the
        processing table "STATUS". Each element should be a string.
    failed_states : list or np.array
        Each element should be a string
        signifying a state that is returned by the Slurm scheduler that
        should be consider failing or problematic.

    Returns
    -------
    bool
        True if any of the statuses of the jobs given in statuses are
        a member of the failed_states.
    """
    if failed_states is None:
        failed_states = get_failed_states()
    return np.any([status in failed_states for status in statuses])

def get_jobs_in_queue(user=None, include_scron=False, dry_run_level=0):
    """
    Queries the NERSC Slurm database using sacct with appropriate flags to get
    information about specific jobs based on their jobids.

    Parameters
    ----------
    user : str
        NERSC user to query the jobs for
    include_scron : bool
        True if you want to include scron entries in the returned table.
        Default is False.
    dry_run_level : int
        If nonzero, this is a simulated run. Default is 0.
        0 which runs the code normally.
        1 writes all files but doesn't submit any jobs to Slurm.
        2 writes tables but doesn't write scripts or submit anything.
        3 Doesn't write or submit anything but queries Slurm normally for job status.
        4 Doesn't write, submit jobs, or query Slurm.
        5 Doesn't write, submit jobs, or query Slurm; instead it makes up the status of the jobs.

    Returns
    -------
    astropy.table.Table
        Table with the columns JOBID, PARTITION, RESERVATION, NAME, USER, ST, TIME, NODES,
        NODELIST(REASON) for the specified user.
    """
    log = get_logger()
    if user is None:
        if 'USER' in os.environ:
            user = os.environ['USER']
        else:
            user = 'desi'

    cmd = f'squeue -u {user} -o "%i,%P,%v,%j,%u,%t,%M,%D,%R"'
    cmd_as_list = cmd.split()

    header = 'JOBID,PARTITION,RESERVATION,NAME,USER,ST,TIME,NODES,NODELIST(REASON)'
    if dry_run_level > 4:
        log.info("Dry run, would have otherwise queried Slurm with the"
                 +f" following: {' '.join(cmd_as_list)}")
        string = header
        string += f"27650097,cron,(null),scron_ar,{user},PD,0:00,1,(BeginTime)"
        string += f"27650100,cron,(null),scron_nh,{user},PD,0:00,1,(BeginTime)"
        string += f"27650098,cron,(null),scron_up,{user},PD,0:00,1,(BeginTime)"
        string += f"29078887,gpu_ss11,(null),tilenight-20230413-24315,{user},PD,0:00,1,(Priority)"
        string += f"29078892,gpu_ss11,(null),tilenight-20230413-21158,{user},PD,0:00,1,(Priority)"
        string += f"29079325,gpu_ss11,(null),tilenight-20240309-24526,{user},PD,0:00,1,(Dependency)"
        string += f"29079322,gpu_ss11,(null),ztile-22959-thru20240309,{user},PD,0:00,1,(Dependency)"
        string += f"29078883,gpu_ss11,(null),tilenight-20230413-21187,{user},R,10:18,1,nid003960"
        string += f"29079242,regular_milan_ss11,(null),arc-20240309-00229483-a0123456789,{user},PD,0:00,3,(Priority)"
        string += f"29079246,regular_milan_ss11,(null),arc-20240309-00229484-a0123456789,{user},PD,0:00,3,(Priority)"

        # create command to run to exercise subprocess -> stdout parsing
        cmd = 'echo ' + string
        cmd_as_list = ['echo', string]
    elif dry_run_level > 3:
        cmd = 'echo ' + header
        cmd_as_list = ['echo', header]
    else:
        log.info(f"Querying jobs in queue with: {' '.join(cmd_as_list)}")

    #- sacct sometimes fails; try several times before giving up
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            table_as_string = subprocess.check_output(cmd_as_list, text=True,
                                          stderr=subprocess.STDOUT)
            break
        except subprocess.CalledProcessError as err:
            log.error(f'{cmd} job query failure at {datetime.datetime.now()}')
            log.error(f'{cmd_as_list}')
            log.error(f'{err.output=}')
    else:  #- for/else happens if loop doesn't succeed
        msg = f'{cmd} query failed {max_attempts} times; exiting'
        log.critical(msg)
        raise RuntimeError(msg)

    ## remove extra quotes that astropy table does't like
    table_as_string = table_as_string.replace('"','')

    ## remove parenthesis are also not very desirable
    table_as_string = table_as_string.replace('(', '').replace(')', '')


    ## remove node list with hyphen or comma otherwise it will break table reader
    table_as_string = re.sub(r"nid\[[0-9,-]*\]", "multiple nodes", table_as_string)

    try:
        queue_info_table = Table.read(table_as_string, format='ascii.csv')
    except:
        log.info("Table retured by squeue couldn't be parsed. The string was:")
        print(table_as_string)
        raise
    
    for col in queue_info_table.colnames:
        queue_info_table.rename_column(col, col.upper())

    ## If the table is empty, return it immediately, otherwise perform
    ## sanity check and cuts
    if len(queue_info_table) == 0:
        return queue_info_table

    if np.any(queue_info_table['USER']!=user):
        msg = f"Warning {np.sum(queue_info_table['USER']!=user)} " \
              + f"jobs returned were not {user=}\n" \
              + f"{queue_info_table['USER'][queue_info_table['USER']!=user]}"
        log.critical(msg)
        raise ValueError(msg)

    if not include_scron:
        queue_info_table = queue_info_table[queue_info_table['PARTITION'] != 'cron']

    return queue_info_table


def check_queue_count(user=None, include_scron=False, dry_run_level=0):
    """
    Queries the NERSC Slurm database using sacct with appropriate flags to get
    information about specific jobs based on their jobids.

    Parameters
    ----------
    user : str
        NERSC user to query the jobs for
    include_scron : bool
        True if you want to include scron entries in the returned table.
        Default is False.
    dry_run_level : int
        If nonzero, this is a simulated run. Default is 0.
        0 which runs the code normally.
        1 writes all files but doesn't submit any jobs to Slurm.
        2 writes tables but doesn't write scripts or submit anything.
        3 Doesn't write or submit anything but queries Slurm normally for job status.
        4 Doesn't write, submit jobs, or query Slurm.
        5 Doesn't write, submit jobs, or query Slurm; instead it makes up the status of the jobs.

    Returns
    -------
    int
        The number of jobs for that user in the queue (including or excluding
        scron entries depending on include_scron).
    """
    return len(get_jobs_in_queue(user=user, include_scron=include_scron,
                                 dry_run_level=dry_run_level))
