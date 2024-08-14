"""
desispec.workflow.queue
=======================

"""
import os
import re
import numpy as np
from astropy.table import Table, vstack
import subprocess
from desiutil.log import get_logger
import time, datetime

global _cached_slurm_states
_cached_slurm_states = dict()

def get_resubmission_states():
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

    Returns:
        list. A list of strings outlining the job states that should be resubmitted.
    """
    return ['UNSUBMITTED', 'BOOT_FAIL', 'DEADLINE', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'TIMEOUT', 'CANCELLED']


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


def queue_info_from_time_window(start_time=None, end_time=None, user=None, \
                             columns='jobid,jobname,partition,submit,eligible,'+
                                     'start,end,elapsed,state,exitcode',
                             dry_run=0):
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
    dry_run : int
        Whether this is a simulated run or real run. If nonzero, it is a simulation and it returns a default
        table that doesn't query the Slurm scheduler.

    Returns
    -------
    Table
        Table with the columns defined by the input variable 'columns' and information relating
        to all jobs submitted by the specified user in the specified time frame.
    """
    # global queue_info_table
    if dry_run:
        string = 'JobID,JobName,Partition,Submit,Eligible,Start,End,State,ExitCode\n'
        string += '49482394,arc-20211102-00107062-a0123456789,realtime,2021-11-02'\
                  +'T18:31:14,2021-11-02T18:36:33,2021-11-02T18:36:33,2021-11-02T'\
                  +'18:48:32,COMPLETED,0:0' + '\n'
        string += '49482395,arc-20211102-00107063-a0123456789,realtime,2021-11-02'\
                  +'T18:31:16,2021-11-02T18:36:33,2021-11-02T18:48:34,2021-11-02T'\
                  +'18:57:02,COMPLETED,0:0' + '\n'
        string += '49482397,arc-20211102-00107064-a0123456789,realtime,2021-11-02'\
                  +'T18:31:19,2021-11-02T18:36:33,2021-11-02T18:57:05,2021-11-02T'\
                  +'19:06:17,COMPLETED,0:0' + '\n'
        string += '49482398,arc-20211102-00107065-a0123456789,realtime,2021-11-02'\
                  +'T18:31:24,2021-11-02T18:36:33,2021-11-02T19:06:18,2021-11-02T'\
                  +'19:13:59,COMPLETED,0:0' + '\n'
        string += '49482399,arc-20211102-00107066-a0123456789,realtime,2021-11-02'\
                  +'T18:31:27,2021-11-02T18:36:33,2021-11-02T19:14:00,2021-11-02T'\
                  +'19:24:49,COMPLETED,0:0'
        cmd_as_list = ['echo', string]
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
                         'eligible,start,end,elapsed,state,exitcode', dry_run=0):
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
    dry_run : int
        Whether this is a simulated run or real run. If nonzero, it is a simulation and it returns a default
        table that doesn't query the Slurm scheduler.

    Returns
    -------
    Table
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
            results.append(queue_info_from_qids(qids[i:i+nmax], columns=columns, dry_run=dry_run))
        results = vstack(results)
        return results

    ## Turn the queue id's into a list
    ## this should work with str or int type also, though not officially supported
    qid_str = ','.join(np.atleast_1d(qids).astype(str)).replace(' ','')

    cmd_as_list = ['sacct', '-X', '--parsable2', '--delimiter=,',
                   f'--format={columns}', '-j', qid_str]
    if dry_run:
        log.info("Dry run, would have otherwise queried Slurm with the"
                 +f" following: {' '.join(cmd_as_list)}")
        string = 'JobID,JobName,Partition,Submit,Eligible,Start,End,State,ExitCode'
        for jobid, expid in zip(qids, 100000+np.arange(len(qids))):
            string += f'\n{jobid},arc-20211102-{expid:08d}-a0123456789,realtime,2021-11-02'\
                  +'T18:31:14,2021-11-02T18:36:33,2021-11-02T18:36:33,2021-11-02T'\
                  +'18:48:32,COMPLETED,0:0'

        # create command to run to exercise subprocess -> stdout parsing
        cmd_as_list = ['echo', string]
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

def get_queue_states_from_qids(qids, dry_run=0, use_cache=False):
    """
    Queries the NERSC Slurm database using sacct with appropriate flags to get
    information on the job STATE. If use_cache is set and all qids have cached
    values from a previous query, those cached states will be returned instead.

    Parameters
    ----------
    jobids : list or array of ints
        Slurm QID's at NERSC that you want to return information about.
    dry_run : int
        Whether this is a simulated run or real run. If nonzero, it is a simulation and it returns a default
        table that doesn't query the Slurm scheduler.
    use_cache, bool. If True the code first looks for a cached status
        for the qid. If unavailable, then it queries Slurm. Default is False.

    Returns
    -------
    Dict
        Dictionary with the keys as jobids and values as the slurm state of the job.
    """
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
        outtable = queue_info_from_qids(qids, columns='jobid,state', dry_run=dry_run)
        for row in outtable:
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
    _cached_slurm_states[int(qid)] = state

def clear_queue_state_cache():
    """
    Remove all entries from the queue state cache
    """
    global _cached_slurm_states
    _cached_slurm_states.clear()


def update_from_queue(ptable, qtable=None, dry_run=0, ignore_scriptnames=False):
    """
    Given an input prcessing table (ptable) and query table from the Slurm queue (qtable) it cross matches the
    Slurm job ID's and updates the 'state' in the table using the current state in the Slurm scheduler system.

    Args:
        ptable, Table. Processing table that contains the jobs you want updated with the most recent queue table. Must
                       have at least columnns 'LATEST_QID' and 'STATUS'.
        qtable, Table. Table with the columns defined by the input variable 'columns' and information relating
                                 to all jobs submitted by the specified user in the specified time frame.
        ignore_scriptnames, bool. Default is False. Set to true if you do not
                        want to check whether the scriptname matches the jobname
                        return by the slurm scheduler.
        The following are only used if qtable is not provided:
            dry_run, int. Whether this is a simulated run or real run. If nonzero, it is a simulation and it returns a default
                           table that doesn't query the Slurm scheduler.

    Returns:
        ptable, Table. The same processing table as the input except that the "STATUS" column in ptable for all jobs is
                       updated based on the 'STATE' in the qtable (as matched by "LATEST_QID" in the ptable
                       and "JOBID" in the qtable).
    """
    log = get_logger()
    if qtable is None:
        log.info("qtable not provided, querying Slurm using ptable's LATEST_QID set")
        qids = np.array(ptable['LATEST_QID'])
        ## Avoid null valued QID's (set to -99)
        qids = qids[qids > 0]
        qtable = queue_info_from_qids(qids, dry_run=dry_run)

    log.info(f"Slurm returned information on {len(qtable)} jobs out of "
             +f"{len(ptable)} jobs in the ptable. Updating those now.")

    check_scriptname = ('JOBNAME' in qtable.colnames
                        and 'SCRIPTNAME' in ptable.colnames
                        and not ignore_scriptnames)
    if check_scriptname:
        log.info("Will be verifying that the file names are consistent")

    for row in qtable:
        match = (int(row['JOBID']) == ptable['LATEST_QID'])
        if np.any(match):
            ind = np.where(match)[0][0]
            if check_scriptname and ptable['SCRIPTNAME'][ind] not in row['JOBNAME']:
                log.warning(f"For job with expids:{ptable['EXPID'][ind]}"
                            + f" the scriptname is {ptable['SCRIPTNAME'][ind]}"
                            + f" but the jobname in the queue was "
                            + f"{row['JOBNAME']}.")
            state = str(row['STATE']).split(' ')[0]
            ptable['STATUS'][ind] = state

    return ptable

def any_jobs_not_complete(statuses, termination_states=None):
    """
    Returns True if any of the job statuses in the input column of the processing table, statuses, are not complete
    (as based on the list of acceptable final states, termination_states, given as an argument. These should be states
    that are viewed as final, as opposed to job states that require resubmission.

    Args:
        statuses, Table.Column or list or np.array. The statuses in the processing table "STATUS". Each element should
                                                    be a string.
        termination_states, list or np.array. Each element should be a string signifying a state that is returned
                                              by the Slurm scheduler that should be deemed terminal state.

    Returns:
        bool. True if any of the statuses of the jobs given in statuses are NOT a member of the termination states.
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

    Args:
        statuses, Table.Column or list or np.array. The statuses in the
            processing table "STATUS". Each element should be a string.
        failed_states, list or np.array. Each element should be a string
            signifying a state that is returned by the Slurm scheduler that
            should be consider failing or problematic.

    Returns:
        bool. True if any of the statuses of the jobs given in statuses are 
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
        Whether this is a simulated run or real run. If nonzero, it is a
        simulation and it returns a default table that doesn't query the
        Slurm scheduler.

    Returns
    -------
    Table
        Table with the columns JOBID, PARTITION, NAME, USER, ST, TIME, NODES,
        NODELIST(REASON) for the specified user.
    """
    log = get_logger()

    cmd = f'squeue -u {user} -o "%i,%P,%j,%u,%t,%M,%D,%R"'
    cmd_as_list = cmd.split()

    if dry_run_level > 0:
        log.info("Dry run, would have otherwise queried Slurm with the"
                 +f" following: {' '.join(cmd_as_list)}")
        string = 'JOBID,PARTITION,NAME,USER,ST,TIME,NODES,NODELIST(REASON)'
        string += f"27650097,cron,scron_ar,{user},PD,0:00,1,(BeginTime)"
        string += f"27650100,cron,scron_nh,{user},PD,0:00,1,(BeginTime)"
        string += f"27650098,cron,scron_up,{user},PD,0:00,1,(BeginTime)"
        string += f"29078887,gpu_ss11,tilenight-20230413-24315,{user},PD,0:00,1,(Priority)"
        string += f"29078892,gpu_ss11,tilenight-20230413-21158,{user},PD,0:00,1,(Priority)"
        string += f"29079325,gpu_ss11,tilenight-20240309-24526,{user},PD,0:00,1,(Dependency)"
        string += f"29079322,gpu_ss11,ztile-22959-thru20240309,{user},PD,0:00,1,(Dependency)"
        string += f"29078883,gpu_ss11,tilenight-20230413-21187,{user},R,10:18,1,nid003960"
        string += f"29079242,regular_milan_ss11,arc-20240309-00229483-a0123456789,{user},PD,0:00,3,(Priority)"
        string += f"29079246,regular_milan_ss11,arc-20240309-00229484-a0123456789,{user},PD,0:00,3,(Priority)"

        # create command to run to exercise subprocess -> stdout parsing
        cmd = 'echo ' + string
        cmd_as_list = ['echo', string]
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
        Whether this is a simulated run or real run. If nonzero, it is a
        simulation and it returns a default table that doesn't query the
        Slurm scheduler.

    Returns
    -------
    int
        The number of jobs for that user in the queue (including or excluding
        scron entries depending on include_scron).
    """
    return len(get_jobs_in_queue(user=user, include_scron=include_scron,
                                 dry_run_level=dry_run_level))
