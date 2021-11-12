

import os
import numpy as np
from astropy.table import Table
import subprocess
from desiutil.log import get_logger
import time


def get_resubmission_states():
    """
    Defines what Slurm job failure modes should be resubmitted in the hopes of the job succeeding the next time.

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
        list. A list of strings outlining the job states that should be resubmitted.
    """
    return ['UNSUBMITTED', 'BOOT_FAIL', 'DEADLINE', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'TIMEOUT', 'CANCELLED']


def get_termination_states():
    """
    Defines what Slurm job states that are final and aren't in question about needing resubmission.

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
    return ['COMPLETED', 'CANCELLED', 'FAILED']



def queue_info_from_time_window(start_time=None, end_time=None, user=None, \
                             columns='jobid,jobname,partition,submit,eligible,'+
                                     'start,end,elapsed,state,exitcode',
                             dry_run=0):
    """
    Queries the NERSC Slurm database using sacct with appropriate flags to get information within a specified time
    window of all jobs submitted or executed during that time.

    Args:
        start_time, str. String of the form YYYY-mm-ddTHH:MM:SS. Based on the given night and the earliest hour you
                         want to see queue information about.
        end_time, str. String of the form YYYY-mm-ddTHH:MM:SS. Based on the given night and the latest hour you
                         want to see queue information about.
        user, str. The username at NERSC that you want job information about. The default is an the environment name if
                   if exists, otherwise 'desi'.
        columns, str. Comma seperated string of valid sacct column names, in lower case. To be useful for the workflow,
                      it should have MUST have columns "JOBID" and "STATE". Other columns available that aren't included
                      in the default list are: jobid,jobname,partition,submit,eligible,start,end,elapsed,state,exitcode.
                      Other options include: suspended,derivedexitcode,reason,priority,jobname.
        dry_run, int. Whether this is a simulated run or real run. If nonzero, it is a simulation and it returns a default
                       table that doesn't query the Slurm scheduler.

    Returns:
        queue_info_table, Table. Table with the columns defined by the input variable 'columns' and information relating
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
    return queue_info_table


def queue_info_from_qids(qids, columns='jobid,jobname,partition,submit,'+
                         'eligible,start,end,elapsed,state,exitcode',
                         dry_run=0):
    """
    Queries the NERSC Slurm database using sacct with appropriate flags to get information within a specified time
    window of all jobs submitted or executed during that time.

    Args:
        jobids, list or array of ints. Slurm QID's at NERSC that you want to
                      return information about.
        columns, str. Comma seperated string of valid sacct column names, in lower case. To be useful for the workflow,
                      it should have MUST have columns "JOBID" and "STATE". Other columns available that aren't included
                      in the default list are: jobid,jobname,partition,submit,eligible,start,end,elapsed,state,exitcode.
                      Other options include: suspended,derivedexitcode,reason,priority,jobname.
        dry_run, int. Whether this is a simulated run or real run. If nonzero, it is a simulation and it returns a default
                       table that doesn't query the Slurm scheduler.

    Returns:
        queue_info_table, Table. Table with the columns defined by the input variable 'columns' and information relating
                                 to all jobs submitted by the specified user in the specified time frame.
    """
    log = get_logger()
    ## Turn the queue id's into a list
    ## this should work with str or int type also, though not officially supported
    qid_str = ','.join(np.atleast_1d(qids).astype(str)).replace(' ','')

    cmd_as_list = ['sacct', '-X', '--parsable2', '--delimiter=,',
                   f'--format={columns}', '-j', qid_str]
    if dry_run:
        log.info("Dry run, would have otherwise queried Slurm with the"
                 +f" following: {' '.join(cmd_as_list)}")
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
        log.info(f"Querying Slurm with the following: {' '.join(cmd_as_list)}")

    table_as_string = subprocess.check_output(cmd_as_list, text=True,
                                              stderr=subprocess.STDOUT)
    queue_info_table = Table.read(table_as_string, format='ascii.csv')

    for col in queue_info_table.colnames:
        queue_info_table.rename_column(col, col.upper())
    return queue_info_table

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
            ptable['STATUS'][ind] = row['STATE']

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
