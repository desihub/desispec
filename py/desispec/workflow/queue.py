

import os
import numpy as np
from astropy.table import Table
import subprocess
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
    return ['UNSUBMITTED', 'BOOT_FAIL', 'DEADLINE', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'TIMEOUT']


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



def refresh_queue_info_table(start_time=None, end_time=None, user=None, \
                             columns='jobid,state,submit,eligible,start,end,jobname', dry_run=False):
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
                      in the default list are: ther format columns: jobid,state,submit,eligible,start,end,elapsed,
                      suspended,exitcode,derivedexitcode,reason,priority,jobname.
        dry_run, bool. Whether this is a simulated run or real run. If True, it is a simulation and it returns a default
                       table that doesn't query the Slurm scheduler.

    Returns:
        queue_info_table, Table. Table with the columns defined by the input variable 'columns' and information relating
                                 to all jobs submitted by the specified user in the specified time frame.
    """
    # global queue_info_table
    if dry_run:
        string = 'JobID,State,Submit,Start,End,ExitCode,DerivedExitCode,Reason' + '\n'
        string += '30010130,COMPLETED,2020-04-25T13:24:54,2020-04-25T23:58:30,2020-04-26T00:05:41,0:0,0:0,QOSGrpMemLimit' + '\n'
        string += '30010136,COMPLETED,2020-04-25T13:25:10,2020-04-26T00:05:45,2020-04-26T00:13:55,0:0,0:0,QOSGrpMemLimit' + '\n'
        string += '30010141,COMPLETED,2020-04-25T13:25:29,2020-04-26T00:13:55,2020-04-26T00:21:34,0:0,0:0,QOSGrpMemLimit' + '\n'
        string += '30010146,COMPLETED,2020-04-25T13:25:38,2020-04-26T00:21:37,2020-04-26T00:28:56,0:0,0:0,QOSGrpMemLimit' + '\n'
        string += '30010149,COMPLETED,2020-04-25T13:25:44,2020-04-26T00:28:59,2020-04-26T00:36:04,0:0,0:0,QOSGrpMemLimit' + '\n'
        string += '30010152,COMPLETED,2020-04-25T13:25:53,2020-04-26T00:36:05,2020-04-26T00:43:05,0:0,0:0,QOSGrpMemLimit' + '\n'
        string += '30010153,COMPLETED,2020-04-25T13:25:59,2020-04-26T00:43:07,2020-04-26T00:50:21,0:0,0:0,QOSGrpMemLimit' + '\n'
        string += '30010154,COMPLETED,2020-04-25T13:26:08,2020-04-26T00:50:26,2020-04-26T00:57:57,0:0,0:0,QOSGrpMemLimit' + '\n'
        string += '30010157,COMPLETED,2020-04-25T13:26:16,2020-04-26T00:57:58,2020-04-26T01:05:15,0:0,0:0,QOSGrpMemLimit' + '\n'
        string += '30010161,COMPLETED,2020-04-25T13:26:26,2020-04-26T01:05:33,2020-04-26T01:12:53,0:0,0:0,QOSGrpMemLimit'
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

    queue_info_table = Table.read(subprocess.check_output(cmd_as_list, stderr=subprocess.STDOUT, text=True),
                                  format='ascii.csv')
    for col in queue_info_table.colnames:
        queue_info_table.rename_column(col, col.upper())
    return queue_info_table




def update_from_queue(ptable, qtable=None, start_time=None, end_time=None, user=None, dry_run=False,):
    """
    Given an input prcessing table (ptable) and query table from the Slurm queue (qtable) it cross matches the
    Slurm job ID's and updates the 'state' in the table using the current state in the Slurm scheduler system.

    Args:
        ptable, Table. Processing table that contains the jobs you want updated with the most recent queue table. Must
                       have at least columnns 'LATEST_QID' and 'STATUS'.
        qtable, Table. Table with the columns defined by the input variable 'columns' and information relating
                                 to all jobs submitted by the specified user in the specified time frame.

        The following are only used if qtable is not provided:
            start_time, str. String of the form YYYY-mm-ddTHH:MM:SS. Based on the given night and the earliest hour you
                             want to see queue information about.
            end_time, str. String of the form YYYY-mm-ddTHH:MM:SS. Based on the given night and the latest hour you
                             want to see queue information about.
            user, str. The username at NERSC that you want job information about. The default is an the environment name if
                       if exists, otherwise 'desi'.
            dry_run, bool. Whether this is a simulated run or real run. If True, it is a simulation and it returns a default
                           table that doesn't query the Slurm scheduler.

    Returns:
        ptable, Table. The same processing table as the input except that the "STATUS" column in ptable for all jobs is
                       updated based on the 'STATE' in the qtable (as matched by "LATEST_QID" in the ptable
                       and "JOBID" in the qtable).
    """
    if qtable is None:
        qtable = refresh_queue_info_table(start_time=start_time, end_time=end_time, user=user, dry_run=dry_run)

    for row in qtable:
        match = (int(row['JOBID']) == ptable['LATEST_QID'])
        # 'jobid,state,submit,eligible,start,end,jobname'
        if np.any(match):
            ind = np.where(match)[0][0]
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