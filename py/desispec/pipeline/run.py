#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.run
=====================

Tools for running the pipeline.
"""
import os
import errno
import sys
import subprocess as sp


def pid_exists( pid ):
    """Check whether pid exists in the current process table.

    **UNIX only.**

    Args:
        pid (int): A process ID.

    Returns:
        pid_exists (bool): ``True`` if the process exists in the current process table.
    """
    if pid < 0:
        return False
    if pid == 0:
        # According to "man 2 kill" PID 0 refers to every process
        # in the process group of the calling process.
        # On certain systems 0 is a valid PID but we have no way
        # to know that in a portable fashion.
        raise ValueError('invalid PID 0')
    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            # ESRCH == No such process
            return False
        elif err.errno == errno.EPERM:
            # EPERM clearly means there's a process to deny access to
            return True
        else:
            # According to "man 2 kill" possible error values are
            # (EINVAL, EPERM, ESRCH)
            raise
    else:
        return True



class Machine( object ):
    """This class represents the properties of a single machine.  This includes
    the node configuration, etc.
    """

    def __init__( self ):
        pass

    def nodes( self ):
        """Returns the maximum number of nodes to use on this machine.
        """
        return 1

    def cores_per_node( self ):
        """Returns the number of cores per node on this machine.
        """
        return 1

    def proc_spawn( self, com, logfile ):
        """Spawn a process and redirect output to a log file.
        """
        proc = sp.Popen( com, stdout=open ( logfile, "w" ), stderr=sp.STDOUT, stdin=None, close_fds=True )
        return proc.pid

    def proc_wait( self, pid ):
        """Wait for a process to finish.
        """
        if pid_exists( pid ):
            ex = os.wait( pid )
            return ex[1]
        else:
            return 0

    def proc_poll( self, pid ):
        """Check if specified process is still running.
        """
        return pid_exists( pid )

    def job_run( self, com, nodes, ppn, log ):
        """Runs a command on a specified number of nodes, and a specified number
        of processes per node.
        """
        jobid = 0
        return jobid

    def job_wait( self, jobid ):
        """Wait for a job to finish before returning.
        """
        return

    def job_complete( self, jobid ):
        """Is the speficied jobid still running?
        """
        return False


class MachineLocal( Machine ):
    """Local machine definition.  We use one node and one process, and use
    subprocess to run jobs.
    """

    def __init__( self ):
        pass

    def job_run( self, com, nodes, ppn, log ):
        """Placeholder
        """
        return jobid

    def job_wait( self, jobid ):
        """Wait for a job to finish before returning.
        """
        return

    def job_complete( self, jobid ):
        """Is the speficied jobid still running?
        """
        return False

"""
Notes to keep handy while editing:

class MachineEdison

PBS_VERSION=TORQUE-4.2.7
PBS_JOBNAME=STDIN
PBS_ENVIRONMENT=PBS_INTERACTIVE
PBS_O_WORKDIR=/global/u1/k/kisner
PBS_TASKNUM=1
PBS_O_HOME=/global/homes/k/kisner
PBSCOREDUMP=True
PBS_WALLTIME=1800
PBS_GPUFILE=/var/spool/torque/aux//2153436.edique02gpu
PBS_MOMPORT=15003
PBS_O_QUEUE=debug
PBS_O_LOGNAME=kisner
PBS_JOBCOOKIE=C2EA813B2802321F3DF6015972F33122
PBS_NODENUM=0
PBS_NUM_NODES=1
PBS_O_SHELL=/bin/bash
PBS_JOBID=2153436.edique02
PBS_O_HOST=edison07
PBS_VNODENUM=0
PBS_QUEUE=debug
PBS_MICFILE=/var/spool/torque/aux//2153436.edique02mic
PBS_O_SUBMIT_FILTER=/usr/syscom/nsg/sbin/submit_filter
PBS_O_MAIL=/var/mail/kisner
PBS_NP=48
PBS_NUM_PPN=1
PBS_O_SERVER=edique02
PBS_NODEFILE=/var/spool/torque/aux//2153436.edique02
"""
