#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.run
=====================

Tools for running the pipeline.
"""

from __future__ import absolute_import, division, print_function

import os
import errno
import sys
import subprocess as sp

from desispec.log import get_logger


def pid_exists( pid ):
    """Check whether pid exists in the current process table.

    **UNIX only.**  Should work the same as psutil.pid_exists().

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


def subprocess_list(tasks, rank=0):
    log = get_logger()
    #pids = []
    for tsk in tasks:
        runcom = True
        for dep in tsk['inputs']:
            if not os.path.isfile(dep):
                print("missing ", dep)
                err = "dependency {} missing, cannot run task {}".format(dep, " ".join(tsk['command']))
                print(err)
                log.error(err)
                runcom = False
        proc = None
        if runcom:
            # proc = sp.Popen(tsk['command'], stdout=sp.PIPE, stderr=sp.STDOUT)
            # log.info("subproc[{}]: {}".format(proc.pid, " ".join(tsk['command'])))
            # outs, errs = proc.communicate()
            # for line in outs:
            #     log.debug("subproc[{}]:   {}".format(proc.pid, line.rstrip()))
            # proc.wait()
            log.info("subproc: {}".format(" ".join(tsk['command'])))
            ret = sp.call(tsk['command'])
    return


class Machine(object):
    """
    Class representing the number nodes and cores available to us,
    and also the commands needed to run jobs.
    """
    def __init__(self):
        pass

    def nodes(self):
        """
        The number of available nodes.
        """
        return 1

    def node_cores(self):
        """
        The number of cores per node.
        """
        return 1

    def run(self, command, logfile, nodes=1, ppn=1, nthread=1):
        """
        Run the command on the specified number of nodes,
        processes per node, and OMP_NUM_THREADS.  Write
        output to logfile.
        """
        if (nodes != 1) or (ppn != 1):
            raise RuntimeError("Machine base class only supports a single process.")
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = nthread
        proc = sp.Popen(command, env=env, stdout=open(logfile, "w"), stderr=sp.STDOUT, stdin=None, close_fds=True)
        return proc


class MachineSlurm(object):
    """
    Machine class which runs jobs using the srun command from SLURM.
    """
    def __init__(self, max_nodes, cores_per_node, shared_nodes=False):
        self._max_nodes = max_nodes
        self._max_ppn = cores_per_node
        self._max_proc = max_nodes * cores_per_node
        self._avail = self._max_proc
        self._shared = shared_nodes

    def nodes(self):
        """ 
        The number of total nodes.
        """
        return self._max_nodes

    def node_cores(self):
        """
        The number of cores per node.
        """
        return self._max_ppn

    def run(self, command, logfile, nodes=1, ppn=1, nthread=1, use_multiprocess=False):
        """
        Run the command on the specified number of nodes,
        processes per node, and OpenMP threads.  Write
        output to logfile.  If use_multiprocess is True,
        disable CPU binding.
        """
        if use_multiprocess:
            nthread = 1
        if self._shared:
            requested = nodes * ppn * nthread
        else:
            requested = nodes * self._max_ppn
        if requested > self._avail:
            raise RuntimeError("Requested {} cores, but only {} are available".format(requested, self._avail))
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = nthread
        scom = ['srun', '-n', '{}'.format(nodes * ppn), '-c', '{}'.format(nthread)]
        if use_multiprocess:
            scom.extend(['--cpu_bind=no'])
        scom.extend(command)
        proc = sp.Popen(scom, env=env, stdout=open(logfile, "w"), stderr=sp.STDOUT, stdin=None, close_fds=True)
        return proc

