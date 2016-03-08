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
        alldone = True
        for outf in tsk['outputs']:
            if not os.path.isfile(outf):
                alldone = False
        if alldone:
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
            for outf in tsk['outputs']:
                ret = sp.call(['mv', '{}.part'.format(outf), outf])

    return


def shell_job(path, logroot, envsetup, desisetup, commands):
    with open(path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("now=`date +%Y%m%d-%H:%M:%S`\n")
        f.write("log={}_${{now}}\n\n".format(logroot))
        for com in envsetup:
            f.write("{}\n".format(com))
        f.write("\n")
        f.write("source {}\n\n".format(desisetup))
        for com in commands:
            executable = com.split(' ')[0]
            f.write("which {}\n".format(executable))
            f.write("time {} >${{log}} 2>&1\n\n".format(com))
    return


def nersc_job(path, logroot, envsetup, desisetup, commands, nodes=1, nodeproc=1, minutes=10, openmp=False, multiproc=False):
    hours = int(minutes/60)
    fullmin = int(minutes - 60*hours)
    timestr = "{:02d}:{:02d}:00".format(hours, fullmin)
    with open(path, 'w') as f:
        f.write("#!/bin/bash -l\n\n")
        f.write("#SBATCH --partition=debug\n")
        f.write("#SBATCH --account=desi\n")
        f.write("#SBATCH --nodes={}\n".format(nodes))
        f.write("#SBATCH --time={}\n".format(timestr))
        f.write("#SBATCH --job-name=desipipe\n")
        f.write("#SBATCH --output={}_slurm_%j\n".format(logroot))
        f.write("#SBATCH --export=NONE\n\n")
        for com in envsetup:
            f.write("{}\n".format(com))
        f.write("\n")
        f.write("source {}\n\n".format(desisetup))
        f.write("node_cores=0\n")
        f.write("if [ ${NERSC_HOST} = edison ]; then\n")
        f.write("  node_cores=24\n")
        f.write("else\n")
        f.write("  node_cores=32\n")
        f.write("fi\n")
        f.write("\n")
        f.write("nodes={}\n".format(nodes))
        f.write("node_proc={}\n".format(nodeproc))
        f.write("node_thread=$(( node_cores / node_proc ))\n")
        f.write("procs=$(( nodes * node_proc ))\n\n")
        if openmp:
            f.write("export OMP_NUM_THREADS=${node_thread}\n")
            f.write("\n")
        runstr = "srun --export=ALL"
        if multiproc:
            runstr = "{} --cpu_bind=no".format(runstr)
        f.write("run=\"{} -n ${{procs}} -N ${{nodes}} -c ${{node_thread}}\"\n\n".format(runstr))
        f.write("now=`date +%Y%m%d-%H:%M:%S`\n")
        f.write("echo \"job datestamp = ${now}\"\n")
        f.write("log={}_${{now}}\n\n".format(logroot))
        for com in commands:
            executable = com.split(' ')[0]
            f.write("which {}\n".format(executable))
            f.write("time ${{run}} {} >${{log}} 2>&1\n\n".format(com))
    return

