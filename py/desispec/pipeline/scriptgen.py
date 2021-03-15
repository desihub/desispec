#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.scriptgen
=============================

Tools for generating shell and slurm scripts.
"""

from __future__ import absolute_import, division, print_function

import os
import stat
import sys
import re

import numpy as np

from desiutil.log import get_logger

from .. import io

from ..parallel import (dist_uniform, dist_discrete, dist_discrete_all,
    stdouterr_redirected)

from .prod import task_read, task_write

from .plan import nersc_machine, nersc_job_size


def dump_job_env(fh, tfactor, startup, nworker, workersize):
    """Write parameters needed at runtime to an open filehandle.
    """
    fh.write("export DESI_PIPE_RUN_TIMEFACTOR={}\n".format(tfactor))
    fh.write("export DESI_PIPE_RUN_STARTUP={}\n".format(startup))
    fh.write("export DESI_PIPE_RUN_WORKERS={}\n\n".format(nworker))
    fh.write("export DESI_PIPE_RUN_WORKER_SIZE={}\n\n".format(workersize))
    return


def parse_job_env():
    """Retrieve job parameters from the environment.
    """
    par = dict()
    if "DESI_PIPE_RUN_TIMEFACTOR" in os.environ:
        par["timefactor"] = float(os.environ["DESI_PIPE_RUN_TIMEFACTOR"])
    if "DESI_PIPE_RUN_STARTUP" in os.environ:
        par["startup"] = float(os.environ["DESI_PIPE_RUN_STARTUP"])
    if "DESI_PIPE_RUN_WORKERS" in os.environ:
        par["workers"] = int(os.environ["DESI_PIPE_RUN_WORKERS"])
    if "DESI_PIPE_RUN_WORKER_SIZE" in os.environ:
        par["workersize"] = int(os.environ["DESI_PIPE_RUN_WORKER_SIZE"])
    return par


def shell_job(path, logroot, desisetup, commands, comrun="", mpiprocs=1,
              openmp=1,debug=False):
    if len(commands) == 0:
        raise RuntimeError("List of commands is empty")
    with open(path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("now=`date +%Y%m%d-%H%M%S`\n")
        f.write("export STARTTIME=${now}\n")
        f.write("log={}_${{now}}.log\n\n".format(logroot))
        f.write("source {}\n\n".format(desisetup))

        f.write("# Force the script to exit on errors from commands\n")
        f.write("set -e\n\n")

        f.write("export OMP_NUM_THREADS={}\n\n".format(openmp))
        if debug:
            f.write("export DESI_LOGLEVEL=DEBUG\n\n")

        run = ""
        if comrun != "":
            run = "{} {}".format(comrun, mpiprocs)
        for com in commands:
            executable = com.split(" ")[0]
            # f.write("which {}\n".format(executable))
            f.write("echo logging to ${log}\n")
            f.write("time {} {} >>${{log}} 2>&1\n\n".format(run, com))
    mode = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    os.chmod(path, mode)
    return


def nersc_job(jobname, path, logroot, desisetup, commands, machine, queue,
    nodes, cnodes, ppns, minutes, nworker, workersize, multisrun=False,
    openmp=False, multiproc=False, shifterimg=None, debug=False):
    """Create a SLURM script for use at NERSC.

    Args:


    """
    if len(commands) == 0:
        raise RuntimeError("List of commands is empty")
    hostprops = nersc_machine(machine, queue)

    if nodes > hostprops["maxnodes"]:
        raise RuntimeError("request nodes '{}' is too large for {} queue '{}'"\
            .format(nodes, machine, queue))

    if minutes > hostprops["maxtime"]:
        raise RuntimeError("request time '{}' is too long for {} queue '{}'"\
            .format(minutes, machine, queue))

    hours = int(minutes/60)
    fullmin = int(minutes - 60*hours)
    timestr = "{:02d}:{:02d}:00".format(hours, fullmin)

    totalnodes = nodes
    if multisrun:
        # we are running every command as a separate srun
        # and backgrounding them.  In this case, the nodes
        # given are per command, so we need to compute the
        # total.
        totalnodes = nodes * len(commands)

    with open(path, "w") as f:
        f.write("#!/bin/bash -l\n\n")
        if shifterimg is not None:
            f.write("#SBATCH --image={}\n".format(shifterimg))

        for line in hostprops["sbatch"]:
            f.write("{}\n".format(line))

        f.write("#SBATCH --account=desi\n")
        f.write("#SBATCH --nodes={}\n".format(totalnodes))
        f.write("#SBATCH --time={}\n".format(timestr))
        f.write("#SBATCH --job-name={}\n".format(jobname))
        f.write("#SBATCH --output={}_%j.log\n\n".format(logroot))

        f.write("echo Starting slurm script at `date`\n\n")
        f.write("source {}\n\n".format(desisetup))

        f.write("# Force the script to exit on errors from commands\n")
        f.write("set -e\n\n")

        f.write("# Set TMPDIR to be on the ramdisk\n")
        f.write("export TMPDIR=/dev/shm\n\n")

        f.write("cpu_per_core={}\n".format(hostprops["corecpus"]))
        f.write("node_cores={}\n\n".format(hostprops["nodecores"]))

        if debug:
            f.write("export DESI_LOGLEVEL=DEBUG\n\n")

        f.write("now=`date +%Y%m%d-%H%M%S`\n")
        f.write("echo \"job datestamp = ${now}\"\n")
        f.write("log={}_${{now}}.log\n\n".format(logroot))
        f.write("envlog={}_${{now}}.env\n".format(logroot))
        f.write("env > ${envlog}\n\n")
        for com, cn, ppn, nwrk, wrksz in zip(
            commands, cnodes, ppns, nworker, workersize):
            if ppn > hostprops["nodecores"]:
                raise RuntimeError("requested procs per node '{}' is more than"
                    " the number of cores per node on {}".format(ppn, machine))
            f.write("nodes={}\n".format(cn))
            f.write("node_proc={}\n".format(ppn))
            f.write("node_thread=$(( node_cores / node_proc ))\n")
            f.write("node_depth=$(( cpu_per_core * node_thread ))\n")
            f.write("procs=$(( nodes * node_proc ))\n\n")
            dump_job_env(f, hostprops["timefactor"], hostprops["startup"],
                         nwrk, wrksz)
            if openmp:
                f.write("export OMP_NUM_THREADS=${node_thread}\n")
                f.write("export OMP_PLACES=threads\n")
                f.write("export OMP_PROC_BIND=spread\n")
            else:
                f.write("export OMP_NUM_THREADS=1\n")
            f.write("\n")
            runstr = "srun"
            if multiproc:
                runstr = "{} --cpu_bind=no".format(runstr)
                f.write("export KMP_AFFINITY=disabled\n")
                f.write("\n")
            else:
                runstr = "{} --cpu_bind=cores".format(runstr)

            if shifterimg is None:
                f.write("run=\"{} -n ${{procs}} -N ${{nodes}} -c "
                    "${{node_depth}}\"\n\n".format(runstr))
            else:
                f.write("run=\"{} -n ${{procs}} -N ${{nodes}} -c "
                    "${{node_depth}} shifter\"\n\n".format(runstr))

            comlist = com.split(" ")
            executable = comlist.pop(0)
            f.write("ex=`which {}`\n".format(executable))
            f.write("app=\"${ex}.app\"\n")
            f.write("if [ -x ${app} ]; then\n")
            f.write("  if [ ${ex} -nt ${app} ]; then\n")
            f.write("    app=${ex}\n")
            f.write("  fi\n")
            f.write("else\n")
            f.write("  app=${ex}\n")
            f.write("fi\n")
            f.write("echo calling {} at `date`\n\n".format(executable))
            f.write("export STARTTIME=`date +%Y%m%d-%H%M%S`\n")
            f.write("echo ${{run}} ${{app}} {}\n".format(" ".join(comlist)))
            f.write("time ${{run}} ${{app}} {} >>${{log}} 2>&1".format(" ".join(comlist)))
            if multisrun:
                f.write(" &")
            f.write("\n\n")
        if multisrun:
            f.write("wait\n\n")

        f.write("echo done with slurm script at `date`\n")

    return


def batch_shell(tasks_by_type, outroot, logroot, mpirun="", mpiprocs=1,
    openmp=1, db=None):
    """Generate bash script(s) to process lists of tasks.

    Given sets of task lists, generate a script that processes each in order.

    Args:
        tasks_by_type (OrderedDict): Ordered dictionary of the tasks for each
            type to be written to a single job script.
        outroot (str): root output script name.
        logroot (str): root output log name.
        mpirun (str): optional command to use for launching MPI programs.
        mpiprocs (int): if mpirun is specified, use this number of processes.
        openmp (int): value to set for OMP_NUM_THREADS.
        db (DataBase): the pipeline database handle.

    Returns:
        (list): list of generated script files.

    """
    from .tasks.base import task_classes, task_type

    # Get the location of the setup script from the production root.
    proddir = os.path.abspath(io.specprod_root())
    desisetup = os.path.abspath(os.path.join(proddir, "setup.sh"))

    dbstr = ""
    if db is None:
        dbstr = "--nodb"

    coms = list()

    for t, tasklist in tasks_by_type.items():
        if len(tasklist) == 0:
            raise RuntimeError("{} task list is empty".format(t))

        taskfile = "{}_{}.tasks".format(outroot, t)
        task_write(taskfile, tasklist)

        if mpiprocs > 1:
            coms.append("desi_pipe_exec_mpi --tasktype {} --taskfile {} {}"\
                .format(t, taskfile, dbstr))
        else:
            coms.append("desi_pipe_exec --tasktype {} --taskfile {} {}"\
                .format(t, taskfile, dbstr))

    outfile = "{}.sh".format(outroot)

    shell_job(outfile, logroot, desisetup, coms, comrun=mpirun,
        mpiprocs=mpiprocs, openmp=openmp)

    return [ outfile ]


def batch_nersc(tasks_by_type, outroot, logroot, jobname, machine, queue,
    maxtime, maxnodes, nodeprocs=None, openmp=False, multiproc=False, db=None,
    shifterimg=None, debug=False):
    """Generate slurm script(s) to process lists of tasks.

    Given sets of task lists and constraints about the machine, generate slurm
    scripts for use at NERSC.

    Args:
        tasks_by_type (OrderedDict): Ordered dictionary of the tasks for each
            type to be written to a single job script.
        outroot (str): root output script name.
        logroot (str): root output log name.
        jobname (str): the name of the job.
        machine (str): the NERSC machine name.
        queue (str): the name of the queue
        maxtime (int): the maximum run time in minutes.
        maxnodes (int): the maximum number of nodes to use.
        nodeprocs (int): the number of processes to use per node.
        openmp (bool): if True, set OMP_NUM_THREADS to the correct value.
        multiproc (bool): if True, use OMP_NUM_THREADS=1 and disable core
            binding of processes.
        db (DataBase): the pipeline database handle.
        shifter (str): the name of the shifter image to use.
        debug (bool): if True, set DESI log level to DEBUG in the script.

    Returns:
        (list): list of generated slurm files.

    """
    from .tasks.base import task_classes, task_type

    # Get the location of the setup script from the production root.
    proddir = os.path.abspath(io.specprod_root())
    desisetup = os.path.abspath(os.path.join(proddir, "setup.sh"))

    joblist = dict()

    # How many pipeline steps are we trying to pack?
    npacked = len(tasks_by_type)

    for t, tasklist in tasks_by_type.items():
        if len(tasklist) == 0:
            raise RuntimeError("{} task list is empty".format(t))
        # Compute job size for this task type
        if npacked > 1:
            joblist[t] = nersc_job_size(
                t, tasklist, machine, queue, maxtime, maxnodes,
                nodeprocs=nodeprocs, db=db
            )
        else:
            # Safe to load balance
            joblist[t] = nersc_job_size(
                t, tasklist, machine, queue, maxtime, maxnodes,
                nodeprocs=nodeprocs, db=db, balance=True
            )
        # If we are packing multiple pipeline steps, but one of those steps
        # is already too large to fit within queue constraints, then this
        # makes no sense.
        if (len(joblist[t]) > 1) and (npacked > 1):
            log = get_logger()
            log.info("{} {} queue, maxtime={}, maxnodes={}".format(
                machine, queue, maxtime, maxnodes))
            log.info("{} {} tasks -> {} jobs".format(
                len(tasklist), t, len(joblist[t])))
            raise RuntimeError("Cannot batch multiple pipeline steps, "
                               "each with multiple jobs")

    dbstr = ""
    if db is None:
        dbstr = "--nodb"

    scriptfiles = list()

    log = get_logger()

    # Add an extra 20 minutes (!) to the overall job runtime as a buffer
    # against system issues.
    runtimebuffer = 20.0

    if npacked == 1:
        # We have a single pipeline step which might be split into multiple
        # job scripts.
        jindx = 0
        tasktype = list(tasks_by_type.keys())[0]
        for (nodes, ppn, runtime, nworker, workersize, tasks) \
            in joblist[tasktype]:
            joblogroot = None
            joboutroot = None
            if jindx>0:
                joblogroot = "{}_{}".format(logroot, jindx)
                joboutroot = "{}_{}".format(outroot, jindx)
            else:
                joblogroot = logroot
                joboutroot = outroot

            taskfile = "{}.tasks".format(joboutroot)
            task_write(taskfile, tasks)
            coms = [ "desi_pipe_exec_mpi --tasktype {} --taskfile {} {}"\
                .format(tasktype, taskfile, dbstr) ]
            outfile = "{}.slurm".format(joboutroot)

            log.debug("writing job {}".format(outfile))

            runtime += runtimebuffer

            nersc_job(jobname, outfile, joblogroot, desisetup, coms, machine,
                      queue, nodes, [ nodes ], [ ppn ], runtime, [ nworker ],
                      [ workersize ], openmp=openmp, multiproc=multiproc,
                      shifterimg=shifterimg, debug=debug)
            scriptfiles.append(outfile)
            jindx += 1

    else:
        # We are packing multiple pipeline steps into a *single* job script.
        # We have already verified that each step fits within the machine
        # and queue constraints.  We use the largest job size.
        fullnodes = 0
        fullruntime = 0
        for t in tasks_by_type.keys():
            for (nodes, ppn, runtime, nworker, workersize, tasks) in joblist[t]:
                if nodes > fullnodes:
                    fullnodes = nodes
                fullruntime += runtime

        # Verify that this total does not exceed the machine limits
        hostprops = nersc_machine(machine, queue)
        if fullruntime > hostprops["maxtime"]:
            raise RuntimeError("Packed pipeline jobs exceed time limit")

        coms = list()
        ppns = list()
        cnodes = list()
        nwk = list()
        wrksz = list()
        for t, tasklist in tasks_by_type.items():
            (nodes, ppn, runtime, nworker, workersize, tasks) = joblist[t][0]
            taskfile = "{}_{}.tasks".format(outroot, t)
            task_write(taskfile, tasks)
            coms.append("desi_pipe_exec_mpi --tasktype {} --taskfile {} {}"\
                .format(t, taskfile, dbstr))
            ppns.append(ppn)
            cnodes.append(nodes)
            nwk.append(nworker)
            wrksz.append(workersize)

        outfile = "{}.slurm".format(outroot)

        fullruntime += runtimebuffer

        nersc_job(jobname, outfile, logroot, desisetup, coms, machine,
                  queue, fullnodes, cnodes, ppns, fullruntime, nwk, wrksz,
                  openmp=openmp, multiproc=multiproc, shifterimg=shifterimg,
                  debug=debug)
        scriptfiles.append(outfile)

    return scriptfiles
