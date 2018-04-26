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
    stdouterr_redirected, use_mpi)

from .prod import task_read, task_write


def nersc_machine(name, queue):
    """Return the properties of the specified NERSC host.

    Args:
        name (str): the name of the host.  Allowed values are:  edison,
            cori-haswell, and cori-knl.
        queue (str): the queue on the machine (regular, debug, etc)

    Returns:
        dict: properties of this machine.

    """
    props = dict()
    if name == "edison":
        props["sbatch"] = list()
        props["nodecores"] = 24
        props["corecpus"] = 2
        if queue == "debug":
            props["maxnodes"] = 512
            props["maxtime"] = 30
            props["submitlimit"] = 5
            props["sbatch"].append("#SBATCH --partition=debug")
        elif queue == "regular":
            props["maxnodes"] = 2048
            props["maxtime"] = 12 * 60
            props["submitlimit"] = 5000
            props["sbatch"].append("#SBATCH --partition=regular")
        else:
            raise RuntimeError("Unknown {} queue '{}'".format(name, queue))
    elif name == "cori-haswell":
        props["sbatch"] = [
            "#SBATCH --constraint=haswell"
        ]
        props["nodecores"] = 32
        props["corecpus"] = 2
        if queue == "debug":
            props["maxnodes"] = 64
            props["maxtime"] = 30
            props["submitlimit"] = 5
            props["sbatch"].append("#SBATCH --partition=debug")
        elif queue == "regular":
            props["maxnodes"] = 512
            props["maxtime"] = 12 * 60
            props["submitlimit"] = 5000
            props["sbatch"].append("#SBATCH --partition=regular")
        else:
            raise RuntimeError("Unknown {} queue '{}'".format(name, queue))
    elif name == "cori-knl":
        props["sbatch"] = [
            "#SBATCH --constraint=knl,quad,cache",
            "#SBATCH --core-spec=4"
        ]
        props["nodecores"] = 64
        props["corecpus"] = 4
        if queue == "debug":
            props["maxnodes"] = 512
            props["maxtime"] = 30
            props["submitlimit"] = 5
            props["sbatch"].append("#SBATCH --partition=debug")
        elif queue == "regular":
            props["maxnodes"] = 4096
            props["maxtime"] = 12 * 60
            props["submitlimit"] = 5000
            props["sbatch"].append("#SBATCH --partition=regular")
        else:
            raise RuntimeError("Unknown {} queue '{}'".format(name, queue))
    else:
        raise RuntimeError("Unknown machine '{}'".format(name))

    return props


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
    nodes, nodeproc, minutes, multisrun=False, openmp=False, multiproc=False,
    shifterimg=None,debug=False):
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

    if nodeproc > hostprops["nodecores"]:
        raise RuntimeError("requested procs per node '{}' is more than the "
            "the number of cores per node on {}".format(nodeproc, machine))

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

        f.write("cpu_per_core={}\n\n".format(hostprops["corecpus"]))
        f.write("node_cores={}\n\n".format(hostprops["nodecores"]))

        f.write("nodes={}\n".format(nodes))
        f.write("node_proc={}\n".format(nodeproc))
        f.write("node_thread=$(( node_cores / node_proc ))\n")
        f.write("node_depth=$(( cpu_per_core * node_thread ))\n")

        f.write("procs=$(( nodes * node_proc ))\n\n")
        if openmp:
            f.write("export OMP_NUM_THREADS=${node_thread}\n")
            f.write("export OMP_PLACES=threads\n")
            f.write("export OMP_PROC_BIND=spread\n")
        else:
            f.write("export OMP_NUM_THREADS=1\n")
        f.write("\n")

        if debug:
            f.write("export DESI_LOGLEVEL=DEBUG\n\n")

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

        f.write("now=`date +%Y%m%d-%H%M%S`\n")
        f.write("echo \"job datestamp = ${now}\"\n")
        f.write("log={}_${{now}}.log\n\n".format(logroot))
        f.write("envlog={}_${{now}}.env\n".format(logroot))
        f.write("env > ${envlog}\n\n")
        for com in commands:
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


def nersc_job_size(tasktype, tasklist, machine, queue, maxtime, maxnodes,
    nodeprocs=None, db=None):
    """Compute the NERSC job parameters based on constraints.

    Given the list of tasks, query their estimated runtimes and determine
    the "best" job size to use.  If the job is too big to fit the constraints
    then split up the tasks into multiple jobs.

    If maxtime or maxnodes is zero, then the defaults for the queue are used.

    The default behavior is to create jobs that are as large as possible- i.e.
    to run all tasks simultaneously in parallel.  In general, larger jobs with
    a shorter run time will move through the queue faster.  If the job size
    exceeds the maximum number of nodes, then the job size is fixed to this
    maximum and the runtime is extended.  If the runtime exceeds maxtime, then
    the job is split.

    Args:
        tasktype (str): the type of these tasks.
        tasklist (list): the list of tasks.
        machine (str): the nersc machine name,
            e.g. edison, cori-haswell, cori-knl
        queue (str): the nersc queue name, e.g. regular or debug
        maxtime (int): the maximum run time in minutes.
        maxnodes (int): the maximum number of nodes.
        nodeprocs (int): the number of processes per node.
        db (DataBase): the database to pass to the task runtime
            calculation.

    Returns:
        list:  List of tuples (nodes, runtime, tasks) containing one entry
            per job.  Each entry specifies the number of nodes to use, the
            expected total runtime, and the list of tasks for that job.
    """
    from .tasks.base import task_classes, task_type
    log = get_logger()

    if len(tasklist) == 0:
        raise RuntimeError("List of tasks is empty")

    # Get the machine properties
    hostprops = nersc_machine(machine, queue)

    if maxtime <= 0:
        maxtime = hostprops["maxtime"]
    if maxtime > hostprops["maxtime"]:
        raise RuntimeError("requested max time '{}' is too long for {} "
            "queue '{}'".format(maxtime, machine, queue))

    if maxnodes <= 0:
        maxnodes = hostprops["maxnodes"]
    if maxnodes > hostprops["maxnodes"]:
        raise RuntimeError("requested max nodes '{}' is larger than {} "
            "queue '{}'".format(maxtime, machine, queue))

    if nodeprocs is None:
        nodeprocs = hostprops["nodecores"]

    if nodeprocs > hostprops["nodecores"]:
        raise RuntimeError("requested procs per node '{}' is more than the "
            "the number of cores per node on {}".format(nodeprocs, machine))

    log.debug("maxtime = {}, maxnodes = {}, nodeprocs = {}".format(maxtime, maxnodes, nodeprocs))

    # Max number of procs to use per task.
    taskproc = task_classes[tasktype].run_max_procs(nodeprocs)

    # Run times for each task at this concurrency
    tasktimes = [ (x, task_classes[tasktype].run_time(x, nodeprocs,
        db=db)) for x in tasklist ]

    # We want to sort the times so that we can use a simple algorithm.
    tasktimes = list(sorted(tasktimes, key=lambda x: x[1]))[::-1]

    mintasktime = tasktimes[-1][1]
    maxtasktime = tasktimes[0][1]
    log.debug("taskproc = {}".format(taskproc))
    log.debug("tasktimes range = {} ... {}".format(mintasktime, maxtasktime))

    if maxtasktime > maxtime:
        raise RuntimeError("The longest task ({} minutes) exceeds the "
            " requested max runtime ({} minutes)".format(maxtasktime,
            maxtime))

    # Number of workers (as large as possible)
    availproc = maxnodes * nodeprocs
    maxworkers = availproc // taskproc
    nworker = maxworkers
    if nworker > len(tasklist):
        nworker = len(tasklist)
    log.debug("maxworkers = {}".format(maxworkers))
    log.debug("nworker = {}".format(nworker))

    totalnodes = (nworker * taskproc) // nodeprocs
    if totalnodes * nodeprocs < nworker * taskproc:
        totalnodes += 1

    log.debug("totalnodes = {}".format(totalnodes))

    # The returned list of jobs
    ret = list()

    # Create jobs until we run out of tasks
    alldone = False
    while not alldone:
        jobdone = False
        reverse = False
        workertasks = [ list() for w in range(nworker) ]
        workersum = [ 0 for w in range(nworker) ]
        # Distribute tasks until the job gets too big or we run out of
        # tasks.
        while not jobdone:
            # Pass back and forth over the worker list, assigning tasks.
            # Since the tasks are sorted by size, this should result in a
            # rough load balancing.
            for w in range(nworker):
                if len(tasktimes) == 0:
                    jobdone = True
                    break
                wrk = w
                if reverse:
                    wrk = nworker - 1 - w
                if workersum[wrk] + tasktimes[0][1] > maxtime:
                    jobdone = True
                    break
                else:
                    curtask = tasktimes.pop(0)
                    workertasks[wrk].append(curtask[0])
                    workersum[wrk] += curtask[1]
            reverse = not reverse

        # Process this job
        runtime = np.max(workersum)

        # Did we run out of tasks during this job?
        minrun = np.min(workersum)
        if (minrun == 0) and (len(tasktimes) == 0):
            # This was the last job of a multi-job situation, and some workers
            # have no tasks.  Shrink the job size.
            nempty = np.sum([ 1 for w in workersum if w == 0 ])
            nworker = nworker - nempty
            totalnodes = (nworker * taskproc) // nodeprocs
            if totalnodes * nodeprocs < nworker * taskproc:
                totalnodes += 1
            log.debug("shrinking final job size to {} nodes".format(totalnodes))

        outtasks = list()
        for w in workertasks:
            outtasks.extend(w)

        ret.append( (totalnodes, runtime, outtasks) )
        log.debug("job will run on {} nodes for {} minutes on {} tasks".format(totalnodes, runtime, len(outtasks)))

        if len(tasktimes) == 0:
            alldone = True

    return ret


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

        taskfile = "{}_{}.tasks".format(outroot)
        task_write(taskfile, tasklist)

        if mpiprocs > 1:
            coms = [ "desi_pipe_exec_mpi --tasktype {} --taskfile {} {}"\
                .format(t, taskfile, dbstr) ]
        else:
            coms = [ "desi_pipe_exec --tasktype {} --taskfile {} {}"\
                .format(t, taskfile, dbstr) ]

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
        joblist[t] = nersc_job_size(t, tasklist, machine, queue, maxtime,
            maxnodes, nodeprocs=nodeprocs)
        # If we are packing multiple pipeline steps, but one of those steps
        # is already too large to fit within queue constraints, then this
        # makes no sense.
        if (len(joblist[t]) > 1) and (npacked > 1):
            raise RuntimeError("Cannot batch multiple pipeline steps, "
                               "each with multiple jobs")

    dbstr = ""
    if db is None:
        dbstr = "--nodb"

    scriptfiles = list()

    if npacked == 1:
        # We have a single pipeline step which might be split into multiple
        # job scripts.
        jindx = 0
        suffix = True
        if len(joblist) == 1:
            suffix = False
        tasktype = list(tasks_by_type.keys())[0]
        for (nodes, runtime, tasks) in joblist[tasktype]:
            joblogroot = None
            joboutroot = None
            if suffix:
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
            nersc_job(jobname, outfile, joblogroot, desisetup, coms, machine,
                      queue, nodes, nodeprocs, runtime, openmp=openmp,
                      multiproc=multiproc, shifterimg=shifterimg, debug=debug)
            scriptfiles.append(outfile)
            jindx += 1

    else:
        # We are packing multiple pipeline steps into a *single* job script.
        # We have already verified that each step fits within the machine
        # and queue constraints.

        # We use the largest job size.
        fullnodes = 0
        fullruntime = 0
        for t in tasks_by_type.keys():
            for (nodes, runtime, tasks) in joblist[t]:
                if nodes > fullnodes:
                    fullnodes = nodes
                fullruntime += runtime

        # Verify that this total does not exceed the machine limits
        hostprops = nersc_machine(machine, queue)
        if fullruntime > hostprops["maxtime"]:
            raise RuntimeError("Packed pipeline jobs exceed time limit")

        coms = list()
        for t, tasklist in tasks_by_type.items():
            (nodes, runtime, tasks) = joblist[t][0]
            taskfile = "{}_{}.tasks".format(outroot, t)
            task_write(taskfile, tasks)
            coms.append("desi_pipe_exec_mpi --tasktype {} --taskfile {} {}"\
                .format(t, taskfile, dbstr))

        outfile = "{}.slurm".format(outroot)
        nersc_job(jobname, outfile, logroot, desisetup, coms, machine,
                  queue, fullnodes, nodeprocs, fullruntime, openmp=openmp,
                  multiproc=multiproc, shifterimg=shifterimg, debug=debug)
        scriptfiles.append(outfile)

    return scriptfiles
