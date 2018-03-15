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

from ..parallel import (dist_uniform, dist_discrete,
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
            props["sbatch"].append("#SBATCH --partition=debug")
        elif queue == "regular":
            props["maxnodes"] = 2048
            props["maxtime"] = 12 * 60
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
            props["sbatch"].append("#SBATCH --partition=debug")
        elif queue == "regular":
            props["maxnodes"] = 512
            props["maxtime"] = 12 * 60
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
            props["sbatch"].append("#SBATCH --partition=debug")
        elif queue == "regular":
            props["maxnodes"] = 4096
            props["maxtime"] = 12 * 60
            props["sbatch"].append("#SBATCH --partition=regular")
        else:
            raise RuntimeError("Unknown {} queue '{}'".format(name, queue))
    else:
        raise RuntimeError("Unknown machine '{}'".format(name))

    return props


def shell_job(path, logroot, desisetup, commands, comrun="", mpiprocs=1,
              openmp=1,debug=False):
    with open(path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("now=`date +%Y%m%d-%H:%M:%S`\n")
        f.write("export STARTTIME=${now}\n")
        f.write("log={}_${{now}}.log\n\n".format(logroot))
        f.write("source {}\n\n".format(desisetup))

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
        f.write("#SBATCH --output={}_%j.log\n".format(logroot))

        f.write("echo Starting slurm script at `date`\n\n")
        f.write("source {}\n\n".format(desisetup))

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

        f.write("now=`date +%Y%m%d-%H:%M:%S`\n")
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
            f.write("export STARTTIME=`date +%Y%m%d-%H:%M:%S`\n")
            f.write("echo ${{run}} ${{app}} {}\n".format(" ".join(comlist)))
            f.write("time ${{run}} ${{app}} {} >>${{log}} 2>&1".format(" ".join(comlist)))
            if multisrun:
                f.write(" &")
            f.write("\n\n")
        if multisrun:
            f.write("wait\n\n")

        f.write("echo done with slurm script at `date`\n")

    return


def nersc_job_size(tasktype, tasklist, machine, queue, runtime,
    nodeprocs=None, db=None):
    """Compute the NERSC job parameters based on constraints.

    Given the list of tasks, query their estimated runtimes and determine
    the best job size to use.  If the job is too big to fit the constraints
    then split up the tasks into multiple jobs.

    Args:


    Returns:
        list:  List of tuples (nodes, tasks) containing one entry per job.
            Each entry specifies the number of nodes to use and the list of
            tasks for that job.

    """
    from .tasks.base import task_classes, task_type

    # Get the machine properties
    hostprops = nersc_machine(machine, queue)

    if runtime > hostprops["maxtime"]:
        raise RuntimeError("requested time '{}' is too long for {} "
            "queue '{}'".format(runtime, machine, queue))

    if nodeprocs is None:
        nodeprocs = hostprops["nodecores"]

    if nodeprocs > hostprops["nodecores"]:
        raise RuntimeError("requested procs per node '{}' is more than the "
            "the number of cores per node on {}".format(nodeprocs, machine))

    # Max number of procs to use per task.
    taskproc = task_classes[tasktype].run_max_procs(nodeprocs)

    # Run times for each task at this concurrency
    tasktimes = [ (x, task_classes[tasktype].run_time(x, nodeprocs,
        db=db)) for x in tasklist ]

    # We want to sort the times so that we can use a simple greedy
    # "first-fit" algorithm.

    tasktimes = list(sorted(tasktimes, key=lambda x: x[1]))[::-1]

    mintime = tasktimes[-1][1]
    maxtime = tasktimes[0][1]

    if maxtime > runtime:
        raise RuntimeError("The longest task ({} minutes) exceeds the "
            " requested max runtime ({} minutes)".format(maxtime, runtime))

    workertasks = list()
    workersum = list()

    workertasks.append([ tasktimes[0] ])
    workersum.append(tasktimes[0][1])

    for t in tasktimes[1:]:
        if workersum[-1] + t[1] > runtime:
            # move to the next worker
            workertasks.append([ t ])
            workersum.append(t[1])
        else:
            # add to this worker
            workertasks[-1].append(t)
            workersum[-1] += t[1]

    # Compute how many nodes we need
    nworker = len(workersum)
    totalprocs = nworker * taskproc
    totalnodes = totalprocs // nodeprocs
    if totalprocs % nodeprocs != 0:
        totalnodes += 1

    ret = list()

    # Do we need to split this into multiple jobs?
    if totalnodes > hostprops["maxnodes"]:
        # yes...
        maxprocs = hostprops["maxnodes"] * nodeprocs
        raise NotImplementedError("Job splitting not yet implemented")

    else:
        # no...
        outtasks = list()
        for w in workertasks:
            outtasks.extend([ x[0] for x in w ])
        ret.append( (totalnodes, outtasks) )

    return ret


def batch_shell(tasktype, tasklist, outroot, logroot, mpirun="", mpiprocs=1,
    openmp=1, db=None):
    """Generate slurm script(s) to process a list of tasks.

    Given a list of tasks and some constraints about the machine,
    run time, etc, generate shell scripts.

    """
    from .tasks.base import task_classes, task_type

    # Get the location of the setup script from the production root.

    proddir = os.path.abspath(io.specprod_root())
    desisetup = os.path.abspath(os.path.join(proddir, "setup.sh"))

    dbstr = ""
    if db is None:
        dbstr = "--nodb"

    taskfile = "{}.tasks".format(outroot)
    task_write(taskfile, tasklist)

    coms = None
    if mpiprocs > 1:
        coms = [ "desi_pipe_exec_mpi --tasktype {} --taskfile {} {}"\
            .format(tasktype, taskfile, dbstr) ]
    else:
        coms = [ "desi_pipe_exec --tasktype {} --taskfile {} {}"\
            .format(tasktype, taskfile, dbstr) ]

    outfile = "{}.sh".format(outroot)

    shell_job(outfile, logroot, desisetup, coms, comrun=mpirun,
        mpiprocs=mpiprocs, openmp=openmp)

    return [ outfile ]


def batch_nersc(tasktype, tasklist, outroot, logroot, jobname, machine, queue,
    runtime, nodeprocs=None, openmp=False, multiproc=False, db=None,
                shifterimg=None,debug=False):
    """Generate slurm script(s) to process a list of tasks.

    Given a list of tasks and some constraints about the machine,
    run time, etc, generate slurm scripts for use at NERSC.

    """
    from .tasks.base import task_classes, task_type

    # Get the location of the setup script from the production root.

    proddir = os.path.abspath(io.specprod_root())
    desisetup = os.path.abspath(os.path.join(proddir, "setup.sh"))

    # Compute job size.

    joblist = nersc_job_size(tasktype, tasklist, machine, queue, runtime,
        nodeprocs=nodeprocs)

    dbstr = ""
    if db is None:
        dbstr = "--nodb"

    scriptfiles = list()

    jindx = 0
    suffix = True
    if len(joblist) == 1:
        suffix = False
    for (nodes, tasks) in joblist:
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
        nersc_job(jobname, outfile, joblogroot, desisetup, coms, machine, queue,
            nodes, nodeprocs, runtime, openmp=openmp, multiproc=multiproc,
                  shifterimg=shifterimg,debug=debug)
        scriptfiles.append(outfile)
        jindx += 1

    return scriptfiles


    #
    # def step_props(first, last, specs, night):
    #     """
    #     Internal helper function used only in the desi_pipe
    #     script to build names of scripts and directories.
    #     """
    #     specstr = ""
    #     if specs is not None:
    #         specstr = " --spectrographs {}".format(",".join([ "{}".format(x) for x in specs ]))
    #
    #     rundir = io.get_pipe_rundir()
    #     scrdir = os.path.join(rundir, io.get_pipe_scriptdir())
    #     logdir = os.path.join(rundir, io.get_pipe_logdir())
    #
    #     nstr = ""
    #     scrstr = ""
    #     if night is not None:
    #         nstr = " --nights {}".format(night)
    #         scrstr = "{}".format(night)
    #
    #     stepstr = ""
    #     jobname = ""
    #     if first == last:
    #         stepstr = first
    #         if scrstr != "":
    #             stepstr = "{}_{}".format(first, scrstr)
    #         jobname = first
    #     else:
    #         stepstr = "{}-{}".format(first, last)
    #         if scrstr != "":
    #             stepstr = "{}-{}_{}".format(first, last, scrstr)
    #         jobname = "{}_{}".format(first, last)
    #
    #     return (rundir, scrdir, logdir, specstr, nstr, scrstr, stepstr, jobname)
    #
    #
    # def compute_step(img, specdata, specredux, desiroot, setupfile,
    #     first, last, specs, night, ntask, taskproc, tasktime, shell_mpi_run,
    #     shell_maxcores, shell_threads, nersc_host, nersc_maxnodes,
    #     nersc_nodecores, nersc_threads, nersc_mp, nersc_queue_thresh,
    #     queue="debug"):
    #     """
    #     Internal helper function used only in the desi_pipe script to
    #     generate the job scripts.
    #     """
    #
    #     (rundir, scrdir, logdir, specstr, nstr, scrstr, stepstr, jobname) = \
    #         step_props(first, last, specs, night)
    #
    #     totproc = ntask * taskproc
    #
    #     shell_maxprocs = shell_maxcores // shell_threads
    #     shell_procs = shell_maxprocs
    #     if totproc < shell_procs:
    #         shell_procs = totproc
    #
    #     ntdir = scrdir
    #     logntdir = logdir
    #     if night is not None:
    #         ntdir = os.path.join(scrdir, night)
    #         if not os.path.isdir(ntdir):
    #             os.makedirs(ntdir)
    #         logntdir = os.path.join(logdir, night)
    #         if not os.path.isdir(logntdir):
    #             os.makedirs(logntdir)
    #
    #     shell_path = os.path.join(ntdir, "{}.sh".format(stepstr))
    #     shell_log = os.path.join(logntdir, "{}_sh".format(stepstr))
    #
    #     #- no MPI for shell job version so that it can be run from interactive node
    #     com = None
    #     if shell_maxcores == 1:
    #         com = ["desi_pipe_run --first {} --last {}{}{}".format(first, last, specstr, nstr)]
    #     else:
    #         com = ["desi_pipe_run_mpi --first {} --last {}{}{}".format(first, last, specstr, nstr)]
    #
    #     pipe.shell_job(shell_path, shell_log, setupfile, com, comrun=shell_mpi_run,
    #         mpiprocs=shell_procs, threads=shell_threads)
    #
    #     # Compute job size for NERSC runs
    #
    #     core_per_proc = 1
    #     if nersc_threads > 1:
    #         core_per_proc = nersc_threads
    #     elif nersc_mp > 1:
    #         core_per_proc = nersc_mp
    #
    #     nodeproc = nersc_nodecores // core_per_proc
    #
    #     (nodes, procs, time) = job_size(ntask, taskproc, tasktime, nodeproc, nersc_maxnodes)
    #
    #     if nodes > nersc_queue_thresh and queue == "debug":
    #         print("{} nodes too big for debug queue; switching to regular".format(nodes))
    #         queue = "regular"
    #
    #     if time > 30 and queue == "debug":
    #         print("{} minutes too big for debug queue; switching to regular".format(time))
    #         queue = "regular"
    #
    #     com = ["desi_pipe_run_mpi --first {} --last {}{}{}".format(first, last, specstr, nstr)]
    #
    #     # write normal slurm script
    #
    #     nersc_path = os.path.join(ntdir, "{}.slurm".format(stepstr))
    #     nersc_log = os.path.join(logntdir, "{}_slurm".format(stepstr))
    #
    #     pipe.nersc_job(nersc_host, nersc_path, nersc_log, setupfile, com,
    #         nodes=nodes, nodeproc=nodeproc, minutes=time, multisrun=False,
    #         openmp=(nersc_threads > 1), multiproc=(nersc_mp > 1), queue=queue,
    #         jobname=jobname)
    #
    #     nersc_shifter_path = ""
    #     if img is not None:
    #         # write shifter slurm script
    #
    #         nersc_shifter_path = os.path.join(ntdir,
    #             "{}_shifter.slurm".format(stepstr))
    #         nersc_shifter_log = os.path.join(logntdir,
    #             "{}_shifter".format(stepstr))
    #
    #         pipe.nersc_shifter_job(nersc_host, nersc_shifter_path, img, specdata,
    #             specredux, desiroot, nersc_shifter_log, setupfile, com,
    #             nodes=nodes, nodeproc=nodeproc, minutes=time, multisrun=False,
    #             openmp=(nersc_threads > 1), multiproc=(nersc_mp > 1), queue=queue,
    #             jobname=jobname)
    #
    #     return (shell_path, nersc_path, nersc_shifter_path)
