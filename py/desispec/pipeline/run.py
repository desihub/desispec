#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.run
=========================

Tools for running the pipeline.
"""

from __future__ import absolute_import, division, print_function

import os
import stat
import errno
import sys
import re
import pickle
import copy
import traceback
import time
import logging

import numpy as np

from desiutil.log import get_logger
from .. import io
from ..parallel import (dist_uniform, dist_discrete,
    stdouterr_redirected)

from .common import *
from .graph import *

from .task import get_worker
from .state import graph_db_check, graph_db_write
from .plan import load_prod


def run_step(step, grph, opts, comm=None):
    """
    Run a whole single step of the pipeline.

    This function first takes the communicator and the maximum processes
    per task and splits the communicator to form groups of processes of
    the desired size.  It then takes the full dependency graph and extracts
    all the tasks for a given step.  These tasks are then distributed among
    the groups of processes.

    Each process group loops over its assigned tasks.  For each task, it
    redirects stdout/stderr to a per-task file and calls run_task().  If
    any process in the group throws an exception, then the traceback and
    all information (graph and options) needed to re-run the task are written
    to disk.

    After all process groups have finished, the state of the full graph is
    merged from all processes.  This way a failure of one process on one task
    will be propagated as a failed task to all processes.

    Args:
        step (str): the pipeline step to process.
        grph (dict): the dependency graph.
        opts (dict): the global options.
        comm (mpi4py.Comm): the full communicator to use for whole step.

    Returns:
        Nothing.
    """
    log = get_logger()

    nproc = 1
    rank = 0
    if comm is not None:
        nproc = comm.size
        rank = comm.rank

    # Instantiate the worker for this step

    workername = opts["{}_worker".format(step)]
    workeropts = opts["{}_worker_opts".format(step)]
    worker = get_worker(step, workername, workeropts)

    # Get the options for this step.

    options = opts[step]

    # Get the tasks that need to be done for this step.  Mark all completed
    # tasks as done.

    tasks = None
    if rank == 0:
        # For this step, compute all the tasks that we need to do
        alltasks = []
        for name, nd in sorted(grph.items()):
            if nd["type"] == step_file_types[step]:
                alltasks.append(name)

        # For each task, prune if it is finished
        tasks = []
        for t in alltasks:
            if "state" in grph[t]:
                if grph[t]["state"] != "done":
                    tasks.append(t)
            else:
                tasks.append(t)

    if comm is not None:
        tasks = comm.bcast(tasks, root=0)
        grph = comm.bcast(grph, root=0)

    ntask = len(tasks)

    # Now every process has the full list of tasks.  Get the max
    # number of processes from the worker.

    taskproc = worker.max_nproc()
    if taskproc > nproc:
        taskproc = nproc

    # If we have multiple processes for each task, split the communicator.

    comm_group = comm
    comm_rank = None
    group = rank
    ngroup = nproc
    group_rank = 0
    if comm is not None:
        if taskproc > 1:
            ngroup = int(nproc / taskproc)
            group = int(rank / taskproc)
            group_rank = rank % taskproc
            comm_group = comm.Split(color=group, key=group_rank)
            comm_rank = comm.Split(color=group_rank, key=group)
        else:
            comm_group = None
            comm_rank = comm

    # Now we divide up the tasks among the groups of processes as
    # equally as possible.

    group_ntask = 0
    group_firsttask = 0

    if group < ngroup:
        # only assign tasks to whole groups
        if ntask < ngroup:
            if group < ntask:
                group_ntask = 1
                group_firsttask = group
            else:
                group_ntask = 0
        else:
            if step == "zfind":
                # We load balance the bricks across process groups based
                # on the number of targets per brick.  All bricks with
                # < taskproc targets are weighted the same.

                if ntask <= ngroup:
                    # distribute uniform in this case
                    group_firsttask, group_ntask = dist_uniform(ntask, ngroup, group)
                else:
                    bricksizes = [ grph[x]["ntarget"] for x in tasks ]
                    worksizes = [ taskproc if (x < taskproc) else x for x in bricksizes ]

                    if rank == 0:
                        log.debug("zfind {} groups".format(ngroup))
                        workstr = ""
                        for w in worksizes:
                            workstr = "{}{} ".format(workstr, w)
                        log.debug("zfind work sizes = {}".format(workstr))

                    group_firsttask, group_ntask = dist_discrete(worksizes, ngroup, group)

                if group_rank == 0:
                    worksum = np.sum(worksizes[group_firsttask:group_firsttask+group_ntask])
                    log.debug("group {} has tasks {}-{} sum = {}".format(group, group_firsttask, group_firsttask+group_ntask-1, worksum))

            else:
                group_firsttask, group_ntask = dist_uniform(ntask, ngroup, group)

    # Get logging and failure dumping locations

    rundir = io.get_pipe_rundir()

    faildir = os.path.join(rundir, io.get_pipe_faildir())
    logdir = os.path.join(rundir, io.get_pipe_logdir())

    # every group goes and does its tasks...

    failcount = 0
    group_failcount = 0

    if group_ntask > 0:
        for t in range(group_firsttask, group_firsttask + group_ntask):

            # slice out just the graph for this task

            (night, gname) = graph_night_split(tasks[t])

            # check if all inputs exist

            missing = 0
            if group_rank == 0:
                for iname in grph[tasks[t]]['in']:
                    ind = grph[iname]
                    fspath = graph_path(iname)
                    if not os.path.exists(fspath):
                        missing += 1
                        log.error("skipping step {} task {} due to missing input {}".format(step, tasks[t], fspath))
            if comm_group is not None:
                missing = comm_group.bcast(missing, root=0)

            if missing > 0:
                if group_rank == 0:
                    group_failcount += 1
                continue

            nfaildir = os.path.join(faildir, night)
            nlogdir = os.path.join(logdir, night)

            tgraph = graph_slice(grph, names=[tasks[t]], deps=True)
            ffile = os.path.join(nfaildir, "{}_{}.yaml".format(step, tasks[t]))

            # For this task, we will temporarily redirect stdout and stderr
            # to a task-specific log file.

            tasklog = os.path.join(nlogdir, "{}.log".format(gname))
            if group_rank == 0:
                if os.path.isfile(tasklog):
                    os.remove(tasklog)
            if comm_group is not None:
                comm_group.barrier()

            with stdouterr_redirected(to=tasklog, comm=comm_group):
                try:
                    # if the step previously failed, clear that file now
                    if group_rank == 0:
                        if os.path.isfile(ffile):
                            os.remove(ffile)

                    log.debug("running step {} task {} (group {}/{} with {} processes)".format(step, tasks[t], (group+1), ngroup, taskproc))

                    # All processes in comm_group will either return from this or ALL will
                    # raise an exception
                    worker.run(tgraph, tasks[t], options, comm=comm_group)

                    # mark step as done in our group's graph
                    grph[tasks[t]]["state"] = "done"

                except:
                    # The task threw an exception.  We want to dump all information
                    # that will be needed to re-run the task.
                    msg = "FAILED: step {} task {} (group {}/{} with {} processes)".format(step, tasks[t], (group+1), ngroup, taskproc)
                    log.error(msg)
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                    log.error("".join(lines))
                    if group_rank == 0:
                        group_failcount += 1
                        fyml = {}
                        fyml["step"] = step
                        fyml["task"] = tasks[t]
                        fyml["worker"] = workername
                        fyml["worker_opts"] = workeropts
                        fyml["graph"] = tgraph
                        fyml["opts"] = options
                        fyml["procs"] = taskproc
                        log.error("Dumping yaml graph to "+ffile)
                        yaml_write(ffile, fyml)

                    # mark the step as failed in our group"s local graph
                    graph_set_recursive(grph, tasks[t], "fail")

        if comm_group is not None:
            group_failcount = comm_group.bcast(group_failcount, root=0)

    # Now we take the graphs from all groups and merge their states

    failcount = group_failcount

    if comm is not None:
        if group_rank == 0:
            graph_merge(grph, comm=comm_rank)
            failcount = comm_rank.allreduce(failcount)
        if comm_group is not None:
            grph = comm_group.bcast(grph, root=0)
            failcount = comm_group.bcast(failcount, root=0)

    return grph, ntask, failcount


def retry_task(failpath, newopts=None):
    """
    Attempt to re-run a failed task.

    This takes the path to a yaml file containing the information about a
    failed task (such a file is written by run_step() when a task fails).
    This yaml file contains the truncated dependecy graph for the single
    task, as well as the options that were used when running the task.
    It also contains information about the number of processes that were
    being used.

    This function attempts to load mpi4py and use the MPI.COMM_WORLD
    communicator to re-run the task.  If COMM_WORLD has a different number
    of processes than were originally used, a warning is printed.  A
    warning is also printed if the options are being overridden.

    If the task completes successfully, the failed yaml file is deleted.

    Args:
        failpath (str): the path to the failure yaml file.
        newopts (dict): the dictionary of options to use in place of the
            original ones.

    Returns:
        Nothing.
    """

    log = get_logger()

    if not os.path.isfile(failpath):
        raise RuntimeError("failure yaml file {} does not exist".format(failpath))

    fyml = yaml_read(failpath)

    step = fyml["step"]
    name = fyml["task"]
    workername = fyml["workername"]
    workeropts = fyml["worker_opts"]
    grph = fyml["graph"]
    origopts = fyml["opts"]
    nproc = fyml["procs"]

    comm = None
    rank = 0
    nworld = 1

    if nproc > 1:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        nworld = comm.size
        rank = comm.rank
        if nworld != nproc:
            if rank == 0:
                log.warning("WARNING: original task was run with {} processes, re-running with {} instead".format(nproc, nworld))

    opts = origopts
    if newopts is not None:
        log.warning("WARNING: overriding original options")
        opts = newopts

    worker = get_worker(step, workername, workeropts)

    rundir = io.get_pipe_rundir()
    logdir = os.path.join(rundir, io.get_pipe_logdir())
    (night, gname) = graph_night_split(name)

    nlogdir = os.path.join(logdir, night)

    # For this task, we will temporarily redirect stdout and stderr
    # to a task-specific log file.

    tasklog = os.path.join(nlogdir, "{}.log".format(gname))
    if rank == 0:
        if os.path.isfile(tasklog):
            os.remove(tasklog)
    if comm is not None:
        comm.barrier()

    failcount = 0

    with stdouterr_redirected(to=tasklog, comm=comm):
        try:
            log.debug("re-trying step {}, task {} with {} processes".format(step, name, nworld))
            worker.run(grph, name, opts, comm=comm)
        except:
            msg = "FAILED: step {} task {} process {}".format(step, name, rank)
            log.error(msg)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            log.error(''.join(lines))
            if group_rank == 0:
                failcount= 1

    if comm is not None:
        failcount = comm.bcast(failcount, root=0)

    if rank == 0:
        if failcount > 0:
            log.error("{} of {} processes raised an exception".format(failcount, nworld))
        else:
            # success, clear failure file now
            if os.path.isfile(failpath):
                os.remove(failpath)

    return


def run_steps(first, last, spectrographs=None, nightstr=None, comm=None):
    """
    Run multiple sequential pipeline steps.

    This function first takes the communicator and the requested processes
    per task and splits the communicator to form groups of processes of
    the desired size.  It then takes the full dependency graph and extracts
    all the tasks for a given step.  These tasks are then distributed among
    the groups of processes.

    Each process group loops over its assigned tasks.  For each task, it
    redirects stdout/stderr to a per-task file and calls run_task().  If
    any process in the group throws an exception, then the traceback and
    all information (graph and options) needed to re-run the task are written
    to disk.

    After all process groups have finished, the state of the full graph is
    merged from all processes.  This way a failure of one process on one task
    will be propagated as a failed task to all processes.

    Args:
        step (str): the pipeline step to process.
        grph (dict): the dependency graph.
        opts (dict): the global options.
        comm (mpi4py.Comm): the full communicator to use for whole step.

    Returns:
        Nothing.
    """
    log = get_logger()

    rank = 0
    nproc = 1
    if comm is not None:
        rank = comm.rank
        nproc = comm.size

    # get the full graph

    grph = None
    if rank == 0:
        grph = load_prod(nightstr=nightstr, spectrographs=spectrographs)
        graph_db_check(grph)
    if comm is not None:
        grph = comm.bcast(grph, root=0)

    # read run options from disk

    rundir = io.get_pipe_rundir()
    optfile = os.path.join(rundir, "options.yaml")
    opts = None
    if rank == 0:
        opts = yaml_read(optfile)
    if comm is not None:
        opts = comm.bcast(opts, root=0)

    # compute the ordered list of steps to run

    firststep = None
    if first is None:
        firststep = 0
    else:
        s = 0
        for st in step_types:
            if st == first:
                firststep = s
            s += 1

    laststep = None
    if last is None:
        laststep = len(step_types)
    else:
        s = 1
        for st in step_types:
            if st == last:
                laststep = s
            s += 1

    if rank == 0:
        log.info("running steps {} to {}".format(step_types[firststep], step_types[laststep-1]))

    # Mark our steps as in progress

    for st in range(firststep, laststep):
        for name, nd in grph.items():
            if nd["type"] == step_file_types[step_types[st]]:
                if nd["state"] != "done":
                    nd["state"] = "running"

    if rank == 0:
        graph_db_write(grph)

    # Run the steps.  Each step updates the graph in place to track
    # the state of all nodes.

    for st in range(firststep, laststep):

        runfile = None
        if rank == 0:
            log.info("starting step {} at {}".format(step_types[st], time.asctime()))

        grph, ntask, failtask = run_step(step_types[st], grph, opts, comm=comm)

        if rank == 0:
            log.info("completed step {} at {}".format(step_types[st], time.asctime()))
            log.info("  {} total tasks, {} failures".format(ntask, failtask))
            graph_db_write(grph)

        if (ntask > 0) and (ntask == failtask):
            if rank == 0:
                log.info("step {}: all tasks failed, quiting at {}".format(step_types[st], time.asctime()))
            break

        if comm is not None:
            comm.barrier()

    if rank == 0:
        log.info("finished steps {} to {}".format(step_types[firststep], step_types[laststep-1]))

    return


def shell_job(path, logroot, desisetup, commands, comrun="", mpiprocs=1, threads=1):
    with open(path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("now=`date +%Y%m%d-%H:%M:%S`\n")
        f.write("export STARTTIME=${now}\n")
        f.write("log={}_${{now}}.log\n\n".format(logroot))
        f.write("source {}\n\n".format(desisetup))
        f.write("export OMP_NUM_THREADS={}\n\n".format(threads))
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


def nersc_job(path, logroot, desisetup, commands, nodes=1, \
    nodeproc=1, minutes=10, multisrun=False, openmp=False, multiproc=False, \
    queue="debug", jobname="desipipe"):
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
        if queue == "debug":
            f.write("#SBATCH --partition=debug\n")
        else:
            f.write("#SBATCH --partition=regular\n")
        f.write("#SBATCH --account=desi\n")
        f.write("#SBATCH --nodes={}\n".format(totalnodes))
        f.write("#SBATCH --time={}\n".format(timestr))
        f.write("#SBATCH --job-name={}\n".format(jobname))
        f.write("#SBATCH --output={}_%j.log\n".format(logroot))
        f.write("echo Starting slurm script at `date`\n\n")
        f.write("source {}\n\n".format(desisetup))
        f.write("# Set TMPDIR to be on the ramdisk\n")
        f.write("export TMPDIR=/dev/shm\n\n")
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
        runstr = "srun"
        if multiproc:
            runstr = "{} --cpu_bind=no".format(runstr)
        f.write("run=\"{} -n ${{procs}} -N ${{nodes}} -c ${{node_thread}}\"\n\n".format(runstr))
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


def nersc_shifter_job(path, img, specdata, specredux, desiroot, logroot, desisetup, commands, nodes=1, \
    nodeproc=1, minutes=10, multisrun=False, openmp=False, multiproc=False, \
    queue="debug", jobname="desipipe"):

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
        f.write("#SBATCH --image={}\n".format(img))
        if queue == "debug":
            f.write("#SBATCH --partition=debug\n")
        else:
            f.write("#SBATCH --partition=regular\n")
        f.write("#SBATCH --account=desi\n")
        f.write("#SBATCH --nodes={}\n".format(totalnodes))
        f.write("#SBATCH --time={}\n".format(timestr))
        f.write("#SBATCH --job-name={}\n".format(jobname))
        f.write("#SBATCH --output={}_%j.log\n".format(logroot))
        f.write("#SBATCH --volume=\"{}:/desi/root;{}:/desi/spectro_data;{}:/desi/spectro_redux\"\n\n".format(desiroot, specdata, specredux))

        f.write("echo Starting slurm script at `date`\n\n")
        f.write("source {}\n\n".format(desisetup))

        f.write("node_cores=0\n")
        f.write("if [ ${NERSC_HOST} = edison ]; then\n")
        f.write("  module load shifter\n")
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
        runstr = "srun"
        if multiproc:
            runstr = "{} --cpu_bind=no".format(runstr)
        f.write("run=\"{} -n ${{procs}} -N ${{nodes}} -c ${{node_thread}} shifter\"\n\n".format(runstr))
        f.write("now=`date +%Y%m%d-%H:%M:%S`\n")
        f.write("echo \"job datestamp = ${now}\"\n")
        f.write("log={}_${{now}}.log\n\n".format(logroot))
        f.write("envlog={}_${{now}}.env\n".format(logroot))
        f.write("env > ${envlog}\n\n")
        for com in commands:
            comlist = com.split(" ")
            executable = comlist.pop(0)
            f.write("app={}\n".format(executable))
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
