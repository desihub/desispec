#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.plan
======================

Tools for planning pipeline runs.
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
    weighted_partition, stdouterr_redirected, use_mpi)

from .prod import task_read, task_write


def nersc_machine(name, queue):
    """Return the properties of the specified NERSC host.

    Args:
        name (str): the name of the host.  Allowed values are:  cori-haswell
            and cori-knl.
        queue (str): the queue on the machine (regular, debug, etc)

    Returns:
        dict: properties of this machine.

    """
    props = dict()
    if name == "cori-haswell":
        props["sbatch"] = [
            "#SBATCH --constraint=haswell"
        ]
        props["nodecores"] = 32
        props["corecpus"] = 2
        props["nodemem"] = 125.0
        props["timefactor"] = 1.0
        props["startup"] = 2.0
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
        elif queue == "realtime":
            props["maxnodes"] = 10
            props["maxtime"] = 720
            props["submitlimit"] = 5000
            props["sbatch"].append("#SBATCH --exclusive")
            props["sbatch"].append("#SBATCH --qos=realtime")
        else:
            raise RuntimeError("Unknown {} queue '{}'".format(name, queue))
    elif name == "cori-knl":
        props["sbatch"] = [
            "#SBATCH --constraint=knl,quad,cache",
            "#SBATCH --core-spec=4"
        ]
        props["nodecores"] = 64
        props["corecpus"] = 4
        props["nodemem"] = 93.0
        props["timefactor"] = 3.0
        props["startup"] = 2.0
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
        raise RuntimeError("Unknown machine '{}' choice is 'cori-haswell' or 'cori-knl'".format(name))

    return props


def compute_nodes(nworker, taskproc, nodeprocs):
    """Compute number of nodes for the number of workers.

    Args:
        nworker (int):  The number of workers.
        taskproc (int):  The number of processes per task.
        nodeprocs (int):  The number of processes per node.

    Returns:
        (int):  The number of required nodes.

    """
    nds = (nworker * taskproc) // nodeprocs
    if nds * nodeprocs < nworker * taskproc:
        nds += 1
    return nds


def worker_times(tasktimes, workerdist, startup=0.0):
    """Compute the time needed for each worker.

    Args:
        tasktimes (array):  array of individual task times.
        workerdist (list):  List of tuples of indices in taskstimes.
        startup (float):  Startup overhead in minutes for each worker.

    Returns:
        (tuple):  The (worker times, min, max).

    Notes / Examples:
        len(tasktimes) = number of tasks
        len(workerdist) = number of workers
        workerdist[i] = tuple of tasktime indices assigned to worker i
        sum(tasktimes[workerdist[i]]) = expected total time for worker i
    """
    tasktimes = np.asarray(tasktimes)
    workertimes = np.array([startup + np.sum(tasktimes[ii]) for ii in workerdist])
    workermax = np.max(workertimes)
    workermin = np.min(workertimes)
    return workertimes, workermin, workermax


def compute_worker_tasks(tasktype, tasklist, tfactor, nworker,
                         workersize, startup=0.0, db=None, num_nodes=None):
    """Compute the distribution of tasks for specified workers.

    Args:
        tasktype (str):  The task type.
        tasklist (list):  List of tasks, all of type tasktype.
        tfactor (float):  Additional runtime scaling factor.
        nworker (int):  The number of workers.
        workersize (int):  The number of processes in each worker.
        startup (float, optional):  Startup overhead in minutes for each worker.
        db (DataBase, optional): the database to pass to the task runtime
            calculation.
        num_nodes (int, optional): number of nodes over which the workers are distributed

    Returns:
        (tuple):  The (sorted tasks, sorted runtime weights, dist) results
        where dist is the a list of tuples (one per worker) indicating
        the indices of tasks for that worker in the
        returned sorted list of tasks.

    """
    from .tasks.base import task_classes, task_type
    log = get_logger()

    # Run times for each task at this concurrency
    tasktimes = [(x, tfactor * task_classes[tasktype].run_time(
                    x, workersize, db=db)) for x in tasklist]

    # Sort the tasks by runtime to improve the partitioning
    # NOTE: sorting is unnecessary when using weighted_partition instead of
    #       dist_discrete_all, but leaving for now while comparing/debugging
    tasktimes = list(sorted(tasktimes, key=lambda x: x[1]))[::-1]
    mintasktime = tasktimes[-1][1]
    maxtasktime = tasktimes[0][1]
    log.debug("task runtime range = {:.2f} ... {:.2f}".format(mintasktime, maxtasktime))

    # Split the task names and times
    worktasks = [x[0] for x in tasktimes]
    workweights = [x[1] for x in tasktimes]

    # Distribute tasks
    workdist = None
    if len(workweights) == nworker:
        # One task per worker
        workdist = [[i,] for i in range(nworker)]
    else:
        # workdist = dist_discrete_all(workweights, nworker)
        if num_nodes is not None:
            workers_per_node = (nworker + num_nodes - 1 ) // num_nodes
        else:
            workers_per_node = None

        workdist = weighted_partition(workweights, nworker,
            groups_per_node=workers_per_node)

    # Find the runtime for each worker
    workertimes, workermin, workermax = worker_times(
        workweights, workdist, startup=startup)

    log.debug("worker task assignment:")
    log.debug("  0: {:.2f} minutes".format(workertimes[0]))
    log.debug("      first task {}".format(worktasks[workdist[0][0]]))
    log.debug("      last task {}".format(worktasks[workdist[0][-1]]))
    if nworker > 1:
        log.debug("      ...")
        log.debug("  {}: {:.2f} minutes".format(nworker-1, workertimes[-1]))
        log.debug("      first task {}".format(
            worktasks[workdist[nworker-1][0]]))
        log.debug("      last task {}".format(
                worktasks[workdist[nworker-1][-1]]
            )
        )
    log.debug("range of worker times = {:.2f} ... {:.2f}".format(workermin, workermax))

    return (worktasks, workweights, workdist)


def nersc_job_size(tasktype, tasklist, machine, queue, maxtime, maxnodes,
    nodeprocs=None, db=None, balance=False):
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
            e.g. cori-haswell, cori-knl
        queue (str): the nersc queue name, e.g. regular or debug
        maxtime (int): the maximum run time in minutes.
        maxnodes (int): the maximum number of nodes.
        nodeprocs (int): the number of processes per node.  If None, estimate
            this based on the per-process memory needs of the task and the
            machine properties.
        db (DataBase): the database to pass to the task runtime
            calculation.
        balance (bool): if True, change the number of workers to load
            balance the job.

    Returns:
        list:  List of tuples (nodes, nodeprocs, runtime, nworker, workersize,
            tasks) containing one entry per job.  Each entry specifies the
            number of nodes to use, the expected total runtime, number of
            workers, and the list of tasks for that job.

    """
    from .tasks.base import task_classes, task_type
    log = get_logger()

    log.debug("inputs:")
    log.debug("    tasktype = {}".format(tasktype))
    log.debug("    len(tasklist) = {}".format(len(tasklist)))
    log.debug("    machine = {}".format(machine))
    log.debug("    queue = {}".format(queue))
    log.debug("    maxtime = {}".format(maxtime))
    log.debug("    nodeprocs = {}".format(nodeprocs))

    if len(tasklist) == 0:
        raise RuntimeError("List of tasks is empty")

    # Get the machine properties
    hostprops = nersc_machine(machine, queue)
    log.debug("hostprops={}".format(hostprops))

    if maxtime <= 0:
        maxtime = hostprops["maxtime"]
        log.debug("Using default {} {} maxtime={}".format(
            machine, queue, maxtime))
    if maxtime > hostprops["maxtime"]:
        raise RuntimeError("requested max time '{}' is too long for {} "
            "queue '{}'".format(maxtime, machine, queue))

    if maxnodes <= 0:
        maxnodes = hostprops["maxnodes"]
        log.debug("Using default {} {} maxnodes={}".format(
            machine, queue, maxnodes))
    else:
        log.debug("Using user-specified {} {} maxnodes={}".format(
            machine, queue, maxnodes))
    if maxnodes > hostprops["maxnodes"]:
        raise RuntimeError("requested max nodes '{}' is larger than {} "
            "queue '{}' with {} nodes".format(
                maxnodes, machine, queue, hostprops["maxnodes"]))

    coremem = hostprops["nodemem"] / hostprops["nodecores"]

    # Required memory for each task
    taskfullmems = [ (x, task_classes[tasktype].run_max_mem_task(x, db=db))
                for x in tasklist ]

    # The max memory required by any task
    maxtaskmem = np.max([x[1] for x in taskfullmems])
    log.debug("Maximum memory per task = {}".format(maxtaskmem))

    # Required memory for a single process of the task.
    taskprocmems = [ (x, task_classes[tasktype].run_max_mem_proc(x, db=db))
                    for x in tasklist ]
    maxprocmem = np.max([x[1] for x in taskprocmems])
    log.debug("Maximum memory per process = {}".format(maxprocmem))

    maxnodeprocs = hostprops["nodecores"]
    if maxprocmem > 0.0:
        procmem = coremem
        while procmem < maxprocmem:
            maxnodeprocs = maxnodeprocs // 2
            procmem *= 2
        log.debug("Maximum processes per node based on memory requirements = {}"
                  .format(maxnodeprocs))
    else:
        log.debug("Using default max procs per node ({})".format(maxnodeprocs))

    if nodeprocs is None:
        nodeprocs = maxnodeprocs
    else:
        if nodeprocs > maxnodeprocs:
            log.warning(
                "Cannot use {} procs per node (insufficient memory).  Using {} instead.".format(nodeprocs, maxnodeprocs)
            )
            nodeprocs = maxnodeprocs

    if nodeprocs > hostprops["nodecores"]:
        raise RuntimeError("requested procs per node '{}' is more than the "
            "the number of cores per node on {}".format(nodeprocs, machine))

    log.debug("Using {} processes per node".format(nodeprocs))

    # How many nodes are required to achieve the maximum memory of the largest
    # task?
    mintasknodes = 1
    if maxtaskmem > 0.0:
        mintasknodes += int(maxtaskmem / hostprops["nodemem"])

    # Max number of procs to use per task.
    taskproc = task_classes[tasktype].run_max_procs()
    if taskproc == 0:
        # This means that the task is flexible and can use an arbitrary
        # number of processes.  We assign it the number of processes
        # corresponding to the number of nodes and procs per node dictated
        # by the memory requirements.
        taskproc = mintasknodes * nodeprocs

    log.debug("Using {} processes per task".format(taskproc))

    # Number of workers (as large as possible)
    availproc = maxnodes * nodeprocs
    maxworkers = availproc // taskproc
    nworker = maxworkers
    if nworker > len(tasklist):
        nworker = len(tasklist)
    log.debug("Initial number of workers = {}".format(nworker))

    # Number of nodes
    nodes = compute_nodes(nworker, taskproc, nodeprocs)
    log.debug("Required nodes = {}".format(nodes))

    # Estimate the startup cost of each worker as a constant based on the
    # job size.
    startup_scale = nodes // 200
    startup_time = (1.0 + startup_scale) * hostprops["startup"]
    log.debug("Using {} minutes for worker startup time".format(startup_time))

    # Compute the distribution of tasks to these workers
    (worktasks, worktimes, workdist) = compute_worker_tasks(
        tasktype, tasklist, hostprops["timefactor"], nworker, taskproc,
        startup=startup_time, db=db)
    log.debug("Task times range from {} to {} minutes"
              .format(worktimes[0], worktimes[-1]))

    # Compute the times for each worker
    workertimes, workermin, workermax = worker_times(
        worktimes, workdist, startup=startup_time)
    log.debug("Initial worker times range from {} to {} minutes"
              .format(workermin, workermax))

    # Examine the maximum time needed for all workers.  If this is too large
    # for the requested maximum run time, then we need to split the job.
    # If we have a single job, then we optionally load balance by reducing
    # the job size and extending the run time.

    final = list()

    if workermax > maxtime:
        # We must split the job.  The tasks are already sorted from large to
        # small.  To decide where to split, we accumulate tasks unti we
        # get to the walltime threshold.
        log.debug(
            "Max worker time ({}) is larger than maximum allowed time ({})"
            .format(workermax, maxtime)
        )
        log.debug("Splitting job")
        maxminutes = maxtime * nworker
        jobminutes = startup_time * nworker
        jobtasks = list()
        jindx = 0
        for tsk, tsktime in zip(worktasks, worktimes):
            if jobminutes + tsktime > maxminutes:
                # Close out this job.  We pass the list of tasks through
                # this calculation function to ensure that everything matches
                # the same calculation that will be done at runtime.
                (jobworktasks, jobworktimes, jobworkdist) = \
                    compute_worker_tasks(
                        tasktype, jobtasks, hostprops["timefactor"],
                        nworker, taskproc, startup=startup_time, db=db)
                jobworkertimes, jobworkermin, jobworkermax = worker_times(
                    jobworktimes, jobworkdist, startup=startup_time)
                log.debug(
                    "Split job {} has {} tasks and max time {}"
                    .format(jindx, len(jobworktasks), jobworkermax)
                )
                final.append(
                    (nodes, nodeprocs, jobworkermax, nworker, taskproc,
                     jobworktasks)
                )
                jindx += 1
            else:
                # Accumulate task to this job
                jobtasks.append(tsk)

        # Close out any remaining job
        if len(jobtasks) > 0:
            (jobworktasks, jobworktimes, jobworkdist) = \
                compute_worker_tasks(
                    tasktype, jobtasks, hostprops["timefactor"],
                    nworker, taskproc, startup=startup_time, db=db)
            jobworkertimes, jobworkermin, jobworkermax = worker_times(
                jobworktimes, jobworkdist, startup=startup_time)
            log.debug(
                "Split job {} has {} tasks and max time {}"
                .format(jindx, len(jobworktasks), jobworkermax)
            )
            final.append(
                (nodes, nodeprocs, jobworkermax, nworker, taskproc,
                 jobworktasks)
            )
    elif balance:
        log.debug("Checking for load imbalance as requested")
        # We are load balancing a single job
        while workermax > 1.5 * workermin:
            # pretty bad imbalance...
            if (nworker > 2) and (workermax < 0.5 * maxtime):
                # We don't want to go lower than 2 workers, since that
                # allows one worker to do the "big" task and the other
                # worker to do everything else.  We also can double the
                # runtime if it will exceed our maximum.
                nworker = nworker // 2
                log.debug(
                    "Job is imbalanced, reducing workers to {}"
                    .format(nworker)
                )
                # Recompute job sizes
                nodes = compute_nodes(nworker, taskproc, nodeprocs)
                log.debug("Number of nodes now = {}".format(nodes))
                (worktasks, worktimes, workdist) = compute_worker_tasks(
                    tasktype, tasklist, hostprops["timefactor"], nworker,
                    taskproc, startup=startup_time, db=db)
                workertimes, workermin, workermax = worker_times(
                    worktimes, workdist, startup=startup_time)
                log.debug("Worker times range from {} to {} minutes"
                          .format(workermin, workermax))
            else:
                log.debug(
                    "Job is imbalanced, but there are too few workers or the runtime is already too long."
                )
                break
        log.debug(
            "Adding job with {} tasks, {} workers, and max time {} on {} nodes"
            .format(len(worktasks), nworker, workermax, nodes)
        )
        final.append((nodes, nodeprocs, workermax, nworker, taskproc,
                      worktasks))
    else:
        # We just have one job
        log.debug(
            "Adding job with {} tasks, {} workers, and max time {} on {} nodes"
            .format(len(worktasks), nworker, workermax, nodes)
        )
        final.append((nodes, nodeprocs, workermax, nworker, taskproc,
                      worktasks))

    return final
