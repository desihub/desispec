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
import sys

import numpy as np

from desiutil.log import get_logger

from .. import io

from ..parallel import (dist_uniform, dist_discrete, dist_discrete_all,
    stdouterr_redirected, use_mpi)

from .prod import load_prod

from .db import check_tasks


def run_task(name, opts, comm=None, logfile=None, db=None):
    """Run a single task.

    Based on the name of the task, call the appropriate run function for that
    task.  Log output to the specified file.  Run using the specified MPI
    communicator and optionally update state to the specified database.

    Note:  This function DOES NOT check the database or filesystem to see if
    the task has been completed or if its dependencies exist.  It assumes that
    some higher-level code has done that if necessary.

    Args:
        name (str): the name of this task.
        opts (dict): options to use for this task.
        comm (mpi4py.MPI.Comm): optional MPI communicator.
        logfile (str): output log file.  If None, do not redirect output to a
            file.
        db (pipeline.db.DB): The optional database to update.

    Returns:
        int: the total number of processes that failed.

    """
    from .tasks.base import task_classes, task_type
    log = get_logger()

    ttype = task_type(name)

    nproc = 1
    rank = 0
    if comm is not None:
        nproc = comm.size
        rank = comm.rank

    if rank == 0:
        if (logfile is not None) and os.path.isfile(logfile):
            os.remove(logfile)
        # Mark task as in progress
        if db is not None:
            task_classes[ttype].state_set(db=db,name=name,state="running")

    failcount = 0
    if logfile is None:
        # No redirection
        if db is None:
            failcount = task_classes[ttype].run(name, opts, comm=comm)
        else:
            failcount = task_classes[ttype].run_and_update(db, name, opts,
                comm=comm)
    else:
        with stdouterr_redirected(to=logfile, comm=comm):
            if db is None:
                failcount = task_classes[ttype].run(name, opts, comm=comm)
            else:
                failcount = task_classes[ttype].run_and_update(db, name, opts,
                    comm=comm)

    return failcount


def run_task_simple(name, opts, comm=None):
    """Run a single task with no DB or log redirection.

    This a wrapper around run_task() for use without a database and with no
    log redirection.  See documentation for that function.

    Args:
        name (str): the name of this task.
        opts (dict): options to use for this task.
        comm (mpi4py.MPI.Comm): optional MPI communicator.

    Returns:
        int: the total number of processes that failed.

    """
    return run_task(name, opts, comm=comm, logfile=None, db=None)


def run_task_list(tasktype, tasklist, opts, comm=None, db=None, force=False):
    """Run a collection of tasks of the same type.

    This function requires that the DESI environment variables are set to
    point to the current production directory.

    This function first takes the communicator and uses the maximum processes
    per task to split the communicator and form groups of processes of
    the desired size.  It then takes the list of tasks and uses their relative
    run time estimates to assign tasks to the process groups.  Each process
    group loops over its assigned tasks.

    If the database is not specified, no state tracking will be done and the
    filesystem will be checked as needed to determine the current state.

    Only tasks that are ready to run (based on the filesystem checks or the
    database) will actually be attempted.

    Args:
        tasktype (str): the pipeline step to process.
        tasklist (list): the list of tasks.  All tasks should be of type
            "tasktype" above.
        opts (dict): the global options (for example, as read from the
            production options.yaml file).
        comm (mpi4py.Comm): the full communicator to use for whole set of tasks.
        db (pipeline.db.DB): The optional database to update.
        force (bool): If True, ignore database and filesystem state and just
            run the tasks regardless.

    Returns:
        tuple: the number of ready tasks, and the number that failed.

    """
    from .tasks.base import task_classes, task_type
    log = get_logger()

    nproc = 1
    rank = 0
    if comm is not None:
        nproc = comm.size
        rank = comm.rank

    # Compute the number of processes that share a node.

    procs_per_node = 1
    if comm is not None:
        import mpi4py.MPI as MPI
        nodecomm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0)
        procs_per_node = nodecomm.size

    # Get the options for this task type.

    options = opts[tasktype]

    # Get the tasks that still need to be done.

    runtasks = None
    ntask = None
    ndone = None

    if rank == 0:
        if force:
            # Run everything
            runtasks = tasklist
            ntask = len(runtasks)
            ndone = 0
        else:
            # Actually check which things need to be run.
            states = check_tasks(tasklist, db=db)
            runtasks = [ x for x in tasklist if states[x] == "ready" ]
            ntask = len(runtasks)
            ndone = len([ x for x in tasklist if states[x] == "done" ])

        log.debug("Number of {} tasks ready to run is {} (total is {})".format(tasktype,len(runtasks),len(tasklist)))

    if comm is not None:
        runtasks = comm.bcast(runtasks, root=0)
        ntask = comm.bcast(ntask, root=0)
        ndone = comm.bcast(ndone, root=0)

    # Get the weights for each task.  Since this might trigger DB lookups, only
    # the root process does this.

    weights = None
    if rank == 0:
        weights = [ task_classes[tasktype].run_time(x, procs_per_node, db=db) \
            for x in runtasks ]
    if comm is not None:
        weights = comm.bcast(weights, root=0)

    # Now every process has the full list of tasks.  Get the max
    # number of processes for this task type

    taskproc = task_classes[tasktype].run_max_procs(procs_per_node)
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
            if ntask <= ngroup:
                # distribute uniform in this case
                group_firsttask, group_ntask = dist_uniform(ntask, ngroup,
                    group)
            else:
                group_firsttask, group_ntask = dist_discrete(weights, ngroup,
                    group)

    # every group goes and does its tasks...

    rundir = io.get_pipe_rundir()
    logdir = os.path.join(rundir, io.get_pipe_logdir())

    failcount = 0
    group_failcount = 0

    if group_ntask > 0:

        log.debug("rank #{} Group number of task {}, first task {}".format(rank,group_ntask,group_firsttask))

        for t in range(group_firsttask, group_firsttask + group_ntask):
            # For this task, determine the output log file.  If the task has
            # the "night" key in its name, then use that subdirectory.
            # Otherwise, if it has the "pixel" key, use the appropriate
            # subdirectory.
            tt = task_type(runtasks[t])
            fields = task_classes[tt].name_split(runtasks[t])

            tasklog = None
            if "night" in fields:
                tasklogdir = os.path.join(logdir, io.get_pipe_nightdir(),
                                          "{:08d}".format(fields["night"]))
                # (this directory should have been made during the prod update)
                tasklog = os.path.join(tasklogdir,
                    "{}.log".format(runtasks[t]))
            elif "pixel" in fields:
                tasklogdir = os.path.join(logdir, "healpix",
                    io.healpix_subdirectory(fields["nside"],fields["pixel"]))
                # When creating this directory, there MIGHT be conflicts from
                # multiple processes working on pixels in the same
                # sub-directories...
                try :
                    if not os.path.isdir(os.path.dirname(tasklogdir)):
                        os.makedirs(os.path.dirname(tasklogdir))
                except FileExistsError:
                    pass
                try :
                    if not os.path.isdir(tasklogdir):
                        os.makedirs(tasklogdir)
                except FileExistsError:
                    pass
                tasklog = os.path.join(tasklogdir,
                    "{}.log".format(runtasks[t]))

            failedprocs = run_task(runtasks[t], options, comm=comm_group,
                logfile=tasklog, db=db)

            if failedprocs > 1:
                group_failcount += 1

    failcount = group_failcount

    # Every process in each group has the fail count for the tasks assigned to
    # its group.  To get the total onto all processes, we just have to do an
    # allreduce across the rank communicator.

    if comm_rank is not None:
        failcount = comm_rank.allreduce(failcount)

    if db is not None and rank == 0 :
        # postprocess the successful tasks

        log.debug("postprocess the successful tasks")

        states = db.get_states(runtasks)

        log.debug("states={}".format(states))
        log.debug("runtasks={}".format(runtasks))


        with db.cursor() as cur :
            for name in runtasks :
                if states[name] == "done" :
                    log.debug("postprocessing {}".format(name))
                    task_classes[tasktype].postprocessing(db,name,cur)


    log.debug("rank #{} done".format(rank))

    return ntask, ndone, failcount


def run_task_list_db(tasktype, tasklist, comm=None):
    """Run a list of tasks using the pipeline DB and options.

    This is a wrapper around run_task_list which uses the production database
    and global options file.

    Args:
        tasktype (str): the pipeline step to process.
        tasklist (list): the list of tasks.  All tasks should be of type
            "tasktype" above.
        comm (mpi4py.Comm): the full communicator to use for whole set of tasks.

    Returns:
        tuple: the number of ready tasks, and the number that failed.

    """
    (db, opts) = load_prod("w")
    return run_task_list(tasktype, tasklist, opts, comm=comm, db=db)


def dry_run(tasktype, tasklist, opts, procs, procs_per_node, db=None,
    launch="mpirun -np", force=False):
    """Compute the distribution of tasks and equivalent commands.

    This function takes similar arguments as run_task_list() except simulates
    the data distribution and commands that would be run if given the specified
    number of processes and processes per node.

    This can be used to debug issues with the runtime concurrency or the
    actual options that will be passed to the underying main() entry points
    for each task.

    This function requires that the DESI environment variables are set to
    point to the current production directory.

    Only tasks that are ready to run (based on the filesystem checks or the
    database) will actually be attempted.

    NOTE: Since this function is just informative and for interactive use,
    we print information directly to STDOUT rather than logging.

    Args:
        tasktype (str): the pipeline step to process.
        tasklist (list): the list of tasks.  All tasks should be of type
            "tasktype" above.
        opts (dict): the global options (for example, as read from the
            production options.yaml file).
        procs (int): the number of processes to simulate.
        procs_per_node (int): the number of processes per node to simulate.
        db (pipeline.db.DB): The optional database to update.
        launch (str): The launching command for a job.  This is just a
            convenience and prepended to each command before the number of
            processes.
        force (bool): If True, ignore database and filesystem state and just
            run the tasks regardless.

    Returns:
        Nothing.

    """
    from .tasks.base import task_classes, task_type

    prefix = "DRYRUN:  "

    # Get the options for this task type.

    options = dict()
    if tasktype in opts:
        options = opts[tasktype]

    # Get the tasks that still need to be done.

    runtasks = None

    if force:
        # Run everything
        runtasks = tasklist
    else:
        # Actually check which things need to be run.
        states = check_tasks(tasklist, db=db)
        runtasks = [ x for x in tasklist if (states[x] == "ready") ]

    ntask = len(runtasks)
    print("{}{} tasks out of {} are ready to run (or be re-run)".format(prefix,
        ntask, len(tasklist)))
    sys.stdout.flush()

    # Get the weights for each task.

    weights = [ task_classes[tasktype].run_time(x, procs_per_node, db=db) \
            for x in runtasks ]

    # Get the max number of processes for this task type

    taskproc = task_classes[tasktype].run_max_procs(procs_per_node)
    if taskproc > procs:
        print("{}task type '{}' can use {} processes per task.  Limiting "
            " this to {} as requested".format(prefix, tasktype, taskproc,
            procs))
        sys.stdout.flush()
        taskproc = procs

    # If we have multiple processes for each task, create groups

    ngroup = procs
    if taskproc > 1:
        ngroup = int(procs / taskproc)

    print("{}using {} groups of {} processes each".format(prefix,
        ngroup, taskproc))
    sys.stdout.flush()

    if ngroup * taskproc < procs:
        print("{}{} processes remain and will sit idle".format(prefix,
            (procs - ngroup * taskproc)))
        sys.stdout.flush()

    # Now we divide up the tasks among the groups of processes as
    # equally as possible.

    group_firsttask = None
    group_ntask = None

    if ntask <= ngroup:
        group_firsttask = [ x for x in range(ntask) ]
        group_ntask = [ 1 for x in range(ntask) ]
        if ntask < ngroup:
            group_firsttask.extend([ 0 for x in range(ngroup - ntask) ])
            group_ntask.extend([ 0 for x in range(ngroup - ntask) ])
    else:
        dist = dist_discrete_all(weights, ngroup)
        group_firsttask = [ x[0] for x in dist ]
        group_ntask = [ x[1] for x in dist ]

    rundir = io.get_pipe_rundir()
    logdir = os.path.join(rundir, io.get_pipe_logdir())

    maxruntime = 0
    print("{}".format(prefix))
    sys.stdout.flush()

    for g in range(ngroup):
        first = group_firsttask[g]
        nt = group_ntask[g]
        if nt == 0:
            continue
        gruntime = np.sum(weights[first:first+nt])
        if gruntime > maxruntime:
            maxruntime = gruntime
        print("{}group {} estimated runtime is {} minutes".format(prefix,
            g, gruntime))
        sys.stdout.flush()
        for t in range(first, first + nt):
            # For this task, determine the output log file.  If the task has
            # the "night" key in its name, then use that subdirectory.
            # Otherwise, if it has the "pixel" key, use the appropriate
            # subdirectory.
            tt = task_type(runtasks[t])
            fields = task_classes[tt].name_split(runtasks[t])

            tasklog = None
            if "night" in fields:
                tasklogdir = os.path.join(logdir, io.get_pipe_nightdir(),
                    "{:08d}".format(fields["night"]))
                tasklog = os.path.join(tasklogdir,
                    "{}.log".format(runtasks[t]))
            elif "pixel" in fields:
                tasklogdir = os.path.join(logdir,
                    io.healpix_subdirectory(fields["nside"],fields["pixel"]))
                tasklog = os.path.join(tasklogdir,
                    "{}.log".format(runtasks[t]))

            com = task_classes[tt].run_cli(runtasks[t], options, taskproc,
                                           launch=launch, log=tasklog, db=db) # need to db for some tasks

            print("{}  {}".format(prefix, com))
            sys.stdout.flush()

        print("{}".format(prefix))
        sys.stdout.flush()

    print("{}Total estimated runtime is {} minutes".format(prefix,
        maxruntime))
    sys.stdout.flush()

    return
