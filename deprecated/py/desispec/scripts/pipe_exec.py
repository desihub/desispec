#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.scripts.pipe_exec
==========================

Run one or more pipeline tasks.
"""

from __future__ import absolute_import, division, print_function

import sys
import os
import time
import datetime
import numpy as np
import argparse
import re
import warnings

import desispec.io as io

from desiutil.log import get_logger

import desispec.pipeline as pipe


def parse(options=None):
    parser = argparse.ArgumentParser(description="Run pipeline tasks of a "
        "single type")
    parser.add_argument("--tasktype", required=False, default=None,
        help="The type of the input tasks.")
    parser.add_argument("--force", required=False, default=False,
        action="store_true", help="Run tasks regardless of DB or file state.")
    parser.add_argument("--nodb", required=False, default=False,
        action="store_true", help="Do not use the production database.")
    parser.add_argument("--taskfile", required=False, default=None,
        help="Use a file containing the list of tasks.  If not specified, "
        "use --task or read list of tasks from STDIN")
    parser.add_argument("--task", required=False, default=None,
        help="Single task to run")

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args, comm=None):
    t1 = datetime.datetime.now()

    log = get_logger()

    rank = 0
    nproc = 1
    if comm is not None:
        rank = comm.rank
        nproc = comm.size

    # Check start up time.

    if rank == 0:
        if "STARTTIME" in os.environ:
            try:
                t0 = datetime.datetime.strptime(os.getenv("STARTTIME"), "%Y%m%d-%H%M%S")
                dt = t1 - t0
                minutes, seconds = dt.seconds//60, dt.seconds%60
                log.info("Python startup time: {} min {} sec".format(minutes, seconds))
            except ValueError:
                log.error("unable to parse $STARTTIME={}".format(os.getenv("STARTTIME")))
        else:
            log.info("Python startup time unknown since $STARTTIME not set")
        sys.stdout.flush()

    # raw and production locations

    rawdir = os.path.abspath(io.rawdata_root())
    proddir = os.path.abspath(io.specprod_root())

    if rank == 0:
        log.info("Starting at {}".format(time.asctime()))
        log.info("  Using raw dir {}".format(rawdir))
        log.info("  Using spectro production dir {}".format(proddir))
        sys.stdout.flush()

    # Get task list from disk or from STDIN

    tasklist = None
    if args.taskfile is not None:
        # One process reads the file and broadcasts
        if rank == 0:
            tasklist = pipe.prod.task_read(args.taskfile)
        if comm is not None:
            tasklist = comm.bcast(tasklist, root=0)
    elif args.task is not None:
        tasklist = [args.task,]
    else:
        # Every process has the same STDIN contents.
        tasklist = list()
        for line in sys.stdin:
            tasklist.append(line.rstrip())

    # Do we actually have any tasks?
    if len(tasklist) == 0:
        # warnings.warn("Task list is empty", RuntimeWarning)
        log.error("Task list is empty")
        sys.exit(1)

    # Derive tasktype from tasklist if needed; otherwise filter by tasktype
    tasktypes = [tmp.split('_')[0] for tmp in tasklist]
    if args.tasktype is None:
        if len(set(tasktypes)) == 1:
            args.tasktype = tasktypes[0]
        else:
            log.error("Multiple task types found in input list: {}".format(
                set(tasktypes)))
            sys.exit(1)

    elif len(set(tasktypes)) > 0:
        tasklist = [t for t in tasklist if t.startswith(args.tasktype+'_')]

    # run it!

    (db, opts) = pipe.load_prod("w")

    ntask = len(tasklist)
    ready = None
    done = None
    failed = None

    if args.nodb:
        ready, done, failed = pipe.run_task_list(args.tasktype, tasklist, opts,
                                           comm=comm, db=None, force=args.force)
    else:
        ready, done, failed = pipe.run_task_list(args.tasktype, tasklist, opts,
                                           comm=comm, db=db, force=args.force)

    t2 = datetime.datetime.now()

    if rank == 0:
        log.info("  {} tasks already done, {} tasks were ready, and {} failed".format(done, ready, failed))
        dt = t2 - t1
        minutes, seconds = dt.seconds//60, dt.seconds%60
        log.info("Run time: {} min {} sec".format(minutes, seconds))
        sys.stdout.flush()

    if comm is not None:
        comm.barrier()

    # Did we have any ready tasks that were not already done?
    # Note: if there were no ready tasks, but some were already
    # done, then we want to exit with a "0" error code.  This will
    # allow the calling script to continue with other pipeline steps
    # and / or allow other dependent jobs run.
    if done == 0:
        # nothing is done
        if ready == 0:
            if rank == 0:
                warnings.warn("No tasks were ready or done", RuntimeWarning)
            sys.exit(1)
        if (failed == ready) and (failed > 1) :
            # all tasks failed (and there are more than one)
            if rank == 0:
                warnings.warn("All tasks that were run failed", RuntimeWarning)
            sys.exit(1)
    else:
        # At least some tasks were done- we return zero so that future
        # jobs can run.
        if (ready > 0) and (failed == ready):
            if rank == 0:
                warnings.warn("All tasks that were run failed", RuntimeWarning)
            sys.exit(1)

    return
