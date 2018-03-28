#!/usr/bin/env python
#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-

"""Run one or more pipeline tasks.
"""

from __future__ import absolute_import, division, print_function

import sys
import os
import time
import datetime
import numpy as np
import argparse
import re

import desispec.io as io

from desiutil.log import get_logger

import desispec.pipeline as pipe


def parse(options=None):
    parser = argparse.ArgumentParser(description="Run pipeline tasks of a "
        "single type")
    parser.add_argument("--tasktype", required=True, default=None,
        help="The type of the input tasks.")
    parser.add_argument("--nodb", required=False, default=False,
        action="store_true", help="Do not use the production database.")
    parser.add_argument("--taskfile", required=False, default=None,
        help="Use a file containing the list of tasks.  If not specified, "
        "read list of tasks from STDIN")

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
        tasklist = pipe.prod.task_read(args.taskfile)
    else:
        # Read from STDIN.
        tasklist = list()
        for line in sys.stdin:
            tasklist.append(line.rstrip())

    # run it!

    (db, opts) = pipe.load_prod("w")
    failed = None
    if args.nodb:
        failed = pipe.run_task_list(args.tasktype, tasklist, opts, comm=comm,
            db=None)
    else:
        failed = pipe.run_task_list(args.tasktype, tasklist, opts, comm=comm, db=db)

    t2 = datetime.datetime.now()

    if rank == 0:
        log.info("  {} tasks failed".format(failed))
        dt = t2 - t1
        minutes, seconds = dt.seconds//60, dt.seconds%60
        log.info("Run time: {} min {} sec".format(minutes, seconds))
        sys.stdout.flush()

    return
