#!/usr/bin/env python
#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-

"""
Run one or more pipeline steps.
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
from desispec.log import get_logger
import desispec.pipeline as pipe


def parse(options=None):

    parser = argparse.ArgumentParser( description="Run steps of the pipeline at a given concurrency." )
    parser.add_argument( "--first", required=False, default=None, help="first step of the pipeline to run" )
    parser.add_argument( "--last", required=False, default=None, help="last step of the pipeline to run")
    parser.add_argument("--nights", required=False, default=None, help="comma separated (YYYYMMDD) or regex pattern")
    parser.add_argument("--spectrographs", required=False, default=None, help="process only this comma-separated list of spectrographs")
    
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

    # raw and production locations

    rawdir = os.path.abspath(io.rawdata_root())
    proddir = os.path.abspath(io.specprod_root())

    if rank == 0:
        log.info("starting at {}".format(time.asctime()))
        log.info("using raw dir {}".format(rawdir))
        log.info("using spectro production dir {}".format(proddir))

    # run it!

    pipe.run_steps(args.first, args.last, spectrographs=args.spectrographs, nightstr=args.nights, comm=comm)
    t2 = datetime.datetime.now()
    
    if rank == 0:
        if "STARTTIME" in os.environ:
            try:
                t0 = datetime.datetime.strptime(os.getenv("STARTTIME"), "%Y%m%d-%H:%M:%S")
                dt = t1 - t0
                minutes, seconds = dt.seconds//60, dt.seconds%60
                log.info("Python startup time: {} min {} sec".format(minutes, seconds))
            except ValueError:
                log.error("unable to parse $STARTTIME={}".format(os.getenv("STARTTIME")))
        else:
            log.info("python startup time unknown since $STARTTIME not set")
        
        dt = t2 - t1
        minutes, seconds = dt.seconds//60, dt.seconds%60
        log.info("Run time: {} min {} sec".format(minutes, seconds))

