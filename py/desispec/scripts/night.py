# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.scripts.night
======================

Entry points for start/update/end night scripts.
"""
from __future__ import (absolute_import, division, print_function,
    unicode_literals)

import sys
import os
import argparse
import subprocess as sp
import time

from desiutil.log import get_logger

from .. import io

from .. import pipeline as pipe


def stage_from_command():
    """
    Extract the processing stage from the executable command.

    Returns:
        str: The extracted stage.

    Raises:
        KeyError: If the extracted string does not match the set of stages.
    """
    stage = os.path.basename(sys.argv[0]).split("_")[1]
    if stage not in ("start", "update", "end"):
        raise KeyError("Command does not match one of the stages!")
    return stage


def parse(stage, options=None):
    """
    Parse command-line options for start/update/end night scripts.

    Args:
        stage (str): The stage of the launch, one of "start", "update", "end".
        options : iterable

    Returns:
        :class:`argparse.Namespace`: The parsed command-line options.
    """
    desc = {"start": "Begin DESI pipeline processing for a particular night.",
            "update": "Run DESI pipeline on new data for a particular night.",
            "end": "Conclude DESI pipeline processing for a particular night."}

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
        description=desc[stage])

    parser.add_argument("--night", required=True, type=str,
        help="Night ID in the form YYYYMMDD.")

    parser.add_argument("--nersc", required=False, default=None,
        help="run jobs on this NERSC system (edison | cori-haswell "
        "| cori-knl)")

    parser.add_argument("--nersc_queue", required=False, default="regular",
        help="use this NERSC queue (debug | regular)")

    parser.add_argument("--nersc_runtime", required=False, type=int,
        default=30, help="Then maximum run time (in minutes) for a single "
        " NERSC job.  If the list of tasks cannot be run in this time,"
        " multiple job scripts will be written")

    parser.add_argument("--nersc_shifter", required=False, default=None,
        help="The shifter image to use for NERSC jobs")

    parser.add_argument("--mpi_procs", required=False, type=int, default=1,
        help="The number of MPI processes to use for non-NERSC shell "
        "scripts (default 1)")

    parser.add_argument("--mpi_run", required=False, type=str,
        default="", help="The command to launch MPI programs "
        "for non-NERSC shell scripts (default do not use MPI)")

    parser.add_argument("--procs_per_node", required=False, type=int,
        default=0, help="The number of processes to use per node.  If not "
        "specified it uses a default value for each machine.")

    parser.add_argument("--db-postgres-user", type=str, required=False,
        default="desidev_ro", help="If using postgres, connect as this "
        "user for read-only access")

    parser.add_argument("--debug", required=False, default=False,
        action="store_true", help="debugging messages in job logs")

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    status = 0
    log = get_logger()

    int_night = 0
    if status == 0:
        try:
            int_night = int(args.night)
        except ValueError:
            log.critical("Could not convert night \"{}\" to an integer!"\
                .format(args.night))
            status = 3

    if status == 0:
        year = int_night // 10000
        month = (int_night - year*10000) // 100
        day = int_night - year*10000 - month*100
        try:
            assert 1969 < year < 2038
            assert 0 < month < 13
            assert 0 < day < 32
        except AssertionError:
            log.critical("Value for night \"{}\" is not a valid calendar "
                "date!".format(args.night))
            status = 4

    if status == 0:
        try:
            raw = io.rawdata_root()
        except AssertionError:
            log.critical("Raw data location not set")
            status = 5

    if status == 0:
        try:
            proddir = io.specprod_root()
        except AssertionError:
            log.critical("Production location not set")
            status = 6

    return args, status


def main():
    """
    Entry point for :command:`desi_start_night`,
    :command:`desi_update_night` and :command:`desi_end_night`.

    Returns:
        :class:`int`: An integer suitable for passing to :func:`sys.exit`.
    """
    # Find stage
    try:
        stage = stage_from_command()
    except:
        return 2

    # Parse and check arguments
    args, status = parse(stage)
    if status != 0:
        return status

    chaincom = ["desi_pipe", "chain", "--night", "{}".format(args.night)]

    if stage == "start":
        ttstr = ",".join(["pix", "psf", "psfnight", "traceshift",
            "extract", "fiberflat", "fiberflatnight"])
        chaincom.extend(["--tasktypes", ttstr])

    elif stage == "update":
        ttstr = ",".join(["pix", "psf", "psfnight", "traceshift",
            "extract", "fiberflat", "fiberflatnight", "sky", "starfit",
            "fluxcalib", "cframe"])
        chaincom.extend(["--tasktypes", ttstr])

    elif stage == "end":
        ttstr = ",".join(["pix", "psf", "psfnight", "traceshift",
            "extract", "fiberflat", "fiberflatnight", "sky", "starfit",
            "fluxcalib", "cframe", "spectra", "redshift"])
        chaincom.extend(["--tasktypes", ttstr])

    else:
        return 2

    if args.nersc is not None:
        chaincom.extend(["--nersc", args.nersc])
        chaincom.extend(["--nersc_queue", args.nersc_queue])
        chaincom.extend(["--nersc_runtime", "{}".format(args.nersc_runtime)])
        if args.nersc_shifter is not None:
            chaincom.extend(["--nersc_shifter", args.nersc_shifter])
        if args.procs_per_node > 0:
            chaincom.extend(["--procs_per_node",
                "{}".format(args.procs_per_node)])
    else:
        if args.mpi_procs > 1:
            chaincom.extend(["--mpi_procs", "{}".format(args.mpi_procs)])
        if args.mpi_run != "":
            chaincom.extend(["--mpi_run", args.mpi_run])

    if args.db_postgres_user != "desidev_ro":
        chaincom.extend(["--db-postgres-user", args.db_postgres_user])

    if args.debug:
        chaincom.extend(["--debug"])

    # Call desi_pipe chain

    log = get_logger()
    proddir = io.specprod_root()
    scrdir = io.get_pipe_scriptdir()
    ntdir = os.path.join(proddir, io.get_pipe_rundir(),
        io.get_pipe_scriptdir(), io.get_pipe_nightdir())

    if not os.path.isdir(ntdir):
        os.makedirs(ntdir)

    thisnight = os.path.join(ntdir, "{}".format(args.night))

    if not os.path.isdir(thisnight):
        os.makedirs(thisnight)

    reldir = os.path.join(io.get_pipe_nightdir(), "{}".format(args.night))

    chaincom.extend(["--outdir", reldir])

    log.info("Production {}:".format(proddir))
    log.info("  Running stage {} for night {}.".format(stage, args.night))
    log.info("  Chain command:  {}".format(" ".join(chaincom)))
    sys.stdout.flush()

    jobstr = sp.check_output(" ".join(chaincom), shell=True,
        universal_newlines=True)
    jobs = jobstr.split(",")

    if len(jobs) > 0:
        log.info("  Final job ID(s) are:  {}".format(":".join(jobs)))
        sys.stdout.flush()

    return 0
