#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.prod
======================

Functions for updating and loading a production.
"""

from __future__ import absolute_import, division, print_function

import os
import re
import sys

import numpy as np

from yaml import load as yload
from yaml import dump as ydump
try:
    from yaml import CLoader as YLoader
except ImportError:
    from yaml import Loader as YLoader

import healpy as hp

from desiutil.log import get_logger

from .. import io

from .defs import prod_options_name

from .db import load_db


def yaml_write(path, input):
    """Write a dictionary to a file.

    Args:
        path (str): the output file name.
        input (dict): the data.

    Returns:
        nothing.
    """
    with open(path, "w") as f:
        ydump(input, f, default_flow_style=False)
    return


def yaml_read(path):
    """Read a dictionary from a file.

    Args:
        path (str): the input file name.

    Returns:
        dict: the data.
    """
    data = None
    with open(path, "r") as f:
        data = yload(f, Loader=YLoader)
    return data


def task_write(path, tasklist):
    """Write a task list to a text file or STDOUT.

    If the path is None, write lines to STDOUT.  In all cases, write a special
    termination line so that this stream or file can be passed into the
    task_read function.

    Args:
        path (str): the output file name.
        tasklist (list): the data.

    Returns:
        nothing.
    """
    if path is None:
        for tsk in tasklist:
            sys.stdout.write("{}\n".format(tsk))
        sys.stdout.write("#END\n")
    else:
        with open(path, "w") as f:
            for tsk in tasklist:
                f.write("{}\n".format(tsk))
            f.write("#END\n")
    return


def task_read(path):
    """Read a task list from a text file or STDIN.

    Lines that begin with '#' are ignored as comments.  If the path is None,
    lines are read from STDIN until an EOF marker is received.

    Args:
        path (str): the input file name.

    Returns:
        list: the list of tasks.

    """
    data = list()
    compat = re.compile(r"^#.*")
    if path is None:
        endpat = re.compile(r"^#END.*")
        for line in sys.stdin:
            if endpat.match(line) is not None:
                break
            if compat.match(line) is None:
                data.append(line.rstrip())
    else:
        with open(path, "r") as f:
            for line in f:
                if compat.match(line) is None:
                    data.append(line.rstrip())
    return data


def select_nights(allnights, nightstr):
    """Select nights based on regex matches.

    Given a list of nights, select all nights matching the specified
    patterns and return this subset.

    Args:
        allnights (list): list of all nights as strings
        nightstr (str): comma-separated list of regex patterns.

    Returns:
        list: list of nights that match the patterns.
    """

    nights = []
    if nightstr is not None:
        nightsel = nightstr.split(",")
        for sel in nightsel:
            pat = re.compile(sel)
            for nt in allnights:
                mat = pat.match(nt)
                if mat is not None:
                    if nt not in nights:
                        nights.append(nt)
        nights = sorted(nights)
    else:
        nights = sorted(allnights)

    return nights


def update_prod(nightstr=None, hpxnside=64, expid=None):
    """Create or update a production directory tree.

    For a given production, create the directory hierarchy and the starting
    default options.yaml file if it does not exist.  Also initialize the
    production DB if it does not exist.  Then update the DB with one or more
    nights from the raw data.  Nights to update may be specified by a list of
    simple regex matches.

    Args:
        nightstr (str): comma-separated list of regex patterns.
        hpxnside (int): The nside value to use for spectral grouping.
        expid (int): Only update a single exposure.  If this is specified,
            then nightstr must contain only a single night.

    """
    from .tasks.base import task_classes, task_type

    rawdir = os.path.abspath(io.rawdata_root())
    proddir = os.path.abspath(io.specprod_root())

    # create main directories if they don"t exist

    if not os.path.isdir(proddir):
        os.makedirs(proddir)

    cal2d = os.path.join(proddir, "calibnight")
    if not os.path.isdir(cal2d):
        os.makedirs(cal2d)

    expdir = os.path.join(proddir, "exposures")
    if not os.path.isdir(expdir):
        os.makedirs(expdir)

    predir = os.path.join(proddir, "preproc")
    if not os.path.isdir(predir):
        os.makedirs(predir)

    specdir = os.path.join(proddir, "spectra-{}".format(hpxnside))
    if not os.path.isdir(specdir):
        os.makedirs(specdir)

    rundir = io.get_pipe_rundir()
    if not os.path.isdir(rundir):
        os.makedirs(rundir)

    scriptdir = os.path.join(rundir, io.get_pipe_scriptdir())
    if not os.path.isdir(scriptdir):
        os.makedirs(scriptdir)

    logdir = os.path.join(rundir, io.get_pipe_logdir())
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    nightscrdir = os.path.join(scriptdir, io.get_pipe_nightdir())
    if not os.path.isdir(nightscrdir):
        os.makedirs(nightscrdir)

    nightlogdir = os.path.join(logdir, io.get_pipe_nightdir())
    if not os.path.isdir(nightlogdir):
        os.makedirs(nightlogdir)

    pixlogdir = os.path.join(logdir, io.get_pipe_pixeldir())
    if not os.path.isdir(pixlogdir):
        os.makedirs(pixlogdir)

    optfile = os.path.join(rundir, prod_options_name)
    if not os.path.isfile(optfile):
        opts = dict()
        for tt, tc in task_classes.items():
            tdict = { tt : tc.run_defaults() }
            opts.update(tdict)
        yaml_write(optfile, opts)

    # Load the database, this will create and initialize it if it does not
    # exist.

    dbpath = io.get_pipe_database()
    db = load_db(dbpath, "w")

    # Get list of available nights

    allnights = []
    nightpat = re.compile(r"\d{8}")
    for root, dirs, files in os.walk(rawdir, topdown=True):
        for d in dirs:
            nightmat = nightpat.match(d)
            if nightmat is not None:
                allnights.append(d)
        break

    # Select the requested nights

    nights = select_nights(allnights, nightstr)
    if (expid is not None) and (len(nights) > 1):
        raise RuntimeError("If updating a production for one exposure, only "
                           "a single night should be specified.")

    # Create per-night directories and update the DB for each night.

    for nt in nights:
        nexpdir = os.path.join(expdir, nt)
        if not os.path.isdir(nexpdir):
            os.makedirs(nexpdir)
        npredir = os.path.join(predir, nt)
        if not os.path.isdir(npredir):
            os.makedirs(npredir)
        ndir = os.path.join(cal2d, nt)
        if not os.path.isdir(ndir):
            os.makedirs(ndir)
        nlog = os.path.join(nightlogdir, nt)
        if not os.path.isdir(nlog):
            os.makedirs(nlog)
        nscr = os.path.join(nightscrdir, nt)
        if not os.path.isdir(nscr):
            os.makedirs(nscr)

        db.update(nt, hpxnside, expid)

        # make per-exposure dirs
        exps = None
        with db.cursor() as cur:
            if expid is None:
                cur.execute(\
                    "select expid from fibermap where night = {}".format(nt))
            else:
                # This query is essential a check that the expid is valid.
                cur.execute("select expid from fibermap where night = {} "
                            "and expid = {}".format(nt, expid))
            exps = [ int(x[0]) for x in cur.fetchall() ]
        for ex in exps:
            fdir = os.path.join(nexpdir, "{:08d}".format(ex))
            if not os.path.isdir(fdir):
                os.makedirs(fdir)
            fdir = os.path.join(npredir, "{:08d}".format(ex))
            if not os.path.isdir(fdir):
                os.makedirs(fdir)

    return


def load_prod(mode="w", user=None):
    """Load the database and options for a production.

    This loads the database from the production location defined by the usual
    DESI environment variables.  It also loads the global options file for
    the production.

    Args:
        mode (str): open mode for sqlite database ("r" or "w").
        user (str): for postgresql, an alternate user name for opening the DB.
            This can be used to connect as a user with read-only access.

    Returns:
        tuple: (pipeline.db.DataBase, dict) The database for the production
            and the global options dictionary.

    """
    dbpath = io.get_pipe_database()
    db = load_db(dbpath, mode=mode, user=user)

    rundir = io.get_pipe_rundir()
    optfile = os.path.join(rundir, prod_options_name)
    opts = yaml_read(optfile)

    return (db, opts)
