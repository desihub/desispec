#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.state
===========================

Functions for manipulating the state of objects in the dependency graph.
"""

from __future__ import absolute_import, division, print_function

import os
import glob
import subprocess as sp

import numpy as np

from desiutil.log import get_logger
from .. import util
from .. import io

from .common import *

from .graph import *


def graph_db_info():
    """
    Return information about the runtime database.

    Currently this returns info about the temporary yaml files
    that contain a dump of the state.  In the future this will
    return connection information about the database.

    Args: None

    Returns:
        tuple with elements
            - file: name of the most recent state file
            - stime: modification time of the file
            - jobid: the slurm job name or POSIX PID
            - running: True if the job is still running
    """
    proddir = os.path.abspath(io.specprod_root())
    rundir = os.path.join(proddir, "run")

    file = ""
    stime = 0
    first = ""
    last = ""
    jobid = "-1"
    running = False

    statepat = re.compile(r'.*state_(.*).yaml')
    slrmpat = re.compile(r'slurm-(.*)')

    # Find the newest state file

    for stfile in glob.glob(os.path.join(rundir, "state_*.yaml")):
        thistime = os.path.getmtime(stfile)
        if thistime > stime:
            file = stfile
            stime = thistime
            statemat = statepat.match(stfile)
            if statemat is None:
                raise RuntimeError("state file matches glob but not regex- should never get here!")
            jobid = statemat.group(1)

    # See if this job is still running

    slrmmat = slrmpat.match(jobid)
    if slrmmat is None:
        # we were just using bash...
        pid = int(jobid)
        if util.pid_exists(pid):
            running = True
    else:
        slrmid = int(slrmmat.group(1))
        state = sp.check_output("squeue -j {} 2>/dev/null | tail -1 | gawk '{{print $10}}'".format(slrmid), shell=True)
        if state == 'R':
            running = True

    return (file, stime, jobid, running)


def graph_db_check(grph):
    """
    Check the state of all objects in a graph.

    This sets the state of all objects in the graph based on external
    information.  This might eventually involve querying a database.
    For now, the filesystem is checked for the existance of the object.
    Currently this marks all nodes as "none" or "done".  The "running" and
    "fail" states are overridden.  This may change in the future.

    Args:
        grph (dict): the dependency graph.

    Returns:
        Nothing.  The graph is modified in place.
    """
    for name, nd in grph.items():

        if type == "night":
            nd["state"] = "done"
            continue

        path = graph_path(name)

        if not os.path.isfile(path):
            # file does not exist
            nd["state"] = "none"
            continue

        if os.path.islink(path):
            # this is a fake symlink- always done
            nd["state"] = "done"
            continue

        tout = os.path.getmtime(path)

        stale = False
        for input in nd["in"]:
            if grph[input]["type"] == "night":
                continue
            inpath = graph_path(input)
            # if the input file exists, check if its timestamp
            # is newer than the output.
            if os.path.isfile(inpath):
                tin = os.path.getmtime(inpath)
                if tin > tout:
                    nd["state"] = "none"
                    stale = True

        if not stale:
            nd["state"] = "done"

    return


def graph_db_read(file):
    """
    Load the graph and all state info.

    Construct the graph from the runtime database.  For now, this
    just reads a yaml dump.

    Args:
        file (str): the path to the file to write.

    Returns:
        dict: The dependency graph.
    """
    return yaml_read(file)


def graph_db_write(grph):
    """
    Synchronize graph data to disk.

    This takes the in-memory graph and the states of all objects
    and writes this information to disk.  For now, this just dumps
    to a yaml file.  In the future, this function will modify a
    database.

    Args:
        grph (dict): the dependency graph.

    Returns:
        Nothing.
    """
    proddir = os.path.abspath(io.specprod_root())
    rundir = os.path.join(proddir, "run")

    jobid = None
    if "SLURM_JOBID" in os.environ:
        jobid = "slurm-{}".format(os.environ["SLURM_JOBID"])
    else:
        jobid = os.getpid()

    stateroot = "state_{}".format(jobid)
    statefile = os.path.join(rundir, "{}.yaml".format(stateroot))
    statedot = os.path.join(rundir, "{}.dot".format(stateroot))

    yaml_write(statefile, grph)
    with open(statedot, "w") as f:
        graph_dot(grph, f)

    return
