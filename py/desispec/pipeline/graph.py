#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.graph
======================

Dependency graph manipulation.
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import re
import copy

from .. import io

from .common import *


_state_colors = {
    "none": "#000000",
    "done": "#00ff00",
    "fail": "#ff0000",
    "wait": "#ffff00"
}


_graph_sep = "_"


def graph_name(*args):
    """
    Join night and object names into the unique graph name.

    Args:
        args (list): strings to join.

    Returns (str):
        the name in the graph.
    """
    if len(args) > 0:
        return _graph_sep.join(args)
    else:
        return ""


def graph_name_split(name):
    """
    Split graph object name into night and the remainder.

    Args:
        name (str): the object name.

    Returns:
        tuple containing the night and the remainder of the name.
    """
    patstr = "([0-9]{{8}}){}(.*)".format(_graph_sep)
    pat = re.compile(patstr)
    mat = pat.match(name)
    ret = ("", name)
    if mat is not None:
        night = mat.group(1)
        obj = mat.group(2)
        ret = (night, obj)
    return ret


def graph_path(type, name):
    """
    Convert object name in the graph to a filesystem path.

    Although the names of objects in the dependency graph loosely
    map to file names, this is not required.  This function uses
    the canonical desispec.io.meta.findfile function to get the
    filesystem path of an object.

    Args:
        type (str): object type.
        name (str): the name of the object in the graph.

    Returns (str):
        Filesystem path to object.
    """
    path = ""
    if type == "night":
        proddir = os.path.abspath(io.specprod_root())
        path = os.path.join(proddir, "exposures", name)
    else:
        night = None
        expid = None
        camera = None
        brickname = None
        band = None
        spectrograph = None
        if type == "fibermap":
            patstr = "([0-9]{{8}}){}fibermap-([0-9]{{8}})".format(_graph_sep)
            pat = re.compile(patstr)
            mat = pat.match(name)
            if mat is None:
                raise RuntimeError("{} is not a valid fibermap name".format(name))
            night = mat.group(1)
            expid = int(mat.group(2))
        elif type == "pix":
            patstr = "([0-9]{{8}}){}pix-([brz][0-9])-([0-9]{{8}})".format(_graph_sep)
            pat = re.compile(patstr)
            mat = pat.match(name)
            if mat is None:
                raise RuntimeError("{} is not a valid pix name".format(name))
            night = mat.group(1)
            camera = mat.group(2)
            expid = int(mat.group(3))
        elif type == "psfboot":
            patstr = "([0-9]{{8}}){}psfboot-([brz][0-9])".format(_graph_sep)
            pat = re.compile(patstr)
            mat = pat.match(name)
            if mat is None:
                raise RuntimeError("{} is not a valid psfboot name".format(name))
            night = mat.group(1)
            camera = mat.group(2)
        elif type == "psf":
            patstr = "([0-9]{{8}}){}psf-([brz][0-9])-([0-9]{{8}})".format(_graph_sep)
            pat = re.compile(patstr)
            mat = pat.match(name)
            if mat is None:
                raise RuntimeError("{} is not a valid psf name".format(name))
            night = mat.group(1)
            camera = mat.group(2)
            expid = int(mat.group(3))
        elif type == "psfnight":
            patstr = "([0-9]{{8}}){}psfnight-([brz][0-9])".format(_graph_sep)
            pat = re.compile(patstr)
            mat = pat.match(name)
            if mat is None:
                raise RuntimeError("{} is not a valid psfnight name".format(name))
            night = mat.group(1)
            camera = mat.group(2)
        elif type == "frame":
            patstr = "([0-9]{{8}}){}frame-([brz][0-9])-([0-9]{{8}})".format(_graph_sep)
            pat = re.compile(patstr)
            mat = pat.match(name)
            if mat is None:
                raise RuntimeError("{} is not a valid frame name".format(name))
            night = mat.group(1)
            camera = mat.group(2)
            expid = int(mat.group(3))
        elif type == "fiberflat":
            patstr = "([0-9]{{8}}){}fiberflat-([brz][0-9])-([0-9]{{8}})".format(_graph_sep)
            pat = re.compile(patstr)
            mat = pat.match(name)
            if mat is None:
                raise RuntimeError("{} is not a valid fiberflat name".format(name))
            night = mat.group(1)
            camera = mat.group(2)
            expid = int(mat.group(3))
        elif type == "sky":
            patstr = "([0-9]{{8}}){}sky-([brz][0-9])-([0-9]{{8}})".format(_graph_sep)
            pat = re.compile(patstr)
            mat = pat.match(name)
            if mat is None:
                raise RuntimeError("{} is not a valid sky name".format(name))
            night = mat.group(1)
            camera = mat.group(2)
            expid = int(mat.group(3))
        elif type == "stdstars":
            patstr = "([0-9]{{8}}){}stdstars-([0-9])-([0-9]{{8}})".format(_graph_sep)
            pat = re.compile(patstr)
            mat = pat.match(name)
            if mat is None:
                raise RuntimeError("{} is not a valid standard star name".format(name))
            night = mat.group(1)
            spectrograph = int(mat.group(2))
            expid = int(mat.group(3))
        elif type == "calib":
            patstr = "([0-9]{{8}}){}calib-([brz][0-9])-([0-9]{{8}})".format(_graph_sep)
            pat = re.compile(patstr)
            mat = pat.match(name)
            if mat is None:
                raise RuntimeError("{} is not a valid calibration name".format(name))
            night = mat.group(1)
            camera = mat.group(2)
            expid = int(mat.group(3))
        elif type == "cframe":
            patstr = "([0-9]{{8}}){}cframe-([brz][0-9])-([0-9]{{8}})".format(_graph_sep)
            pat = re.compile(patstr)
            mat = pat.match(name)
            if mat is None:
                raise RuntimeError("{} is not a valid cframe name".format(name))
            night = mat.group(1)
            camera = mat.group(2)
            expid = int(mat.group(3))
        elif type == "brick":
            patstr = "brick-([brz])-(.*)"
            pat = re.compile(patstr)
            mat = pat.match(name)
            if mat is None:
                raise RuntimeError("{} is not a valid brick name".format(name))
            band = mat.group(1)
            brickname = mat.group(2)
        elif type == "zbest":
            patstr = "zbest-(.*)"
            pat = re.compile(patstr)
            mat = pat.match(name)
            if mat is None:
                raise RuntimeError("{} is not a valid zbest name".format(name))
            brickname = mat.group(1)
        else:
            raise RuntimeError("unknown type {}".format(type))

        path = io.findfile(type, night=night, expid=expid, camera=camera, brickname=brickname, band=band, spectrograph=spectrograph)
    return path


def graph_prune(grph, name, descend=False):
    """
    Remove a node from the dependency graph.

    Args:
        grph (dict): the graph.
        name (str): the node to remove.
        descend (bool): if True, recursively remove objects
            that depend on this node.

    Returns:
        nothing- graph is modified in place.
    """

    # unlink from parents
    for p in grph[name]["in"]:
        grph[p]["out"].remove(name)
    if descend:
        # recursively process children
        for c in grph[name]["out"]:
            graph_prune(grph, c, descend=True)
    else:
        # not removing children, so only unlink
        for c in grph[name]["out"]:
            grph[c]["in"].remove(name)
    del grph[name]
    return


def graph_slice(grph, names=None, types=None, deps=False):
    """
    Select nodes based on name and type.

    Create a new graph that has nodes of the specified name or
    type, and optionally also copy the direct inputs to these
    nodes.

    Args:
        grph (dict): the graph.
        names (list): list of node names to keep.
        types (list): list of node types to keep.
        deps (bool): if True, keep direct inputs to selected nodes.

    Returns (dict):
        the new graph.
    """
    if types is None:
        types = graph_types

    newgrph = {}
    
    # First copy directly selected nodes
    for name, nd in grph.items():
        if (names is not None) and (name not in names):
            continue
        if nd["type"] not in types:
            continue
        newgrph[name] = copy.deepcopy(nd)

    # Now optionally grab all direct inputs
    if deps:
        for name, nd in list(newgrph.items()):
            for p in nd["in"]:
                if p not in newgrph:
                    newgrph[p] = copy.deepcopy(grph[p])

    # Now remove links that we have pruned
    current_items = list(newgrph.items())
    for name, nd in current_items:
        newin = []
        for p in nd["in"]:
            if p in newgrph:
                newin.append(p)
        nd["in"] = newin
        newout = []
        for c in nd["out"]:
            if c in newgrph:
                newout.append(c)
        nd["out"] = newout

    return newgrph


def graph_slice_spec(grph, spectrographs=None):
    """
    Select graph objects based on spectrograph.

    Create a new graph that uses only the specified list
    of spectrographs.

    Args:
        grph (dict): the graph.
        spectrographs (list): list of ints

    Returns (dict):
        the new graph.
    """

    newgrph = copy.deepcopy(grph)
    if spectrographs is None:
        spectrographs = list(range(10))
    current_items = list(newgrph.items())
    for name, nd in current_items:
        if "spec" in nd:
            if int(nd["spec"]) not in spectrographs:
                graph_prune(newgrph, name, descend=False)
    return newgrph


def graph_set_recursive(grph, name, state):
    """
    Recursively set the state of a node of the graph.

    Descend the graph, recursively marking outputs
    with the same state.

    Args:
        grph (dict): the dependency graph.
        name (str): the node name.
        state (str): the state to set.

    Returns:
        nothing.
    """
    
    # recursively process children
    for c in grph[name]["out"]:
        graph_set_recursive(grph, c, state)
    
    # set top node state
    grph[name]["state"] = state

    return


def graph_dot(grph, f):
    """
    Create a DOT file for graph visualization.

    This writes DOT commands to the specified file handle.

    Args:
        grph (dict): the dependency graph.
        f (stream): an open stream that supports the write() method.

    Returns:
        Nothing.
    """

    # For visualization, we rank nodes of the same type together.

    rank = {}
    for t in graph_types:
        rank[t] = []

    for name, nd in grph.items():
        if nd["type"] not in graph_types:
            raise RuntimeError("graph node {} has invalid type {}".format(name, nd["type"]))
        rank[nd["type"]].append(name)

    tab = "    "
    f.write("\n// DESI Plan\n\n")
    f.write("digraph DESI {\n")
    f.write("splines=false;\n")
    f.write("overlap=false;\n")
    f.write("{}rankdir=LR\n".format(tab))

    # organize nodes into subgraphs

    for t in graph_types:
        f.write('{}subgraph cluster{} {{\n'.format(tab, t))
        f.write('{}{}label="{}";\n'.format(tab, tab, t))
        f.write('{}{}newrank=true;\n'.format(tab, tab))
        f.write('{}{}rank=same;\n'.format(tab, tab))
        for name in sorted(rank[t]):
            nd = grph[name]
            props = "[shape=box,penwidth=3"
            if "state" in nd:
                props = "{},color=\"{}\"".format(props, _state_colors[nd["state"]])
            else:
                props = "{},color=\"{}\"".format(props, _state_colors["none"])
            props = "{}]".format(props)
            f.write('{}{}"{}" {};\n'.format(tab, tab, name, props))
        f.write("{}}}\n".format(tab))

    # write dependencies

    for t in graph_types:
        for name in sorted(rank[t]):
            for child in grph[name]["out"]:
                f.write('{}"{}" -> "{}" [penwidth=1,color="#999999"];\n'.format(tab, name, child))

    # write rank grouping

    # for t in types:
    #     if (t == "night") and len(rank[t]) == 1:
    #         continue
    #     f.write("{}{{ rank=same ".format(tab))
    #     for name in sorted(rank[t]):
    #         f.write(""{}" ".format(name))
    #     f.write(" }\n")

    f.write("}\n\n")

    return


def graph_merge(grph, comm=None):
    """
    Merge graph states across multiple processes.

    Each process has the same graph of objects, but those objects have
    different states based on which tasks a process has run locally.
    This function reconciles the state of all objects between processes
    and re-broadcasts the result so that all processes have the same
    state information.

    Args:
        grph (dict): the dependency graph.
        comm (mpi4py.MPI.Comm): the MPI communicator.

    Returns:
        Nothing.
    """

    if comm is None:
        return
    elif comm.size == 1:
        return
    
    # check that we have the same list of nodes on all processes.  Then
    # merge the states.  "fail" overrides "None", and "done" overrides
    # them both.
    
    states = {}
    names = sorted(grph.keys())
    for n in names:
        states[n] = grph[n]["state"]

    priority = {
        "none" : 0,
        "wait" : 1,
        "fail" : 2,
        "done" : 3
    }

    for p in range(1, comm.size):

        if comm.rank == 0:
            # print("proc {} receiving from {}".format(comm.rank, p))
            # sys.stdout.flush()
            pstates = comm.recv(source=p, tag=p)
            pnames = sorted(pstates.keys())
            if pnames != names:
                raise RuntimeError("names of all objects must be the same when merging graph states")
            for n in names:
                if priority[pstates[n]] > priority[states[n]]:
                    states[n] = pstates[n]

        elif comm.rank == p:
            # print("proc {} sending to {}".format(comm.rank, 0))
            # sys.stdout.flush()
            comm.send(states, dest=0, tag=p)
        comm.barrier()

    # broadcast final merged state back to all processes.
    # print("proc {} hit bcast of states".format(comm.rank))
    # sys.stdout.flush()
    states = comm.bcast(states, root=0)

    # update process-local graph
    for n in names:
        grph[n]["state"] = states[n]

    # print("proc {} ending merge".format(comm.rank))
    # sys.stdout.flush()
    return

