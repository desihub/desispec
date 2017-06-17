#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.plan
==========================

Functions for planning a production and manipulating the dependency graph.
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import time
import glob
import re
import copy

import numpy as np

import healpy as hp

from .. import io
from desiutil.log import get_logger
import desimodel.footprint

from .common import *
from .graph import *

from .task import (get_worker, default_workers,
    default_options)


def select_nights(allnights, nightstr):
    """
    Select nights based on regex matches.

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


def create_prod(nightstr=None, extra={}, specs=None, fakepix=False, hpxnside=64):
    """
    Create or update a production.

    For a given production, create the directory hierarchy and
    dependency graphs for all nights.  Also create the default
    options file if it does not exist and the bash and slurm
    scripts to run the pipeline.

    Args:
        nightstr (str): comma-separated list of regex patterns.
        extra (dict): dictionary of extra options for the task
            workers that will be added to the initial options.yaml
            file for the production.  The keys are the worker
            class names and the values are dictionaries that are
            passed to the constructor of each worker class.
        specs (list): list of spectrographs to use.
        fakepix (bool): if True, skip the checks for input pixel
            data files.  Assume that all spectrographs / cameras
            have data.  Useful for planning when simulating frame
            files directly.
        hpxnside (int): The nside value to use for spectral grouping.

    Returns:
        tuple containing the number of exposures of each type
        and the spectral groups.
    """

    rawdir = os.path.abspath(io.rawdata_root())
    proddir = os.path.abspath(io.specprod_root())

    if specs is None:
        specs = [ x for x in range(10) ]

    expnightcount = {}

    # create main directories if they don"t exist

    if not os.path.isdir(proddir):
        os.makedirs(proddir)

    cal2d = os.path.join(proddir, "calib2d")
    if not os.path.isdir(cal2d):
        os.makedirs(cal2d)

    calpsf = os.path.join(cal2d, "psf")
    if not os.path.isdir(calpsf):
        os.makedirs(calpsf)

    expdir = os.path.join(proddir, "exposures")
    if not os.path.isdir(expdir):
        os.makedirs(expdir)

    specdir = os.path.join(proddir, "spectra-{}".format(hpxnside))
    if not os.path.isdir(specdir):
        os.makedirs(specdir)

    plandir = io.get_pipe_plandir()
    if not os.path.isdir(plandir):
        os.makedirs(plandir)

    rundir = io.get_pipe_rundir()
    if not os.path.isdir(rundir):
        os.makedirs(rundir)

    faildir = os.path.join(rundir, io.get_pipe_faildir())
    if not os.path.isdir(faildir):
        os.makedirs(faildir)

    scriptdir = os.path.join(rundir, io.get_pipe_scriptdir())
    if not os.path.isdir(scriptdir):
        os.makedirs(scriptdir)

    logdir = os.path.join(rundir, io.get_pipe_logdir())
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    optfile = os.path.join(rundir, "options.yaml")
    if not os.path.isfile(optfile):
        opts = default_options(extra=extra)
        yaml_write(optfile, opts)

    # get list of nights

    allnights = []
    nightpat = re.compile(r"\d{8}")
    for root, dirs, files in os.walk(rawdir, topdown=True):
        for d in dirs:
            nightmat = nightpat.match(d)
            if nightmat is not None:
                allnights.append(d)
        break

    nights = select_nights(allnights, nightstr)

    # create per-night directories

    allpix = {}

    for nt in nights:
        nexpdir = os.path.join(expdir, nt)
        if not os.path.isdir(nexpdir):
            os.makedirs(nexpdir)
        ndir = os.path.join(cal2d, nt)
        if not os.path.isdir(ndir):
            os.makedirs(ndir)
        ndir = os.path.join(calpsf, nt)
        if not os.path.isdir(ndir):
            os.makedirs(ndir)
        nfail = os.path.join(faildir, nt)
        if not os.path.isdir(nfail):
            os.makedirs(nfail)
        nlog = os.path.join(logdir, nt)
        if not os.path.isdir(nlog):
            os.makedirs(nlog)

        grph, expcount, npix = graph_night(nt, specs, fakepix, hpxnside)

        for pix in npix:
            if pix in allpix:
                allpix[pix] += npix[pix]
            else:
                allpix[pix] = npix[pix]

        expnightcount[nt] = expcount
        with open(os.path.join(plandir, "{}.dot".format(nt)), "w") as f:
            graph_dot(grph, f)
        yaml_write(os.path.join(plandir, "{}.yaml".format(nt)), grph)

        # make per-exposure dirs
        for name, node in grph.items():
            if node["type"] == "fibermap":
                fdir = os.path.join(nexpdir, "{:08d}".format(node["id"]))
                if not os.path.isdir(fdir):
                    os.makedirs(fdir)

    return expnightcount, allpix


def graph_night(rawnight, specs, fakepix, hpxnside=64):
    """
    Generate the dependency graph for one night of data.

    Each node of the graph is a dictionary, with required keys "type",
    "in", and "out".  Where "in" and "out" are lists of other nodes.
    Extra keys for each type are allowed.  Some keys (band, spec, etc)
    are technically redundant (since they could be obtained by working
    back up the graph to the raw data properties), however this is done
    for convenience.

    Args:
        rawnight (str): The night to process.
        specs (list): List of integer spectrographs to use.
        fakepix (bool): If True, do not check for the existence of input
            pixel files.  Assume that data for all spectrographs and cameras
            exists.
        hpxnside (int): The nside value to use for spectral grouping.

    Returns:
        tuple containing

            - Dependency graph, as nested dictionaries.
            - exposure counts: dictionary of the number of exposures of
              each type.
            - dictionary of spectra groups for each fibermap.

    """
    log = get_logger()

    grph = {}

    node = {}
    node["type"] = "night"
    node["in"] = []
    node["out"] = []
    grph[rawnight] = node

    allpix = {}

    expcount = {}
    expcount["flat"] = 0
    expcount["arc"] = 0
    expcount["science"] = 0

    # First, insert raw data into the graph.  We use the existence of the raw data
    # as a filter over spectrographs.  Spectrographs whose raw data do not exist
    # are excluded from the graph.

    expid = io.get_exposures(rawnight, raw=True)

    campat = re.compile(r"([brz])([0-9])")

    keepspec = set()

    for ex in sorted(expid):
        # get the fibermap for this exposure
        fibermap = io.get_raw_files("fibermap", rawnight, ex)

        # Read the fibermap to get the exposure type, and while we are at it,
        # also accumulate the total list of spectra groups.  We use the list of
        # spectrographs to select ONLY the groups that are actually hit by
        # fibers from our chosen spectrographs.

        fmdata = io.read_fibermap(fibermap)
        flavor = fmdata.meta["FLAVOR"]
        fmpix = {}

        if flavor == "arc":
            expcount["arc"] += 1
        elif flavor == "flat":
            expcount["flat"] += 1
        else:
            expcount["science"] += 1
            ra = np.array(fmdata["RA_TARGET"], dtype=np.float64)
            dec = np.array(fmdata["DEC_TARGET"], dtype=np.float64)
            bad = np.where(fmdata["TARGETID"] < 0)[0]
            ra[bad] = 0.0
            dec[bad] = 0.0
            # pix = hp.ang2pix(hpxnside, ra, dec, nest=True, lonlat=True)
            pix = desimodel.footprint.radec2pix(hpxnside, ra, dec)
            pix[bad] = -1
            for fm in zip(fmdata["SPECTROID"], pix):
                if fm[1] >= 0:
                    if fm[0] in specs:
                        if fm[1] in fmpix:
                            fmpix[fm[1]] += 1
                        else:
                            fmpix[fm[1]] = 1
            for fmp in fmpix:
                if fmp in allpix:
                    allpix[fmp] += fmpix[fmp]
                else:
                    allpix[fmp] = fmpix[fmp]

        node = {}
        node["type"] = "fibermap"
        node["id"] = ex
        node["flavor"] = flavor
        node["nside"] = hpxnside
        node["pixels"] = fmpix
        node["in"] = [rawnight]
        node["out"] = []
        name = graph_name(rawnight, "fibermap-{:08d}".format(ex))

        grph[name] = node
        grph[rawnight]["out"].append(name)

        # get the raw exposures
        raw = {}
        if fakepix:
            # build the dictionary manually
            for band in ["b", "r", "z"]:
                for spec in specs:
                    cam = "{}{}".format(band, spec)
                    filename = "pix-{}{}-{:08d}.fits".format(band, spec, ex)
                    path = os.path.join(io.specprod_root(), rawnight, filename)
                    raw[cam] = path
        else:
            # take the intersection of existing pix files and our
            # selected spectrographs.
            allraw = io.get_raw_files("pix", rawnight, ex)
            for band in ["b", "r", "z"]:
                for spec in specs:
                    cam = "{}{}".format(band, spec)
                    if cam in allraw:
                        raw[cam] = allraw[cam]

        for cam in sorted(raw.keys()):
            cammat = campat.match(cam)
            if cammat is None:
                raise RuntimeError("invalid camera string {}".format(cam))
            band = cammat.group(1)
            spec = cammat.group(2)

            keepspec.update(spec)

            node = {}
            node["type"] = "pix"
            node["id"] = ex
            node["band"] = band
            node["spec"] = spec
            node["flavor"] = flavor
            node["in"] = [rawnight]
            node["out"] = []
            name = graph_name(rawnight, "pix-{}{}-{:08d}".format(band, spec, ex))

            grph[name] = node
            grph[rawnight]["out"].append(name)

    keep = sorted(keepspec)

    # Now that we have added all the raw data to the graph, we work our way
    # through the processing steps.

    # This step is a placeholder, in case we want to combine information from
    # multiple flats or arcs before running bootcalib.  We mark these bootcalib
    # outputs as depending on all arcs and flats, but in reality we may just
    # use the first or last set.

    # Since each psfboot file takes multiple exposures as input, we first
    # create those nodes.

    for band in ["b", "r", "z"]:
        for spec in keep:
            name = graph_name(rawnight, "psfboot-{}{}".format(band, spec))
            node = {}
            node["type"] = "psfboot"
            node["band"] = band
            node["spec"] = spec
            node["in"] = []
            node["out"] = []
            grph[name] = node

    current_items = list(grph.items())
    for name, nd in current_items:
        if nd["type"] != "pix":
            continue
        if (nd["flavor"] != "flat") and (nd["flavor"] != "arc"):
            continue
        band = nd["band"]
        spec = nd["spec"]
        bootname = graph_name(rawnight, "psfboot-{}{}".format(band, spec))
        grph[bootname]["in"].append(name)
        nd["out"].append(bootname)

    # Next is full PSF estimation.  Inputs are the arc image and the bootcalib
    # output file.  We also add nodes for the combined psfs.

    for band in ["b", "r", "z"]:
        for spec in keep:
            name = graph_name(rawnight, "psfnight-{}{}".format(band, spec))
            node = {}
            node["type"] = "psfnight"
            node["band"] = band
            node["spec"] = spec
            node["in"] = []
            node["out"] = []
            grph[name] = node

    #- cache current graph items so we can update graph as we go
    current_items = list(grph.items())
    for name, nd in current_items:
        if nd["type"] != "pix":
            continue
        if nd["flavor"] != "arc":
            continue
        band = nd["band"]
        spec = nd["spec"]
        id = nd["id"]
        bootname = graph_name(rawnight, "psfboot-{}{}".format(band, spec))
        psfname = graph_name(rawnight, "psf-{}{}-{:08d}".format(band, spec, id))
        psfnightname = graph_name(rawnight, "psfnight-{}{}".format(band, spec))
        node = {}
        node["type"] = "psf"
        node["band"] = band
        node["spec"] = spec
        node["id"] = id
        node["in"] = [name, bootname]
        node["out"] = [psfnightname]
        grph[psfname] = node
        grph[bootname]["out"].append(psfname)
        grph[psfnightname]["in"].append(psfname)
        nd["out"].append(psfname)

    # Now we extract the flats and science frames using the nightly psf

    #- cache current graph items so we can update graph as we go
    current_items = list(grph.items())
    for name, nd in current_items:
        if nd["type"] != "pix":
            continue
        if nd["flavor"] == "arc":
            continue
        band = nd["band"]
        spec = nd["spec"]
        id = nd["id"]
        flavor = nd["flavor"]
        framename = graph_name(rawnight, "frame-{}{}-{:08d}".format(band, spec, id))
        psfnightname = graph_name(rawnight, "psfnight-{}{}".format(band, spec))
        fmname = graph_name(rawnight, "fibermap-{:08d}".format(id))
        node = {}
        node["type"] = "frame"
        node["band"] = band
        node["spec"] = spec
        node["id"] = id
        node["flavor"] = flavor
        node["in"] = [name, fmname, psfnightname]
        node["out"] = []
        grph[framename] = node
        grph[psfnightname]["out"].append(framename)
        grph[fmname]["out"].append(framename)
        nd["out"].append(framename)

    # Now build the fiberflats for each flat exposure.  We keep a list of all
    # available fiberflats while we are looping over them, since we"ll need
    # that in the next step to select the "most recent" fiberflat.

    flatexpid = {}

    #- cache current graph items so we can update graph as we go
    current_items = list(grph.items())
    for name, nd in current_items:
        if nd["type"] != "frame":
            continue
        if nd["flavor"] != "flat":
            continue
        band = nd["band"]
        spec = nd["spec"]
        id = nd["id"]
        flatname = graph_name(rawnight, "fiberflat-{}{}-{:08d}".format(band, spec, id))
        node = {}
        node["type"] = "fiberflat"
        node["band"] = band
        node["spec"] = spec
        node["id"] = id
        node["in"] = [name]
        node["out"] = []
        grph[flatname] = node
        nd["out"].append(flatname)
        cam = "{}{}".format(band, spec)
        if cam not in flatexpid:
            flatexpid[cam] = []
        flatexpid[cam].append(id)

    # This is a small helper function to return the "most recent fiberflat"
    # that came before the current exposure.

    def last_flat(cam, expid):
        flatid = None
        flatname = None
        if cam in flatexpid:
            for fid in sorted(flatexpid[cam]):
                if (flatid is None):
                    flatid = fid
                elif (fid > flatid) and (fid < id):
                    flatid = fid
        if flatid is not None:
            flatname = graph_name(rawnight, "fiberflat-{}{}-{:08d}".format(band,
                spec, fid))
        else:
            # This means we don't have any flats for this night.
            # Probably this is because we are going to inject
            # already-calibrated simulation data into the production.
            # If this was really a mistake, then it will be caught
            # at runtime when the sky step fails.
            pass
        return flatid, flatname

    #- cache current graph items so we can update graph as we go
    current_items = list(grph.items())
    for name, nd in current_items:
        if nd["type"] != "frame":
            continue
        if nd["flavor"] == "flat":
            continue
        band = nd["band"]
        spec = nd["spec"]
        id = nd["id"]
        cam = "{}{}".format(band, spec)
        flatid, flatname = last_flat(cam, id)
        skyname = graph_name(rawnight, "sky-{}{}-{:08d}".format(band, spec, id))
        node = {}
        node["type"] = "sky"
        node["band"] = band
        node["spec"] = spec
        node["id"] = id
        node["in"] = [name]
        if flatname is not None:
            node["in"].append(flatname)
            grph[flatname]["out"].append(skyname)
        node["out"] = []
        grph[skyname] = node
        nd["out"].append(skyname)

    # Construct the standard star files.  These are one per spectrograph,
    # and depend on the frames and the corresponding flats and sky files.

    stdgrph = {}

    #- cache current graph items so we can update graph as we go
    current_items = list(grph.items())
    for name, nd in current_items:
        if nd["type"] != "frame":
            continue
        if nd["flavor"] == "flat":
            continue
        band = nd["band"]
        spec = nd["spec"]
        id = nd["id"]

        starname = graph_name(rawnight, "stdstars-{}-{:08d}".format(spec, id))
        # does this spectrograph exist yet in the graph?
        if starname not in stdgrph:
            fmname = graph_name(rawnight, "fibermap-{:08d}".format(id))
            grph[fmname]["out"].append(starname)
            node = {}
            node["type"] = "stdstars"
            node["spec"] = spec
            node["id"] = id
            node["in"] = [fmname]
            node["out"] = []
            stdgrph[starname] = node

        cam = "{}{}".format(band, spec)
        flatid, flatname = last_flat(cam, id)
        skyname = graph_name(rawnight, "sky-{}{}-{:08d}".format(band, spec, id))
        stdgrph[starname]["in"].extend([skyname, name])
        if flatname is not None:
            stdgrph[starname]["in"].extend([flatname])
            grph[flatname]["out"].append(starname)
        nd["out"].append(starname)
        grph[skyname]["out"].append(starname)

    grph.update(stdgrph)

    # Construct calibration files

    #- cache current graph items so we can update graph as we go
    current_items = list(grph.items())
    for name, nd in current_items:
        if nd["type"] != "frame":
            continue
        if nd["flavor"] == "flat":
            continue
        band = nd["band"]
        spec = nd["spec"]
        id = nd["id"]
        cam = "{}{}".format(band, spec)
        flatid, flatname = last_flat(cam, id)
        skyname = graph_name(rawnight, "sky-{}{}-{:08d}".format(band, spec, id))
        starname = graph_name(rawnight, "stdstars-{}-{:08d}".format(spec, id))
        calname = graph_name(rawnight, "calib-{}{}-{:08d}".format(band, spec, id))
        node = {}
        node["type"] = "calib"
        node["band"] = band
        node["spec"] = spec
        node["id"] = id
        node["in"] = [name, skyname, starname]
        if flatname is not None:
            node["in"].extend([flatname])
            grph[flatname]["out"].append(calname)
        node["out"] = []
        grph[calname] = node
        grph[skyname]["out"].append(calname)
        grph[starname]["out"].append(calname)
        nd["out"].append(calname)

    # Build cframe files

    #- cache current graph items so we can update graph as we go
    current_items = list(grph.items())
    for name, nd in current_items:
        if nd["type"] != "frame":
            continue
        if nd["flavor"] == "flat":
            continue
        band = nd["band"]
        spec = nd["spec"]
        id = nd["id"]
        cam = "{}{}".format(band, spec)
        flatid, flatname = last_flat(cam, id)
        skyname = graph_name(rawnight, "sky-{}{}-{:08d}".format(band, spec, id))
        calname = graph_name(rawnight, "calib-{}{}-{:08d}".format(band, spec, id))
        cfname = graph_name(rawnight, "cframe-{}{}-{:08d}".format(band, spec, id))
        node = {}
        node["type"] = "cframe"
        node["band"] = band
        node["spec"] = spec
        node["id"] = id
        node["in"] = [name, skyname, calname]
        if flatname is not None:
            node["in"].extend([flatname])
            grph[flatname]["out"].append(cfname)
        node["out"] = []
        grph[cfname] = node
        grph[skyname]["out"].append(cfname)
        grph[calname]["out"].append(cfname)
        nd["out"].append(cfname)

    # Spectra / Zbest dependencies

    for p in allpix:
        zname = "zbest-{}-{}".format(hpxnside, p)
        sname = "spectra-{}-{}".format(hpxnside, p)

        node = {}
        node["type"] = "spectra"
        node["nside"] = hpxnside
        node["pixel"] = p
        node["in"] = []
        node["out"] = [zname]
        grph[sname] = node
            
        node = {}
        node["type"] = "zbest"
        node["nside"] = hpxnside
        node["pixel"] = p
        node["ntarget"] = allpix[p]
        node["in"] = [sname]
        node["out"] = []
        grph[zname] = node

    #- cache current graph items so we can update graph as we go
    current_items = list(grph.items())
    for name, nd in current_items:
        if nd["type"] != "fibermap":
            continue
        if nd["flavor"] == "arc":
            continue
        if nd["flavor"] == "flat":
            continue
        id = nd["id"]
        fmpix = nd["pixels"]
        for band in ["b", "r", "z"]:
            for spec in keep:
                cfname = graph_name(rawnight, "cframe-{}{}-{:08d}".format(band, spec, id))
                for p in fmpix:
                    sname = "spectra-{}-{}".format(hpxnside, p)
                    grph[sname]["in"].append(cfname)
                    grph[cfname]["out"].append(sname)

    return (grph, expcount, allpix)


def load_prod(nightstr=None, spectrographs=None, progress=None):
    """
    Load the dependency graph for a production.

    This loads the dependency graphs for one or more nights.  It
    can also filter the graph based on regex matching the night and
    using a subset of the spectrographs.

    Args:
        nightstr (str): comma-separated list of regex patterns.
        spectrographs (str): comma-separated list of spectrographs.

    Returns:
        dict: The full multi-night graph with selections applied.
    """

    proddir = os.path.abspath(io.specprod_root())

    plandir = os.path.join(proddir, "plan")

    allnights = []
    planpat = re.compile(r"([0-9]{8})\.yaml")
    for root, dirs, files in os.walk(plandir, topdown=True):
        for f in files:
            planmat = planpat.match(f)
            if planmat is not None:
                night = planmat.group(1)
                allnights.append(night)
        break

    # select nights to use

    nights = None
    if nightstr is None:
        nights = allnights
    else:
        nights = select_nights(allnights, nightstr)

    # select the spectrographs to use

    spects = []
    if spectrographs is None:
        for s in range(10):
            spects.append(s)
    else:
        spc = spectrographs.split(",")
        for s in spc:
            spects.append(int(s))

    # load the graphs from selected nights and merge.  We must also merge
    # the cframe dependencies from all spectra objects from all nights. 
    # We don't use graph_slice / graph_prune here since we are purposely
    # leaving "dangling" dependencies in the graph.

    groups = {}
    grph = {}
    import pprint

    for n in nights:
        nightfile = os.path.join(plandir, "{}.yaml".format(n))
        ngrph = yaml_read(nightfile)

        # Slice out spectrographs that we want.
        sgrph = graph_slice_spec(ngrph, spectrographs=spects)

        # Split our graph into just the spectra nodes and everything
        # else.
        ngrph = {}
        for name, nd in sgrph.items():
            if nd["type"] == "spectra":
                if name not in groups:
                    groups[name] = copy.deepcopy(nd)
                else:
                    for cf in nd["in"]:
                        if cf not in groups[name]["in"]:
                            groups[name]["in"].append(cf)
            else:
                ngrph[name] = copy.deepcopy(nd)

        # Update the main graph with everything except the spectral groups
        grph.update(ngrph)

    # Now update the combined graph with the merged spectra objects
    grph.update(groups)

    return grph
