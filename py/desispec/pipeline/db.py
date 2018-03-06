#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.db
===========================

Pipeline processing database
"""

from __future__ import absolute_import, division, print_function

import os

import re

import numpy as np

from desiutil.log import get_logger

from .. import io

import fitsio

from .defs import (task_states, task_int_to_state, task_state_to_int, task_name_sep)


def task_types():
    """Get the list of possible task types that are supported.

    Returns:
        list: The list of supported task types.

    """
    from .tasks.base import task_classes, task_type
    return list(sorted(task_classes.keys()))


def all_tasks(night, nside):
    """Get all possible tasks for a single night.

    This uses the filesystem to query the raw data for a particular night and
    return a dictionary containing all possible tasks for each task type.  For
    objects which span multiple nights (e.g. spectra, zbest), this returns the
    tasks which are touched by the given night.

    Args:
        night (str): The night to scan for tasks.
        nside (int): The HEALPix NSIDE value to use.

    Returns:
        dict: a dictionary whose keys are the task types and where each value
            is a list of task properties.

    """
    import desimodel.footprint


    log = get_logger()

    log.debug("io.get_exposures night={}".format(night))

    expid = io.get_exposures(night, raw=True)

    full = dict()
    for t in task_types():
        full[t] = list()

    for ex in sorted(expid):

        log.debug("read fibermap for exposure {}".format(ex))

        # get the fibermap for this exposure
        fibermap = io.get_raw_files("fibermap", night, ex)

        #fmdata = io.read_fibermap(fibermap)
        #flavor = fmdata.meta["FLAVOR"]

        fmdata,header = fitsio.read(fibermap,header=True)
        flavor = header["FLAVOR"].strip().lower()



        fmpix = dict()
        if (flavor != "arc") and (flavor != "flat"):
            # This will be used to track which healpix pixels are
            # touched by fibers from each spectrograph.
            ra = np.array(fmdata["RA_TARGET"], dtype=np.float64)
            dec = np.array(fmdata["DEC_TARGET"], dtype=np.float64)
            bad = np.where(fmdata["TARGETID"] < 0)[0]
            ra[bad] = 0.0
            dec[bad] = 0.0
            pix = desimodel.footprint.radec2pix(nside, ra, dec)
            pix[bad] = -1
            # FIXME: how are we storing this info in the database?
            # for fm in zip(fmdata["SPECTROID"], pix):
            #     if fm[1] >= 0:
            #         if fm[0] in specs:
            #             if fm[1] in fmpix:
            #                 fmpix[fm[1]] += 1
            #             else:
            #                 fmpix[fm[1]] = 1
            # for fmp in fmpix:
            #     if fmp in allpix:
            #         allpix[fmp] += fmpix[fmp]
            #     else:
            #         allpix[fmp] = fmpix[fmp]

        fmprops = dict()
        fmprops["night"]  = int(night)
        fmprops["expid"]  = int(ex)
        fmprops["flavor"] = flavor
        fmprops["state"]  = "done"

        full["fibermap"].append(fmprops)


        rdprops = dict()
        rdprops["night"]  = int(night)
        rdprops["expid"]  = int(ex)
        rdprops["flavor"] = flavor
        rdprops["state"]  = "done"

        full["rawdata"].append(rdprops)

        # Add the preprocessed pixel files
        for band in ['b', 'r', 'z']: # need to open the rawdat file to see how many spectros and cameras are there
            for spec in range(10): #
                pixprops = dict()
                pixprops["night"] = int(night)
                pixprops["band"] = band
                pixprops["spec"] = spec
                pixprops["expid"] = int(ex)
                pixprops["flavor"] = flavor
                pixprops["state"] = "ready"
                full["pix"].append(pixprops)

                if flavor == "arc" :
                    # Add the PSF files
                    props = dict()
                    props["night"] = int(night)
                    props["band"] = band
                    props["spec"] = spec
                    props["expid"] = int(ex)
                    props["state"] = "waiting" # see defs.task_states
                    full["psf"].append(props)

                    # Add a PSF night file if does not exist
                    exists=False
                    for entry in full["psfnight"] :
                        if entry["night"]==props["night"] \
                           and entry["band"]==props["band"] \
                           and entry["spec"]==props["spec"] :
                            exists=True
                            break
                    if not exists :
                         props = dict()
                         props["night"] = int(night)
                         props["band"] = band
                         props["spec"] = spec
                         props["state"] = "waiting" # see defs.task_states
                         full["psfnight"].append(props)

                if flavor != "arc" :
                    # Add extractions
                    props = dict()
                    props["night"] = int(night)
                    props["band"] = band
                    props["spec"] = spec
                    props["expid"] = int(ex)
                    props["state"] = "waiting" # see defs.task_states
                    full["extract"].append(props)
                
                if flavor == "flat" :
                    # Add a fiberflat task
                    props = dict()
                    props["night"] = int(night)
                    props["band"] = band
                    props["spec"] = spec
                    props["expid"] = int(ex)
                    props["state"] = "waiting" # see defs.task_states
                    full["fiberflat"].append(props)

    log.debug("done")
    return full



#
#
# def graph_night(rawnight, specs, fakepix, hpxnside=64):
#     """
#     Generate the dependency graph for one night of data.
#
#     Each node of the graph is a dictionary, with required keys "type",
#     "in", and "out".  Where "in" and "out" are lists of other nodes.
#     Extra keys for each type are allowed.  Some keys (band, spec, etc)
#     are technically redundant (since they could be obtained by working
#     back up the graph to the raw data properties), however this is done
#     for convenience.
#
#     Args:
#         rawnight (str): The night to process.
#         specs (list): List of integer spectrographs to use.
#         fakepix (bool): If True, do not check for the existence of input
#             pixel files.  Assume that data for all spectrographs and cameras
#             exists.
#         hpxnside (int): The nside value to use for spectral grouping.
#
#     Returns:
#         tuple containing
#
#             - Dependency graph, as nested dictionaries.
#             - exposure counts: dictionary of the number of exposures of
#               each type.
#             - dictionary of spectra groups for each fibermap.
#
#     """
#     log = get_logger()
#
#     grph = {}
#
#     node = {}
#     node["type"] = "night"
#     node["in"] = []
#     node["out"] = []
#     grph[rawnight] = node
#
#     allpix = {}
#
#     expcount = {}
#     expcount["flat"] = 0
#     expcount["arc"] = 0
#     expcount["science"] = 0
#
#     # First, insert raw data into the graph.  We use the existence of the raw data
#     # as a filter over spectrographs.  Spectrographs whose raw data do not exist
#     # are excluded from the graph.
#
#     expid = io.get_exposures(rawnight, raw=True)
#
#     campat = re.compile(r"([brz])([0-9])")
#
#     keepspec = set()
#
#     for ex in sorted(expid):
#         # get the fibermap for this exposure
#         fibermap = io.get_raw_files("fibermap", rawnight, ex)
#
#         # Read the fibermap to get the exposure type, and while we are at it,
#         # also accumulate the total list of spectra groups.  We use the list of
#         # spectrographs to select ONLY the groups that are actually hit by
#         # fibers from our chosen spectrographs.
#
#         fmdata = io.read_fibermap(fibermap)
#         flavor = fmdata.meta["FLAVOR"]
#         fmpix = {}
#
#         if flavor == "arc":
#             expcount["arc"] += 1
#         elif flavor == "flat":
#             expcount["flat"] += 1
#         else:
#             expcount["science"] += 1
#             ra = np.array(fmdata["RA_TARGET"], dtype=np.float64)
#             dec = np.array(fmdata["DEC_TARGET"], dtype=np.float64)
#             bad = np.where(fmdata["TARGETID"] < 0)[0]
#             ra[bad] = 0.0
#             dec[bad] = 0.0
#             # pix = hp.ang2pix(hpxnside, ra, dec, nest=True, lonlat=True)
#             pix = desimodel.footprint.radec2pix(hpxnside, ra, dec)
#             pix[bad] = -1
#             for fm in zip(fmdata["SPECTROID"], pix):
#                 if fm[1] >= 0:
#                     if fm[0] in specs:
#                         if fm[1] in fmpix:
#                             fmpix[fm[1]] += 1
#                         else:
#                             fmpix[fm[1]] = 1
#             for fmp in fmpix:
#                 if fmp in allpix:
#                     allpix[fmp] += fmpix[fmp]
#                 else:
#                     allpix[fmp] = fmpix[fmp]
#
#         node = {}
#         node["type"] = "fibermap"
#         node["id"] = ex
#         node["flavor"] = flavor
#         node["nside"] = hpxnside
#         node["pixels"] = fmpix
#         node["in"] = [rawnight]
#         node["out"] = []
#         name = graph_name(rawnight, "fibermap-{:08d}".format(ex))
#
#         grph[name] = node
#         grph[rawnight]["out"].append(name)
#
#         # get the raw exposures
#         raw = {}
#         if fakepix:
#             # build the dictionary manually
#             for band in ["b", "r", "z"]:
#                 for spec in specs:
#                     cam = "{}{}".format(band, spec)
#                     filename = "pix-{}{}-{:08d}.fits".format(band, spec, ex)
#                     path = os.path.join(io.specprod_root(), rawnight, filename)
#                     raw[cam] = path
#         else:
#             # take the intersection of existing pix files and our
#             # selected spectrographs.
#             allraw = io.get_raw_files("pix", rawnight, ex)
#             for band in ["b", "r", "z"]:
#                 for spec in specs:
#                     cam = "{}{}".format(band, spec)
#                     if cam in allraw:
#                         raw[cam] = allraw[cam]
#
#         for cam in sorted(raw.keys()):
#             cammat = campat.match(cam)
#             if cammat is None:
#                 raise RuntimeError("invalid camera string {}".format(cam))
#             band = cammat.group(1)
#             spec = cammat.group(2)
#
#             keepspec.update(spec)
#
#             node = {}
#             node["type"] = "pix"
#             node["id"] = ex
#             node["band"] = band
#             node["spec"] = spec
#             node["flavor"] = flavor
#             node["in"] = [rawnight]
#             node["out"] = []
#             name = graph_name(rawnight, "pix-{}{}-{:08d}".format(band, spec, ex))
#
#             grph[name] = node
#             grph[rawnight]["out"].append(name)
#
#     keep = sorted(keepspec)
#
#     # Now that we have added all the raw data to the graph, we work our way
#     # through the processing steps.
#
#     # This step is a placeholder, in case we want to combine information from
#     # multiple flats or arcs before running bootcalib.  We mark these bootcalib
#     # outputs as depending on all arcs and flats, but in reality we may just
#     # use the first or last set.
#
#     # Since each psfboot file takes multiple exposures as input, we first
#     # create those nodes.
#
#     for band in ["b", "r", "z"]:
#         for spec in keep:
#             name = graph_name(rawnight, "psfboot-{}{}".format(band, spec))
#             node = {}
#             node["type"] = "psfboot"
#             node["band"] = band
#             node["spec"] = spec
#             node["in"] = []
#             node["out"] = []
#             grph[name] = node
#
#     current_items = list(grph.items())
#     for name, nd in current_items:
#         if nd["type"] != "pix":
#             continue
#         if (nd["flavor"] != "flat") and (nd["flavor"] != "arc"):
#             continue
#         band = nd["band"]
#         spec = nd["spec"]
#         bootname = graph_name(rawnight, "psfboot-{}{}".format(band, spec))
#         grph[bootname]["in"].append(name)
#         nd["out"].append(bootname)
#
#     # Next is full PSF estimation.  Inputs are the arc image and the bootcalib
#     # output file.  We also add nodes for the combined psfs.
#
#     for band in ["b", "r", "z"]:
#         for spec in keep:
#             name = graph_name(rawnight, "psfnight-{}{}".format(band, spec))
#             node = {}
#             node["type"] = "psfnight"
#             node["band"] = band
#             node["spec"] = spec
#             node["in"] = []
#             node["out"] = []
#             grph[name] = node
#
#     #- cache current graph items so we can update graph as we go
#     current_items = list(grph.items())
#     for name, nd in current_items:
#         if nd["type"] != "pix":
#             continue
#         if nd["flavor"] != "arc":
#             continue
#         band = nd["band"]
#         spec = nd["spec"]
#         id = nd["id"]
#         bootname = graph_name(rawnight, "psfboot-{}{}".format(band, spec))
#         psfname = graph_name(rawnight, "psf-{}{}-{:08d}".format(band, spec, id))
#         psfnightname = graph_name(rawnight, "psfnight-{}{}".format(band, spec))
#         node = {}
#         node["type"] = "psf"
#         node["band"] = band
#         node["spec"] = spec
#         node["id"] = id
#         node["in"] = [name, bootname]
#         node["out"] = [psfnightname]
#         grph[psfname] = node
#         grph[bootname]["out"].append(psfname)
#         grph[psfnightname]["in"].append(psfname)
#         nd["out"].append(psfname)
#
#     # Now we extract the flats and science frames using the nightly psf
#
#     #- cache current graph items so we can update graph as we go
#     current_items = list(grph.items())
#     for name, nd in current_items:
#         if nd["type"] != "pix":
#             continue
#         if nd["flavor"] == "arc":
#             continue
#         band = nd["band"]
#         spec = nd["spec"]
#         id = nd["id"]
#         flavor = nd["flavor"]
#         framename = graph_name(rawnight, "frame-{}{}-{:08d}".format(band, spec, id))
#         psfnightname = graph_name(rawnight, "psfnight-{}{}".format(band, spec))
#         fmname = graph_name(rawnight, "fibermap-{:08d}".format(id))
#         node = {}
#         node["type"] = "frame"
#         node["band"] = band
#         node["spec"] = spec
#         node["id"] = id
#         node["flavor"] = flavor
#         node["in"] = [name, fmname, psfnightname]
#         node["out"] = []
#         grph[framename] = node
#         grph[psfnightname]["out"].append(framename)
#         grph[fmname]["out"].append(framename)
#         nd["out"].append(framename)
#
#     # Now build the fiberflats for each flat exposure.  We keep a list of all
#     # available fiberflats while we are looping over them, since we"ll need
#     # that in the next step to select the "most recent" fiberflat.
#
#     flatexpid = {}
#
#     #- cache current graph items so we can update graph as we go
#     current_items = list(grph.items())
#     for name, nd in current_items:
#         if nd["type"] != "frame":
#             continue
#         if nd["flavor"] != "flat":
#             continue
#         band = nd["band"]
#         spec = nd["spec"]
#         id = nd["id"]
#         flatname = graph_name(rawnight, "fiberflat-{}{}-{:08d}".format(band, spec, id))
#         node = {}
#         node["type"] = "fiberflat"
#         node["band"] = band
#         node["spec"] = spec
#         node["id"] = id
#         node["in"] = [name]
#         node["out"] = []
#         grph[flatname] = node
#         nd["out"].append(flatname)
#         cam = "{}{}".format(band, spec)
#         if cam not in flatexpid:
#             flatexpid[cam] = []
#         flatexpid[cam].append(id)
#
#     # This is a small helper function to return the "most recent fiberflat"
#     # that came before the current exposure.
#
#     def last_flat(cam, expid):
#         flatid = None
#         flatname = None
#         if cam in flatexpid:
#             for fid in sorted(flatexpid[cam]):
#                 if (flatid is None):
#                     flatid = fid
#                 elif (fid > flatid) and (fid < id):
#                     flatid = fid
#         if flatid is not None:
#             flatname = graph_name(rawnight, "fiberflat-{}{}-{:08d}".format(band,
#                 spec, fid))
#         else:
#             # This means we don't have any flats for this night.
#             # Probably this is because we are going to inject
#             # already-calibrated simulation data into the production.
#             # If this was really a mistake, then it will be caught
#             # at runtime when the sky step fails.
#             pass
#         return flatid, flatname
#
#     #- cache current graph items so we can update graph as we go
#     current_items = list(grph.items())
#     for name, nd in current_items:
#         if nd["type"] != "frame":
#             continue
#         if nd["flavor"] == "flat":
#             continue
#         band = nd["band"]
#         spec = nd["spec"]
#         id = nd["id"]
#         cam = "{}{}".format(band, spec)
#         flatid, flatname = last_flat(cam, id)
#         skyname = graph_name(rawnight, "sky-{}{}-{:08d}".format(band, spec, id))
#         node = {}
#         node["type"] = "sky"
#         node["band"] = band
#         node["spec"] = spec
#         node["id"] = id
#         node["in"] = [name]
#         if flatname is not None:
#             node["in"].append(flatname)
#             grph[flatname]["out"].append(skyname)
#         node["out"] = []
#         grph[skyname] = node
#         nd["out"].append(skyname)
#
#     # Construct the standard star files.  These are one per spectrograph,
#     # and depend on the frames and the corresponding flats and sky files.
#
#     stdgrph = {}
#
#     #- cache current graph items so we can update graph as we go
#     current_items = list(grph.items())
#     for name, nd in current_items:
#         if nd["type"] != "frame":
#             continue
#         if nd["flavor"] == "flat":
#             continue
#         band = nd["band"]
#         spec = nd["spec"]
#         id = nd["id"]
#
#         starname = graph_name(rawnight, "stdstars-{}-{:08d}".format(spec, id))
#         # does this spectrograph exist yet in the graph?
#         if starname not in stdgrph:
#             fmname = graph_name(rawnight, "fibermap-{:08d}".format(id))
#             grph[fmname]["out"].append(starname)
#             node = {}
#             node["type"] = "stdstars"
#             node["spec"] = spec
#             node["id"] = id
#             node["in"] = [fmname]
#             node["out"] = []
#             stdgrph[starname] = node
#
#         cam = "{}{}".format(band, spec)
#         flatid, flatname = last_flat(cam, id)
#         skyname = graph_name(rawnight, "sky-{}{}-{:08d}".format(band, spec, id))
#         stdgrph[starname]["in"].extend([skyname, name])
#         if flatname is not None:
#             stdgrph[starname]["in"].extend([flatname])
#             grph[flatname]["out"].append(starname)
#         nd["out"].append(starname)
#         grph[skyname]["out"].append(starname)
#
#     grph.update(stdgrph)
#
#     # Construct calibration files
#
#     #- cache current graph items so we can update graph as we go
#     current_items = list(grph.items())
#     for name, nd in current_items:
#         if nd["type"] != "frame":
#             continue
#         if nd["flavor"] == "flat":
#             continue
#         band = nd["band"]
#         spec = nd["spec"]
#         id = nd["id"]
#         cam = "{}{}".format(band, spec)
#         flatid, flatname = last_flat(cam, id)
#         skyname = graph_name(rawnight, "sky-{}{}-{:08d}".format(band, spec, id))
#         starname = graph_name(rawnight, "stdstars-{}-{:08d}".format(spec, id))
#         calname = graph_name(rawnight, "calib-{}{}-{:08d}".format(band, spec, id))
#         node = {}
#         node["type"] = "calib"
#         node["band"] = band
#         node["spec"] = spec
#         node["id"] = id
#         node["in"] = [name, skyname, starname]
#         if flatname is not None:
#             node["in"].extend([flatname])
#             grph[flatname]["out"].append(calname)
#         node["out"] = []
#         grph[calname] = node
#         grph[skyname]["out"].append(calname)
#         grph[starname]["out"].append(calname)
#         nd["out"].append(calname)
#
#     # Build cframe files
#
#     #- cache current graph items so we can update graph as we go
#     current_items = list(grph.items())
#     for name, nd in current_items:
#         if nd["type"] != "frame":
#             continue
#         if nd["flavor"] == "flat":
#             continue
#         band = nd["band"]
#         spec = nd["spec"]
#         id = nd["id"]
#         cam = "{}{}".format(band, spec)
#         flatid, flatname = last_flat(cam, id)
#         skyname = graph_name(rawnight, "sky-{}{}-{:08d}".format(band, spec, id))
#         calname = graph_name(rawnight, "calib-{}{}-{:08d}".format(band, spec, id))
#         cfname = graph_name(rawnight, "cframe-{}{}-{:08d}".format(band, spec, id))
#         node = {}
#         node["type"] = "cframe"
#         node["band"] = band
#         node["spec"] = spec
#         node["id"] = id
#         node["in"] = [name, skyname, calname]
#         if flatname is not None:
#             node["in"].extend([flatname])
#             grph[flatname]["out"].append(cfname)
#         node["out"] = []
#         grph[cfname] = node
#         grph[skyname]["out"].append(cfname)
#         grph[calname]["out"].append(cfname)
#         nd["out"].append(cfname)
#
#     # Spectra / Zbest dependencies
#
#     for p in allpix:
#         zname = "zbest-{}-{}".format(hpxnside, p)
#         sname = "spectra-{}-{}".format(hpxnside, p)
#
#         node = {}
#         node["type"] = "spectra"
#         node["nside"] = hpxnside
#         node["pixel"] = p
#         node["in"] = []
#         node["out"] = [zname]
#         grph[sname] = node
#
#         node = {}
#         node["type"] = "zbest"
#         node["nside"] = hpxnside
#         node["pixel"] = p
#         node["ntarget"] = allpix[p]
#         node["in"] = [sname]
#         node["out"] = []
#         grph[zname] = node
#
#     #- cache current graph items so we can update graph as we go
#     current_items = list(grph.items())
#     for name, nd in current_items:
#         if nd["type"] != "fibermap":
#             continue
#         if nd["flavor"] == "arc":
#             continue
#         if nd["flavor"] == "flat":
#             continue
#         id = nd["id"]
#         fmpix = nd["pixels"]
#         for band in ["b", "r", "z"]:
#             for spec in keep:
#                 cfname = graph_name(rawnight, "cframe-{}{}-{:08d}".format(band, spec, id))
#                 for p in fmpix:
#                     sname = "spectra-{}-{}".format(hpxnside, p)
#                     grph[sname]["in"].append(cfname)
#                     grph[cfname]["out"].append(sname)
#
#     return (grph, expcount, allpix)



def check_tasks(tasklist, db=None, inputs=None):
    """Check a list of tasks and return their state.

    If the database is specified, it is used to check the state of the tasks
    and their dependencies.  Otherwise the filesystem is checked.

    Args:
        tasklist (list): list of tasks.
        db (pipeline.db.DB): The optional database to use.
        inputs (dict): optional dictionary containing the only input
            dependencies that should be considered.

    Returns:
        dict: The current state of all tasks.

    """
    from .tasks.base import task_classes, task_type
    states = dict()

    if db is None:
        # Check the filesystem to see which tasks are done.  Since we don't
        # have a DB, we can only distinguish between "waiting", "ready", and
        # "done" states.
        for tsk in tasklist:
            tasktype = task_type(tsk)
            st = "waiting"

            # Check dependencies
            ready = True
            deps = task_classes[tasktype].deps(tsk, db=db, inputs=inputs)
            for k, v in deps.items():
                if not isinstance(v, list):
                    v = [ v ]
                for dp in v:
                    deptype = task_type(dp)
                    depfiles = task_classes[deptype].paths(dp)
                    for odep in depfiles:
                        if not os.path.isfile(odep):
                            ready = False
                            break
            if ready:
                st = "ready"
                done = True
                # Check outputs
                outfiles = task_classes[tasktype].paths(tsk)
                for out in outfiles:
                    if not os.path.isfile(out):
                        done = False
                        break
                if done:
                    st = "done"

            states[tsk] = st
    else:
        states = db.get_states(tasklist)

    return states


class DataBase:
    """Class for tracking pipeline processing objects and state.
    """
    def __init__(self):
        self.conn = None
        return


    def get_states_type(self, tasktype, tasks):
        """Efficiently get the state of many tasks of a single type.

        Args:
            tasktype (str): the type of these tasks.
            tasks (list): list of task names.

        Returns:
            dict: the state of each task.

        """
        states = None
        namelist = ",".join([ '"{}"'.format(x) for x in tasks ])

        log = get_logger()

        log.debug("opening db")
        with self.conn as cn:
            cur = cn.cursor()
            log.debug("selecting in db")
            cur.execute(\
                'select name, state from {} where name in ({})'.format(tasktype,
                namelist))
            st = cur.fetchall()
            log.debug("done")
            states = { x[0] : task_int_to_state[x[1]] for x in st }
        return states


    def get_states(self, tasks):
        """Efficiently get the state of many tasks at once.

        Args:
            tasks (list): list of task names.

        Returns:
            dict: the state of each task.

        """
        from .tasks.base import task_classes, task_type
        # First find the type of each task.
        ttypes = dict()
        for tsk in tasks:
            ttypes[tsk] = task_type(tsk)

        # Sort tasks into types, so that we can build one query per type.
        taskbytype = dict()
        for t in task_types():
            taskbytype[t] = list()
        for tsk in tasks:
            taskbytype[ttypes[tsk]].append(tsk)

        # Query the list of tasks of each type

        states = dict()
        for t, tlist in taskbytype.items():
            if len(tlist) > 0:
                states.update(self.get_states_type(t, tlist))

        return states


    def set_states_type(self, tasktype, tasks):
        """Efficiently get the state of many tasks of a single type.

        Args:
            tasktype (str): the type of these tasks.
            tasks (list): list of tuples containing the task name and the
                state to set.

        Returns:
            Nothing.

        """
        states = None
        namelist = ",".join([ '"{}"'.format(x) for x in tasks ])

        log = get_logger()

        log.debug("opening db")
        with self.conn as cn:
            cur = cn.cursor()
            log.debug("updating in db")
            cur.execute('begin transaction')
            for tsk in tasks:
                cur.execute('update {} set state = {} where name = "{}"'.format(tasktype, task_state_to_int[tsk[1]], tsk[0]))
            log.debug("done")
        return


    def set_states(self, tasks):
        """Efficiently set the state of many tasks at once.

        Args:
            tasks (list): list of tuples containing the task name and the
                state to set.

        Returns:
            Nothing.

        """
        from .tasks.base import task_classes, task_type
        # First find the type of each task.
        ttypes = dict()
        for tsk in tasks:
            ttypes[tsk[0]] = task_type(tsk[0])

        # Sort tasks into types
        taskbytype = dict()
        for t in task_types():
            taskbytype[t] = list()
        for tsk in tasks:
            taskbytype[ttypes[tsk[0]]].append(tsk)

        # Process each type
        for t, tlist in taskbytype.items():
            if len(tlist) > 0:
                self.set_states_type(t, tlist)
        return


    def update(self, night, nside):
        """Update DB based on raw data.

        This will use the usual io.meta functions to find raw exposures.  For
        each exposure, the fibermap and all following objects will be added to
        the DB.

        Args:
            night (str): The night to scan for updates.

            nside (int): The current NSIDE value used for pixel grouping.
        """
        from .tasks.base import task_classes, task_type

        log = get_logger()

        # Update DB tables, if needed
        self.initdb()

        alltasks = all_tasks(night, nside)

        with self.conn as con:
            cur = con.cursor()

            # read what is already in db
            tasks_in_db = {}
            for tt in task_types():
                if tt == "spectra" : continue
                cur.execute(\
                            "select name from {} where night={}"\
                            .format(tt, night))
                tasks_in_db[tt] = [ x for (x, ) in cur.fetchall()]

            cur.execute("begin transaction")
            for tt in task_types():
                log.debug("updating {} ...".format(tt))
                for tsk in alltasks[tt]:
                    tname = task_classes[tt].name_join(tsk)
                    if tname not in tasks_in_db[tt] :
                        log.debug("adding {}".format(tname))
                        task_classes[tt].insert(cur, tsk)

            cur.execute("commit")

        return


    def sync(self, night):
        """Update states of tasks based on filesystem.

        Go through all tasks in the DB for the given night and determine their
        state on the filesystem.  Then update the DB state to match.

        Args:
            night (str): The night to scan for updates.

        """
        # Update DB tables, if needed
        self.initdb()

        tasks_in_db = None
        # Grab existing nightly tasks
        with self.conn as cn:
            cur = cn.cursor()
            tasks_in_db = {}
            for tt in task_types():
                if tt == "spectra":
                    continue
                cur.execute(\
                            "select name from {} where night = {}"\
                            .format(tt, night))
                tasks_in_db[tt] = [ x for (x, ) in cur.fetchall() ]

        # FIXME: for spectra and zbest tasks, we should use the extra tables
        # to check only the files touched by this night's data.

        # For each task type, check status WITHOUT the DB, then set state.
        for tt in task_types():
            tstates = check_tasks(tasks_in_db[tt], db=None)
            st = [ (x, tstates[x]) for x in tasks_in_db[tt] ]
            self.set_states_type(tt, st)

        return


    def getready(self):
        """Update DB, changing waiting to ready depending on status of dependencies .

        """
        from .tasks.base import task_classes, task_type

        log = get_logger()


        with self.conn as con:

            cur = con.cursor()

            for tt in task_types():

                # for each type of task, get the list of tasks in waiting mode
                cur.execute('select name from {} where state={}'.format(tt,task_state_to_int["waiting"]))
                tasks = [ x for (x, ) in cur.fetchall()]

                if len(tasks)>0 :
                    log.debug("checking {} {} tasks ...".format(len(tasks),tt))

                for tsk in tasks :
                    # for each task in waiting mode, get the dependencies
                    deps = task_classes[tt].deps(tsk,db=self,inputs=None)
                    ready = True
                    for dep in deps :
                        # for each dependency, guess its type
                        deptype  = dep.split(task_name_sep)[0]
                        # based on the type and dependency name, read state from db
                        depstate =  task_classes[deptype].state_get(db=self,name=dep)
                        ready   &=  (depstate=="done") # ready if all dependencies are done
                    if ready :
                        # change state to ready
                        log.debug("{} is ready to run".format(tsk))
                        task_classes[tt].state_set(db=self,name=tsk,state="ready")

        return


class DataBaseSqlite(DataBase):
    """Pipeline database using sqlite3 as the backend.

    Args:
        path (str): the filesystem path of the database to open.  If None, then
            a temporary database is created in memory.
        mode (str): if "r", the database is open in read-only mode.  If "w",
            the database is open in read-write mode and created if necessary.

    """
    def __init__(self, path, mode):
        super(DataBaseSqlite, self).__init__()

        self._path = path
        self._mode = mode

        create = True
        if (self._path is not None) and os.path.exists(self._path):
            create = False

        if self._mode == 'r' and create:
            raise RuntimeError("cannot open a non-existent DB in read-only "
                " mode")

        self.connstr = None

        # This timeout is in seconds
        self._busytime = 1000

        # Journaling options
        self._journalmode = "persist"
        self._syncmode = "normal"

        self._open()

        if create:
            self.initdb()
        return


    def _open(self):
        import sqlite3

        if self._path is None:
            # We are opening an in-memory DB
            self.conn = sqlite3.connect(":memory:")
        else:
            try:
                # only python3 supports uri option
                if self._mode == 'r':
                    self.connstr = 'file:{}?mode=ro'.format(self._path)
                else:
                    self.connstr = 'file:{}?mode=rwc'.format(self._path)
                self.conn = sqlite3.connect(self.connstr, uri=True,
                    timeout=self._busytime)
            except:
                self.conn = sqlite3.connect(self._path, timeout=self._busytime)
        if self._mode == 'w':
            # In read-write mode, set the journaling
            self.conn.execute("pragma journal_mode={}"\
                .format(self._journalmode))
            self.conn.execute("pragma synchronous={}".format(self._syncmode))

        # Other tuning options
        self.conn.execute("pragma temp_store=memory")
        self.conn.execute("pragma page_size=4096")
        self.conn.execute("pragma cache_size=4000")
        return


    def initdb(self):
        """Create DB tables for all tasks if they do not exist.
        """
        # check existing tables
        tables_in_db = None
        with self.conn as cn:
            cur = cn.cursor()
            cur.execute('select name FROM sqlite_master WHERE type="table"')
            tables_in_db = [x for (x, ) in cur.fetchall()]

        # Create a table for every task type
        from .tasks.base import task_classes, task_type
        for tt, tc in task_classes.items():
            if tt not in tables_in_db:
                tc.create(self)
        return


class DataBasePostgres(DataBase):
    """Pipeline database using PostgreSQL as the backend.

    Args:
        host (str): The database server.
        port (int): The connection port.
        dbname (str): The database to connect.
        user (str): The user name for the connection.  The password should be
            stored in the ~/.pgpass file.
        schema (str): The schema within the database.  If this is specified,
            then the database is assumed to exist.  Otherwise the schema is
            computed from a hash of the production location and will be
            created.

    """
    def __init__(self, host, port, dbname, user, schema=None):
        super(DataBasePostgres, self).__init__()

        self._schema = schema
        self._user = user
        self._dbname = dbname
        self._host = host
        self._port = port

        self._proddir = os.path.abspath(io.specprod_root())

        self._open()
        return


    def _compute_schema(self):
        import hashlib
        md = hashlib.md5()
        md.update(self._proddir.encode())
        return "pipe_{}".format(md.hexdigest())


    def _open(self):
        import psycopg2 as pg2

        create = False
        if self._schema is None:
            create = True
            self._schema = self._compute_schema()

        # Open connection
        self.conn = pg2.connect(host=self._host, port=self._port,
            user=self._user, dbname=self._dbname)

        # Check existence of the schema.  If we were not passed the schema
        # in the constructor, it means that we are creating a new prod, so any
        # existing schema should be wiped and recreated.

        with self.conn as cn:
            cur = cn.cursor()
            # See if our schema already exists...
            com = "select exists(select 1 from pg_namespace where nspname = '{}')".format(self._schema)
            cur.execute(com)
            have_schema = cur.fetchone()[0]
            print("have_schema =",have_schema)

            if create:
                if have_schema:
                    # We need to wipe it first
                    com = "drop schema {} cascade".format(self._schema)
                    print(com,flush=True)
                    cur.execute(com)
                com = "create schema {} authorization {}"\
                    .format(self._schema, self._user)
                print(com,flush=True)
                cur.execute(com)
                # Create a table of information about this prod
                com = "create table {}.info (key text unique, val text)"\
                    .format(self._schema)
                print(com,flush=True)
                cur.execute(com)
                com = "insert into {}.info values ('{}', '{}')"\
                    .format(self._schema, "path", self._proddir)
                print(com,flush=True)
                cur.execute(com)
                if 'USER' in os.environ:
                    com = "insert into {}.info values ('{}', '{}')"\
                        .format(self._schema, "created_by", os.environ['USER'])
                    print(com,flush=True)
                    cur.execute(com)
            else:
                if not have_schema:
                    raise RuntimeError("Postgres schema for production {} does"
                        " not exist.  Make sure you create the production with"
                        " postgres options and source the top-level setup.sh"
                        " file.".format(self._proddir))
            com = "set search_path to {}".format(self._schema)
            print(com,flush=True)
            cur.execute(com)

        # If we are creating the prod, initialize the tables.
        if create:
            self.initdb()
        return


    @property
    def schema(self):
        return self._schema


    def initdb(self):
        """Create DB tables for all tasks if they do not exist.
        """
        # check existing tables
        tables_in_db = None
        with self.conn as cn:
            cur = cn.cursor()
            cur.execute('select tablename from pg_tables where schemaname =  "{}"'.format(self.schema))
            tables_in_db = [x for (x, ) in cur.fetchall()]

        # Create a table for every task type
        from .tasks.base import task_classes, task_type
        for tt, tc in task_classes.items():
            if tt not in tables_in_db:
                tc.create(self)
        return


def load_db(dbstring, mode="w", user=None):
    """Load a database from a connection string.

    This instantiates either an sqlite or postgresql database using a string.
    If this string begins with "postgresql:", then it is taken to be the
    information needed to connect to a postgres server.  Otherwise it is
    assumed to be a filesystem path to use with sqlite.  The mode is only
    meaningful when using sqlite.  Postgres permissions are controlled through
    the user permissions.

    Args:
        dbstring (str): either a filesystem path (sqlite) or a colon-separated
            string of connection properties in the form
            "postresql:<host>:<port>:<dbname>:<user>:<schema>".
        mode (str): for sqlite, the mode.
        user (str): for postgresql, an alternate user name for opening the DB.
            This can be used to connect as a user with read-only access.

    Returns:
        DataBase: a derived database class of the appropriate type.

    """
    if re.search(r"postgresql:", dbstring) is not None:
        props = dbstring.split(":")
        host = props[1]
        port = int(props[2])
        dbname = props[3]
        username = props[4]
        if user is not None:
            username = user
        schema = None
        if len(props) > 5:
            # Our DB string also contains the name of an existing
            # schema.
            schema = props[5]
        return DataBasePostgres(host=host, port=port, dbname=dbname,
            user=username, schema=schema)
    else:
        return DataBaseSqlite(dbstring, mode)
