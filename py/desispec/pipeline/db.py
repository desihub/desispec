#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.db
====================

Pipeline processing database
"""

from __future__ import absolute_import, division, print_function

import os

import re
from collections import OrderedDict

from contextlib import contextmanager

import numpy as np

from desiutil.log import get_logger

from .. import io

import fitsio

from .defs import (task_states, task_int_to_state, task_state_to_int, task_name_sep)


def all_task_types():
    """Get the list of possible task types that are supported.

    Returns:
        list: The list of supported task types.

    """
    from . import tasks
    from .tasks.base import default_task_chain
    ttypes = ["fibermap", "rawdata"]
    ttypes.extend(tasks.base.default_task_chain)
    # Insert qadata after cframe
    idx = ttypes.index('cframe')
    ttypes.insert(idx+1, 'qadata')
    return ttypes


def task_sort(tasks):
    """Sort a list of tasks by type.

    This takes a list of arbitrary tasks and sorts them by type.  The result
    is placed in an ordered dictionary of lists in run order.

    Args:
        tasks (list): the list of input tasks.

    Returns:
        (OrderedDict): ordered dictionary of tasks sorted by type.

    """
    from .tasks.base import task_classes, task_type
    sort = dict()
    ttypes = all_task_types()
    for tp in ttypes:
        sort[tp] = list()

    for tsk in tasks:
        sort[task_type(tsk)].append(tsk)

    ret = OrderedDict()
    for tp in ttypes:
        if len(sort[tp]) > 0:
            ret[tp] = sort[tp]
    return ret


def all_tasks(night, nside, expid=None):
    """Get all possible tasks for a single night.

    This uses the filesystem to query the raw data for a particular night and
    return a dictionary containing all possible tasks for each task type.  For
    objects which span multiple nights (e.g. spectra, redrock), this returns the
    tasks which are touched by the given night.

    Args:
        night (str): The night to scan for tasks.
        nside (int): The HEALPix NSIDE value to use.
        expid (int): Only get tasks for this single exposure.

    Returns:
        dict: a dictionary whose keys are the task types and where each value
            is a list of task properties.

    """
    import desimodel.footprint

    log = get_logger()

    log.debug("io.get_exposures night={}".format(night))

    expids = io.get_exposures(night, raw=True)

    full = dict()
    for t in all_task_types():
        full[t] = list()

    healpix_frames = []

    if expid is not None:
        if expid not in expids:
            raise RuntimeError("exposure ID {} not valid for night {}"\
                               .format(expid, night))
        expids = [ expid ]

    for ex in sorted(expids):

        # get the fibermap for this exposure
        fibermap = io.get_raw_files("fibermap", night, ex)

        log.debug("read {}".format(fibermap))

        fmdata = io.read_fibermap(fibermap)
        header = fmdata.meta

        # fmdata, header = fitsio.read(fibermap, 'FIBERMAP', header=True)
        flavor = header["FLAVOR"].strip().lower()
        if flavor not in ["arc","flat","science"] :
            log.error("Do not know what do to with fibermap flavor '{}' for file '{}".format(flavor,fibermap))
            raise ValueError("Do not know what do to with fibermap flavor '{}' for file '{}".format(flavor,fibermap))

        fmpix = dict()
        if (flavor != "arc") and (flavor != "flat"):
            # This will be used to track which healpix pixels are
            # touched by fibers from each spectrograph.
            ra = np.array(fmdata["TARGET_RA"], dtype=np.float64)
            dec = np.array(fmdata["TARGET_DEC"], dtype=np.float64)

            # rm NaN (possible depending on versions of fiberassign)
            valid_coordinates  = (np.isnan(ra)==False)&(np.isnan(dec)==False)

            for spectro in np.unique( fmdata["SPECTROID"] ) :
                ii=np.where(fmdata["SPECTROID"][valid_coordinates]==spectro)[0]
                if ii.size == 0 : continue
                pixels  = desimodel.footprint.radec2pix(nside, ra[valid_coordinates][ii], dec[valid_coordinates][ii])
                for pixel in np.unique(pixels) :
                    props = dict()
                    props["night"] = int(night)
                    props["expid"] = int(ex)
                    props["spec"]  = spectro
                    props["nside"] = nside
                    props["pixel"] = pixel
                    props["ntargets"] = np.sum(pixels==pixel)
                    healpix_frames.append(props)
            # all spectro at once
            pixels  = np.unique(desimodel.footprint.radec2pix(nside, ra[valid_coordinates], dec[valid_coordinates]))
            for pixel in pixels :
                props = dict()
                props["pixel"] = pixel
                props["nside"] = nside
                props["state"]  = "waiting"
                exists=False
                for entry in full["spectra"] :
                    if entry["pixel"]==props["pixel"] :
                        exists=True
                        break
                if not exists : full["spectra"].append(props)
                exists=False
                for entry in full["redshift"] :
                    if entry["pixel"]==props["pixel"] :
                        exists=True
                        break
                if not exists : full["redshift"].append(props)

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
        for band in ['b', 'r', 'z']:
            # need to open the rawdata file to see how many spectros
            # and cameras are there
            for spec in np.unique( fmdata["SPECTROID"] ) :
                pixprops = dict()
                pixprops["night"] = int(night)
                pixprops["band"] = band
                pixprops["spec"] = spec
                pixprops["expid"] = int(ex)
                pixprops["flavor"] = flavor
                pixprops["state"] = "ready"
                full["preproc"].append(pixprops)

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

                    # Add traceshift
                    full["traceshift"].append(props)

                    # Add extractions
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
                    # Add a fiberflat night file if does not exist
                    exists=False
                    for entry in full["fiberflatnight"] :
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
                         full["fiberflatnight"].append(props)

                if flavor != "arc" and flavor != "flat":
                    # Add sky
                    props = dict()
                    props["night"] = int(night)
                    props["band"] = band
                    props["spec"] = spec
                    props["expid"] = int(ex)
                    props["state"] = "waiting" # see defs.task_states
                    full["sky"].append(props)
                    # Add fluxcalib
                    full["fluxcalib"].append(props)
                    # Add cframe
                    full["cframe"].append(props)
                    # Add QA
                    full["qadata"].append(props)

                    # Add starfit if does not exist
                    exists=False
                    for entry in full["starfit"] :
                        if entry["night"]==props["night"] \
                           and entry["expid"]==props["expid"] \
                           and entry["spec"]==props["spec"] :
                            exists=True
                            break
                    if not exists :
                         props = dict()
                         props["night"] = int(night)
                         props["expid"] = int(ex)
                         props["spec"] = spec
                         props["state"] = "waiting" # see defs.task_states
                         full["starfit"].append(props)

    log.debug("done")
    return full , healpix_frames


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
            deps = task_classes[tasktype].deps(tsk, db=db, inputs=inputs)

            if len(deps)==0 :
                # do not set state to ready of tasks with 0 dependencies
                ready = False
            else :
                ready = True
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
        self._conn = None
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
        namelist = ",".join([ "'{}'".format(x) for x in tasks ])

        log = get_logger()
        log.debug("opening db")

        with self.cursor() as cur:
            log.debug("selecting in db")
            cur.execute(\
                'select name, state from {} where name in ({})'.format(tasktype,
                namelist))
            st = cur.fetchall()
            log.debug("done")
            states = { x[0] : task_int_to_state[x[1]] for x in st }
        return states


    def count_task_states(self, tasktype):
        """Return a dictionary of how many tasks are in each state

        Args:
            tasktype (str): the type of these tasks.

        Returns:
            dict: keyed by state, values are number of tasks in that state0
        """
        state_count = OrderedDict()
        for state in task_states:
            state_count[state] = 0

        with self.cursor() as cur:
            cur.execute( 'select name, state from {}'.format(tasktype))
            for name, intstate in cur.fetchall():
                state_count[task_int_to_state[intstate]] += 1

        return state_count


    def get_states(self, tasks):
        """Efficiently get the state of many tasks at once.

        Args:
            tasks (list): list of task names.

        Returns:
            dict: the state of each task.

        """
        from .tasks.base import task_classes, task_type

        # Sort by type
        taskbytype = task_sort(tasks)

        # Get state of each type
        states = dict()
        for t, tlist in taskbytype.items():
            states.update(self.get_states_type(t, tlist))

        return states


    def set_states_type(self, tasktype, tasks, postprocessing=True):
        """Efficiently get the state of many tasks of a single type.

        Args:
            tasktype (str): the type of these tasks.
            tasks (list): list of tuples containing the task name and the
                state to set.

        Returns:
            Nothing.

        """
        from .tasks.base import task_classes

        log = get_logger()
        log.debug("opening db")

        with self.cursor() as cur:
            log.debug("updating in db")
            for tsk in tasks:
                cur.execute("update {} set state = {} where name = '{}'".format(tasktype, task_state_to_int[tsk[1]], tsk[0]))
                if postprocessing and tsk[1]=="done" :
                    task_classes[tasktype].postprocessing(db=self,name=tsk[0],cur=cur)
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
        for t in all_task_types():
            taskbytype[t] = list()
        for tsk in tasks:
            taskbytype[ttypes[tsk[0]]].append(tsk)

        # Process each type
        for t, tlist in taskbytype.items():
            if len(tlist) > 0:
                self.set_states_type(t, tlist)
        return


    def get_submitted(self, tasks):
        """Return the submitted flag for the list of tasks.

        Args:
            tasks (list): list of task names.

        Returns:
            (dict): the boolean submitted state of each task (True means that
                the task has been submitted).

        """
        from .tasks.base import task_type
        # Sort by type
        taskbytype = task_sort(tasks)

        # Process each type
        submitted = dict()
        for t, tlist in taskbytype.items():
            if (t == "spectra") or (t == "redshift"):
                raise RuntimeError("spectra and redshift tasks do not have submitted flag.")
            namelist = ",".join([ "'{}'".format(x) for x in tlist ])
            with self.cursor() as cur:
                cur.execute(\
                    'select name, submitted from {} where name in ({})'.format(t, namelist))
                sb = cur.fetchall()
                submitted.update({ x[0] : x[1] for x in sb })
        return submitted


    def set_submitted_type(self, tasktype, tasks, unset=False):
        """Flag a list of tasks of a single type as submitted.

        Args:
            tasktype (str): the type of these tasks.
            tasks (list): list of task names.
            unset (bool): if True, invert the behavior and unset the submitted
                flag for these tasks.

        Returns:
            Nothing.

        """
        val = 1
        if unset:
            val = 0
        with self.cursor() as cur:
            for tsk in tasks:
                cur.execute("update {} set submitted = {} where name = '{}'".format(tasktype, val, tsk))
        return


    def set_submitted(self, tasks, unset=False):
        """Flag a list of tasks as submitted.

        Args:
            tasks (list): list of task names.
            unset (bool): if True, invert the behavior and unset the submitted
                flag for these tasks.

        Returns:
            Nothing.

        """
        from .tasks.base import task_type
        # Sort by type
        taskbytype = task_sort(tasks)

        # Process each type
        for t, tlist in taskbytype.items():
            if (t == "spectra") or (t == "redshift"):
                raise RuntimeError("spectra and redshift tasks do not have submitted flag.")
            self.set_submitted_type(tlist, unset=unset)
        return


    def update(self, night, nside, expid=None):
        """Update DB based on raw data.

        This will use the usual io.meta functions to find raw exposures.  For
        each exposure, the fibermap and all following objects will be added to
        the DB.

        Args:
            night (str): The night to scan for updates.
            nside (int): The current NSIDE value used for pixel grouping.
            expid (int): Only update the DB for this exposure.

        """
        from .tasks.base import task_classes, task_type

        log = get_logger()

        alltasks, healpix_frames = all_tasks(night, nside, expid=expid)

        with self.cursor() as cur:
            # insert or ignore all healpix_frames
            log.debug("updating healpix_frame ...")
            for entry in healpix_frames:
                # see if we already have this entry
                cmd = "select exists(select 1 from healpix_frame where (expid = {} and spec = {} and nside = {} and pixel = {} ))".format(entry["expid"], entry["spec"], entry["nside"], entry["pixel"])
                cur.execute(cmd)
                have_row = cur.fetchone()[0]

                if not have_row:
                    cur.execute("insert into healpix_frame (night,expid,spec,nside,pixel,ntargets,state) values({},{},{},{},{},{},{})".format(entry["night"],entry["expid"],entry["spec"],entry["nside"],entry["pixel"],entry["ntargets"],0))

            # read what is already in db
            tasks_in_db = {}
            for tt in all_task_types():
                cur.execute("select name from {}".format(tt))
                tasks_in_db[tt] = [ x for (x, ) in cur.fetchall()]

            for tt in all_task_types():
                log.debug("updating {} ...".format(tt))
                for tsk in alltasks[tt]:
                    tname = task_classes[tt].name_join(tsk)
                    if tname not in tasks_in_db[tt] :
                        log.debug("adding {}".format(tname))
                        task_classes[tt].insert(cur, tsk)

        return


    def sync(self, night, specdone=False):
        """Update states of tasks based on filesystem.

        Go through all tasks in the DB for the given night and determine their
        state on the filesystem.  Then update the DB state to match.

        Args:
            night (str): The night to scan for updates.
            specdone: If true, set spectra to done if files exist.
        """
        from .tasks.base import task_classes
        log = get_logger()

        # Get the list of task types excluding spectra and redshifts,
        # which will be handled separately.
        ttypes = [ t for t in all_task_types() if (t != "spectra") \
            and (t != "redshift") ]

        tasks_in_db = None
        # Grab existing nightly tasks
        with self.cursor() as cur:
            tasks_in_db = {}
            for tt in ttypes:
                cur.execute("select name from {} where night = {}"\
                    .format(tt, night))
                tasks_in_db[tt] = [ x for (x, ) in cur.fetchall() ]

        # For each task type, check status WITHOUT the DB, then set state.
        # Save out the cframe states for later use with the healpix_frame table
        cfstates = None
        for tt in ttypes:
            tstates = check_tasks(tasks_in_db[tt], db=None)
            st = [ (x, tstates[x]) for x in tasks_in_db[tt] ]
            self.set_states_type(tt, st)
            if tt == "cframe":
                cfstates = tstates.copy()

        # Now examine the spectra and redshift files.  If the files exist,
        # we assume they are done and completely up to date.  If the files
        # are not up to date, they must be manually deleted in order for the
        # sync to correctly reconstruct the database state.

        pixrows = self.select_healpix_frame({"night" : night})
        # First check the existence of the files touched by this night
        spec_exists = dict()
        red_exists = dict()
        for row in pixrows:
            if row["pixel"] in spec_exists:
                continue
            spec_name = task_classes["spectra"].name_join(row)
            red_name = task_classes["redshift"].name_join(row)

            # Check spectra outputs
            outfiles = task_classes["spectra"].paths(spec_name)
            spec_exists[row["pixel"]] = True
            for out in outfiles:
                if not os.path.isfile(out):
                    spec_exists[row["pixel"]] = False
                    break

            # Check redshift outputs
            outfiles = task_classes["redshift"].paths(red_name)
            red_exists[row["pixel"]] = True
            for out in outfiles:
                if not os.path.isfile(out):
                    red_exists[row["pixel"]] = False
                    break

        # Now use all this info.  Some internal helpers to avoid code
        # duplication
        def set_hpx_frame_0(row, spec, red, cur):
            self.update_healpix_frame_state(row, 0, cur)
            task_classes["spectra"].state_set(
                self, spec, "waiting", cur)
            task_classes["redshift"].state_set(
                self, red, "waiting", cur)
            return

        def set_hpx_frame_1(row, spec, red, cur):
            self.update_healpix_frame_state(row, 1, cur)
            # getready() will do this for us:
            #task_classes["spectra"].state_set(
            #    self, spec, "ready", cur)
            task_classes["redshift"].state_set(
                self, red, "waiting", cur)
            return

        def set_hpx_frame_2(row, spec, red, cur):
            self.update_healpix_frame_state(row, 2, cur)
            task_classes["spectra"].state_set(
                self, spec, "done", cur)
            # getready() will do this:
            #task_classes["redshift"].state_set(
            #    self, red, "ready", cur)
            return

        def set_hpx_frame_3(row, spec, red, cur):
            self.update_healpix_frame_state(row, 3, cur)
            task_classes["spectra"].state_set(
                self, spec, "done", cur)
            task_classes["redshift"].state_set(
                self, red, "done", cur)
            return

        with self.cursor() as cur:
            for row in pixrows:
                cfdone = True
                cfprops = row.copy()
                for band in ["b", "r", "z"]:
                    cfprops["band"] = band
                    cf_name = task_classes["cframe"].name_join(cfprops)
                    if cfstates[cf_name] != "done":
                        cfdone = False

                spec_name = task_classes["spectra"].name_join(row)
                red_name = task_classes["redshift"].name_join(row)

                if (not cfdone) and (not specdone) :
                    # The cframes do not exist, so reset the state of the
                    # spectra and redshift tasks.
                    set_hpx_frame_0(row, spec_name, red_name, cur)
                else:
                    # The cframe exists...
                    if spec_exists[row["pixel"]]:
                        if red_exists[row["pixel"]]:
                            # We are all done (state 3)
                            set_hpx_frame_3(row, spec_name, red_name, cur)
                        else:
                            # We are only at state 2
                            set_hpx_frame_2(row, spec_name, red_name, cur)
                    else:
                        # We are at just at state 1
                        set_hpx_frame_1(row, spec_name, red_name, cur)

        # Update ready state of tasks
        self.getready(night=night)

        return


    def cleanup(self, tasktypes=None, expid=None, cleanfailed=False,
                cleansubmitted=False):
        """Reset states of tasks.

        Any tasks that are marked as "running" will have their
        state reset to "ready".  This can be called if a job dies before
        completing all tasks.

        Args:
            tasktypes (list): if not None, clean up only tasks of these types.
            expid (int): if not None, only clean tasks related to this
                exposure ID.  Note that tasks which are independent of
                an expid (psfnight, fiberflatnight, spectra, redshift)
                will be ignored if this option is given.
            cleanfailed (bool): if True, also reset failed tasks to ready.
            cleansubmitted (bool): if True, set submitted flag to False.

        """
        tasks_running = None

        alltypes = all_task_types()
        ttypes = None
        if tasktypes is None:
            ttypes = alltypes
        else:
            for tt in tasktypes:
                if tt not in alltypes:
                    raise RuntimeError("Cannot clean invalid task type {}"\
                        .format(tt))
            ttypes = tasktypes

        # Grab existing nightly tasks
        with self.cursor() as cur:
            tasks_running = {}
            for tt in ttypes:
                hasexpid = (tt not in ["psfnight", "fiberflatnight", "spectra",
                            "redshift"])
                if hasexpid:
                    # This task type has an expid property.
                    cmd = None
                    if expid is not None:
                        # We are cleaning only a single exposure.
                        cmd = "select name from {} where expid = {} and ( state = {}".format(tt, expid, task_state_to_int["running"])
                    else:
                        # We are cleaning all exposures for this task type.
                        cmd = "select name from {} where ( state = {}".format(tt, task_state_to_int["running"])
                    if cleanfailed:
                        cmd = "{} or state = {} )".format(cmd,
                            task_state_to_int["failed"])
                    else:
                        cmd = "{} )".format(cmd)
                    cur.execute(cmd)
                    tasks_running[tt] = [ x for (x, ) in cur.fetchall() ]
                    if cleansubmitted:
                        if expid is not None:
                            cmd = "update {} set submitted = 0 where expid = {}".format(tt, expid)
                        else:
                            cmd = "update {} set submitted = 0".format(tt)
                        cur.execute(cmd)
                else:
                    # This task type has no concept of an exposure ID
                    if expid is not None:
                        # We specified an exposure ID, which makes no sense
                        # for this task type.  Skip it.
                        tasks_running[tt] = list()
                        continue
                    else:
                        # cleanup this task type.
                        cmd = "select name from {} where ( state = {}".format(tt, task_state_to_int["running"])
                        if cleanfailed:
                            cmd = "{} or state = {} )".format(cmd,
                                task_state_to_int["failed"])
                        else:
                            cmd = "{} )".format(cmd)
                        cur.execute(cmd)
                        tasks_running[tt] = [ x for (x, ) in cur.fetchall() ]
                        if cleansubmitted:
                            if (tt != "spectra") and (tt != "redshift"):
                                cmd = "update {} set submitted = 0".format(tt)
                                cur.execute(cmd)

        for tt in ttypes:
            if len(tasks_running[tt]) > 0:
                st = [ (x, "waiting") for x in tasks_running[tt] ]
                self.set_states_type(tt, st)

        self.getready()

        return


    def getready(self, night=None):
        """Update DB, changing waiting to ready depending on status of dependencies .

        Args:
            night (str): The night to process.

        """
        from .tasks.base import task_classes, task_type
        log = get_logger()

        # Get the list of task types excluding spectra and redshifts,
        # which will be handled separately.
        ttypes = [ t for t in all_task_types() if (t != "spectra") \
            and (t != "redshift") ]

        with self.cursor() as cur:
            for tt in ttypes:
                # for each type of task, get the list of tasks in waiting mode
                cmd = "select name from {} where state = {}".format(tt, task_state_to_int["waiting"])
                if night is not None:
                    cmd = "{} and night = {}".format(cmd, night)
                cur.execute(cmd)
                tasks = [ x for (x, ) in cur.fetchall()]
                if len(tasks) > 0:
                    log.debug("checking {} {} tasks ...".format(len(tasks),tt))
                for tsk in tasks:
                    task_classes[tt].getready(db=self, name=tsk, cur=cur)

            for tt in [ "spectra" , "redshift" ]:
                if tt == "spectra":
                    required_healpix_frame_state = 1
                    # means we have a cframe
                elif tt == "redshift":
                    required_healpix_frame_state = 2
                    # means we have an updated spectra file

                cur.execute('select nside,pixel from healpix_frame where state = {}'.format(required_healpix_frame_state))
                entries = cur.fetchall()
                for entry in entries :
                    log.debug("{} of pixel {} is ready to run".format(tt,entry[1]))
                    cur.execute('update {} set state = {} where nside = {} and pixel = {}'.format(tt,task_state_to_int["ready"],entry[0],entry[1]))

                log.debug("checking waiting {} tasks to see if they are done...".format(tt))
                cmd = "select pixel from {} where state = {}".format(tt, task_state_to_int["waiting"])
                cur.execute(cmd)
                pixels = [ x for (x, ) in cur.fetchall()]
                if len(pixels) > 0:
                    log.debug("checking {} {} ...".format(len(pixels),tt))
                    if tt == "spectra":
                        required_healpix_frame_state = 2
                    elif tt == "redshift":
                        required_healpix_frame_state = 3
                    for pixel in pixels:
                        cur.execute('select pixel from healpix_frame where pixel = {} and state != {}'.format(pixel,required_healpix_frame_state))
                        entries = cur.fetchall()
                        if len(entries)==0 :
                            log.debug("{} task of pixel {} is done".format(tt,pixel))
                            cur.execute('update {} set state = {} where pixel = {}'.format(tt,task_state_to_int["done"],pixel))
        return


    def update_healpix_frame_state(self, props, state, cur):
        if "expid" in props :
            # update from a cframe
            cmd = "update healpix_frame set state = {} where expid = {} and spec = {} and state = {}".format(state,props["expid"],props["spec"],props["state"])
        else :
            # update from a spectra or redshift task
            cmd = "update healpix_frame set state = {} where nside = {} and pixel = {} and state = {}".format(state,props["nside"],props["pixel"],props["state"])

        if cur is None :
            with self.cursor() as cur:
                cur.execute(cmd)
        else :
            cur.execute(cmd)
        return


    def select_healpix_frame(self, props):
        res = []
        with self.cursor() as cur:
            cmd = "select * from healpix_frame where "
            first=True
            for k in props.keys() :
                if not first : cmd += " and "
                first=False
                cmd += "{}={}".format(k,props[k])
            cur.execute(cmd)
            entries = cur.fetchall()
            # convert that to list of dictionaries
            for entry in entries :
                tmp = dict()
                for i, k in enumerate(["night", "expid", "spec", "nside",
                                       "pixel", "ntargets", "state"]):
                    tmp[k] = entry[i]
                res.append(tmp)
        return res


    def create_healpix_frame_table(self) :
        with self.cursor() as cur:
            cmd = "create table healpix_frame (night integer, expid integer, spec integer, nside integer, pixel integer, ntargets integer, state integer, unique(expid, spec, nside, pixel))"
            cur.execute(cmd)

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

        self._connstr = None

        # This timeout is in seconds
        self._busytime = 1000

        # Journaling options
        self._journalmode = "persist"
        self._syncmode = "normal"

        if create:
            self.initdb()
        return


    def _open(self):
        import sqlite3

        if self._path is None:
            # We are opening an in-memory DB
            self._conn = sqlite3.connect(":memory:")
        else:
            try:
                # only python3 supports uri option
                if self._mode == 'r':
                    self._connstr = 'file:{}?mode=ro'.format(self._path)
                else:
                    self._connstr = 'file:{}?mode=rwc'.format(self._path)
                self._conn = sqlite3.connect(self._connstr, uri=True,
                    timeout=self._busytime)
            except:
                self._conn = sqlite3.connect(self._path, timeout=self._busytime)
        if self._mode == 'w':
            # In read-write mode, set the journaling
            self._conn.execute("pragma journal_mode={}"\
                .format(self._journalmode))
            self._conn.execute("pragma synchronous={}".format(self._syncmode))
            # Other tuning options
            self._conn.execute("pragma temp_store=memory")
            self._conn.execute("pragma page_size=4096")
            self._conn.execute("pragma cache_size=4000")
        return


    def _close(self):
        del self._conn
        self._conn = None
        return


    @contextmanager
    def cursor(self):
        import sqlite3
        self._open()
        cur = self._conn.cursor()
        cur.execute("begin transaction")
        try:
            yield cur
        except sqlite3.DatabaseError as err:
            log = get_logger()
            log.error(err)
            cur.execute("rollback")
            raise err
        else:
            try:
                cur.execute("commit")
            except sqlite3.OperationalError:
                #- sqlite3 in py3.5 can't commit a read-only finished transaction
                pass
        finally:
            del cur
            self._close()


    def initdb(self):
        """Create DB tables for all tasks if they do not exist.
        """
        # check existing tables
        tables_in_db = None
        with self.cursor() as cur:
            cur.execute("select name FROM sqlite_master WHERE type='table'")
            tables_in_db = [x for (x, ) in cur.fetchall()]

        # Create a table for every task type
        from .tasks.base import task_classes, task_type
        for tt, tc in task_classes.items():
            if tt not in tables_in_db:
                tc.create(self)

        if "healpix_frame" not in tables_in_db:
            self.create_healpix_frame_table()
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
        authorize (str): If creating the schema, this is the list of
            additional roles that should be granted access.

    """
    def __init__(self, host, port, dbname, user, schema=None, authorize=None):
        super(DataBasePostgres, self).__init__()

        self._schema = schema
        self._user = user
        self._dbname = dbname
        self._host = host
        self._port = port
        self._authorize = authorize

        self._proddir = os.path.abspath(io.specprod_root())

        create = False
        if self._schema is None:
            create = True
            self._schema = self._compute_schema()

        if create:
            self.initdb()
        return


    def _compute_schema(self):
        import hashlib
        md = hashlib.md5()
        md.update(self._proddir.encode())
        return "pipe_{}".format(md.hexdigest())


    def _open(self):
        import psycopg2 as pg2
        import time
        import numpy.random

        # Open connection.  If psycopg2 raises an exception, then sleep
        # for a random time interval and keep trying.
        maxtry = 10
        ntry = 0
        while True:
            try:
                self._conn = pg2.connect(host=self._host, port=self._port,
                    user=self._user, dbname=self._dbname)
            except pg2.OperationalError as err:
                log = get_logger()
                log.debug("PostgreSQL connection failed with '{}', will sleep and retry".format(err))
                if ntry > maxtry:
                    log.error(err)
                    break
                numpy.random.seed(int(time.time()))
                sec = numpy.random.uniform() * 3.0
                time.sleep(sec)
                ntry += 1
            else:
                break

        return


    def _close(self):
        del self._conn
        self._conn = None
        return


    @property
    def schema(self):
        return self._schema


    def _have_schema(self, cur):
        com = "select exists(select 1 from pg_namespace where nspname = '{}')".format(self._schema)
        cur.execute(com)
        return cur.fetchone()[0]


    @contextmanager
    def cursor(self, skipcheck=False):
        import psycopg2
        self._open()
        cur = self._conn.cursor()
        if not skipcheck:
            have_schema = self._have_schema(cur)
            if not have_schema:
                raise RuntimeError("Postgres schema for production {} does"
                    " not exist.  Make sure you create the production with"
                    " postgres options and source the top-level setup.sh"
                    " file.".format(self._proddir))
        cur.execute("set search_path to '{}'".format(self._schema))
        cur.execute("begin transaction")
        try:
            yield cur
        except psycopg2.DatabaseError as err:
            log = get_logger()
            log.error(err)
            cur.execute("rollback")
            raise err
        else:
            cur.execute("commit")
        finally:
            del cur
            self._close()


    def initdb(self):
        """Create DB tables for all tasks if they do not exist.
        """
        log = get_logger()
        # Check existence of the schema.  If we were not passed the schema
        # in the constructor, it means that we are creating a new prod, so any
        # existing schema should be wiped and recreated.
        tables_in_db = None
        with self.cursor(skipcheck=True) as cur:
            # See if our schema already exists...
            have_schema = self._have_schema(cur)
            if have_schema:
                # We need to wipe it first
                com = "drop schema {} cascade".format(self._schema)
                log.debug(com)
                cur.execute(com)
            com = "create schema {} authorization {}"\
                .format(self._schema, self._user)
            log.debug(com)
            cur.execute(com)

            if self._authorize is not None:
                com = "grant usage on schema {} to {}"\
                    .format(self._schema, self._authorize)
                log.debug(com)
                cur.execute(com)

                com = "alter default privileges in schema {} grant select on tables to {}".format(self._schema, self._authorize)
                log.debug(com)
                cur.execute(com)

                com = "alter default privileges in schema {} grant select,usage on sequences to {}".format(self._schema, self._authorize)
                log.debug(com)
                cur.execute(com)

                com = "alter default privileges in schema {} grant execute on functions to {}".format(self._schema, self._authorize)
                log.debug(com)
                cur.execute(com)

                com = "alter default privileges in schema {} grant usage on types to {}".format(self._schema, self._authorize)
                log.debug(com)
                cur.execute(com)

            # Create a table of information about this prod
            com = "create table {}.info (key text unique, val text)"\
                .format(self._schema)
            log.debug(com)
            cur.execute(com)
            com = "insert into {}.info values ('{}', '{}')"\
                .format(self._schema, "path", self._proddir)
            log.debug(com)
            cur.execute(com)
            if 'USER' in os.environ:
                com = "insert into {}.info values ('{}', '{}')"\
                    .format(self._schema, "created_by", os.environ['USER'])
                log.debug(com)
                cur.execute(com)

            # check existing tables
            cur.execute("select tablename from pg_tables where schemaname =  '{}'".format(self.schema))
            tables_in_db = [x for (x, ) in cur.fetchall()]

        # Create a table for every task type
        from .tasks.base import task_classes, task_type
        for tt, tc in task_classes.items():
            if tt not in tables_in_db:
                tc.create(self)

        if "healpix_frame" not in tables_in_db:
            self.create_healpix_frame_table()

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
