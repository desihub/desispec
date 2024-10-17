#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.tasks.traceshift
==================================

"""

from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from ..defs import (task_name_sep, task_state_to_int, task_int_to_state)

from ...util import option_list

from ...io import findfile

from .base import (BaseTask, task_classes)

from desiutil.log import get_logger

import sys,re,os,copy

# NOTE: only one class in this file should have a name that starts with "Task".

class TaskTraceShift(BaseTask):
    """Class containing the properties of one trace shift task.
    """
    def __init__(self):
        super(TaskTraceShift, self).__init__()
        # then put int the specifics of this class
        # _cols must have a state
        self._type = "traceshift"
        self._cols = [
            "night",
            "band",
            "spec",
            "expid",
            "state"
        ]
        self._coltypes = [
            "integer",
            "text",
            "integer",
            "integer",
            "integer"
        ]
        # _name_fields must also be in _cols
        self._name_fields  = ["night","band","spec","expid"]
        self._name_formats = ["08d","s","d","08d"]

    def _paths(self, name):
        """See BaseTask.paths.
        """
        props = self.name_split(name)
        camera = "{}{}".format(props["band"], props["spec"])
        return [ findfile("psf", night=props["night"], expid=props["expid"],
            camera=camera, groupname=None, nside=None, band=props["band"],
            spectrograph=props["spec"]) ]

    def _deps(self, name, db, inputs):
        """See BaseTask.deps.
        """
        from .base import task_classes
        props = self.name_split(name)
        deptasks = {
            "image" : task_classes["preproc"].name_join(props),
            "psf" : task_classes["psfnight"].name_join(props)
            }
        return deptasks

    def _run_max_procs(self):
        # This is a serial task.
        return 1

    def _run_time(self, name, procs, db):
        # Run time on one proc on machine with scale factor == 1.0
        return 2.0


    def _run_defaults(self):
        """See BaseTask.run_defaults.
        """
        opts = {}
        opts["degxx"] = 2
        opts["degxy"] = 2
        opts["degyx"] = 0
        opts["degyy"] = 0
        opts["auto"]  = True
        return opts


    def _option_list(self, name, opts):
        """Build the full list of options.

        This includes appending the filenames and incorporating runtime
        options.
        """
        from .base import task_classes, task_type

        deps = self.deps(name)
        options = {}
        options["image"]    = task_classes["preproc"].paths(deps["image"])[0]
        options["psf"]      = task_classes["psfnight"].paths(deps["psf"])[0]
        options["outpsf"]   = self.paths(name)[0]

        options.update(opts)
        return option_list(options)

    def _run_cli(self, name, opts, procs, db):
        """See BaseTask.run_cli.
        """
        entry = "desi_compute_trace_shifts"
        optlist = self._option_list(name, opts)
        com = "{} {}".format(entry, " ".join(optlist))
        return com

    def _run(self, name, opts, comm, db):
        """See BaseTask.run.
        """
        from ...scripts import trace_shifts
        optlist = self._option_list(name, opts)
        args = trace_shifts.parse(optlist)
        if comm is None :
            trace_shifts.main(args)
        else :
            trace_shifts.main_mpi(args, comm=comm)
        return

    def postprocessing(self, db, name, cur):
        """For successful runs, postprocessing on DB"""
        # run getready for all extraction with same night,band,spec
        props = self.name_split(name)
        log  = get_logger()
        for tt in ["extract"] :
            cmd = "select name from {} where night={} and expid={} and band='{}' and spec={} and state=0".format(tt,props["night"],props["expid"],props["band"],props["spec"])
            cur.execute(cmd)
            tasks = [ x for (x,) in cur.fetchall() ]
            log.debug("checking {} {}".format(tt,tasks))
            for task in tasks :
                task_classes[tt].getready( db=db,name=task,cur=cur)
