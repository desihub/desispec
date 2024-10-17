#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.tasks.preproc
===============================

"""

from __future__ import absolute_import, division, print_function

import os
import re

from collections import OrderedDict

from ..defs import (task_name_sep, task_state_to_int, task_int_to_state)

from ...util import option_list

from ...io import findfile

from .base import (BaseTask, task_classes)

from desiutil.log import get_logger

import numpy as np

# NOTE: only one class in this file should have a name that starts with "Task".

class TaskPreproc(BaseTask):
    """Class containing the properties of one preprocessed pixel file.
    """
    def __init__(self):
        # do that first
        super(TaskPreproc, self).__init__()
        # then put int the specifics of this class
        # _cols must have a state
        self._type = "preproc"
        self._cols = [
            "night",
            "band",
            "spec",
            "expid",
            "flavor",
            "state"
        ]
        self._coltypes = [
            "integer",
            "text",
            "integer",
            "integer",
            "text",
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
        return [ findfile("preproc", night=props["night"], expid=props["expid"],
            camera=camera, groupname=None, nside=None, band=props["band"],
            spectrograph=props["spec"]) ]


    def _deps(self, name, db, inputs):
        """See BaseTask.deps.
        """
        from .base import task_classes
        props = self.name_split(name)
        deptasks = {
             "fibermap" : task_classes["fibermap"].name_join(props),
             "rawdata" : task_classes["rawdata"].name_join(props)
        }
        return deptasks


    def _run_max_procs(self):
        # This is a serial task.
        return 1

    def _run_time(self, name, procs, db):
        # Run time on one proc on machine with scale factor == 1.0
        return 3.0

    def _run_max_mem_proc(self, name, db):
        # Per-process memory requirements
        return 0


    def _run_defaults(self):
        """See BaseTask.run_defaults.
        """
        return dict()


    def _option_list(self, name, opts):
        """Build the full list of options.

        This includes appending the filenames and incorporating runtime
        options.
        """
        from .base import task_classes, task_type

        dp = self.deps(name)

        options = OrderedDict()
        options.update(opts)

        props = self.name_split(name)
        options["infile"] = task_classes["rawdata"].paths(dp["rawdata"])[0]
        options["cameras"] = "{}{}".format(props["band"],props["spec"])

        outfile = self.paths(name)[0]
        options["outfile"] = outfile

        return option_list(options)


    def _run_cli(self, name, opts, procs, db):
        """See BaseTask.run_cli.
        """
        entry = "desi_preproc"
        optlist = self._option_list(name, opts)
        com = "{} {}".format(entry, " ".join(optlist))
        return com


    def _run(self, name, opts, comm, db):
        """See BaseTask.run.
        """
        from ...scripts import preproc
        optlist = self._option_list(name, opts)
        args = preproc.parse(optlist)
        preproc.main(args)
        return

    def postprocessing(self, db, name, cur):
        """For successful runs, postprocessing on DB"""
        # run getready for all extraction with same night,band,spec
        props = self.name_split(name)
        log  = get_logger()
        tt  = "psf"
        cmd = "select name from {} where night={} and band='{}' and spec={} and expid={} and state=0".format(tt,props["night"],props["band"],props["spec"],props["expid"])
        cur.execute(cmd)
        tasks = [ x for (x,) in cur.fetchall() ]
        log.debug("checking {}".format(tasks))
        for task in tasks:
            task_classes[tt].getready(db=db, name=task, cur=cur)
        tt  = "traceshift"
        cmd = "select name from {} where night={} and band='{}' and spec={} and expid={} and state=0".format(tt,props["night"],props["band"],props["spec"],props["expid"])
        cur.execute(cmd)
        tasks = [ x for (x,) in cur.fetchall() ]
        log.debug("checking {}".format(tasks))
        for task in tasks:
            task_classes[tt].getready(db=db, name=task, cur=cur)
