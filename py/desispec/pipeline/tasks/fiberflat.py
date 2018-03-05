#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from ..defs import (task_name_sep, task_state_to_int, task_int_to_state)

from ...util import option_list

from ...io import findfile

from .base import BaseTask

import sys,re,os,copy

# NOTE: only one class in this file should have a name that starts with "Task".

class TaskFiberflat(BaseTask):
    """Class containing the properties of one extraction task.
    """
    def __init__(self):
        super(TaskFiberflat, self).__init__()
        # then put int the specifics of this class
        # _cols must have a state
        self._type = "fiberflat"
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
        return [ findfile("fiberflat", night=props["night"], expid=props["expid"],
            camera=camera, groupname=None, nside=None, band=props["band"],
            spectrograph=props["spec"]) ]

    def _deps(self, name, db, inputs):
        """See BaseTask.deps.
        """
        from .base import task_classes
        props = self.name_split(name)
        extract_task = task_classes["extract"].name_join(props)
        deptasks = [extract_task]
        return deptasks

    def _run_max_procs(self, procs_per_node):
        """See BaseTask.run_max_procs.
        """
        return 1

    def _run_time(self, name, procs_per_node, db=None):
        """See BaseTask.run_time.
        """
        return 4 # 2 minutes on edison

    def _run_defaults(self):
        """See BaseTask.run_defaults.
        """
        opts = {}
        return opts

    def _option_list(self, name, opts):
        """Build the full list of options.

        This includes appending the filenames and incorporating runtime
        options.
        """
        from .base import task_classes, task_type

        options = OrderedDict()

        deplist = self.deps(name)
        framefile = None
        for dp in deplist:
            if re.search("extract", dp) is not None:
                framefile = task_classes["extract"].paths(dp)[0]
        if framefile is None :
            raise RuntimeError("dependency list must include input frame file")
        
        outfile = self.paths(name)[0]

        options = {}
        options["infile"]    = framefile
        options["outfile"]   = outfile
        options.update(opts)
        return option_list(options)

    def _run_cli(self, name, opts, procs):
        """See BaseTask.run_cli.
        """
        entry = "desi_compute_fiberflat"
        optlist = self._option_list(name, opts)
        com = "{} {}".format(entry, " ".join(optlist))
        return com

    def _run(self, name, opts, comm):
        """See BaseTask.run.
        """
        from ...scripts import fiberflat
        optlist = self._option_list(name, opts)
        args = fiberflat.parse(optlist)
        fiberflat.main(args)
        return
