#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.tasks.rawdata
===============================

Please add module-level documentation.
"""

from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from ..defs import (task_name_sep, task_state_to_int, task_int_to_state)

from ...util import option_list

from ...io import findfile

from .base import BaseTask


# NOTE: only one class in this file should have a name that starts with "Task".

class TaskRawdata(BaseTask):
    """Class containing the properties of one rawdata.

    Since rawdatas have no dependencies and are not created by the pipeline,
    this class is just used to specify names, etc.

    """
    def __init__(self):
        # do that first
        super(TaskRawdata, self).__init__()
        # then put int the specifics of this class
        # _cols must have a state
        self._type = "rawdata"
        self._cols = [
            "night",
            "expid",
            "flavor",
            "state"
        ]
        self._coltypes = [
            "integer",
            "integer",
            "text",
            "integer"
        ]
        # _name_fields must also be in _cols
        self._name_fields  = ["night","expid"]
        self._name_formats = ["d","08d"]



    def _paths(self, name):
        """See BaseTask.paths.
        """
        props = self.name_split(name)
        return [ findfile("raw", night=props["night"],
            expid=props["expid"]) ]

    def _deps(self, name, db, inputs):
        """See BaseTask.deps.
        """
        return dict()

    def _run_max_procs(self):
        # This is a fake task.
        return 1

    def _run_time(self, name, procs, db):
        # Fake task
        return 0.0


    def _run_defaults(self):
        """See BaseTask.run_defaults.
        """
        return dict()


    def _run_cli(self, name, opts, procs, db):
        """See BaseTask.run_cli.
        """
        return ""


    def _run(self, name, opts, comm, db):
        """See BaseTask.run.
        """
        return
