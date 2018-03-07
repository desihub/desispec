#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from .base import BaseTask


# NOTE: only one class in this file should have a name that starts with "Task".

class TaskRedshift(BaseTask):
    """Class containing the properties of one spectra task.
    """
    def __init__(self):
        super(TaskRedshift, self).__init__()
        # then put int the specifics of this class
        # _cols must have a state
        self._type = "redshift"
        self._cols = [
            "nside",
            "pixel",
            "state"
        ]
        self._coltypes = [
            "integer",
            "integer",
            "integer"
        ]
        # _name_fields must also be in _cols
        self._name_fields  = ["nside","pixel"]
        self._name_formats = ["d","d"]
    

    def _deps(self, name, db, inputs):
        """See BaseTask.deps.
        """
        return dict()

    def run_max_procs(self, procs_per_node):
        return 20

    def run_time(self, name, procs_per_node, db=None):
        """See BaseTask.run_time.
        """
        return 15 # in general faster but convergence slower for some realizations

    def _run_defaults(self):
        """See BaseTask.run_defaults.
        """
        return {}

    def _option_list(self, name, opts):
        return []
    
    def _run(self, name, opts, comm):
        """See BaseTask.run.
        """
        return

