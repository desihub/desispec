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

import sys,re,os,glob

# NOTE: only one class in this file should have a name that starts with "Task".

class TaskPSFNight(BaseTask):
    """Class containing the properties of one PSF combined night task.
    """
    def __init__(self):
        super(TaskPSFNight, self).__init__()
        # then put int the specifics of this class
        # _cols must have a state
        self._type = "psfnight"
        self._cols = [
            "night",
            "band",
            "spec",
            "state"
        ]
        self._coltypes = [
            "integer",
            "text",
            "integer",
            "integer"
        ]
        # _name_fields must also be in _cols
        self._name_fields  = ["night","band","spec"]
        self._name_formats = ["08d","s","d"]
        
    def _paths(self, name):
        """See BaseTask.paths.
        """
        props = self.name_split(name)
        camera = "{}{}".format(props["band"], props["spec"])
        return [ findfile("psfnight", night=props["night"],
            camera=camera, groupname=None, nside=None, band=props["band"],
            spectrograph=props["spec"]) ]

    def _deps(self, name, db, inputs):
        """See BaseTask.deps.
        """
        return dict()
        
    def _run_max_procs(self, procs_per_node):
        """See BaseTask.run_max_procs.
        """
        return 1

    def _run_time(self, name, procs_per_node, db=None):
        """See BaseTask.run_time.
        """
        return 1

    def _run_defaults(self):
        """See BaseTask.run_defaults.
        """
        return {}

    def _option_dict(self, name, opts):
        """Build the full list of options.

        This includes appending the filenames and incorporating runtime
        options.
        """
        from .base import task_classes, task_type

        options = OrderedDict()
        options["output"] = self.paths(name)[0]

        # look for psf for this night on disk
        options["input"]  = []
        props = self.name_split(name)
        camera = "{}{}".format(props["band"], props["spec"])
        dummy_expid    = 99999999
        template_input = findfile("psf", night=props["night"], expid=dummy_expid,
                                  camera=camera,
                                  band=props["band"],
                                  spectrograph=props["spec"])
        template_input = template_input.replace("{:08d}".format(dummy_expid),"*")
        options["input"]  = glob.glob(template_input)
        return options

    def _option_list(self, name, opts):
        """Build the full list of options.

        This includes appending the filenames and incorporating runtime
        options.
        """
        return option_list(self._option_dict(name,opts))
        
    def _run_cli(self, name, opts, procs, db=None):
        """See BaseTask.run_cli.
        """
        optlist = self._option_list(name, opts)
        com = "# command line for psfnight not implemented"
        return com

    def _run(self, name, opts, comm, db=None):
        """See BaseTask.run.
        """
        from ...scripts import specex
        optdict = self._option_dict(name, opts)
        specex.mean_psf(optdict["input"], optdict["output"])
        
        return
        
