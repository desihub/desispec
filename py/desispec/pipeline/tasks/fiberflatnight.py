#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.tasks.fiberflatnight
======================================

Please add module-level documentation.
"""

from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from ..defs import (task_name_sep, task_state_to_int, task_int_to_state)

from ...util import option_list

from ...io import findfile

from .base import (BaseTask, task_classes)

from desiutil.log import get_logger

import numpy as np

import sys,re,os,glob

# NOTE: only one class in this file should have a name that starts with "Task".

class TaskFiberflatNight(BaseTask):
    """Class containing the properties of one fiberflat combined night task.
    """
    def __init__(self):
        super(TaskFiberflatNight, self).__init__()
        # then put int the specifics of this class
        # _cols must have a state
        self._type = "fiberflatnight"
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
        return [ findfile("fiberflatnight", night=props["night"],
            camera=camera, groupname=None, nside=None, band=props["band"],
            spectrograph=props["spec"]) ]

    def _deps(self, name, db, inputs):
        """See BaseTask.deps.
        """
        return dict()

    def _run_max_procs(self):
        # This is a serial task.
        return 1

    def _run_time(self, name, procs, db):
        # Run time on one proc on machine with scale factor == 1.0
        return 1

    def _run_defaults(self):
        """See BaseTask.run_defaults.
        """
        return {}

    def _option_list(self, name, opts):
        """Build the full list of options.

        This includes appending the filenames and incorporating runtime
        options.
        """
        from .base import task_classes, task_type

        options = OrderedDict()
        options["outfile"] = self.paths(name)[0]

        # look for psf for this night on disk
        props = self.name_split(name)
        camera = "{}{}".format(props["band"], props["spec"])
        dummy_expid    = 99999999
        template_input = findfile("fiberflat", night=props["night"], expid=dummy_expid,
                                  camera=camera,
                                  band=props["band"],
                                  spectrograph=props["spec"])
        template_input = template_input.replace("{:08d}".format(dummy_expid),"*")
        options["infile"]  = glob.glob(template_input)
        return option_list(options)

    def _run_cli(self, name, opts, procs, db):
        """See BaseTask.run_cli.
        """
        return "desi_average_fiberflat {}".format(self._option_list(name, opts))

    def _run(self, name, opts, comm, db):
        """See BaseTask.run.
        """
        from ...scripts import average_fiberflat
        optlist = self._option_list(name, opts)
        args = average_fiberflat.parse(optlist)
        average_fiberflat.main(args)

        return

    def getready(self, db, name, cur):
        """Checks whether dependencies are ready"""
        log  = get_logger()

        # look for the state of psf with same night,band,spectro
        props = self.name_split(name)

        cmd = "select state from fiberflat where night={} and band='{}' and spec={}".format(props["night"],props["band"],props["spec"])
        cur.execute(cmd)
        states = np.array([ x for (x,) in cur.fetchall() ])
        log.debug("states={}".format(states))

        # fiberflatnight ready if all fiberflat from the night have been processed, and at least one is done (failures are allowed)
        n_done   = np.sum(states==task_state_to_int["done"])
        n_failed = np.sum(states==task_state_to_int["failed"])

        ready    = (n_done > 0) & ( (n_done + n_failed) == states.size )
        if ready :
            self.state_set(db=db,name=name,state="ready",cur=cur)


    def postprocessing(self, db, name, cur):
        """For successful runs, postprocessing on DB"""
        # run getready for all sky with same night,band,spec
        props = self.name_split(name)
        log  = get_logger()
        tt  = "sky"
        cmd = "select name from {} where night={} and band='{}' and spec={} and state=0".format(tt,props["night"],props["band"],props["spec"])
        cur.execute(cmd)
        tasks = [ x for (x,) in cur.fetchall() ]
        log.debug("checking {}".format(tasks))
        for task in tasks :
            task_classes[tt].getready( db=db,name=task,cur=cur)
