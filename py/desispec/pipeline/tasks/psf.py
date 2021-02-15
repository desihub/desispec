#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from collections import OrderedDict

from ..defs import (task_name_sep, task_state_to_int, task_int_to_state)

from ...util import option_list

from ...io import findfile

from .base import (BaseTask, task_classes)

from desiutil.log import get_logger

import sys,re,os

# NOTE: only one class in this file should have a name that starts with "Task".

class TaskPSF(BaseTask):
    """Class containing the properties of one PSF task.
    """
    def __init__(self):
        super(TaskPSF, self).__init__()
        # then put int the specifics of this class
        # _cols must have a state
        self._type = "psf"
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
            "input-image" : task_classes["preproc"].name_join(props)
        }
        return deptasks


    def _run_max_procs(self):
        # 20 bundles per camera
        return 20


    def _run_time(self, name, procs, db):
        # Time when running on max procs on machine with scale
        # factor 1.0
        mprc = self._run_max_procs()
        return (20.0 / procs) * mprc


    def _run_defaults(self):
        """See BaseTask.run_defaults.
        """
        opts = {}
        opts["trace-deg-wave"] = 7
        opts["trace-deg-x"] = 7
        opts["trace-prior-deg"] = 4

        envname="DESI_SPECTRO_CALIB"
        if not envname in os.environ :
            raise KeyError("need to set DESI_SPECTRO_CALIB env. variable")

        return opts


    def _option_list(self, name, opts):
        """Build the full list of options.

        This includes appending the filenames and incorporating runtime
        options.
        """
        from .base import task_classes, task_type

        options = OrderedDict()

        deps  = self.deps(name)
        props = self.name_split(name)

        # make a copy, so we can remove some entries
        opts_copy = opts.copy()

        options["input-image"] = task_classes["preproc"].paths(deps["input-image"])[0]
        options["output-psf"]  = self.paths(name)

        if "specmin" in opts_copy:
            options["specmin"] = opts_copy["specmin"]
            del opts_copy["specmin"]

        if "nspec" in opts_copy:
            options["nspec"] = opts_copy["nspec"]
            del opts_copy["nspec"]

        if len(opts_copy) > 0:
            extarray = option_list(opts_copy)
            options["extra"] = " ".join(extarray)

        return option_list(options)


    def _run_cli(self, name, opts, procs, db):
        """See BaseTask.run_cli.
        """

        entry = "desi_compute_psf"
        if procs > 1:
            entry = "desi_compute_psf_mpi"
        return "{} {}".format(entry, " ".join(self._option_list(name, opts)))


    def _run(self, name, opts, comm, db):
        """See BaseTask.run.
        """
        from ...scripts import specex
        optlist = self._option_list(name, opts)

        args = specex.parse(optlist)
        specex.main(args, comm=comm)
        return


    def postprocessing(self, db, name, cur):
        """For successful runs, postprocessing on DB"""
        # run getready on all psfnight with same night,band,spec
        props = self.name_split(name)
        log  = get_logger()
        tt="psfnight"
        cmd = "select name from {} where night={} and band='{}' and spec={} and state=0".format(tt,props["night"],props["band"],props["spec"])
        cur.execute(cmd)
        tasks = [ x for (x,) in cur.fetchall() ]
        log.debug("checking {}".format(tasks))
        for task in tasks :
            task_classes[tt].getready( db=db,name=task,cur=cur)
