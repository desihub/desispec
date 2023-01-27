#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.tasks.extract
===============================

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

class TaskExtract(BaseTask):
    """Class containing the properties of one extraction task.
    """
    def __init__(self):
        super(TaskExtract, self).__init__()
        # then put int the specifics of this class
        # _cols must have a state
        self._type = "extract"
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
        return [ findfile("frame", night=props["night"], expid=props["expid"],
            camera=camera, groupname=None, nside=None, band=props["band"],
            spectrograph=props["spec"]) ]

    def _deps(self, name, db, inputs):
        """See BaseTask.deps.
        """
        from .base import task_classes
        props = self.name_split(name)
        deptasks = {
            "input" : task_classes["preproc"].name_join(props),
            "fibermap" : task_classes["fibermap"].name_join(props),
            "psf" : task_classes["traceshift"].name_join(props)
            }
        return deptasks

    def _run_max_procs(self):
        # 20 bundles per camera
        return 20


    def _run_time(self, name, procs, db):
        # Time when running on max procs on machine with scale
        # factor 1.0
        mprc = self._run_max_procs()
        return (7.0 / procs) * mprc


    def _run_defaults(self):
        """See BaseTask.run_defaults.
        """
        opts = {}
        opts["regularize"] = 0.0
        opts["nwavestep"] = 50
        opts["verbose"] = False
        opts["heliocentric_correction"] = False
        opts["wavelength_b"] = "3579.0,5939.0,0.8"
        opts["wavelength_r"] = "5635.0,7731.0,0.8"
        opts["wavelength_z"] = "7445.0,9824.0,0.8"
        opts["psferr"] = 0.05
        return opts


    def _option_list(self, name, opts):
        """Build the full list of options.

        This includes appending the filenames and incorporating runtime
        options.
        """
        from .base import task_classes, task_type

        deps = self.deps(name)
        options = {}
        options["input"]    = task_classes["preproc"].paths(deps["input"])[0]
        options["fibermap"] = task_classes["fibermap"].paths(deps["fibermap"])[0]
        options["psf"]      = task_classes["traceshift"].paths(deps["psf"])[0]
        options["output"]   = self.paths(name)[0]

        # extract the wavelength range from the options, depending on the band
        props = self.name_split(name)
        optscopy = copy.deepcopy(opts)
        wkey = "wavelength_{}".format(props["band"])
        wave = optscopy[wkey]
        del optscopy["wavelength_b"]
        del optscopy["wavelength_r"]
        del optscopy["wavelength_z"]
        optscopy["wavelength"] = wave

        options.update(optscopy)
        return option_list(options)

    def _run_cli(self, name, opts, procs, db):
        """See BaseTask.run_cli.
        """
        entry = "desi_extract_spectra"
        optlist = self._option_list(name, opts)
        com = "{} {}".format(entry, " ".join(optlist))
        return com

    def _run(self, name, opts, comm, db):
        """See BaseTask.run.
        """
        from ...scripts import extract
        optlist = self._option_list(name, opts)
        args = extract.parse(optlist)
        if comm is None :
            extract.main(args)
        else :
            extract.main_mpi(args, comm=comm)
        return

    def postprocessing(self, db, name, cur):
        """For successful runs, postprocessing on DB"""
        # run getready for all extraction with same night,band,spec
        props = self.name_split(name)
        log  = get_logger()
        for tt in ["fiberflat","sky"] :
            cmd = "select name from {} where night={} and expid={} and band='{}' and spec={} and state=0".format(tt,props["night"],props["expid"],props["band"],props["spec"])
            cur.execute(cmd)
            tasks = [ x for (x,) in cur.fetchall() ]
            log.debug("checking {} {}".format(tt,tasks))
            for task in tasks :
                task_classes[tt].getready( db=db,name=task,cur=cur)
