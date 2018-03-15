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
            "input-image" : task_classes["pix"].name_join(props)
        }
        return deptasks


    def _run_max_procs(self, procs_per_node):
        """See BaseTask.run_max_procs.
        """
        return 20


    def _run_time(self, name, procs_per_node, db=None):
        """See BaseTask.run_time.
        """
        return 15 # convergence slower for some realizations


    def _run_defaults(self):
        """See BaseTask.run_defaults.
        """
        opts = {}
        opts["trace-deg-wave"] = 7
        opts["trace-deg-x"] = 7
        opts["trace-prior-deg"] = 4

        envname="DESI_CCD_CALIBRATION_DATA"
        if not envname in os.environ :
            raise KeyError("need to set DESI_CCD_CALIBRATION_DATA env. variable")

        # default for now is the simulation directory
        # think in the future to use another directory
        opts["input-psf-dir"]   = "{}/SIM".format(os.environ[envname])

        # to get the lampline location, look in our path for specex
        # and use that install prefix to find the data directory.
        # if that directory does not exist, use a default NERSC
        # location.
        opts["lamplines"] = \
            "/project/projectdirs/desi/software/edison/specex/specex-0.3.9/data/specex_linelist_desi.txt"
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exefile = os.path.join(path, "desi_psf_fit")
            if os.path.isfile(exefile) and os.access(exefile, os.X_OK):
                specexdir = os.path.join(path, "..", "data")
                opts["lamplines"] = os.path.join(specexdir,
                    "specex_linelist_desi.txt")

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

        inputpsf = "psf-{}{}.fits".format(props["band"],props["spec"])
        if "input-psf-dir" in opts :
            inputpsf = os.path.join(opts["input-psf-dir"],inputpsf)

        options["input-psf"]   = inputpsf
        options["input-image"] = task_classes["pix"].paths(deps["input-image"])[0]
        options["output-psf"]  = self.paths(name)


        if len(opts) > 0:
            opts_wo_input_dir = opts.copy()
            opts_wo_input_dir.pop("input-psf-dir")
            extarray = option_list(opts_wo_input_dir)
            options["extra"] = " ".join(extarray)

        return option_list(options)

    def _run_cli(self, name, opts, procs, db):
        """See BaseTask.run_cli.
        """

        entry = "desi_compute_psf"
        if procs > 1:
            entry = "desi_compute_psf_mpi"
        return "{} {}".format(entry, self._option_list(name, opts))

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
