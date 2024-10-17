#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.tasks.starfit
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

from desiutil.log import get_logger

# NOTE: only one class in this file should have a name that starts with "Task".

class TaskStarFit(BaseTask):
    """Class containing the properties of one extraction task.
    """
    def __init__(self):
        super(TaskStarFit, self).__init__()
        # then put int the specifics of this class
        # _cols must have a state
        self._type = "starfit"
        self._cols = [
            "night",
            "spec",
            "expid",
            "state"
        ]
        self._coltypes = [
            "integer",
            "integer",
            "integer",
            "integer"
        ]
        # _name_fields must also be in _cols
        self._name_fields  = ["night","spec","expid"]
        self._name_formats = ["08d","d","08d"]

    def _paths(self, name):
        """See BaseTask.paths.
        """
        props = self.name_split(name)
        return [ findfile("stdstars", night=props["night"], expid=props["expid"],
                          groupname=None, nside=None, camera=None, band=None,
                          spectrograph=props["spec"]) ]

    def _deps(self, name, db, inputs):
        """See BaseTask.deps.
        """
        from .base import task_classes
        props = self.name_split(name)

        # we need all the cameras for the fit of standard stars
        deptasks = dict()
        for band in ["b","r","z"] :
            props_and_band       = props.copy()
            props_and_band["band"] = band
            deptasks[band+"-frame"]=task_classes["extract"].name_join(props_and_band)
            deptasks[band+"-fiberflat"]=task_classes["fiberflatnight"].name_join(props_and_band)
            deptasks[band+"-sky"]=task_classes["sky"].name_join(props_and_band)
        return deptasks

    def _run_max_procs(self):
        # This is a serial task.
        return 1

    def _run_time(self, name, procs, db):
        # Run time on one proc on machine with scale factor == 1.0
        return 35.0

    def _run_max_mem_proc(self, name, db):
        # Per-process memory requirements
        return 5.0


    def _run_defaults(self):
        """See BaseTask.run_defaults.
        """
        import glob

        log = get_logger()

        opts = {}
        starmodels = None
        if "DESI_BASIS_TEMPLATES" in os.environ:
            filenames = sorted(glob.glob(os.environ["DESI_BASIS_TEMPLATES"]+"/stdstar_templates_*.fits"))
            if len(filenames) > 0 :
                starmodels = filenames[-1]
            else:
                filenames = sorted(glob.glob(os.environ["DESI_BASIS_TEMPLATES"]+"/star_templates_*.fits"))
                log.warning('Unable to find stdstar templates in {}; using star templates instead'.format(
                    os.getenv('DESI_BASIS_TEMPLATES')))
                if len(filenames) > 0 :
                    starmodels = filenames[-1]
                else:
                    msg = 'Unable to find stdstar or star templates in {}'.format(
                        os.getenv('DESI_BASIS_TEMPLATES'))
                    log.error(msg)
                    raise RuntimeError(msg)
        else:
            log.error("DESI_BASIS_TEMPLATES not set!")
            raise RuntimeError("could not find the stellar templates")

        opts["starmodels"] =  starmodels

        opts["delta-color"] = 0.2
        opts["color"] = "G-R"

        return opts


    def _option_list(self, name, opts):
        """Build the full list of options.

        This includes appending the filenames and incorporating runtime
        options.
        """
        from .base import task_classes, task_type


        log = get_logger()

        deps = self.deps(name)
        options = {}
        ### options["ncpu"] = 1
        options["outfile"] = self.paths(name)[0]
        options["frames"] = []
        options["skymodels"] = []
        options["fiberflats"] = []

        # frames skymodels fiberflats
        props = self.name_split(name)
        for band in ["b", "r", "z"] :
            props_and_band = props.copy()
            props_and_band["band"] = band

            task  = task_classes["extract"].name_join(props_and_band)
            frame_filename = task_classes["extract"].paths(task)[0]

            task  = task_classes["fiberflatnight"].name_join(props_and_band)
            fiberflat_filename = task_classes["fiberflatnight"].paths(task)[0]

            task  = task_classes["sky"].name_join(props_and_band)
            sky_filename = task_classes["sky"].paths(task)[0]

            # check all files exist
            if os.path.isfile(frame_filename) \
               and os.path.isfile(fiberflat_filename) \
               and os.path.isfile(sky_filename) :

                options["frames"].append(frame_filename)
                options["skymodels"].append(sky_filename)
                options["fiberflats"].append(fiberflat_filename)

            else :
                log.warning("missing band {} for {}".format(band,name))

        options.update(opts)
        return option_list(options)

    def _run_cli(self, name, opts, procs, db):
        """See BaseTask.run_cli.
        """
        entry = "desi_fit_stdstars"
        optlist = self._option_list(name, opts)
        com = "{} {}".format(entry, " ".join(optlist))
        return com

    def _run(self, name, opts, comm, db):
        """See BaseTask.run.
        """
        from ...scripts import stdstars
        optlist = self._option_list(name, opts)
        args = stdstars.parse(optlist)
        stdstars.main(args)
        return

    def postprocessing(self, db, name, cur):
        """For successful runs, postprocessing on DB"""
        # run getready on all fierflatnight with same night,band,spec
        props = self.name_split(name)
        log  = get_logger()
        tt="fluxcalib"
        cmd = "select name from {} where night={} and expid={} and spec={} and state=0".format(tt,props["night"],props["expid"],props["spec"])
        cur.execute(cmd)
        tasks = [ x for (x,) in cur.fetchall() ]
        log.debug("checking {}".format(tasks))
        for task in tasks :
            task_classes[tt].getready( db=db,name=task,cur=cur)
