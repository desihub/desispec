#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from .base import BaseTask, task_classes, task_type
from ...io import findfile
from ...util import option_list
from redrock.external.desi import rrdesi
from desiutil.log import get_logger

import os

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
    
    def _paths(self, name):
        """See BaseTask.paths.
        """
        props = self.name_split(name)
        hpix = props["pixel"]
        nside = props["nside"]
        zbest = findfile("zbest", groupname=hpix, nside=nside)
        redrock = findfile("redrock", groupname=hpix, nside=nside)
        return [zbest, redrock]
    
    def _deps(self, name, db, inputs):
        """See BaseTask.deps.
        """
        props = self.name_split(name)
        deptasks = {
            "infile" : task_classes["spectra"].name_join(props)
        }
        return deptasks

    def run_max_procs(self, procs_per_node):
        return procs_per_node

    def run_time(self, name, procs_per_node, db=None):
        """See BaseTask.run_time.
        """
        return 15 # in general faster but convergence slower for some realizations

    def _run_defaults(self):
        """See BaseTask.run_defaults.
        """
        return {}

    def _option_list(self, name, opts):
        """Build the full list of options.

        This includes appending the filenames and incorporating runtime
        options.
        """
        
        zbestfile, redrockfile = self.paths(name)
        outdir  = os.path.dirname(zbestfile)
        
        options = {}
        options["output"] = redrockfile
        options["zbest"] = zbestfile
        options.update(opts)
        
        optarray = option_list(options)
        
        deps = self.deps(name)
        specfile = task_classes["spectra"].paths(deps["infile"])[0]
        optarray.append(specfile)
        
        return optarray

    
    def _run_cli(self, name, opts, procs, db):
        """See BaseTask.run_cli.
        """
        entry = "rrdesi_mpi"
        optlist = self._option_list(name, opts)
        return "{} {}".format(entry, " ".join(optlist))

    def _run(self, name, opts, comm, db):
        """See BaseTask.run.
        """        
        optlist = self._option_list(name, opts)
        rrdesi(options=optlist, comm=comm)
        return

    def postprocessing(self, db, name, cur):
        """For successful runs, postprocessing on DB"""
        props=self.name_split(name)
        props["state"]=2 # selection, only those for which we had already updated the spectra
        db.update_healpix_frame_state(props,state=3,cur=cur) # 3=redshifts have been updated
