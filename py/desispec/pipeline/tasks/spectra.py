#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from ..defs import (task_name_sep, task_state_to_int, task_int_to_state)

from ...util import option_list

from ...io import findfile

from .base import BaseTask


# NOTE: only one class in this file should have a name that starts with "Task".

class TaskSpectra(BaseTask):
    """Class containing the properties of one spectra task.
    """
    def __init__(self):
        super(TaskSpectra, self).__init__()
        # then put int the specifics of this class
        # _cols must have a state
        self._type = "spectra"
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
        return [ findfile("spectra", night=None, expid=None,
                          camera=None, groupname=props["pixel"], nside=props["nside"], band=None,
                          spectrograph=None) ]
    
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

    def _option_list(self, name, opts, db):
        
        # we do need db access for spectra
        
        from .base import task_classes, task_type
        # get pixel
        props = self.name_split(name)
        # get list of exposures and spectrographs by selecting entries in the healpix_frame table with state = 1
        # which means that there is a new cframe intersecting the pixel
        entries = db.select_healpix_frame({"pixel":props["pixel"],"nside":props["nside"],"state":1})
        # now select cframe with same props
        cframes = []
        for entry in entries :
            for band in ["b","r","z"] :
                entry_and_band = entry.copy()
                entry_and_band["band"] = band
                print ("DEBUGDEBUG",entry_and_band)
                # this will match cframes with same expid and spectro
                taskname = task_classes["cframe"].name_join(entry_and_band) 
                filename = task_classes["cframe"].paths(taskname)[0]
                print ("DEBUGDEBUG filename=",filename)
                cframes.append(filename)
                
        options = {}
        options["infile"]  = cframes
        options["outfile"] = self.paths(name)[0]
        
        return option_list(options)
    
    def _run_cli(self, name, opts, procs, db):
        """See BaseTask.run_cli.
        """
        entry = "???"
        optlist = self._option_list(name, opts, db)
        return "{} {}".format(entry, " ".join(optlist))
        
    def _run(self, name, opts, comm, db):
        """See BaseTask.run.
        """
        return

