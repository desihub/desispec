#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.tasks.redshift
================================

Please add module-level documentation.
"""

from __future__ import absolute_import, division, print_function

import numpy as np

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
        redrock = findfile("redrock", groupname=hpix, nside=nside)
        rrdetails = findfile("rrdetails", groupname=hpix, nside=nside)
        return [redrock, rrdetails]

    def _deps(self, name, db, inputs):
        """See BaseTask.deps.
        """
        props = self.name_split(name)
        deptasks = {
            "infile" : task_classes["spectra"].name_join(props)
        }
        return deptasks

    def _run_max_procs(self):
        # Redshifts can run on any number of procs.
        return 0

    def _run_time(self, name, procs, db):
        # Run time on one task on machine with scale factor == 1.0.
        # This should depend on the total number of unique targets, which is
        # not known a priori.  Instead, we compute the total targets and reduce
        # this by some factor.
        if db is not None:
            props = self.name_split(name)
            entries = db.select_healpix_frame(
                {"pixel":props["pixel"],
                 "nside":props["nside"]}
            )
            ntarget = np.sum([x["ntargets"] for x in entries])
            neff = 0.3 * ntarget
            # 2.5 seconds per targets
            tm = 1 + 2.5 * 0.0167 * neff
        else:
            tm = 60

        return tm

    def _run_max_mem_proc(self, name, db):
        # Per-process memory requirements.  This is determined by the largest
        # Spectra file that must be read and broadcast.  We compute that size
        # assuming no coadd and using the total number of targets falling in
        # our pixel.
        mem = 0.0
        if db is not None:
            props = self.name_split(name)
            entries = db.select_healpix_frame(
                {"pixel":props["pixel"],
                 "nside":props["nside"]}
            )
            ntarget = np.sum([x["ntargets"] for x in entries])
            # DB entry is for one exposure and spectrograph.
            mem = 0.2 + 0.0002 * 3 * ntarget
        return mem

    def _run_max_mem_task(self, name, db):
        # This returns the total aggregate memory needed for the task,
        # which should be based on the larger of:
        #  1) the total number of unique (coadded) targets.
        #  2) the largest spectra file times the number of processes
        # Since it is not easy to calculate (1), and the constraint for (2)
        # is already encapsulated in the per-process memory requirements,
        # we return zero here.  This effectively selects one node.
        mem = 0.0
        return mem


    def _run_defaults(self):
        """See BaseTask.run_defaults.
        """
        return {'no-mpi-abort': True}

    def _option_list(self, name, opts):
        """Build the full list of options.

        This includes appending the filenames and incorporating runtime
        options.
        """

        redrockfile, rrdetailsfile = self.paths(name)
        outdir  = os.path.dirname(redrockfile)

        options = {}
        options["details"] = rrdetailsfile
        options["outfile"] = redrockfile
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

    def run_and_update(self, db, name, opts, comm=None):
        """Run the task and update DB state.

        The state of the task is marked as "done" if the command completes
        without raising an exception and if the output files exist.

        It is specific for redshift because the healpix_frame table has to be updated

        Args:
            db (pipeline.db.DB): The database.
            name (str): the name of this task.
            opts (dict): options to use for this task.
            comm (mpi4py.MPI.Comm): optional MPI communicator.

        Returns:
            int: the number of processes that failed.

        """
        nproc = 1
        rank = 0
        if comm is not None:
            nproc = comm.size
            rank = comm.rank

        failed = self.run(name, opts, comm=comm, db=db)

        if rank == 0:
            if failed > 0:
                self.state_set(db, name, "failed")
            else:
                outputs = self.paths(name)
                done = True
                for out in outputs:
                    if not os.path.isfile(out):
                        done = False
                        failed = nproc
                        break
                if done:
                    props=self.name_split(name)
                    props["state"]=2 # selection, only those for which we had already updated the spectra
                    with db.cursor() as cur :
                        self.state_set(db, name, "done",cur=cur)
                        db.update_healpix_frame_state(props,state=3,cur=cur) # 3=redshifts have been updated
                else:
                    self.state_set(db, name, "failed")
        return failed
