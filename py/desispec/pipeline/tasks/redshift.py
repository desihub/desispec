#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-

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

    def _run_max_mem(self, name, db):
        mem = 0.0
        return mem

    def _run_max_procs(self, procs_per_node):
        return procs_per_node

    def _run_time(self, name, procs_per_node, db):
        # One minute of overhead, plus about 1 second per target.
        tm = 1
        if db is not None:
            props = self.name_split(name)
            # FIXME:  the healpix_frame table has the number of targets
            # for each spectrograph in the given pixel.  What we need here
            # is the total number of unique targets per pixel.  Once the
            # "coadd" task is implemented, get the total unique targets from the
            # coadd table.  As a temporary proxy, we base this number on the
            # max number of targets from a single frame.
            entries = db.select_healpix_frame(
                {"pixel":props["pixel"],
                 "nside":props["nside"]}
            )
            maxtarg = np.max([x["ntargets"] for x in entries])
            # 2.5 seconds per max targets in a single cframe (change this
            # to be based on number of targets in coadd)
            tm += 2.5 * 0.0167 * maxtarg
        return tm

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
