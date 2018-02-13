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


# NOTE: only one class in this file should have a name that starts with "Task".

class TaskPSF(BaseTask):
    """Class containing the properties of one PSF task.
    """
    def __init__(self):
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
        super(TaskPSF, self).__init__()


    def _name_split(self, name):
        """See BaseTask.name_split.
        """
        fields = name.split(task_name_sep)
        if (len(fields) != 5) or (fields[1] != "psf"):
            raise RuntimeError("name \"{}\" not valid for a psf".format(name))
        ret = dict()
        ret["night"] = int(fields[0])
        ret["band"] = fields[2]
        ret["spec"] = int(fields[3])
        ret["expid"] = int(fields[4])
        return ret


    def _name_join(self, props):
        """See BaseTask.name_join.
        """
        return "{:08d}{}psf{}{:s}{}{:d}{}{:08d}".format(props["night"],
            task_name_sep, task_name_sep, props["band"], task_name_sep,
            props["spec"], task_name_sep, props["expid"])


    def _paths(self, name):
        """See BaseTask.paths.
        """
        props = self.name_split(name)
        camera = "{}{}".format(props["band"], props["spec"])
        return [ findfile("psf", night=props["night"], expid=props["expid"],
            camera=camera, groupname=None, nside=None, band=props["band"],
            spectrograph=props["spec"]) ]


    def _create(self, db):
        """See BaseTask.create.
        """
        with db.conn as con:
            createstr = "create table psf (name text unique"
            for col in zip(self._cols, self._coltypes):
                createstr = "{}, {} {}".format(createstr, col[0], col[1])
            createstr = "{})".format(createstr)
            con.execute(createstr)
        return


    def _insert(self, db, name, **kwargs):
        """See BaseTask.insert.
        """
        props = self.name_split(name)
        with db.conn as con:
            cur = con.cursor()
            cur.execute("insert into psf values (\"{}\", {}, \"{}\", {}, "
                "{}, {})".format(name, props["night"], props["band"],
                props["spec"], props["expid"], task_state_to_int["waiting"]))
            con.commit()
        return


    def _retrieve(self, db, name):
        """See BaseTask.retrieve.
        """
        ret = dict()
        with db.conn as con:
            cur = con.cursor()
            cur.execute(\
                "select * from psf where name = \"{}\"".format(name))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("task {} not in database".format(name))
            ret["name"] = name
            ret["night"] = row[1]
            ret["band"] = row[2]
            ret["spec"] = row[3]
            ret["expid"] = row[4]
            ret["state"] = task_int_to_state(row[5])
        return ret


    def _state_set(self, db, name, state):
        """See BaseTask.state_set.
        """
        with db.conn as con:
            cur = con.cursor()
            cur.execute("insert into psf(state) values "
                "({})".format(task_state_to_int(state)))
            con.commit()
        return


    def _state_get(self, db, name):
        """See BaseTask.state_get.
        """
        st = None
        with db.conn as con:
            cur = con.cursor()
            cur.execute(\
                "select state from psf where name = \"{}\"".format(name))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("task {} not in database".format(name))
            st = task_int_to_state(row[0])
        return st


    def _deps(self, name, db):
        """See BaseTask.deps.
        """
        from .base import task_classes, task_type

        props = self.name_split(name)
        boottask = task_classes["psfboot"].name_join(props)
        pixtask = task_classes["pix"].name_join(props)
        deptasks = [
            boottask,
            pixtask
        ]
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

        bootfile = None
        pixfile = None
        deplist = self.deps(name)
        for dp in deplist:
            if re.search("psfboot", dp) is not None:
                bootfile = task_classes["psfboot"].paths(dp)[0]
            if re.search("pix", dp) is not None:
                pixfile = task_classes["pix"].paths(dp)[0]
        if (bootfile is None) or (pixfile is None):
            raise RuntimeError("dependency list must include bootstrap "
                "and image files")
        options["input-image"] = pixfile
        options["input-psf"]   = bootfile
        options["output-psf"]  = self.path(name)

        if len(opts) > 0:
            extarray = option_list(opts)
            options["extra"] = " ".join(extarray)

        return option_list(options)


    def _cli_command(self, optlist, procs):
        """Return the equivalent command-line interface from the full
        option list (with files already appended).
        """
        entry = "desi_compute_psf"
        if procs > 1:
            entry = "desi_compute_psf_mpi"
        optstr = " ".join(optlist)
        return "{} {}".format(entry, optstr)


    def _run_cli(self, name, opts, procs):
        """See BaseTask.run_cli.
        """
        optlist = self._option_list(name, opts)
        return self._cli_command(optlist, procs)


    def _run(self, name, opts, comm):
        """See BaseTask.run.
        """
        from ...scripts import specex
        optlist = self._option_list(name, opts)

        args = specex.parse(optarray)
        specex.main(args, comm=comm)
        return
