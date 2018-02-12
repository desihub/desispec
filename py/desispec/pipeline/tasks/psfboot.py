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

class TaskPSFBoot(BaseTask):
    """Class containing the properties of one PSF bootstrap task.
    """
    def __init__(self):
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
        super(TaskPSFBoot, self).__init__()


    def _name_split(self, name):
        """See BaseTask.name_split.
        """
        fields = name.split(task_name_sep)
        if (len(fields) != 4) or (fields[1] != "psfboot"):
            raise RuntimeError("name \"{}\" not valid for a "
                "psfboot".format(name))
        ret = dict()
        ret["night"] = int(fields[0])
        ret["band"] = fields[2]
        ret["spec"] = int(fields[3])
        return ret


    def _name_join(self, props):
        """See BaseTask.name_join.
        """
        return "{:08d}{}psfboot{}{:s}{}{:d}".format(props["night"],
            task_name_sep, task_name_sep, props["band"], task_name_sep,
            props["spec"])


    def _paths(self, name):
        """See BaseTask.paths.
        """
        props = self.name_split(name)
        camera = "{}{}".format(props["band"], props["spec"])
        return [ findfile("psfboot", night=props["night"],
            camera=camera, groupname=None, nside=None, band=props["band"],
            spectrograph=props["spec"]) ]


    def _create(self, db):
        """See BaseTask.create.
        """
        with db.conn as con:
            createstr = "create table psfboot (name text unique"
            for col in zip(self._cols, self._coltypes):
                createstr = "{}, {}, {}".format(createstr, col[0], col[1])
            createstr = "{})".format(createstr)
            con.execute(createstr)
        return


    def _insert(self, db, name, **kwargs):
        """See BaseTask.insert.
        """
        props = self.name_split(name)
        with db.conn as con:
            cur = con.cursor()
            cur.execute('insert into psfboot values (\"{}\", {}, \"{}\", {}, '
                '{})'.format(name, props["night"], props["band"],
                props["spec"], task_state_to_int["waiting"]))
            con.commit()
        return


    def _retrieve(self, db, name):
        """See BaseTask.retrieve.
        """
        ret = dict()
        with db.conn as con:
            cur = con.cursor()
            cur.execute(\
                'select * from psfboot where name = "{}"'.format(name))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("task {} not in database".format(name))
            ret["name"] = name
            ret["night"] = row[1]
            ret["band"] = row[2]
            ret["spec"] = row[3]
            ret["state"] = task_int_to_state(row[4])
        return ret


    def _state_set(self, db, name, state):
        """See BaseTask.state_set.
        """
        with db.conn as con:
            cur = con.cursor()
            cur.execute('insert into psfboot(state) values '
                '({})'.format(task_state_to_int(state)))
            con.commit()
        return


    def _state_get(self, db, name):
        """See BaseTask.state_get.
        """
        st = None
        with db.conn as con:
            cur = con.cursor()
            cur.execute(\
                'select state from psfboot where name = "{}"'.format(name))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("task {} not in database".format(name))
            st = task_int_to_state(row[0])
        return st


    def _deps(self, name):
        """See BaseTask.deps.
        """
        from ._taskclass import task_classes
        props = self.name_split(name)

        arctasks = list()
        flattasks = list()
        if db is None:
            # Since we don't have a DB, we have to query the filesystem.


        else:
            # Query the pre-processed images for the arcs and flats.
            with db.conn as con
                cur = con.cursor()
                cur.execute(\
                    'select name from pix where night = {} and band = "{}" and '
                    'spec = {} and flavor = "arc"'.format(props["night"],
                    props["band"], props["spec"]))
                arctasks.extend(cur.fetchall())
                cur.execute(\
                    'select name from pix where night = {} and band = "{}" and '
                    'spec = {} and flavor = "flat"'.format(props["night"],
                    props["band"], props["spec"]))
                flattasks.extend(cur.fetchall())

        # Sort by expid
        arctasks = list(sorted(arctasks, key=lambda x: x["expid"]))
        flattasks = list(sorted(flattasks, key=lambda x: x["expid"]))

        # Just use the first one
        deptasks = [
            arctasks[0],
            flattasks[0]
        ]
        return deptasks


    def _run_max_procs(self, procs_per_node):
        """See BaseTask.run_max_procs.
        """
        return 1


    def _run_time(self, db, name):
        """See BaseTask.run_time.
        """
        return 30


    def _run_defaults(self):
        """See BaseTask.run_defaults.
        """
        opts = {}
        return opts


    def _option_list(self, name, opts):
        """Build the full list of options.

        This includes appending the filenames and incorporating runtime
        options.
        """
        from ._taskclass import task_classes

        #FIXME: change this once the depencency list is a dictionary.

        options = OrderedDict()

        bootfile = None
        pixfile = None
        deplist = self.deps(name)
        for dp in deplist:
            if re.search("psfboot", dp) is not None:
                bootfile = task_classes["psfboot"].path(dp)
            if re.search("pix", dp) is not None:
                pixfile = task_classes["pix"].path(dp)
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
        optlist = self._option_list(name, opts)

        args = specex.parse(optarray)
        specex.main(args, comm=comm)

        return



    def run(self, db, name, opts, comm=None):
        """Run the PSF bootstrap.

        This just uses the first arc for now.

        Args:
            db (pipeline.db.DB): The database.
            name (str): the name of this task.
            opts (dict): options to use for this task.
            comm (mpi4py.MPI.Comm): optional MPI communicator.
        """
        if comm is not None:
            if comm.size > 1:
                raise RuntimeError("PSF bootstrap should only be called with one process")

        log = get_logger()

        node = grph[task]
        night, obj = graph_night_split(task)
        (temp, band, spec) = graph_name_split(obj)
        cam = "{}{}".format(band, spec)

        arcs = []
        flats = []
        for input in node["in"]:
            inode = grph[input]
            if inode["flavor"] == "arc":
                arcs.append(input)
            elif inode["flavor"] == "flat":
                flats.append(input)
        if len(arcs) == 0:
            raise RuntimeError("no arc images found!")
        if len(flats) == 0:
            raise RuntimeError("no flat images found!")
        firstarc = sorted(arcs)[0]
        firstflat = sorted(flats)[0]

        arcpath = graph_path(firstarc)
        flatpath = graph_path(firstflat)
        outpath = graph_path(task)

        #qapath = io.findfile("qa_bootcalib", night=night, camera=cam, band=band, spectrograph=spec)

        # build list of options
        options = {}
        options["fiberflat"] = flatpath
        options["arcfile"] = arcpath
        #options["qafile"] = qapath
        options["outfile"] = outpath
        options.update(opts)
        optarray = option_list(options)

        # at debug level, log the equivalent commandline
        com = ["RUN", "desi_bootcalib"]
        com.extend(optarray)
        log.info(" ".join(com))

        args = bootcalib.parse(optarray)

        bootcalib.main(args)

        return
