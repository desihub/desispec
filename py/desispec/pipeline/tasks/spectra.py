#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function


# NOTE: only one class in this file should have a name that starts with "Task".

class TaskSpectra(object):
    """Class containing the properties of one spectra task.
    """
    def __init__(self):
        self._cols = [
            "nside",
            "pixel"
        ]
        self._coltypes = [
            "integer",
            "integer"
        ]


    def name_split(self, name):
        fields = name.split(task_name_sep)
        if (len(fields) != 3) or (fields[0] != "spectra"):
            raise RuntimeError("name \"{}\" not valid for a "
                "spectra".format(name))
        ret = dict()
        ret["nside"] = int(fields[1])
        ret["pixel"] = int(fields[2])
        return ret


    def name_join(self, props):
        return "spectra{}{:d}{}{:d}".format(task_name_sep, props["nside"],
            task_name_sep, props["pixel"])


    def create(self, db):
        with db.conn as con:
            createstr = "create table spectra (name text unique"
            for col in zip(self._cols, self._coltypes):
                createstr = "{}, {} {}".format(createstr, col[0], col[1])
            createstr = "{})".format(createstr)
            con.execute(createstr)
        return


    def insert(self, db, name, **kwargs):
        fields = name.split(task_name_sep)
        with db.conn as con
            cur = con.cursor()
            cur.execute("insert into spectra values (\"{}\", {}, "
                "{})".format(name, fields[1], fields[2],
                task_state_to_int("none")))
            con.commit()
        return


    def retrieve(self, db, name):
        ret = dict()
        with db.conn as con
            cur = con.cursor()
            cur.execute(\
                "select * from spectra where name = \"{}\"".format(name))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("task {} not in database".format(name))
            ret["name"] = name
            ret["nside"] = row[1]
            ret["pixel"] = row[2]
            ret["state"] = task_int_to_state(row[3])
        return ret


    def state_set(self, db, name, state):
        with db.conn as con
            cur = con.cursor()
            cur.execute("insert into spectra(state) values "
                "({})".format(task_state_to_int(state)))
            con.commit()
        return


    def state_get(self, db, name):
        st = None
        with db.conn as con
            cur = con.cursor()
            cur.execute(\
                "select state from spectra where name = \"{}\"".format(name))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("task {} not in database".format(name))
            st = task_int_to_state(row[0])
        return st


    def deps(self, db, name):
        fields = name.split(task_name_sep)
        return list()


    def run_weight(self, db, name):
        return 1


    def run_max_nproc(self, db, name):
        return 20


    def run_time(self, db, name):
        return 15 # in general faster but convergence slower for some realizations


    def run_defaults(self, db, name):
        opts = {}
        opts["trace-deg-wave"] = 7
        opts["trace-deg-x"] = 7
        opts["trace-prior-deg"] = 4

        # to get the lampline location, look in our path for specex
        # and use that install prefix to find the data directory.
        # if that directory does not exist, use a default NERSC
        # location.
        opts["lamplines"] = "/project/projectdirs/desi/software/edison/specex/specex-0.3.9/data/specex_linelist_desi.txt"
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exefile = os.path.join(path, "desi_psf_fit")
            if os.path.isfile(exefile) and os.access(exefile, os.X_OK):
                specexdir = os.path.join(path, "..", "data")
                opts["lamplines"] = os.path.join(specexdir, "specex_linelist_desi.txt")

        return opts


    def run(self, db, name, opts, comm=None):
        """Run the PSF estimation.

        This calls the MPI wrapper around calls to the (serial, but
        threaded) libspecex routines which do a per-bundle estimate.

        Args:
            db (pipeline.db.DB): The database.
            name (str): the name of this task.
            opts (dict): options to use for this task.
            comm (mpi4py.MPI.Comm): optional MPI communicator.
        """
        nproc = 1
        rank = 0
        if comm is not None:
            nproc = comm.size
            rank = comm.rank

        log = get_logger()

        node = grph[task]
        night, obj = graph_night_split(task)
        (temp, band, spec, expid) = graph_name_split(obj)
        cam = "{}{}".format(band, spec)

        pix = []
        inpsf = []
        for input in node["in"]:

            print("DEBUG  input , grph[input]=", input,grph[input])

            inode = grph[input]
            if inode["type"] == "psfboot":
                inpsf.append(input)
            elif inode["type"] == "pix":
                pix.append(input)
        if len(inpsf) != 1:
            raise RuntimeError("specex needs exactly one input psf file")
        if len(pix) != 1:
            raise RuntimeError("specex needs exactly one input image file")
        inpsffile = graph_path(inpsf[0])
        imgfile = graph_path(pix[0])
        outfile = graph_path(task)

        options = {}
        options["input-image"] = imgfile
        options["input-psf"]   = inpsffile
        options["output-psf"]  = outfile
        #if log.getEffectiveLevel() == DEBUG:
        #    options["debug"] = True

        if len(opts) > 0:
            extarray = option_list(opts)
            options["extra"] = " ".join(extarray)

        optarray = option_list(options)

        # at debug level, write out the equivalent commandline
        if rank == 0:
            com = ["RUN", "desi_compute_psf"]
            com.extend(optarray)
            log.info(" ".join(com))

        args = specex.parse(optarray)
        specex.main(args, comm=comm)

        return











        def __init__(self, features, ):
            self._features = [
                "type",
                "nside",
                "pixel"
            ]
            self._extra = {
                "targets" : "blob"
            }


        @property
        def features(self):
            """The list of common features describing this task.

            Returns:
                list: the list of features, which must all be defined in
                    pipeline.defs.task_features.
            """
            return self._features


        @property
        def extra(self):
            """Extra properties of this object.

            These properties will not be used when constructing the task name, but
            they will be used as additional database columns.  The returned
            dictionary has keys which are the property names and values which are
            valid SQL datatypes like "text" or "integer".

            Returns:
                dict: dictionary of properties and their database types.
            """
            return self._extra


        def dependencies(self, db, task):
            """Compute dependency tasks.

            Given a database and a dictionary describing one task, compute and
            return all the dependent tasks.  We keep the task information as a
            dictionary to avoid converting back and forth between strings and the
            features needed for database lookups as we recursively find
            dependencies.

            Args:
                db (pipeline.db.DB): The database.
                task (dict): A dictionary of task features and their values for
                    one particular task.

            Returns:
                list:  a list of tasks, each one a dictionary of features like the
                    input to this method.

            """
            return list()
