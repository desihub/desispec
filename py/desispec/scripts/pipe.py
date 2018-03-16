#!/usr/bin/env python
#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-

"""
Interactive control of the pipeline
"""

from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import re
import glob
import subprocess
import numpy as np

from .. import io

from .. import pipeline as pipe


class clr:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    def disable(self):
        self.HEADER = ""
        self.OKBLUE = ""
        self.OKGREEN = ""
        self.WARNING = ""
        self.FAIL = ""
        self.ENDC = ""


class PipeUI(object):

    def __init__(self):
        self.pref = "DESI"

        parser = argparse.ArgumentParser(
            description="DESI pipeline control",
            usage="""desi_pipe <command> [options]

Where supported commands are:
   create   Create a new production.
   tasks    Get all possible tasks for a given type and states.
   check    Check the status of tasks.
   dryrun   Return the equivalent command line entrypoint for tasks.
   script   Generate a shell or slurm script.
   run      Generate a script and run it.
   env      Print current production location.
   update   Update an existing production.
   getready Auto-Update of prod DB.
   sync     Synchronize DB state based on the filesystem.
   status   Overview of production.
   top      Live display of production database.
""")
        parser.add_argument("command", help="Subcommand to run")
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")
            parser.print_help()
            sys.exit(0)

        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()


    def env(self):
        rawdir = io.rawdata_root()
        proddir = io.specprod_root()
        print("{}{:<22} = {}{}{}".format(self.pref, "Raw data directory", clr.OKBLUE, rawdir, clr.ENDC))
        print("{}{:<22} = {}{}{}".format(self.pref, "Production directory", clr.OKBLUE, proddir, clr.ENDC))
        return


    def create(self):
        parser = argparse.ArgumentParser(\
            description="Create a new production",
            usage="desi_pipe create [options] (use --help for details)")

        parser.add_argument("--root", required=False, default=None,
            help="value to use for DESI_ROOT")

        parser.add_argument("--data", required=False, default=None,
            help="value to use for DESI_SPECTRO_DATA")

        parser.add_argument("--redux", required=False, default=None,
            help="value to use for DESI_SPECTRO_REDUX")

        parser.add_argument("--prod", required=False, default=None,
            help="value to use for SPECPROD")

        parser.add_argument("--basis", required=False, default=None,
            help="value to use for DESI_BASIS_TEMPLATES")

        parser.add_argument("--calib", required=False, default=None,
            help="value to use for DESI_CCD_CALIBRATION_DATA")

        parser.add_argument("--db-sqlite", required=False, default=False,
            action="store_true", help="Use SQLite database backend.")

        parser.add_argument("--db-sqlite-path", type=str, required=False,
            default=None, help="Override path to SQLite DB")

        parser.add_argument("--db-postgres", required=False, default=False,
            action="store_true", help="Use PostgreSQL database backend.  "
            "You must correctly configure your ~/.pgpass file!")

        parser.add_argument("--db-postgres-host", type=str, required=False,
            default="nerscdb03.nersc.gov", help="Set PostgreSQL hostname")

        parser.add_argument("--db-postgres-port", type=int, required=False,
            default=5432, help="Set PostgreSQL port number")

        parser.add_argument("--db-postgres-name", type=str, required=False,
            default="desidev", help="Set PostgreSQL DB name")

        parser.add_argument("--db-postgres-user", type=str, required=False,
            default="desidev_admin", help="Set PostgreSQL user name")

        parser.add_argument("--db-postgres-authorized", type=str,
            required=False, default="desidev_ro",
            help="Additional PostgreSQL users / roles to authorize")

        parser.add_argument("--nside", required=False, type=int, default=64,
            help="HEALPix nside value to use for spectral grouping.")

        args = parser.parse_args(sys.argv[2:])

        # Check desi root location

        desiroot = None
        if args.root is not None:
            desiroot = os.path.abspath(args.root)
            os.environ["DESI_ROOT"] = desiroot
        elif "DESI_ROOT" in os.environ:
            desiroot = os.environ["DESI_ROOT"]
        else:
            print("You must set DESI_ROOT in your environment or "
                "use the --root commandline option")
            sys.exit(0)

        # Check raw data location

        rawdir = None
        if args.data is not None:
            rawdir = os.path.abspath(args.data)
            os.environ["DESI_SPECTRO_DATA"] = rawdir
        elif "DESI_SPECTRO_DATA" in os.environ:
            rawdir = os.environ["DESI_SPECTRO_DATA"]
        else:
            print("You must set DESI_SPECTRO_DATA in your environment or "
                "use the --data commandline option")
            sys.exit(0)

        # Check production name

        prodname = None
        if args.prod is not None:
            prodname = args.prod
            os.environ["SPECPROD"] = prodname
        elif "SPECPROD" in os.environ:
            prodname = os.environ["SPECPROD"]
        else:
            print("You must set SPECPROD in your environment or use the "
                "--prod commandline option")
            sys.exit(0)

        # Check spectro redux location

        specdir = None
        if args.redux is not None:
            specdir = os.path.abspath(args.redux)
            os.environ["DESI_SPECTRO_REDUX"] = specdir
        elif "DESI_SPECTRO_REDUX" in os.environ:
            specdir = os.environ["DESI_SPECTRO_REDUX"]
        else:
            print("You must set DESI_SPECTRO_REDUX in your environment or "
                "use the --redux commandline option")
            sys.exit(0)

        proddir = os.path.join(specdir, prodname)
        if os.path.exists(proddir):
            print("Production {} exists.".format(proddir))
            print("Either remove this directory if you want to start fresh")
            print("or use 'desi_pipe update' to update a production.")
            sys.stdout.flush()
            sys.exit(0)

        # Check basis template location

        basis = None
        if args.basis is not None:
            basis = os.path.abspath(args.basis)
            os.environ["DESI_BASIS_TEMPLATES"] = basis
        elif "DESI_BASIS_TEMPLATES" in os.environ:
            basis = os.environ["DESI_BASIS_TEMPLATES"]
        else:
            print("You must set DESI_BASIS_TEMPLATES in your environment or "
                "use the --basis commandline option")
            sys.exit(0)

        # Check calibration location

        cabib = None
        if args.calib is not None:
            calib = os.path.abspath(args.calib)
            os.environ["DESI_CCD_CALIBRATION_DATA"] = calib
        elif "DESI_CCD_CALIBRATION_DATA" in os.environ:
            calib = os.environ["DESI_CCD_CALIBRATION_DATA"]
        else:
            print("You must set DESI_CCD_CALIBRATION_DATA in your "
                "environment or use the --calib commandline option")
            sys.exit(0)

        # Construct our DB connection string

        dbpath = None
        if args.db_postgres:
            # We are creating a new postgres backend. Explicitly create the
            # database, so that we can get the schema key.
            db = pipe.DataBasePostgres(host=args.db_postgres_host,
                port=args.db_postgres_port, dbname=args.db_postgres_name,
                user=args.db_postgres_user, schema=None,
                authorize=args.db_postgres_authorized)

            dbprops = [
                "postgresql",
                args.db_postgres_host,
                "{}".format(args.db_postgres_port),
                args.db_postgres_name,
                args.db_postgres_user,
                db.schema
            ]
            dbpath = ":".join(dbprops)
            os.environ["DESI_SPECTRO_DB"] = dbpath

        elif args.db_sqlite:
            # We are creating a new sqlite backend
            if args.db_sqlite_path is not None:
                # We are using a non-default path
                dbpath = os.path.abspath(args.db_sqlite_path)
            else:
                # We are using sqlite with the default location
                dbpath = os.path.join(proddir, "desi.db")
                if not os.path.isdir(proddir):
                    os.makedirs(proddir)

            # Create the database
            db = pipe.DataBaseSqlite(dbpath, "w")

            os.environ["DESI_SPECTRO_DB"] = dbpath

        elif "DESI_SPECTRO_DB" in os.environ:
            # We are using an existing prod
            dbpath = os.environ["DESI_SPECTRO_DB"]

        else:
            # Error- we have to get the DB info from somewhere
            print("You must set DESI_SPECTRO_DB in your environment or "
                "use the --db-sqlite or --db-postgres commandline options")
            sys.exit(0)

        pipe.update_prod(nightstr=None, hpxnside=args.nside)

        # create setup shell snippet

        setupfile = os.path.abspath(os.path.join(proddir, "setup.sh"))
        with open(setupfile, "w") as s:
            s.write("# Generated by desi_pipe\n")
            s.write("export DESI_ROOT={}\n\n".format(desiroot))
            s.write("export DESI_BASIS_TEMPLATES={}\n".format(basis))
            s.write("export DESI_CCD_CALIBRATION_DATA={}\n\n".format(calib))
            s.write("export DESI_SPECTRO_DATA={}\n".format(rawdir))
            s.write("export DESI_SPECTRO_REDUX={}\n".format(specdir))
            s.write("export SPECPROD={}\n".format(prodname))
            s.write("export DESI_SPECTRO_DB={}\n".format(dbpath))
            s.write("\n")
            if "DESI_LOGLEVEL" in os.environ:
                s.write("export DESI_LOGLEVEL=\"{}\"\n\n"\
                    .format(os.environ["DESI_LOGLEVEL"]))
            else:
                s.write("#export DESI_LOGLEVEL=\"DEBUG\"\n\n")

        print("\nTo use this production, you should do:")
        print("%> source {}\n".format(setupfile))

        return


    def update(self):
        parser = argparse.ArgumentParser(description="Update a production",
            usage="desi_pipe update [options] (use --help for details)")

        parser.add_argument("--nights", required=False, default=None,
            help="comma separated (YYYYMMDD) or regex pattern- only nights "
            "matching these patterns will be examined.")

        parser.add_argument("--nside", required=False, type=int, default=64,
            help="HEALPix nside value to use for spectral grouping.")

        args = parser.parse_args(sys.argv[2:])

        pipe.update_prod(nightstr=args.nights, hpxnside=args.nside)

        return


    def tasks(self):
        availtypes = ",".join(pipe.db.task_types())

        parser = argparse.ArgumentParser(description="Get all tasks of a "
            "particular type for one or more nights",
            usage="desi_pipe tasks [options] (use --help for details)")

        parser.add_argument("--tasktype", required=True, default=None,
            help="task type ({})".format(availtypes))

        parser.add_argument("--nights", required=False, default=None,
            help="comma separated (YYYYMMDD) or regex pattern- only nights "
            "matching these patterns will be examined.")

        parser.add_argument("--states", required=False, default=None,
            help="comma separated list of states (see defs.py).  Only tasks "
            "in these states will be returned.")

        parser.add_argument("--taskfile", required=False, default=None,
            help="write tasks to this file (if not specified, write to STDOUT)")

        parser.add_argument("--db-postgres-user", type=str, required=False,
            default="desidev_ro", help="If using postgres, connect as this "
            "user for read-only access")

        args = parser.parse_args(sys.argv[2:])

        states = None
        if args.states is None:
            states = pipe.task_states
        else:
            states = args.states.split(",")
            for s in states:
                if s not in pipe.task_states:
                    print("Task state '{}' is not valid".format(s))
                    sys.exit(0)

        dbpath = io.get_pipe_database()
        db = pipe.load_db(dbpath, mode="r", user=args.db_postgres_user)

        allnights = io.get_nights(strip_path=True)
        nights = pipe.prod.select_nights(allnights, args.nights)
        ntlist = ",".join(nights)

        tasks = list()
        with db.cursor() as cur:

            if args.tasktype == "spectra" or args.tasktype == "redshift" :
                
                cmd = "select pixel from healpix_frame where night in ({})".format(ntlist)
                cur.execute(cmd)
                pixels = np.unique([ x for (x,) in cur.fetchall() ]).tolist()
                pixlist = ",".join([ str(p) for p in pixels])
                cmd = "select name,state from {} where pixel in ({})".format( args.tasktype,pixlist)
                cur.execute(cmd)
                tasks = [ x for (x, y) in cur.fetchall() if \
                          pipe.task_int_to_state[y] in states ]
                
            else :
                cmd = "select name, state from {} where night in ({})"\
                    .format(args.tasktype, ntlist)
                cur.execute(cmd)
                tasks = [ x for (x, y) in cur.fetchall() if \
                          pipe.task_int_to_state[y] in states ]

        pipe.prod.task_write(args.taskfile, tasks)

        return


    def getready(self):
        dbpath = io.get_pipe_database()
        db     = pipe.load_db(dbpath, mode="w")
        db.getready()


    def check(self):
        parser = argparse.ArgumentParser(\
            description="Check the state of pipeline tasks",
            usage="desi_pipe check [options] (use --help for details)")

        parser.add_argument("--taskfile", required=False, default=None,
            help="read tasks from this file (if not specified, read from "
            "STDIN)")

        parser.add_argument("--nodb", required=False, default=False,
            action="store_true", help="Do not use the production database.")

        parser.add_argument("--db-postgres-user", type=str, required=False,
            default="desidev_ro", help="If using postgres, connect as this "
            "user for read-only access")

        args = parser.parse_args(sys.argv[2:])

        tasks = pipe.prod.task_read(args.taskfile)

        db = None
        if not args.nodb:
            dbpath = io.get_pipe_database()
            db = pipe.load_db(dbpath, mode="r", user=args.db_postgres_user)

        states = pipe.db.check_tasks(tasks, db=db)

        for tsk in tasks:
            print("{} : {}".format(tsk, states[tsk]))
        sys.stdout.flush()

        return


    def sync(self):
        availtypes = ",".join(pipe.db.task_types())

        parser = argparse.ArgumentParser(\
            description="Synchronize DB state based on the filesystem.",
            usage="desi_pipe sync [options] (use --help for details)")

        parser.add_argument("--nights", required=False, default=None,
            help="comma separated (YYYYMMDD) or regex pattern- only nights "
            "matching these patterns will be examined.")

        args = parser.parse_args(sys.argv[2:])

        dbpath = io.get_pipe_database()
        db = pipe.load_db(dbpath, mode="w")

        allnights = io.get_nights(strip_path=True)
        nights = pipe.prod.select_nights(allnights, args.nights)

        for nt in nights:
            db.sync(nt)
        return


    def cleanup(self):
        parser = argparse.ArgumentParser(\
            description="Clean up stale task states in the DB",
            usage="desi_pipe cleanup [options] (use --help for details)")

        parser.add_argument("--failed", required=False, default=False,
            action="store_true", help="Also clear failed states")

        args = parser.parse_args(sys.argv[2:])

        dbpath = io.get_pipe_database()
        db = pipe.load_db(dbpath, mode="w")
        db.cleanup(cleanfailed=args.failed)
        return


    def _parse_run_opts(self, parser):
        """Internal function to parse options for running.

        This provides a consistent set of run-time otpions for the
        "dryrun", "script", and "run" commands.

        """
        availtypes = ",".join(pipe.db.task_types())
        scrdir = io.get_pipe_scriptdir()

        parser.add_argument("--tasktype", required=True, default=None,
            help="task type ({})".format(availtypes))

        parser.add_argument("--taskfile", required=False, default=None,
            help="read tasks from this file (if not specified, read from "
            "STDIN)")

        parser.add_argument("--nersc", required=False, default=None,
            help="write a script for this NERSC system (edison | cori-haswell "
            "| cori-knl)")

        parser.add_argument("--nersc_queue", required=False, default="regular",
            help="write a script for this NERSC queue (debug | regular)")

        parser.add_argument("--nersc_runtime", required=False, type=int,
            default=30, help="Then maximum run time (in minutes) for a single "
            " job.  If the list of tasks cannot be run in this time, multiple "
            " job scripts will be written")

        parser.add_argument("--nersc_shifter", required=False, default=None,
            help="The shifter image to use for NERSC jobs")

        parser.add_argument("--mpi_procs", required=False, type=int, default=1,
            help="The number of MPI processes to use for non-NERSC shell "
            "scripts (default 1)")

        parser.add_argument("--mpi_run", required=False, type=str,
            default="", help="The command to launch MPI programs "
            "for non-NERSC shell scripts (default do not use MPI)")

        parser.add_argument("--procs_per_node", required=False, type=int,
            default=0, help="The number of processes to use per node.  If not "
            "specified it uses a default value for each machine.")

        parser.add_argument("--outdir", required=False, default=scrdir,
            help="put scripts and logs in this directory relative to the "
            "production 'run' directory.")

        parser.add_argument("--nodb", required=False, default=False,
            action="store_true", help="Do not use the production database.")

        parser.add_argument("--db-postgres-user", type=str, required=False,
            default="desidev_ro", help="If using postgres, connect as this "
            "user for read-only access")

        parser.add_argument("--debug", required=False, default=False,
            action="store_true", help="debugging messages in job logs")

        args = parser.parse_args(sys.argv[2:])

        return args


    def dryrun(self):

        parser = argparse.ArgumentParser(description="Print equivalent "
            "command-line jobs that would be run given the tasks and total"
            "number of processes",
            usage="desi_pipe dryrun [options] (use --help for details)")

        args = self._parse_run_opts(parser)

        tasks = pipe.prod.task_read(args.taskfile)

        (db, opts) = pipe.prod.load_prod("r")
        if args.nodb:
            db = None

        ppn = args.procs_per_node

        if args.nersc is None:
            # Not running at NERSC
            if ppn <= 0:
                ppn = args.mpi_procs
            pipe.run.dry_run(args.tasktype, tasks, opts, args.mpi_procs,
                ppn, db=db, launch="mpirun -n", force=False)
        else:
            # Running at NERSC
            hostprops = pipe.scriptgen.nersc_machine(args.nersc,
                args.nersc_queue)
            if ppn <= 0:
                ppn = hostprops["nodecores"]

            joblist = pipe.scriptgen.nersc_job_size(args.tasktype, tasks,
                args.nersc, args.nersc_queue, args.nersc_runtime, nodeprocs=ppn,
                db=db)

            launch="srun -n"
            for (jobnodes, jobtasks) in joblist:
                jobprocs = jobnodes * ppn
                pipe.run.dry_run(args.tasktype, jobtasks, opts, jobprocs,
                    ppn, db=db, launch=launch, force=False)

        return


    def _gen_script(self, args):

        proddir = os.path.abspath(io.specprod_root())

        tasks = pipe.prod.task_read(args.taskfile)

        outsubdir = args.outdir

        outdir = os.path.join(proddir, io.get_pipe_rundir(), outsubdir)

        mstr = "shell"
        if args.nersc is not None:
            mstr = args.nersc

        outstr = "{}_{}".format(args.tasktype, mstr)
        outscript = os.path.join(outdir, outstr)
        outlog = os.path.join(outdir, outstr)

        (db, opts) = pipe.prod.load_prod("r")
        if args.nodb:
            db = None

        ppn = args.procs_per_node

        # FIXME: Add openmp / multiproc function to task classes and
        # call them here.

        scripts = None

        if args.nersc is None:
            # Not running at NERSC
            scripts = pipe.scriptgen.batch_shell(args.tasktype, tasks,
                outscript, outlog, mpirun=args.mpi_run,
                mpiprocs=args.mpi_procs, openmp=1, db=db)

        else:
            # Running at NERSC
            if ppn <= 0:
                hostprops = pipe.scriptgen.nersc_machine(args.nersc,
                    args.nersc_queue)
                ppn = hostprops["nodecores"]

            scripts = pipe.scriptgen.batch_nersc(args.tasktype, tasks,
                outscript, outlog, args.tasktype, args.nersc, args.nersc_queue,
                args.nersc_runtime, nodeprocs=ppn, openmp=False,
                multiproc=False, db=db, shifterimg=args.nersc_shifter,
                debug=args.debug)

        return scripts


    def script(self):
        parser = argparse.ArgumentParser(description="Create a batch script "
            "for the list of tasks.  If the --nersc option is not given, "
            "create a shell script that optionally uses mpirun.",
            usage="desi_pipe script [options] (use --help for details)")

        args = self._parse_run_opts(parser)

        scripts = self._gen_script(args)
        print(",".join(scripts))

        return


    def run(self):
        parser = argparse.ArgumentParser(description="Create and run a batch "
            "script for the list of tasks.  If the --nersc option is not "
            "given, create a shell script that optionally uses mpirun.",
            usage="desi_pipe run [options] (use --help for details)")

        args = self._parse_run_opts(parser)

        print("Not yet implemented")
        sys.stdout.flush()
        return

        scripts = self._gen_script(args)

        return


    def status(self):
        parser = argparse.ArgumentParser(\
            description="Explore status of pipeline tasks",
            usage="desi_pipe status [options] (use --help for details)")

        parser.add_argument("--nodb", required=False, default=False,
            action="store_true", help="Do not use the production database.")

        args = parser.parse_args(sys.argv[2:])

        db = None
        if not args.nodb:
            dbpath = io.get_pipe_database()
            db = pipe.db.DataBase(dbpath, "r")

        tasktypes = pipe.db.task_types()

        states = pipe.db.check_tasks(tasks, db=db)

        for tsk in tasks:
            print("{} : {}".format(tsk, states[tsk]))
        sys.stdout.flush()

        return



    #
    # def all(self):
    #     self.load_state()
    #     # go through the current state and accumulate success / failure
    #     status = {}
    #     for st in pipe.step_types:
    #         status[st] = {}
    #         status[st]["total"] = 0
    #         status[st]["none"] = 0
    #         status[st]["running"] = 0
    #         status[st]["fail"] = 0
    #         status[st]["done"] = 0
    #
    #     fts = pipe.file_types_step
    #     for name, nd in self.grph.items():
    #         tp = nd["type"]
    #         if tp in fts.keys():
    #             status[fts[tp]]["total"] += 1
    #             status[fts[tp]][nd["state"]] += 1
    #
    #     for st in pipe.step_types:
    #         beg = ""
    #         if status[st]["done"] == status[st]["total"]:
    #             beg = clr.OKGREEN
    #         elif status[st]["fail"] > 0:
    #             beg = clr.FAIL
    #         elif status[st]["running"] > 0:
    #             beg = clr.WARNING
    #         print("{}    {}{:<12}{} {:>5} tasks".format(self.pref, beg, st, clr.ENDC, status[st]["total"]))
    #     print("")
    #     return
    #
    #
    # def step(self):
    #     parser = argparse.ArgumentParser(description="Details about a particular pipeline step")
    #     parser.add_argument("step", help="Step name (allowed values are: bootcalib, specex, psfcombine, extract, fiberflat, sky, stdstars, fluxcal, procexp, and zfind).")
    #     parser.add_argument("--state", required=False, default=None, help="Only list tasks in this state (allowed values are: done, fail, running, none)")
    #     # now that we"re inside a subcommand, ignore the first
    #     # TWO argvs
    #     args = parser.parse_args(sys.argv[2:])
    #
    #     if args.step not in pipe.step_types:
    #         print("Unrecognized step name")
    #         parser.print_help()
    #         sys.exit(0)
    #
    #     self.load_state()
    #
    #     tasks_done = []
    #     tasks_none = []
    #     tasks_fail = []
    #     tasks_running = []
    #
    #     fts = pipe.step_file_types[args.step]
    #     for name, nd in self.grph.items():
    #         tp = nd["type"]
    #         if tp == fts:
    #             stat = nd["state"]
    #             if stat == "done":
    #                 tasks_done.append(name)
    #             elif stat == "fail":
    #                 tasks_fail.append(name)
    #             elif stat == "running":
    #                 tasks_running.append(name)
    #             else:
    #                 tasks_none.append(name)
    #
    #     if (args.state is None) or (args.state == "done"):
    #         for tsk in sorted(tasks_done):
    #             print("{}    {}{:<20}{}".format(self.pref, clr.OKGREEN, tsk, clr.ENDC))
    #     if (args.state is None) or (args.state == "fail"):
    #         for tsk in sorted(tasks_fail):
    #             print("{}    {}{:<20}{}".format(self.pref, clr.FAIL, tsk, clr.ENDC))
    #     if (args.state is None) or (args.state == "running"):
    #         for tsk in sorted(tasks_running):
    #             print("{}    {}{:<20}{}".format(self.pref, clr.WARNING, tsk, clr.ENDC))
    #     if (args.state is None) or (args.state == "none"):
    #         for tsk in sorted(tasks_none):
    #             print("{}    {:<20}".format(self.pref, tsk))
    #
    #
    # def task(self):
    #     parser = argparse.ArgumentParser(description="Details about a specific pipeline task")
    #     parser.add_argument("task", help="Task name (as displayed by the \"step\" command).")
    #     parser.add_argument("--log", required=False, default=False, action="store_true", help="Print the log and traceback, if applicable")
    #     parser.add_argument("--retry", required=False, default=False, action="store_true", help="Retry the specified task")
    #     parser.add_argument("--opts", required=False, default=None, help="Retry using this options file")
    #     # now that we're inside a subcommand, ignore the first
    #     # TWO argvs
    #     args = parser.parse_args(sys.argv[2:])
    #
    #     self.load_state()
    #
    #     if args.task not in self.grph.keys():
    #         print("Task {} not found in graph.".format(args.task))
    #         sys.exit(0)
    #
    #     nd = self.grph[args.task]
    #     stat = nd["state"]
    #
    #     beg = ""
    #     if stat == "done":
    #         beg = clr.OKGREEN
    #     elif stat == "fail":
    #         beg = clr.FAIL
    #     elif stat == "running":
    #         beg = clr.WARNING
    #
    #     filepath = pipe.graph_path(args.task)
    #
    #     (night, gname) = pipe.graph_night_split(args.task)
    #     nfaildir = os.path.join(self.faildir, night)
    #     nlogdir = os.path.join(self.logdir, night)
    #
    #     logpath = os.path.join(nlogdir, "{}.log".format(gname))
    #
    #     ymlpath = os.path.join(nfaildir, "{}_{}.yaml".format(pipe.file_types_step[nd["type"]], args.task))
    #
    #     if args.retry:
    #         if stat != "fail":
    #             print("Task {} has not failed, cannot retry".format(args.task))
    #         else:
    #             if os.path.isfile(ymlpath):
    #                 newopts = None
    #                 if args.opts is not None:
    #                     newopts = pipe.yaml_read(args.opts)
    #                 try:
    #                     pipe.retry_task(ymlpath, newopts=newopts)
    #                 finally:
    #                     self.grph[args.task]["state"] = "done"
    #                     pipe.graph_db_write(self.grph)
    #             else:
    #                 print("Failure yaml dump does not exist!")
    #     else:
    #         print("{}{}:".format(self.pref, args.task))
    #         print("{}    state = {}{}{}".format(self.pref, beg, stat, clr.ENDC))
    #         print("{}    path = {}".format(self.pref, filepath))
    #         print("{}    logfile = {}".format(self.pref, logpath))
    #         print("{}    inputs required:".format(self.pref))
    #         for d in sorted(nd["in"]):
    #             print("{}      {}".format(self.pref, d))
    #         print("{}    output dependents:".format(self.pref))
    #         for d in sorted(nd["out"]):
    #             print("{}      {}".format(self.pref, d))
    #         print("")
    #
    #         if args.log:
    #             print("=========== Begin Log =============")
    #             print("")
    #             with open(logpath, "r") as f:
    #                 logdata = f.read()
    #                 print(logdata)
    #             print("")
    #             print("============ End Log ==============")
    #             print("")
    #
    #     return

    def top(self):
        parser = argparse.ArgumentParser(\
            description="Live overview of the production state",
            usage="desi_pipe top [options] (use --help for details)")

        parser.add_argument("--refresh", required=False, type=int, default=10,
            help="The number of seconds between DB queries")

        args = parser.parse_args(sys.argv[2:])

        import signal
        import time
        import numpy as np

        def signal_handler(signal, frame):
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)

        dbpath = io.get_pipe_database()
        db = pipe.load_db(dbpath, mode="r")

        tasktypes = pipe.tasks.base.default_task_chain

        header_state = ""
        for s in pipe.task_states:
            header_state = "{} {:8s}|".format(header_state, s)

        sep = "------------------------------------------------------------------------------"

        header = "{}\n{:26s}|{}\n{}".format(sep, "         Task Type        ",
            header_state, sep)

        while True:
            taskstates = {}
            with db.cursor() as cur:
                for t in tasktypes:
                    taskstates[t] = {}
                    cmd = "select state from {}".format(t)
                    cur.execute(cmd)
                    st = np.array([ int(x[0]) for x in cur.fetchall() ])
                    for s in pipe.task_states:
                        taskstates[t][s] = \
                            np.sum(st == pipe.task_state_to_int[s])

            print("\033c")
            print(header)
            for t in tasktypes:
                line = "{:26s}|".format(t)
                for s in pipe.task_states:
                    line = "{}{:9d}|".format(line, taskstates[t][s])
                print(line)
            print(sep)
            print(" (Use Ctrl-C to Quit) ")
            sys.stdout.flush()
            time.sleep(args.refresh)

        return


def main():
    p = PipeUI()
