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
from collections import OrderedDict

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
   chain    Run all ready tasks for multiple pipeline steps.
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
            sys.exit(1)

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

        parser.add_argument("--force", action = "store_true",
            help="force DB creation even if prod exists on disk (useful for simulations")

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
            sys.exit(1)

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
            sys.exit(1)

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
            sys.exit(1)

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
            sys.exit(1)

        proddir = os.path.join(specdir, prodname)
        if os.path.exists(proddir) and not args.force :
            print("Production {} exists.".format(proddir))
            print("Either remove this directory if you want to start fresh")
            print("or use 'desi_pipe update' to update a production")
            print("or rerun with --force option.")
            sys.stdout.flush()
            sys.exit(1)

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
            sys.exit(1)

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
            sys.exit(1)

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
            sys.exit(1)

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


    def _get_tasks(self, db, tasktype, states, nights):
        ntlist = ",".join(nights)
        tasks = list()
        with db.cursor() as cur:

            if tasktype == "spectra" or tasktype == "redshift":

                cmd = "select pixel from healpix_frame where night in ({})".format(ntlist)
                cur.execute(cmd)
                pixels = np.unique([ x for (x,) in cur.fetchall() ]).tolist()
                pixlist = ",".join([ str(p) for p in pixels])
                cmd = "select name,state from {} where pixel in ({})".format(tasktype, pixlist)
                cur.execute(cmd)
                tasks = [ x for (x, y) in cur.fetchall() if \
                          pipe.task_int_to_state[y] in states ]

            else :
                cmd = "select name, state from {} where night in ({})"\
                    .format(tasktype, ntlist)
                cur.execute(cmd)
                tasks = [ x for (x, y) in cur.fetchall() if \
                          pipe.task_int_to_state[y] in states ]
        return tasks


    def tasks(self):
        availtypes = ",".join(pipe.db.all_task_types())

        parser = argparse.ArgumentParser(description="Get all tasks of a "
            "particular type for one or more nights",
            usage="desi_pipe tasks [options] (use --help for details)")

        parser.add_argument("--tasktypes", required=True, default=None,
            help="comma separated list of task types ({})".format(availtypes))

        parser.add_argument("--nights", required=False, default=None,
            help="comma separated (YYYYMMDD) or regex pattern- only nights "
            "matching these patterns will be examined.")

        parser.add_argument("--states", required=False, default=None,
            help="comma separated list of states (see defs.py).  Only tasks "
            "in these states will be returned.")

        parser.add_argument("--nosubmitted", required=False, default=False,
            action="store_true", help="Skip all tasks flagged as submitted")

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
                    sys.exit(1)

        dbpath = io.get_pipe_database()
        db = pipe.load_db(dbpath, mode="r", user=args.db_postgres_user)

        allnights = io.get_nights(strip_path=True)
        nights = pipe.prod.select_nights(allnights, args.nights)

        ttypes = args.tasktypes.split(',')
        tasktypes = list()
        for tt in pipe.tasks.base.default_task_chain:
            if tt in ttypes:
                tasktypes.append(tt)

        all_tasks = list()
        for tt in tasktypes:
            tasks = self._get_tasks(db, tt, states, nights)
            if args.nosubmitted:
                if (tt != "spectra") and (tt != "redshift"):
                    sb = db.get_submitted(tasks)
                    tasks = [ x for x in tasks if not sb[x] ]
            all_tasks.extend(tasks)

        pipe.prod.task_write(args.taskfile, all_tasks)

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
        availtypes = ",".join(pipe.db.all_task_types())

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

        parser.add_argument("--submitted", required=False, default=False,
            action="store_true", help="Also clear submitted flag")

        args = parser.parse_args(sys.argv[2:])

        dbpath = io.get_pipe_database()
        db = pipe.load_db(dbpath, mode="w")
        db.cleanup(cleanfailed=args.failed, cleansubmitted=args.submitted)
        return


    def _parse_run_opts(self, parser):
        """Internal function to parse options for running.

        This provides a consistent set of run-time otpions for the
        "dryrun", "script", and "run" commands.

        """
        parser.add_argument("--nersc", required=False, default=None,
            help="write a script for this NERSC system (edison | cori-haswell "
            "| cori-knl)")

        parser.add_argument("--nersc_queue", required=False, default="regular",
            help="write a script for this NERSC queue (debug | regular)")

        parser.add_argument("--nersc_maxtime", required=False, type=int,
            default=0, help="Then maximum run time (in minutes) for a single "
            " job.  If the list of tasks cannot be run in this time, multiple "
            " job scripts will be written.  Default is the maximum time for "
            " the specified queue.")

        parser.add_argument("--nersc_maxnodes", required=False, type=int,
            default=0, help="The maximum number of nodes to use.  Default "
            " is the maximum nodes for the specified queue.")

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

        parser.add_argument("--outdir", required=False, default=None,
            help="Put task scripts and logs in this directory relative to the "
            "production 'scripts' directory.  Default puts task directory "
            "in the main scripts directory.")

        parser.add_argument("--debug", required=False, default=False,
            action="store_true", help="debugging messages in job logs")

        return parser


    def dryrun(self):
        availtypes = ",".join(pipe.db.all_task_types())

        parser = argparse.ArgumentParser(description="Print equivalent "
            "command-line jobs that would be run given the tasks and total"
            "number of processes",
            usage="desi_pipe dryrun [options] (use --help for details)")

        parser.add_argument("--taskfile", required=False, default=None,
            help="read tasks from this file (if not specified, read from "
            "STDIN)")

        parser = self._parse_run_opts(parser)

        parser.add_argument("--nodb", required=False, default=False,
            action="store_true", help="Do not use the production database.")

        parser.add_argument("--db-postgres-user", type=str, required=False,
            default="desidev_ro", help="If using postgres, connect as this "
            "user for read-only access")

        args = parser.parse_args(sys.argv[2:])

        tasks = pipe.prod.task_read(args.taskfile)
        tasks_by_type = pipe.db.task_sort(tasks)

        (db, opts) = pipe.prod.load_prod("r")
        if args.nodb:
            db = None

        ppn = args.procs_per_node

        if args.nersc is None:
            # Not running at NERSC
            if ppn <= 0:
                ppn = args.mpi_procs
            for tt, tlist in tasks_by_type.items():
                pipe.run.dry_run(tt, tlist, opts, args.mpi_procs,
                    ppn, db=db, launch="mpirun -n", force=False)
        else:
            # Running at NERSC
            hostprops = pipe.scriptgen.nersc_machine(args.nersc,
                args.nersc_queue)
            if ppn <= 0:
                ppn = hostprops["nodecores"]

            for tt, tlist in tasks_by_type.items():
                joblist = pipe.scriptgen.nersc_job_size(tt, tlist,
                    args.nersc, args.nersc_queue, args.nersc_maxtime,
                    args.nersc_maxnodes, nodeprocs=ppn, db=db)

                launch="srun -n"
                for (jobnodes, jobtime, jobtasks) in joblist:
                    jobprocs = jobnodes * ppn
                    pipe.run.dry_run(tt, jobtasks, opts, jobprocs,
                        ppn, db=db, launch=launch, force=False)

        return


    def _gen_scripts(self, tasks_by_type, nodb, args):
        ttypes = list(tasks_by_type.keys())
        jobname = ttypes[0]
        if len(ttypes) > 1:
            jobname = "{}-{}".format(ttypes[0], ttypes[-1])

        proddir = os.path.abspath(io.specprod_root())

        import datetime
        now = datetime.datetime.now()
        outtaskdir = "{}_{:%Y%m%d-%H%M%S}".format(jobname, now)

        if args.outdir is None:
            outdir = os.path.join(proddir, io.get_pipe_rundir(),
                io.get_pipe_scriptdir(), outtaskdir)
        else:
            outdir = os.path.join(proddir, io.get_pipe_rundir(),
                io.get_pipe_scriptdir(), args.outdir, outtaskdir)

        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        mstr = "run"
        if args.nersc is not None:
            mstr = args.nersc

        outscript = os.path.join(outdir, mstr)
        outlog = os.path.join(outdir, mstr)

        (db, opts) = pipe.prod.load_prod("r")
        if nodb:
            db = None

        ppn = args.procs_per_node

        # FIXME: Add openmp / multiproc function to task classes and
        # call them here.

        scripts = None

        if args.nersc is None:
            # Not running at NERSC
            scripts = pipe.scriptgen.batch_shell(tasks_by_type,
                outscript, outlog, mpirun=args.mpi_run,
                mpiprocs=args.mpi_procs, openmp=1, db=db)

        else:
            # Running at NERSC
            if ppn <= 0:
                hostprops = pipe.scriptgen.nersc_machine(args.nersc,
                    args.nersc_queue)
                ppn = hostprops["nodecores"]

            scripts = pipe.scriptgen.batch_nersc(tasks_by_type,
                outscript, outlog, jobname, args.nersc, args.nersc_queue,
                args.nersc_maxtime, args.nersc_maxnodes, nodeprocs=ppn,
                openmp=False, multiproc=False, db=db,
                shifterimg=args.nersc_shifter, debug=args.debug)

        return scripts


    def script(self):
        availtypes = ",".join(pipe.db.all_task_types())

        parser = argparse.ArgumentParser(description="Create batch script(s) "
            "for the list of tasks.  If the --nersc option is not given, "
            "create shell script(s) that optionally uses mpirun.  Prints"
            " the list of generated scripts to STDOUT.",
            usage="desi_pipe script [options] (use --help for details)")

        parser.add_argument("--taskfile", required=False, default=None,
            help="read tasks from this file (if not specified, read from "
            "STDIN)")

        parser = self._parse_run_opts(parser)

        parser.add_argument("--nodb", required=False, default=False,
            action="store_true", help="Do not use the production database.")

        parser.add_argument("--db-postgres-user", type=str, required=False,
            default="desidev_ro", help="If using postgres, connect as this "
            "user for read-only access")

        args = parser.parse_args(sys.argv[2:])

        tasks = pipe.prod.task_read(args.taskfile)

        if len(tasks) > 0:
            tasks_by_type = pipe.db.task_sort(tasks)
            scripts = self._gen_scripts(tasks_by_type, args.nodb, args)
            print(",".join(scripts))
        else:
            import warnings
            warnings.warn("Input task list is empty", RuntimeWarning)

        return


    def _run_scripts(self, scripts, deps=None, slurm=False):
        import subprocess as sp

        depstr = ""
        if deps is not None:
            depstr = "-d afterok"
            for d in deps:
                depstr = "{}:{}".format(depstr, d)

        jobids = list()
        if slurm:
            # submit each job and collect the job IDs
            for scr in scripts:
                sout = sp.check_output("sbatch {} {}".format(depstr, scr),
                    shell=True, universal_newlines=True)
                jid = sout.split()[3]
                print("submitted job {} script {}".format(jid,scr))
                jobids.append(jid)
        else:
            # run the scripts one at a time
            for scr in scripts:
                rcode = sp.call(scr, shell=True)
                if rcode != 0:
                    print("WARNING:  script {} had return code = {}"\
                          .format(scr, rcode))
        return jobids


    def run(self):
        availtypes = ",".join(pipe.db.all_task_types())

        parser = argparse.ArgumentParser(description="Create and run batch "
            "script(s) for the list of tasks.  If the --nersc option is not "
            "given, create shell script(s) that optionally uses mpirun.",
            usage="desi_pipe run [options] (use --help for details)")

        parser.add_argument("--taskfile", required=False, default=None,
            help="Read tasks from this file (if not specified, read from "
            "STDIN).  Tasks of all types will be packed into a single job!")

        parser.add_argument("--nosubmitted", required=False, default=False,
            action="store_true", help="Skip all tasks flagged as submitted")

        parser.add_argument("--depjobs", required=False, default=None,
            help="comma separated list of slurm job IDs to specify as "
            "dependencies of this current job.")

        parser = self._parse_run_opts(parser)

        parser.add_argument("--nodb", required=False, default=False,
            action="store_true", help="Do not use the production database.")

        args = parser.parse_args(sys.argv[2:])

        tasks = pipe.prod.task_read(args.taskfile)

        if len(tasks) > 0:
            tasks_by_type = pipe.db.task_sort(tasks)
            tasktypes = list(tasks_by_type.keys())
            # We are packing everything into one job
            scripts = self._gen_scripts(tasks_by_type, args.nodb, args)

            deps = None
            slurm = False
            if args.nersc is not None:
                slurm = True
            if args.depjobs is not None:
                deps = args.depjobs.split(',')

            # Run the jobs
            if not args.nodb:
                # We can use the DB, mark tasks as submitted.
                if slurm:
                    dbpath = io.get_pipe_database()
                    db = pipe.load_db(dbpath, mode="w")
                    for tt in tasktypes:
                        if (tt != "spectra") and (tt != "redshift"):
                            db.set_submitted_type(tt, tasks_by_type[tt])

            jobids = self._run_scripts(scripts, deps=deps, slurm=slurm)
            if len(jobids) > 0:
                print(",".join(jobids))
        else:
            import warnings
            warnings.warn("Input task list is empty", RuntimeWarning)
        return


    def chain(self):
        parser = argparse.ArgumentParser(description="Create a chain of jobs"
            " using all ready tasks for each specified step.  The order of"
            " the pipeline steps is fixed, regardless of the order specified"
            " by the --tasktypes option.",
            usage="desi_pipe chain [options] (use --help for details)")

        parser.add_argument("--tasktypes", required=False, default=",".join(pipe.tasks.base.default_task_chain),
            help="comma separated list of slurm job IDs to specify as "
            "dependencies of this current job.")

        parser.add_argument("--nights", required=False, default=None,
            help="comma separated (YYYYMMDD) or regex pattern- only nights "
            "matching these patterns will be generated.")

        parser.add_argument("--states", required=False, default=None,
            help="comma separated list of states (see defs.py).  Only tasks "
            "in these states will be scheduled.")

        parser.add_argument("--pack", required=False, default=False,
            action="store_true", help="Pack the chain of pipeline steps "
            "into a single job script.")

        parser.add_argument("--nosubmitted", required=False, default=False,
            action="store_true", help="Skip all tasks flagged as submitted")

        parser.add_argument("--depjobs", required=False, default=None,
            help="comma separated list of slurm job IDs to specify as "
            "dependencies of this current job.")

        parser = self._parse_run_opts(parser)

        args = parser.parse_args(sys.argv[2:])

        print("Step(s) to run:",args.tasktypes)

        machprops = None
        if args.nersc is not None:
            machprops = pipe.scriptgen.nersc_machine(args.nersc,
                args.nersc_queue)

        # FIXME:  we should support task selection by exposure ID as well.

        states = None
        if args.states is None:
            states = pipe.task_states
        else:
            states = args.states.split(",")
            for s in states:
                if s not in pipe.task_states:
                    print("Task state '{}' is not valid".format(s))
                    sys.exit(1)
        ttypes = args.tasktypes.split(',')
        tasktypes = list()
        for tt in pipe.tasks.base.default_task_chain:
            if tt in ttypes:
                tasktypes.append(tt)

        if (machprops is not None) and (not args.pack):
            if len(tasktypes) > machprops["submitlimit"]:
                print("Queue {} on machine {} limited to {} jobs."\
                    .format(args.nersc_queue, args.nersc,
                    machprops["submitlimit"]))
                print("Use a different queue or shorter chains of tasks.")
                sys.exit(1)

        slurm = False
        if args.nersc is not None:
            slurm = True

        dbpath = io.get_pipe_database()
        db = pipe.load_db(dbpath, mode="w")

        allnights = io.get_nights(strip_path=True)
        nights = pipe.prod.select_nights(allnights, args.nights)

        outdeps = None
        indeps = None
        if args.depjobs is not None:
            indeps = args.depjobs.split(',')

        tasks_by_type = OrderedDict()
        for tt in tasktypes:
            # Get the tasks.  We select by state and submitted status.
            tasks = self._get_tasks(db, tt, states, nights)
            if args.nosubmitted:
                if (tt != "spectra") and (tt != "redshift"):
                    sb = db.get_submitted(tasks)
                    tasks = [ x for x in tasks if not sb[x] ]

            if len(tasks) == 0:
                import warnings
                warnings.warn("Input task list for '{}' is empty".format(tt),
                              RuntimeWarning)
                break
            tasks_by_type[tt] = tasks

        scripts = None
        tscripts = None
        if args.pack:
            # We are packing everything into one job
            scripts = self._gen_scripts(tasks_by_type, False, args)
        else:
            # Generate individual scripts
            tscripts = dict()
            for tt in tasktypes:
                onetype = OrderedDict()
                onetype[tt] = tasks_by_type[tt]
                tscripts[tt] = self._gen_scripts(onetype, False, args)

        # Run the jobs
        if slurm:
            for tt in tasktypes:
                if (tt != "spectra") and (tt != "redshift"):
                    db.set_submitted_type(tt, tasks_by_type[tt])

        outdeps = None
        if args.pack:
            # Submit one job
            outdeps = self._run_scripts(scripts, deps=indeps, slurm=slurm)
        else:
            # Loop over task types submitting jobs and tracking dependencies.
            for tt in tasktypes:
                outdeps = self._run_scripts(tscripts[tt], deps=indeps,
                    slurm=slurm)
                if len(outdeps) > 0:
                    indeps = outdeps
                else:
                    indeps = None

        if outdeps is not None and len(outdeps) > 0:
            print(",".join(outdeps))

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

        tasktypes = pipe.db.all_task_types()

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
    #         sys.exit(1)
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
    #         sys.exit(1)
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

        parser.add_argument("--db-postgres-user", type=str, required=False,
            default="desidev_ro", help="If using postgres, connect as this "
            "user for read-only access")

        parser.add_argument("--once", required=False, action="store_true",
            default=False, help="Print info once without clearing the terminal")

        args = parser.parse_args(sys.argv[2:])

        import signal
        import time
        import numpy as np

        def signal_handler(signal, frame):
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)

        dbpath = io.get_pipe_database()
        db = pipe.load_db(dbpath, mode="r", user=args.db_postgres_user)

        tasktypes = pipe.tasks.base.default_task_chain

        header_state = ""
        for s in pipe.task_states:
            header_state = "{} {:8s}|".format(header_state, s)
        header_state = "{} {:8s}|".format(header_state, "submit")

        sep = "----------------+---------+---------+---------+---------+---------+---------+"

        header = "{}\n{:16s}|{}\n{}".format(sep, "   Task Type",
            header_state, sep)

        def print_status(clear=False):
            taskstates = {}
            tasksub = {}
            with db.cursor() as cur:
                for t in tasktypes:
                    taskstates[t] = {}
                    cmd = "select state from {}".format(t)
                    cur.execute(cmd)
                    st = np.array([ int(x[0]) for x in cur.fetchall() ])
                    for s in pipe.task_states:
                        taskstates[t][s] = \
                            np.sum(st == pipe.task_state_to_int[s])
                    if (t != "spectra") and (t != "redshift"):
                        cmd = "select submitted from {}".format(t)
                        cur.execute(cmd)
                        isub = [ int(x[0]) for x in cur.fetchall() ]
                        tasksub[t] = np.sum(isub).astype(int)
            if clear:
                print("\033c")
            print(header)
            for t in tasktypes:
                line = "{:16s}|".format(t)
                for s in pipe.task_states:
                    line = "{}{:9d}|".format(line, taskstates[t][s])
                if t in tasksub:
                    line = "{}{:9d}|".format(line, tasksub[t])
                else:
                    line = "{}{:9s}|".format(line, "       NA")
                print(line)
            print(sep)
            if clear:
                print(" (Use Ctrl-C to Quit) ")
            sys.stdout.flush()

        if args.once:
            print_status(clear=False)
        else:
            while True:
                print_status(clear=True)
                time.sleep(args.refresh)

        return


def main():
    p = PipeUI()
