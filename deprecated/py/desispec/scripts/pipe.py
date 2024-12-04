#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.scripts.pipe
=====================

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

from desiutil.log import get_logger

from .. import pipeline as pipe

from ..pipeline import control as control


class PipeUI(object):

    def __init__(self):
        self.pref = "DESI"

        parser = argparse.ArgumentParser(
            description="DESI pipeline control",
            usage="""desi_pipe <command> [options]

Where supported commands are (use desi_pipe <command> --help for details):
   (------- High-Level -------)
   create   Create a new production.
   go       Run a full production.
   update   Update an existing production.
   top      Live display of production database.
   status   Overview of production.
   (------- Mid-Level --------)
   chain    Run all ready tasks for multiple pipeline steps.
   cleanup  Reset "running" (or optionally "failed") tasks back to "ready".
   (------- Low-Level --------)
   tasks    Get all possible tasks for a given type and states.
   check    Check the status of tasks.
   dryrun   Return the equivalent command line entrypoint for tasks.
   script   Generate a shell or slurm script.
   run      Generate a script and run it.
   getready Auto-Update of prod DB.
   sync     Synchronize DB state based on the filesystem.
   env      Print current production location.
   query    Direct sql query to the database.

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
        print("{}{:<22} = {}{}{}".format(
            self.pref, "Raw data directory", control.clr.OKBLUE, rawdir,
            control.clr.ENDC)
        )
        print("{}{:<22} = {}{}{}".format(
            self.pref, "Production directory", control.clr.OKBLUE, proddir,
            control.clr.ENDC)
        )
        return

    def query(self):
        parser = argparse.ArgumentParser(\
            description="Query the DB",
                                         usage="desi_pipe query 'sql_command' [--rw] (use --help for details)")
        parser.add_argument('cmd', metavar='cmd', type=str,
                            help="SQL command in quotes, like 'select * from preproc'")
        parser.add_argument("--rw", action = "store_true",
                            help="read/write mode (use with care, experts only). Default is read only")
        args = parser.parse_args(sys.argv[2:])
        dbpath = io.get_pipe_database()
        if args.rw :
            mode="w"
        else :
            mode="r"
        db = pipe.load_db(dbpath, mode=mode)
        with db.cursor() as cur:
            cur.execute(args.cmd)
            st = cur.fetchall()
            for entry in st :
                line=""
                for prop in entry :
                    line += " {}".format(prop)
                print(line)

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
            help="value to use for DESI_SPECTRO_CALIB")

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

        control.create(
            root=args.root,
            data=args.data,
            redux=args.redux,
            prod=args.prod,
            force=args.force,
            basis=args.basis,
            calib=args.calib,
            db_sqlite=args.db_sqlite,
            db_sqlite_path=args.db_sqlite_path,
            db_postgres=args.db_postgres,
            db_postgres_host=args.db_postgres_host,
            db_postgres_port=args.db_postgres_port,
            db_postgres_name=args.db_postgres_name,
            db_postgres_user=args.db_postgres_user,
            db_postgres_authorized=args.db_postgres_authorized,
            nside=args.nside)

        return


    def update(self):
        parser = argparse.ArgumentParser(description="Update a production",
            usage="desi_pipe update [options] (use --help for details)")

        parser.add_argument("--nights", required=False, default=None,
            help="comma separated (YYYYMMDD) or regex pattern- only nights "
            "matching these patterns will be examined.")

        parser.add_argument("--nside", required=False, type=int, default=64,
            help="HEALPix nside value to use for spectral grouping.")

        parser.add_argument("--expid", required=False, type=int, default=-1,
            help="Only update the production for a single exposure ID.")

        args = parser.parse_args(sys.argv[2:])

        expid = None
        if args.expid >= 0:
            expid = args.expid

        control.update(nightstr=args.nights, nside=args.nside,
            expid=expid)

        return


    def tasks(self):
        availtypes = ",".join(pipe.tasks.base.default_task_chain)

        parser = argparse.ArgumentParser(description="Get all tasks of a "
            "particular type for one or more nights",
            usage="desi_pipe tasks [options] (use --help for details)")

        parser.add_argument("--tasktypes", required=False, default=availtypes,
            help="comma separated list of task types ({})".format(availtypes))

        parser.add_argument("--nights", required=False, default=None,
            help="comma separated (YYYYMMDD) or regex pattern- only nights "
            "matching these patterns will be examined.")

        parser.add_argument("--expid", required=False, type=int, default=-1,
            help="Only select tasks for a single exposure ID.")

        parser.add_argument("--spec", required=False, type=int, default=-1,
            help="Only select tasks for a single spectrograph.")

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

        expid = None
        if args.expid >= 0:
            expid = args.expid

        spec = None
        if args.spec >= 0:
            spec = args.spec

        states = None
        if args.states is not None:
            states = args.states.split(",")

        ttypes = None
        if args.tasktypes is not None:
            ttypes = args.tasktypes.split(",")

        control.tasks(
            ttypes,
            nightstr=args.nights,
            states=states,
            expid=expid,
            spec=spec,
            nosubmitted=args.nosubmitted,
            db_postgres_user=args.db_postgres_user,
            taskfile=args.taskfile)

        return


    def getready(self):
        parser = argparse.ArgumentParser(description="Update database to "
            "for one or more nights to ensure that forward dependencies "
            "know that they are ready.",
            usage="desi_pipe getready [options] (use --help for details)")

        parser.add_argument("--nights", required=False, default=None,
            help="comma separated (YYYYMMDD) or regex pattern- only nights "
            "matching these patterns will be examined.")

        args = parser.parse_args(sys.argv[2:])

        dbpath = io.get_pipe_database()
        db = pipe.load_db(dbpath, mode="w")

        control.getready(db, nightstr=args.nights)

        return


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

        states = control.check_tasks(tasks, db=db)

        for tsk in tasks:
            print("{} : {}".format(tsk, states[tsk]))
        sys.stdout.flush()

        return


    def sync(self):
        availtypes = ",".join(pipe.tasks.base.default_task_chain)

        parser = argparse.ArgumentParser(\
            description="Synchronize DB state based on the filesystem.",
            usage="desi_pipe sync [options] (use --help for details)")

        parser.add_argument("--nights", required=False, default=None,
            help="comma separated (YYYYMMDD) or regex pattern- only nights "
            "matching these patterns will be examined.")
        parser.add_argument("--force-spec-done", action="store_true",
            help="force setting spectra file to state done if file exists independently of state of parent cframes.")

        args = parser.parse_args(sys.argv[2:])

        dbpath = io.get_pipe_database()
        db = pipe.load_db(dbpath, mode="w")

        control.sync(db, nightstr=args.nights,specdone=args.force_spec_done)

        return


    def cleanup(self):
        availtypes = ",".join(pipe.tasks.base.default_task_chain)

        parser = argparse.ArgumentParser(\
            description="Clean up stale task states in the DB",
            usage="desi_pipe cleanup [options] (use --help for details)")

        parser.add_argument("--failed", required=False, default=False,
            action="store_true", help="Also clear failed states")

        parser.add_argument("--submitted", required=False, default=False,
            action="store_true", help="Also clear submitted flag")

        parser.add_argument("--tasktypes", required=False, default=None,
            help="comma separated list of task types to clean ({})".format(availtypes))

        parser.add_argument("--expid", required=False, type=int, default=-1,
            help="Only clean tasks for this exposure ID.")

        args = parser.parse_args(sys.argv[2:])

        dbpath = io.get_pipe_database()
        db = pipe.load_db(dbpath, mode="w")

        ttypes = None
        if args.tasktypes is not None:
            ttypes = args.tasktypes.split(",")

        expid = None
        if args.expid >= 0:
            expid = args.expid

        control.cleanup(
            db,
            ttypes,
            failed=args.failed,
            submitted=args.submitted,
            expid=expid)

        return


    def _check_nersc_host(self, args):
        """Modify the --nersc argument based on the environment.
        """
        if args.shell:
            # We are forcibly generating shell scripts.
            args.nersc = None
        else:
            if args.nersc is None:
                if "NERSC_HOST" in os.environ:
                    if os.environ["NERSC_HOST"] == "cori":
                        args.nersc = "cori-haswell"
                    else:
                        args.nersc = os.environ["NERSC_HOST"]
        return


    def _parse_run_opts(self, parser):
        """Internal function to parse options for running.

        This provides a consistent set of run-time otpions for the
        "dryrun", "script", and "run" commands.

        """
        parser.add_argument("--nersc", required=False, default=None,
            help="write a script for this NERSC system (cori-haswell "
            "| cori-knl).  Default uses $NERSC_HOST")

        parser.add_argument("--shell", required=False, default=False,
            action="store_true",
            help="generate normal bash scripts, even if run on a NERSC system")

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
        availtypes = ",".join(pipe.tasks.base.default_task_chain)

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

        parser.add_argument("--force", required=False, default=False,
            action="store_true", help="print commands for all tasks, not"
            " only the ready ones")

        args = parser.parse_args(sys.argv[2:])

        self._check_nersc_host(args)

        tasks = pipe.prod.task_read(args.taskfile)

        control.dryrun(
            tasks,
            nersc=args.nersc,
            nersc_queue=args.nersc_queue,
            nersc_maxtime=args.nersc_maxtime,
            nersc_maxnodes=args.nersc_maxnodes,
            nersc_shifter=args.nersc_shifter,
            mpi_procs=args.mpi_procs,
            mpi_run=args.mpi_run,
            nodb=args.nodb,
            db_postgres_user=args.db_postgres_user,
            force=args.force)

        return


    def script(self):
        availtypes = ",".join(pipe.tasks.base.default_task_chain)

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

        self._check_nersc_host(args)

        scripts = control.script(
            args.taskfile,
            nersc=args.nersc,
            nersc_queue=args.nersc_queue,
            nersc_maxtime=args.nersc_maxtime,
            nersc_maxnodes=args.nersc_maxnodes,
            nersc_shifter=args.nersc_shifter,
            mpi_procs=args.mpi_procs,
            mpi_run=args.mpi_run,
            procs_per_node=args.procs_per_node,
            nodb=args.nodb,
            out=args.outdir,
            db_postgres_user=args.db_postgres_user)

        if len(scripts) > 0:
            print(",".join(scripts))

        return


    def run(self):
        availtypes = ",".join(pipe.tasks.base.default_task_chain)

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

        self._check_nersc_host(args)

        deps = None
        if args.depjobs is not None:
            deps = args.depjobs.split(",")

        jobids = control.run(
            args.taskfile,
            nosubmitted=args.nosubmitted,
            depjobs=deps,
            nersc=args.nersc,
            nersc_queue=args.nersc_queue,
            nersc_maxtime=args.nersc_maxtime,
            nersc_maxnodes=args.nersc_maxnodes,
            nersc_shifter=args.nersc_shifter,
            mpi_procs=args.mpi_procs,
            mpi_run=args.mpi_run,
            procs_per_node=args.procs_per_node,
            nodb=args.nodb,
            out=args.outdir,
            debug=args.debug)

        if len(jobids) > 0:
            print(",".join(jobids))

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

        parser.add_argument("--expid", required=False, type=int, default=-1,
            help="Only select tasks for a single exposure ID.")

        parser.add_argument("--spec", required=False, type=int, default=-1,
            help="Only select tasks for a single spectrograph.")

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

        parser.add_argument("--dryrun", action="store_true",
                            help="do not submit the jobs.")

        parser = self._parse_run_opts(parser)

        args = parser.parse_args(sys.argv[2:])

        self._check_nersc_host(args)

        expid = None
        if args.expid >= 0:
            expid = args.expid

        spec = None
        if args.spec >= 0:
            spec = args.spec

        states = None
        if args.states is not None:
            states = args.states.split(",")

        deps = None
        if args.depjobs is not None:
            deps = args.depjobs.split(",")

        jobids = control.chain(
            args.tasktypes.split(","),
            nightstr=args.nights,
            states=states,
            expid=expid,
            spec=spec,
            pack=args.pack,
            nosubmitted=args.nosubmitted,
            depjobs=deps,
            nersc=args.nersc,
            nersc_queue=args.nersc_queue,
            nersc_maxtime=args.nersc_maxtime,
            nersc_maxnodes=args.nersc_maxnodes,
            nersc_shifter=args.nersc_shifter,
            mpi_procs=args.mpi_procs,
            mpi_run=args.mpi_run,
            procs_per_node=args.procs_per_node,
            out=args.outdir,
            debug=args.debug,
            dryrun=args.dryrun)

        if jobids is not None and len(jobids) > 0:
            print(",".join(jobids))

        return


    def go(self):
        parser = argparse.ArgumentParser(description="Run a full production "
            "from start to finish.  This will pack steps into 3 jobs per night"
            " and then run redshift fitting after all nights are done.  Note "
            "that if you are running multiple nights you should use the "
            "regular queue.",
            usage="desi_pipe go [options] (use --help for details)")

        parser.add_argument("--nights", required=False, default=None,
            help="comma separated (YYYYMMDD) or regex pattern- only nights "
            "matching these patterns will be generated.")

        parser.add_argument("--states", required=False, default=None,
            help="comma separated list of states. This argument is "
            "passed to chain (see desi_pipe chain --help for more info).")
        parser.add_argument("--resume", action = 'store_true',
            help="same as --states waiting,ready")

        parser.add_argument("--dryrun", action="store_true",
            help="do not submit the jobs.")

        parser = self._parse_run_opts(parser)

        args = parser.parse_args(sys.argv[2:])

        if args.resume :
            if args.states is not None :
                print("Ambiguous arguments: cannot specify --states along with --resume option which would overwrite the list of states.")
                return
            else :
                args.states="waiting,ready"

        self._check_nersc_host(args)

        allnights = io.get_nights(strip_path=True)
        nights = pipe.prod.select_nights(allnights, args.nights)

        log = get_logger()

        blocks = [
            ["preproc", "psf", "psfnight"],
            ["traceshift", "extract"],
            ["fiberflat", "fiberflatnight", "sky", "starfit", "fluxcalib",
             "cframe"],
        ]

        nightlast = list()

        states = args.states
        if states is not None :
            states = states.split(",")

        for nt in nights:
            previous = None
            log.info("Submitting processing chains for night {}".format(nt))
            for blk in blocks:
                jobids = control.chain(
                    blk,
                    nightstr="{}".format(nt),
                    pack=True,
                    depjobs=previous,
                    nersc=args.nersc,
                    nersc_queue=args.nersc_queue,
                    nersc_maxtime=args.nersc_maxtime,
                    nersc_maxnodes=args.nersc_maxnodes,
                    nersc_shifter=args.nersc_shifter,
                    mpi_procs=args.mpi_procs,
                    mpi_run=args.mpi_run,
                    procs_per_node=args.procs_per_node,
                    out=args.outdir,
                    states=states,
                    debug=args.debug,
                    dryrun=args.dryrun)
                if jobids is not None and len(jobids)>0 :
                    previous = [ jobids[-1] ]
            if previous is not None and len(previous)>0 :
                nightlast.append(previous[-1])

        # Submit spectal grouping
        jobids = control.chain(
            ["spectra"],
            pack=True,
            depjobs=nightlast,
            nersc=args.nersc,
            nersc_queue=args.nersc_queue,
            nersc_maxtime=args.nersc_maxtime,
            nersc_maxnodes=args.nersc_maxnodes,
            nersc_shifter=args.nersc_shifter,
            mpi_procs=args.mpi_procs,
            mpi_run=args.mpi_run,
            procs_per_node=args.procs_per_node,
            out=args.outdir,
            states=states,
            debug=args.debug,
            dryrun=args.dryrun)

        previous = None
        if jobids is not None and len(jobids)>0 :
            previous = [ jobids[-1] ]

        # Submit redshifts (and coadds)
        jobids = control.chain(
            ["redshift"],
            pack=True,
            depjobs=previous,
            nersc=args.nersc,
            nersc_queue=args.nersc_queue,
            nersc_maxtime=args.nersc_maxtime,
            nersc_maxnodes=args.nersc_maxnodes,
            nersc_shifter=args.nersc_shifter,
            mpi_procs=args.mpi_procs,
            mpi_run=args.mpi_run,
            procs_per_node=args.procs_per_node,
            out=args.outdir,
            states=states,
            debug=args.debug,
            dryrun=args.dryrun)

        return


    def status(self):
        availtypes = ",".join(pipe.tasks.base.default_task_chain)

        parser = argparse.ArgumentParser(\
            description="Explore status of pipeline tasks",
            usage="desi_pipe status [options] (use --help for details)")

        parser.add_argument("--task", required=False, default=None,
            help="get log information about this specific task")

        parser.add_argument("--tasktypes", required=False, default=None,
            help="comma separated list of task types ({})".format(availtypes))

        parser.add_argument("--nights", required=False, default=None,
            help="comma separated (YYYYMMDD) or regex pattern- only nights "
            "matching these patterns will be examined.")

        parser.add_argument("--expid", required=False, type=int, default=None,
            help="Only select tasks for a single exposure ID.")

        parser.add_argument("--spec", required=False, type=int, default=None,
            help="Only select tasks for a single spectrograph.")

        parser.add_argument("--states", required=False, default=None,
            help="comma separated list of states (see defs.py).  Only tasks "
            "in these states will be returned.")

        parser.add_argument("--db-postgres-user", type=str, required=False,
            default="desidev_ro", help="If using postgres, connect as this "
            "user for read-only access")

        args = parser.parse_args(sys.argv[2:])

        ttypes = None
        if args.tasktypes is not None:
            ttypes = args.tasktypes.split(",")

        states = None
        if args.states is not None:
            states = args.states.split(",")

        control.status(
            task=args.task, tasktypes=ttypes, nightstr=args.nights,
            states=states, expid=args.expid, spec=args.spec,
            db_postgres_user=args.db_postgres_user
        )

        return


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
