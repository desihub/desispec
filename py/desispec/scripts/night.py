# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.scripts.night
======================

Automated nightly processing.
"""
from __future__ import (absolute_import, division, print_function,
    unicode_literals)

import sys
import os
import argparse
import subprocess as sp
import time

import fitsio

from desiutil.log import get_logger

from .. import io

from .. import pipeline as pipe

errs = {
    "usage" : 1,
    "pipefail" : 2,
    "io" : 3
}


class Nightly(object):

    def __init__(self):
        self.pref = "DESINIGHT"
        self.log = get_logger()

        parser = argparse.ArgumentParser(
            description="DESI nightly processing",
            usage="""desi_night <command> [options]

Where supported commands are:
  update    Process an incoming exposure as much as possible.  Arc exposures
            will trigger PSF estimation.  If the nightly PSF exists, then a
            flat exposure will be extracted and a fiberflat will be created.
            If the nightly PSF exists, then a science exposure will be
            extracted.  If the nightly fiberflat exists, then a science
            exposure will be calibrated.
  arcs      All arcs are done, proceed with nightly PSF.
  flats     All flats are done, proceed with nightly fiberflat.
  redshift  Regroup spectra and process all updated redshifts.
""")
        parser.add_argument("command", help="Subcommand to run")
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")
            parser.print_help()
            sys.exit(errs["usage"])

        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()


    def _update_db(self, night):
        try:
            sp.check_call("desi_pipe update --nights {}".format(night),
                          shell=True, universal_newlines=True)
        except:
            self.log.error("{}: desi_pipe update failed".format(self.pref))
            sys.exit(errs["pipefail"])
        return


    def _exposure_flavor(self, db, night, expid=None):
        # What exposures are we using?
        exp_by_flavor = dict()
        for flv in ["arc", "flat", "science"]:
            exp_by_flavor[fl] = list()

        if expid is not None:
            # Only use this one
            with db.cursor() as cur:
                cmd = "select expid, flavor from fibermap where night = {} and expid = {}".format(args.night, args.expid)
                cur.execute(cmd)
                for result in cur.fetchall():
                    exp_by_flavor[result[1]].append(result[0])
        else:
            # Get them all
            with db.cursor() as cur:
                cmd = "select expid, flavor from fibermap where night = {}"\
                    .format(args.night, args.expid)
                cur.execute(cmd)
                for result in cur.fetchall():
                    exp_by_flavor[result[1]].append(result[0])
        return exp_by_flavor


    def _select_exposures(self, db, night, table, expid=None):
        # Get the state and submitted status for all selected tasks.
        # Technically this is redundant since the jobs run by desi_pipe chain
        # will do the same checks- we are just avoiding calls that are not
        # needed.
        cmd = "select name, expid, state, submitted from {} where night = {}".format(table, night)
        if expid is not None:
            # Only use this one exposure
            cmd = "{} and expid = {}".format(cmd, expid)

        exps = set()
        with db.cursor() as cur:
            cur.execute(cmd)
            for result in cur.fetchall():
                if (pipe.task_int_to_state(result[2]) != "done") and \
                    (result[3] != 1):
                    self.log.info("{}: found unprocessed {} exposure {}".format(self.pref, table, result[1]))
                    exps.update(result[1])
        return exps


    def _write_jobid(self, root, night, expid, jobid):
        outdir = os.path.join(pipe.io.specprod_root(),
                              pipe.io.get_pipe_rundir(),
                              pipe.io.get_pipe_scriptdir(),
                              pipe.io.get_pipe_nightdir(),
                              night)
        outfile = os.path.join(outdir, "{}_{:08d}_{}".format(root, expid, jobid))
        with open(outfile, "w") as f:
            f.write(time.ctime())
            f.write("\n")
        return


    def _read_jobids(self, root, night):
        outdir = os.path.join(pipe.io.specprod_root(),
                              pipe.io.get_pipe_rundir(),
                              pipe.io.get_pipe_scriptdir(),
                              pipe.io.get_pipe_nightdir(),
                              night)
        pat = re.compile(r"{}_(\d{{8}})_(.*)".format(root))
        jobids = list()
        for root, dirs, files in os.walk(outdir, topdown=True):
            for f in files:
                mat = pat.match(f)
                if mat is not None:
                    jobids.append(mat.group(2))
            break
        return jobids


    def _chain_args(self, args):
        optlist = [
            "nersc",
            "nersc_queue",
            "nersc_maxtime",
            "nersc_maxnodes",
            "nersc_shifter",
            "mpi_procs",
            "mpi_run",
            "procs_per_node",
            "debug"
        ]
        varg = vars(args)
        opts = ""
        for k, v in varg.items():
            if k in optlist:
                if isinstance(v, bool):
                    opts = "{} --{}".format(opts, k)
                else:
                    opts = "{} --{} {}".format(opts, k, v)
        return opts


    def _run_psf(self, args, db, night, expid=None):
        exps = self._select_exposures(db, night, "psf", expid=expid)
        cargs = self._chain_args(args)
        jobids = list()
        for ex in exps:
            com = "desi_pipe chain --tasktypes preproc,psf --pack --nosubmitted --nights {} --outdir {} --expid {} {}".format(night, os.path.join("night", night), ex, cargs)
            try:
                self.log.info("{}:  running {}".format(self.pref, com))
                jid = sp.check_output(com, shell=True, universal_newlines=True)
                jobids.append(jid)
            except:
                self.log.error("{}: failure running {}".format(self.pref, com))
                sys.exit(errs["pipefail"])
        return jobids


    def _run_extract(self, args, exp_by_flavor, db, night, expid=None, deps=None):
        exps = self._select_exposures(db, night, "frame", expid=expid)
        cargs = self._chain_args(args)
        jobids = list()
        for ex in exps:
            com = None
            if ex in exp_by_flavor["flat"]:
                # Process through fiberflat
                com = "desi_pipe chain --tasktypes preproc,traceshift,extract,fiberflat --pack --nosubmitted --nights {} --outdir {} --expid {} {}".format(night, os.path.join("night", night), ex, cargs)
            else:
                # Just extract
                com = "desi_pipe chain --tasktypes preproc,traceshift,extract --pack --nosubmitted --nights {} --outdir {} --expid {} {}".format(night, os.path.join("night", night), ex, cargs)
            if deps is not None:
                com = "{} --depjobs {}".format(com, deps)
            try:
                self.log.info("{}:  running {}".format(self.pref, com))
                jid = sp.check_output(com, shell=True, universal_newlines=True)
                jobids.append(jid)
            except:
                self.log.error("{}: failure running {}".format(self.pref, com))
                sys.exit(errs["pipefail"])
        return jobids


    def _run_calib(self, args, db, night, expid=None, deps=None):
        exps = self._select_exposures(db, night, "cframe", expid=expid)
        cargs = self._chain_args(args)
        jobids = list()
        for ex in exps:
            com = "desi_pipe chain --tasktypes sky,starfit,fluxcalib,cframe --pack --nosubmitted --nights {} --outdir {} --expid {} {}".format(night, os.path.join("night", night), ex, cargs)
            if deps is not None:
                com = "{} --depjobs {}".format(com, deps)
            try:
                self.log.info("{}:  running {}".format(self.pref, com))
                jid = sp.check_output(com, shell=True, universal_newlines=True)
                jobids.append(jid)
            except:
                self.log.error("{}: failure running {}".format(self.pref, com))
                sys.exit(errs["pipefail"])
        return jobids


    def _check_nightly(self, ttype, db, night):
        ready = True
        deps = None
        tnight = "{}night".format(ttype)
        cmd = "select name, state, submitted from {} where night = {}".format(tnight, night)
        with db.cursor() as cur:
            cur.execute(cmd)
            for result in cur.fetchall():
                if pipe.task_int_to_state(result[1]) != "done":
                    ready = False
        if not ready:
            # Did we already submit a job?
            nids = self._read_jobids(tnight, night)
            if len(nids) > 0:
                ready = True
                deps = ",".join(nids)
        return read, deps


    def _pipe_opts(self, parser):
        """Internal function to parse options passed to desi_pipe.
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

        parser.add_argument("--debug", required=False, default=False,
            action="store_true", help="debugging messages in job logs")

        return parser


    def update(self):
        parser = argparse.ArgumentParser(description="Run processing on "
            "new data", usage="desi_night update [options] (use --help for "
            "details)")

        parser.add_argument("--night", required=True, default=None,
            help="The night in YYYYMMDD format.")

        parser.add_argument("--expid", required=False, type=int, default=-1,
            help="Only process this exposure.")

        parser = self._pipe_opts(parser)

        args = parser.parse_args(sys.argv[2:])

        # First update the DB
        self._update_db(args.night)

        # Now load the DB
        dbpath = io.get_pipe_database()
        db = pipe.load_db(dbpath, mode="w")

        # Get our exposures to consider and their flavors
        expid = None
        if args.expid >= 0:
            expid = args.expid
        expid_by_flavor = self._exposure_flavor(db, args.night, expid=expid)

        # If there are any arcs, we always process them
        for ex in expid_by_flavor["arc"]:
            jobids = self._run_psf(args, db, night, expid=ex)
            #FIXME: once we have a job table in the DB, the job ID will
            # be recorded automatically.  Until then, we record the PSF job
            # IDs in some file names so that the psfnight job can get the
            # dependencies correct.
            for jid in jobids.split(","):
                self._write_jobid("psf", night, ex, jid)

        # Check whether psfnight tasks are done or submitted
        ntpsfready, ntpsfdeps = self._check_nightly("psf", db, night)

        # Check whether fiberflatnight tasks are done or submitted
        ntflatready, ntflatdeps = self._check_nightly("fiberflat", db, night)

        if ntpsfready:
            # We can do extractions
            for ex in expid_by_flavor["flat"]:
                jobids = self._run_extract(args, exp_by_flavor, db, night,
                                           expid=ex, deps=ntpsfdeps)
                #FIXME: once we have a job table in the DB, the job ID will
                # be recorded automatically.  Until then, we record the
                # fiberflat job IDs in some file names so that the
                # fiberflatnight job can get the dependencies correct.
                for jid in jobids.split(","):
                    self._write_jobid("fiberflat", night, ex, jid)

            for ex in expid_by_flavor["science"]:
                exids = self._run_extract(args, exp_by_flavor, db, night,
                                          expid=ex, deps=ntpsfdeps)
                if ntflatready:
                    # We can submit calibration jobs too.
                    alldeps = "{},{}".format(ntflatdeps, ",".join(exids))
                    calids = self._run_calib(args, db, night, expid=ex, deps=alldeps)

        return


    def arcs(self):
        parser = argparse.ArgumentParser(description="Arcs are finished, "
            "create nightly PSF", usage="desi_night arcs [options] (use "
            "--help for details)")

        parser.add_argument("--night", required=True, default=None,
            help="The night in YYYYMMDD format.")

        parser = self._pipe_opts(parser)

        args = parser.parse_args(sys.argv[2:])

        # First update the DB
        self._update_db(args.night)

        # Now load the DB
        dbpath = io.get_pipe_database()
        db = pipe.load_db(dbpath, mode="w")

        # Check whether psfnight tasks are already done or submitted
        ntpsfready, ntpsfdeps = self._check_nightly("psf", db, night)

        if ntpsfready:
            if ntpsfdeps is None:
                self.log.info("{}: psfnight for {} already done".format(self.pref, night))
            else:
                self.log.info("{}: psfnight for {} already submitted to queue (job = {})".format(self.pref, night, ntpsfdeps))
        else:
            # Safe to run.  Get the job IDs of any previous psf tasks.
            psfjobs = self._read_jobids("psf", night)
            cargs = self._chain_args(args)
            com = "desi_pipe chain --tasktypes psfnight --pack --nosubmitted --nights {} --outdir {} {}".format(night, os.path.join("night", night), cargs)
            if len(psfjobs) > 0:
                com = "{} --depjobs {}".format(com, psfjobs)
            try:
                self.log.info("{}:  running {}".format(self.pref, com))
                jid = sp.check_output(com, shell=True, universal_newlines=True)
                self._write_jobid("psfnight", night, "NA", jid)
            except:
                self.log.error("{}: failure running {}".format(self.pref, com))
                sys.exit(errs["pipefail"])

        return


    def flats(self):
        parser = argparse.ArgumentParser(description="Flats are finished, "
            "create nightly fiberflat", usage="desi_night flats [options] (use "
            "--help for details)")

        parser.add_argument("--night", required=True, default=None,
            help="The night in YYYYMMDD format.")

        parser = self._pipe_opts(parser)

        args = parser.parse_args(sys.argv[2:])

        # First update the DB
        self._update_db(args.night)

        # Now load the DB
        dbpath = io.get_pipe_database()
        db = pipe.load_db(dbpath, mode="w")

        # Check whether psfnight tasks are already done or submitted
        ntflatready, ntflatdeps = self._check_nightly("fiberflat", db, night)

        if ntflatready:
            if ntflatdeps is None:
                self.log.info("{}: fiberflatnight for {} already done".format(self.pref, night))
            else:
                self.log.info("{}: fiberflatnight for {} already submitted to queue (job = {})".format(self.pref, night, ntpsfdeps))
        else:
            # Safe to run.  Get the job IDs of any previous fiberflat tasks.
            flatjobs = self._read_jobids("fiberflat", night)
            cargs = self._chain_args(args)
            com = "desi_pipe chain --tasktypes fiberflatnight --pack --nosubmitted --nights {} --outdir {} {}".format(night, os.path.join("night", night), cargs)
            if len(flatjobs) > 0:
                com = "{} --depjobs {}".format(com, flatjobs)
            try:
                self.log.info("{}:  running {}".format(self.pref, com))
                jid = sp.check_output(com, shell=True, universal_newlines=True)
                self._write_jobid("fiberflatnight", night, "NA", jid)
            except:
                self.log.error("{}: failure running {}".format(self.pref, com))
                sys.exit(errs["pipefail"])

        return


    def redshift(self):
        parser = argparse.ArgumentParser(description="Run spectra grouping and redshifts", usage="desi_night redshift [options] (use "
            "--help for details)")

        parser.add_argument("--night", required=True, default=None,
            help="The night in YYYYMMDD format.")

        parser = self._pipe_opts(parser)

        args = parser.parse_args(sys.argv[2:])

        # First update the DB
        self._update_db(args.night)

        # Run it
        cargs = self._chain_args(args)
        com = "desi_pipe chain --tasktypes spectra,redshift --pack --nosubmitted --nights {} --outdir {} {}".format(night, os.path.join("night", night), cargs)
        try:
            self.log.info("{}:  running {}".format(self.pref, com))
            sp.check_call(com, shell=True, universal_newlines=True)
        except:
            self.log.error("{}: failure running {}".format(self.pref, com))
            sys.exit(errs["pipefail"])

        return
