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
import re
import copy
import argparse
import subprocess as sp
import time
import warnings

import fitsio

from desiutil.log import get_logger

from .. import io

from .. import pipeline as pipe

from ..pipeline import control as control


errs = {
    "usage" : 1,
    "pipefail" : 2,
    "io" : 3
}


class Nightly(object):

    def __init__(self):
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
  redshifts Regroup spectra and process all updated redshifts.
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
        control.update(nightstr=night)
        return


    def _exposure_flavor(self, db, night, expid=None):
        # What exposures are we using?
        exp_by_flavor = dict()
        for flv in ["arc", "flat", "science"]:
            exp_by_flavor[flv] = list()

        if expid is not None:
            # Only use this one
            with db.cursor() as cur:
                cmd = "select expid, flavor from fibermap where night = {} and expid = {}".format(night, expid)
                cur.execute(cmd)
                for result in cur.fetchall():
                    exp_by_flavor[result[1]].append(result[0])
        else:
            # Get them all
            with db.cursor() as cur:
                cmd = "select expid, flavor from fibermap where night = {}"\
                    .format(night)
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
                if (pipe.task_int_to_state[result[2]] != "done") and \
                    (result[3] != 1):
                    if result[1] not in exps:
                        self.log.info("found unprocessed {} exposure {}".format(table, result[1]))
                        exps.add(result[1])
        return list(sorted(exps))


    def _write_jobid(self, root, night, expid, jobid):
        outdir = os.path.join(io.specprod_root(),
                              io.get_pipe_rundir(),
                              io.get_pipe_scriptdir(),
                              io.get_pipe_nightdir(),
                              night)
        fname = "{}_{:08d}_{}".format(root, expid, jobid)
        outfile = os.path.join(outdir, fname)
        with open(outfile, "w") as f:
            f.write(time.ctime())
            f.write("\n")
        return


    def _read_jobids(self, ttype, night):
        outdir = os.path.join(io.specprod_root(),
                              io.get_pipe_rundir(),
                              io.get_pipe_scriptdir(),
                              io.get_pipe_nightdir(),
                              night)
        pat = re.compile(r"{}_(\d{{8}})_(.*)".format(ttype))
        jobids = list()
        for root, dirs, files in os.walk(outdir, topdown=True):
            for f in files:
                mat = pat.match(f)
                if mat is not None:
                    jobids.append(mat.group(2))
            break
        return jobids


    def _small(self, args):
        small = copy.copy(args)
        if small.nersc_maxnodes_small is not None:
            small.nersc_maxnodes = small.nersc_maxnodes_small
        return small


    def _run_chain(self, args, exps, db, night, tasktypes, deps=None,
                   spec=None):
        log = get_logger()
        jobids = list()
        if exps is None:
            log.info("Running chain for night = {}, tasktypes = {}, "
                "deps = {}".format(night, tasktypes, deps))
            jobids = control.chain(
                tasktypes,
                nightstr=night,
                spec=spec,
                pack=True,
                nosubmitted=True,
                depjobs=deps,
                nersc=args.nersc,
                nersc_queue=args.nersc_queue,
                nersc_maxtime=args.nersc_maxtime,
                nersc_maxnodes=args.nersc_maxnodes,
                nersc_shifter=args.nersc_shifter,
                mpi_procs=args.mpi_procs,
                mpi_run=args.mpi_run,
                procs_per_node=args.procs_per_node,
                out=os.path.join("night", night),
                debug=False)
            log.debug('Job IDs {}'.format(jobids))
        else:
            for ex in exps:
                log.info("Running chain for night = {}, tasktypes = {}, "
                    "expid = {}, deps = {}".format(night, tasktypes, ex, deps))
                exjobids = control.chain(
                    tasktypes,
                    nightstr=night,
                    expid=ex,
                    spec=spec,
                    pack=True,
                    nosubmitted=True,
                    depjobs=deps,
                    nersc=args.nersc,
                    nersc_queue=args.nersc_queue,
                    nersc_maxtime=args.nersc_maxtime,
                    nersc_maxnodes=args.nersc_maxnodes,
                    nersc_shifter=args.nersc_shifter,
                    mpi_procs=args.mpi_procs,
                    mpi_run=args.mpi_run,
                    procs_per_node=args.procs_per_node,
                    out=os.path.join("night", night),
                    debug=False)
                log.debug('Job IDs {}'.format(exjobids))
                jobids.extend(exjobids)
        return jobids


    def _run_psf(self, args, db, night, expid=None, spec=None):
        exps = self._select_exposures(db, night, "psf", expid=expid)
        return self._run_chain(args, exps, db, night, ["preproc", "psf"],
            spec=spec)


    def _run_extract(self, args, exp_by_flavor, db, night, expid=None,
                     deps=None, spec=None):
        exps = self._select_exposures(db, night, "extract", expid=expid)
        jobids = list()
        for ex in exps:
            jids = None
            # Regardless of exposure type, preprocess and traceshift in a
            # single job.
            trids = self._run_chain(self._small(args), [ex], db, night,
                ["preproc", "traceshift"], deps=deps, spec=spec)
            # Now either extract or also do fiberflat.
            if ex in exp_by_flavor["flat"]:
                jids = self._run_chain(args, [ex], db, night,
                    ["extract", "fiberflat"], deps=trids, spec=spec)
            else:
                jids = self._run_chain(args, [ex], db, night,
                    ["extract"], deps=trids, spec=spec)
            jobids.extend(jids)
        return jobids


    def _run_calib(self, args, db, night, expid=None, deps=None, spec=None):
        exps = self._select_exposures(db, night, "cframe", expid=expid)
        # Swap in the modified args for "small" jobs
        return self._run_chain(self._small(args), exps, db, night,
            ["sky", "starfit", "fluxcalib", "cframe"], deps=deps, spec=spec)


    def _check_nightly(self, ttype, db, night):
        ready = True
        deps = None
        tnight = "{}night".format(ttype)
        cmd = "select name, state, submitted from {} where night = {}".format(tnight, night)
        with db.cursor() as cur:
            cur.execute(cmd)
            for result in cur.fetchall():
                if pipe.task_int_to_state[result[1]] != "done":
                    ready = False
        if not ready:
            # Did we already submit a job?
            nids = self._read_jobids(tnight, night)
            if len(nids) > 0:
                ready = True
                deps = nids
        return ready, deps


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


    def _pipe_opts(self, parser):
        """Internal function to parse options passed to desi_night.
        """
        parser.add_argument("--nersc", required=False, default=None,
            help="write a script for this NERSC system (cori-haswell "
            "| cori-knl).  Default uses $NERSC_HOST")

        parser.add_argument("--shell", required=False, default=False,
            action="store_true",
            help="generate normal bash scripts, even if run on a NERSC system")

        parser.add_argument("--nersc_queue", required=False, default="regular",
            help="write a script for this NERSC queue (debug | regular)")

        parser.add_argument("--nersc_queue_redshifts", required=False,
            default=None, help="Use this NERSC queue for redshifts. "
            "Defaults to same as --nersc_queue.")

        parser.add_argument("--nersc_maxtime", required=False, type=int,
            default=0, help="Then maximum run time (in minutes) for a single "
            " job.  If the list of tasks cannot be run in this time, multiple "
            " job scripts will be written.  Default is the maximum time for "
            " the specified queue.")

        parser.add_argument("--nersc_maxnodes", required=False, type=int,
            default=0, help="The maximum number of nodes to use.  Default "
            " is the maximum nodes for the specified queue.")

        parser.add_argument("--nersc_maxnodes_small", required=False, type=int,
            default=0, help="The maximum number of nodes to use for 'small' "
            "steps like the per-night psf and fiberflat.  Default is to use the"
            " same value as --nersc_maxnodes.")

        parser.add_argument("--nersc_maxnodes_redshifts", required=False,
            type=int, default=0, help="The maximum number of nodes to use for "
            " redshifts.  Default is to use --nersc_maxnodes.")

        parser.add_argument("--nersc_shifter", required=False, default=None,
            help="The shifter image to use for NERSC jobs")

        parser.add_argument("--mpi_procs", required=False, type=int, default=1,
            help="The number of MPI processes to use for non-NERSC shell "
            "scripts (default 1)")

        parser.add_argument("--mpi_run", required=False, type=str, default="",
            help="The command to launch MPI programs "
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

        parser.add_argument("--spec", required=False, type=int, default=-1,
            help="Only process a single spectrograph.  (FOR DEBUGGING ONLY)")

        parser = self._pipe_opts(parser)

        args = parser.parse_args(sys.argv[2:])

        self._check_nersc_host(args)

        # First update the DB
        self._update_db(args.night)

        # Now load the DB
        dbpath = io.get_pipe_database()
        db = pipe.load_db(dbpath, mode="w")

        # Get our exposures to consider and their flavors
        expid = None
        if args.expid >= 0:
            expid = args.expid

        spec = None
        if args.spec >= 0:
            spec = args.spec

        expid_by_flavor = self._exposure_flavor(db, args.night, expid=expid)

        # If there are any arcs, we always process them
        for ex in expid_by_flavor["arc"]:
            jobids = self._run_psf(args, db, args.night, expid=ex, spec=spec)
            #FIXME: once we have a job table in the DB, the job ID will
            # be recorded automatically.  Until then, we record the PSF job
            # IDs in some file names so that the psfnight job can get the
            # dependencies correct.
            for jid in jobids:
                self._write_jobid("psf", args.night, ex, jid)

        # Check whether psfnight tasks are done or submitted
        ntpsfready, ntpsfdeps = self._check_nightly("psf", db, args.night)

        # Check whether fiberflatnight tasks are done or submitted
        ntflatready, ntflatdeps = self._check_nightly("fiberflat", db, args.night)

        if ntpsfready:
            # We can do extractions
            for ex in expid_by_flavor["flat"]:
                jobids = self._run_extract(args, expid_by_flavor, db, args.night, expid=ex, deps=ntpsfdeps, spec=spec)
                #FIXME: once we have a job table in the DB, the job ID will
                # be recorded automatically.  Until then, we record the
                # fiberflat job IDs in some file names so that the
                # fiberflatnight job can get the dependencies correct.
                for jid in jobids:
                    self._write_jobid("fiberflat", args.night, ex, jid)

            for ex in expid_by_flavor["science"]:
                exids = self._run_extract(args, expid_by_flavor, db, args.night, expid=ex, deps=ntpsfdeps, spec=spec)
                if ntflatready:
                    # We can submit calibration jobs too.
                    alldeps = None
                    if len(exids) > 0:
                        alldeps = list(exids)
                    if ntflatdeps is not None:
                        if alldeps is None:
                            alldeps = list(ntflatdeps)
                        else:
                            alldeps.extend(ntflatdeps)
                    calids = self._run_calib(args, db, args.night, expid=ex,
                        deps=alldeps)
                    for cid in calids:
                        self._write_jobid("cframe", args.night, ex, cid)
                else:
                    allexp = [ "{}".format(x) for x in expid_by_flavor["science"] ]
                    msg = "Attempting to update processing of science exposures before the nightly fiberflat has been submitted.  Calibration has been skipped for the following exposures: {}  You should resubmit these exposures after running 'desi_night flats'".format(",".join(allexp))
                    warnings.warn(msg, RuntimeWarning)
        else:
            if (len(expid_by_flavor["flat"]) > 0) or (len(expid_by_flavor["science"]) > 0):
                allexp = [ "{}".format(x) for x in expid_by_flavor["science"] ]
                allexp.extend([ "{}".format(x) for x in expid_by_flavor["flat"] ])
                msg = "Attempting to update processing with flats and/or science exposures before the nightly PSF has been submitted.  The following exposures have been skipped:  {}  You should resubmit these exposures after running 'desi_night arcs'".format(",".join(allexp))
                warnings.warn(msg, RuntimeWarning)

        return


    def arcs(self):
        parser = argparse.ArgumentParser(description="Arcs are finished, "
            "create nightly PSF", usage="desi_night arcs [options] (use "
            "--help for details)")

        parser.add_argument("--night", required=True, default=None,
            help="The night in YYYYMMDD format.")

        parser.add_argument("--spec", required=False, type=int, default=-1,
            help="Only select tasks for a single spectrograph.  (FOR DEBUGGING ONLY)")

        parser = self._pipe_opts(parser)

        args = parser.parse_args(sys.argv[2:])

        self._check_nersc_host(args)

        spec = None
        if args.spec >= 0:
            spec = args.spec

        # First update the DB
        self._update_db(args.night)

        # Now load the DB
        dbpath = io.get_pipe_database()
        db = pipe.load_db(dbpath, mode="w")

        # Check whether psfnight tasks are already done or submitted
        ntpsfready, ntpsfdeps = self._check_nightly("psf", db, args.night)

        if ntpsfready:
            if ntpsfdeps is None:
                self.log.info("psfnight for {} already done".format(args.night))
            else:
                self.log.info("psfnight for {} already submitted to queue (job = {})".format(args.night, ntpsfdeps))
        else:
            # Safe to run.  Get the job IDs of any previous psf tasks.
            psfjobs = self._read_jobids("psf", args.night)
            deps = None
            if len(psfjobs) > 0:
                deps = psfjobs
            jid = self._run_chain(self._small(args), None, db, args.night,
                ["psfnight"], deps=deps, spec=spec)
            self._write_jobid("psfnight", args.night, 0, jid[0])
        return


    def flats(self):
        parser = argparse.ArgumentParser(description="Flats are finished, "
            "create nightly fiberflat", usage="desi_night flats [options] (use "
            "--help for details)")

        parser.add_argument("--night", required=True, default=None,
            help="The night in YYYYMMDD format.")

        parser.add_argument("--spec", required=False, type=int, default=-1,
            help="Only select tasks for a single spectrograph.  (FOR DEBUGGING ONLY)")

        parser = self._pipe_opts(parser)

        args = parser.parse_args(sys.argv[2:])

        self._check_nersc_host(args)

        spec = None
        if args.spec >= 0:
            spec = args.spec

        # First update the DB
        self._update_db(args.night)

        # Now load the DB
        dbpath = io.get_pipe_database()
        db = pipe.load_db(dbpath, mode="w")

        # Check whether psfnight tasks are already done or submitted
        ntflatready, ntflatdeps = self._check_nightly("fiberflat", db, args.night)

        if ntflatready:
            if ntflatdeps is None:
                self.log.info("fiberflatnight for {} already done".format(args.night))
            else:
                self.log.info("fiberflatnight for {} already submitted to queue (job = {})".format(args.night, ntflatdeps))
        else:
            # Safe to run.  Get the job IDs of any previous fiberflat tasks.
            flatjobs = self._read_jobids("fiberflat", args.night)
            deps = None
            if len(flatjobs) > 0:
                deps = flatjobs
            jid = self._run_chain(self._small(args), None, db, args.night,
                ["fiberflatnight"], deps=deps, spec=spec)
            self._write_jobid("fiberflatnight", args.night, 0, jid[0])
        return


    def redshifts(self):
        parser = argparse.ArgumentParser(description="Run spectra grouping and redshifts", usage="desi_night redshifts [options] (use "
            "--help for details)")

        parser.add_argument("--night", required=True, default=None,
            help="The night in YYYYMMDD format.")

        parser = self._pipe_opts(parser)

        args = parser.parse_args(sys.argv[2:])

        self._check_nersc_host(args)

        # First update the DB
        self._update_db(args.night)

        # Now load the DB
        dbpath = io.get_pipe_database()
        db = pipe.load_db(dbpath, mode="w")

        # Get the list of submitted cframe jobs.  Use these as our dependencies.
        cframejobs = self._read_jobids("cframe", args.night)

        # Run it
        redargs = copy.copy(args)
        if redargs.nersc_queue_redshifts is not None:
            redargs.nersc_queue = redargs.nersc_queue_redshifts
        if redargs.nersc_maxnodes_redshifts is not None:
            redargs.nersc_maxnodes = redargs.nersc_maxnodes_redshifts
        jid = self._run_chain(redargs, None, db, args.night, "spectra,redshift",
                              deps=cframejobs)

        return
