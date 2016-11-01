#!/usr/bin/env python
#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-


from __future__ import absolute_import, division, print_function

import sys
import os
import stat
import numpy as np
import argparse
import re

from .. import io
from ..parallel import dist_discrete

from .. import pipeline as pipe

    
def compute_step(setupfile, envcom, first, last, specs, night, 
    ntask, taskproc, shell_mpi_run, shell_maxcores, shell_threads, nersc_maxnodes,
    nersc_nodecores, nersc_threads, nersc_mp, nersc_queue_thresh,
    queue="debug", minutes=30,
    ):
    specstr = ""
    if specs is not None:
        specstr = " --spectrographs {}".format(",".join([ "{}".format(x) for x in specs ]))

    rundir = io.get_pipe_rundir()
    scrdir = os.path.join(rundir, io.get_pipe_scriptdir())
    logdir = os.path.join(rundir, io.get_pipe_logdir())

    if (nersc_threads > 1) and (nersc_mp > 1):
        raise RuntimeError("set either nersc_threads or nersc_mp, but not both")

    nstr = ""
    scrstr = "all"
    if night is not None:
        nstr = " --nights {}".format(night)
        scrstr = "{}".format(night)

    if first == last:
        stepstr = "{}_{}".format(first, scrstr)
        jobname = first
    else:
        stepstr = "{}-{}_{}".format(first, last, scrstr)
        jobname = "{}_{}".format(first, last)

    com = ["desi_pipe_run --first {} --last {}{}{}".format(first, last, specstr, nstr)]

    totproc = ntask * taskproc

    shell_maxprocs = int(shell_maxcores / shell_threads)
    shell_procs = shell_maxprocs
    if totproc < shell_procs:
        shell_procs = totproc

    shell_path = os.path.join(scrdir, "{}.sh".format(stepstr))
    shell_log = os.path.join(logdir, "{}_sh".format(stepstr))

    #- no MPI for shell job version so that it can be run from interactive node
    if shell_maxcores == 1:
        com = ["desi_pipe_run --nompi --first {} --last {}{}{}".format(first, last, specstr, nstr)]
    else:
        com = ["desi_pipe_run --first {} --last {}{}{}".format(first, last, specstr, nstr)]
    pipe.shell_job(shell_path, shell_log, envcom, setupfile, com, comrun=shell_mpi_run, mpiprocs=shell_procs, threads=shell_threads)

    #- MPI for standard batch job (to be written below)
    com = ["desi_pipe_run --first {} --last {}{}{}".format(first, last, specstr, nstr)]

    core_per_proc = 1
    if nersc_threads > 1:
        core_per_proc = nersc_threads
    elif nersc_mp > 1:
        core_per_proc = nersc_mp

    nodeproc = int(nersc_nodecores / core_per_proc)
    nodes = int(totproc / nodeproc)
    if nodes * nodeproc != totproc:
        nodes += 1
    if nodes > nersc_maxnodes:
        nodes = nersc_maxnodes

    if nodes > nersc_queue_thresh and queue == "debug":
        print("{} nodes too big for debug queue; switching to regular".format(nodes))
        queue = "regular"

    nersc_path = os.path.join(scrdir, "{}.slurm".format(stepstr))
    nersc_log = os.path.join(logdir, "{}_slurm".format(stepstr))

    pipe.nersc_job(nersc_path, nersc_log, envcom, setupfile, com, nodes=nodes,
        nodeproc=nodeproc, minutes=minutes, multisrun=False, openmp=(nersc_threads > 1),
        multiproc=(nersc_mp > 1), queue=queue, jobname=jobname)

    return (shell_path, nersc_path)


def parse(options=None):
    parser = argparse.ArgumentParser(description="Set up pipeline runs for a production.")
    parser.add_argument("--raw", required=False, default=None, help="raw data directory")
    parser.add_argument("--redux", required=False, default=None, help="output directory")
    parser.add_argument("--prod", required=False, default=None, help="output production name")
    parser.add_argument("--nights", required=False, default=None, help="comma separated (YYYYMMDD) or regex pattern")
    
    parser.add_argument("--env", required=False, default=None, help="text file with environment setup commands")
    
    parser.add_argument("--nersc_host", required=False, default="edison", help="NERSC slurm scripts host name (edison|cori)")

    parser.add_argument("--nersc_max_nodes", required=False, default=None, help="NERSC slurm scripts max nodes to use.  Default is size of debug queue max.")

    parser.add_argument("--shell_mpi_run", required=False, default="mpirun -np", help="bash scripts command to launch MPI pipeline steps.  If --shell_max_cores is 1, this is ignored.")
    parser.add_argument("--shell_max_cores", required=False, default=1, help="bash scripts max cores to use.")

    parser.add_argument("--fakeboot", required=False, default=False, action="store_true", help="bypass bootcalib")

    parser.add_argument("--fakepsf", required=False, default=False, action="store_true", help="bypass specex")

    parser.add_argument("--spectrographs", required=False, default=None, help="process only this comma-separated list of spectrographs")

    parser.add_argument("--debug", required=False, default=False, action="store_true", help="in setup script, set log level to DEBUG")

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args):

    # Check raw data location

    rawdir = args.raw
    if rawdir is None:
        if "DESI_SPECTRO_DATA" in os.environ:
            rawdir = io.rawdata_root()
        else:
            raise RuntimeError("You must set DESI_SPECTRO_DATA in your environment or use the --raw commandline option")
    else:
        rawdir = os.path.abspath(rawdir)
        os.environ["DESI_SPECTRO_DATA"] = rawdir

    # Check production name

    prodname = args.prod
    if prodname is None:
        if "SPECPROD" in os.environ:
            prodname = os.environ["SPECPROD"]
        else:
            raise RuntimeError("You must set SPECPROD in your environment or use the --prod commandline option")
    else:
        os.environ["SPECPROD"] = prodname

    # Check spectro redux location

    proddir = None

    specdir = args.redux
    if specdir is None:
        if "DESI_SPECTRO_REDUX" in os.environ:
            specdir = os.environ["DESI_SPECTRO_REDUX"]
            proddir = io.specprod_root()
        else:
            raise RuntimeError("You must set DESI_SPECTRO_REDUX in your environment or use the --redux commandline option")
    else:
        specdir = os.path.abspath(specdir)
        proddir = os.path.join(specdir, prodname)
        os.environ["DESI_SPECTRO_REDUX"] = specdir

    # Check the machine limits we are using for this production

    nodecores = 0
    maxnodes = 0
    if args.nersc_host == "edison":
        nodecores = 24
        if args.nersc_max_nodes is not None:
            maxnodes = int(args.nersc_max_nodes)
        else:
            maxnodes = 512
    elif args.nerschost == "cori":
        nodecores = 32
        if args.nersc_max_nodes is not None:
            maxnodes = int(args.nersc_max_nodes)
        else:
            maxnodes = 64
    else:
        raise RuntimeError("unknown nersc host")

    shell_maxcores = int(args.shell_max_cores)
    shell_mpi_run = ""
    if shell_maxcores > 1:
        shell_mpi_run = "{}".format(args.shell_mpi_run)

    # Update output directories and plans

    print("Working with production {} :".format(proddir))

    print("  Updating plans ...")
    expnightcount, allbricks = pipe.create_prod(nightstr=args.nights)
    totcount = {}
    totcount["flat"] = 0
    totcount["arc"] = 0
    totcount["science"] = 0
    for k, v in expnightcount.items():
        totcount["flat"] += v["flat"]
        totcount["arc"] += v["arc"]
        totcount["science"] += v["science"]

    # create setup shell snippet

    print("  Creating setup.sh ...")
    setupfile = os.path.abspath(os.path.join(proddir, "setup.sh"))
    with open(setupfile, "w") as s:
        s.write("# Generated by desi_pipe\n")
        s.write("export DESI_SPECTRO_DATA={}\n".format(rawdir))
        s.write("export DESI_SPECTRO_REDUX={}\n".format(specdir))
        s.write("export SPECPROD={}\n".format(prodname))
        s.write("\n")
        if args.debug:
            s.write("export DESI_LOGLEVEL=\"DEBUG\"\n\n")
        else:
            s.write("#export DESI_LOGLEVEL=\"DEBUG\"\n\n")

    # read in the environment setup, if needed

    envcom = []
    if args.env is not None:
        print("  Reading environment initialization from {} ...".format(args.env))
        with open(args.env, "r") as f:
            for line in f:
                envcom.append(line.rstrip())
    else:
        print("  No environment initialization specified")

    # which nights and spectrographs are we using?

    print("  Selecting nights ...")

    specs = [ x for x in range(10) ]
    if args.spectrographs is not None:
        specs = [ int(x) for x in args.spectrographs.split(",") ]
    nspect = len(specs)

    allnights = []
    nightpat = re.compile(r"\d{8}")
    for root, dirs, files in os.walk(rawdir, topdown=True):
        for d in dirs:
            nightmat = nightpat.match(d)
            if nightmat is not None:
                allnights.append(d)
        break

    nights = pipe.select_nights(allnights, args.nights)

    # Get the workers from the main options file.  If we are just
    # updating an existing production, the user may have changed
    # the options to select a new worker for some tasks.  So we
    # always check this before generating scripts which need info
    # about the max processes supported by each worker.

    print("  Finding max processes supported by workers ...")

    rundir = io.get_pipe_rundir()
    optfile = os.path.join(rundir, "options.yaml")
    opts = pipe.yaml_read(optfile)

    workermax = {}
    for step in pipe.step_types:
        worker = pipe.get_worker(step, opts["{}_worker".format(step)], opts["{}_worker_opts".format(step)])
        workermax[step] = worker.max_nproc()
        print("    {} : {} processes per task".format(step, workermax[step]))

    # create scripts for processing

    print("  Generating scripts ...")

    all_slurm = []
    all_shell = []
    nt_slurm = {}
    nt_shell = {}
    for nt in nights:
        nt_slurm[nt] = []
        nt_shell[nt] = []

    # bootcalib

    if not args.fakeboot:

        ntask = len(nights) * 3 * nspect
        multip = 2

        scr_shell, scr_slurm = compute_step(setupfile, envcom, "bootstrap", "bootstrap", specs, None, ntask, workermax["bootstrap"], shell_mpi_run, shell_maxcores, 1, maxnodes, nodecores, 1, multip, maxnodes)
        all_shell.append(scr_shell)
        all_slurm.append(scr_slurm)

        for nt in nights:

            ntask = 3 * nspect

            scr_shell, scr_slurm = compute_step(setupfile, envcom, "bootstrap", "bootstrap", specs, nt, ntask, workermax["bootstrap"], shell_mpi_run, shell_maxcores, 1, maxnodes, nodecores, 1, multip, maxnodes)
            nt_shell[nt].append(scr_shell)
            nt_slurm[nt].append(scr_slurm)

    else:

        cal2d = os.path.join(proddir, "calib2d")
        calpsf = os.path.join(cal2d, "psf")

        for nt in nights:

            calpsfnight = os.path.join(calpsf, nt)
            if not os.path.isdir(calpsfnight):
                os.makedirs(calpsfnight)

            for band in ["b", "r", "z"]:
                for spec in range(10):
                    cam = "{}{}".format(band, spec)
                    target = os.path.join(os.environ["DESIMODEL"], "data", "specpsf", "psf-{}.fits".format(band))
                    lnk = os.path.join(calpsfnight, "psfboot-{}{}.fits".format(band, spec))
                    if not os.path.islink(lnk):
                        os.symlink(target, lnk)

    # specex

    if not args.fakepsf:

        ntask = totcount["arc"] * 3 * nspect
        threads = 2

        scr_shell, scr_slurm = compute_step(setupfile, envcom, "psf", "psf", specs, None, ntask, workermax["psf"], shell_mpi_run, shell_maxcores, 1, maxnodes, nodecores, threads, 1, maxnodes)
        all_shell.append(scr_shell)
        all_slurm.append(scr_slurm)

        for nt in nights:

            ntask = expnightcount[nt]["arc"] * 3 * nspect

            scr_shell, scr_slurm = compute_step(setupfile, envcom, "psf", "psf", specs, nt, ntask, workermax["psf"], shell_mpi_run, shell_maxcores, 1, maxnodes, nodecores, threads, 1, maxnodes)
            nt_shell[nt].append(scr_shell)
            nt_slurm[nt].append(scr_slurm)

        # psfcombine

        ntask = len(nights) * 3 * nspect

        scr_shell, scr_slurm = compute_step(setupfile, envcom, "psfcombine", "psfcombine", specs, None, ntask, workermax["psfcombine"], shell_mpi_run, shell_maxcores, 1, maxnodes, nodecores, 1, 1, maxnodes)
        all_shell.append(scr_shell)
        all_slurm.append(scr_slurm)

        for nt in nights:

            ntask = 3 * nspect

            scr_shell, scr_slurm = compute_step(setupfile, envcom, "psfcombine", "psfcombine", specs, nt, ntask, workermax["psfcombine"], shell_mpi_run, shell_maxcores, 1, maxnodes, nodecores, 1, 1, maxnodes)
            nt_shell[nt].append(scr_shell)
            nt_slurm[nt].append(scr_slurm)

    else:

        cal2d = os.path.join(proddir, "calib2d")
        calpsf = os.path.join(cal2d, "psf")

        for nt in nights:

            calpsfnight = os.path.join(calpsf, nt)
            if not os.path.isdir(calpsfnight):
                os.makedirs(calpsfnight)

            for band in ["b", "r", "z"]:
                for spec in range(10):
                    cam = "{}{}".format(band, spec)
                    target = os.path.join(calpsfnight, "psfboot-{}{}.fits".format(band, spec))
                    lnk = os.path.join(calpsfnight, "psfnight-{}{}.fits".format(band, spec))
                    if not os.path.islink(lnk):
                        os.symlink(target, lnk)

    # extract

    ntask = (totcount["flat"] + totcount["science"]) * 3 * nspect
    threads = 1

    scr_shell, scr_slurm = compute_step(setupfile, envcom, "extract", "extract", specs, None, ntask, workermax["extract"], shell_mpi_run, shell_maxcores, 1, maxnodes, nodecores, threads, 1, maxnodes)
    all_shell.append(scr_shell)
    all_slurm.append(scr_slurm)

    for nt in nights:

        ntask = (expnightcount[nt]["flat"] + expnightcount[nt]["science"]) * 3 * nspect

        scr_shell, scr_slurm = compute_step(setupfile, envcom, "extract", "extract", specs, nt, ntask, workermax["extract"], shell_mpi_run, shell_maxcores, 1, maxnodes, nodecores, threads, 1, maxnodes)
        nt_shell[nt].append(scr_shell)
        nt_slurm[nt].append(scr_slurm)

    # calibration

    ntask = totcount["science"] * 3 * nspect
    multip = 1      #- turning off multiprocessing

    scr_shell, scr_slurm = compute_step(setupfile, envcom, "fiberflat", "calibrate", specs, None, ntask, 1, shell_mpi_run, shell_maxcores, 1, maxnodes, nodecores, 1, multip, maxnodes)
    all_shell.append(scr_shell)
    all_slurm.append(scr_slurm)

    for nt in nights:

        ntask = expnightcount[nt]["science"] * 3 * nspect

        scr_shell, scr_slurm = compute_step(setupfile, envcom, "fiberflat", "calibrate", specs, nt, ntask, 1, shell_mpi_run, shell_maxcores, 1, maxnodes, nodecores, 1, multip, maxnodes)
        nt_shell[nt].append(scr_shell)
        nt_slurm[nt].append(scr_slurm)

    # make bricks - serial only for now!

    rundir = io.get_pipe_rundir()
    scrdir = os.path.join(rundir, io.get_pipe_scriptdir())
    logdir = os.path.join(rundir, io.get_pipe_logdir())

    brickcom = []
    for nt in nights:
        brickcom.append("desi_make_bricks --night {}".format(nt))

    shell_path = os.path.join(scrdir, "bricks.sh")
    shell_log = os.path.join(logdir, "bricks_sh.log")
    pipe.shell_job(shell_path, shell_log, envcom, setupfile, brickcom, comrun=shell_mpi_run, mpiprocs=1, threads=1)

    nersc_path = os.path.join(scrdir, "bricks.slurm")
    nersc_log = os.path.join(logdir, "bricks_slurm.log")
    pipe.nersc_job(nersc_path, nersc_log, envcom, setupfile, brickcom, nodes=1,
        nodeproc=1, minutes=30, multisrun=False, openmp=False, multiproc=False,
        queue="debug", jobname="bricks")

    all_shell.append(shell_path)
    all_slurm.append(nersc_path)

    # redshift fitting

    # We guess at the job size to use.  In order for the load balancing
    # to be effective, we use a smaller number of workers and a larger
    # process group size.

    ntask = len(allbricks.keys())
    efftask = int( ntask / 4 )

    # redmonster can do about 10 targets in 30 minutes.  We
    # add an extra fudge factor since the load balancing above
    # means that most workers will have at least 2 tasks, even
    # if one is very large.
    redtime = 30
    redqueue = "debug"
    largest = np.max(np.asarray([ allbricks[x] for x in allbricks.keys() ]))
    increments = int(2.0 * (float(largest) / 10.0) / float(workermax["redshift"])) + 1
    if increments > 1:
        redqueue = "regular"
        redtime = 30 * increments

    scr_shell, scr_slurm = compute_step(setupfile, envcom, "redshift", "redshift", specs,
        None, efftask, workermax["redshift"], shell_mpi_run, shell_maxcores,
        1, maxnodes, nodecores, 1, 1, maxnodes, queue=redqueue, minutes=redtime)
    all_shell.append(scr_shell)
    all_slurm.append(scr_slurm)

    # Make high-level shell scripts which run or submit the steps

    run_slurm_all = os.path.join(scrdir, "run_slurm_all.sh")
    with open(run_slurm_all, "w") as f:
        f.write("#!/bin/bash\n\n")
        first = True
        for scr in all_slurm:
            if first:
                f.write("jobid=`sbatch {} | awk '{{print $4}}'`\n\n".format(scr))
                first = False
            else:
                f.write("jobid=`sbatch -d afterok:${{jobid}} {} | awk '{{print $4}}'`\n\n".format(scr))

    run_shell_all = os.path.join(scrdir, "run_shell_all.sh")
    with open(run_shell_all, "w") as f:
        f.write("#!/bin/bash\n\n")
        for scr in all_slurm:
            f.write("bash {}\n\n".format(scr))

    mode = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    os.chmod(run_slurm_all, mode)
    os.chmod(run_shell_all, mode)

    for nt in nights:
        run_slurm_nt = os.path.join(scrdir, "run_slurm_{}.sh".format(nt))
        with open(run_slurm_nt, "w") as f:
            f.write("#!/bin/bash\n\n")
            first = True
            for scr in nt_slurm[nt]:
                if first:
                    f.write("jobid=`sbatch {} | awk '{{print $4}}'`\n\n".format(scr))
                    first = False
                else:
                    f.write("jobid=`sbatch -d afterok:${{jobid}} {} | awk '{{print $4}}'`\n\n".format(scr))

        run_shell_nt = os.path.join(scrdir, "run_shell_{}.sh".format(nt))
        with open(run_shell_nt, "w") as f:
            f.write("#!/bin/bash\n\n")
            for scr in nt_shell[nt]:
                f.write("bash {}\n\n".format(scr))

        os.chmod(run_slurm_nt, mode)
        os.chmod(run_shell_nt, mode)

