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


def step_props(first, last, specs, night):
    """
    Internal helper function used only in the desi_pipe
    script to build names of scripts and directories.
    """
    specstr = ""
    if specs is not None:
        specstr = " --spectrographs {}".format(",".join([ "{}".format(x) for x in specs ]))

    rundir = io.get_pipe_rundir()
    scrdir = os.path.join(rundir, io.get_pipe_scriptdir())
    logdir = os.path.join(rundir, io.get_pipe_logdir())

    nstr = ""
    scrstr = "all"
    if night is not None:
        nstr = " --nights {}".format(night)
        scrstr = "{}".format(night)

    stepstr = ""
    jobname = ""
    if first == last:
        stepstr = "{}_{}".format(first, scrstr)
        jobname = first
    else:
        stepstr = "{}-{}_{}".format(first, last, scrstr)
        jobname = "{}_{}".format(first, last)

    return (rundir, scrdir, logdir, specstr, nstr, scrstr, stepstr, jobname)

    
def compute_step_shell(setupfile, first, last, specs, night, 
    ntask, taskproc, shell_mpi_run, shell_maxcores, shell_threads
    ):
    """
    Internal helper function used only in the desi_pipe script to
    generate bash (non-NERSC) job scripts.
    """
    
    (rundir, scrdir, logdir, specstr, nstr, scrstr, stepstr, jobname) = step_props(first, last, specs, night)

    totproc = ntask * taskproc

    shell_maxprocs = int(shell_maxcores / shell_threads)
    shell_procs = shell_maxprocs
    if totproc < shell_procs:
        shell_procs = totproc

    shell_path = os.path.join(scrdir, "{}.sh".format(stepstr))
    shell_log = os.path.join(logdir, "{}_sh".format(stepstr))

    #- no MPI for shell job version so that it can be run from interactive node
    com = None
    if shell_maxcores == 1:
        com = ["desi_pipe_run --first {} --last {}{}{}".format(first, last, specstr, nstr)]
    else:
        com = ["desi_pipe_run_mpi --first {} --last {}{}{}".format(first, last, specstr, nstr)]
    pipe.shell_job(shell_path, shell_log, setupfile, com, comrun=shell_mpi_run, mpiprocs=shell_procs, threads=shell_threads)

    return shell_path


def compute_step_slurm(setupfile, first, last, specs, night, 
    ntask, taskproc, nersc_maxnodes,
    nersc_nodecores, nersc_threads, nersc_mp, nersc_queue_thresh,
    queue="debug", minutes=30
    ):
    """
    Internal helper function used only in the desi_pipe script to
    generate NERSC job scripts.
    """

    (rundir, scrdir, logdir, specstr, nstr, scrstr, stepstr, jobname) = step_props(first, last, specs, night)

    totproc = ntask * taskproc

    com = ["desi_pipe_run_mpi --first {} --last {}{}{}".format(first, last, specstr, nstr)]

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

    pipe.nersc_job(nersc_path, nersc_log, setupfile, com, nodes=nodes,
        nodeproc=nodeproc, minutes=minutes, multisrun=False, openmp=(nersc_threads > 1),
        multiproc=(nersc_mp > 1), queue=queue, jobname=jobname)

    return nersc_path


def compute_step_shifter(img, specdata, specredux, desiroot, setupfile, 
    first, last, specs, night, 
    ntask, taskproc, nersc_maxnodes,
    nersc_nodecores, nersc_threads, nersc_mp, nersc_queue_thresh,
    queue="debug", minutes=30
    ):
    """
    Internal helper function used only in the desi_pipe script to
    generate NERSC job scripts that use shifter.
    """

    (rundir, scrdir, logdir, specstr, nstr, scrstr, stepstr, jobname) = step_props(first, last, specs, night)

    totproc = ntask * taskproc

    com = ["desi_pipe_run_mpi --first {} --last {}{}{}".format(first, last, specstr, nstr)]

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

    nersc_path = os.path.join(scrdir, "{}_shifter.slurm".format(stepstr))
    nersc_log = os.path.join(logdir, "{}_shifter".format(stepstr))

    pipe.nersc_shifter_job(nersc_path, img, specdata, specredux, desiroot, nersc_log, setupfile, com, nodes=nodes,
        nodeproc=nodeproc, minutes=minutes, multisrun=False, openmp=(nersc_threads > 1),
        multiproc=(nersc_mp > 1), queue=queue, jobname=jobname)

    return nersc_path


def compute_step(img, specdata, specredux, desiroot, setupfile, 
    first, last, specs, night, ntask, taskproc, shell_mpi_run, shell_maxcores,
    shell_threads, nersc_maxnodes, nersc_nodecores, nersc_threads, nersc_mp, 
    nersc_queue_thresh, queue="debug", minutes=30
    ):
    """
    Internal helper function used only in the desi_pipe script to
    generate the job scripts.
    """

    scr_shell = compute_step_shell(setupfile, first, last, specs, night, 
                ntask, taskproc, shell_mpi_run, shell_maxcores, shell_threads)

    scr_slurm = compute_step_slurm(setupfile, first, last, specs, night, 
                ntask, taskproc, nersc_maxnodes, nersc_nodecores, nersc_threads, 
                nersc_mp, nersc_queue_thresh, queue=queue, minutes=minutes)

    scr_shifter = ""
    if img is not None:
        scr_shifter = compute_step_shifter(img, specdata, specredux, desiroot, setupfile, first, last, specs, night, ntask, taskproc, 
            nersc_maxnodes, nersc_nodecores, nersc_threads, nersc_mp, 
            nersc_queue_thresh, queue=queue, minutes=minutes)

    return (scr_shell, scr_slurm, scr_shifter)


def parse(options=None):
    parser = argparse.ArgumentParser(description="Set up pipeline runs for a production.")
    parser.add_argument("--data", required=False, default=None, help="override DESI_SPECTRO_DATA")
    parser.add_argument("--redux", required=False, default=None, help="override DESI_SPECTRO_REDUX")
    parser.add_argument("--prod", required=False, default=None, help="override SPECPROD")
    
    parser.add_argument("--starmodels", required=False, default=None, help="override the default star model file")

    parser.add_argument("--nights", required=False, default=None, help="comma separated (YYYYMMDD) or regex pattern")
    
    parser.add_argument("--nersc_host", required=False, default="edison", help="NERSC slurm scripts host name (edison|cori|coriknl)")

    parser.add_argument("--nersc_queue", required=False, default="regular", help="NERSC queue to use.")

    parser.add_argument("--nersc_max_nodes", required=False, default=None, help="NERSC slurm scripts max nodes to use.  Default is size of debug queue max.")

    parser.add_argument("--shell_mpi_run", required=False, default="mpirun -np", help="bash scripts command to launch MPI pipeline steps.  If --shell_max_cores is 1, this is ignored.")
    parser.add_argument("--shell_max_cores", required=False, default=1, help="bash scripts max cores to use.")

    parser.add_argument("--fakeboot", required=False, default=False, action="store_true", help="bypass bootcalib")

    parser.add_argument("--fakepsf", required=False, default=False, action="store_true", help="bypass specex")

    parser.add_argument("--fakepix", required=False, default=False, action="store_true", help="bypass checks for pixel data.  Useful when skipping extraction step in simulations.")

    parser.add_argument("--spectrographs", required=False, default=None, help="process only this comma-separated list of spectrographs")

    parser.add_argument("--debug", required=False, default=False, action="store_true", help="in setup script, set log level to DEBUG")

    parser.add_argument("--shifter", required=False, default=None, help="shifter image to use in alternate slurm scripts")

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args):

    # Check raw data location

    rawdir = args.data
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

    # Check DESIMODEL

    if "DESIMODEL" not in os.environ:
        raise RuntimeError("You must set DESIMODEL in your environment")

    # Check DESI_ROOT

    if "DESI_ROOT" not in os.environ:
        raise RuntimeError("You must set DESI_ROOT in your environment")
    desiroot = os.environ["DESI_ROOT"]

    # Add any extra options to the initial options.yaml file

    extra = {}
    if args.starmodels is not None:
        extra["Stdstars"] = {}
        extra["Stdstars"]["starmodels"] = args.starmodels

    # Check the machine limits we are using for this production

    nodecores = 0
    maxnodes = 0
    queuethresh = 0
    if args.nersc_host == "edison":
        nodecores = 24
        queuethresh = 512
        if args.nersc_max_nodes is not None:
            maxnodes = int(args.nersc_max_nodes)
        else:
            maxnodes = 2048
    elif args.nerschost == "cori":
        nodecores = 32
        queuethresh = 64
        if args.nersc_max_nodes is not None:
            maxnodes = int(args.nersc_max_nodes)
        else:
            maxnodes = 512
    elif args.nerschost == "coriknl":
        nodecores = 64
        queuethresh = 512
        if args.nersc_max_nodes is not None:
            maxnodes = int(args.nersc_max_nodes)
        else:
            maxnodes = 4096
    else:
        raise RuntimeError("unknown nersc host")

    shell_maxcores = int(args.shell_max_cores)
    shell_mpi_run = ""
    if shell_maxcores > 1:
        shell_mpi_run = "{}".format(args.shell_mpi_run)

    # Select our spectrographs

    specs = [ x for x in range(10) ]
    if args.spectrographs is not None:
        specs = [ int(x) for x in args.spectrographs.split(",") ]
    nspect = len(specs)

    # Update output directories and plans

    print("Working with production {} :".format(proddir))

    print("  Updating plans ...")
    expnightcount, allbricks = pipe.create_prod(nightstr=args.nights, 
        extra=extra, specs=specs, fakepix=args.fakepix)
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

    # which nights and are we using?

    print("  Selecting nights ...")

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
    all_shifter = []

    nt_slurm = {}
    nt_shell = {}
    nt_shifter = {}
    for nt in nights:
        nt_slurm[nt] = []
        nt_shell[nt] = []
        nt_shifter[nt] = []

    # bootcalib

    if not args.fakeboot:

        ntask = len(nights) * 3 * nspect
        taskproc = workermax["bootstrap"]
        step_threads = 1
        step_mp = 2
        nt = None
        first = "bootstrap"
        last = "bootstrap"

        scr_shell, scr_slurm, scr_shifter = compute_step(args.shifter, 
            rawdir, specdir, desiroot, setupfile, first, last, specs, nt, 
            ntask, taskproc, shell_mpi_run, shell_maxcores, 1, maxnodes,
            nodecores, step_threads, step_mp, queuethresh)
        all_shell.append(scr_shell)
        all_slurm.append(scr_slurm)
        all_shifter.append(scr_shifter)

        #scr_shell, scr_slurm = compute_step(setupfile, "bootstrap", "bootstrap", specs, None, ntask, workermax["bootstrap"], shell_mpi_run, shell_maxcores, 1, maxnodes, nodecores, 1, multip, queuethresh)
        
        for nt in nights:

            ntask = 3 * nspect

            scr_shell, scr_slurm, scr_shifter = compute_step(args.shifter, 
                rawdir, specdir, desiroot, setupfile, first, last, specs, nt, 
                ntask, taskproc, shell_mpi_run, shell_maxcores, 1, maxnodes,
                nodecores, step_threads, step_mp, queuethresh)
            nt_shell[nt].append(scr_shell)
            nt_slurm[nt].append(scr_slurm)
            nt_shifter[nt].append(scr_shifter)

            # scr_shell, scr_slurm = compute_step(setupfile, "bootstrap", "bootstrap", specs, nt, ntask, workermax["bootstrap"], shell_mpi_run, shell_maxcores, 1, maxnodes, nodecores, 1, multip, queuethresh)

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
        taskproc = workermax["psf"]
        step_threads = 2
        step_mp = 1
        nt = None
        first = "psf"
        last = "psf"

        scr_shell, scr_slurm, scr_shifter = compute_step(args.shifter, 
            rawdir, specdir, desiroot, setupfile, first, last, specs, nt, 
            ntask, taskproc, shell_mpi_run, shell_maxcores, 1, maxnodes,
            nodecores, step_threads, step_mp, queuethresh)
        all_shell.append(scr_shell)
        all_slurm.append(scr_slurm)
        all_shifter.append(scr_shifter)

        # scr_shell, scr_slurm = compute_step(setupfile, "psf", "psf", specs, None, ntask, workermax["psf"], shell_mpi_run, shell_maxcores, 1, maxnodes, nodecores, threads, 1, queuethresh)
        # all_shell.append(scr_shell)
        # all_slurm.append(scr_slurm)

        for nt in nights:

            ntask = expnightcount[nt]["arc"] * 3 * nspect

            scr_shell, scr_slurm, scr_shifter = compute_step(args.shifter, 
                rawdir, specdir, desiroot, setupfile, first, last, specs, nt, 
                ntask, taskproc, shell_mpi_run, shell_maxcores, 1, maxnodes,
                nodecores, step_threads, step_mp, queuethresh)
            nt_shell[nt].append(scr_shell)
            nt_slurm[nt].append(scr_slurm)
            nt_shifter[nt].append(scr_shifter)

        # psfcombine

        ntask = len(nights) * 3 * nspect
        taskproc = workermax["psfcombine"]
        step_threads = 1
        step_mp = 1
        nt = None
        first = "psfcombine"
        last = "psfcombine"

        scr_shell, scr_slurm, scr_shifter = compute_step(args.shifter, 
            rawdir, specdir, desiroot, setupfile, first, last, specs, nt, 
            ntask, taskproc, shell_mpi_run, shell_maxcores, 1, maxnodes,
            nodecores, step_threads, step_mp, queuethresh)
        all_shell.append(scr_shell)
        all_slurm.append(scr_slurm)
        all_shifter.append(scr_shifter)

        # scr_shell, scr_slurm = compute_step(setupfile, "psfcombine", "psfcombine", specs, None, ntask, workermax["psfcombine"], shell_mpi_run, shell_maxcores, 1, maxnodes, nodecores, 1, 1, queuethresh)
        # all_shell.append(scr_shell)
        # all_slurm.append(scr_slurm)

        for nt in nights:

            ntask = 3 * nspect

            scr_shell, scr_slurm, scr_shifter = compute_step(args.shifter, 
                rawdir, specdir, desiroot, setupfile, first, last, specs, nt, 
                ntask, taskproc, shell_mpi_run, shell_maxcores, 1, maxnodes,
                nodecores, step_threads, step_mp, queuethresh)
            nt_shell[nt].append(scr_shell)
            nt_slurm[nt].append(scr_slurm)
            nt_shifter[nt].append(scr_shifter)

            # scr_shell, scr_slurm = compute_step(setupfile, "psfcombine", "psfcombine", specs, nt, ntask, workermax["psfcombine"], shell_mpi_run, shell_maxcores, 1, maxnodes, nodecores, 1, 1, queuethresh)
            # nt_shell[nt].append(scr_shell)
            # nt_slurm[nt].append(scr_slurm)

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
    taskproc = workermax["extract"]
    step_threads = 1
    step_mp = 1
    nt = None
    first = "extract"
    last = "extract"

    scr_shell, scr_slurm, scr_shifter = compute_step(args.shifter, 
        rawdir, specdir, desiroot, setupfile, first, last, specs, nt, 
        ntask, taskproc, shell_mpi_run, shell_maxcores, 1, maxnodes,
        nodecores, step_threads, step_mp, queuethresh)
    all_shell.append(scr_shell)
    all_slurm.append(scr_slurm)
    all_shifter.append(scr_shifter)

    # scr_shell, scr_slurm = compute_step(setupfile, "extract", "extract", specs, None, ntask, workermax["extract"], shell_mpi_run, shell_maxcores, 1, maxnodes, nodecores, threads, 1, queuethresh)
    # all_shell.append(scr_shell)
    # all_slurm.append(scr_slurm)

    for nt in nights:

        ntask = (expnightcount[nt]["flat"] + expnightcount[nt]["science"]) * 3 * nspect

        scr_shell, scr_slurm, scr_shifter = compute_step(args.shifter, 
            rawdir, specdir, desiroot, setupfile, first, last, specs, nt, 
            ntask, taskproc, shell_mpi_run, shell_maxcores, 1, maxnodes,
            nodecores, step_threads, step_mp, queuethresh)
        nt_shell[nt].append(scr_shell)
        nt_slurm[nt].append(scr_slurm)
        nt_shifter[nt].append(scr_shifter)

        # scr_shell, scr_slurm = compute_step(setupfile, "extract", "extract", specs, nt, ntask, workermax["extract"], shell_mpi_run, shell_maxcores, 1, maxnodes, nodecores, threads, 1, queuethresh)
        # nt_shell[nt].append(scr_shell)
        # nt_slurm[nt].append(scr_slurm)

    # calibration

    ntask = totcount["science"] * 3 * nspect
    taskproc = 1
    step_threads = 1
    step_mp = 1
    nt = None
    first = "fiberflat"
    last = "calibrate"

    scr_shell, scr_slurm, scr_shifter = compute_step(args.shifter, 
        rawdir, specdir, desiroot, setupfile, first, last, specs, nt, 
        ntask, taskproc, shell_mpi_run, shell_maxcores, 1, maxnodes,
        nodecores, step_threads, step_mp, queuethresh)
    all_shell.append(scr_shell)
    all_slurm.append(scr_slurm)
    all_shifter.append(scr_shifter)

    # multip = 1      #- turning off multiprocessing

    # scr_shell, scr_slurm = compute_step(setupfile, "fiberflat", "calibrate", specs, None, ntask, 1, shell_mpi_run, shell_maxcores, 1, maxnodes, nodecores, 1, multip, queuethresh)
    # all_shell.append(scr_shell)
    # all_slurm.append(scr_slurm)

    for nt in nights:

        ntask = expnightcount[nt]["science"] * 3 * nspect

        scr_shell, scr_slurm, scr_shifter = compute_step(args.shifter, 
            rawdir, specdir, desiroot, setupfile, first, last, specs, nt, 
            ntask, taskproc, shell_mpi_run, shell_maxcores, 1, maxnodes,
            nodecores, step_threads, step_mp, queuethresh)
        nt_shell[nt].append(scr_shell)
        nt_slurm[nt].append(scr_slurm)
        nt_shifter[nt].append(scr_shifter)

        # scr_shell, scr_slurm = compute_step(setupfile, "fiberflat", "calibrate", specs, nt, ntask, 1, shell_mpi_run, shell_maxcores, 1, maxnodes, nodecores, 1, multip, queuethresh)
        # nt_shell[nt].append(scr_shell)
        # nt_slurm[nt].append(scr_slurm)

    # make bricks - serial only for now!

    rundir = io.get_pipe_rundir()
    scrdir = os.path.join(rundir, io.get_pipe_scriptdir())
    logdir = os.path.join(rundir, io.get_pipe_logdir())

    brickcom = []
    for nt in nights:
        brickcom.append("desi_make_bricks --night {}".format(nt))

    shell_path = os.path.join(scrdir, "bricks.sh")
    shell_log = os.path.join(logdir, "bricks_sh")
    pipe.shell_job(shell_path, shell_log, setupfile, brickcom, comrun=shell_mpi_run, mpiprocs=1, threads=1)
    all_shell.append(shell_path)

    nersc_path = os.path.join(scrdir, "bricks.slurm")
    nersc_log = os.path.join(logdir, "bricks_slurm")
    pipe.nersc_job(nersc_path, nersc_log, setupfile, brickcom, nodes=1,
        nodeproc=1, minutes=30, multisrun=False, openmp=False, multiproc=False,
        queue="debug", jobname="bricks")
    all_slurm.append(nersc_path)

    if args.shifter is not None:
        nersc_path = os.path.join(scrdir, "bricks_shifter.slurm")
        nersc_log = os.path.join(logdir, "bricks_shifter")
        pipe.nersc_shifter_job(nersc_path, args.shifter, 
            rawdir, specdir, desiroot, nersc_log, setupfile, brickcom, nodes=1,
            nodeproc=1, minutes=30, multisrun=False, openmp=False, multiproc=False,
            queue="debug", jobname="bricks")
        all_shifter.append(nersc_path)

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

    taskproc = workermax["redshift"]
    step_threads = 1
    step_mp = 1
    nt = None
    first = "redshift"
    last = "redshift"

    scr_shell, scr_slurm, scr_shifter = compute_step(args.shifter, 
        rawdir, specdir, desiroot, setupfile, first, last, specs, nt, 
        efftask, taskproc, shell_mpi_run, shell_maxcores, 1, maxnodes,
        nodecores, step_threads, step_mp, queuethresh, queue=redqueue, minutes=redtime)
    all_shell.append(scr_shell)
    all_slurm.append(scr_slurm)
    all_shifter.append(scr_shifter)

    # scr_shell, scr_slurm = compute_step(setupfile, "redshift", "redshift", specs,
    #     None, efftask, workermax["redshift"], shell_mpi_run, shell_maxcores,
    #     1, maxnodes, nodecores, 1, 1, queuethresh, queue=redqueue, minutes=redtime)
    # all_shell.append(scr_shell)
    # all_slurm.append(scr_slurm)

    # Make high-level shell scripts which run or submit the steps

    mode = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH

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
    os.chmod(run_slurm_all, mode)

    if args.shifter is not None:
        run_shifter_all = os.path.join(scrdir, "run_shifter_all.sh")
        with open(run_shifter_all, "w") as f:
            f.write("#!/bin/bash\n\n")
            first = True
            for scr in all_shifter:
                if first:
                    f.write("jobid=`sbatch {} | awk '{{print $4}}'`\n\n".format(scr))
                    first = False
                else:
                    f.write("jobid=`sbatch -d afterok:${{jobid}} {} | awk '{{print $4}}'`\n\n".format(scr))
        os.chmod(run_shifter_all, mode)

    run_shell_all = os.path.join(scrdir, "run_shell_all.sh")
    with open(run_shell_all, "w") as f:
        f.write("#!/bin/bash\n\n")
        for scr in all_shell:
            f.write("bash {}\n\n".format(scr))
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
        os.chmod(run_slurm_nt, mode)

        if args.shifter is not None:
            run_shifter_nt = os.path.join(scrdir, "run_shifter_{}.sh".format(nt))
            with open(run_shifter_nt, "w") as f:
                f.write("#!/bin/bash\n\n")
                first = True
                for scr in nt_shifter[nt]:
                    if first:
                        f.write("jobid=`sbatch {} | awk '{{print $4}}'`\n\n".format(scr))
                        first = False
                    else:
                        f.write("jobid=`sbatch -d afterok:${{jobid}} {} | awk '{{print $4}}'`\n\n".format(scr))
            os.chmod(run_shifter_nt, mode)

        run_shell_nt = os.path.join(scrdir, "run_shell_{}.sh".format(nt))
        with open(run_shell_nt, "w") as f:
            f.write("#!/bin/bash\n\n")
            for scr in nt_shell[nt]:
                f.write("bash {}\n\n".format(scr))
        os.chmod(run_shell_nt, mode)

    # Create some helper scripts which run all the data in chains for each night

    scfile = os.path.join(scrdir, "run_shell.sh")
    sc = open(scfile, "w")
    sc.write("#!/bin/bash\n\n")
    for nt in nights:
        run_nt = os.path.join(scrdir, "run_shell_{}.sh".format(nt))
        sc.write("{}\n\n".format(run_nt))
    sc.close()
    os.chmod(scfile, mode)

    scfile = os.path.join(scrdir, "run_slurm.sh")
    sc = open(scfile, "w")
    sc.write("#!/bin/bash\n\n")
    for nt in nights:
        run_nt = os.path.join(scrdir, "run_slurm_{}.sh".format(nt))
        sc.write("{}\n\n".format(run_nt))
    sc.close()
    os.chmod(scfile, mode)

    scfile = os.path.join(scrdir, "run_shifter.sh")
    sc = open(scfile, "w")
    sc.write("#!/bin/bash\n\n")
    for nt in nights:
        run_nt = os.path.join(scrdir, "run_shifter_{}.sh".format(nt))
        sc.write("{}\n\n".format(run_nt))
    sc.close()
    os.chmod(scfile, mode)
