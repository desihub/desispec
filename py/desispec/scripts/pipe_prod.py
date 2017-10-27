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
from ..parallel import dist_discrete, dist_balanced

from .. import pipeline as pipe


def job_size(ntask, taskprocs, tasktime, nodeproc, maxnodes):
    """
    Internal helper function used only in the desi_pipe
    script to compute the size of batch jobs.
    """
    maxprocs = nodeproc * maxnodes
    maxworkers = maxprocs // taskprocs

    fullprocs = ntask * taskprocs

    nodes = None
    time = None
    procs = None
    
    if fullprocs <= maxprocs:
        # we can run fully unpacked, with every process group
        # assigned one task.
        time = tasktime
        procs = fullprocs
    else:
        # we want to repack as efficiently as possible
        workers = dist_balanced(ntask, maxworkers)
        nworker = len(workers)
        procs = nworker * taskprocs
        time = tasktime * workers[0][1]
    
    nodes = procs // nodeproc + 1
    return (nodes, procs, time)


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
    scrstr = ""
    if night is not None:
        nstr = " --nights {}".format(night)
        scrstr = "{}".format(night)

    stepstr = ""
    jobname = ""
    if first == last:
        stepstr = first
        if scrstr != "":
            stepstr = "{}_{}".format(first, scrstr)
        jobname = first
    else:
        stepstr = "{}-{}".format(first, last)
        if scrstr != "":
            stepstr = "{}-{}_{}".format(first, last, scrstr)
        jobname = "{}_{}".format(first, last)

    return (rundir, scrdir, logdir, specstr, nstr, scrstr, stepstr, jobname)


def compute_step(img, specdata, specredux, desiroot, setupfile, 
    first, last, specs, night, ntask, taskproc, tasktime, shell_mpi_run, 
    shell_maxcores, shell_threads, nersc_host, nersc_maxnodes,
    nersc_nodecores, nersc_threads, nersc_mp, nersc_queue_thresh,
    queue="debug"):
    """
    Internal helper function used only in the desi_pipe script to
    generate the job scripts.
    """

    (rundir, scrdir, logdir, specstr, nstr, scrstr, stepstr, jobname) = \
        step_props(first, last, specs, night)

    totproc = ntask * taskproc

    shell_maxprocs = shell_maxcores // shell_threads
    shell_procs = shell_maxprocs
    if totproc < shell_procs:
        shell_procs = totproc

    ntdir = scrdir
    logntdir = logdir
    if night is not None:
        ntdir = os.path.join(scrdir, night)
        if not os.path.isdir(ntdir):
            os.makedirs(ntdir)
        logntdir = os.path.join(logdir, night)
        if not os.path.isdir(logntdir):
            os.makedirs(logntdir)

    shell_path = os.path.join(ntdir, "{}.sh".format(stepstr))
    shell_log = os.path.join(logntdir, "{}_sh".format(stepstr))

    #- no MPI for shell job version so that it can be run from interactive node
    com = None
    if shell_maxcores == 1:
        com = ["desi_pipe_run --first {} --last {}{}{}".format(first, last, specstr, nstr)]
    else:
        com = ["desi_pipe_run_mpi --first {} --last {}{}{}".format(first, last, specstr, nstr)]
    
    pipe.shell_job(shell_path, shell_log, setupfile, com, comrun=shell_mpi_run,
        mpiprocs=shell_procs, threads=shell_threads)

    # Compute job size for NERSC runs

    core_per_proc = 1
    if nersc_threads > 1:
        core_per_proc = nersc_threads
    elif nersc_mp > 1:
        core_per_proc = nersc_mp

    nodeproc = nersc_nodecores // core_per_proc

    (nodes, procs, time) = job_size(ntask, taskproc, tasktime, nodeproc, nersc_maxnodes)

    if nodes > nersc_queue_thresh and queue == "debug":
        print("{} nodes too big for debug queue; switching to regular".format(nodes))
        queue = "regular"

    if time > 30 and queue == "debug":
        print("{} minutes too big for debug queue; switching to regular".format(time))
        queue = "regular"

    com = ["desi_pipe_run_mpi --first {} --last {}{}{}".format(first, last, specstr, nstr)]

    # write normal slurm script

    nersc_path = os.path.join(ntdir, "{}.slurm".format(stepstr))
    nersc_log = os.path.join(logntdir, "{}_slurm".format(stepstr))

    pipe.nersc_job(nersc_host, nersc_path, nersc_log, setupfile, com, 
        nodes=nodes, nodeproc=nodeproc, minutes=time, multisrun=False, 
        openmp=(nersc_threads > 1), multiproc=(nersc_mp > 1), queue=queue, 
        jobname=jobname)

    nersc_shifter_path = ""
    if img is not None:
        # write shifter slurm script

        nersc_shifter_path = os.path.join(ntdir, 
            "{}_shifter.slurm".format(stepstr))
        nersc_shifter_log = os.path.join(logntdir, 
            "{}_shifter".format(stepstr))

        pipe.nersc_shifter_job(nersc_host, nersc_shifter_path, img, specdata,
            specredux, desiroot, nersc_shifter_log, setupfile, com, 
            nodes=nodes, nodeproc=nodeproc, minutes=time, multisrun=False, 
            openmp=(nersc_threads > 1), multiproc=(nersc_mp > 1), queue=queue, 
            jobname=jobname)

    return (shell_path, nersc_path, nersc_shifter_path)


def parse(options=None):
    parser = argparse.ArgumentParser(description="Set up pipeline runs for a production.")
    parser.add_argument("--data", required=False, default=None, help="override DESI_SPECTRO_DATA")
    parser.add_argument("--redux", required=False, default=None, help="override DESI_SPECTRO_REDUX")
    parser.add_argument("--prod", required=False, default=None, help="override SPECPROD")
    
    parser.add_argument("--starmodels", required=False, default=None, help="override the default star model file")

    parser.add_argument("--nights", required=False, default=None, help="comma separated (YYYYMMDD) or regex pattern")
    
    parser.add_argument("--nersc_host", required=False, default="edison", help="NERSC slurm scripts host name (edison|cori|coriknl)")

    parser.add_argument("--nersc_queue", required=False, default="regular", help="NERSC queue to use.")

    parser.add_argument("--nersc_max_nodes", required=False, default=None, help="NERSC slurm scripts max nodes to use.  Default is 1/3 of the machine.")

    parser.add_argument("--shell_mpi_run", required=False, default="mpirun -np", help="bash scripts command to launch MPI pipeline steps.  If --shell_max_cores is 1, this is ignored.")
    parser.add_argument("--shell_max_cores", required=False, default=1, help="bash scripts max cores to use.")

    parser.add_argument("--fakeboot", required=False, default=False, action="store_true", help="bypass bootcalib")

    parser.add_argument("--fakepsf", required=False, default=False, action="store_true", help="bypass specex")

    parser.add_argument("--fakepix", required=False, default=False, action="store_true", help="bypass checks for pixel data.  Useful when skipping extraction step in simulations.")

    parser.add_argument("--spectrographs", required=False, default=None, help="process only this comma-separated list of spectrographs")

    parser.add_argument("--debug", required=False, default=False, action="store_true", help="in setup script, set log level to DEBUG")

    parser.add_argument("--shifter", required=False, default=None, help="shifter image to use in alternate slurm scripts")

    parser.add_argument("--nside", required=False, type=int, default=64, 
        help="HEALPix nside value to use for spectral grouping.")

    parser.add_argument("--redshift_nodes", required=False, type=int, 
        default=1, help="Number of nodes to use for redshift fitting "
        "in a single worker.")

    parser.add_argument("--redshift_spec_per_minute", required=False, 
        type=int, default=80, help="Number of spectra that can be "
        "processed in a minute on a single node using all templates.")

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
    elif args.nersc_host == "cori":
        nodecores = 32
        queuethresh = 64
        if args.nersc_max_nodes is not None:
            maxnodes = int(args.nersc_max_nodes)
        else:
            maxnodes = 512
    elif args.nersc_host == "coriknl":
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

    # Add any extra options to the initial options.yaml file

    extra = {}
    if args.starmodels is not None:
        extra["Stdstars"] = {}
        extra["Stdstars"]["starmodels"] = args.starmodels

    #
    # Redrock runtime notes from 2017-09.  All numbers rounded
    # down to be as conservative as possible:
    #   Small test (2% DC, one day, one spectrograph)
    #     edison:  1 node per worker, 24 procs per node,
    #            ~3500 spectra in 30 min =~ 110 spec per node min
    #     knl: 1 node per worker, 32 procs per node (not using 64
    #            due to memory constraint- will gain 2x after fix),
    #            ~800 spectra in 35 min =~ 20 spec per node min
    #   Medium test (2% DC, one day)
    #     edison:  2 nodes per worker, 24 procs per node,
    #            ~6000 spectra in 40 min =~ 75 spec per node min
    #     edison:  1 nodes per worker, 24 procs per node,
    #            ~3000 spectra in 30 min =~ 100 spec per node min
    #

    extra["Redrock"] = {}
    extra["Redrock"]["nodes"] = args.redshift_nodes
    extra["Redrock"]["nodeprocs"] = nodecores // 2
    extra["Redrock"]["spec_per_minute"] = args.redshift_spec_per_minute

    # Select our spectrographs

    specs = [ x for x in range(10) ]
    if args.spectrographs is not None:
        specs = [ int(x) for x in args.spectrographs.split(",") ]
    nspect = len(specs)

    # Update output directories and plans

    print("Working with production {} :".format(proddir))

    print("  Updating plans ...")
    expnightcount, allpix = pipe.create_prod(nightstr=args.nights, 
        extra=extra, specs=specs, fakepix=args.fakepix, hpxnside=args.nside)
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

    workerhandles = {}
    workermax = {}
    workertime = {}
    workernames = {}
    for step in pipe.step_types:
        workernames[step] = opts["{}_worker".format(step)]
        worker = pipe.get_worker(step, opts["{}_worker".format(step)], 
            opts["{}_worker_opts".format(step)])
        workermax[step] = worker.max_nproc()
        workertime[step] = worker.task_time()
        workerhandles[step] = worker
        print("    {} : {} processes per task".format(step, workermax[step]))

    # create scripts for processing

    print("  Generating scripts ...")

    nt_slurm = {}
    nt_shell = {}
    nt_shifter = {}
    for nt in nights:
        nt_slurm[nt] = []
        nt_shell[nt] = []
        nt_shifter[nt] = []

    # bootcalib

    if not args.fakeboot:

        taskproc = workermax["bootstrap"]
        taskmin = workertime["bootstrap"]
        step_threads = 1
        step_mp = 2
        nt = None
        first = "bootstrap"
        last = "bootstrap"
        
        for nt in nights:

            ntask = 3 * nspect

            scr_shell, scr_slurm, scr_shifter = compute_step(args.shifter, 
                rawdir, specdir, desiroot, setupfile, first, last, specs, nt, 
                ntask, taskproc, taskmin, shell_mpi_run, shell_maxcores, 1, 
                args.nersc_host, maxnodes, nodecores, step_threads, step_mp,
                queuethresh, queue=args.nersc_queue)
            nt_shell[nt].append(scr_shell)
            nt_slurm[nt].append(scr_slurm)
            nt_shifter[nt].append(scr_shifter)

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

        taskproc = workermax["psf"]
        taskmin = workertime["psf"]
        step_threads = 2
        step_mp = 1
        nt = None
        first = "psf"
        last = "psf"

        for nt in nights:

            ntask = expnightcount[nt]["arc"] * 3 * nspect

            scr_shell, scr_slurm, scr_shifter = compute_step(args.shifter, 
                rawdir, specdir, desiroot, setupfile, first, last, specs, nt, 
                ntask, taskproc, taskmin, shell_mpi_run, shell_maxcores, 1, 
                args.nersc_host, maxnodes, nodecores, step_threads, step_mp,
                queuethresh, queue=args.nersc_queue)
            nt_shell[nt].append(scr_shell)
            nt_slurm[nt].append(scr_slurm)
            nt_shifter[nt].append(scr_shifter)

        # psfcombine

        taskproc = workermax["psfcombine"]
        taskmin = workertime["psfcombine"]
        step_threads = 1
        step_mp = 1
        nt = None
        first = "psfcombine"
        last = "psfcombine"

        for nt in nights:

            ntask = 3 * nspect

            scr_shell, scr_slurm, scr_shifter = compute_step(args.shifter, 
                rawdir, specdir, desiroot, setupfile, first, last, specs, nt, 
                ntask, taskproc, taskmin, shell_mpi_run, shell_maxcores, 1, 
                args.nersc_host, maxnodes, nodecores, step_threads, step_mp,
                queuethresh, queue=args.nersc_queue)
            nt_shell[nt].append(scr_shell)
            nt_slurm[nt].append(scr_slurm)
            nt_shifter[nt].append(scr_shifter)

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

    if not args.fakepix:

        taskproc = workermax["extract"]
        taskmin = workertime["extract"]
        step_threads = 1 # faster to use max number of processes per node
        step_mp = 1
        nt = None
        first = "extract"
        last = "extract"

        for nt in nights:

            ntask = (expnightcount[nt]["flat"] + expnightcount[nt]["science"]) * 3 * nspect

            scr_shell, scr_slurm, scr_shifter = compute_step(args.shifter, 
                rawdir, specdir, desiroot, setupfile, first, last, specs, nt, 
                ntask, taskproc, taskmin, shell_mpi_run, shell_maxcores, 1, 
                args.nersc_host, maxnodes, nodecores, step_threads, step_mp,
                queuethresh, queue=args.nersc_queue)
            nt_shell[nt].append(scr_shell)
            nt_slurm[nt].append(scr_slurm)
            nt_shifter[nt].append(scr_shifter)

    # calibration

    taskproc = 1
    
    tasktime = 0
    tasktime += workertime["fiberflat"]
    tasktime += workertime["sky"]
    tasktime += workertime["stdstars"]
    tasktime += workertime["fluxcal"]
    tasktime += workertime["calibrate"]

    step_threads = 2
    step_mp = 1
    nt = None
    first = "fiberflat"
    last = "calibrate"

    for nt in nights:

        ntask = expnightcount[nt]["science"] * 3 * nspect

        scr_shell, scr_slurm, scr_shifter = compute_step(args.shifter, 
            rawdir, specdir, desiroot, setupfile, first, last, specs, nt, 
            ntask, taskproc, tasktime, shell_mpi_run, shell_maxcores, 1, 
            args.nersc_host, maxnodes, nodecores, step_threads, step_mp, 
            queuethresh, queue=args.nersc_queue)
        nt_shell[nt].append(scr_shell)
        nt_slurm[nt].append(scr_slurm)
        nt_shifter[nt].append(scr_shifter)

    # Make spectral groups.  The groups are distributed, and we use
    # approximately 5 spectra per process.  We also use one process
    # per two cores for more memory.

    # On KNL, the timing for ~5 spectra per process is about 1.5 hours.
    # we use that empirical metric here to estimate the run time with
    # some margin.

    ngroup = len(allpix.keys())

    spectime = 120
    specprocs = ngroup // 5
    specnodeprocs = nodecores // 2
    specnodes = specprocs // specnodeprocs
    if specnodes == 0:
        specnodes = 1
    specprocs = specnodes * specnodeprocs

    specqueue = args.nersc_queue
    if spectime > 30 and specqueue == "debug":
        specqueue = "regular"

    rundir = io.get_pipe_rundir()
    scrdir = os.path.join(rundir, io.get_pipe_scriptdir())
    logdir = os.path.join(rundir, io.get_pipe_logdir())

    speccom = ["desi_group_spectra --pipeline"]

    shell_path = os.path.join(scrdir, "spectra.sh")
    shell_log = os.path.join(logdir, "spectra_sh")
    pipe.shell_job(shell_path, shell_log, setupfile, speccom, comrun=shell_mpi_run, mpiprocs=shell_maxcores, threads=1)

    nersc_path = os.path.join(scrdir, "spectra.slurm")
    nersc_log = os.path.join(logdir, "spectra_slurm")
    pipe.nersc_job(args.nersc_host, nersc_path, nersc_log, setupfile, 
        speccom, nodes=specnodes, nodeproc=specnodeprocs, minutes=spectime, 
        multisrun=False, openmp=True, multiproc=False, 
        queue=specqueue, jobname="groupspectra")

    if args.shifter is not None:
        nersc_path = os.path.join(scrdir, "spectra_shifter.slurm")
        nersc_log = os.path.join(logdir, "spectra_shifter")
        pipe.nersc_shifter_job(args.nersc_host, nersc_path, args.shifter, 
            rawdir, specdir, desiroot, nersc_log, setupfile, speccom, 
            nodes=specnodes, nodeproc=specnodeprocs, minutes=30, 
            multisrun=False, openmp=False, multiproc=False,
            queue=specqueue, jobname="groupspectra")

    # Redshift fitting.  Use estimated spectra per node minute.

    red_worker_nodes = workerhandles["redshift"].max_nodes()
    red_nodeprocs = workerhandles["redshift"].node_procs()
    red_spec_per_node_min = workerhandles["redshift"].spec_per_min()
    red_totalspec = np.sum([ y for x, y in allpix.items() ])

    red_runtime = workertime["redshift"]

    ntask = len(allpix.keys())
    
    nworker = 1 + red_totalspec // (red_runtime * red_worker_nodes *
        red_spec_per_node_min)
    
    red_nodes = nworker * red_worker_nodes

    red_queue = args.nersc_queue
    if red_runtime > 30 and red_queue == "debug":
        red_queue = "regular"
    
    rundir = io.get_pipe_rundir()
    scrdir = os.path.join(rundir, io.get_pipe_scriptdir())
    logdir = os.path.join(rundir, io.get_pipe_logdir())

    redcom = ["desi_pipe_run_mpi --first redshift --last redshift"]

    shell_path = os.path.join(scrdir, "redshift.sh")
    shell_log = os.path.join(logdir, "redshift_sh")
    pipe.shell_job(shell_path, shell_log, setupfile, redcom, 
        comrun=shell_mpi_run, mpiprocs=shell_maxcores, threads=1)

    nersc_path = os.path.join(scrdir, "redshift.slurm")
    nersc_log = os.path.join(logdir, "redshift_slurm")
    pipe.nersc_job(args.nersc_host, nersc_path, nersc_log, setupfile, 
        redcom, nodes=red_nodes, nodeproc=red_nodeprocs, 
        minutes=red_runtime, multisrun=False, openmp=True, multiproc=False, 
        queue=red_queue, jobname="redshift")

    if args.shifter is not None:
        nersc_path = os.path.join(scrdir, "redshift_shifter.slurm")
        nersc_log = os.path.join(logdir, "redshift_shifter")
        pipe.nersc_shifter_job(args.nersc_host, nersc_path, args.shifter, 
            rawdir, specdir, desiroot, nersc_log, setupfile, redcom, 
            nodes=red_nodes, nodeproc=red_nodeprocs, minutes=red_runtime, 
            multisrun=False, openmp=True, multiproc=False,
            queue=red_queue, jobname="redshift")


    # Make high-level shell scripts which run or submit the steps

    mode = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH

    for nt in nights:
        ntdir = os.path.join(scrdir, nt)
        if not os.path.isdir(ntdir):
            os.makedirs(ntdir)
        run_slurm_nt = os.path.join(ntdir, "run_slurm_{}.sh".format(nt))
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
            run_shifter_nt = os.path.join(ntdir, "run_shifter_{}.sh".format(nt))
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

        run_shell_nt = os.path.join(ntdir, "run_shell_{}.sh".format(nt))
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
        ntdir = os.path.join(scrdir, nt)
        run_nt = os.path.join(ntdir, "run_shell_{}.sh".format(nt))
        sc.write("{}\n\n".format(run_nt))
    sc.close()
    os.chmod(scfile, mode)

    scfile = os.path.join(scrdir, "run_slurm.sh")
    sc = open(scfile, "w")
    sc.write("#!/bin/bash\n\n")
    for nt in nights:
        ntdir = os.path.join(scrdir, nt)
        run_nt = os.path.join(ntdir, "run_slurm_{}.sh".format(nt))
        sc.write("{}\n\n".format(run_nt))
    sc.close()
    os.chmod(scfile, mode)

    if args.shifter is not None:
        scfile = os.path.join(scrdir, "run_shifter.sh")
        sc = open(scfile, "w")
        sc.write("#!/bin/bash\n\n")
        for nt in nights:
            ntdir = os.path.join(scrdir, nt)
            run_nt = os.path.join(scrdir, "run_shifter_{}.sh".format(nt))
            sc.write("{}\n\n".format(run_nt))
        sc.close()
        os.chmod(scfile, mode)
