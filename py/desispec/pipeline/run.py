#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.run
=====================

Tools for running the pipeline.
"""

from __future__ import absolute_import, division, print_function

import os
import errno
import sys
import subprocess as sp
import re
import pickle

import yaml

import desispec
from desispec.log import get_logger
from .plan import *


def default_options():
    allopts = {}

    opts = {}
    opts['trace-only'] = False
    opts['legendre-degree'] = 6
    allopts['bootcalib'] = opts

    opts = {}
    opts['flux-hdu'] = 1
    opts['ivar-hdu'] = 2
    opts['mask-hdu'] = 3
    opts['header-hdu'] = 1
    opts['xcoord-hdu'] = 1
    opts['ycoord-hdu'] = 1
    opts['psfmodel'] = 'GAUSSHERMITE'
    opts['half_size_x'] = 8
    opts['half_size_y'] = 5
    opts['verbose'] = False
    opts['gauss_hermite_deg'] = 6
    opts['legendre_deg_wave'] = 4
    opts['legendre_deg_x'] = 1
    opts['trace_deg_wave'] = 6
    opts['trace_deg_x'] = 6
    allopts['specex'] = opts

    opts = {}
    opts['regularize'] = 0.0
    opts['nwavestep'] = 50
    opts['verbose'] = False
    allopts['extract'] = opts

    allopts['fiberflat'] = {}

    allopts['sky'] = {}

    allopts['stdstars'] = {}

    allopts['fluxcal'] = {}

    allopts['procexp'] = {}

    allopts['makebricks'] = {}

    allopts['zfind'] = {}

    return allopts


def write_options(path, opts):
    with open(path, 'w') as f:
        yaml.dump(opts, f, default_flow_style=False)
    return


def read_options(path):
    opts = None
    with open(path, 'r') as f:
        opts = yaml.load(f)
    return opts


def qa_path(datafile, suffix="_QA"):
    dir = os.path.dirname(datafile)
    base = os.path.basename(datafile)
    root, ext = os.path.splitext(base)
    qafile = os.path.join(dir, "{}{}.pdf".format(root, suffix))
    return qafile


def option_list(opts):
    optlist = []
    for key, val in opts.items():
        keystr = "--{}".format(key)
        if isinstance(val, (bool,)):
            if val:
                optlist.append(keystr)
                optlist.append()
        else:
            optlist.append(keystr)
            if isinstance(val, (float,)):
                optlist.append("{:.14e}".format(val))
            else:
                optlist.append("{}".format(val))
    return optlist


def finish_task(name, node):
    # eventually we will mark this as complete in a database...
    pass


def is_finished(rawdir, proddir, grph, name):
    # eventually, we could check a database to get this info...

    if step == 'bootcalib':
        outpath = graph_path_psfboot(proddir, name)
        if not os.path.isfile(outpath):
            return False
        tout = os.path.getmtime(outpath)
        for input in grph[name]['in']:
            inpath = graph_path_pix(rawdir, input)
            # if the input file exists, check if its timestamp
            # is newer than the output.
            if os.path.isfile(inpath):
                tin = os.path.getmtime(inpath)
                if tin > tout:
                    return False

    elif step == 'specex':
        pass

    elif step == 'extract':
        pass
    
    elif step == 'fiberflat':
        pass
    
    elif step == 'sky':
        pass
    
    elif step == 'stdstars':
        pass
    
    elif step == 'fluxcal':
        pass
    
    elif step == 'procexp':
        pass
    
    elif step == 'makebricks':
        pass
    
    elif step == 'zfind':
        pass
    
    else:
        raise RuntimeError("Unknown pipeline step {}".format(step))

    return True


def run_task(step, rawdir, proddir, grph, opts, comm=None):
    if step not in graph_types:
        raise ValueError("step type {} not recognized".format(step))
    # Verify that there is only a single node in the graph
    # of the desired step.  The graph should already have
    # been sliced before calling this task.
    nds = []
    for name, nd in grph.items():
        if nd['type'] == step:
            nds.append(name)
    if len(nds) != 1:
        raise RuntimeError("run_task should only be called with a graph containing a single node to process")

    name = nds[0]
    node = grph[name]

    # step-specific operations

    if step == 'bootcalib':
        # The inputs to this step include *all* the arcs and flats for the
        # night.  Here we sort them into the list of arcs and the list of
        # flats, and simply choose the first one of each.
        arcs = []
        flats = []
        for input in node['in']:
            inode = grph[input]
            if inode['type'] == 'arc':
                arcs.append(input)
            elif inode['type'] == 'flat':
                flats.append(input)
        firstarc = sorted(arcs)[0]
        firstflat = sorted(flats)[0]
        # build list of options
        arcpath = graph_path_pix(rawdir, firstarc)
        flatpath = graph_path_pix(rawdir, firstflat)
        outpath = graph_path_psfboot(proddir, name)
        qapath = run_qa_path(outpath)
        options = {}
        options['fiberflat'] = flatpath
        options['arcfile'] = arcpath
        options['qafile'] = qapath
        options['outfile'] = outpath
        options.update(opts)
        optarray = option_list(options)
        args = desispec.scripts.bootcalib.parse(optarray)
        desispec.scripts.bootcalib.main(args)

    elif step == 'specex':
        pass

    elif step == 'extract':
        pass
    
    elif step == 'fiberflat':
        pass
    
    elif step == 'sky':
        pass
    
    elif step == 'stdstars':
        pass
    
    elif step == 'fluxcal':
        pass
    
    elif step == 'procexp':
        pass
    
    elif step == 'makebricks':
        pass
    
    elif step == 'zfind':
        pass
    
    else:
        raise RuntimeError("Unknown pipeline step {}".format(step))

    return


def run_step(step, rawdir, proddir, grph, opts, comm=None, taskproc=1):
    log = get_logger()

    nproc = 1
    rank = 0
    if comm is not None:
        nproc = comm.size
        rank = comm.rank

    # Get the tasks that need to be done for this step.

    tasks = None
    if rank == 0:
        # For this step, compute all the tasks that we need to do
        alltasks = []
        for name, nd in grph.items():
            if nd['type'] == step:
                alltasks.append(name)

        # For each task, prune if it is finished
        tasks = []
        for t in alltasks:
            if not is_finished(rawdir, proddir, grph, t):
                tasks.append(t)

    if comm is not None:
        tasks = comm.bcast(tasks, root=0)

    ntask = len(tasks)

    # Get the options for this step.

    options = opts[step]

    # Now every process has the full list of tasks.  If we have multiple
    # processes for each task, split the communicator.

    comm_group = comm
    group = rank
    ngroup = nproc
    group_rank = 0
    if comm is not None:
        if taskproc > 1:
            ngroup = int(nproc / taskproc)
            group = int(rank / taskproc)
            group_rank = rank % ngroup
            comm_group = comm.Split(color=group, key=group_rank)
        else:
            from mpi4py import MPI
            comm_group = MPI.COMM_SELF

    # Now we divide up the tasks among the groups of processes as
    # equally as possible.

    group_ntask = int(ntask / ngroup)
    group_firsttask = 0
    leftover = ntask % ngroup
    if group < leftover:
        group_ntask += 1
        group_firsttask = group * group_ntask
    else:
        group_firsttask = ((group_ntask + 1) * leftover) + (group_ntask * (group - leftover))

    # every group goes and does its tasks...

    faildir = os.path.join(proddir, 'failed')

    for t in range(ntask):
        if (t >= group_firsttask) and (t < group_firsttask + group_ntask):
            # slice out just the graph for this task
            tgraph = graph_slice(grph, names=[tasks[t]], deps=True)
            pfile = os.path.join(faildir, "fail_{}_{}.pkl".format(step, tasks[t]))
            try:
                # if the step previously failed, clear that file now
                if os.path.isfile(pfile):
                    os.remove(pfile)
                run_task(step, rawdir, proddir, tgraph, options, comm=comm_group)
            except:
                # The task threw an exception.  We want to dump all information
                # that will be needed to re-run the run_task() function on just
                # this task.
                msg = "FAILED: step {} task {} (group {}/{} with {} processes)".format(step, tasks[t], (group+1), ngroup, taskproc)
                log.error(msg)
                fpkl = {}
                fpkl['step'] = step
                fpkl['rawdir'] = rawdir
                fpkl['proddir'] = proddir
                fpkl['task'] = tasks[t]
                fpkl['graph'] = tgraph
                fpkl['opts'] = options
                fpkl['procs'] = taskproc
                pfile = "fail_{}_{}.pkl".format(step, tasks[t])
                with open(pfile, 'wb') as p:
                    pickle.dump(fpkl, p)


def retry_task(failpath, newopts=None):
    log = get_logger()

    if not os.path.isfile(failpath):
        raise RuntimeError("failure pickle file {} does not exist".format(failpath))

    fpkl = None
    with open(failpath, 'rb') as p:
        fpkl = pickle.load(p)

    step = fpkl['step']
    rawdir = fpkl['rawdir']
    proddir = fpkl['proddir']
    name = fpkl['task']
    grph = fpkl['graph']
    origopts = fpkl['opts']
    nproc = fpkl['procs']

    comm = None
    if nproc > 1:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        nworld = comm.size
        if nworld != nproc:
            if comm.rank == 0:
                log.warn("WARNING: original task was run with {} processes, re-running with {} instead".format(nproc, nworld))

    opts = origopts
    if newopts is not None:
        log.warn("WARNING: overriding original options")
        opts = newopts

    try:
        run_task(step, rawdir, proddir, grph, opts, comm=comm)
    except:
        log.error("Retry Failed")
    else:
        os.remove(failpath)
    return


def pid_exists( pid ):
    """Check whether pid exists in the current process table.

    **UNIX only.**  Should work the same as psutil.pid_exists().

    Args:
        pid (int): A process ID.

    Returns:
        pid_exists (bool): ``True`` if the process exists in the current process table.
    """
    if pid < 0:
        return False
    if pid == 0:
        # According to "man 2 kill" PID 0 refers to every process
        # in the process group of the calling process.
        # On certain systems 0 is a valid PID but we have no way
        # to know that in a portable fashion.
        raise ValueError('invalid PID 0')
    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            # ESRCH == No such process
            return False
        elif err.errno == errno.EPERM:
            # EPERM clearly means there's a process to deny access to
            return True
        else:
            # According to "man 2 kill" possible error values are
            # (EINVAL, EPERM, ESRCH)
            raise
    else:
        return True


def subprocess_list(tasks, rank=0):
    log = get_logger()
    #pids = []
    for tsk in tasks:
        runcom = True
        newest_in = 0
        for dep in tsk['inputs']:
            if not os.path.isfile(dep):
                err = "dependency {} missing, cannot run task {}".format(dep, " ".join(tsk['command']))
                log.error(err)
                runcom = False
            else:
                t = os.path.getmtime(dep)
                if t > newest_in:
                    newest_in = t
        alldone = True
        if len(tsk['outputs']) == 0:
            alldone = False
        for outf in tsk['outputs']:
            if not os.path.isfile(outf):
                alldone = False
            else:
                t = os.path.getmtime(outf)
                if t < newest_in:
                    alldone = False
        if alldone:
            runcom = False
        proc = None
        if runcom:
            # proc = sp.Popen(tsk['command'], stdout=sp.PIPE, stderr=sp.STDOUT)
            # log.info("subproc[{}]: {}".format(proc.pid, " ".join(tsk['command'])))
            # outs, errs = proc.communicate()
            # for line in outs:
            #     log.debug("subproc[{}]:   {}".format(proc.pid, line.rstrip()))
            # proc.wait()
            log.info("subproc: {}".format(" ".join(tsk['command'])))
            ret = sp.call(tsk['command'])
            for outf in tsk['outputs']:
                ret = sp.call(['mv', '{}.part'.format(outf), outf])

    return


def shell_job(path, logroot, envsetup, desisetup, commands):
    with open(path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("now=`date +%Y%m%d-%H:%M:%S`\n")
        f.write("log={}_${{now}}.log\n\n".format(logroot))
        for com in envsetup:
            f.write("{}\n".format(com))
        f.write("\n")
        f.write("source {}\n\n".format(desisetup))
        for com in commands:
            executable = com.split(' ')[0]
            f.write("which {}\n".format(executable))
            f.write("time {} >>${{log}} 2>&1\n\n".format(com))
    return


def nersc_job(path, logroot, envsetup, desisetup, commands, nodes=1, nodeproc=1, minutes=10, multisrun=False, openmp=False, multiproc=False):
    hours = int(minutes/60)
    fullmin = int(minutes - 60*hours)
    timestr = "{:02d}:{:02d}:00".format(hours, fullmin)

    totalnodes = nodes
    if multisrun:
        # we are running every command as a separate srun
        # and backgrounding them.  In this case, the nodes
        # given are per command, so we need to compute the
        # total.
        totalnodes = nodes * len(commands)

    with open(path, 'w') as f:
        f.write("#!/bin/bash -l\n\n")
        if totalnodes > 512:
            f.write("#SBATCH --partition=regular\n")
        else:
            f.write("#SBATCH --partition=debug\n")
        f.write("#SBATCH --account=desi\n")
        f.write("#SBATCH --nodes={}\n".format(totalnodes))
        f.write("#SBATCH --time={}\n".format(timestr))
        f.write("#SBATCH --job-name=desipipe\n")
        f.write("#SBATCH --output={}_slurm_%j.log\n".format(logroot))
        f.write("#SBATCH --export=NONE\n\n")
        for com in envsetup:
            f.write("{}\n".format(com))
        f.write("\n")
        f.write("source {}\n\n".format(desisetup))
        f.write("node_cores=0\n")
        f.write("if [ ${NERSC_HOST} = edison ]; then\n")
        f.write("  node_cores=24\n")
        f.write("else\n")
        f.write("  node_cores=32\n")
        f.write("fi\n")
        f.write("\n")
        f.write("nodes={}\n".format(nodes))
        f.write("node_proc={}\n".format(nodeproc))
        f.write("node_thread=$(( node_cores / node_proc ))\n")
        f.write("procs=$(( nodes * node_proc ))\n\n")
        if openmp:
            f.write("export OMP_NUM_THREADS=${node_thread}\n")
            f.write("\n")
        runstr = "srun --export=ALL"
        if multiproc:
            runstr = "{} --cpu_bind=no".format(runstr)
        f.write("run=\"{} -n ${{procs}} -N ${{nodes}} -c ${{node_thread}}\"\n\n".format(runstr))
        f.write("now=`date +%Y%m%d-%H:%M:%S`\n")
        f.write("echo \"job datestamp = ${now}\"\n")
        f.write("log={}_${{now}}.log\n\n".format(logroot))
        for com in commands:
            executable = com.split(' ')[0]
            f.write("which {}\n".format(executable))
            f.write("time ${{run}} {} >>${{log}} 2>&1".format(com))
            if multisrun:
                f.write(" &")
            f.write("\n\n")
        if multisrun:
            f.write("wait\n\n")
    return

