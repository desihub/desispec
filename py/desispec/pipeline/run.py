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
import stat
import errno
import sys
import re
import pickle
import copy
import traceback
import time
import logging
from contextlib import contextmanager

import numpy as np

import yaml

import desispec

import desispec.log
from desispec.log import get_logger
from desispec.util import default_nproc, dist_uniform, dist_discrete
from .plan import *
from .utils import option_list

import desispec.scripts.bootcalib as bootcalib
import desispec.scripts.specex as specex
import desispec.scripts.extract as extract
import desispec.scripts.fiberflat as fiberflat
import desispec.scripts.sky as skypkg
import desispec.scripts.stdstars as stdstars
import desispec.scripts.fluxcalibration as fluxcal
import desispec.scripts.procexp as procexp
import desispec.scripts.zfind as zfind


run_step_types = [
    'bootcalib',
    'specex',
    'psfcombine',
    'extract',
    'fiberflat',
    'sky',
    'stdstars',
    'fluxcal',
    'procexp',
    'zfind'
]


step_file_types = {
    'bootcalib' : ['psfboot'],
    'specex' : ['psf'],
    'psfcombine' : ['psfnight'],
    'extract' : ['frame'],
    'fiberflat' : ['fiberflat'],
    'sky' : ['sky'],
    'stdstars' : ['stdstars'],
    'fluxcal' : ['calib'],
    'procexp' : ['cframe'],
    'zfind' : ['zbest']
}


file_types_step = {
    'psfboot' : 'bootcalib',
    'psf' : 'specex',
    'psfnight' : 'psfcombine',
    'frame' : 'extract',
    'fiberflat' : 'fiberflat',
    'sky' : 'sky',
    'stdstars' : 'stdstars',
    'calib' : 'fluxcal',
    'cframe' : 'procexp',
    'zbest' : 'zfind'
}


run_states = [
    'done',
    'fail',
    'wait'
]


def qa_path(datafile, suffix="_QA", figformat='pdf', qaformat='yaml'):
    '''
    Transforms a data filename into the QA filenames
    
    Args:
        datafile : full path to a data file
        
    Options:
        suffix : add to the data file name before the extension
        figformat : 'pdf', 'jpg', or 'png' format of figure
    
    Returns (qafile, qafig)
        full path names to output QA data file (yaml) and plot files
    '''
    dir = os.path.dirname(datafile)
    base = os.path.basename(datafile)
    root, ext = os.path.splitext(base)
    qafile = os.path.join(dir, "{}{}.{}".format(root, suffix, qaformat))
    qafig = os.path.join(dir, "{}{}.{}".format(root, suffix, figformat))
    return qafile, qafig


def finish_task(name, node):
    # eventually we will mark this as complete in a database...
    pass


def is_finished(rawdir, proddir, grph, name):
    '''
    Determine whether a single data object is finished.

    This checks whether a data object is finished by testing whether 
    the output file exists and is newer than its inputs.

    Args:
        rawdir (str): the path to the raw data directory.
        proddir (str): the path to the production directory.
        grph (dict): the dependency graph.
        name (str): the object name.

    Returns (bool):
        True if the object is finished, False otherwise.
    '''
    # eventually, we could check a database to get this info...

    type = grph[name]['type']

    if type == 'night':
        return True

    outpath = graph_path(rawdir, proddir, name, type)
    if not os.path.isfile(outpath):
        return False

    if os.path.islink(outpath):
        # this is a fake bootcalib symlink
        return True

    tout = os.path.getmtime(outpath)

    for input in grph[name]['in']:
        if grph[input]['type'] == 'night':
            continue
        inpath = graph_path(rawdir, proddir, input, grph[input]['type'])
        # if the input file exists, check if its timestamp
        # is newer than the output.
        if os.path.isfile(inpath):
            tin = os.path.getmtime(inpath)
            if tin > tout:
                return False
    return True


def prod_state(rawdir, proddir, grph):
    '''
    Check the completion state of all objects in a graph.

    This scans over an entire dependency graph and tests each object
    for whether it is finished.  If the object is done, it marks the
    the state of the node in the graph.

    Args:
        rawdir (str): the path to the raw data directory.
        proddir (str): the path to the production directory.
        grph (dict): the dependency graph.

    Returns:
        Nothing.  The graph is modified in place.
    '''
    for name, nd in grph.items():
        if is_finished(rawdir, proddir, grph, name):
            nd['state'] = 'done'
    return


def run_task(step, rawdir, proddir, grph, opts, comm=None):
    '''
    Run a single pipeline task.

    This function takes a truncated graph containing a single node
    of the specified type and the nodes representing the inputs for
    the task.

    Args:
        step (str): the pipeline step type.
        rawdir (str): the path to the raw data directory.
        proddir (str): the path to the production directory.
        grph (dict): the truncated dependency graph.
        opts (dict): the global options dictionary.
        comm (mpi4py.Comm): the optional MPI communicator to use.

    Returns:
        Nothing.
    '''

    if step not in step_file_types:
        raise ValueError("step type {} not recognized".format(step))

    log = get_logger()

    # Verify that there is only a single node in the graph
    # of the desired step.  The graph should already have
    # been sliced before calling this task.
    nds = []
    for name, nd in grph.items():
        if nd['type'] in step_file_types[step]:
            nds.append(name)
    if len(nds) != 1:
        raise RuntimeError("run_task should only be called with a graph containing a single node to process")

    name = nds[0]
    node = grph[name]

    nproc = 1
    rank = 0
    if comm is not None:
        nproc = comm.size
        rank = comm.rank

    # step-specific operations

    if step == 'bootcalib':

        # The inputs to this step include *all* the arcs and flats for the
        # night.  Here we sort them into the list of arcs and the list of
        # flats, and simply choose the first one of each.
        arcs = []
        flats = []
        for input in node['in']:
            inode = grph[input]
            if inode['flavor'] == 'arc':
                arcs.append(input)
            elif inode['flavor'] == 'flat':
                flats.append(input)
        if len(arcs) == 0:
            raise RuntimeError("no arc images found!")
        if len(flats) == 0:
            raise RuntimeError("no flat images found!")
        firstarc = sorted(arcs)[0]
        firstflat = sorted(flats)[0]
        # build list of options
        arcpath = graph_path_pix(rawdir, firstarc)
        flatpath = graph_path_pix(rawdir, firstflat)
        outpath = graph_path_psfboot(proddir, name)
        qafile, qafig = qa_path(outpath)
        options = {}
        options['fiberflat'] = flatpath
        options['arcfile'] = arcpath
        options['qafile'] = qafile
        ### options['qafig'] = qafig
        options['outfile'] = outpath
        options.update(opts)
        optarray = option_list(options)

        # at debug level, write out the equivalent commandline
        if rank == 0:
            com = ['RUN', 'desi_bootcalib']
            com.extend(optarray)
            log.debug(" ".join(com))

        args = bootcalib.parse(optarray)

        sys.stdout.flush()
        if rank == 0:
            #print("proc {} call bootcalib main".format(rank))
            #sys.stdout.flush()
            bootcalib.main(args)
            #print("proc {} returned from bootcalib main".format(rank))
            #sys.stdout.flush()
        #print("proc {} finish runtask bootcalib".format(rank))
        #sys.stdout.flush()

    elif step == 'specex':

        # get input files
        pix = []
        boot = []
        for input in node['in']:
            inode = grph[input]
            if inode['type'] == 'psfboot':
                boot.append(input)
            elif inode['type'] == 'pix':
                pix.append(input)
        if len(boot) != 1:
            raise RuntimeError("specex needs exactly one psfboot file")
        if len(pix) == 0:
            raise RuntimeError("specex needs exactly one image file")
        bootfile = graph_path_psfboot(proddir, boot[0])
        imgfile = graph_path_pix(rawdir, pix[0])
        outfile = graph_path_psf(proddir, name)
        outdir = os.path.dirname(outfile)

        options = {}
        options['input'] = imgfile
        options['bootfile'] = bootfile
        options['output'] = outfile
        if log.getEffectiveLevel() == desispec.log.DEBUG:
            options['verbose'] = True
        if len(opts) > 0:
            extarray = option_list(opts)
            options['extra'] = " ".join(extarray)

        optarray = option_list(options)

        # at debug level, write out the equivalent commandline
        if rank == 0:
            com = ['RUN', 'desi_compute_psf']
            com.extend(optarray)
            log.debug(" ".join(com))

        args = specex.parse(optarray)
        specex.main(args, comm=comm)

    elif step == 'psfcombine':

        outfile = graph_path_psfnight(proddir, name)
        infiles = []
        for input in node['in']:
            infiles.append(graph_path_psf(proddir, input))

        if rank == 0:
            specex.mean_psf(infiles, outfile)

    elif step == 'extract':
        
        pix = []
        psf = []
        fm = []
        band = None
        for input in node['in']:
            inode = grph[input]
            if inode['type'] == 'psfnight':
                psf.append(input)
            elif inode['type'] == 'pix':
                pix.append(input)
                band = inode['band']
            elif inode['type'] == 'fibermap':
                fm.append(input)
        if len(psf) != 1:
            raise RuntimeError("extraction needs exactly one psfnight file")
        if len(pix) != 1:
            raise RuntimeError("extraction needs exactly one image file")
        if len(fm) != 1:
            raise RuntimeError("extraction needs exactly one fibermap file")

        imgfile = graph_path_pix(rawdir, pix[0])
        psffile = graph_path_psfnight(proddir, psf[0])
        fmfile = graph_path_fibermap(rawdir, fm[0])
        outfile = graph_path_frame(proddir, name)

        options = {}
        options['input'] = imgfile
        options['fibermap'] = fmfile
        options['psf'] = psffile
        options['output'] = outfile

        # extract the wavelength range from the options, depending on the band

        optscopy = copy.deepcopy(opts)
        wkey = "wavelength_{}".format(band)
        wave = optscopy[wkey]
        del optscopy['wavelength_b']
        del optscopy['wavelength_r']
        del optscopy['wavelength_z']
        optscopy['wavelength'] = wave

        options.update(optscopy)
        optarray = option_list(options)

        # at debug level, write out the equivalent commandline
        if rank == 0:
            com = ['RUN', 'desi_extract_spectra']
            com.extend(optarray)
            log.debug(" ".join(com))

        args = extract.parse(optarray)
        extract.main_mpi(args, comm=comm)
    
    elif step == 'fiberflat':

        if len(node['in']) != 1:
            raise RuntimeError('fiberflat should have only one input frame')
        framefile = graph_path_frame(proddir, node['in'][0])
        outfile = graph_path_fiberflat(proddir, name)
        qafile, qafig = qa_path(outfile)

        options = {}
        options['infile'] = framefile
        options['qafile'] = qafile
        options['qafig'] = qafig
        options['outfile'] = outfile
        options.update(opts)
        optarray = option_list(options)

        # at debug level, write out the equivalent commandline
        if rank == 0:
            com = ['RUN', 'desi_compute_fiberflat']
            com.extend(optarray)
            log.debug(" ".join(com))

        args = fiberflat.parse(optarray)

        if rank == 0:
            fiberflat.main(args)
    
    elif step == 'sky':
        
        frm = []
        flat = []
        for input in node['in']:
            inode = grph[input]
            if inode['type'] == 'frame':
                frm.append(input)
            elif inode['type'] == 'fiberflat':
                flat.append(input)
        if len(frm) != 1:
            raise RuntimeError("sky needs exactly one frame file")
        if len(flat) != 1:
            raise RuntimeError("sky needs exactly one fiberflat file")

        framefile = graph_path_frame(proddir, frm[0])
        flatfile = graph_path_fiberflat(proddir, flat[0])
        outfile = graph_path_sky(proddir, name)
        qafile, qafig = qa_path(outfile)

        options = {}
        options['infile'] = framefile
        options['fiberflat'] = flatfile
        options['qafile'] = qafile
        options['qafig'] = qafig
        options['outfile'] = outfile
        options.update(opts)
        optarray = option_list(options)

        # at debug level, write out the equivalent commandline
        if rank == 0:
            com = ['RUN', 'desi_compute_sky']
            com.extend(optarray)
            log.debug(" ".join(com))

        args = skypkg.parse(optarray)

        if rank == 0:
            skypkg.main(args)
    
    elif step == 'stdstars':

        frm = []
        flat = []
        sky = []
        flatexp = None
        specgrph = None
        for input in node['in']:
            inode = grph[input]
            if inode['type'] == 'frame':
                frm.append(input)
                specgrph = inode['spec']
            elif inode['type'] == 'fiberflat':
                flat.append(input)
                flatexp = inode['id']
            elif inode['type'] == 'sky':
                sky.append(input)

        outfile = graph_path_stdstars(proddir, name)
        qafile, qafig = qa_path(outfile)
        
        framefiles = [graph_path_frame(proddir, x) for x in frm]
        skyfiles = [graph_path_sky(proddir, x) for x in sky]
        flatfiles = [graph_path_fiberflat(proddir, x) for x in flat]

        options = {}
        options['frames'] = framefiles
        options['skymodels'] = skyfiles
        options['fiberflats'] = flatfiles
        options['outfile'] = outfile
        options['ncpu'] = str(default_nproc)
        #- TODO: no QA for fitting standard stars yet
        
        options.update(opts)
        optarray = option_list(options)

        # at debug level, write out the equivalent commandline
        if rank == 0:
            com = ['RUN', 'desi_fit_stdstars']
            com.extend(optarray)
            log.debug(" ".join(com))

        args = stdstars.parse(optarray)

        if rank == 0:
            stdstars.main(args)
    
    elif step == 'fluxcal':

        frm = []
        flat = []
        sky = []
        star = []
        for input in node['in']:
            inode = grph[input]
            if inode['type'] == 'frame':
                frm.append(input)
            elif inode['type'] == 'fiberflat':
                flat.append(input)
            elif inode['type'] == 'sky':
                sky.append(input)
            elif inode['type'] == 'stdstars':
                star.append(input)
        if len(frm) != 1:
            raise RuntimeError("fluxcal needs exactly one frame file")
        if len(flat) != 1:
            raise RuntimeError("fluxcal needs exactly one fiberflat file")
        if len(sky) != 1:
            raise RuntimeError("fluxcal needs exactly one sky file")
        if len(star) != 1:
            raise RuntimeError("fluxcal needs exactly one star file")

        framefile = graph_path_frame(proddir, frm[0])
        flatfile = graph_path_fiberflat(proddir, flat[0])
        skyfile = graph_path_sky(proddir, sky[0])
        starfile = graph_path_stdstars(proddir, star[0])
        outfile = graph_path_calib(proddir, name)
        qafile, qafig = qa_path(outfile)

        options = {}
        options['infile'] = framefile
        options['fiberflat'] = flatfile
        options['qafile'] = qafile
        options['qafig'] = qafig
        options['sky'] = skyfile
        options['models'] = starfile
        options['outfile'] = outfile
        options.update(opts)
        optarray = option_list(options)

        # at debug level, write out the equivalent commandline
        if rank == 0:
            com = ['RUN', 'desi_compute_fluxcalibration']
            com.extend(optarray)
            log.debug(" ".join(com))

        args = fluxcal.parse(optarray)

        if rank == 0:
            fluxcal.main(args)
    
    elif step == 'procexp':
        
        frm = []
        flat = []
        sky = []
        cal = []
        for input in node['in']:
            inode = grph[input]
            if inode['type'] == 'frame':
                frm.append(input)
            elif inode['type'] == 'fiberflat':
                flat.append(input)
            elif inode['type'] == 'sky':
                sky.append(input)
            elif inode['type'] == 'calib':
                cal.append(input)
        if len(frm) != 1:
            raise RuntimeError("procexp needs exactly one frame file")
        if len(flat) != 1:
            raise RuntimeError("procexp needs exactly one fiberflat file")
        if len(sky) != 1:
            raise RuntimeError("procexp needs exactly one sky file")
        if len(cal) != 1:
            raise RuntimeError("procexp needs exactly one calib file")

        framefile = graph_path_frame(proddir, frm[0])
        flatfile = graph_path_fiberflat(proddir, flat[0])
        skyfile = graph_path_sky(proddir, sky[0])
        calfile = graph_path_calib(proddir, cal[0])
        outfile = graph_path_cframe(proddir, name)

        options = {}
        options['infile'] = framefile
        options['fiberflat'] = flatfile
        options['sky'] = skyfile
        options['calib'] = calfile
        options['outfile'] = outfile
        options.update(opts)
        optarray = option_list(options)

        # at debug level, write out the equivalent commandline
        if rank == 0:
            com = ['RUN', 'desi_process_exposure']
            com.extend(optarray)
            log.debug(" ".join(com))

        args = procexp.parse(optarray)

        if rank == 0:
            procexp.main(args)
    
    elif step == 'zfind':
        brick = node['brick']
        outfile = graph_path_zbest(proddir, name)
        qafile, qafig = qa_path(outfile)
        options = {}
        options['brick'] = brick
        options['outfile'] = outfile
        #- TODO: no QA for desi_zfind yet
        options.update(opts)
        optarray = option_list(options)

        # at debug level, write out the equivalent commandline
        if rank == 0:
            com = ['RUN', 'desi_zfind']
            com.extend(optarray)
            log.debug(" ".join(com))

        args = zfind.parse(optarray)
        zfind.main(args, comm=comm)

    else:
        raise RuntimeError("Unknown pipeline step {}".format(step))

    #sys.stdout.flush()
    if comm is not None:
        #print("proc {} hit runtask barrier".format(rank))
        #sys.stdout.flush()
        comm.barrier()
    #print("proc {} finish runtask".format(rank))
    #sys.stdout.flush()

    return


@contextmanager
def stdouterr_redirected(to=os.devnull, comm=None):
    '''
    Based on http://stackoverflow.com/questions/5081657

    import os

    with stdouterr_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    sys.stdout.flush()
    sys.stderr.flush()
    fd = sys.stdout.fileno()
    fde = sys.stderr.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd
        sys.stderr.close() # + implicit flush()
        os.dup2(to.fileno(), fde) # fd writes to 'to' file
        sys.stderr = os.fdopen(fde, 'w') # Python writes to fd
        # update desi logging to use new stdout
        log = get_logger()
        while len(log.handlers) > 0:
            h = log.handlers[0]
            log.removeHandler(h)
        # Add the current stdout.
        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(levelname)s:%(filename)s:%(lineno)s:%(funcName)s: %(message)s')
        ch.setFormatter(formatter)
        log.addHandler(ch)

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        pto = to
        if comm is None:
            with open(pto, 'w') as file:
                _redirect_stdout(to=file)
        else:
            pto = "{}_{}".format(to, comm.rank)
            with open(pto, 'w') as file:
                _redirect_stdout(to=file)
        try:
            if (comm is None) or (comm.rank == 0):
                log.info("Begin log redirection to {} at {}".format(to, time.asctime()))
            sys.stdout.flush()
            yield # allow code to be run with the redirected stdout
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different
            if comm is not None:
                # concatenate per-process files
                comm.barrier()
                if comm.rank == 0:
                    with open(to, 'w') as outfile:
                        for p in range(comm.size):
                            outfile.write("================= Process {} =================\n".format(p))
                            fname = "{}_{}".format(to, p)
                            with open(fname) as infile:
                                outfile.write(infile.read())
                            os.remove(fname)
                comm.barrier()

            if (comm is None) or (comm.rank == 0):
                log.info("End log redirection to {} at {}".format(to, time.asctime()))
            sys.stdout.flush()
            
    return


def run_step(step, rawdir, proddir, grph, opts, comm=None, taskproc=1):
    '''
    Run a whole single step of the pipeline.

    This function first takes the communicator and the requested processes
    per task and splits the communicator to form groups of processes of
    the desired size.  It then takes the full dependency graph and extracts 
    all the tasks for a given step.  These tasks are then distributed among
    the groups of processes.

    Each process group loops over its assigned tasks.  For each task, it
    redirects stdout/stderr to a per-task file and calls run_task().  If
    any process in the group throws an exception, then the traceback and
    all information (graph and options) needed to re-run the task are written
    to disk.

    After all process groups have finished, the state of the full graph is
    merged from all processes.  This way a failure of one process on one task
    will be propagated as a failed task to all processes.

    Args:
        step (str): the pipeline step to process.
        rawdir (str): the path to the raw data directory.
        proddir (str): the path to the production directory.
        grph (dict): the dependency graph.
        opts (dict): the global options.
        comm (mpi4py.Comm): the full communicator to use for whole step.
        taskproc (int): the number of processes to use for a single task.

    Returns:
        Nothing.
    '''
    log = get_logger()

    nproc = 1
    rank = 0
    if comm is not None:
        nproc = comm.size
        rank = comm.rank

    if taskproc > nproc:
        raise RuntimeError("cannot have {} processes per task with only {} processes".format(taskproc, nproc))

    # Get the tasks that need to be done for this step.  Mark all completed
    # tasks as done.

    tasks = None
    if rank == 0:
        # For this step, compute all the tasks that we need to do
        alltasks = []
        for name, nd in sorted(grph.items()):
            if nd['type'] in step_file_types[step]:
                alltasks.append(name)

        # For each task, prune if it is finished
        tasks = []
        for t in alltasks:
            if 'state' in grph[t]:
                if grph[t]['state'] != 'done':
                    tasks.append(t)
            else:
                tasks.append(t)

    if comm is not None:
        tasks = comm.bcast(tasks, root=0)
        grph = comm.bcast(grph, root=0)

    ntask = len(tasks)

    # Get the options for this step.

    options = opts[step]

    # Now every process has the full list of tasks.  If we have multiple
    # processes for each task, split the communicator.

    comm_group = comm
    comm_rank = None
    group = rank
    ngroup = nproc
    group_rank = 0
    if comm is not None:
        if taskproc > 1:
            ngroup = int(nproc / taskproc)
            group = int(rank / taskproc)
            group_rank = rank % taskproc
            comm_group = comm.Split(color=group, key=group_rank)
            comm_rank = comm.Split(color=group_rank, key=group)
        else:
            comm_group = None
            comm_rank = comm

    # Now we divide up the tasks among the groups of processes as
    # equally as possible.

    group_ntask = 0
    group_firsttask = 0

    if group < ngroup:
        # only assign tasks to whole groups
        if ntask < ngroup:
            if group < ntask:
                group_ntask = 1
                group_firsttask = group
            else:
                group_ntask = 0
        else:
            if step == 'zfind':
                # We load balance the bricks across process groups based
                # on the number of targets per brick.  All bricks with 
                # < taskproc targets are weighted the same.

                if ntask <= ngroup:
                    # distribute uniform in this case
                    group_firsttask, group_ntask = dist_uniform(ntask, ngroup, group)
                else:
                    bricksizes = [ grph[x]['ntarget'] for x in tasks ]
                    worksizes = [ taskproc if (x < taskproc) else x for x in bricksizes ]

                    if rank == 0:
                        log.debug("zfind {} groups".format(ngroup))
                        workstr = ""
                        for w in worksizes:
                            workstr = "{}{} ".format(workstr, w)
                        log.debug("zfind work sizes = {}".format(workstr))

                    group_firsttask, group_ntask = dist_discrete(worksizes, ngroup, group)

                if group_rank == 0:
                    worksum = np.sum(worksizes[group_firsttask:group_firsttask+group_ntask])
                    log.debug("group {} has tasks {}-{} sum = {}".format(group, group_firsttask, group_firsttask+group_ntask-1, worksum))

            else:
                group_firsttask, group_ntask = dist_uniform(ntask, ngroup, group)

    # every group goes and does its tasks...

    faildir = os.path.join(proddir, 'run', 'failed')
    logdir = os.path.join(proddir, 'run', 'logs')

    failcount = 0
    group_failcount = 0

    if group_ntask > 0:
        for t in range(group_firsttask, group_firsttask + group_ntask):
            # if group_rank == 0:
            #     print("group {} starting task {}".format(group, tasks[t]))
            #     sys.stdout.flush()
            # slice out just the graph for this task

            (night, gname) = graph_name_split(tasks[t])
            nfaildir = os.path.join(faildir, night)
            nlogdir = os.path.join(logdir, night)

            tgraph = graph_slice(grph, names=[tasks[t]], deps=True)
            ffile = os.path.join(nfaildir, "{}_{}.yaml".format(step, tasks[t]))
            
            # For this task, we will temporarily redirect stdout and stderr
            # to a task-specific log file.

            tasklog = os.path.join(nlogdir, "{}.log".format(gname))
            if group_rank == 0:
                if os.path.isfile(tasklog):
                    os.remove(tasklog)
            if comm_group is not None:
                comm_group.barrier()

            with stdouterr_redirected(to=tasklog, comm=comm_group):
                try:
                    # if the step previously failed, clear that file now
                    if group_rank == 0:
                        if os.path.isfile(ffile):
                            os.remove(ffile)
                    # if group_rank == 0:
                    #     print("group {} runtask {}".format(group, tasks[t]))
                    #     sys.stdout.flush()
                    log.debug("running step {} task {} (group {}/{} with {} processes)".format(step, tasks[t], (group+1), ngroup, taskproc))
                    run_task(step, rawdir, proddir, tgraph, options, comm=comm_group)
                    # mark step as done in our group's graph
                    # if group_rank == 0:
                    #     print("group {} start graph_mark {}".format(group, tasks[t]))
                    #     sys.stdout.flush()
                    graph_mark(grph, tasks[t], state='done', descend=False)
                    # if group_rank == 0:
                    #     print("group {} end graph_mark {}".format(group, tasks[t]))
                    #     sys.stdout.flush()
                except:
                    # The task threw an exception.  We want to dump all information
                    # that will be needed to re-run the run_task() function on just
                    # this task.
                    group_failcount += 1
                    msg = "FAILED: step {} task {} (group {}/{} with {} processes)".format(step, tasks[t], (group+1), ngroup, taskproc)
                    log.error(msg)
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                    log.error(''.join(lines))
                    fyml = {}
                    fyml['step'] = step
                    fyml['rawdir'] = rawdir
                    fyml['proddir'] = proddir
                    fyml['task'] = tasks[t]
                    fyml['graph'] = tgraph
                    fyml['opts'] = options
                    fyml['procs'] = taskproc
                    if not os.path.isfile(ffile):
                        log.error('Dumping yaml graph to '+ffile)
                        # we are the first process to hit this
                        with open(ffile, 'w') as f:
                            yaml.dump(fyml, f, default_flow_style=False)
                    # mark the step as failed in our group's local graph
                    graph_mark(grph, tasks[t], state='fail', descend=True)

        if comm_group is not None:
            comm_group.barrier()
            group_failcount = comm_group.allreduce(group_failcount)

    # Now we take the graphs from all groups and merge their states

    failcount = group_failcount
    #sys.stdout.flush()
    if comm is not None:
        # print("proc {} hit merge barrier".format(rank))
        # sys.stdout.flush()
        # comm.barrier()
        if group_rank == 0:
            # print("proc {} joining merge".format(rank))
            # sys.stdout.flush()
            graph_merge_state(grph, comm=comm_rank)
            failcount = comm_rank.allreduce(failcount)
        if comm_group is not None:
            # print("proc {} joining bcast".format(rank))
            # sys.stdout.flush()
            grph = comm_group.bcast(grph, root=0)
            failcount = comm_group.bcast(failcount, root=0)

    return grph, ntask, failcount


def retry_task(failpath, newopts=None):
    '''
    Attempt to re-run a failed task.

    This takes the path to a yaml file containing the information about a
    failed task (such a file is written by run_step() when a task fails).
    This yaml file contains the truncated dependecy graph for the single
    task, as well as the options that were used when running the task.
    It also contains information about the number of processes that were
    being used.

    This function attempts to load mpi4py and use the MPI.COMM_WORLD
    communicator to re-run the task.  If COMM_WORLD has a different number
    of processes than were originally used, a warning is printed.  A
    warning is also printed if the options are being overridden.

    If the task completes successfully, the failed yaml file is deleted.

    Args:
        failpath (str): the path to the failure yaml file.
        newopts (dict): the dictionary of options to use in place of the
            original ones.

    Returns:
        Nothing.
    '''

    log = get_logger()

    if not os.path.isfile(failpath):
        raise RuntimeError("failure yaml file {} does not exist".format(failpath))

    fyml = None
    with open(failpath, 'r') as f:
        fyml = yaml.load(f)

    step = fyml['step']
    rawdir = fyml['rawdir']
    proddir = fyml['proddir']
    name = fyml['task']
    grph = fyml['graph']
    origopts = fyml['opts']
    nproc = fyml['procs']

    comm = None
    rank = 0

    if nproc > 1:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        nworld = comm.size
        rank = comm.rank
        if nworld != nproc:
            if rank == 0:
                log.warning("WARNING: original task was run with {} processes, re-running with {} instead".format(nproc, nworld))

    opts = origopts
    if newopts is not None:
        log.warning("WARNING: overriding original options")
        opts = newopts

    logdir = os.path.join(proddir, 'run', 'logs')
    (night, gname) = graph_name_split(name)

    nlogdir = os.path.join(logdir, night)
            
    # For this task, we will temporarily redirect stdout and stderr
    # to a task-specific log file.

    tasklog = os.path.join(nlogdir, "{}.log".format(gname))
    if rank == 0:
        if os.path.isfile(tasklog):
            os.remove(tasklog)
    if comm is not None:
        comm.barrier()

    failcount = 0

    with stdouterr_redirected(to=tasklog, comm=comm):
        try:
            log.debug("re-trying step {}, task {} with {} processes".format(step, name, nworld))
            run_task(step, rawdir, proddir, grph, opts, comm=comm)
        except:
            failcount += 1
            msg = "FAILED: step {} task {} process {}".format(step, name, rank)
            log.error(msg)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            log.error(''.join(lines))
            
    if comm is not None:
        comm.barrier()
        failcount = comm.allreduce(failcount)

    if rank == 0:
        if failcount > 0:
            log.error("{} of {} processes raised an exception".format(failcount, nworld))
        else:
            # success, clear failure file now
            if os.path.isfile(failpath):
                os.remove(failpath)

    return


def run_steps(first, last, rawdir, proddir, spectrographs=None, nightstr=None, comm=None):
    '''
    Run multiple sequential pipeline steps.

    

    This function first takes the communicator and the requested processes
    per task and splits the communicator to form groups of processes of
    the desired size.  It then takes the full dependency graph and extracts 
    all the tasks for a given step.  These tasks are then distributed among
    the groups of processes.

    Each process group loops over its assigned tasks.  For each task, it
    redirects stdout/stderr to a per-task file and calls run_task().  If
    any process in the group throws an exception, then the traceback and
    all information (graph and options) needed to re-run the task are written
    to disk.

    After all process groups have finished, the state of the full graph is
    merged from all processes.  This way a failure of one process on one task
    will be propagated as a failed task to all processes.

    Args:
        step (str): the pipeline step to process.
        rawdir (str): the path to the raw data directory.
        proddir (str): the path to the production directory.
        grph (dict): the dependency graph.
        opts (dict): the global options.
        comm (mpi4py.Comm): the full communicator to use for whole step.
        taskproc (int): the number of processes to use for a single task.

    Returns:
        Nothing.
    '''
    log = get_logger()

    rank = 0
    nproc = 1
    if comm is not None:
        rank = comm.rank
        nproc = comm.size

    # get the full graph

    grph = None
    if rank == 0:
        grph = graph_read_prod(proddir, nightstr=nightstr, spectrographs=spectrographs)
        prod_state(rawdir, proddir, grph)
    if comm is not None:
        grph = comm.bcast(grph, root=0)

    # read run options from disk

    rundir = os.path.join(proddir, "run")
    optfile = os.path.join(rundir, "options.yaml")
    opts = None
    if rank == 0:
        opts = read_options(optfile)
    if comm is not None:
        opts = comm.bcast(opts, root=0)

    # compute the ordered list of steps to run

    firststep = None
    if first is None:
        firststep = 0
    else:
        s = 0
        for st in run_step_types:
            if st == first:
                firststep = s
            s += 1

    laststep = None
    if last is None:
        laststep = len(run_step_types)
    else:
        s = 1
        for st in run_step_types:
            if st == last:
                laststep = s
            s += 1

    if rank == 0:
        log.info("running steps {} to {}".format(run_step_types[firststep], run_step_types[laststep-1]))

    # Assign the desired number of processes per task

    steptaskproc = {}
    for st in run_step_types:
        steptaskproc[st] = 1

    steptaskproc['bootcalib'] = 1
    steptaskproc['specex'] = 20
    steptaskproc['psfcombine'] = 1
    steptaskproc['extract'] = 20
    steptaskproc['fiberflat'] = 1
    steptaskproc['sky'] = 1
    steptaskproc['stdstars'] = 1
    steptaskproc['fluxcal'] = 1
    steptaskproc['procexp'] = 1
    steptaskproc['zfind'] = 48

    jobid = None
    if rank == 0:
        if 'SLURM_JOBID' in os.environ:
            jobid = "slurm-{}".format(os.environ['SLURM_JOBID'])
        else:
            jobid = os.getpid()

    statefile = None
    statedot = None
    if rank == 0:
        stateroot = "state_{}-{}_{}".format(run_step_types[firststep], run_step_types[laststep-1], jobid)
        statefile = os.path.join(rundir, "{}.yaml".format(stateroot))
        statedot = os.path.join(rundir, "{}.dot".format(stateroot))

    # Mark our steps as in progress

    for st in range(firststep, laststep):
        for name, nd in grph.items():
            if nd['type'] in step_file_types[run_step_types[st]]:
                if 'state' in nd:
                    if nd['state'] != 'done':
                        graph_mark(grph, name, 'wait')
                else:
                    graph_mark(grph, name, 'wait')

    if rank == 0:
        graph_write(statefile, grph)
        with open(statedot, 'w') as f:
            graph_dot(grph, f)

    # Run the steps.  Each step updates the graph in place to track
    # the state of all nodes.

    for st in range(firststep, laststep):
        runfile = None
        if rank == 0:
            log.info("starting step {} at {}".format(run_step_types[st], time.asctime()))
        taskproc = steptaskproc[run_step_types[st]]
        if taskproc > nproc:
            taskproc = nproc

        grph, ntask, failtask = run_step(run_step_types[st], rawdir, proddir, grph, opts, comm=comm, taskproc=taskproc)
        if rank == 0:
            log.info("  {} total tasks, {} failures".format(ntask, failtask))

        if ntask == failtask:
            if rank == 0:
                log.info("step {}: all tasks failed, quiting at {}".format(run_step_types[st], time.asctime()))
            break

        if comm is not None:
            comm.barrier()
        if rank == 0:
            log.info("completed step {} at {}".format(run_step_types[st], time.asctime()))
        if rank == 0:
            graph_write(statefile, grph)
            with open(statedot, 'w') as f:
                graph_dot(grph, f)
            log.info("finished steps {} to {}".format(run_step_types[firststep], run_step_types[laststep-1]))

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


def shell_job(path, logroot, envsetup, desisetup, commands, comrun="", mpiprocs=1, threads=1):
    with open(path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("now=`date +%Y%m%d-%H:%M:%S`\n")
        f.write('export STARTTIME=${now}\n')
        f.write("log={}_${{now}}.log\n\n".format(logroot))
        for com in envsetup:
            f.write("{}\n".format(com))
        f.write("\n")
        f.write("source {}\n\n".format(desisetup))
        f.write("export OMP_NUM_THREADS={}\n\n".format(threads))
        run = ""
        if comrun != "":
            run = "{} {}".format(comrun, mpiprocs)
        for com in commands:
            executable = com.split(' ')[0]
            # f.write("which {}\n".format(executable))
            f.write("echo logging to ${log}\n")
            f.write("time {} {} >>${{log}} 2>&1\n\n".format(run, com))
    mode = stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    os.chmod(path, mode)
    return


def nersc_job(path, logroot, envsetup, desisetup, commands, nodes=1, \
    nodeproc=1, minutes=10, multisrun=False, openmp=False, multiproc=False, \
    queue='debug', jobname='desipipe'):
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
        if queue == 'debug':
            f.write("#SBATCH --partition=debug\n")
        else:
            f.write("#SBATCH --partition=regular\n")
        f.write("#SBATCH --account=desi\n")
        f.write("#SBATCH --nodes={}\n".format(totalnodes))
        f.write("#SBATCH --time={}\n".format(timestr))
        f.write("#SBATCH --job-name={}\n".format(jobname))
        f.write("#SBATCH --output={}_%j.log\n".format(logroot))
        f.write("#SBATCH --export=NONE\n\n")
        f.write("echo Starting slurm script at `date`\n\n")
        for com in envsetup:
            f.write("{}\n".format(com))
        f.write("\n")
        f.write("source {}\n\n".format(desisetup))
        f.write("# Set TMPDIR to be on the ramdisk\n")
        f.write("export TMPDIR=/dev/shm\n\n")
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
            comlist = com.split(' ')
            executable = comlist.pop(0)
            f.write("ex=`which {}`\n".format(executable))
            f.write("app=\"${ex}.app\"\n")
            f.write("if [ -x ${app} ]; then\n")
            f.write("  if [ ${ex} -nt ${app} ]; then\n")
            f.write("    app=${ex}\n")
            f.write("  fi\n")
            f.write("else\n")
            f.write("  app=${ex}\n")
            f.write("fi\n")
            f.write("echo calling desi_pipe_run at `date`\n\n")
            f.write('export STARTTIME=`date +%Y%m%d-%H:%M:%S`\n')
            f.write("echo ${{run}} ${{app}} {}\n".format(' '.join(comlist)))
            f.write("time ${{run}} ${{app}} {} >>${{log}} 2>&1".format(' '.join(comlist)))
            if multisrun:
                f.write(" &")
            f.write("\n\n")
        if multisrun:
            f.write("wait\n\n")

        f.write("echo done with slurm script at `date`\n")

    return

