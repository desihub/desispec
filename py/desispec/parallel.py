"""
desispec.parallel
==================

Helper functions and classes for dealing with parallelization
and related topics.
"""
from __future__ import print_function, absolute_import, division

import os
import sys
import time
from contextlib import contextmanager
import logging

import numpy as np

from .log import get_logger


# Multiprocessing environment setup

default_nproc = None
"""Default number of multiprocessing processes. Set globally on first import."""

if "SLURM_CPUS_PER_TASK" in os.environ:
    default_nproc = int(os.environ["SLURM_CPUS_PER_TASK"])
else:
    import multiprocessing as _mp
    default_nproc = max(1, _mp.cpu_count() // 2)


# Functions for static distribution

def dist_uniform(nwork, workers, id):
    """
    Statically distribute some number of items among workers.

    This assumes that each item has equal weight, and that they
    should be divided into contiguous blocks of items and
    assigned to workers in rank order.

    This function returns the index of the first item and the
    number of items for the specified worker ID.

    Args:
        nwork (int): the number of things to distribute.
        workers (int): the number of workers.

    Returns (tuple):
        A tuple of ints, containing the first item and number
        of items.
    """

    ntask = 0
    firsttask = 0

    # if ID is out of range, ignore it
    if id < workers:
        if nwork < workers:
            if id < nwork:
                ntask = 1
                firsttask = id
        else:
            ntask = int(nwork / workers)
            leftover = nwork % workers
            if id < leftover:
                ntask += 1
                firsttask = id * ntask
            else:
                firsttask = ((ntask + 1) * leftover) + (ntask * (id - leftover))
    return (firsttask, ntask)


def distribute_required_groups(A, max_per_group):
    ngroup = 1
    total = 0
    for i in range(A.shape[0]):
        total += A[i]
        if total > max_per_group:
            total = A[i]
            ngroup += 1
    return ngroup

def distribute_partition(A, k):
    low = np.max(A)
    high = np.sum(A)
    while low < high:
        mid = low + int((high - low) / 2)
        required = distribute_required_groups(A, mid)
        if required <= k:
            high = mid
        else:
            low = mid + 1
    return low

def dist_discrete(worksizes, workers, id, pow=1.0):
    """
    Distribute indivisible blocks of items between groups.

    Given some contiguous blocks of items which cannot be 
    subdivided, distribute these blocks to the specified
    number of groups in a way which minimizes the maximum
    total items given to any group.  Optionally weight the
    blocks by a power of their size when computing the
    distribution.

    This is effectively the "Painter"s Partition Problem".

    Args:
        worksizes (list): The sizes of the indivisible blocks.
        workers (int): The number of workers.
        id (int): The worker ID whose range should be returned.
        pow (float): The power to use for weighting

    Returns:
        A tuple.  The first element of the tuple is the first 
        block assigned to the worker ID, and the second element 
        is the number of blocks assigned to the worker.
    """
    chunks = np.array(worksizes, dtype=np.int64)
    weights = np.power(chunks.astype(np.float64), pow)
    max_per_proc = float(distribute_partition(weights.astype(np.int64), workers))

    target = np.sum(weights) / workers

    dist = []

    off = 0
    curweight = 0.0
    proc = 0
    for cur in range(0, weights.shape[0]):
        if curweight + weights[cur] > max_per_proc:
            dist.append( (off, cur-off) )
            over = curweight - target
            curweight = weights[cur] + over
            off = cur
            proc += 1
        else:
            curweight += weights[cur]

    dist.append( (off, weights.shape[0]-off) )

    if len(dist) != workers:
        raise RuntimeError("Number of distributed groups different than number requested")

    return dist[id]



@contextmanager
def stdouterr_redirected(to=os.devnull, comm=None):
    """
    Redirect stdout and stderr to a file.

    The general technique is based on:

    http://stackoverflow.com/questions/5081657

    One difference here is that each process in the communicator
    redirects to a different temporary file, and the upon exit
    from the context the rank zero process concatenates these
    in order to the file result.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    fd = sys.stdout.fileno()
    fde = sys.stderr.fileno()

    log = get_logger()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(out_to, err_to):
        sys.stdout.close() # + implicit flush()
        os.dup2(out_to.fileno(), fd) # fd writes to "to" file
        sys.stdout = os.fdopen(fd, "w") # Python writes to fd
        
        sys.stderr.close() # + implicit flush()
        os.dup2(err_to.fileno(), fde) # fd writes to "to" file
        sys.stderr = os.fdopen(fde, "w") # Python writes to fd
        
        # update desi logging to use new stdout
        while len(log.handlers) > 0:
            h = log.handlers[0]
            log.removeHandler(h)
        # Add the current stdout.
        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(levelname)s:%(filename)s:%(lineno)s:%(funcName)s: %(message)s")
        ch.setFormatter(formatter)
        log.addHandler(ch)

    # redirect both stdout and stderr to the same file

    if (comm is None) or (comm.rank == 0):
        log.debug("Begin log redirection to {} at {}".format(to, time.asctime()))
    sys.stdout.flush()
    sys.stderr.flush()
    pto = to
    if comm is None:
        with open(pto, "w") as file:
            _redirect_stdout(out_to=file, err_to=file)
    else:
        pto = "{}_{}".format(to, comm.rank)
        with open(pto, "w") as file:
            _redirect_stdout(out_to=file, err_to=file)

    old_stdout = os.fdopen(os.dup(fd), "w")
    old_stderr = os.fdopen(os.dup(fde), "w")

    try:
        yield # allow code to be run with the redirected stdout
    finally:
        sys.stdout.flush()
        sys.stderr.flush()

        # restore old stdout and stderr

        _redirect_stdout(out_to=old_stdout, err_to=old_stderr)

        if comm is not None:
            # concatenate per-process files
            comm.barrier()
            if comm.rank == 0:
                with open(to, "w") as outfile:
                    for p in range(comm.size):
                        outfile.write("================= Process {} =================\n".format(p))
                        fname = "{}_{}".format(to, p)
                        with open(fname) as infile:
                            outfile.write(infile.read())
                        os.remove(fname)
            comm.barrier()

        if (comm is None) or (comm.rank == 0):
            log.debug("End log redirection to {} at {}".format(to, time.asctime()))
        sys.stdout.flush()
        sys.stderr.flush()
            
    return
