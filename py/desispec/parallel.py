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
import io
from contextlib import contextmanager
import logging
import ctypes

import numpy as np

from desiutil.log import get_logger


# C file descriptors for stderr and stdout, used in redirection
# context manager.

libc = ctypes.CDLL(None)
c_stdout = None
c_stderr = None
try:
    # Linux systems
    c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')
    c_stderr = ctypes.c_void_p.in_dll(libc, 'stderr')
except:
    try:
        # Darwin
        c_stdout = ctypes.c_void_p.in_dll(libc, '__stdoutp')
        c_stderr = ctypes.c_void_p.in_dll(libc, '__stdoutp')
    except:
        # Neither!
        pass

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


def dist_balanced(nwork, maxworkers):
    """
    Distribute items between a flexible number of workers.

    This assumes that each item has equal weight, and that they
    should be divided into contiguous blocks of items and
    assigned to workers in rank order.

    If the number of workers is less than roughly sqrt(nwork), then
    we do not reduce the number of workers and the result is the same
    as the dist_uniform function.  If there are more workers than this,
    then the number of workers is reduced until all workers have close
    to the same number of tasks.

    Args:
        nwork (int): The number of work items.
        maxworkers (int): The maximum number of workers.  The actual
            number may be less than this.

    Returns:
        A list of tuples, one for each worker.  The first element 
        of the tuple is the first item assigned to the worker, 
        and the second element is the number of items assigned to 
        the worker.
    """
    workers = maxworkers

    if nwork < workers:
        workers = nwork
    else:
        ntask = nwork // workers
        leftover = nwork % workers
        while (leftover != 0) and (leftover + ntask < workers):
            workers -= 1
            ntask = nwork // workers
            leftover = nwork % workers
    
    ret = []
    for w in range(workers):
        wfirst = None
        wtasks = None
        if w < leftover:
            wtasks = ntask + 1
            wfirst = w * ntask
        else:
            wtasks = ntask
            wfirst = ((ntask + 1) * leftover) + (ntask * (w - leftover))
        ret.append( (wfirst, wtasks) )

    return ret


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
def stdouterr_redirected(to=None, comm=None):
    """
    Redirect stdout and stderr to a file.

    The general technique is based on:

    http://stackoverflow.com/questions/5081657
    http://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/

    One difference here is that each process in the communicator
    redirects to a different temporary file, and the upon exit
    from the context the rank zero process concatenates these
    in order to the file result.

    Args:
        to (str): The output file name.
        comm (mpi4py.MPI.Comm): The optional MPI communicator.
    """

    # The currently active POSIX file descriptors
    fd_out = sys.stdout.fileno()
    fd_err = sys.stderr.fileno()

    # The DESI logger
    log = get_logger()

    def _redirect(out_to, err_to):

        # Flush the C-level buffers
        if c_stdout is not None:
            libc.fflush(c_stdout)
        if c_stderr is not None:
            libc.fflush(c_stderr)

        # This closes the python file handles, and marks the POSIX
        # file descriptors for garbage collection- UNLESS those
        # are the special file descriptors for stderr/stdout.
        sys.stdout.close()
        sys.stderr.close()

        # Close fd_out/fd_err if they are open, and copy the
        # input file descriptors to these.
        os.dup2(out_to, fd_out)
        os.dup2(err_to, fd_err)

        # Create a new sys.stdout / sys.stderr that points to the
        # redirected POSIX file descriptors.  In Python 3, these
        # are actually higher level IO objects.
        if sys.version_info[0] < 3:
            sys.stdout = os.fdopen(fd_out, "wb")
            sys.stderr = os.fdopen(fd_err, "wb")
        else:
            # Python 3 case
            sys.stdout = io.TextIOWrapper(os.fdopen(fd_out, 'wb'))
            sys.stderr = io.TextIOWrapper(os.fdopen(fd_err, 'wb'))

        # update DESI logging to use new stdout
        while len(log.handlers) > 0:
            h = log.handlers[0]
            log.removeHandler(h)
        # Add the current stdout.
        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(levelname)s:%(filename)s:%(lineno)s:%(funcName)s: %(message)s")
        ch.setFormatter(formatter)
        log.addHandler(ch)

    # redirect both stdout and stderr to the same file

    if to is None:
        to = "/dev/null"

    if (comm is None) or (comm.rank == 0):
        log.debug("Begin log redirection to {} at {}".format(to, time.asctime()))

    # Save the original file descriptors so we can restore them later
    saved_fd_out = os.dup(fd_out)
    saved_fd_err = os.dup(fd_err)

    try:
        pto = to
        if comm is not None:
            if to != "/dev/null":
                pto = "{}_{}".format(to, comm.rank)

        # open python file, which creates low-level POSIX file
        # descriptor.
        file = open(pto, "w")

        # redirect stdout/stderr to this new file descriptor.
        _redirect(out_to=file.fileno(), err_to=file.fileno())

        yield # allow code to be run with the redirected output

        # close python file handle, which will mark POSIX file
        # descriptor for garbage collection.  That is fine since
        # we are about to overwrite those in the finally clause.
        file.close()

    finally:
        # restore old stdout and stderr
        _redirect(out_to=saved_fd_out, err_to=saved_fd_err)

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

        # flush python handles for good measure
        sys.stdout.flush()
        sys.stderr.flush()

    return
