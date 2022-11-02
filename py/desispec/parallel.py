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
import warnings

import numpy as np

import desiutil.log
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

# MPI environment availability
def use_mpi():
    """Return whether we can use MPI."""
    if ("NERSC_HOST" in os.environ) and ("SLURM_JOB_NAME" not in os.environ):
        return False
    else:
        try:
            import mpi4py.MPI as MPI
            return True
        except ImportError:
            return False

    return False

# Functions for static distribution

def dist_uniform(nwork, nworkers, id=None):
    """
    Statically distribute some number of items among workers.

    This assumes that each item has equal weight, and that they
    should be divided into contiguous blocks of items and
    assigned to workers in rank order.

    This function returns the index of the first item and the
    number of items for the specified worker ID, or the information
    for all workers.

    Args:
        nwork (int): the number of things to distribute.
        nworkers (int): the number of workers.
        id (int): optionally return just the tuple associated with
            this worker

    Returns (tuple):
        A tuple of ints, containing the first item and number
        of items.  If id=None, then return a list containing the tuple
        for all workers.

    """
    dist = []

    for i in range(nworkers):
        ntask = nwork // nworkers
        firsttask = 0
        leftover = nwork % nworkers
        if i < leftover:
            ntask += 1
            firsttask = i * ntask
        else:
            firsttask = ((ntask + 1) * leftover) + (ntask * (i - leftover))
        dist.append( (firsttask, ntask) )

    if id is not None:
        if id < len(dist):
            return dist[id]
        else:
            raise RuntimeError("worker ID is out of range")
    else:
        return dist


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

def dist_discrete_all(worksizes, nworkers, power=1.0):
    """Distribute indivisible blocks of items between groups.

    Given some contiguous blocks of items which cannot be
    subdivided, distribute these blocks to the specified
    number of groups in a way which minimizes the maximum
    total items given to any group.  Optionally weight the
    blocks by a power of their size when computing the
    distribution.

    This is effectively the "Painter"s Partition Problem".

    Args:
        worksizes (list): The sizes of the indivisible blocks.
        nworkers (int): The number of workers.
        pow (float): The power to use for weighting

    Returns:
        list of length nworkers; each element is a list of indices of
            worksizes assigned to that worker.
    """
    chunks = np.array(worksizes, dtype=np.int64)
    weights = np.power(chunks.astype(np.float64), power)
    max_per_proc = float(distribute_partition(weights.astype(np.int64), nworkers))

    if len(worksizes) < nworkers:
        warnings.warn("Too many workers ({}) for {} work items.  Some workers"
            " idle.".format(nworkers, len(worksizes)), RuntimeWarning)

    dist = []

    off = 0
    curweight = 0.0
    for cur in range(0, weights.shape[0]):
        if curweight + weights[cur] > max_per_proc:
            ## dist.append( (off, cur-off) )
            dist.append( list(range(off, cur)) )
            curweight = weights[cur]
            off = cur
        else:
            curweight += weights[cur]

    if (nworkers - len(dist) > 0):
        # There are still workers remaining.
        if (nworkers - len(dist) == 1):
            # There is exactly one worker remaining.  Assign it the rest of
            # the work
            if weights.shape[0] > off:
                ## dist.append((off, weights.shape[0] - off))
                dist.append( list(range(off, weights.shape[0])) )
            else:
                ## dist.append((off, 0))
                dist.append( list() )
        else:
            # We have multiple remaining workers.  This can happen in cases
            # of extreme load imbalance.  Distribute the remaining work
            # uniformly.
            remain = dist_uniform(weights.shape[0]-off, nworkers-len(dist))
            for i in range(nworkers - len(dist)):
                ## dist.append( (off + remain[i][0], remain[i][1]) )
                ioff = off + remain[i][0]
                n = remain[i][1]
                dist.append( list(range(ioff, ioff+n)) )

    if len(dist) < nworkers:
        # The load imbalance was really bad.  Just warn and assign the
        # remaining workers zero items.
        warnings.warn("Load imbalance.  Some work items are so large that not all workers have items.", RuntimeWarning)
        for i in range(len(dist), nworkers):
            ## dist.append( (off, 0) )
            dist.append( list() )

    return dist


def dist_discrete(worksizes, nworkers, workerid, power=1.0):
    """Distribute indivisible blocks of items between groups.

    Given some contiguous blocks of items which cannot be
    subdivided, distribute these blocks to the specified
    number of groups in a way which minimizes the maximum
    total items given to any group.  Optionally weight the
    blocks by a power of their size when computing the
    distribution.

    This is effectively the "Painter"s Partition Problem".

    Args:
        worksizes (list): The sizes of the indivisible blocks.
        nworkers (int): The number of workers.
        workerid (int): The worker ID whose range should be returned.
        power (float): The power to use for weighting

    Returns:
        A tuple.  The first element of the tuple is the first
        block assigned to the worker ID, and the second element
        is the number of blocks assigned to the worker.
    """
    allworkers = dist_discrete_all(worksizes, nworkers, power=power)
    return allworkers[workerid]

def weighted_partition(weights, n, groups_per_node=None):
    '''
    Partition `weights` into `n` groups with approximately same sum(weights)

    Args:
        weights: array-like weights
        n: number of groups

    Returns list of lists of indices of weights for each group

    Notes:
        compared to `dist_discrete_all`, this function allows non-contiguous
        items to be grouped together which allows better balancing.
    '''
    #- sumweights will track the sum of the weights that have been assigned
    #- to each group so far
    sumweights = np.zeros(n, dtype=float)

    #- Initialize list of lists of indices for each group
    groups = list()
    for i in range(n):
        groups.append(list())

    #- Assign items from highest weight to lowest weight, always assigning
    #- to whichever group currently has the fewest weights
    weights = np.asarray(weights)
    for i in np.argsort(-weights):
        j = np.argmin(sumweights)
        groups[j].append(i)
        sumweights[j] += weights[i]

    assert len(groups) == n

    #- Reorder groups to spread out large items across different nodes
    #- NOTE: this isn't perfect, e.g. study
    #-   weighted_partition(np.arange(12), 6, groups_per_node=2)
    #- even better would be to zigzag back and forth across the nodes instead
    #- of loop across the nodes.
    if groups_per_node is None:
        return groups
    else:
        distributed_groups = [None,] * len(groups)
        num_nodes = (n + groups_per_node - 1) // groups_per_node
        i = 0
        for noderank in range(groups_per_node):
            for inode in range(num_nodes):
                j = inode*groups_per_node + noderank
                if i < n and j < n:
                    distributed_groups[j] = groups[i]
                    i += 1

        #- do a final check that all groups were assigned
        for i in range(len(distributed_groups)):
            assert distributed_groups[i] is not None, 'group {} not set'.format(i)

        return distributed_groups

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
    nproc = 1
    rank = 0
    if comm is not None:
        nproc = comm.size
        rank = comm.rank

    # The currently active POSIX file descriptors
    fd_out = sys.stdout.fileno()
    fd_err = sys.stderr.fileno()

    # The DESI loggers.
    desi_loggers = desiutil.log._desiutil_log_root

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
        for name, logger in desi_loggers.items():
            hformat = None
            while len(logger.handlers) > 0:
                h = logger.handlers[0]
                if hformat is None:
                    hformat = h.formatter._fmt
                logger.removeHandler(h)
            # Add the current stdout.
            ch = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(hformat, datefmt='%Y-%m-%dT%H:%M:%S')
            ch.setFormatter(formatter)
            logger.addHandler(ch)

    # redirect both stdout and stderr to the same file

    if to is None:
        to = "/dev/null"

    if rank == 0:
        log = get_logger()
        log.info("Begin log redirection to {} at {}".format(to, time.asctime()))

    # Save the original file descriptors so we can restore them later
    saved_fd_out = os.dup(fd_out)
    saved_fd_err = os.dup(fd_err)

    try:
        pto = to
        if to != "/dev/null":
            pto = "{}_{}".format(to, rank)

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
        # flush python handles for good measure
        sys.stdout.flush()
        sys.stderr.flush()

        # restore old stdout and stderr
        _redirect(out_to=saved_fd_out, err_to=saved_fd_err)

        if nproc > 1:
            comm.barrier()

        # concatenate per-process files
        if rank == 0 and to != "/dev/null":
            with open(to, "w") as outfile:
                for p in range(nproc):
                    outfile.write("================ Start of Process {} ================\n".format(p))
                    fname = "{}_{}".format(to, p)
                    with open(fname) as infile:
                        outfile.write(infile.read())
                    outfile.write("================= End of Process {} =================\n\n".format(p))
                    os.remove(fname)

        if nproc > 1:
            comm.barrier()

        if rank == 0:
            log = get_logger()
            log.info("End log redirection to {} at {}".format(to, time.asctime()))

        # flush python handles for good measure
        sys.stdout.flush()
        sys.stderr.flush()

    return


def take_turns(comm, at_a_time, func, *args, **kwargs):
    """
    Processes call a function in groups.

    Any extra positional and keyword arguments are passed to the function.

    Args:
        comm:  mpi4py.MPI.Comm or None.
        at_a_time (int): the maximum number of processes to run at a time.
        func: the function to call.

    Returns:
        The return value on each process is the return value of the function.

    """
    if comm is None:
        # just run the function
        return func(*args, **kwargs)

    nproc = comm.size
    rank = comm.rank

    if at_a_time >= nproc:
        # every process just runs at once
        return func(*args, **kwargs)

    # we split the communicator to enforce a fixed number of processes
    # running at once.

    groupsize = nproc // at_a_time
    if nproc % at_a_time != 0:
        groupsize += 1

    group = rank // groupsize
    group_rank =  rank % groupsize

    comm_group = comm.Split(color=group, key=group_rank)

    # within each group, processes take turns

    ret = None
    for p in range(comm_group.size):
        if p == comm_group.rank:
            ret = func(*args, **kwargs)
        comm_group.barrier()

    return ret
