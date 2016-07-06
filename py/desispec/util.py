"""
Utility functions for desispec
"""

import os
import numpy as np

#- Default number of processes to use for multiprocessing
if 'SLURM_CPUS_PER_TASK' in os.environ.keys():
    default_nproc = int(os.environ['SLURM_CPUS_PER_TASK'])
else:
    import multiprocessing as _mp
    default_nproc = max(1, _mp.cpu_count() // 2)


# Distribute some number of things among some number
# of workers as evenly as possible.

def dist_uniform(nwork, workers, id):
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


# This is effectively the "Painter's Partition Problem".

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

    Args:
        worksizes (list): The sizes of the indivisible blocks.
        workers (int): The number of workers.
        id (int): The worker ID whose range should be returned.
        pow (float): The power to use for weighting

    Returns:
        A tuple.  The first element of the tuple is the first 
        item assigned to the worker ID, and the second element 
        is the number of items assigned to the worker.
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


def mask32(mask):
    '''
    Return an input mask as unsigned 32-bit
    
    Raises ValueError if 64-bit input can't be cast to 32-bit without losing
    info (i.e. if it contains values > 2**32-1)
    '''
    if mask.dtype in (
        np.dtype('i4'),  np.dtype('u4'),
        np.dtype('>i4'), np.dtype('>u4'),
        np.dtype('<i4'), np.dtype('<u4'),
        ):
        if mask.dtype.isnative:
            return mask.view('u4')
        else:
            return mask.astype('u4')
            
    elif mask.dtype in (
        np.dtype('i8'),  np.dtype('u8'),
        np.dtype('>i8'), np.dtype('>u8'),
        np.dtype('<i8'), np.dtype('<u8'),
        ):
        if mask.dtype.isnative:
            mask64 = mask.view('u8')
        else:
            mask64 = mask.astype('i8')
        if np.any(mask64 > 2**32-1):
            raise ValueError("mask with values above 2**32-1 can't be cast to 32-bit")
        return np.asarray(mask, dtype='u4')
        
    elif mask.dtype in (
        np.dtype('bool'), np.dtype('bool8'),
        np.dtype('i2'),  np.dtype('u2'),
        np.dtype('>i2'), np.dtype('>u2'),
        np.dtype('<i2'), np.dtype('<u2'),
        np.dtype('i1'),  np.dtype('u1'),
        np.dtype('>i1'), np.dtype('>u1'),
        np.dtype('<i1'), np.dtype('<u1'),
        ):
        return np.asarray(mask, dtype='u4')
    else:
        raise ValueError("Can't cast dtype {} to unsigned 32-bit".format(mask.dtype))

def night2ymd(night):
    """
    parse night YEARMMDD string into tuple of integers (year, month, day)
    """
    assert isinstance(night, str), 'night is not a string'
    assert len(night) == 8, 'invalid YEARMMDD night string '+night
    
    year = int(night[0:4])
    month = int(night[4:6])
    day = int(night[6:8])
    if month < 1 or 12 < month:
        raise ValueError('YEARMMDD month should be 1-12, not {}'.format(month))
    if day < 1 or 31 < day:
        raise ValueError('YEARMMDD day should be 1-31, not {}'.format(day))
        
    return (year, month, day)
    
def ymd2night(year, month, day):
    """
    convert year, month, day integers into cannonical YEARMMDD night string
    """
    return "{:04d}{:02d}{:02d}".format(year, month, day)
    
def combine_ivar(ivar1, ivar2):
    """
    Returns the combined inverse variance of two inputs, making sure not to
    divide by 0 in the process.
    
    ivar1 and ivar2 may be scalar or ndarray but must have the same dimensions
    """
    iv1 = np.atleast_1d(ivar1)  #- handle list, tuple, ndarray, and scalar input
    iv2 = np.atleast_1d(ivar2)
    assert np.all(iv1 >= 0), 'ivar1 has negative elements'
    assert np.all(iv2 >= 0), 'ivar2 has negative elements'
    assert iv1.shape == iv2.shape, 'shape mismatch {} vs. {}'.format(iv1.shape, iv2.shape)
    ii = (iv1 > 0) & (iv2 > 0)
    ivar = np.zeros(iv1.shape)
    ivar[ii] = 1.0 / (1.0/iv1[ii] + 1.0/iv2[ii])
    
    #- Convert back to python float if input was scalar
    if isinstance(ivar1, (float, int)):
        return float(ivar)
    #- If input was 0-dim numpy array, convert back to 0-di
    elif ivar1.ndim == 0:
        return np.asarray(ivar[0])
    else:
        return ivar


_matplotlib_backend = None

def set_backend(backend='agg'):
    global _matplotlib_backend
    if _matplotlib_backend is None:
        _matplotlib_backend = backend
        import matplotlib
        matplotlib.use(_matplotlib_backend)
    return


