"""
desispec.gpu
============

Utility functions for working with GPUs
"""

import os
from desiutil.log import get_logger

#- Require both cupy and numba.cuda,
#- but track availability separately for debugging
try:
    import cupy
    import cupyx.scipy.ndimage
    _cupy_available = cupy.is_available()  #- True if cupy detects a GPU
except ImportError:
    _cupy_available = False

try:
    import numba.cuda
    _numba_cuda_available = numba.cuda.is_available()
except ImportError:
    _numba_cuda_available = False

#- If $DESI_NO_GPU is set to anything, don't use a GPU.
#- This is primarily for debugging to globally disable GPU usage.
if 'DESI_NO_GPU' in os.environ:
    _desi_use_gpu = False
else:
    _desi_use_gpu = True

def is_gpu_available():
    """Return whether cupy and numba.cuda are installed and a GPU
    is available to use, and $DESI_NO_GPU is *not* set"""
    return _cupy_available and _numba_cuda_available and _desi_use_gpu

def free_gpu_memory():
    """Release all cupy GPU memory; ok to call even if no GPUs"""
    if is_gpu_available():
        mempool = cupy.get_default_memory_pool()
        mempool.free_all_blocks()

def redistribute_gpu_ranks(comm, method='round-robin'):
    """Redistribute which MPI ranks are assigned to which GPUs

    Args:
        comm: MPI communicator

    Options:
        method: 'round-robin' (default) or 'contiguous'

    Returns:
        device_id assigned (-1 if no GPUs)

    'round-robin' assigns cyclically, e.g. 8 ranks on 4 GPUs would
    be assigned [0,1,2,3,0,1,2,3].
    'continuous' assigns contiguous ranks to the same GPU, e.g.
    [0,0,1,1,2,2,3,3,4,4]
    """
    device_id = -1  #- default if no GPUs
    if is_gpu_available():
        log = get_logger()
        ngpu = cupy.cuda.runtime.getDeviceCount()
        if comm is None:
            device_id = 0
            cupy.cuda.Device(device_id).use()
            log.info(f'No MPI communicator; assigning process to GPU {device_id}/{ngpu}')
        else:
            if method == 'round-robin':
                device_id = comm.rank % ngpu
            elif method == 'contiguous':
                device_id = int(comm.rank / ngpu)
            else:
                msg = f'method should be "round-robin" or "contiguous", not "{method}"'
                log.error(msg)
                raise ValueError(msg)

            cupy.cuda.Device(device_id).use()
            log.debug('Assigning rank=%d to GPU=%d/%d', comm.rank, device_id, ngpu)

            device_assignments = comm.gather(device_id, root=0)
            if comm.rank == 0:
                log.info(f'Assigned MPI ranks to GPUs {device_assignments}')

    return device_id



