"""
desispec.workflow.batch
=======================

Utilities for working with slurm batch queues.
"""

import os
from importlib import resources
import yaml

from desiutil.log import get_logger

_config_cache = dict()
def get_config(name):
    """
    Return configuration dictionary for system `name`

    Args:
        name (str): e.g. cori-haswell, cori-knl, dirac, perlmutter-gpu, ...

    Returns dictionary with keys:
        * site: location of system, e.g. 'NERSC'
        * cores_per_node: number of physical cores per node
        * threads_per_core: hyperthreading / SMT per core
        * memory: memory per node in GB
        * timefactor: scale time estimates by this amount on this system
        * gpus_per_node: number of GPUs per node
        * batch_opts: list of additional batch options for script header
    """
    if name is None:
        name = default_system()

    global _config_cache
    if name in _config_cache:
        return _config_cache[name]

    configfile = resources.files('desispec').joinpath('data/batch_config.yaml')
    with open(configfile) as fx:
        config = yaml.safe_load(fx)

    #- Add the name for reference, in case it was default selected
    config['name'] = name

    #- Add to cache so that we don't have to re-read batch_config.yaml every time
    _config_cache[name] = config[name]

    return config[name]


def default_system(jobdesc=None, no_gpu=False):
    """
    Guess default system to use based on environment

    Args:
        jobdesc (str): Description of the job in the processing table (optional).
        no_gpu (bool): Don't use GPU's even if available. Default False.

    Returns:
         name (str): default system name to use
    """
    log = get_logger()
    name = None
    if 'NERSC_HOST' in os.environ:
        if os.environ['NERSC_HOST'] == 'perlmutter':
            ## HARDCODED: for now arcs and biases can't use gpu's, so use cpu's
            if jobdesc in ['linkcal', 'arc', 'biasnight', 'biaspdark',
                           'ccdcalib', 'badcol', 'psfnight', 'pdark' ]:
                name = 'perlmutter-cpu'
            elif no_gpu:
                name = 'perlmutter-cpu'
            else:
                name = 'perlmutter-gpu'
    elif os.path.isdir('/clusterfs/dirac1'):
        name = 'dirac'

    if name is None:
        msg = 'Unable to determine default batch system from environment'
        log.error(msg)
        raise RuntimeError(msg)
    else:
        log.info(f'Guessing default batch system {name}')

    return name


def parse_reservation(reservation, jobdesc):
    """
    Parse reservation name into cpu/gpu reservation based upon jobdesc

    Args:
        reservation (str): resvname or resvname_cpu,resvname_gpu or None
        jobdesc (str): job description string e.g. 'arc', 'flat', 'tilenight'

    Returns:
        cpu_reservation_name, gpu_reservation_name

    If a single reservation name is provided, return both cpu/gpu as the same.
    If either is 'none' (case-insensitive), return None for that reservation
    """
    if reservation is None:
        return reservation

    tmp = reservation.split(',')
    if len(tmp) == 1:
        reservation_cpu = reservation_gpu = reservation
    elif len(tmp) == 2:
        reservation_cpu, reservation_gpu = tmp
    else:
        raise ValueError(f'Unable to parse {reservation} as rescpu,resgpu')

    if reservation_cpu.lower() == 'none':
        reservation_cpu = None

    if reservation_gpu.lower() == 'none':
        reservation_gpu = None

    system_name = default_system(jobdesc)
    config = get_config(system_name)

    if 'gpus_per_node' not in config or config['gpus_per_node'] == 0:
        return reservation_cpu
    else:
        return reservation_gpu


def determine_resources(ncameras, jobdesc, nexps=1, forced_runtime=None, queue=None, system_name=None):
    """
    Determine the resources that should be assigned to the batch script given what
    desi_proc needs for the given input information.

    Args:
        ncameras (int): number of cameras to be processed
        jobdesc (str): type of data being processed
        nexps (int, optional): the number of exposures processed in this step
        queue (str, optional): the Slurm queue to be submitted to. Currently not used.
        system_name (str, optional): batch compute system, e.g. cori-haswell or perlmutter-gpu

    Returns:
        tuple: A tuple containing:

        * ncores: int, number of cores (actually 2xphysical cores) that should be submitted via "-n {ncores}"
        * nodes:  int, number of nodes to be requested in the script. Typically  (ncores-1) // cores_per_node + 1
        * runtime: int, the max time requested for the script in minutes for the processing.
    """
    if system_name is None:
        system_name = default_system(jobdesc=jobdesc)

    config = get_config(system_name)
    log = get_logger()
    jobdesc = jobdesc.upper()

    nspectro = (ncameras - 1) // 3 + 1
    nodes = None
    if jobdesc in ('ARC', 'TESTARC'):
        ncores          = 20 * (10*(ncameras+1)//20) # lowest multiple of 20 exceeding 10 per camera
        ncores, runtime = ncores + 1, 45             # + 1 for worflow.schedule scheduler proc
    elif jobdesc in ('FLAT', 'TESTFLAT'):
        runtime = 40
        if system_name.startswith('perlmutter'):
            ncores = config['cores_per_node']
        else:
            ncores = 20 * nspectro
    elif jobdesc == 'TILENIGHT':
        runtime  = int(60. / 140. * ncameras * nexps) # 140 frames per node hour
        runtime += 40                                 # overhead
        ncores = config['cores_per_node']
        if not system_name.startswith('perlmutter'):
            msg = 'tilenight cannot run on system_name={}'.format(system_name)
            log.critical(msg)
            raise ValueError(msg)
    elif jobdesc in ('SKY', 'TWILIGHT', 'SCIENCE','PRESTDSTAR'):
        runtime = 30
        if system_name.startswith('perlmutter'):
            ncores = config['cores_per_node']
        else:
            ncores = 20 * nspectro
    elif jobdesc in ('DARK', 'BADCOL'):
        ncores, runtime = ncameras, 5
    elif jobdesc in ('BIASNIGHT', 'BIASPDARK'):
        ## Jobs are memory limited, so use 15 cores per node
        ## and split work of 30 cameras across 2 nodes
        nodes = (ncameras // 16) + 1 # 2 nodes unless ncameras <= 15
        ncores = 15
        ## 5 minutes base plus 2 mins per loop over dark exposures
        ## Old was 8 minutes base plus 4 mins
        pdarkcores = min([ncameras*nexps, nodes*config['cores_per_node']])
        runtime = 5 + 2.*(float(nodes*config['cores_per_node'])/float(pdarkcores))
    elif jobdesc in ('PDARK'):
        nodes = 1 
        # can do 1 core per camera per exp, but limit to cores available
        ncores = min([ncameras*nexps, nodes*config['cores_per_node']])
        ## 4 minutes base plus 4 mins per loop over dark exposures 
        ## Old was 4 minutes base plus 2 mins
        runtime = 5 + 2.*(float(nodes*config['cores_per_node'])/float(ncores))
    elif jobdesc == 'CCDCALIB':
        nodes = 1
        ncores, runtime = ncameras, 7 # 5 mins after perlmutter system scaling factor
    elif jobdesc == 'ZERO':
        ncores, runtime = 2, 5
    elif jobdesc == 'PSFNIGHT':
        ncores, runtime = ncameras, 5
    elif jobdesc == 'NIGHTLYFLAT':
        ncores, runtime = ncameras, 5
    elif jobdesc == 'STDSTARFIT':
        #- Special case hardcode: stdstar parallelism maxes out at ~30 cores
        #- and on KNL, it OOMs above that anyway.
        #- This might be more related to using a max of 30 standards, not that
        #- there are 30 cameras (coincidence).
        #- Use 32 as power of 2 for core packing
        ncores = 32
        runtime = 8+2*nexps
    elif jobdesc == 'POSTSTDSTAR':
        runtime = 10
        ncores = ncameras
    elif jobdesc == 'NIGHTLYBIAS':
        ncores, runtime = 15, 5
        nodes = 2
    elif jobdesc in ['PEREXP', 'PERNIGHT', 'CUMULATIVE', 'CUSTOMZTILE']:
        if system_name.startswith('perlmutter'):
            nodes, runtime = 1, 50  #- timefactor will bring time back down
        else:
            nodes, runtime = 2, 30
        ncores = nodes * config['cores_per_node']
    elif jobdesc == 'HEALPIX':
        nodes = 1
        runtime = 100
        ncores = nodes * config['cores_per_node']
    elif jobdesc == 'LINKCAL':
        nodes, ncores = 1, 1
        runtime = 5.
    else:
        msg = 'unknown jobdesc={}'.format(jobdesc)
        log.critical(msg)
        raise ValueError(msg)

    if forced_runtime is not None:
        runtime = forced_runtime

    if nodes is None:
        nodes = (ncores - 1) // config['cores_per_node'] + 1

    # - Arcs and flats make good use of full nodes, but throttle science
    # - exposures to 5 nodes to enable two to run together in the 10-node
    # - realtime queue, since their wallclock is dominated by less
    # - efficient sky and fluxcalib steps
    if jobdesc in ('ARC', 'TESTARC'):#, 'FLAT', 'TESTFLAT'):
        max_realtime_nodes = 10
    else:
        max_realtime_nodes = 5

    #- Pending further optimizations, use same number of nodes in all queues
    ### if (queue == 'realtime') and (nodes > max_realtime_nodes):
    if (nodes > max_realtime_nodes):
        nodes = max_realtime_nodes
        ncores = config['cores_per_node'] * nodes
        if jobdesc in ('ARC', 'TESTARC'):
            # adjust for workflow.schedule scheduler proc
            ncores = ((ncores - 1) // 20) * 20 + 1

    #- Allow KNL jobs to be slower than Haswell,
    #- except for ARC so that we don't have ridiculously long times
    #- (Normal arc is still ~15 minutes, albeit with a tail)
    if jobdesc not in ['ARC', 'TESTARC']:
        runtime *= config['timefactor']

    #- Do not allow runtime to be less than 5 min
    if runtime < 5:
        runtime = 5

    #- Add additional overhead factor if needed
    if 'NERSC_RUNTIME_OVERHEAD' in os.environ:
        t = os.environ['NERSC_RUNTIME_OVERHEAD']
        log.info(f'Adding $NERSC_RUNTIME_OVERHEAD={t} minutes to batch runtime request')
        runtime += float(runtime)

    return ncores, nodes, runtime



