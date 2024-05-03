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
        if os.environ['NERSC_HOST'] == 'cori':
            name = 'cori-haswell'
        elif os.environ['NERSC_HOST'] == 'perlmutter':
            ## HARDCODED: for now arcs and biases can't use gpu's, so use cpu's
            if jobdesc in ['linkcal', 'arc', 'nightlybias', 'ccdcalib',
                           'badcol', 'psfnight', 'nightlyflat']:
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



