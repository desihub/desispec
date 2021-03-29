"""
desispec.workflow.batch: utilities for working with slurm batch queues
"""

import os
from pkg_resources import resource_filename
import yaml

from desiutil.log import get_logger

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

    configfile = resource_filename('desispec', 'data/batch_config.yaml')
    with open(configfile) as fx:
        config = yaml.safe_load(fx)

    #- Add the name for reference, in case it was default selected
    config['name'] = name

    return config[name]

def default_system():
    """
    Guess default system to use based on environment

    Returns default name to use
    """
    log = get_logger()
    name = None
    if 'NERSC_HOST' in os.environ:
        if os.environ['NERSC_HOST'] == 'cori':
            name = 'cori-haswell'
        elif os.environ['NERSC_HOST'] == 'perlmutter':
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




