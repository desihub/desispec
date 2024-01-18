"""
desispec.io.params
==================

IO routines for parameter values
"""
from __future__ import print_function, absolute_import, division

import yaml

from importlib import resources

# CACHE
_params_cache = {}

def read_params(filename=None, reload=False):
    """Read parameter data from file
    """
    global _params_cache  # Cache
    if filename is None:
        filename = resources.files('desispec').joinpath('data/params/desispec_param.yml')

    # Init
    if (filename not in _params_cache) or (reload is True):
        # Read yaml
        with open(filename, 'r') as infile:
            _params_cache[filename] = yaml.safe_load(infile)
    # Return
    return _params_cache[filename]



