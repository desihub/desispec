"""
desispec.io.params
===============

IO routines for parameter values
"""
from __future__ import print_function, absolute_import, division

import yaml

from pkg_resources import resource_filename


def read_params(filename=None):
    """Read parameter data from file
    """
    global params  # Caching

    # File
    if filename is None:
        filename = resource_filename('desispec','/data/params/desispec_param.yml')
    # Read yaml
    with open(filename, 'r') as infile:
        params = yaml.load(infile)
    # Return
    return params



