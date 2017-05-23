"""
desispec.io.params
===============

IO routines for parameter values
"""
from __future__ import print_function, absolute_import, division

import yaml

from pkg_resources import resource_filename


def read_obj_param(filename=None):
    """Read obj parameter data from file
    """
    # File
    if filename is None:
        filename = resource_filename('desispec','/data/params/desi_obj.yml')
    # Read yaml
    with open(filename, 'r') as infile:
        obj_params = yaml.load(infile)
    # Return
    return obj_params



