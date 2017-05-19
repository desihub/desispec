"""
desispec.io.params
===============

IO routines for parameter values
"""
from __future__ import print_function, absolute_import, division

import os, yaml

import desispec
param_path = desispec.__path__[0]+'/data/params/'


def read_obj_param(filename=None):
    """Read obj parameter data from file
    """
    # File
    if filename is None:
        filename = param_path+'desi_obj.yml'
    # Read yaml
    with open(param_path+'desi_obj.yml', 'r') as infile:
        obj_params = yaml.load(infile)
    # Return
    return obj_params



