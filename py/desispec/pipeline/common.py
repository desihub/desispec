#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.defs
=========================

Common definitions needed by pipeline modules.
"""

from __future__ import absolute_import, division, print_function

from yaml import load as yload
from yaml import dump as ydump
try:
    from yaml import CLoader as YLoader
except ImportError:
    from yaml import Loader as YLoader


graph_types = [
    "night",
    "fibermap",
    "pix",
    "psfboot",
    "psf",
    "psfnight",
    "frame",
    "fiberflat",
    "sky",
    "stdstars",
    "calib",
    "cframe",
    "brick",
    "zbest"
]
"""Object types used in the graph."""


step_types = [
    "bootstrap",
    "psf",
    "psfcombine",
    "extract",
    "fiberflat",
    "sky",
    "stdstars",
    "fluxcal",
    "calibrate",
    "redshift"
]
"""The list of pipeline processing steps"""


step_file_types = {
    "bootstrap" : "psfboot",
    "psf" : "psf",
    "psfcombine" : "psfnight",
    "extract" : "frame",
    "fiberflat" : "fiberflat",
    "sky" : "sky",
    "stdstars" : "stdstars",
    "fluxcal" : "calib",
    "calibrate" : "cframe",
    "redshift" : "zbest"
}
"""The output object type associated with each step."""


file_types_step = {
    "psfboot" : "bootstrap",
    "psf" : "psf",
    "psfnight" : "psfcombine",
    "frame" : "extract",
    "fiberflat" : "fiberflat",
    "sky" : "sky",
    "stdstars" : "stdstars",
    "calib" : "fluxcal",
    "cframe" : "calibrate",
    "zbest" : "redshift"
}
"""The pipeline step associated with each object type."""


default_workers = {
    "bootstrap" : "Bootcalib",
    "psf" : "Specex",
    "psfcombine" : "SpecexCombine",
    "extract" : "Specter",
    "fiberflat" : "Fiberflat",
    "sky" : "Sky",
    "stdstars" : "Stdstars",
    "fluxcal" : "Fluxcal",
    "calibrate" : "Procexp",
    "redshift" : "Redmonster"    
}
"""The default worker type for each pipeline step."""


run_states = [
    "none",
    "done",
    "fail",
    "running"
]
"""The valid states of each pipeline task."""


def yaml_write(path, input):
    """
    Write a dictionary to a file.

    Args:
        path (str): the output file name.
        input (dict): the data.

    Returns:
        nothing.
    """
    with open(path, "w") as f:
        ydump(input, f, default_flow_style=False)
    return


def yaml_read(path, progress=None):
    """
    Read a dictionary from a file.

    Args:
        path (str): the input file name.

    Returns:
        dict: the data.
    """
    data = None
    with open(path, "r") as f:
        data = yload(f, Loader=YLoader)
    return data

