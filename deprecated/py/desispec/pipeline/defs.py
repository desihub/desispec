#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.defs
======================

Common definitions needed by pipeline modules.
"""

from __future__ import absolute_import, division, print_function


task_states = [
    "waiting",
    "ready",
    "running",
    "done",
    "failed"
]
"""The valid states of each pipeline task."""

task_state_to_int = {
    "waiting" : 0,
    "ready" : 1,
    "running" : 2,
    "done" : 3,
    "failed" : 4
}

task_int_to_state = {
    0 : "waiting",
    1 : "ready",
    2 : "running",
    3 : "done",
    4 : "failed"
}


state_colors = {
    "waiting": "#000000",
    "ready" : "#0000ff",
    "running": "#ffff00",
    "done": "#00ff00",
    "failed": "#ff0000",
}
"""State colors used for visualization."""


task_name_sep = "_"
"""The separator string used for building object names."""

prod_options_name = "options.yaml"
"""The name of the options file inside the run directory."""
