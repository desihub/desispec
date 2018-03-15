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


task_states = [
    "waiting",
    "ready",
    "running",
    "done",
    "fail"
]
"""The valid states of each pipeline task."""

task_state_to_int = {
    "waiting" : 0,
    "ready" : 1,
    "running" : 2,
    "done" : 3,
    "fail" : 4
}

task_int_to_state = {
    0 : "waiting",
    1 : "ready",
    2 : "running",
    3 : "done",
    4 : "fail"
}


state_colors = {
    "waiting": "#000000",
    "ready" : "#0000ff",
    "running": "#ffff00",
    "done": "#00ff00",
    "fail": "#ff0000",
}
"""State colors used for visualization."""


task_name_sep = "_"
"""The separator string used for building object names."""

prod_options_name = "options.yaml"
"""The name of the options file inside the run directory."""


#
# REFACTOR:
# All the code below is dead.



#
# object_types = [
#     "fibermap",
#     "pix",
#     "psfboot",
#     "psf",
#     "psfnight",
#     "frame",
#     "fiberflat",
#     "sky",
#     "stdstars",
#     "calib",
#     "cframe",
#     "spectra",
#     "zbest"
# ]
# """Object types used in the pipeline."""
#
# task_features = [
#     "type",     # The task type itself (fibermap, frame, etc)
#     "night",    # The observing night
#     "flavor"    # The exposure flavor (arc, flat, etc)
#     "band",     # The band {r, b, z}
#     "spec",     # The spectrograph {0 ... 9}
#     "expid",    # The exposure ID
#     "nside",    # The Healpix NSIDE value
#     "pixel"     # The Healpix pixel value
# ]
# """Possible features that each task might have."""
#
#
# task_feature_format = {
#     "type" : "s",
#     "night" : "08d",
#     "flavor" : "s",
#     "band" : "s",
#     "spec" : "d",
#     "expid" : "08d",
#     "nside" : "d",
#     "pixel" : "d"
# }
# """The string format of each feature."""
#
#
# task_feature_db_format = {
#     "type" : "text",
#     "night" : "integer",
#     "flavor" : "text",
#     "band" : "text",
#     "spec" : "integer",
#     "expid" : "integer",
#     "nside" : "integer",
#     "pixel" : "integer"
# }
# """The string format of each feature."""

#
# object_props = {
#     "fibermap" : [
#         "night",
#         "type",
#         "expid"
#     ],
#     "pix" : [
#         "night",
#         "type",
#         "band",
#         "spec",
#         "expid"
#     ],
#     "psfboot" : [
#         "night",
#         "type",
#         "band",
#         "spec"
#     ],
#     "psf" : [
#         "night",
#         "type",
#         "band",
#         "spec",
#         "expid"
#     ],
#     "psfnight" : [
#         "night",
#         "type",
#         "band",
#         "spec"
#     ],
#     "frame" : [
#         "night",
#         "type",
#         "band",
#         "spec",
#         "expid"
#     ],
#     "fiberflat" : [
#         "night",
#         "type",
#         "band",
#         "spec",
#         "expid"
#     ],
#     "sky" : [
#         "night",
#         "type",
#         "band",
#         "spec",
#         "expid"
#     ],
#     "stdstars" : [
#         "night",
#         "type",
#         "spec",
#         "expid"
#     ],
#     "calib" : [
#         "night",
#         "type",
#         "band",
#         "spec",
#         "expid"
#     ],
#     "cframe" : [
#         "night",
#         "type",
#         "band",
#         "spec",
#         "expid"
#     ],
#     "spectra" : [
#         "type",
#         "nside",
#         "pixel"
#     ],
#     "zbest" : [
#         "type",
#         "nside",
#         "pixel"
#     ]
# }
# """The specific features for each object type."""
#
#
# task_types = [
#     "bootstrap",
#     "psf",
#     "psfcombine",
#     "extract",
#     "fiberflat",
#     "sky",
#     "stdstars",
#     "fluxcal",
#     "calibrate",
#     "redshift"
# ]
# """The list of pipeline processing tasks"""
#
#
# task_to_object = {
#     "bootstrap" : "psfboot",
#     "psf" : "psf",
#     "psfcombine" : "psfnight",
#     "extract" : "frame",
#     "fiberflat" : "fiberflat",
#     "sky" : "sky",
#     "stdstars" : "stdstars",
#     "fluxcal" : "calib",
#     "calibrate" : "cframe",
#     "redshift" : "zbest"
# }
# """The output object type associated with each task."""
#
#
# object_to_task = {
#     "psfboot" : "bootstrap",
#     "psf" : "psf",
#     "psfnight" : "psfcombine",
#     "frame" : "extract",
#     "fiberflat" : "fiberflat",
#     "sky" : "sky",
#     "stdstars" : "stdstars",
#     "calib" : "fluxcal",
#     "cframe" : "calibrate",
#     "zbest" : "redshift"
# }
# """The pipeline task associated with each object type."""
#
#
# default_workers = {
#     "bootstrap" : "Bootcalib",
#     "psf" : "Specex",
#     "psfcombine" : "SpecexCombine",
#     "extract" : "Specter",
#     "fiberflat" : "Fiberflat",
#     "sky" : "Sky",
#     "stdstars" : "Stdstars",
#     "fluxcal" : "Fluxcal",
#     "calibrate" : "Procexp",
#     "redshift" : "Redrock"
# }
# """The default worker for each type of task."""
