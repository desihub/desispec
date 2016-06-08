#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.utils
=====================

Utilities for the pipeline.
"""

from __future__ import absolute_import, division, print_function

import os
import errno
import sys
import re


def option_list(opts):
    optlist = []
    for key, val in opts.items():
        keystr = "--{}".format(key)
        if isinstance(val, (bool,)):
            if val:
                optlist.append(keystr)
        else:
            optlist.append(keystr)
            if isinstance(val, (float,)):
                optlist.append("{:.14e}".format(val))
            elif isinstance(val, (list, tuple)):
                optlist.extend(val)
            else:
                optlist.append("{}".format(val))
    return optlist

#- Default number of processes to use for multiprocessing
if 'SLURM_CPUS_PER_TASK' in os.environ.keys():
    default_nproc = int(os.environ['SLURM_CPUS_PER_TASK'])
else:
    import multiprocessing as _mp
    default_nproc = max(1, _mp.cpu_count() // 2)
