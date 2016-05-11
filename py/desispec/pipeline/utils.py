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
                optlist.append()
        else:
            optlist.append(keystr)
            if isinstance(val, (float,)):
                optlist.append("{:.14e}".format(val))
            else:
                optlist.append("{}".format(val))
    return optlist

