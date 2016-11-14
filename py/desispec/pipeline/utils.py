#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.utils
=======================

Utilities for the pipeline.
"""
from __future__ import absolute_import, division, print_function
import numbers

def option_list(opts):
    """Convert key, value pairs into command-line options.

    Parameters
    ----------
    opts : dict-like
        Convert a dictionary into command-line options.

    Returns
    -------
    :class:`list`
        A list of command-line options.
    """
    optlist = []
    for key, val in opts.items():
        keystr = "--{}".format(key)
        if isinstance(val, bool):
            if val:
                optlist.append(keystr)
        else:
            optlist.append(keystr)
            if isinstance(val, float):
                optlist.append("{:.14e}".format(val))
            elif isinstance(val, (list, tuple)):
                optlist.extend(val)
            else:
                optlist.append("{}".format(val))
    return optlist
