"""
desispec.log
============

This is a transitional dummy wrapper on ``desiutil.log``.
"""
from __future__ import absolute_import
# from warnings import warn

from desiutil.log import DEBUG, INFO, WARNING, ERROR, CRITICAL
from desiutil.log import get_logger as _desiutil_get_logger

def get_logger(*args, **kwargs):
    """Transitional dummy wrapper on ``desiutil.log.get_logger()``.
    """
    # warn("desispec.log is deprecated, please use desiutil.log.",
    #      DeprecationWarning)
    log = _desiutil_get_logger(*args, **kwargs)
    log.warn("desispec.log is deprecated, please use desiutil.log.")
    return log
