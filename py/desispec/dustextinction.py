"""
desispec.dustextinction
========================

Milky Way dust extinction curve routines.
"""
from __future__ import absolute_import
import numpy as np

try:
    from desiutil.dust import ext_odonnell, ext_ccm
except ImportError as err:
    from desiutil.log import get_logger
    log = get_logger()
    msg = 'Please update your desiutil checkout to include the latest dust module.'
    log.error(msg)
    raise err
