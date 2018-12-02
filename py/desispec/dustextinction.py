"""
desispec.dustextinction
========================

Milky Way dust extinction curve routines.
"""
from __future__ import absolute_import
import numpy as np

try:
    from desiutil.dust import ext_odonnell, ext_ccm
except:
    from desiutil.log import get_logger
    log = get_logger()
    log.warning('Please update your desiutil checkout to include the latest dust module.')
