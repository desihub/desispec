#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.io
===========

Tools for data and metadata I/O.
"""

# help with 2to3 support
from __future__ import absolute_import, division

from .meta import (findfile, get_exposures, get_files, get_raw_files, 
    rawdata_root, specprod_root, validate_night)
from .frame import read_frame, write_frame
from .sky import read_sky, write_sky
from .fiberflat import read_fiberflat, write_fiberflat
from .fibermap import read_fibermap, write_fibermap, empty_fibermap
from .brick import Brick
from .qa import read_qa_frame, read_qa_data, write_qa_frame, write_qa_brick
from .zfind import read_zbest, write_zbest
from .image import read_image, write_image
from .util import (header2wave, fitsheader, native_endian, makepath,
    write_bintable, iterfiles)
from .fluxcalibration import (
    read_stdstar_templates, write_stdstar_models, read_stdstar_models,
    read_flux_calibration, write_flux_calibration)
from .filters import load_filter
from .download import download, filepath2url
from .crc import memcrc, cksum
from .database import (load_brick, is_night, load_night, is_flavor, load_flavor,
    get_bricks_by_name, get_brickid_by_name, load_data)

from desispec.preproc import read_bias, read_pixflat, read_mask
from desispec.io.raw import read_raw, write_raw
