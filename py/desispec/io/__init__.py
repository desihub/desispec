# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.io
===========

Tools for data and metadata I/O.
"""
from __future__ import absolute_import
# The line above will help with 2to3 support.

from .brick import Brick
from .download import download, filepath2url
from .fiberflat import read_fiberflat, write_fiberflat
from .fibermap import read_fibermap, write_fibermap, empty_fibermap
from .filters import load_filter
from .fluxcalibration import (read_stdstar_templates, write_stdstar_models,
                              read_stdstar_models, read_flux_calibration,
                              write_flux_calibration)
from .frame import read_frame, write_frame
from .image import read_image, write_image
from .meta import (findfile, get_exposures, get_files, get_raw_files,
                   rawdata_root, specprod_root, validate_night,
                   get_pipe_plandir, get_pipe_rundir, get_pipe_scriptdir,
                   get_pipe_logdir, get_pipe_faildir)
from .qa import (read_qa_frame, read_qa_data, write_qa_frame, write_qa_brick,
                 load_qa_frame, write_qa_exposure, write_qa_prod)
from .raw import read_raw, write_raw
from .sky import read_sky, write_sky
from .util import (header2wave, fitsheader, native_endian, makepath,
                   write_bintable, iterfiles)
from .zfind import read_zbest, write_zbest

# Why is this even here?
from desispec.preproc import read_bias, read_pixflat, read_mask
