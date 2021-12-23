# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.io
===========

Tools for data and metadata I/O.
"""
from __future__ import absolute_import
# The line above will help with 2to3 support.

import warnings
warnings.filterwarnings('ignore', message="'.*nanomaggies.* did not parse as fits unit.*")
warnings.filterwarnings('ignore', message=".*'10\*\*6 arcsec.* did not parse as fits unit.*")

# from .download import download, filepath2url
from .fiberflat import read_fiberflat, write_fiberflat
from .fibermap import read_fibermap, write_fibermap, empty_fibermap
from .filters import load_filter,load_legacy_survey_filter
from .fluxcalibration import (read_stdstar_templates, write_stdstar_models,
                              read_stdstar_models, read_flux_calibration,
                              write_flux_calibration, read_average_flux_calibration)
from .spectra import (read_spectra, write_spectra, read_frame_as_spectra,
                      read_tile_spectra)
from .frame import read_meta_frame, read_frame, write_frame
from .xytraceset import read_xytraceset, write_xytraceset
from .image import read_image, write_image
from .meta import (findfile, get_exposures, get_files, get_raw_files,
                   rawdata_root, specprod_root, validate_night, qaprod_root,
                   get_pipe_rundir, get_pipe_scriptdir, get_pipe_database,
                   get_pipe_logdir, get_reduced_frames, get_pipe_pixeldir,
                   get_nights, get_pipe_nightdir, find_exposure_night,
                   shorten_filename)
from .params import read_params
from .qa import (read_qa_frame, read_qa_data, write_qa_frame, write_qa_brick,
                 load_qa_frame, write_qa_exposure, write_qa_multiexp, load_qa_multiexp,
                 qafile_from_framefile)
from .exposure_tile_qa import (read_exposure_qa, write_exposure_qa, read_tile_qa, write_tile_qa)
from .raw import read_raw, write_raw
from .sky import read_sky, write_sky
from .skycorr import (read_skycorr, write_skycorr, read_skycorr_pca, write_skycorr_pca)
from .util import (header2wave, fitsheader, native_endian, makepath,
                   write_bintable, iterfiles, healpix_degrade_fixed,
                   healpix_subdirectory, replace_prefix)

# Why is this even here?
# Commented out by JXP as this causes a circular import on Python 3.7
#from desispec.preproc import read_bias, read_pixflat, read_mask
