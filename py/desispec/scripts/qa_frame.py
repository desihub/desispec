# Script for generating QA for a single Frame
from __future__ import absolute_import, division

from desispec.qa import QA_Prod
from desiutil.log import get_logger
import argparse
import numpy as np


def parse(options=None):
    parser = argparse.ArgumentParser(description="Generate Production Level QA")
    parser.add_argument('--frame_file', type = str, required=True,
                        help='Frame filename including path as needed')
    parser.add_argument('--night', type = str, required=True,
                        help='Night of the exposure')
    parser.add_argument('--reduxdir', type = str, default = None, metavar = 'PATH',
                        help = 'Override default path ($DESI_SPECTRO_REDUX/$SPECPROD) to processed data.')
    parser.add_argument('--make_plots', default=False, action="store_true",
                        help = 'Generate QA figs too?')


    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args) :

    from desispec.io import meta
    from desispec.io import read_meta_frame
    from desispec.qa.qa_frame import qaframe_from_frame
    log=get_logger()

    log.info("starting")
    if args.reduxdir is None:
        specprod_dir = meta.specprod_root()
    else:
        specprod_dir = args.reduxdir

    # Generate qaframe (and figures?)
    qaframe_from_frame(args.night, args.frame_file,
                       specprod_dir=specprod_dir, make_plots=args.make_plots)


