# Script for generating QA for a single Frame
from __future__ import absolute_import, division

from desiutil.log import get_logger
import argparse


def parse(options=None):
    parser = argparse.ArgumentParser(description="Generate Frame Level QA [v1.1]")
    parser.add_argument('--frame_file', type = str, required=True,
                        help='Frame filename.  Full path is not required nor desired. ')
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
    from desispec.qa.qa_frame import qaframe_from_frame
    log=get_logger()

    log.info("starting")
    if args.reduxdir is None:
        specprod_dir = meta.specprod_root()
    else:
        specprod_dir = args.reduxdir

    # Generate qaframe (and figures?)
    _ = qaframe_from_frame(args.frame_file, specprod_dir=specprod_dir, make_plots=args.make_plots)


