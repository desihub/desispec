# Script for generating QA for a full exposure
from __future__ import absolute_import, division

from desiutil.log import get_logger
import argparse

from desispec.qa import __offline_qa_version__

def parse(options=None):
    parser = argparse.ArgumentParser(description="Generate Exposure Level QA [v{:s}]".format(__offline_qa_version__))
    parser.add_argument('--expid', type = int, required=True, help='Exposure ID')
    parser.add_argument('--qatype', type = str, required=True,
                        help="Type of QA to generate [fiberflat, s2n]")
    parser.add_argument('--channels', type=str, help="List of channels to include. Default = b,r,z]")
    parser.add_argument('--reduxdir', type = str, default = None, metavar = 'PATH',
                        help = 'Override default path ($DESI_SPECTRO_REDUX/$SPECPROD) to processed data.')
    parser.add_argument('--rebuild', default=False, action="store_true",
                        help = 'Regenerate the QA files for this exposure?')


    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args) :

    from desispec.io import meta
    from desispec.qa.qa_plots import exposure_fiberflat, exposure_s2n
    from desispec.qa.qa_exposure import QA_Exposure
    from desispec.io.meta import find_exposure_night

    log=get_logger()

    log.info("starting")
    if args.reduxdir is None:
        specprod_dir = meta.specprod_root()
    else:
        specprod_dir = args.reduxdir
    if args.channels is None:
        channels = ['b','r','z']
    else:
        channels = [iarg for iarg in args.channels.split(',')]

    # Fiber QA
    if args.qatype == 'fibermap':
        for channel in channels:
            exposure_fiberflat(channel, args.expid, 'meanflux')

    # S/N
    if args.qatype == 's2n':
        # Find night
        night = find_exposure_night(args.expid)
        # Instantiate
        qa_exp = QA_Exposure(args.expid, night, 'science', specprod_dir=specprod_dir, no_load=~args.rebuild)
        # Rebuild?
        if args.rebuild:
            qa_exp.build_qa_data(rebuild=True)
        # Figure time
        exposure_s2n(qa_exp, 'resid')
        import pdb; pdb.set_trace()



