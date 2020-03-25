# Script for generating QA for a full exposure
from __future__ import absolute_import, division

from desiutil.log import get_logger
import argparse

from desispec.qa import __offline_qa_version__

def parse(options=None):
    parser = argparse.ArgumentParser(description="Generate Exposure Level QA [v{:s}]".format(__offline_qa_version__))
    parser.add_argument('--expid', type=int, required=True, help='Exposure ID')
    parser.add_argument('--qatype', type=str, required=False,
                        help="Type of QA to generate [fiberflat, s2n]")
    parser.add_argument('--channels', type=str, help="List of channels to include. Default = b,r,z]")
    parser.add_argument('--specprod_dir', type = str, default=None, metavar='PATH',
                        help='Override default path to processed data.')
    parser.add_argument('--qaprod_dir', type=str, default=None, metavar='PATH',
                        help='Override default path to QA data.')
    parser.add_argument('--rebuild', default=False, action="store_true",
                        help = 'Regenerate the QA files for this exposure?')
    parser.add_argument('--qamulti_root', type=str, default=None,
                        help='Root name for a set of slurped QA files (e.g. mini_qa). Uses $SPECPROD/QA for path')
    parser.add_argument('--slurp', type=str, default=None,
                        help='Root name for slurp QA file to add to (e.g. mini_qa). File must already exist.  Uses $SPECPROD/QA for path')

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

    # Setup
    if args.specprod_dir is None:
        specprod_dir = meta.specprod_root()
    else:
        specprod_dir = args.specprod_dir
    if args.qaprod_dir is None:
        qaprod_dir = meta.qaprod_root(specprod_dir=specprod_dir)
    else:
        qaprod_dir = args.qaprod_dir
    if args.channels is None:
        channels = ['b','r','z']
    else:
        channels = [iarg for iarg in args.channels.split(',')]

    # Find night
    night = find_exposure_night(args.expid, specprod_dir=specprod_dir)

    # Instantiate
    qa_exp = QA_Exposure(args.expid, night, specprod_dir=specprod_dir,
                         qaprod_dir=qaprod_dir,
                         no_load=args.rebuild, multi_root=args.qamulti_root)
    # Rebuild?
    if args.rebuild:
        qa_exp.build_qa_data(rebuild=True)

    # Fiber QA
    if args.qatype == 'fiberflat':
        for channel in channels:
            exposure_fiberflat(channel, args.expid, 'meanflux')

    # S/N
    if args.qatype == 's2n':
        # S2N table
        qa_exp.s2n_table()
        # Figure time
        exposure_s2n(qa_exp, 'resid', specprod_dir=specprod_dir)

    # Slurp into a file?
    if args.slurp is not None:
        qa_exp.slurp_into_file(args.slurp)


