# Script for generating QA from a Production run
from __future__ import absolute_import, division

import argparse
import numpy as np

from  desispec import _version as desis_v

def parse(options=None):
    parser = argparse.ArgumentParser(description="Generate/Analyze Production Level QA [v{:s}]".format(desis_v.__offline_qa_version__))

    parser.add_argument('--reduxdir', type = str, default = None, required=False,
                        help = 'Override default path ($DESI_SPECTRO_REDUX/$SPECPROD) to processed data.')
    parser.add_argument('--make_frameqa', type = int, default = 0,
                        help = 'Bitwise flag to control remaking the QA files (1) and figures (2) for each frame in the production')
    parser.add_argument('--slurp', default = False, action='store_true',
                        help = 'slurp production QA files into one?')
    parser.add_argument('--remove', default = False, action='store_true',
                        help = 'remove frame QA files?')
    parser.add_argument('--clobber', default=False, action='store_true',
                        help='clobber existing QA files?')
    parser.add_argument('--channel_hist', type=str, default=None,
                        help='Generate channel histogram(s)')
    parser.add_argument('--time_series', type=str, default=None,
                        help='Generate time series plot. Input is QATYPE-METRIC, e.g. SKYSUB-MED_RESID')
    parser.add_argument('--html', default = False, action='store_true',
                        help = 'Generate HTML files?')

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args) :

    from desispec.qa import QA_Prod
    from desispec.qa import html
    from desiutil.log import get_logger
    from desispec.io import meta

    log=get_logger()

    log.info("starting")
    # Initialize
    if args.reduxdir is None:
        specprod_dir = meta.specprod_root()
    else:
        specprod_dir = args.reduxdir

    qa_prod = QA_Prod(specprod_dir)

    # Remake Frame QA?
    if args.make_frameqa > 0:
        log.info("(re)generating QA related to frames")
        if (args.make_frameqa % 4) >= 2:
            make_frame_plots = True
        else:
            make_frame_plots = False
        # Run
        if (args.make_frameqa & 2**0) or (args.make_frameqa & 2**1):
            qa_prod.make_frameqa(make_plots=make_frame_plots, clobber=args.clobber)

    # Slurp?
    if args.slurp:
        qa_prod.slurp(make=(args.make_frameqa > 0), remove=args.remove)

    # Channel histograms
    if args.channel_hist is not None:
        # imports
        from matplotlib.backends.backend_pdf import PdfPages
        from desispec.qa import qa_plots as dqqp
        #
        qa_prod.load_data()
        outfile = qa_prod.prod_name+'_chist.pdf'
        pp = PdfPages(outfile)
        # Default?
        if args.channel_hist == 'default':
            dqqp.prod_channel_hist(qa_prod, 'FIBERFLAT', 'MAX_RMS', pp=pp, close=False)
            dqqp.prod_channel_hist(qa_prod, 'SKYSUB', 'MED_RESID', xlim=(-15,15), pp=pp, close=False)
            dqqp.prod_channel_hist(qa_prod, 'FLUXCALIB', 'MAX_ZP_OFF', pp=pp, close=False)
        # Finish
        print("Writing {:s}".format(outfile))
        pp.close()

    # Time plots
    if args.time_series is not None:
        # QATYPE-METRIC
        from desispec.qa import qa_plots as dqqp
        qa_prod.load_data()
        # Run
        qatype, metric = args.time_series.split('-')
        outfile= specprod_dir+'/QA/QA_time_{:s}.png'.format(args.time_series)
        dqqp.prod_time_series(qa_prod, qatype, metric, outfile=outfile)

    # HTML
    if args.html:
        html.calib()
        html.make_exposures()
        html.toplevel()
