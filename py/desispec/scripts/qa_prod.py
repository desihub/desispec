# Script for generating QA from a Production run
from __future__ import absolute_import, division

import argparse
import numpy as np

from desispec.qa import __offline_qa_version__

def parse(options=None):
    parser = argparse.ArgumentParser(description="Generate/Analyze Production Level QA [v{:s}]".format(__offline_qa_version__))

    parser.add_argument('--make_frameqa', type = int, default = 0,
                        help = 'Bitwise flag to control remaking the QA files (1) and figures (2) for each frame in the production')
    parser.add_argument('--slurp', default = False, action='store_true',
                        help = 'slurp production QA files into one per night?')
    parser.add_argument('--remove', default = False, action='store_true',
                        help = 'remove frame QA files?')
    parser.add_argument('--clobber', default=False, action='store_true',
                        help='clobber existing QA files?')
    parser.add_argument('--channel_hist', type=str, default=None,
                        help='Generate channel histogram(s)')
    parser.add_argument('--time_series', type=str, default=None,
                        help='Generate time series plot. Input is QATYPE-METRIC, e.g. SKYSUB-RESID')
    parser.add_argument('--bright_dark', type=int, default=0,
                        help='Restrict to bright/dark (flag: 0=all; 1=bright; 2=dark; only used in time_series)')
    parser.add_argument('--html', default = False, action='store_true',
                        help = 'Generate HTML files?')
    parser.add_argument('--qaprod_dir', type=str, default=None, help='Path to where QA is generated.  Default is qaprod_dir')
    parser.add_argument('--specprod_dir', type=str, default=None, help='Path to spectro production folder.  Default is specprod_dir')
    parser.add_argument('--night', type=str, default=None, help='Only process this night')
    parser.add_argument('--S2N_plot', default=False, action='store_true',
                        help = 'Generate a S/N plot for the production (vs. xaxis)')
    parser.add_argument('--ZP_plot', default=False, action='store_true',
                        help = 'Generate a ZP plot for the production (vs. xaxis)')
    parser.add_argument('--xaxis', type=str, default='MJD', help='Specify x-axis for S/N and ZP plots')

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
    from desispec.qa import qa_plots as dqqp

    log=get_logger()

    log.info("starting")
    # Initialize
    if args.specprod_dir is None:
        specprod_dir = meta.specprod_root()
    else:
        specprod_dir = args.specprod_dir
    if args.qaprod_dir is None:
        qaprod_dir = meta.qaprod_root(specprod_dir=specprod_dir)
    else:
        qaprod_dir = args.qaprod_dir

    qa_prod = QA_Prod(specprod_dir, qaprod_dir=qaprod_dir)

    # Restrict to a nights
    restrict_nights = [args.night] if args.night is not None else None

    # Remake Frame QA?
    if args.make_frameqa > 0:
        log.info("(re)generating QA related to frames")
        if (args.make_frameqa % 4) >= 2:
            make_frame_plots = True
        else:
            make_frame_plots = False
        # Run
        if (args.make_frameqa & 2**0) or (args.make_frameqa & 2**1):
            # Allow for restricted nights
            qa_prod.make_frameqa(make_plots=make_frame_plots, clobber=args.clobber,
                                 restrict_nights=restrict_nights)

    # Slurp and write?
    if args.slurp:
        qa_prod.qaexp_outroot = qaprod_dir 
        qa_prod.slurp_nights(make=(args.make_frameqa > 0), remove=args.remove, write_nights=True,
                             restrict_nights=restrict_nights)

    # Channel histograms
    if args.channel_hist is not None:
        # imports
        from matplotlib.backends.backend_pdf import PdfPages
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
        qa_prod.load_data()
        # Run
        qatype, metric = args.time_series.split('-')
        outfile= qaprod_dir+'/QA_time_{:s}.png'.format(args.time_series)
        dqqp.prod_time_series(qa_prod, qatype, metric, outfile=outfile, bright_dark=args.bright_dark)

    # <S/N> plot
    if args.S2N_plot:
        # Load up
        qa_prod.load_data()
        qa_prod.load_exposure_s2n()
        # Plot
        outfile= qaprod_dir+'/QA_S2N_{:s}.png'.format(args.xaxis)
        dqqp.prod_avg_s2n(qa_prod, optypes=['ELG', 'LRG', 'QSO'], xaxis=args.xaxis, outfile=outfile)

    # ZP plot
    if args.ZP_plot:
        # Load up
        qa_prod.load_data()
        # Plot
        outfile= qaprod_dir+'/QA_ZP_{:s}.png'.format(args.xaxis)
        dqqp.prod_ZP(qa_prod, xaxis=args.xaxis, outfile=outfile)

    # HTML
    if args.html:
        html.calib(qaprod_dir=qaprod_dir, specprod_dir=specprod_dir)
        html.make_exposures(qaprod_dir=qaprod_dir)
        html.toplevel(qaprod_dir=qaprod_dir)
