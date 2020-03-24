# Script for analyzing QA from a Night
from __future__ import absolute_import, division

import argparse

from desispec.qa import __offline_qa_version__

def parse(options=None):
    parser = argparse.ArgumentParser(description="Analyze Night Level QA [v{:s}]".format(__offline_qa_version__))

    #parser.add_argument('--channel_hist', type=str, default=None,
    #                    help='Generate channel histogram(s)')
    parser.add_argument('--expid_series', default=False, action='store_true',
                        help='Generate exposure series plots.')
    parser.add_argument('--bright_dark', type=int, default=0,
                        help='Restrict to bright/dark (flag: 0=all; 1=bright; 2=dark; only used in time_series)')
    parser.add_argument('--qaprod_dir', type=str, default=None, help='Path to where QA is generated.  Default is qaprod_dir')
    parser.add_argument('--specprod_dir', type=str, default=None, help='Path to spectro production folder.  Default is specprod_dir')
    parser.add_argument('--night', type=str, help='Night; required')
    #parser.add_argument('--S2N_plot', default=False, action='store_true',
    #                    help = 'Generate a S/N plot for the production (vs. xaxis)')
    #parser.add_argument('--ZP_plot', default=False, action='store_true',
    #                    help = 'Generate a ZP plot for the production (vs. xaxis)')
    #parser.add_argument('--xaxis', type=str, default='MJD', help='Specify x-axis for S/N and ZP plots')

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args) :

    from desispec.qa import QA_Night
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

    qa_prod = QA_Night(args.night, specprod_dir=specprod_dir, qaprod_dir=qaprod_dir)

    # Exposure plots
    if args.expid_series:
        # QATYPE-METRIC
        qa_prod.load_data()
        # SKYSUB RESID
        qatype, metric = 'SKYSUB', 'RESID' #args.expid_series.split('-')
        outfile = qaprod_dir+'/QA_{:s}_expid_{:s}-{:s}.png'.format(args.night, qatype, metric)
        dqqp.prod_time_series(qa_prod, qatype, metric, outfile=outfile,
                              bright_dark=args.bright_dark, exposures=True,
                              night=args.night, horiz_line=0.)
        # FLUXCALIB ZP
        qatype, metric = 'FLUXCALIB', 'ZP' #args.expid_series.split('-')
        outfile = qaprod_dir+'/QA_{:s}_expid_{:s}-{:s}.png'.format(args.night, qatype, metric)
        dqqp.prod_time_series(qa_prod, qatype, metric, outfile=outfile,
                              bright_dark=args.bright_dark, exposures=True,
                              night=args.night)

    ''' The stuff down here does not work, or has not been tested on Night QA
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
            dqqp.prod_channel_hist(qa_prod, 'SKYSUB', 'RESID', xlim=(-15,15), pp=pp, close=False)
            dqqp.prod_channel_hist(qa_prod, 'FLUXCALIB', 'MAX_ZP_OFF', pp=pp, close=False)
        # Finish
        print("Writing {:s}".format(outfile))
        pp.close()
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
    '''
