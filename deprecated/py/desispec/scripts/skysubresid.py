"""
desispec.scripts.skysubresid
============================

Script for generating plots on SkySub residuals
"""
from __future__ import absolute_import, division

from desiutil.log import get_logger
import argparse
import numpy as np

from desispec.qa import __offline_qa_version__

def parse(options=None):
    parser = argparse.ArgumentParser(description="Generate QA on Sky Subtraction residuals [v{:s}]".format(__offline_qa_version__))
    parser.add_argument('--expid', type=int, help='Generate exposure plot on given exposure')
    parser.add_argument('--channels', type=str, help='List of channels to include')
    parser.add_argument('--prod', default=False, action="store_true", help="Results for full production run")
    parser.add_argument('--gauss', default=False, action="store_true", help="Expore Gaussianity for full production run")
    parser.add_argument('--nights', type=str, help='List of nights to limit prod plots')
    parser.add_argument('--skyline', default=False, action="store_true", help="Skyline residuals?")
    parser.add_argument('--qaprod_dir', type=str, default=None, help='Path to where QA figure files are generated.  Default is qaprod_dir')

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args



def main(args) :
    # imports
    import glob
    from desispec.io import findfile, makepath
    from desispec.io import get_exposures
    from desispec.io import get_files, get_nights
    from desispec.io import get_reduced_frames
    from desispec.io import specprod_root
    from desispec.io import qaprod_root
    from desispec.qa import utils as qa_utils
    import copy
    import pdb

    # Init
    specprod_dir = specprod_root()

    # Log
    log=get_logger()
    log.info("starting")

    # Path
    if args.qaprod_dir is not None:
        qaprod_dir = args.qaprod_dir
    else:
        qaprod_dir = qaprod_root()

    # Channels
    if args.channels is not None:
        channels = [iarg for iarg in args.channels.split(',')]
    else:
        channels = ['b','r','z']

    # Sky dict
    sky_dict = dict(wave=[], skyflux=[], res=[], count=0)
    channel_dict = dict(b=copy.deepcopy(sky_dict),
                        r=copy.deepcopy(sky_dict),
                        z=copy.deepcopy(sky_dict),
                        )
    # Nights
    if args.nights is not None:
        nights = [iarg for iarg in args.nights.split(',')]
    else:
        nights = None

    # Exposure plot?
    if args.expid is not None:
        # Nights
        if nights is None:
            nights = get_nights()
        nights.sort()
        # Find the exposure
        for night in nights:
            if args.expid in get_exposures(night, specprod_dir=specprod_dir):
                frames_dict = get_files(filetype=str('cframe'), night=night,
                                    expid=args.expid, specprod_dir=specprod_dir)
                # Loop on channel
                #for channel in ['b','r','z']:
                for channel in ['z']:
                    channel_dict[channel]['cameras'] = []
                    for camera, cframe_fil in frames_dict.items():
                        if channel in camera:
                            sky_file = findfile(str('sky'), night=night, camera=camera,
                                expid=args.expid, specprod_dir=specprod_dir)
                            wave, flux, res, _ = qa_utils.get_skyres(cframe_fil)
                            # Append
                            channel_dict[channel]['wave'].append(wave)
                            channel_dict[channel]['skyflux'].append(np.log10(np.maximum(flux,1e-1)))
                            channel_dict[channel]['res'].append(res)
                            channel_dict[channel]['cameras'].append(camera)
                            channel_dict[channel]['count'] += 1
                    if channel_dict[channel]['count'] > 0:
                        from desispec.qa.qa_plots import skysub_resid_series  # Hidden to help with debugging
                        skysub_resid_series(channel_dict[channel], 'wave',
                             outfile=qaprod_dir+'/QA_skyresid_wave_expid_{:d}{:s}.png'.format(args.expid, channel))
                        skysub_resid_series(channel_dict[channel], 'flux',
                             outfile=qaprod_dir+'/QA_skyresid_flux_expid_{:d}{:s}.png'.format(args.expid, channel))
        return


    # Skyline
    if args.skyline:
        from desispec.qa.qa_plots import skyline_resid
        # Loop on channel
        for channel in channels:
            cframes = get_reduced_frames(nights=nights, channels=[channel])
            if len(cframes) > 0:
                log.info("Loading sky residuals for {:d} cframes".format(len(cframes)))
                if len(cframes) == 1:
                    log.error('len(cframes)==1; starting debugging')
                    pdb.set_trace() # Need to call differently
                else:
                    sky_wave, sky_flux, sky_res, sky_ivar = qa_utils.get_skyres(
                        cframes, flatten=False)
                # Plot
                outfile=args.outdir+'/skyline_{:s}.png'.format(channel)
                log.info("Plotting to {:s}".format(outfile))
                skyline_resid(channel, sky_wave, sky_flux, sky_res, sky_ivar,
                              outfile=outfile)
        return

    # Full Prod Plot?
    if args.prod:
        from desispec.qa.qa_plots import skysub_resid_dual
        # Loop on channel
        for channel in channels:
            cframes = get_reduced_frames(nights=nights, channels=[channel])
            if len(cframes) > 0:
                log.info("Loading sky residuals for {:d} cframes".format(len(cframes)))
                sky_wave, sky_flux, sky_res, _ = qa_utils.get_skyres(cframes)
                # Plot
                outfile=qaprod_dir+'/skyresid_prod_dual_{:s}.png'.format(channel)
                makepath(outfile)
                log.info("Plotting to {:s}".format(outfile))
                skysub_resid_dual(sky_wave, sky_flux, sky_res, outfile=outfile)
        return

    # Test sky noise for Gaussianity
    if args.gauss:
        from desispec.qa.qa_plots import skysub_gauss
        # Loop on channel
        for channel in channels:
            cframes = get_reduced_frames(nights=nights, channels=[channel])
            if len(cframes) > 0:
                # Cut down for debugging
                #cframes = [cframes[ii] for ii in range(15)]
                #
                log.info("Loading sky residuals for {:d} cframes".format(len(cframes)))
                sky_wave, sky_flux, sky_res, sky_ivar = qa_utils.get_skyres(cframes)
                # Plot
                log.info("Plotting..")
                outfile=qaprod_dir+'/skyresid_prod_gauss_{:s}.png'.format(channel)
                makepath(outfile)
                skysub_gauss(sky_wave, sky_flux, sky_res, sky_ivar,
                                  outfile=outfile)
        return
