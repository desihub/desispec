# Script for generating plots on SkySub residuals
from __future__ import absolute_import, division

from desiutil.log import get_logger
import argparse
import numpy as np


def parse(options=None):
    parser = argparse.ArgumentParser(description="Generate QA on Sky Subtraction residuals")
    parser.add_argument('--reduxdir', type = str, default = None, metavar = 'PATH',
                        help = 'Override default path ($DESI_SPECTRO_REDUX/$SPECPROD) to processed data.')
    parser.add_argument('--expid', type=int, help='Generate exposure plot on given exposure')
    parser.add_argument('--night', type=str, help='Generate night plot on given night')
    parser.add_argument('--nights', type=str, help='List of nights to include for prod plot')
    parser.add_argument('--channels', type=str, help='List of channels to include')
    #parser.add_argument('--frame', type=str, help='List of exposure IDs')

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args



def main(args) :
    # imports
    import glob
    from desispec.io import findfile
    from desispec.io import get_exposures
    from desispec.io import get_files
    from desispec.io import read_frame
    from desispec.io import get_reduced_frames
    from desispec.io.sky import read_sky
    from desispec.io import specprod_root
    from desispec.qa import utils as qa_utils
    import copy
    import pdb

    # Path
    if args.reduxdir is not None:
        specprod_dir = args.reduxdir
    else:
        specprod_dir = specprod_root()

    # Nights
    path_nights = glob.glob(specprod_dir+'/exposures/*')
    nights = [ipathn[ipathn.rfind('/')+1:] for ipathn in path_nights]
    nights.sort()

    # Sky dict
    sky_dict = dict(wave=[], skyflux=[], res=[], count=0)
    channel_dict = dict(b=copy.deepcopy(sky_dict),
                        r=copy.deepcopy(sky_dict),
                        z=copy.deepcopy(sky_dict),
                        )

    # Log
    log=get_logger()
    log.info("starting")

    # Exposure plot?
    if args.expid is not None:
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
                            wave, flux, res, _ = qa_utils.get_skyres(cframe_fil, sky_file)#, sub_sky=True)
                            # Append
                            channel_dict[channel]['wave'].append(wave)
                            channel_dict[channel]['skyflux'].append(np.log10(np.maximum(flux,1e-1)))
                            channel_dict[channel]['res'].append(res)
                            channel_dict[channel]['cameras'].append(camera)
                            channel_dict[channel]['count'] += 1
                    if channel_dict[channel]['count'] > 0:
                        from desispec.qa.qa_plots import skysub_resid_series  # Hidden to help with debugging
                        skysub_resid_series(channel_dict[channel], 'wave',
                             outfile='QA_skyresid_wave_expid_{:d}{:s}.png'.format(args.expid, channel))
                        skysub_resid_series(channel_dict[channel], 'flux',
                                            outfile='QA_skyresid_flux_expid_{:d}{:s}.png'.format(args.expid, channel))
        return


    # Full Prod Plot
    # Nights
    if args.nights is not None:
        nights = [iarg for iarg in args.nights.split(',')]
    else:
        nights = None

    # Channels
    if args.channels is not None:
        channels = [iarg for iarg in args.channels.split(',')]
    else:
        channels = ['b','r','z']

    # Loop on nights
    # Sky dict
    sky_dict = dict(wave=[], skyflux=[], res=[], count=0)
    channel_dict = dict(b=copy.deepcopy(sky_dict),
                        r=copy.deepcopy(sky_dict),
                        z=copy.deepcopy(sky_dict),
                        )
    # Loop on channel
    from desispec.qa.qa_plots import skysub_resid_dual
    for channel in channels:
        cframes = get_reduced_frames(nights=nights, channels=[channel])
        if len(cframes) > 0:
            log.info("Loading sky residuals for {:d} cframes".format(len(cframes)))
            sky_wave, sky_flux, sky_res, _ = qa_utils.get_skyres(cframes)
            # Plot
            log.info("Plotting..")
            skysub_resid_dual(sky_wave, sky_flux, sky_res,
                         outfile='skyresid_prod_dual_{:s}.png'.format(channel))
