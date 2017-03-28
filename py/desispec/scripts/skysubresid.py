# Script for generating plots on SkySub residuals
from __future__ import absolute_import, division

from desiutil.log import get_logger
import argparse
import numpy as np


def parse(options=None):
    parser = argparse.ArgumentParser(description="Generate QA on Sky Subtraction residuals")

    parser.add_argument('--specprod_dir', type = str, default = None, required=True,
                        help = 'Path containing the exposures/directory to use')
    parser.add_argument('--expids', type=str, help='List of exposure IDs')
    parser.add_argument('--nights', type=str, help='List of nights to include')
    parser.add_argument('--channels', type=str, help='List of channels to include')

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
    from desispec.io.sky import read_sky
    from desispec.qa.qa_plots import skysub_resid
    import copy
    import pdb

    # Log
    log=get_logger()
    log.info("starting")

    # Exposures?
    if args.expids is not None:
        expids = [int(iarg) for iarg in args.expids.split(',')]
    else:
        expids = 'all'

    # Nights?
    if args.nights is not None:
        gdnights = [iarg for iarg in args.nights.split(',')]
    else:
        gdnights = 'all'

    # Channels?
    if args.channels is not None:
        gdchannels = [iarg for iarg in args.channels.split(',')]
    else:
        gdchannels = 'all'

    # Sky dict
    sky_dict = dict(wave=[], skyflux=[], res=[], count=0)
    channel_dict = dict(b=copy.deepcopy(sky_dict),
                        r=copy.deepcopy(sky_dict),
                        z=copy.deepcopy(sky_dict),
                        )
    # Loop on nights
    path_nights = glob.glob(args.specprod_dir+'/exposures/*')
    nights = [ipathn[ipathn.rfind('/')+1:] for ipathn in path_nights]
    for night in nights:
        if gdnights == 'all':
            pass
        else:
            if night not in gdnights:
                continue
                # Get em
        for exposure in get_exposures(night, specprod_dir = args.specprod_dir):
            # Check against input expids
            if expids == 'all':
                pass
            else:
                if exposure not in expids:
                    continue
            # Get em
            frames_dict = get_files(filetype=str('cframe'), night=night,
                    expid=exposure, specprod_dir=args.specprod_dir)
            for camera, cframe_fil in frames_dict.items():
                channel = camera[0]
                # Check against input
                if gdchannels == 'all':
                    pass
                else:
                    if channel not in gdchannels:
                        continue
                # Load frame
                log.info('Loading {:s}'.format(cframe_fil))
                cframe = read_frame(cframe_fil)
                if cframe.meta['FLAVOR'] in ['flat','arc']:  # Probably can't happen
                    continue
                # Sky
                sky_file = findfile(str('sky'), night=night, camera=camera,
                                    expid=exposure, specprod_dir=args.specprod_dir)
                skymodel = read_sky(sky_file)
                # Resid
                skyfibers = np.where(cframe.fibermap['OBJTYPE'] == 'SKY')[0]
                res = cframe.flux[skyfibers]
                flux = skymodel.flux[skyfibers] # Residuals
                tmp = np.outer(np.ones(flux.shape[0]), cframe.wave)
                # Append
                #from xastropy.xutils import xdebug as xdb
                #xdb.set_trace()
                channel_dict[channel]['wave'].append(tmp.flatten())
                channel_dict[channel]['skyflux'].append(
                        np.log10(np.maximum(flux.flatten(),1e-1)))
                channel_dict[channel]['res'].append(res.flatten())
                channel_dict[channel]['count'] += 1
    # Figure
    for channel in ['b', 'r', 'z']:
        if channel_dict[channel]['count'] > 0:
            sky_wave = np.concatenate(channel_dict[channel]['wave'])
            sky_flux = np.concatenate(channel_dict[channel]['skyflux'])
            sky_res = np.concatenate(channel_dict[channel]['res'])
            # Plot
            skysub_resid(sky_wave, sky_flux, sky_res,
                         outfile='tmp{:s}.png'.format(channel))
