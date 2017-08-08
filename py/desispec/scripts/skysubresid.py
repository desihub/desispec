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
    #parser.add_argument('--prod', type=str, help='Generate night plot')
    #parser.add_argument('--channels', type=str, help='List of channels to include')
    #parser.add_argument('--frame', type=str, help='List of exposure IDs')

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args

def get_skyres(cframe_fil, sky_file):
    from desispec.io import read_frame
    from desispec.io.sky import read_sky

    cframe = read_frame(cframe_fil)
    if cframe.meta['FLAVOR'] in ['flat','arc']:
        raise ValueError("Bad flavor for exposure: {:s}".format(cframe_fil))

    # Sky
    skymodel = read_sky(sky_file)
    # Resid
    skyfibers = np.where(cframe.fibermap['OBJTYPE'] == 'SKY')[0]
    res = cframe.flux[skyfibers]
    flux = skymodel.flux[skyfibers] # Residuals
    wave = np.outer(np.ones(flux.shape[0]), cframe.wave).flatten()
    # Return
    return wave, flux, res


def main(args) :
    # imports
    import glob
    from desispec.io import findfile
    from desispec.io import get_exposures
    from desispec.io import get_files
    from desispec.io import specprod_root
    from desispec.qa.qa_plots import skysub_resid_series, skysub_resid_dual
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
                for channel in ['b']:
                    channel_dict[channel]['cameras'] = []
                    for camera, cframe_fil in frames_dict.items():
                        if channel in camera:
                            sky_file = findfile(str('sky'), night=night, camera=camera,
                                expid=args.expid, specprod_dir=specprod_dir)
                            wave, flux, res = get_skyres(cframe_fil, sky_file)
                            # Append
                            channel_dict[channel]['wave'].append(wave)
                            channel_dict[channel]['skyflux'].append(np.log10(np.maximum(flux.flatten(),1e-1)))
                            channel_dict[channel]['res'].append(res.flatten())
                            channel_dict[channel]['cameras'].append(camera)
                            channel_dict[channel]['count'] += 1
                    if channel_dict[channel]['count'] > 0:
                        skysub_resid_series(channel_dict[channel], 'wave',
                             outfile='QA_skyresid_wave_expid_{:d}{:s}.png'.format(args.expid, channel))
                        skysub_resid_series(channel_dict[channel], 'flux',
                                            outfile='QA_skyresid_flux_expid_{:d}{:s}.png'.format(args.expid, channel))
        return




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

    # Loop on nights
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
                sky_file = findfile(str('sky'), night=night, camera=camera,
                                    expid=exposure, specprod_dir=specprod_dir)
                # Append
                channel_dict[channel]['wave'].append(tmp.flatten())
                channel_dict[channel]['skyflux'].append(np.log10(np.maximum(flux.flatten(),1e-1)))
                channel_dict[channel]['res'].append(res.flatten())
                channel_dict[channel]['count'] += 1
    # Figure
    for channel in ['b', 'r', 'z']:
        if channel_dict[channel]['count'] > 0:
            sky_wave = np.concatenate(channel_dict[channel]['wave'])
            sky_flux = np.concatenate(channel_dict[channel]['skyflux'])
            sky_res = np.concatenate(channel_dict[channel]['res'])
            # Plot
            skysub_resid_dual(sky_wave, sky_flux, sky_res,
                         outfile='tmp{:s}.png'.format(channel))
