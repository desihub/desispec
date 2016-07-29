# Script for generating QA from a Production run
from __future__ import absolute_import, division

from desispec.log import get_logger
import argparse
import numpy as np


def parse(options=None):
    parser = argparse.ArgumentParser(description="Generate QA on Sky Subtraction residuals")

    parser.add_argument('--specprod_dir', type = str, default = None, required=True,
                        help = 'Path containing the exposures/directory to use')
    parser.add_argument('--expids', type = int, help = 'List of exposure IDs')

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
    import pdb

    # Log
    log=get_logger()
    log.info("starting")

    # Exposures?
    if args.expids is not None:
        expids = [int(iarg) for iarg in args.expids.split(',')]
    else:
        expids = 'all'

    # Sky dict
    sky_dict = dict(wave=[], flux=[], res=[], count=0)
    channel_dict = dict(b=sky_dict.copy(), r=sky_dict.copy(), z=sky_dict.copy())

    # Loop on nights
    path_nights = glob.glob(args.specprod_dir+'/exposures/*')
    nights = [ipathn[ipathn.rfind('/')+1:] for ipathn in path_nights]
    for night in nights:
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
                # Load frame
                cframe = read_frame(cframe_fil)
                if cframe.meta['FLAVOR'] in ['flat','arc']:  # Probably can't happen
                    continue
                # Sky
                sky_file = findfile(str('sky'), night=night, camera=camera,
                                    expid=exposure, specprod_dir=args.specprod_dir)
                skymodel = read_sky(sky_file)
                # Resid
                skyfibers = np.where(cframe.fibermap['OBJTYPE'] == 'SKY')[0]
                flux = cframe.flux[skyfibers]
                res = flux - skymodel.flux[skyfibers] # Residuals
                pdb.set_trace()
                # Append
                channel_dict[channel]['wave'].append(cframe.wave[skyfibers])
                channel_dict[channel]['flux'].append(flux)
                channel_dict[channel]['res'].append(res)
                channel_dict[channel]['count'] += 1



