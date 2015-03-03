#!/usr/bin/env python
#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-

import argparse
import os.path
import glob

import desispec.io

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', action = 'store_true',
        help = 'Provide verbose reporting of progress.')
    parser.add_argument('--night', type = str, default = None, metavar = 'YYYYMMDD',
        help = 'Night to process in the format YYYYMMDD')
    parser.add_argument('--specprod', type = str, default = None, metavar = 'PATH',
        help = 'Override default path ($DESI_SPECTRO_REDUX/$PRODNAME) to processed data.')
    args = parser.parse_args()

    if args.night is None:
        print 'Missing required night argument.'
        return -1

    try:
        # Loop over exposures available for this night.
        for exposure in desispec.io.get_exposures(args.night,specprod = args.specprod):
            # Ignore exposures with no fibermap, assuming they are calibration data.
            fibermap_path = desispec.io.findfile(filetype = 'fibermap',night = args.night,
                expid = exposure,specprod = args.specprod)
            if not os.path.exists(fibermap_path):
                if args.verbose:
                    print 'Skipping exposure %08d with no fibermap.' % exposure
                continue
            # Open the fibermap.
            fibermap_data,fibermap_hdr = desispec.io.read_fibermap(fibermap_path)
            # Get the set of bricknames used in this fibermap.
            bricknames = set(fibermap_data['BRICKNAME'])
            # Get the list of per-camera cframes available for this exposure.
            cframes = desispec.io.get_files(filetype = 'cframe',night = args.night,
                expid = exposure,specprod = args.specprod)
            if args.verbose:
                print 'Exposure %08d: %d bricks, cframes: %s' % (
                    exposure,len(bricknames),' '.join(cframes.keys()))

    except RuntimeError,e:
        print str(e)
        return -2

if __name__ == '__main__':
    main()
