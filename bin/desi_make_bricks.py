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
    parser.add_argument('--night', default = None, metavar = 'YYYYMMDD',
        help = 'Night to process in the format YYYYMMDD')
    args = parser.parse_args()

    if args.night is None:
        print 'Missing required night argument.'
        return -1

    try:
        # Loop over exposures available for this night.
        for exposure in desispec.io.get_exposures(args.night):
            # Ignore exposures with no fibermap, assuming they are calibration data.
            fibermap_path = desispec.io.findfile(filetype = 'fibermap', night = args.night,
                expid = exposure)
            print fibermap_path
            if not os.path.exists(fibermap_path):
                if args.verbose:
                    print 'Skipping exposure %d with no fibermap.' % exposure
                continue
            # Open the fibermap.
            fibermap_data,fibermap_hdr = desispec.io.read_fibermap(fibermap_path)
            print fibermap_data.dtype
            # Look for cframes associated with this exposure.
            cframe_path = desispec.io.findfile(filetype = 'cframe',night = args.night,
                expid = exposure, camera = '*')
            for entry in glob.glob(cframe_path):
                print entry

    except RuntimeError,e:
        print str(e)
        return -2

if __name__ == '__main__':
    main()
