#!/usr/bin/env python
#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-

"""
This script finds cosmics in a pre-processed image and write the result in the mask extension of an output image
(output can be same as input).
"""

from desispec.io import image
from desispec.cosmics import reject_cosmic_rays
from desispec.log import get_logger
import argparse
import numpy as np


def main() :
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--infile', type = str, default = None, required=True,
                        help = 'path of DESI exposure image fits file')
    parser.add_argument('--outfile', type = str, default = None, 
                        help = 'path of DESI output exposure image fits file (default is overwriting input with new mask)')
    
    args = parser.parse_args()
    
    
    if args.outfile is not None :
        outfile=args.outfile
    else :
        outfile=args.infile
    

    log = get_logger()
    log.info("starting finding cosmics in %s"%args.infile)
    
    img=image.read_image(args.infile)
    
    reject_cosmic_rays(img)
    
    log.info("writing data and new mask in %s"%outfile)
    image.write_image(outfile, img, meta=img.meta)
        
    log.info("done")
    
if __name__ == '__main__':
    main()
