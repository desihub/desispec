"""
desispec.scripts.preproc
========================

Command line wrappers for pre-processing a DESI raw exposure
"""
from __future__ import absolute_import, division

import argparse

import os
from desispec import io
from desispec.log import get_logger
log = get_logger()

def parse(options=None):
    parser = argparse.ArgumentParser(
        description="Preprocess DESI raw data",
        epilog='''By default, all HDUs of the input file are processed and
written to pix*.fits files in the current directory; use --outdir to override
the output directory location.  --pixfile PIXFILE can be used to
override a single output filename but if only a single
camera is given with --cameras.
--bias/--pixflat/--mask can specify the calibration files
to use, but also only if a single camera is specified.
        ''')
    parser.add_argument('--infile', type = str, default = None, required=True,
                        help = 'path of DESI raw data file')
    parser.add_argument('--outdir', type = str, default = None, required=False,
                        help = 'output directory')
    parser.add_argument('--pixfile', type = str, default = None, required=False,
                        help = 'output preprocessed pixfile')
    parser.add_argument('--cameras', type = str, default = None, required=False,
                        help = 'comma separated list of cameras')
    parser.add_argument('--bias', type = str, default = None, required=False,
                        help = 'bias image calibration file')
    parser.add_argument('--pixflat', type = str, default = None, required=False,
                        help = 'pixflat image calibration file')
    parser.add_argument('--mask', type = str, default = None, required=False,
                        help = 'mask image calibration file')

    #- uses sys.argv if options=None
    args = parser.parse_args(options)
    
    return args
    
def main(args=None):
    if args is None:
        args = parse()
    elif isinstance(args, (list, tuple)):
        args = parse(args)
        
    if args.cameras is None:
        args.cameras = [c+str(i) for c in 'brz' for i in range(10)]
    else:
        args.cameras = args.cameras.split(',')
    
    if (args.bias is not None) or (args.pixflat is not None) or (args.mask is not None):
        if len(args.cameras) > 1:
            raise ValueError('must use only one camera with --bias, --pixflat, --mask options')
    
    if (args.pixfile is not None) and len(args.cameras) > 1:
            raise ValueError('must use only one camera with --pixfile option')

    if args.outdir is None:
        args.outdir = os.getcwd()
    
    for camera in args.cameras:
        try:
            img = io.read_raw(args.infile, camera,
                bias=args.bias, pixflat=args.pixflat, mask=args.mask)
        except IOError:
            log.error('Camera {} not in {}'.format(camera, args.infile))
            continue

        if args.pixfile is None:
            night = img.meta['NIGHT']
            expid = img.meta['EXPID']
            pixfile = io.findfile('pix', night=night, expid=expid, camera=camera,
                                  outdir=args.outdir)
        else:
            pixfile = args.pixfile

        io.write_image(pixfile, img)
        
        
    
    