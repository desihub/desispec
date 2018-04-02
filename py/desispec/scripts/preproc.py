"""
desispec.scripts.preproc
========================

Command line wrappers for pre-processing a DESI raw exposure
"""
from __future__ import absolute_import, division

import argparse

import os
import sys
from desispec import io
from desiutil.log import get_logger
log = get_logger()

def parse(options=None):
    parser = argparse.ArgumentParser(
        description="Preprocess DESI raw data",
        epilog='''By default, all HDUs of the input file are processed and
written to pix*.fits files in the current directory; use --outdir to override
the output directory location.  --outfile FILENAME can be used to
override a single output filename but only if a single
camera is given with --cameras.
--bias/--pixflat/--mask can specify the calibration files
to use, but also only if a single camera is specified.
        ''')
    parser.add_argument('-i','--infile', type = str, default = None, required=True,
                        help = 'path of DESI raw data file')
    parser.add_argument('--outdir', type = str, default = None, required=False,
                        help = 'output directory')
    parser.add_argument('--pixfile', type = str, default = None, required=False,
                        help = 'DEPRECATED: use --outfile instead')
    parser.add_argument('-o','--outfile', type = str, default = None, required=False,
                        help = 'output preprocessed image file')
    parser.add_argument('--cameras', type = str, default = None, required=False,
                        help = 'comma separated list of cameras')
    parser.add_argument('--bias', type = str, default = None, required=False,
                        help = 'bias image calibration file')
    parser.add_argument('--dark', type = str, default = None, required=False,
                        help = 'dark image calibration file')
    parser.add_argument('--pixflat', type = str, default = None, required=False,
                        help = 'pixflat image calibration file')
    parser.add_argument('--mask', type = str, default = None, required=False,
                        help = 'mask image calibration file')
    parser.add_argument('--nobias', action = 'store_true',
                        help = 'no bias subtraction')
    parser.add_argument('--nodark', action = 'store_true',
                        help = 'no dark subtraction')
    parser.add_argument('--nopixflat',action = 'store_true',
                        help = 'no pixflat correction')
    parser.add_argument('--nomask', action = 'store_true',
                        help = 'no prior masking of pixels')
    parser.add_argument('--nocrosstalk', action = 'store_true',
                        help = 'no cross-talk correction')
    parser.add_argument('--nocosmic', action='store_true',
                        help = 'do not try and reject cosmic rays')
    parser.add_argument('--nogain', action='store_true',
                        help = 'do not apply gain correction') 
    
    parser.add_argument('--cosmics-nsig', type = float, default = 6, required=False,
                        help = 'for cosmic ray rejection : number of sigma above background required')
    parser.add_argument('--cosmics-cfudge', type = float, default = 3, required=False,
                        help = 'for cosmic ray rejection : number of sigma inconsistent with PSF required')
    parser.add_argument('--cosmics-c2fudge', type = float, default = 0.8, required=False,
                        help = 'for cosmic ray rejection : fudge factor applied to PSF')

    parser.add_argument('--bkgsub', action='store_true',
                        help = 'do a background subtraction prior to cosmic ray rejection')
    
    parser.add_argument('--zero-masked', action='store_true',
                        help = 'set to zero the flux of masked pixels (for convenience to display images, no impact on analysis)')
    parser.add_argument('--no-ccd-calib-filename', action='store_true',
                        help = 'do not read calibration data from yaml file in desispec')
    parser.add_argument('--ccd-calib-filename', required=False, default=None,
                        help = 'specify a difference ccd calibration filename (for dev. purpose), default is in desispec/data/ccd')
    parser.add_argument('--fill-header', type = str, default = None,  nargs ='*', help="fill camera header with contents of those of other hdus")
    
    #- uses sys.argv if options=None
    args = parser.parse_args(options)

    return args

def main(args=None):
    if args is None:
        args = parse()
    elif isinstance(args, (list, tuple)):
        args = parse(args)

    bias=True
    if args.bias : bias=args.bias
    if args.nobias : bias=False
    dark=True
    if args.dark : dark=args.dark
    if args.nodark : dark=False
    pixflat=True
    if args.pixflat : pixflat=args.pixflat
    if args.nopixflat : pixflat=False
    mask=True
    if args.mask : mask=args.mask
    if args.nomask : mask=False



    if args.cameras is None:
        args.cameras = [c+str(i) for c in 'brz' for i in range(10)]
    else:
        args.cameras = args.cameras.split(',')

    if (args.bias is not None) or (args.pixflat is not None) or (args.mask is not None) or (args.dark is not None):
        if len(args.cameras) > 1:
            raise ValueError('must use only one camera with --bias, --dark, --pixflat, --mask options')

    if (args.pixfile is not None):
        log.warning('--pixfile is deprecated; please use --outfile instead')
        if args.outfile is None:
            args.outfile = args.pixfile
        else:
            log.critical("Set --outfile not --pixfile and certainly not both")
            sys.exit(1)

    if (args.outfile is not None) and len(args.cameras) > 1:
            raise ValueError('must use only one camera with --outfile option')

    if args.outdir is None:
        args.outdir = os.getcwd()
        log.warning('--outdir not specified; using {}'.format(args.outdir))

    ccd_calibration_filename = None

    if args.no_ccd_calib_filename :
        ccd_calibration_filename = False
    elif args.ccd_calib_filename is not None :
        ccd_calibration_filename = args.ccd_calib_filename


    for camera in args.cameras:
        try:
            img = io.read_raw(args.infile, camera,

                              bias=bias, dark=dark, pixflat=pixflat, mask=mask, bkgsub=args.bkgsub,
                              nocosmic=args.nocosmic,                              
                              cosmics_nsig=args.cosmics_nsig,
                              cosmics_cfudge=args.cosmics_cfudge,
                              cosmics_c2fudge=args.cosmics_c2fudge,
                              ccd_calibration_filename=ccd_calibration_filename,
                              nocrosstalk=args.nocrosstalk,
                              nogain=args.nogain,
                              fill_header=args.fill_header
            )
        except IOError:
            log.error('Error while reading or preprocessing camera {} in {}'.format(camera, args.infile))
            continue

        if(args.zero_masked) :
            img.pix *= (img.mask==0)

        if args.outfile is None:
            night = img.meta['NIGHT']
            expid = img.meta['EXPID']
            outfile = io.findfile('preproc', night=night, expid=expid, camera=camera,
                                  outdir=args.outdir)
        else:
            outfile = args.outfile

        io.write_image(outfile, img)
