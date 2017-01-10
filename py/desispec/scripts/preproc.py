"""
desispec.scripts.preproc
========================

Command line wrappers for pre-processing a DESI raw exposure
"""
from __future__ import absolute_import, division

import argparse
from pkg_resources import resource_exists, resource_filename

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

    parser.add_argument('--cosmics-nsig', type = float, default = 6, required=False,
                        help = 'for cosmic ray rejection : number of sigma above background required')
    parser.add_argument('--cosmics-cfudge', type = float, default = 3, required=False,
                        help = 'for cosmic ray rejection : number of sigma inconsistent with PSF required')
    parser.add_argument('--cosmics-c2fudge', type = float, default = 0.8, required=False,
                        help = 'for cosmic ray rejection : fudge factor applied to PSF')

    parser.add_argument('--bkgsub', action='store_true',
                        help = 'do a background subtraction prior to cosmic ray rejection')
    parser.add_argument('--nocosmic', action='store_true', 
                        help = 'do not try and reject cosmic rays')
    parser.add_argument('--no-ccd-calib-filename', action='store_true',
                        help = 'do not read calibration data from yaml file in desispec')
    parser.add_argument('--ccd-calib-filename', required=False, default=None,
                        help = 'specify a difference ccd calibration filename (for dev. purpose), default is in desispec/data/ccd')
    
    
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

    if args.no_ccd_calib_filename :
        ccd_calibration_filename = None
    else :
        if args.ccd_calib_filename is not None :
            ccd_calibration_filename = args.ccd_calib_filename
        else :
            # find it in desispec
            srch_file = "data/ccd/ccd_calibration.yaml"
            if not resource_exists('desispec', srch_file):
                log.error("Cannot find CCD calibration file {:s}".format(srch_file))
                ccd_calibration_filename = None        
            else :
                ccd_calibration_filename=resource_filename('desispec', srch_file)
    
    for camera in args.cameras:
        try:
            img = io.read_raw(args.infile, camera,

                              bias=args.bias, pixflat=args.pixflat, mask=args.mask, bkgsub=args.bkgsub, nocosmic=args.nocosmic,
                              cosmics_nsig=args.cosmics_nsig,
                              cosmics_cfudge=args.cosmics_cfudge,
                              cosmics_c2fudge=args.cosmics_c2fudge,
                              ccd_calibration_filename=ccd_calibration_filename
            )
        except IOError:
            log.error('Error while reading or preprocessing camera {} in {}'.format(camera, args.infile))
            continue

        if args.pixfile is None:
            night = img.meta['NIGHT']
            expid = img.meta['EXPID']
            pixfile = io.findfile('pix', night=night, expid=expid, camera=camera,
                                  outdir=args.outdir)
        else:
            pixfile = args.pixfile

        io.write_image(pixfile, img)
        
        
    
    
