"""
desispec.scripts.sky
====================

"""

from __future__ import absolute_import, division

from desispec.io import read_frame
from desispec.io import read_fiberflat
from desispec.io import write_sky
from desispec.io import shorten_filename
from desispec.io import write_skycorr
from desispec.io import read_skycorr_pca
from desispec.io import read_skygradpca
from desispec.io import read_tpcorrparam
from desispec.skycorr import SkyCorr
from desispec.fiberflat import apply_fiberflat
from desispec.sky import compute_sky
from desispec.cosmics import reject_cosmic_rays_1d
from desiutil.log import get_logger
import argparse
import numpy as np


def parse(options=None):
    parser = argparse.ArgumentParser(description="Compute the sky model.")

    parser.add_argument('-i','--infile', type = str, default = None, required=True,
                        help = 'path of DESI exposure frame fits file')
    parser.add_argument('--fiberflat', type = str, default = None, required=True,
                        help = 'path of DESI fiberflat fits file')
    parser.add_argument('-o','--outfile', type = str, default = None, required=True,
                        help = 'path of DESI sky fits file')
    parser.add_argument('--cosmics-nsig', type = float, default = 0, required=False,
                        help = 'n sigma rejection for cosmics in 1D (default, no rejection)')
    parser.add_argument('--no-extra-variance', action='store_true',
                        help = 'do not increase sky model variance based on chi2 on sky lines')
    parser.add_argument('--adjust-wavelength', action='store_true',
                        help = 'adjust wavelength calibration of sky model on sky lines to improve sky subtraction for all fibers')
    parser.add_argument('--adjust-lsf', action='store_true',
                        help = 'adjust LSF width of sky model on sky lines to improve sky subtraction for all fibers')
    parser.add_argument('--adjust-with-more-fibers', action='store_true',
                        help = 'use more fibers than just the sky fibers for the adjustements')
    parser.add_argument('--save-adjustments', type = str , default = None, required=False,
                        help = 'save adjustments of wavelength calib and LSF width in table')
    parser.add_argument('--pca-corr', type = str , default = None, required=False,
                        help = 'use this PCA frames file for interpolation of wavelength and/or LSF adjustment')
    parser.add_argument('--fit-offsets', action = 'store_true', default = False, required=False,
                        help = 'fit offsets in sectors of CCD specified in calib yaml file, like OFFCOLSD:"2057:3715" for columns 2057 to 3715 (excluded), in amplifier D')
    parser.add_argument('--skygradpca', type=str, default=None, required=False,
                        help = 'file name of sky gradient PCA file for fitting sky gradients')
    parser.add_argument('--tpcorrparam', type=str, default=None, required=False,
                        help = 'file name of tpcorr parameter file for fitting fiber throughputs')
    parser.add_argument('--exclude-sky-targetids', type = str, default = None, required = False,
                        help = 'List of TARGETIDs to exclude from sky calculation (comma separated)')
    parser.add_argument('--override-sky-targetids', type = str, default = None, required = False,
                        help='List of TARGETIDs to use as skies to completely override fibermap info (comma separated)')

    args = parser.parse_args(options)

    return args


def main(args=None) :

    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    log=get_logger()

    if args.save_adjustments and ((not args.adjust_lsf) or (not args.adjust_wavelength)) :
        mess="need both options --adjust-wavelength and --adjust-lsf to run with --save-adjustments"
        log.error(mess)
        raise Exception(mess)

    if args.exclude_sky_targetids is not None:
        exclude_sky_targetids=[int(_) for _ in args.exclude_sky_targetids.split(',')]
    else:
        exclude_sky_targetids = None
    if args.override_sky_targetids is not None:
        override_sky_targetids=[int(_) for _ in args.override_sky_targetids.split(',')]
    else:
        override_sky_targetids = None

    log.info("starting")

    # read exposure to load data and get range of spectra
    frame = read_frame(args.infile)
    specmin, specmax = np.min(frame.fibers), np.max(frame.fibers)

    if args.cosmics_nsig>0 : # Reject cosmics
        reject_cosmic_rays_1d(frame,args.cosmics_nsig)

    # read fiberflat
    fiberflat = read_fiberflat(args.fiberflat)

    # apply fiberflat to sky fibers
    apply_fiberflat(frame, fiberflat)

    # load pca corr if set
    if args.pca_corr is not None :
        pcacorr = read_skycorr_pca(args.pca_corr)
    else :
        pcacorr = None

    if args.skygradpca is not None:
        skygradpca = read_skygradpca(args.skygradpca)
    else:
        skygradpca = None

    if args.tpcorrparam is not None:
        tpcorrparam = read_tpcorrparam(args.tpcorrparam)
    else:
        tpcorrparam = None



    # compute sky model
    skymodel = compute_sky(frame,add_variance=(not args.no_extra_variance),\
                           adjust_wavelength=args.adjust_wavelength,\
                           adjust_lsf=args.adjust_lsf,\
                           only_use_skyfibers_for_adjustments=(not args.adjust_with_more_fibers),\
                           pcacorr=pcacorr,fit_offsets=args.fit_offsets,fiberflat=fiberflat,
                           skygradpca=skygradpca, tpcorrparam=tpcorrparam,
                           exclude_sky_targetids=exclude_sky_targetids,
                           override_sky_targetids=override_sky_targetids
    )

    if args.save_adjustments is not None :
        skycorr=SkyCorr(wave=skymodel.wave,dwave=skymodel.dwave,dlsf=skymodel.dlsf,header=skymodel.header)
        write_skycorr(args.save_adjustments,skycorr)
        log.info("wrote {}".format(args.save_adjustments))

    # record inputs
    frame.meta['IN_FRAME'] = shorten_filename(args.infile)
    frame.meta['FIBERFLT'] = shorten_filename(args.fiberflat)

    # write result
    write_sky(args.outfile, skymodel, frame.meta)
    log.info("successfully wrote %s"%args.outfile)
