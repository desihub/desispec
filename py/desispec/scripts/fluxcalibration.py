
from __future__ import absolute_import, division


import desispec.fluxcalibration
from desispec.io import read_frame
from desispec.io import read_fiberflat
from desispec.io import read_sky
from desispec.io import shorten_filename
from desispec.io.fluxcalibration import read_stdstar_models
from desispec.io.fluxcalibration import write_flux_calibration
from desispec.fiberflat import apply_fiberflat
from desispec.sky import subtract_sky
from desispec.fluxcalibration import compute_flux_calibration, isStdStar
from desiutil.log import get_logger
from desitarget.targets import main_cmx_or_sv
from desispec.fiberbitmasking import get_fiberbitmasked_frame

import argparse
import os
import os.path
import numpy as np
import sys
from astropy.table import Table

def parse(options=None):
    parser = argparse.ArgumentParser(description="Compute the flux calibration for a DESI frame using precomputed spectro-photometrically calibrated stellar models.")

    parser.add_argument('--infile', type = str, default = None, required=True,
                        help = 'path of DESI exposure frame fits file')
    parser.add_argument('--fiberflat', type = str, default = None, required=True,
                        help = 'path of DESI fiberflat fits file')
    parser.add_argument('--sky', type = str, default = None, required=True,
                        help = 'path of DESI sky fits file')
    parser.add_argument('--models', type = str, default = None, required=True,
                        help = 'path of spectro-photometric stellar spectra fits file')
    parser.add_argument('--selected-calibration-stars', type = str, default = None, required=False,
                        help = 'path to table with list of pre-selected calibration stars')
    parser.add_argument('--chi2cut', type = float, default = 0., required=False,
                        help = 'apply a reduced chi2 cut for the selection of stars')
    parser.add_argument('--chi2cut-nsig', type = float, default = 0., required=False,
                        help = 'discard n-sigma outliers from the reduced chi2 of the standard star fit')
    parser.add_argument('--color', type = str, default = None, required=False,
                        help = 'color used for filtering. Can be G-R R-Z or GAIA-BP-RP or GAIA-G-RP')
    parser.add_argument('--min-color', type = float, default = None, required=False,
                        help = 'only consider stars with color greater than this')
    parser.add_argument('--delta-color-cut', type = float, default = 0.2, required=False,
                        help = 'discard model stars with different broad-band color from imaging')
    parser.add_argument('--nostdcheck', dest='nostdcheck',
                        help='Do not check the standards against flags in the FIBERMAP; just use objects in the model file', action='store_true')
    parser.add_argument('--outfile', type = str, default = None, required=True,
                        help = 'path of DESI flux calbration fits file')
    parser.add_argument('--qafile', type=str, default=None, required=False,
                        help='path of QA file.')
    parser.add_argument('--qafig', type = str, default = None, required=False,
                        help = 'path of QA figure file')
    parser.add_argument('--highest-throughput', type = int, default = 0, required=False,
                        help = 'use this number of stars ranked by highest throughput to normalize transmission (for DESI commissioning)')
    parser.add_argument('--seeing-fwhm', type = float, default = 1.1, required=False,
                        help = 'seeing FWHM in arcsec, used for fiberloss correction')
    parser.add_argument('--nsig-flux-scale', type = float, default = 3, required=False,
                       help = 'n sigma cutoff of the flux scale among standard stars')
    parser.add_argument("--use-gpu", action="store_true", help="Use GPUs")

    parser.set_defaults(nostdcheck=False)
    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args) :

    log=get_logger()

    cmd = ['desi_compute_fluxcalibration',]
    for key, value in args.__dict__.items():
        if value is not None:
            cmd += ['--'+key, str(value)]
    cmd = ' '.join(cmd)
    log.info(cmd)

    log.info("read frame")
    # read frame
    frame = read_frame(args.infile)

    # Set fibermask flagged spectra to have 0 flux and variance
    frame = get_fiberbitmasked_frame(frame, bitmask='flux',ivar_framemask=True)

    log.info("apply fiberflat")
    # read fiberflat
    fiberflat = read_fiberflat(args.fiberflat)

    # apply fiberflat
    apply_fiberflat(frame, fiberflat)

    log.info("subtract sky")
    # read sky
    skymodel=read_sky(args.sky)

    # subtract sky
    subtract_sky(frame, skymodel)

    log.info("compute flux calibration")

    # read models
    model_flux, model_wave, model_fibers, model_metadata=read_stdstar_models(args.models)

    if args.selected_calibration_stars is not None :
        table=Table.read(args.selected_calibration_stars)
        good=table["VALID"]==1
        good_models = np.in1d( model_fibers , table["FIBER"][good] )
        log.info("Selected {} good stars, fibers = {}, from {}".format(np.sum(good_models),model_fibers[good_models],args.selected_calibration_stars))
        model_flux   = model_flux[good_models]
        model_fibers = model_fibers[good_models]
        model_metadata = model_metadata[good_models]

        if args.delta_color_cut > 0 :
            log.warning("will ignore color cut because a preselected list of stars was given")
            args.delta_color_cut = 0
        if args.min_color is not None :
            log.warning("will ignore min color because a preselected list of stars was given")
            args.min_color = None
        if args.chi2cut_nsig > 0 :
            log.warning("will ignore chi2 cut because a preselected list of stars was given")
            args.chi2cut_nsig = 0
        if args.nsig_flux_scale > 0 :
            log.warning("set nsig_flux_scale because a preselected list of stars was given")
            args.nsig_flux_scale = 0.
    ok=np.ones(len(model_metadata),dtype=bool)

    if args.chi2cut > 0 :
        log.info("apply cut CHI2DOF<{}".format(args.chi2cut))
        good = (model_metadata["CHI2DOF"]<args.chi2cut)
        bad  = ~good
        ok  &= good
        if np.any(bad) :
            log.info(" discard {} stars with CHI2DOF= {}".format(np.sum(bad),list(model_metadata["CHI2DOF"][bad])))

    legacy_filters = ('G-R', 'R-Z')
    gaia_filters = ('GAIA-BP-RP', 'GAIA-G-RP')
    model_column_list = model_metadata.columns.names
    if args.color is None:
        if 'MODEL_G-R' in model_column_list:
            color = 'G-R'
        elif 'MODEL_GAIA-BP-RP' in model_column_list:
            log.info('Using Gaia filters')
            color ='GAIA-BP-RP'
        else:
            log.error("Can't find either G-R or BP-RP color in the model file.")
            sys.exit(15)
    else:
        if args.color not in legacy_filters and args.color not in gaia_filters:
            log.error('Color name {} is not allowed, must be one of {} {}'.format(args.color, legacy_filters,gaia_filters))
            sys.exit(14)
        color = args.color
        if color not in model_column_list:
            # This should't happen
            log.error('The color {} was not computed in the models'.format(color))
            sys.exit(16)


    if args.delta_color_cut > 0 :
        # check dust extinction values for those stars
        stars_ebv = np.array(frame.fibermap[model_fibers % 500]["EBV"])
        log.info("min max E(B-V) for std stars = {:4.3f} {:4.3f}".format(np.min(stars_ebv),np.max(stars_ebv)))
        star_gr_reddening_relative_error = 0.2 * stars_ebv
        log.info("Consider a g-r reddening sys. error in the range {:4.3f} {:4.3f}".format(np.min(star_gr_reddening_relative_error),np.max(star_gr_reddening_relative_error)))


        log.info("apply cut |delta color|<{}+reddening error".format(args.delta_color_cut))
        good = (np.abs(model_metadata["MODEL_"+color]-model_metadata["DATA_"+color])<args.delta_color_cut+star_gr_reddening_relative_error)
        bad  = ok&(~good)
        ok  &= good
        if np.any(bad) :
            vals=model_metadata["MODEL_"+color][bad]-model_metadata["DATA_"+color][bad]
            log.info(" discard {} stars with dcolor= {}".format(np.sum(bad),list(vals)))

    if args.min_color is not None :
        log.info("apply cut DATA_{}>{}".format(color, args.min_color))
        good = (model_metadata["DATA_{}".format(color)]>args.min_color)
        bad  = ok&(~good)
        ok  &= good
        if np.any(bad) :
            vals=model_metadata["DATA_{}".format(color)][bad]
            log.info(" discard {} stars with {}= {}".format(np.sum(bad),color,list(vals)))

    if args.chi2cut_nsig > 0 :
        # automatically reject stars that ar chi2 outliers
        mchi2=np.median(model_metadata["CHI2DOF"])
        rmschi2=np.std(model_metadata["CHI2DOF"])
        maxchi2=mchi2+args.chi2cut_nsig*rmschi2
        log.info("apply cut CHI2DOF<{} based on chi2cut_nsig={}".format(maxchi2,args.chi2cut_nsig))
        good = (model_metadata["CHI2DOF"]<=maxchi2)
        bad  = ok&(~good)
        ok  &= good
        if np.any(bad) :
            log.info(" discard {} stars with CHI2DOF={}".format(np.sum(bad),list(model_metadata["CHI2DOF"][bad])))

    ok=np.where(ok)[0]
    if ok.size == 0 :
        log.error("selection cuts discarded all stars")
        sys.exit(12)
    nstars=model_flux.shape[0]
    nbad=nstars-ok.size
    if nbad>0 :
        log.warning("discarding %d star(s) out of %d because of cuts"%(nbad,nstars))
        model_flux=model_flux[ok]
        model_fibers=model_fibers[ok]
        model_metadata=model_metadata[:][ok]

    stdcheck = not args.nostdcheck

    # check that the model_fibers are actually standard stars
    fibermap = frame.fibermap

    ## check whether star fibers from args.models are consistent with fibers from fibermap
    ## if not print the OBJTYPE from fibermap for the fibers numbers in args.models and exit
    if stdcheck:
        fibermap_std_indices = np.where(isStdStar(fibermap))[0]
        if np.any(~np.in1d(model_fibers%500, fibermap_std_indices)):
            target_colnames, target_masks, survey = main_cmx_or_sv(fibermap)
            colname =  target_colnames[0]
            for i in model_fibers%500:
                log.error("inconsistency with spectrum {}, OBJTYPE={}, {}={} in fibermap".format(
                i, fibermap["OBJTYPE"][i], colname, fibermap[colname][i]))
            sys.exit(12)
    else:
        fibermap_std_indices = model_fibers % 500
    # Make sure the fibers of interest aren't entirely masked.
    if np.sum(np.sum(frame.ivar[model_fibers%500, :] == 0, axis=1) == frame.nwave) == len(model_fibers):
        log.warning('All standard-star spectra are masked!')
        return

    if not args.use_gpu: desispec.fluxcalibration.use_gpu = False
    fluxcalib = compute_flux_calibration(frame, model_wave, model_flux,
            model_fibers%500,
            highest_throughput_nstars=args.highest_throughput,
            exposure_seeing_fwhm=args.seeing_fwhm,
            stdcheck=stdcheck, nsig_flux_scale= args.nsig_flux_scale)

    # QA
    if (args.qafile is not None):

        from desispec.io import write_qa_frame
        from desispec.io.qa import load_qa_frame
        from desispec.qa import qa_plots

        log.info("performing fluxcalib QA")
        # Load
        qaframe = load_qa_frame(args.qafile, frame_meta=frame.meta, flavor=frame.meta['FLAVOR'])
        # Run
        #import pdb; pdb.set_trace()
        qaframe.run_qa('FLUXCALIB', (frame, fluxcalib))
        # Write
        if args.qafile is not None:
            write_qa_frame(args.qafile, qaframe)
            log.info("successfully wrote {:s}".format(args.qafile))
        # Figure(s)
        if args.qafig is not None:
            qa_plots.frame_fluxcalib(args.qafig, qaframe, frame, fluxcalib)

    # record inputs
    frame.meta['IN_FRAME'] = shorten_filename(args.infile)
    frame.meta['IN_SKY']   = shorten_filename(args.sky)
    frame.meta['FIBERFLT'] = shorten_filename(args.fiberflat)
    frame.meta['STDMODEL'] = shorten_filename(args.models)

    # write result
    write_flux_calibration(args.outfile, fluxcalib, header=frame.meta)

    log.info("successfully wrote %s"%args.outfile)
