
from __future__ import absolute_import, division


from desispec.io import read_frame
from desispec.io import read_fiberflat
from desispec.io import read_sky
from desispec.io import write_qa_frame
from desispec.io.fluxcalibration import read_stdstar_models
from desispec.io.fluxcalibration import write_flux_calibration
from desispec.io.qa import load_qa_frame
from desispec.fiberflat import apply_fiberflat
from desispec.sky import subtract_sky
from desispec.fluxcalibration import compute_flux_calibration
from desispec.log import get_logger
from desispec.qa import qa_plots

import argparse
import os
import os.path
import numpy as np
import sys


def parse(options=None):
    parser = argparse.ArgumentParser(description="Compute the flux calibration for a DESI frame using precomputed spectro-photometrically calibrated stellar models.")

    parser.add_argument('--infile', type = str, default = None, required=True,
                        help = 'path of DESI exposure frame fits file')
    parser.add_argument('--fiberflat', type = str, default = None, required=True,
                        help = 'path of DESI fiberflat fits file')
    parser.add_argument('--sky', type = str, default = None, required=True,
                        help = 'path of DESI sky fits file')
    parser.add_argument('--models', type = str, default = None, required=True,
                        help = 'path of spetro-photometric stellar spectra fits file')
    parser.add_argument('--chi2cut', type = float, default = 0., required=False,
                        help = 'apply a reduced chi2 cut for the selection of stars')
    parser.add_argument('--outfile', type = str, default = None, required=True,
                        help = 'path of DESI flux calbration fits file')
    parser.add_argument('--qafile', type=str, default=None, required=False,
                        help='path of QA file.')
    parser.add_argument('--qafig', type = str, default = None, required=False,
                        help = 'path of QA figure file')
    
    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args) :

    log=get_logger()

    log.info("read frame")
    # read frame
    frame = read_frame(args.infile)

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
    model_flux,model_wave,model_fibers,model_metadata=read_stdstar_models(args.models)

    if args.chi2cut > 0 :
        ok = np.where(model_metadata["CHI2DOF"]<args.chi2cut)[0]
        if ok.size == 0 :
            log.error("chi2cut has discarded all stars")
            sys.exit(12)
        nstars=model_flux.shape[0]
        nbad=nstars-ok.size
        if nbad>0 :
            log.warning("discarding %d star(s) out of %d because of chi2cut"%(nbad,nstars))
            model_flux=model_flux[ok]
            model_fibers=model_fibers[ok]
            model_metadata=model_metadata[:][ok]
    

    # check that the model_fibers are actually standard stars
    fibermap = frame.fibermap

    ## check whether star fibers from args.models are consistent with fibers from fibermap
    ## if not print the OBJTYPE from fibermap for the fibers numbers in args.models and exit
    
    w = np.where(fibermap["OBJTYPE"][model_fibers] != 'STD')[0]
    if len(w)>0:
        for i in model_fibers:
            log.error("inconsistency with spectrum %d, OBJTYPE='%s' in fibermap"%(i,fibermap["OBJTYPE"][i]))
        sys.exit(12)

    fluxcalib = compute_flux_calibration(frame, model_wave, model_flux, model_fibers)

    # QA
    if (args.qafile is not None):
        log.info("performing fluxcalib QA")
        # Load
        qaframe = load_qa_frame(args.qafile, frame, flavor=frame.meta['FLAVOR'])
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

    # write result
    write_flux_calibration(args.outfile, fluxcalib, header=frame.meta)

    log.info("successfully wrote %s"%args.outfile)

