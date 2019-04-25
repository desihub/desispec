"""
desispec.scripts.qproc
========================

Run DESI qproc on a given exposure
"""


import argparse
import sys,os
import time
import numpy as np
import astropy.io.fits as fits
from desiutil.log import get_logger
from desispec.util import runcmd
from desispec.io import read_raw,read_image,read_fibermap,write_image,write_fiberflat,read_fiberflat
from desispec.io.fluxcalibration import read_average_flux_calibration
from desispec.io.xytraceset import read_xytraceset
from desispec.calibfinder import CalibFinder
from desispec.qproc.qframe import QFrame
from desispec.qproc.io import read_qframe,write_qframe
from desispec.qproc.qextract import qproc_boxcar_extraction
from desispec.qproc.qfiberflat import qproc_apply_fiberflat,qproc_compute_fiberflat
from desispec.qproc.qsky import qproc_sky_subtraction
from desispec.qproc.util import parse_fibers

def parse(options=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""Quick extraction and processing script. The input image file is either a raw data fits file from the ICS (in which case one must specify the camera) or a preprocessed image. The output is a frame (a series of spectra) that can be saved to disk and/or displayed. An approximate fiber flat field correction can be computed if the input is a dome screen exposure. For on-sky images, fiber flat field and a rudimentary sky subtraction can be performed. This script relies on the existence of a DESI_SPECTRO_CALIB (older version used DESI_CCD_CALIBRATION_DATA) environment variable pointing to a local copy of the DESI calibration SVN repository.
""",
                                     epilog="""Example: desi_qproc -i desi-00003577.fits --camera r1 --fibermap fibermap-00003577.fits --fibers 12:15 --plot"""
)
    parser.add_argument('-i','--image', type = str, default = None, required = True,
                        help = 'path to image fits file (either preprocessed or raw image)')
    parser.add_argument('-c','--camera', type = str, default = None, required = False,
                        help = 'has to specify camera if input image is raw data')
    parser.add_argument('-f','--fibermap', type = str, default = None, required = False,
                        help = 'path to fibermap file')
    parser.add_argument('-o','--outframe', type = str, default = None, required = False,
                        help = 'path to output qframe file')
    parser.add_argument('-p','--psf', type = str, default = None, required = False,
                        help = 'path to psf fits file to get the trace coordinates (default is psf in $DESI_CCD_CALIBRATION_DATA)')
    parser.add_argument('--output-preproc', type = str, default = None, required = False,
                        help = 'save the preprocessed image in this file.')
    parser.add_argument('--output-rawframe', type = str, default = None, required = False,
                        help = 'save the raw (before flatfield,sky sub,calibration) extracted qframe to this file.')
    parser.add_argument('--output-skyframe', type = str, default = None, required = False,
                        help = 'save the sky model in a qframe file (need --skysub)')
    parser.add_argument('--shifted-psf', type = str, default = None, required = False,
                        help = 'estimate spectral trace shifts and save them in this file prior to extraction.')
    parser.add_argument('--fibers', type=str, default = None, required = False,
                        help = 'defines from_to which fiber to work on. (ex: --fibers=50:60,4 means that only fibers 4, and fibers from 50 to 60 (excluded) will be extracted)')
    parser.add_argument('--width', type=int, default=7, required=False,
                        help = 'extraction line width (in pixels)')
    parser.add_argument('--plot', action='store_true',
                        help = 'plot result')
    parser.add_argument('--compute-fiberflat', type = str, default = None, required = False,
                        help = 'compute flat and save it to this file')
    parser.add_argument('--apply-fiberflat', action='store_true',
                        help = 'apply fiber flat field (use default from $DESI_CCD_CALIBRATION_DATA if input-fiberflat not provide)')
    parser.add_argument('--input-fiberflat', type = str, default = None, required = False,
                        help = 'use this fiberflat file and apply it')
    parser.add_argument('--skysub', action='store_true',
                        help = 'perform as simple sky subtraction')
    parser.add_argument('--fluxcalib', action='store_true',
                        help = 'flux calibration')
   
    args = parser.parse_args(options)

    return args


def main(args=None):
    
    if args is None:
        args = parse()
    elif isinstance(args, (list, tuple)):
        args = parse(args)

    t0   = time.time()
    log  = get_logger()

    # guess if it is a preprocessed or a raw image
    hdulist   = fits.open(args.image)
    is_input_preprocessed = ("IMAGE" in hdulist)&("IVAR" in hdulist)
    primary_header  = hdulist[0].header
    hdulist.close()

    if is_input_preprocessed :
        image   = read_image(args.image)
    else :
        if args.camera is None :
            print("ERROR: Need to specify camera to open a raw fits image (with all cameras in different fits HDUs)")
            print("Try adding the option '--camera xx', with xx in {brz}{0-9}, like r7,  or type 'desi_qproc --help' for more options")
            sys.exit(12)
        image = read_raw(args.image, args.camera)
        if args.output_preproc is not None :
            write_image(args.output_preproc, image)

    cfinder = None

    if args.psf is None :
        if cfinder is None :
            cfinder = CalibFinder([image.meta,primary_header])
        args.psf = cfinder.findfile("PSF")
        log.info(" Using PSF {}".format(args.psf))


    if args.shifted_psf :
        if ( not is_input_preprocessed ) and ( args.output_preproc is None ) :
            print("ERROR: The PSF shifting requires as an input a preprocessed image on disk. So, please rerun with the option --output-preproc xxx.fits")
            print("Type 'desi_qproc --help' for more options")
            sys.exit(12)
        if is_input_preprocessed :
            image_filename = args.image
        else :
            image_filename = args.output_preproc

        cmd = "desi_compute_trace_shifts --image {} --psf {} --outpsf {} --auto".format(image_filename,args.psf,args.shifted_psf)
        if runcmd(cmd) !=0:
            log.error('desi_compute_trace_shifts failed')
            sys.exit(12)
        # now use the shifted psf for extractions
        args.psf = args.shifted_psf

    tset    = read_xytraceset(args.psf)



    # add fibermap
    if args.fibermap :
        fibermap = read_fibermap(args.fibermap)
    else :
        fibermap = None

    qframe  = qproc_boxcar_extraction(tset,image,width=args.width, fibermap=fibermap)

    if args.output_rawframe is not None :
        write_qframe(args.output_rawframe,qframe)
        log.info("wrote raw extracted frame in {}".format(args.output_rawframe))

    if args.compute_fiberflat is not None :
        fiberflat = qproc_compute_fiberflat(qframe)
        #write_qframe(args.compute_fiberflat,qflat)
        write_fiberflat(args.compute_fiberflat,fiberflat,header=qframe.meta)
        log.info("wrote fiberflat in {}".format(args.compute_fiberflat))

    if args.apply_fiberflat or args.input_fiberflat :

        if args.input_fiberflat is None :
            if cfinder is None :
                cfinder = CalibFinder([image.meta,primary_header])
            args.input_fiberflat = cfinder.findfile("FIBERFLAT")

        log.info("applying fiber flat {}".format(args.input_fiberflat))
        flat = read_fiberflat(args.input_fiberflat)
        qproc_apply_fiberflat(qframe,flat)

    if args.skysub :
        log.info("sky subtraction")
        if args.output_skyframe is not None :
            skyflux=qproc_sky_subtraction(qframe,return_skymodel=True)
            sqframe=QFrame(qframe.wave,skyflux,np.ones(skyflux.shape))
            write_qframe(args.output_skyframe,sqframe)
            log.info("wrote sky model in {}".format(args.output_skyframe))
        else :
            qproc_sky_subtraction(qframe)

    if args.fluxcalib :
        if cfinder is None :
            cfinder = CalibFinder([image.meta,primary_header])
        fluxcalib_filename = cfinder.findfile("FLUXCALIB")
        fluxcalib = read_average_flux_calibration(fluxcalib_filename)
        log.info("read average calib in {}".format(fluxcalib_filename))
        seeing  = qframe.meta["SEEING"]
        airmass = qframe.meta["AIRMASS"]
        exptime = qframe.meta["EXPTIME"]
        exposure_calib = fluxcalib.value(seeing=seeing,airmass=airmass)
        for q in range(qframe.nspec) :
            fiber_calib=np.interp(qframe.wave[q],fluxcalib.wave,exposure_calib)*exptime
            inv_calib = (fiber_calib>0)/(fiber_calib + (fiber_calib==0))
            qframe.flux[q] *= inv_calib
            qframe.ivar[q] *= fiber_calib**2*(fiber_calib>0)

    fibers  = parse_fibers(args.fibers)
    if fibers is None : 
        fibers = qframe.flux.shape[0]
    else :
        ii = np.arange(qframe.fibers.size)[np.in1d(qframe.fibers,fibers)]
        if ii.size == 0 :
            log.error("no such fibers in frame,")
            log.error("fibers are in range [{}:{}]".format(qframe.fibers[0],qframe.fibers[-1]+1))
            sys.exit(12)
        qframe = qframe[ii]

    if args.outframe is not None :
        write_qframe(args.outframe,qframe)
        log.info("wrote {}".format(args.outframe))

    t1 = time.time()
    log.info("all done in {:3.1f} sec".format(t1-t0))

    if args.plot :
        log.info("plotting {} spectra".format(qframe.wave.shape[0]))

        import matplotlib.pyplot as plt
        fig = plt.figure()
        for i in range(qframe.wave.shape[0]) :
            j=(qframe.ivar[i]>0)
            plt.plot(qframe.wave[i,j],qframe.flux[i,j])
        plt.grid()
        plt.xlabel("wavelength")
        plt.ylabel("flux")
        plt.show()
