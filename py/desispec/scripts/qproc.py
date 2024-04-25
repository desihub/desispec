"""
desispec.scripts.qproc
======================

Run DESI qproc on a given exposure
"""


import argparse
import sys,os
import time
import numpy as np
import astropy.io.fits as fits
from desiutil.log import get_logger
from desispec.util import option_list
from desispec.io import read_raw,read_image,read_fibermap,write_image,write_fiberflat,read_fiberflat
from desispec.io.fluxcalibration import read_average_flux_calibration
from desispec.io.xytraceset import read_xytraceset,write_xytraceset
import desispec.scripts.trace_shifts as trace_shifts_script
from desispec.trace_shifts import write_traces_in_psf
from desispec.calibfinder import CalibFinder
from desispec.qproc.qframe import QFrame
from desispec.qproc.io import read_qframe,write_qframe
from desispec.qproc.qextract import qproc_boxcar_extraction
from desispec.qproc.qfiberflat import qproc_apply_fiberflat,qproc_compute_fiberflat
from desispec.qproc.qsky import qproc_sky_subtraction
from desispec.util import parse_fibers
from desispec.qproc.qarc import process_arc
from desispec.qproc.flavor import check_qframe_flavor

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
                        help = 'path to input psf fits file to get the trace coordinates (default is psf in $DESI_CCD_CALIBRATION_DATA)')
    parser.add_argument('--output-preproc', type = str, default = None, required = False,
                        help = 'save the preprocessed image in this file.')
    parser.add_argument('--output-rawframe', type = str, default = None, required = False,
                        help = 'save the raw (before flatfield,sky sub,calibration) extracted qframe to this file.')
    parser.add_argument('--output-skyframe', type = str, default = None, required = False,
                        help = 'save the sky model in a qframe file (need --skysub)')
    parser.add_argument('--output-psf', type = str, default = None, required = False,
                        help = 'save psf (after shifts of gaussian sigma adjustment) to this file.')
    parser.add_argument('--fibers', type=str, default = None, required = False,
                        help = 'defines from_to which fiber to work on. (ex: --fibers=50:60,4 means that only fibers 4, and fibers from 50 to 60 (excluded) will be extracted)')
    parser.add_argument('--width', type=int, default=7, required=False,
                        help = 'extraction line width (in pixels)')
    parser.add_argument('--plot', action='store_true',
                        help = 'plot result')
    parser.add_argument('--compute-lsf-sigma', action="store_true",
                        help = 'estimate lsf sigma (for arc lamp exposures)')
    parser.add_argument('--shift-psf', action="store_true",
                        help = 'estimate spectral trace shifts')
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
    parser.add_argument('--auto-output-dir', type = str, default = '.', required = False,
                        help = 'Output directory when running the script in auto mode')
    parser.add_argument('--auto', action = 'store_true', help = 'auto-decide the list of processes to run based on the input. Output files are saved in an output directory which is by default the current working directory but can be modified with the option --auto-output-dir')
    parser.add_argument('--fallback-on-dark-not-found', action="store_true", help="fall back to DESI_SPECTRO_CALIB darks if dark files are missing from DESI_SPECTRO_DARK, else fail")

    args = parser.parse_args(options)

    return args


def main(args=None):

    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    t0   = time.time()
    log  = get_logger()

    # guess if it is a preprocessed or a raw image
    hdulist   = fits.open(args.image)
    is_input_preprocessed = ("IMAGE" in hdulist)&("IVAR" in hdulist)
    primary_header  = hdulist[0].header
    input_has_fibermap = ("FIBERMAP" in hdulist)
    hdulist.close()


    if is_input_preprocessed :
        image   = read_image(args.image)
    else :
        if args.camera is None :
            print("ERROR: Need to specify camera to open a raw fits image (with all cameras in different fits HDUs)")
            print("Try adding the option '--camera xx', with xx in {brz}{0-9}, like r7,  or type 'desi_qproc --help' for more options")
            sys.exit(12)
        image = read_raw(args.image, args.camera, args.fibermap, fill_header=[1,],
                         fallback_on_dark_not_found=args.fallback_on_dark_not_found,
                         no_cte_corr=True,
                         )

    if args.auto :
        log.debug("AUTOMATIC MODE")
        try :
            night = image.meta['NIGHT']
            if not 'EXPID' in image.meta :
                if 'EXPNUM' in image.meta :
                    log.warning('using EXPNUM {} for EXPID'.format(image.meta['EXPNUM']))
                    image.meta['EXPID'] = image.meta['EXPNUM']
            expid = image.meta['EXPID']
        except KeyError as e :
            log.error("Need at least NIGHT and EXPID (or EXPNUM) to run in auto mode. Retry without the --auto option.")
            log.error(str(e))
            sys.exit(12)

        indir = os.path.dirname(args.image)
        if args.fibermap is None :
            #- first check if input image has a fibermap
            if input_has_fibermap:
                log.debug('auto-mode: using FIBERMAP in input image file')
                args.fibermap = args.image
            else:
                filename = '{}/fibermap-{:08d}.fits'.format(indir, expid)
                if os.path.isfile(filename) :
                    log.debug("auto-mode: found a fibermap, {}, using it!".format(filename))
                    args.fibermap = filename

        if args.output_preproc is None :
            if not is_input_preprocessed :
                args.output_preproc = '{}/preproc-{}-{:08d}.fits'.format(args.auto_output_dir, args.camera.lower(), expid)
                log.debug("auto-mode: will write preproc in "+args.output_preproc)
            else :
                log.debug("auto-mode: will not write preproc because input is a preprocessed image")

        if args.auto_output_dir != '.' :
            if not os.path.isdir(args.auto_output_dir) :
                log.debug("auto-mode: creating directory "+args.auto_output_dir)
                os.makedirs(args.auto_output_dir)

    if args.output_preproc is not None :
        write_image(args.output_preproc, image)

    cfinder = None

    if args.psf is None :
        if cfinder is None :
            cfinder = CalibFinder([image.meta,primary_header],fallback_on_dark_not_found=args.fallback_on_dark_not_found)
        args.psf = cfinder.findfile("PSF")
        log.info(" Using PSF {}".format(args.psf))

    tset    = read_xytraceset(args.psf)

    # add fibermap
    if args.fibermap :
        if os.path.isfile(args.fibermap) :
            fibermap = read_fibermap(args.fibermap)
        elif hasattr(image, 'fibermap') and image.fibermap is not None:
            log.info('Using fibermap from preproc')
            fibermap = image.fibermap
        else :
            log.error("no fibermap file {}".format(args.fibermap))
            fibermap = None
    else :
        fibermap = None

    # select fibermap for one spectrograph
    if fibermap is not None:
        try:
            if args.camera is not None:
                camera = args.camera
            else:
                camera = image.meta['CAMERA']
            petal = int(camera[-1])
            fibermap = fibermap[fibermap['PETAL_LOC'] == petal]

        except Exception as e:
            log.error(e)


    if "OBSTYPE" in image.meta :
        obstype = image.meta["OBSTYPE"].upper()
        image.meta["OBSTYPE"]=obstype # make sure it's upper case
        qframe = None
    else :
        log.warning("No OBSTYPE keyword, trying to guess ...")
        qframe  = qproc_boxcar_extraction(tset,image,width=args.width, fibermap=fibermap)
        input_flavor = None
        if "FLAVOR" in image.meta : input_flavor = image.meta["FLAVOR"]
        obstype = check_qframe_flavor(qframe,input_flavor=input_flavor).upper()
        image.meta["OBSTYPE"]=obstype

    log.info("OBSTYPE = '{}'".format(obstype))

    if args.auto  :

        # now set the things to do
        if obstype == "SKY" or obstype == "TWILIGHT" or obstype == "SCIENCE" :

            args.shift_psf       = True
            args.output_psf      = '{}/psf-{}-{:08d}.fits'.format(args.auto_output_dir, args.camera, expid)
            args.output_rawframe = '{}/qframe-{}-{:08d}.fits'.format(args.auto_output_dir, args.camera, expid)
            args.apply_fiberflat = True
            args.skysub          = True
            args.output_skyframe = '{}/qsky-{}-{:08d}.fits'.format(args.auto_output_dir, args.camera, expid)
            args.fluxcalib       = True
            args.outframe        = '{}/qcframe-{}-{:08d}.fits'.format(args.auto_output_dir, args.camera, expid)

        elif obstype == "ARC" or obstype == "TESTARC" :

            args.shift_psf       = True
            args.output_psf      = '{}/psf-{}-{:08d}.fits'.format(args.auto_output_dir, args.camera, expid)
            args.output_rawframe = '{}/qframe-{}-{:08d}.fits'.format(args.auto_output_dir, args.camera, expid)
            args.compute_lsf_sigma = True

        elif obstype == "FLAT" or obstype == "TESTFLAT" or obstype == "OTHER" :
            args.shift_psf       = True
            args.output_psf      = '{}/psf-{}-{:08d}.fits'.format(args.auto_output_dir, args.camera, expid)
            args.output_rawframe = '{}/qframe-{}-{:08d}.fits'.format(args.auto_output_dir, args.camera, expid)
            args.compute_fiberflat = '{}/qfiberflat-{}-{:08d}.fits'.format(args.auto_output_dir, args.camera, expid)

    if args.shift_psf :

        # using the trace shift script
        if args.auto  :
            options = option_list({"psf":args.psf,"image":"dummy","outpsf":"dummy","degyy":0,"continuum":((obstype=="FLAT")|(obstype=="TESTFLAT")),"sky":((obstype=="SCIENCE")|(obstype=="SKY"))})
        else :
            options = option_list({"psf":args.psf,"image":"dummy","outpsf":"dummy","degyy":0})
        tmp_args = trace_shifts_script.parse(options=options)
        tset = trace_shifts_script.fit_trace_shifts(image=image,args=tmp_args)

    qframe  = qproc_boxcar_extraction(tset,image,width=args.width, fibermap=fibermap)

    if tset.meta is not None :
        # add traceshift info in the qframe, this will be saved in the qframe header
        if qframe.meta is None :
            qframe.meta = dict()
        for k in tset.meta.keys() :
            qframe.meta[k] = tset.meta[k]

    if args.output_rawframe is not None :
        write_qframe(args.output_rawframe,qframe)
        log.info("wrote raw extracted frame in {}".format(args.output_rawframe))


    if args.compute_lsf_sigma :
        tset = process_arc(qframe,tset,linelist=None,npoly=2,nbins=2)

    if args.output_psf is not None :
        for k in qframe.meta :
            if k not in tset.meta :
                tset.meta[k] = qframe.meta[k]
        write_xytraceset(args.output_psf,tset)

    if args.compute_fiberflat is not None :
        fiberflat = qproc_compute_fiberflat(qframe)
        #write_qframe(args.compute_fiberflat,qflat)
        write_fiberflat(args.compute_fiberflat,fiberflat,header=qframe.meta)
        log.info("wrote fiberflat in {}".format(args.compute_fiberflat))

    if args.apply_fiberflat or args.input_fiberflat :

        if args.input_fiberflat is None :
            if cfinder is None :
                cfinder = CalibFinder([image.meta,primary_header],fallback_on_dark_not_found=args.fallback_on_dark_not_found)
            try :
                args.input_fiberflat = cfinder.findfile("FIBERFLAT")
            except KeyError as e :
                log.error("no FIBERFLAT for this spectro config")
                sys.exit(12)
        log.info("applying fiber flat {}".format(args.input_fiberflat))
        flat = read_fiberflat(args.input_fiberflat)
        qproc_apply_fiberflat(qframe,flat)

    if args.skysub :
        log.info("sky subtraction")
        if args.output_skyframe is not None :
            skyflux=qproc_sky_subtraction(qframe,return_skymodel=True)
            sqframe=QFrame(qframe.wave,skyflux,np.ones(skyflux.shape),fibermap=fibermap)
            write_qframe(args.output_skyframe,sqframe)
            log.info("wrote sky model in {}".format(args.output_skyframe))
        else :
            qproc_sky_subtraction(qframe)

    if args.fluxcalib :
        if cfinder is None :
            cfinder = CalibFinder([image.meta,primary_header],fallback_on_dark_not_found=args.fallback_on_dark_not_found)
        # check for flux calib
        if cfinder.haskey("FLUXCALIB") :
            fluxcalib_filename = cfinder.findfile("FLUXCALIB")
            fluxcalib = read_average_flux_calibration(fluxcalib_filename)
            log.info("read average calib in {}".format(fluxcalib_filename))
            if "SEEING" in qframe.meta and qframe.meta["SEEING"] is not None:
                seeing  = qframe.meta["SEEING"]
            elif "ETCSEENG" in qframe.meta and qframe.meta["ETCSEENG"] is not None:
                seeing  = qframe.meta["ETCSEENG"]
            elif "PMSEEING" in qframe.meta and qframe.meta["PMSEEING"] is not None:
                seeing  = qframe.meta["PMSEEING"]
            else :
                log.warning("no SEEING information; using 1.1 arcsec FWHM")
                seeing = 1.1

            if "AIRMASS" in qframe.meta and qframe.meta["AIRMASS"] is not None:
                airmass = qframe.meta["AIRMASS"]
            else :
                log.warning("no AIRMASS information; using 1.0")
                airmass = 1.0

            exptime = qframe.meta["EXPTIME"]
            exposure_calib = fluxcalib.value(seeing=seeing,airmass=airmass)
            for q in range(qframe.nspec) :
                fiber_calib=np.interp(qframe.wave[q],fluxcalib.wave,exposure_calib)*exptime
                inv_calib = (fiber_calib>0)/(fiber_calib + (fiber_calib==0))
                qframe.flux[q] *= inv_calib
                qframe.ivar[q] *= fiber_calib**2*(fiber_calib>0)

            # add keyword in header giving the calibration factor applied at a reference wavelength
            band = qframe.meta["CAMERA"].upper()[0]
            if band == "B" :
                refwave=4500
            elif band == "R" :
                refwave=6500
            else :
                refwave=8500
            calvalue = np.interp(refwave,fluxcalib.wave,exposure_calib)*exptime
            qframe.meta["CALWAVE"]=refwave
            qframe.meta["CALVALUE"]=calvalue
        else :
            log.error("Cannot calibrate fluxes because no FLUXCALIB keywork in calibration files")

    if args.fibers is not None :
        fibers  = parse_fibers(args.fibers)
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
