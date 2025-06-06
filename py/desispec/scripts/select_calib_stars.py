"""
desispec.scripts.select_calib_stars
===================================

"""
import os,sys
import numpy as np
import argparse
import fitsio

from astropy.table import Table

from desiutil.log import get_logger
from desispec.io import read_stdstar_models,read_frame,read_fiberflat,read_sky
from desispec.io.util import get_tempfilename
from desispec.fiberflat import apply_fiberflat
from desispec.sky import subtract_sky
from desispec.fiberfluxcorr import flat_to_psf_flux_correction

def parse(options=None):

    parser = argparse.ArgumentParser(
        prog='desi_select_calib_stars',
        description="Select calibration stars from spectro/photo flux across all petals"
        )

    parser.add_argument('--models', type=str, required=True, nargs="*", help = 'Input list of std stars model files')
    parser.add_argument('--frames',type=str, required=True, nargs="*", help = 'Input list of r-camera frame files from a given exposure (one per spectrograph)')
    parser.add_argument('--fiberflats', type = str, required=True, nargs="*", help = 'path of DESI r-camera fiberflat fits files (one per spectrograph)')
    parser.add_argument('--skys', type = str, required=True, nargs="*", help = 'path of DESI r-camera sky fits files (one per spectrograph)')
    parser.add_argument('-o','--outfile', type=str, default=None, required=True, help = 'Output table with list of calibration stars')
    parser.add_argument('--delta-color-cut', type = float, default = 0.1, required=False, help = 'discard model stars with different broad-band color from imaging')
    parser.add_argument('--wavemin', type = float, default = 6000., required=False, help = 'min wavelength in Angstrom, used to measure rapidely the broadband flux')
    parser.add_argument('--wavemax', type = float, default = 7300., required=False, help = 'max wavelength in Angstrom, used to measure rapidely the broadband flux')

    args = parser.parse_args(options)

    return args

def main(args=None):

    log=get_logger()

    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    log.info("reading inputs")
    frames=dict()
    fiberflats=dict()
    skys=dict()
    stars=dict()

    for filename in args.frames :
        frame = read_frame(filename)
        if frame.spectrograph is not None :
            spectro = frame.spectrograph
        else :
            spectro = frame.meta["SPECGRPH"]
        frames[spectro] = frame

    for filename in args.fiberflats :
        fiberflat  = read_fiberflat(filename)
        spectro = fiberflat.header["SPECGRPH"]
        fiberflats[spectro] = fiberflat

    for filename in args.skys :
        sky  = read_sky(filename)
        spectro = sky.header["SPECGRPH"]
        skys[spectro] = sky

    for filename in args.models :
        flux, wave, fibers, metadata = read_stdstar_models(filename)
        head    = fitsio.read_header(filename,"FIBERMAP")
        fmap    = fitsio.read(filename,"FIBERMAP")
        spectro = head["SPECGRPH"]

        ii=(wave>=args.wavemin)&(wave<=args.wavemax)
        table=Table(metadata)
        table["FIBER"] = fibers
        table["MODELRFLUX"] = np.sum(flux[:,ii],axis=1)
        f2i={f:i for i,f in enumerate(fmap["FIBER"])}
        ii=[f2i[f] for f in fibers]
        table["X"] = fmap["FIBERASSIGN_X"][ii]
        table["Y"] = fmap["FIBERASSIGN_Y"][ii]
        #print(table["FIBER"])
        #print(fmap["FIBER"][ii])
        #sys.exit(12)
        stars[spectro] = table


    log.debug("check spectrographs")
    valid_spectrographs = []
    for spectro in frames.keys() :
        if spectro not in fiberflats.keys() :
            log.warning("missing fiberflat for spectro {}".format(spectro))
            continue
        if spectro not in skys.keys() :
            log.warning("missing sky for spectro {}".format(spectro))
            continue
        if spectro not in stars.keys() :
            log.warning("missing stars for spectro {}".format(spectro))
            continue
        valid_spectrographs.append(spectro)

    log.info("valid spectrographs = {}".format(valid_spectrographs))

    log.info("processing")
    calib_stars = {}
    for k in ["FIBER","RCALIBFRAC","EBV","MODEL_COLOR","DATA_COLOR","X","Y"] :
        calib_stars[k] = []

    for spectro in valid_spectrographs :
        fibers  = stars[spectro]["FIBER"]
        frame   = frames[spectro]
        apply_fiberflat(frame, fiberflats[spectro])
        subtract_sky(frame, skys[spectro])

        f2i = {f:i for i,f in enumerate(frame.fibermap["FIBER"])}
        indices = np.array([f2i[f] for f in fibers])
        log.debug("fiber indices = {}".format(indices))

        jj = np.where((frame.wave>=args.wavemin)&(frame.wave<=args.wavemax))[0]

        if jj.size==0 :
            message="wavelength mismatch: frame.wave=[{},{}] and analysis range = [{},{}]".format(frame.wave[0],frame.wave[-1],args.wavemin,args.wavemax)
            log.error(message)
            raise RuntimeError(message)
        
        # apply point source correction to flux
        psf_correction = flat_to_psf_flux_correction(frame.fibermap,exposure_seeing_fwhm=1.1)
        frame.flux *= psf_correction[:,None]

        rivar = np.sum(frame.ivar[indices][:,jj]*(frame.mask[indices][:,jj]==0),axis=1)
        rflux = np.sum(frame.ivar[indices][:,jj]*frame.flux[indices][:,jj]*(frame.mask[indices][:,jj]==0),axis=1)
        rflux[rivar>0] /= rivar[rivar>0]
        ratio = rflux/stars[spectro]["MODELRFLUX"]
        calib_stars["FIBER"].append(fibers)
        calib_stars["RCALIBFRAC"].append(ratio)
        ebv = frame.fibermap[indices]["EBV"]
        calib_stars["EBV"].append(frame.fibermap[indices]["EBV"])
        calib_stars["X"].append(stars[spectro]["X"])
        calib_stars["Y"].append(stars[spectro]["Y"])
        if "MODEL_G-R" in stars[spectro].dtype.names :
            calib_stars["MODEL_COLOR"].append(stars[spectro]["MODEL_G-R"])
            calib_stars["DATA_COLOR"].append(stars[spectro]["DATA_G-R"])
        elif 'MODEL_GAIA-BP-RP' in stars[spectro].dtype.names :
            calib_stars["MODEL_COLOR"].append(stars[spectro]["MODEL_GAIA-BP-RP"])
            calib_stars["DATA_COLOR"].append(stars[spectro]["DATA_GAIA-BP-RP"])
        else :
            message="Can't find either G-R or BP-RP color in the model file."
            log.error(message)
            raise RuntimeError(message)

    table=Table()
    for k in calib_stars.keys() :
        table[k] = np.hstack(calib_stars[k])
    calib_stars=table

    ok=(calib_stars["RCALIBFRAC"]>0)
    if np.sum(ok)==0 :
        message = "no valid star"
        log.error(message)
        raise RuntimeError(message)

    calib_stars = calib_stars[ok]

    medval = np.median(calib_stars["RCALIBFRAC"])
    if not medval>0 :
        message = "median ratio (meas/model) = {} is not valid".format(medval)
        log.error(message)
        raise RuntimeError(message)
    calib_stars["RCALIBFRAC"] /= medval
    nsig = 3.
    rms = 1.48*np.median(np.abs(calib_stars["RCALIBFRAC"]-1))
    log.info("rms of star r-band calib = {:.3f}".format(rms))
    if rms<0.04 :
        rms = 0.04 # minimal value (a bit arbitrary but we have to set a floor)
    log.info("setting calib variation rejection threshold at {:.3f}".format(3*rms))
    good = np.abs(calib_stars["RCALIBFRAC"]-1)<3*rms

    if args.delta_color_cut > 0 :
        # check dust extinction values for those stars
        reddening_relative_error = 0.2 * calib_stars["EBV"]
        log.info("Consider a reddening sys. error in the range {:4.3f} {:4.3f}".format(np.min(reddening_relative_error),np.max(reddening_relative_error)))
        good &= (np.abs(calib_stars["MODEL_COLOR"]-calib_stars["DATA_COLOR"])<args.delta_color_cut+reddening_relative_error)

    bad  = ~good
    if np.sum(bad) :
        log.info("Discarding {} stars with r-band calib delta = {}".format(np.sum(bad),list(calib_stars["RCALIBFRAC"][bad])))

    calib_stars["VALID"]=good.astype(int)
    tmpfile = get_tempfilename(args.outfile)
    calib_stars.write(tmpfile, overwrite=True)
    os.rename(tmpfile, args.outfile)
    log.info("wrote {}".format(args.outfile))

    return 0
