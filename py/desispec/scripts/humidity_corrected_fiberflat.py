
from __future__ import absolute_import, division

import os
import fitsio
import argparse
import numpy as np

from desiutil.log import get_logger

from desispec.io import read_fiberflat,write_fiberflat,findfile,read_frame
from desispec.io.fiberflat_vs_humidity import get_humidity,read_fiberflat_vs_humidity
from desispec.calibfinder import CalibFinder
from desispec.fiberflat_vs_humidity import compute_humidity_corrected_fiberflat

def parse(options=None):
    parser = argparse.ArgumentParser(description="Compute a fiberflat corrected for variations with humidity.")

    parser.add_argument('-i','--infile', type = str, default = None, required=True,
                        help = 'path of DESI exposure frame fits file')
    parser.add_argument('--fiberflat', type = str, default = None, required=True,
                        help = 'path of DESI fiberflat fits file')
    parser.add_argument('--use-sky-fibers', action = 'store_true',
                        help = 'use sky fibers to improve the correction')
    parser.add_argument('-o','--outfile', type = str, default = None, required=True,
                        help = 'path of output fiberflar file')
    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args) :

    log = get_logger()

    # just read frame header in case we don't need to do anything
    frame_header = fitsio.read_header(args.infile,"FLUX")

    if args.use_sky_fibers :
        # need full frame to adjust correction on data
        frame = read_frame(args.infile)
    else :
        frame = None

    cfinder = CalibFinder([frame_header])
    if not cfinder.haskey("FIBERFLATVSHUMIDITY"):
        log.info("No information on fiberflat vs humidity for camera {}, simply link the input fiberflat".format(frame_header["CAMERA"]))
        if not os.path.islink(args.outfile) :
            relpath=os.path.relpath(args.fiberflat,os.path.dirname(args.outfile))
            os.symlink(relpath,args.outfile)
        return 0

    # read fiberflat
    calib_fiberflat = read_fiberflat(args.fiberflat)

    # read mean fiberflat vs humidity
    filename = cfinder.findfile("FIBERFLATVSHUMIDITY")
    log.info(f"reading {filename}")
    mean_fiberflat_vs_humidity , humidity_array, ffh_wave, ffh_header = read_fiberflat_vs_humidity(filename)
    assert(np.allclose(calib_fiberflat.wave,ffh_wave))

    # now need to find the humidity for this frame and for this fiberflat
    night=frame_header["NIGHT"]
    camera=frame_header["CAMERA"]
    current_frame_humidity =get_humidity(night=night,expid=frame_header["EXPID"],camera=camera)
    log.info("humidity during current exposure={:.2f}".format(current_frame_humidity))



    # we can compute the correction now that we have everything in hand
    improved_fiberflat = compute_humidity_corrected_fiberflat(calib_fiberflat, mean_fiberflat_vs_humidity , humidity_array, current_frame_humidity, frame = frame)

    # add telemetry humidity for the dome flats for the record
    # try to read the night exposure table to get the list of flats
    first_expid = calib_fiberflat.header["EXPID"]
    calib_humidity=[ get_humidity(night,first_expid,camera) ]
    fiberflat_expid=[ first_expid]
    for expid in range(first_expid+1,first_expid+40) :
        filename=findfile("raw",night,expid)
        if not os.path.isfile(filename): continue
        head=fitsio.read_header(filename,1)
        if not "OBSTYPE" in head.keys() or head["OBSTYPE"]!="FLAT" :
            break
        fiberflat_expid.append(expid)
        calib_humidity.append(get_humidity(night,expid,camera))
    log.debug("calib expids={}".format(fiberflat_expid))
    log.debug("calib humidities={}".format(calib_humidity))
    calib_humidity=np.mean(calib_humidity)
    if np.isnan(calib_humidity) :
        log.warning("missing humidity info for fiber flat, use link to input")
        calib_humidity=0.
    else :
        log.info("mean humidity during calibration exposures={:.2f}".format(calib_humidity))
        fit_humidity = improved_fiberflat.header["CALFHUM"]
        if np.abs(fit_humidity-calib_humidity)>10 :
            message="large difference between best fit humidity during dome flats ({:.1f}) and value from telemetry ({:.1f})".format(fit_humidity,calib_humidity)
            if np.abs(fit_humidity-calib_humidity)>20 :
                log.error(message)
                raise RuntimeError(message)
            log.warning(message)

    improved_fiberflat.header["CALTHUM"] = (calib_humidity,"dome flat humidity from telemetry")

    # write it
    write_fiberflat(args.outfile,improved_fiberflat)
    log.info("wrote humidity corrected flat {}".format(args.outfile))

    return 0
