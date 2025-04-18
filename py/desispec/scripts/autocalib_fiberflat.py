"""
desispec.scripts.autocalib_fiberflat
====================================

"""
from __future__ import absolute_import, division
import time

import numpy as np
from desiutil.log import get_logger
from desispec.io import read_fiberflat,write_fiberflat
from desispec.fiberflat import autocalib_fiberflat, average_fiberflat, gradient_correction
from desispec.io import findfile

import argparse


def parse(options=None):
    parser = argparse.ArgumentParser(description="Merge fiber flats from different calibration lamps")
    parser.add_argument('-i','--infile', type = str, default = None, required=True, nargs="*")
    parser.add_argument('--prefix', type = str, required=False, default=None, help = "output filename prefix, including directory (one file per spectrograph), default is findfile('fiberflatnight',night,...,cam)")
    parser.add_argument('--night', type = str, required=False, default=None)
    parser.add_argument('--arm', type = str, required=False, default=None, help="b, r or z")
    parser.add_argument('--average-per-program', action="store_true",help="first average per spectro and program name")
    parser.add_argument('--solve-gradient', action="store_true", help='apply gradient correction')
    parser.add_argument('--gradient-ref-night', type=str, required=False, default=None, help='reference night for gradient correction')

    args = parser.parse_args(options)

    return args

def main(args=None) :

    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    log=get_logger()
    if ( args.night is None or args.arm is None ) and args.prefix is None :
        log.error("ERROR in arguments, need night and arm or prefix for output file names")
        return 1

    log=get_logger()
    log.info("starting at {}".format(time.asctime()))
    inputs=[]
    for filename in args.infile :
        inputs.append(read_fiberflat(filename))


    program=[]
    camera=[]
    expid=[]
    for fflat in inputs :
        program.append(fflat.header["PROGRAM"])
        camera.append(fflat.header["CAMERA"])
        expid.append(fflat.header["EXPID"])
    program=np.array(program)
    camera=np.array(camera)
    expid=np.array(expid)

    ucam = np.unique(camera)
    log.debug("cameras: {}".format(ucam))

    if args.average_per_program :

        uprog = np.unique(program)
        log.info("programs: {}".format(uprog))

        fiberflat_per_program_and_camera = []
        for p in uprog :

            if p.find("CALIB DESI-CALIB-00 to 03")>=0 :
                log.warning("ignore program {}".format(p))
                continue

            log.debug("make sure we have the same list of exposures per camera, for each program")
            common_expid=None
            for c in ucam :
                expid_per_program_and_camera =  expid[(program==p)&(camera==c)]
                log.info("expids with camera={} for program={} : {}".format(c,p,expid_per_program_and_camera))
                if common_expid is None :
                    common_expid = expid_per_program_and_camera
                else :
                    common_expid = np.intersect1d(common_expid,expid_per_program_and_camera)

            log.info("expids with all cameras for program={} : {}".format(p,common_expid))

            for c in ucam :
                fflat_to_average = []
                for e in common_expid :
                    ii = np.where((program==p)&(camera==c)&(expid==e))[0]
                    for i in ii : fflat_to_average.append(inputs[i])
                log.info("averaging {} {} ({} files)".format(p,c,len(fflat_to_average)))
                fiberflat_per_program_and_camera.append(average_fiberflat(fflat_to_average))
        inputs=fiberflat_per_program_and_camera

    else :

        log.debug("make sure we have the same list of exposures per camera, for each program")
        common_expid=None
        for c in ucam :
            expid_per_camera =  expid[(camera==c)]
            log.info("expids with camera={} : {}".format(c,expid_per_camera))
            if common_expid is None :
                common_expid = expid_per_camera
            else :
                common_expid = np.intersect1d(common_expid,expid_per_camera)

        log.info("expids with all cameras : {}".format(common_expid))
        fflat_to_average = []
        for e in common_expid :
            ii = np.where((expid==e))[0]
            for i in ii : fflat_to_average.append(inputs[i])
        inputs = fflat_to_average

    fiberflats = autocalib_fiberflat(inputs)

    if args.solve_gradient or args.gradient_ref_night is not None:
        ref_fiberflats = {}
        if args.gradient_ref_night is None:
            log.info('Solving fiberflat gradient using default fiberflats from $DESI_SPECTRO_CALIB')
            #- find default fiberflats to use as reference
            from desispec.calibfinder import CalibFinder
            for spectro, fiberflat in fiberflats.items():
                cf = CalibFinder([fiberflat.header,])
                filename = cf.findfile('FIBERFLAT')
                log.debug('Reference fiberflat %s %s', spectro, filename)
                ref_fiberflats[spectro] = read_fiberflat(filename)
        else:
            #- use fiberflats from given night as reference
            log.info(f'Solving fiberflat gradient using reference night {args.gradient_ref_night}')
            for spectro in fiberflats.keys():
                ref_filename = findfile('fiberflatnight', night=args.gradient_ref_night, camera="{}{}".format(args.arm, spectro))
                ref_fiberflats[spectro] = read_fiberflat(ref_filename)
        fiberflats = gradient_correction(fiberflats, ref_fiberflats)

    for spectro in fiberflats.keys() :
        if args.prefix :
            ofilename="{}{}-autocal.fits".format(args.prefix,spectro)
        else :
            camera="{}{}".format(args.arm,spectro)
            ofilename=findfile('fiberflatnight', args.night, 0 , camera)
        write_fiberflat(ofilename,fiberflats[spectro])
        log.info("successfully wrote %s"%ofilename)

