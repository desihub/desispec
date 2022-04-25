"""
Coadd spectra
"""

from __future__ import absolute_import, division, print_function

import os,sys
import numpy as np
from astropy.table import Table
import fitsio

from desiutil.log import get_logger
from desispec.io import read_spectra,write_spectra,read_frame
from desispec.coaddition import coadd,coadd_cameras,resample_spectra_lin_or_log
from desispec.pixgroup import frames2spectra
from desispec.specscore import compute_coadd_scores

def parse(options=None):
    import argparse

    parser = argparse.ArgumentParser("Coadd all spectra per target, and optionally resample on linear or logarithmic wavelength grid")
    parser.add_argument("-i","--infile", type=str, nargs='+',
            help="input spectra file or input frame files")
    parser.add_argument("-o","--outfile", type=str,
            help="output spectra file")
    parser.add_argument("--nsig", type=float, default=4,
            help="nsigma rejection threshold for cosmic rays (default %(default)s)")
    parser.add_argument("--lin-step", type=float, default=None,
            help="resampling to single linear wave array of given step in A")
    parser.add_argument("--log10-step", type=float, default=None,
            help="resampling to single log10 wave array of given step in units of log10")
    parser.add_argument("--wave-min", type=float, default=None,
            help="specify the min wavelength in A (default is the min wavelength in the input spectra), used only with option --lin-step or --log10-step")
    parser.add_argument("--wave-max", type=float, default=None,
            help="specify the max wavelength in A (default is the max wavelength in the input spectra, approximate), used only with option --lin-step or --log10-step)")
    parser.add_argument("--fast", action="store_true",
            help="fast resampling, at the cost of correlated pixels and no resolution matrix (used only with option --lin-step or --log10-step)")
    parser.add_argument("--nproc", type=int, default=1,
            help="number of multiprocessing cores to use")
    parser.add_argument("--coadd-cameras", action="store_true",
            help="coadd spectra of different cameras. works only if wavelength grids are aligned")
    parser.add_argument("--onetile", action="store_true",
            help="input spectra are from a single tile")


    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args

def main(args=None):

    log = get_logger()

    if args is None:
        args = parse()

    if args.lin_step is not None and args.log10_step is not None :
        log.critical("cannot have both linear and logarthmic bins :-), choose either --lin-step or --log10-step")
        return 12
    if args.coadd_cameras and ( args.lin_step is not None or args.log10_step is not None ) :
        log.critical("cannot specify a new wavelength binning along with --coadd-cameras option")
        return 12

    if len(args.infile) == 0:
        log.critical("You must specify input files")
        return 12

    log.info("reading input ...")

    # inspect headers
    input_is_frames = False
    input_is_spectra = False
    for filename in args.infile :
        ifile = fitsio.FITS(filename)
        head=ifile[0].read_header()
        identified = False
        if "EXTNAME" in head and head["EXTNAME"]=="FLUX" :
            print(filename,"is a frame")
            input_is_frames = True
            identified = True
            ifile.close()
            continue
        for hdu in ifile :
            head=hdu.read_header()
            if "EXTNAME" in head and head["EXTNAME"].find("_FLUX")>=0 :
                print(filename,"is a spectra")
                input_is_spectra = True
                identified = True
                break
        ifile.close()
        if not identified :
            log.error("{} not identified as frame of spectra file".format(filename))
            sys.exit(1)


    if input_is_frames and input_is_spectra :
        log.error("cannot combine input spectra and frames")
        sys.exit(1)


    if input_is_spectra :
        spectra = read_spectra(args.infile[0])
        for filename in args.infile[1:] :
            log.info("append {}".format(filename))
            spectra.update(read_spectra(filename))
    else: # frames
        frames = dict()
        cameras = {}
        for filename in args.infile:
            frame = read_frame(filename)
            night = frame.meta['NIGHT']
            expid = frame.meta['EXPID']
            camera = frame.meta['CAMERA']
            tileid = frame.meta['TILEID']
            # Add NIGHT and TILEID to fibermap so we can compute COADD_NUMNIGHT
            # and COADD_NUMTILE.
            frame.fibermap['NIGHT'] = night
            frame.fibermap['TILEID'] = tileid
            frames[(night,expid,camera)] = frame
            if args.coadd_cameras:
                cam,spec = camera[0],camera[1]
                # Keep a list of cameras (b,r,z) for each exposure + spec
                if (night,expid) not in cameras.keys():
                    cameras[(night,expid)] = {spec:[cam]}
                elif spec not in cameras[(night,expid)].keys():
                    cameras[(night,expid)][spec] = [cam]
                else:
                    cameras[(night,expid)][spec].append(cam)

        if args.coadd_cameras:
            # If not all 3 cameras are available, remove the incomplete sets
            for (night,expid), camdict in cameras.items():
                for spec,camlist in camdict.items():
                    log.info("Found {} for SP{} on NIGHT {} EXP {}".format(camlist,spec,night,expid))
                    if len(camlist) != 3 or np.any(np.sort(camlist) != np.array(['b','r','z'])):
                        for cam in camlist:
                            frames.pop((night,expid,cam+spec))
                            log.warning("Removing {}{} from Night {} EXP {}".format(cam,spec,night,expid))
                            
        spectra = frames2spectra(frames)

        #- hacks to make SpectraLite like a Spectra
        spectra.fibermap = Table(spectra.fibermap)

        del frames  #- maybe free some memory

    if args.coadd_cameras :
        log.info("coadding cameras ...")
        spectra = coadd_cameras(spectra, cosmics_nsig=args.nsig,
                onetile=args.onetile)
    else :
        log.info("coadding ...")
        coadd(spectra, cosmics_nsig=args.nsig, onetile=args.onetile)

    if args.lin_step is not None :
        log.info("resampling ...")
        spectra = resample_spectra_lin_or_log(spectra, linear_step=args.lin_step, wave_min =args.wave_min, wave_max =args.wave_max, fast = args.fast, nproc = args.nproc)
    if args.log10_step is not None :
        log.info("resampling ...")
        spectra = resample_spectra_lin_or_log(spectra, log10_step=args.log10_step, wave_min =args.wave_min, wave_max =args.wave_max, fast = args.fast, nproc = args.nproc)

    #- Add input files to header, removing previous INFIL* first
    if spectra.meta is None:
        spectra.meta = dict()

    for i in range(1000):
        key = 'INFIL{:03d}'.format(i)
        if key in spectra.meta:
            del spectra.meta[key]
        else:
            break

    for i, filename in enumerate(args.infile):
        spectra.meta['INFIL{:03d}'.format(i)] = os.path.basename(filename)

    log.info("writing {} ...".format(args.outfile))
    write_spectra(args.outfile,spectra)

    log.info("done")
