
"""
Read fibermap and cframe files for all exposures and update or create 
new files that group the spectra by healpix pixel.
"""

from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import sys

import numpy as np

import healpy as hp

from desiutil.log import get_logger

from .. import io as io

from ..parallel import dist_uniform

from .. import pipeline as pipe

from ..spectra import Spectra



def parse(options=None):
    parser = argparse.ArgumentParser(description="Update or create spectral group files.")

    parser.add_argument("--nights", required=False, default=None, help="comma separated (YYYYMMDD) or regex pattern")

    parser.add_argument("--cache", required=False, default=False, action="store_true", help="cache frame data for re-use")

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args, comm=None):

    log = get_logger()

    rank = 0
    nproc = 1
    if comm is not None:
        rank = comm.rank
        nproc = comm.size

    # raw and production locations

    rawdir = os.path.abspath(io.rawdata_root())
    proddir = os.path.abspath(io.specprod_root())

    if rank == 0:
        log.info("Starting at {}".format(time.asctime()))
        log.info("  using raw dir {}".format(rawdir))
        log.info("  using spectro production dir {}".format(proddir))

    # get the full graph and prune out just the objects we need

    grph = None
    if rank == 0:
        grph = pipe.load_prod(nightstr=args.nights)
        sgrph = pipe.graph_slice(grph, types=["spectra"], deps=True)
        pipe.graph_db_check(sgrph)
    if comm is not None:
        grph = comm.bcast(grph, root=0)

    # Get the properties of all spectra and cframes

    allspec = []
    spec_pixel = {}
    spec_paths = {}
    spec_frames = {}

    allframe = []
    frame_paths = {}
    frame_nights = {}
    frame_expids = {}
    frame_bands = {}

    nside = None

    for name, nd in sorted(grph.items()):
        if nd["type"] == "spectra":
            allspec.append(name)
            night, objname = pipe.graph_night_split(name)
            stype, nsidestr, pixstr = pipe.graph_name_split(objname)
            if nside is None:
                nside = int(nsidestr)
            spec_pixel[name] = int(pixstr)
            spec_paths[name] = pipe.graph_path(name)
            spec_frames[name] = nd["in"]
            for cf in nd["in"]:
                night, objname = pipe.graph_night_split(cf)
                ctype, cband, cspec, cexpid = pipe.graph_name_split(objname)
                allframe.append(cf)
                frame_nights[cf] = night
                frame_expids[cf] = cexpid
                frame_bands[cf] = cband
                frame_paths[cf] = pipe.graph_path(cf)

    # Now sort the pixels based on their healpix value

    sortspec = sorted(spec_pixel, key=spec_pixel.get)

    nspec = len(sortspec)

    # Distribute the full set of pixels
    
    dist_pixel = dist_uniform(nspec, nproc)
    if rank == 0:
        log.info("Distributing {} spectral groups among {} processes".format(nspec, nproc))
        sys.stdout.flush()

    if comm is not None:
        comm.barrier()

    # These are our pixels

    if dist_pixel[rank][1] == 0:
        specs = []
    else:
        specs = sortspec[dist_pixel[rank][0]:dist_pixel[rank][0]+dist_pixel[rank][1]]
    nlocal = len(specs)

    # This is the cache of frame data

    framedata = {}

    # Go through our local pixels...

    for sp in specs:
        # Read or initialize this spectral group
        msg = "  ({:04d}) Begin spectral group {} at {}".format(rank, sp, time.asctime())
        log.info(msg)
        sys.stdout.flush()

        specdata = None
        if os.path.isfile(spec_paths[sp]):
            # file exists, read in the current data
            specdata = io.read_spectra(spec_paths[sp], single=True)
        else:
            meta = {}
            meta["NSIDE"] = nside
            meta["HPIX"] = spec_pixel[sp]
            specdata = Spectra(meta=meta, single=True)

        if args.cache:
            # Clean out any cached frame data we don't need
            existing = list(framedata.keys())
            for fr in existing:
                if fr not in spec_frames[sp]:
                    #log.info("frame {} not needed for spec {}, purging".format(fr, sp))
                    del framedata[fr]

            # Read frame data if we don't have it
            for fr in spec_frames[sp]:
                if fr not in framedata:
                    #log.info("frame {} not in cache for spec {}, reading".format(fr, sp))
                    if os.path.isfile(frame_paths[fr]):
                        framedata[fr] = io.read_frame_as_spectra(frame_paths[fr], 
                        frame_nights[fr], frame_expids[fr], frame_bands[fr])
                    else:
                        framedata[fr] = Spectra()

            msg = "    ({:04d}) {} frames resident in memory".format(rank, len(framedata))
            log.info(msg)
            sys.stdout.flush()

        # Update spectral data

        for fr in spec_frames[sp]:
            fdata = None
            if args.cache:
                fdata = framedata[fr]
            else:
                if os.path.isfile(frame_paths[fr]):
                    fdata = io.read_frame_as_spectra(frame_paths[fr], 
                        frame_nights[fr], frame_expids[fr], frame_bands[fr])
                else:
                    fdata = Spectra()
            # Get the targets that hit this pixel.
            targets = []
            fmap = fdata.fmap
            ra = np.array(fmap["RA_TARGET"], dtype=np.float64)
            dec = np.array(fmap["DEC_TARGET"], dtype=np.float64)
            bad = np.where(fmap["TARGETID"] < 0)[0]
            ra[bad] = 0.0
            dec[bad] = 0.0
            pix = hp.ang2pix(nside, ra, dec, nest=True, lonlat=True)
            pix[bad] = -1

            for fm in zip(fmap["TARGETID"], pix):
                if fm[1] == spec_pixel[sp]:
                    targets.append(fm[0])

            if len(targets) > 0:
                # update with data from this frame
                specdata.update(fdata.select(targets=targets))

            del fdata

        # Write out updated data

        io.write_spectra(spec_paths[sp], specdata)

        msg = "  ({:04d}) End spectral group {} at {}".format(rank, sp, time.asctime())
        log.info(msg)
        sys.stdout.flush()


    if comm is not None:
        comm.barrier()
    if rank == 0:
        log.info("Finishing at {}".format(time.asctime()))

    return

