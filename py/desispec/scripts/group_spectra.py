
"""
Read fibermap and cframe files for all exposures and update or create 
new files that group the spectra by healpix pixel.
"""

from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import sys
import re

import numpy as np
import healpy as hp
import fitsio

from desiutil.log import get_logger
import desimodel.footprint

from .. import io as io

from ..parallel import dist_uniform

from .. import pipeline as pipe

from ..spectra import Spectra



def parse(options=None):
    parser = argparse.ArgumentParser(description="Update or create "
        "spectral group files.")

    parser.add_argument("--nights", required=False, default=None,
        help="comma separated (YYYYMMDD) or regex pattern")

    parser.add_argument("--cache", required=False, default=False,
        action="store_true", help="cache frame data for re-use")

    parser.add_argument("--pipeline", required=False, default=False,
        action="store_true", help="use pipeline planning and DB files "
        "to obtain dependency information.")

    parser.add_argument("--hpxnside", required=False, type=int, default=64,
        help="In the case of not using the pipeline info, the HEALPix "
        "NSIDE value to use.")

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
        #- TODO: this could be parallelized, or converted into a pure
        #- geometric calculation without having to read fibermaps
        if not args.pipeline:
            # We have to rescan all cframe files on the fly...
            # Put these into a "fake" dependency graph so that we
            # can treat it the same as a real one later in the code.

            grph = {}
            proddir = os.path.abspath(io.specprod_root())
            expdir = os.path.join(proddir, "exposures")
            specdir = os.path.join(proddir, "spectra")

            allnights = []
            nightpat = re.compile(r"\d{8}")
            for root, dirs, files in os.walk(expdir, topdown=True):
                for d in dirs:
                    nightmat = nightpat.match(d)
                    if nightmat is not None:
                        allnights.append(d)
                break

            nights = pipe.select_nights(allnights, args.nights)

            for nt in nights:
                expids = io.get_exposures(nt)

                for ex in expids:
                    cfiles = io.get_files("cframe", nt, ex)

                    for cam, cf in cfiles.items():
                        cfname = pipe.graph_name(nt, "cframe-{}-{:08d}".format(cam, ex))
                        node = {}
                        node["type"] = "cframe"
                        node["in"] = []
                        node["out"] = []
                        grph[cfname] = node

                        hdr = fitsio.read_header(cf)
                        if hdr["FLAVOR"].strip() == "science":
                            fmdata = fitsio.read(cf, 'FIBERMAP',
                                    columns=('RA_TARGET', 'DEC_TARGET'))

                            ra = fmdata["RA_TARGET"]
                            dec = fmdata["DEC_TARGET"]
                            ok = (ra == ra)  #- strip NaN
                            ra = ra[ok]
                            dec = dec[ok]
                            pix = desimodel.footprint.radec2pix(
                                                args.hpxnside, ra, dec)
                            for p in np.unique(pix):
                                if p >= 0:
                                    sname = "spectra-{}-{}".format(args.hpxnside, p)
                                    if sname not in grph:
                                        node = {}
                                        node["type"] = "spectra"
                                        node["nside"] = args.hpxnside
                                        node["pixel"] = p
                                        node["in"] = []
                                        node["out"] = ['zbest-{}-{}'.format(args.hpxnside, p), ]
                                        grph[sname] = node
                                    grph[sname]["in"].append(cfname)
                                    grph[cfname]["out"].append(sname)

        else:
            grph = pipe.load_prod(nightstr=args.nights)
            sgrph = pipe.graph_slice(grph, types=["spectra"], deps=True)
            pipe.graph_db_check(sgrph)

            #- The pipeline dependency graph associates all frames from an
            #- exposure with the spectra, not just the overlapping frames,
            #- so trim down to just the ones that are needed before reading
            #- the entire frame files
            log.info("Trimming extraneous frames")
            for name, nd in sorted(grph.items()):
                if nd["type"] != "spectra":
                    continue

                keep = list()
                discard = set()
                for cf in nd['in']:
                    #- Check if another frame from same spectro,expid has
                    #- already been discarded
                    night, cfname = pipe.graph_night_split(cf)
                    spectrograph, expid = pipe.graph_name_split(cfname)[2:4]
                    if (spectrograph, expid) in discard:
                        log.debug('{} discarding {}'.format(name, cf))
                        continue

                    framefile = pipe.graph_path(cf)
                    fibermap = fitsio.read(framefile, 'FIBERMAP',
                                columns=('RA_TARGET', 'DEC_TARGET'))
                    #- Strip NaN
                    ii = fibermap['RA_TARGET'] == fibermap['RA_TARGET']
                    fibermap = fibermap[ii]
                    ra, dec = fibermap['RA_TARGET'], fibermap['DEC_TARGET']

                    pix = desimodel.footprint.radec2pix(args.hpxnside, ra, dec)
                    thispix = int(pipe.graph_name_split(name)[2])
                    if thispix in pix:
                        log.debug('{} keeping {}'.format(name, cf))
                        keep.append(cf)
                    else:
                        log.debug('{} discarding {}'.format(name, cf))
                        discard.add((spectrograph, expid))

                nd['in'] = keep

    #- TODO: parallelize this
    #- Check for spectra files that have new frames
    spectra_todo = list()
    if rank == 0:
        for name, nd in sorted(grph.items()):
            if nd["type"] != "spectra":
                continue
            spectrafile = pipe.graph_path(name)
            if not os.path.exists(spectrafile):
                log.info('{} not yet done'.format(name))
                spectra_todo.append(name)
            else:
                fm = fitsio.read(spectrafile, 'FIBERMAP')
                frames_done = list()
                for night, petal, expid in \
                        set(zip(fm['NIGHT'],fm['PETAL_LOC'],fm['EXPID'])):
                    for channel in ['b', 'r', 'z']:
                        tmp = '{}_cframe-{}{}-{:08d}'.format(
                                                      night,channel,petal,expid)
                        frames_done.append(tmp)

                frames_todo = set(nd['in']) - set(frames_done)

                if len(frames_todo) == 0:
                    log.info('All {} frames for {} already in spectra file; skipping'.format(
                        len(nd['in']), name))
                else:
                    log.info('Adding {}/{} frames to {}'.format(
                        len(frames_todo), len(nd['in']), name))
                    nd['in'] = list(frames_todo)
                    spectra_todo.append(name)

        #- Only keep the graph entries with something to do
        grph = pipe.graph_slice(grph, names=spectra_todo, deps=True)

    #- Send graph to all ranks
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
            ndrop = 0
            for fr in existing:
                if fr not in spec_frames[sp]:
                    #log.info("frame {} not needed for spec {}, purging".format(fr, sp))
                    ndrop += 1
                    del framedata[fr]

            # Read frame data if we don't have it
            nadd = 0
            for fr in spec_frames[sp]:
                if fr not in framedata:
                    nadd += 1
                    #log.info("frame {} not in cache for spec {}, reading".format(fr, sp))
                    if os.path.isfile(frame_paths[fr]):
                        framedata[fr] = io.read_frame_as_spectra(frame_paths[fr], 
                        frame_nights[fr], frame_expids[fr], frame_bands[fr])
                    else:
                        framedata[fr] = Spectra()

            msg = "    ({:04d}) dropped {}; added {}; {} frames resident in memory".format(rank, ndrop, nadd, len(framedata))
            log.info(msg)
            sys.stdout.flush()

        # Update spectral data

        for fr in sorted(spec_frames[sp]):
            fdata = None
            if args.cache:
                fdata = framedata[fr]
            else:
                if os.path.isfile(frame_paths[fr]):
                    fdata = io.read_frame_as_spectra(frame_paths[fr], 
                        frame_nights[fr], frame_expids[fr], frame_bands[fr])
                else:
                    log.warning('Missing {}'.format(frame_paths[fr]))
                    fdata = Spectra()
            # Get the targets that hit this pixel.
            targets = []
            fmap = fdata.fibermap
            ra = np.array(fmap["RA_TARGET"], dtype=np.float64)
            dec = np.array(fmap["DEC_TARGET"], dtype=np.float64)
            bad = np.where(fmap["TARGETID"] < 0)[0]
            ra[bad] = 0.0
            dec[bad] = 0.0
            # pix = hp.ang2pix(nside, ra, dec, nest=True, lonlat=True)
            pix = desimodel.footprint.radec2pix(nside, ra, dec)
            pix[bad] = -1

            ii = (specdata.meta['HPIX'] == pix)
            pixtargets = fdata.fibermap['TARGETID'][ii]

            if len(pixtargets) > 0:
                # update with data from this frame
                log.debug('  ({:04d}) Adding {} targets from {} to {}'.format(
                    rank, len(pixtargets), fr, sp))

                specdata.update(fdata.select(targets=pixtargets))

            del fdata

        # Write out updated data

        io.write_spectra(spec_paths[sp], specdata)

        msg = "  ({:04d}) End spectral group {} at {}".format(rank, sp, time.asctime())
        log.info(msg)
        sys.stdout.flush()

    log.info("Rank {} done with {} spectra files at {}".format(
        rank, len(specs), time.asctime()))

    if comm is not None:
        comm.barrier()
    if rank == 0:
        log.info("Finishing at {}".format(time.asctime()))

    return

