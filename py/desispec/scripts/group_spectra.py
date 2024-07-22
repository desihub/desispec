"""
desispec.scripts.group_spectra
==============================

Regroup spectra by healpix.
"""

from __future__ import absolute_import, division, print_function
import os, sys, time

import numpy as np
import argparse
from astropy.table import Table

from desiutil.log import get_logger
from desimodel.footprint import radec2pix

from .. import io
from ..io.meta import shorten_filename
from ..io.util import checkgzip
from ..pixgroup import FrameLite, SpectraLite
from ..pixgroup import add_missing_frames, frames2spectra
from ..coaddition import coadd
from ..util import parse_keyval


def parse(options=None):
    parser = argparse.ArgumentParser(usage = "{prog} [options]")
    parser.add_argument("--reduxdir", type=str,
            help="input redux dir; overrides $DESI_SPECTRO_REDUX/$SPECPROD")
    parser.add_argument("--nights", type=str,
            help="comma separated YEARMMDDs to add")
    parser.add_argument("--survey", type=str,
            help="filter by SURVEY (or FA_SURV if SURVEY is missing in inputs)")
    parser.add_argument("--faprogram", type=str,
            help="filter by FAPRGRM.lower() (or FAFLAVOR mapped to a program for sv1")
    parser.add_argument("--nside", type=int, default=64,
            help="input spectra healpix nside (default %(default)s)")
    parser.add_argument("--healpix", type=int,
            help="nested healpix to generate")
    parser.add_argument("--header", type=str, nargs="*",
            help="KEYWORD=VALUE entries to add to the output header")
    parser.add_argument("--expfile", type=str,
            help="File with NIGHT and EXPID  to use (fits, csv, or ecsv)")
    parser.add_argument("--inframes", type=str, nargs='*',
            help="input frame files; ignore --reduxdir, --nights, --nside")
    parser.add_argument("-o", "--outfile", type=str,
            help="output spectra filename")
    parser.add_argument("-c", "--coaddfile", type=str,
            help="output coadded spectra filename")
    parser.add_argument("--onetile", action="store_true",
            help="input spectra are from a single tile")
    parser.add_argument("--mpi", action="store_true",
            help="use MPI for parallelism")

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


def _read_framefile(filename, nside=None, healpix=None, ifile=None):
    log = get_logger()
    log.debug('Reading %s', filename)
    frame = FrameLite.read(filename)
    if healpix is not None:
        ra, dec = frame.fibermap['TARGET_RA'], frame.fibermap['TARGET_DEC']
        ok = ~np.isnan(ra) & ~np.isnan(dec)
        ra[~ok] = 0.0
        dec[~ok] = 0.0
        allpix = radec2pix(nside, ra, dec)
        ii = np.where((allpix == healpix) & ok)[0]
        if len(ii) == 0:
            log.warning(f"Frame {filename} had no objects in healpix {args.healpix}. Continuing")
            return None

        frame = frame[ii]

    if ifile%10 == 0:
        log.info(f'Finished reading the {ifile}th file at {time.asctime()}')

    return frame

def main(args=None, comm=None):

    t0 = time.time()
    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    if args.mpi:
        if comm is None:
            import mpi4py.MPI
            comm = mpi4py.MPI.COMM_WORLD

        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 1

    #- logger for all ranks
    log = get_logger()

    #- separate logger for rank 0 to avoid a bunch of if rank==0 boilerplate
    if rank == 0:
        log0 = log
    else:
        log0 = get_logger(level='critical')

    if (args.inframes is None) and (args.expfile is None):
        log0.critical('Must provide --inframes or --expfile')
        return 1
    if (args.inframes is not None) and (args.expfile is not None):
        log0.critical('Must use --inframes or --expfile but not both')
        return 1

    log0.info('Starting at {}'.format(time.asctime()))

    #- get list of frames from args.inframes or args.expfile
    if args.inframes is not None:
        framefiles = args.inframes
    else:
        assert args.expfile is not None
        nightexp = None
        if rank == 0:
            log0.info(f'Reading exposures to use from {args.expfile}')
            nightexp = Table.read(args.expfile)

        if comm is not None:
            nightexp = comm.bcast(nightexp, root=0)

        keep = np.ones(len(nightexp), dtype=bool)
        if args.survey is not None:
            log0.info(f'Filtering by SURVEY={args.survey}')
            keep &= nightexp['SURVEY'] == args.survey

        if args.faprogram is not None:
            log0.info(f'Filtering by FAPRGRM={args.faprogram}')
            keep &= nightexp['FAPRGRM'] == args.faprogram

        if args.healpix is not None and 'HEALPIX' in nightexp.colnames:
            log0.info(f'Filtering by healpix={args.healpix}')
            keep &= nightexp['HEALPIX'] == args.healpix

        if args.nights is not None:
            nights = [int(x) for x in args.nights.split(',')]
            log0.info(f'Filtering by night in {nights}')
            keep &= np.isin(nightexp['NIGHT'], nights)

        nightexp = nightexp[keep]
        if len(nightexp) == 0:
            log0.critical('No exposures passed filters')
            return 13

        framefiles = list()
        if rank == 0:
            for night, expid, spectro in nightexp['NIGHT', 'EXPID', 'SPECTRO']:
                for band in ['b', 'r', 'z']:
                    camera = band+str(spectro)
                    framefile = io.findfile('cframe', night, expid, camera,
                        specprod_dir=args.reduxdir, readonly=True)

                    try:
                        framefile = checkgzip(framefile)
                    except FileNotFoundError:
                        log0.warning(f'Missing {framefile} but continuing anyway')
                        continue

                    framefiles.append(framefile)

        if comm is not None:
            framefiles = comm.bcast(framefiles, root=0)

    read_args = [(framefiles[i], args.nside, args.healpix, i) for i in range(len(framefiles))]
    log0.info(f'Reading {len(framefiles)} framefiles at {time.asctime()}')
    frames = list()
    if comm is not None:
        from mpi4py.futures import MPICommExecutor
        with MPICommExecutor(comm, root=0) as pool:
            frames = list(pool.starmap(_read_framefile, read_args))
    else:
        frames = list()
        for rdargs in read_args:
            frames.append(_read_framefile(*rdargs))

    nframes = len(frames)
    if comm is not None:
        nframes = comm.bcast(nframes, root=0)

    if nframes == 0:
        log0.critical('No input frames found')
        return 1

    #- Output is handled by rank 0
    if rank == 0:
        #- convert to dict for frames2spectra
        framesdict = dict()
        for frame in frames:
            if frame is not None:
                night = frame.meta['NIGHT']
                expid = frame.meta['EXPID']
                camera = frame.meta['CAMERA']
                framesdict[(night, expid, camera)] = frame

        log0.info(f'Combining into spectra at {time.asctime()}')
        spectra = frames2spectra(framesdict, pix=args.healpix, nside=args.nside,
                                 onetile=args.onetile)

        #- Record input files
        if spectra.meta is None:
            spectra.meta = dict()

        max_header_names = 1000
        # we can't write the keywords for more files
        nframefiles = len(framefiles)
        for i, filename in enumerate(framefiles[:max_header_names]):
            spectra.meta[f'INFIL{i:03d}'] = shorten_filename(filename)
        if nframefiles > max_header_names:
            log0.warning(f'There were more than {max_header_names} input files. Only {max_header_names} filenames were written into the header')
            spectra.meta['INFILNUM'] = nframefiles
        
        #- Add healpix provenance keywords
        if args.healpix is not None:
            spectra.meta['SPGRP'] = 'healpix'
            spectra.meta['SPGRPVAL'] = args.healpix
            spectra.meta['HPXPIXEL'] = args.healpix
            spectra.meta['HPXNSIDE'] = args.nside
            spectra.meta['HPXNEST'] = True

        #- Add optional header keywords if requested
        if args.header is not None:
            for keyval in args.header:
                key, value = parse_keyval(keyval)
                spectra.meta[key] = value

        if args.outfile is not None:
            log.info('Writing {}'.format(args.outfile))
            io.write_spectra(args.outfile, spectra)

        if args.coaddfile is not None:
            log.info('Coadding spectra')
            #- in-place coadd updates spectra object
            coadd(spectra, onetile=args.onetile)
            log.info('Writing {}'.format(args.coaddfile))
            io.write_spectra(args.coaddfile, spectra)

    if comm is not None:
        comm.barrier()

    duration_minutes = (time.time() - t0) / 60
    log0.info(f'Done at {time.asctime()} duration {duration_minutes:.1f} minutes')

    return 0

