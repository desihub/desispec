"""
Regroup spectra by healpix
"""

from __future__ import absolute_import, division, print_function
import os, sys, time

import numpy as np
from astropy.table import Table

from desiutil.log import get_logger

from .. import io
from ..pixgroup import FrameLite, SpectraLite
from ..pixgroup import (get_exp2healpix_map, add_missing_frames,
        frames2spectra, update_frame_cache, FrameLite)
from ..coaddition import coadd

def parse(options=None):
    import argparse

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

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args

def main(args=None):

    log = get_logger()

    if args is None:
        args = parse()

    if (args.inframes is None) and (args.expfile is None):
        log.critical('Must provide --inframes or --expfile')
        sys.exit(1)
    if (args.inframes is not None) and (args.expfile is not None):
        log.critical('Must use --inframes or --expfile but not both')
        sys.exit(1)

    log.info('Starting at {}'.format(time.asctime()))

    #- get list of frames from args.inframes or args.expfile
    if args.inframes is not None:
        framefiles = args.inframes
    else:
        assert args.expfile is not None
        nightexp = Table.read(args.expfile)

        keep = np.ones(len(nightexp), dtype=bool)
        if args.survey is not None:
            log.info(f'Filtering by SURVEY={args.survey}')
            keep &= nightexp['SURVEY'] == args.survey

        if args.faprogram is not None:
            log.info(f'Filtering by FAPRGRM={args.faprogram}')
            keep &= nightexp['FAPRGRM'] == args.faprogram

        if args.healpix is not None:
            keep &= nightexp['HEALPIX'] == args.healpix

        if args.nights is not None:
            nights = [int(x) for x in args.nights.split(',')]
            keep &= np.isin(nightexp['NIGHT'], nights)

        nightexp = nightexp[keep]
        if len(nightexp) == 0:
            log.critical('No exposures passed filters')
            sys.exit(13)

        framefiles = list()
        for night, expid, spectro in nightexp['NIGHT', 'EXPID', 'SPECTRO']:
            for band in ['b', 'r', 'z']:
                camera = band+str(spectro)
                framefile = io.findfile('cframe', night, expid, camera,
                    specprod_dir=args.reduxdir)
                framefiles.append(framefile)

    frames = dict()
    log.info(f'Reading {len(framefiles)} framefiles')
    for filename in framefiles:
        if os.path.exists(filename):
            log.debug('Reading %s', filename)
            frame = FrameLite.read(filename)
            night = frame.meta['NIGHT']
            expid = frame.meta['EXPID']
            camera = frame.meta['CAMERA']
            frames[(night, expid, camera)] = frame
        else:
            log.error(f'Missing {filename} but continuing anyway')

    if len(frames) == 0:
        log.critical('No input frames found')
        sys.exit(1)

    log.info('Combining into spectra')
    spectra = frames2spectra(frames)

    #- Add optional header keywords if requested
    if args.header is not None:
        if spectra.meta is None:
            spectra.meta = dict()

        for keyval in args.header:
            key, value = keyval.split('=', maxsplit=1)
            try:
                spectra.meta[key] = int(value)
            except ValueError:
                try:
                    spectra.meta[key] = float(value)
                except ValueError:
                    spectra.meta[key] = value

    if args.outfile is not None:
        log.info('Writing {}'.format(args.outfile))
        # spectra.write(args.outfile, header=header)
        io.write_spectra(args.outfile, spectra)
        log.info('Done at {}'.format(time.asctime()))

    if args.coaddfile is not None:
        log.info('Coadding spectra')
        #- in-place coadd updates spectra object
        coadd(spectra, onetile=args.onetile)
        log.info('Writing {}'.format(args.coaddfile))
        # spectra.write(args.coaddfile, header=header)
        io.write_spectra(args.coaddfile, spectra)

    log.info('Done at {}'.format(time.asctime()))

    return 0

