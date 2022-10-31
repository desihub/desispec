"""
Regroup spectra by healpix
"""

from __future__ import absolute_import, division, print_function
import os, sys, time

import numpy as np
import argparse
from astropy.table import Table

from desiutil.log import get_logger

from .. import io
from ..io.meta import shorten_filename
from ..io.util import checkgzip
from ..pixgroup import FrameLite, SpectraLite
from ..pixgroup import (get_exp2healpix_map, add_missing_frames,
        frames2spectra, update_frame_cache, FrameLite)
from ..coaddition import coadd


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

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


def main(args=None):

    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    log = get_logger()

    if (args.inframes is None) and (args.expfile is None):
        log.critical('Must provide --inframes or --expfile')
        return 1
    if (args.inframes is not None) and (args.expfile is not None):
        log.critical('Must use --inframes or --expfile but not both')
        return 1

    log.info('Starting at {}'.format(time.asctime()))

    #- get list of frames from args.inframes or args.expfile
    if args.inframes is not None:
        framefiles = args.inframes
    else:
        assert args.expfile is not None
        log.info(f'Reading exposures to use from {args.expfile}')
        nightexp = Table.read(args.expfile)

        keep = np.ones(len(nightexp), dtype=bool)
        if args.survey is not None:
            log.info(f'Filtering by SURVEY={args.survey}')
            keep &= nightexp['SURVEY'] == args.survey

        if args.faprogram is not None:
            log.info(f'Filtering by FAPRGRM={args.faprogram}')
            keep &= nightexp['FAPRGRM'] == args.faprogram

        if args.healpix is not None and 'HEALPIX' in nightexp.colnames:
            log.info(f'Filtering by healpix={args.healpix}')
            keep &= nightexp['HEALPIX'] == args.healpix

        if args.nights is not None:
            nights = [int(x) for x in args.nights.split(',')]
            log.info(f'Filtering by night in {nights}')
            keep &= np.isin(nightexp['NIGHT'], nights)

        nightexp = nightexp[keep]
        if len(nightexp) == 0:
            log.critical('No exposures passed filters')
            return 13

        framefiles = list()
        for night, expid, spectro in nightexp['NIGHT', 'EXPID', 'SPECTRO']:
            for band in ['b', 'r', 'z']:
                camera = band+str(spectro)
                framefile = io.findfile('cframe', night, expid, camera,
                    specprod_dir=args.reduxdir)
                framefiles.append(framefile)

    frames = dict()
    log.info(f'Reading {len(framefiles)} framefiles')
    foundframefiles = list()
    for filename in framefiles:
        try:
            filename = checkgzip(filename)
        except FileNotFoundError:
            log.warning(f'Missing {filename} but continueing anyway')
            continue

        foundframefiles.append(filename)
        log.debug('Reading %s', filename)
        frame = FrameLite.read(filename)
        night = frame.meta['NIGHT']
        expid = frame.meta['EXPID']
        camera = frame.meta['CAMERA']
        frames[(night, expid, camera)] = frame

    if len(frames) == 0:
        log.critical('No input frames found')
        return 1

    log.info('Combining into spectra')
    spectra = frames2spectra(frames, pix=args.healpix, nside=args.nside)

    if spectra.num_spectra() == 0:
        log.critical(f'No input frame spectra pass nside={args.nside} nested healpix={args.healpix}')
        from desimodel.footprint import radec2pix
        input_hpix = set()
        for frame in frames.values():
            ra = frame.fibermap['TARGET_RA']
            dec = frame.fibermap['TARGET_DEC']
            input_hpix.update(set(radec2pix(args.nside, ra, dec)))
        log.critical(f'Input frames have nside={args.nside} healpix {input_hpix}')
        return 1

    #- Record input files
    if spectra.meta is None:
        spectra.meta = dict()

    for i, filename in enumerate(foundframefiles):
        spectra.meta[f'INFIL{i:03d}'] = shorten_filename(filename)

    #- Add optional header keywords if requested
    if args.header is not None:
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
        io.write_spectra(args.outfile, spectra)

    if args.coaddfile is not None:
        log.info('Coadding spectra')
        #- in-place coadd updates spectra object
        coadd(spectra, onetile=args.onetile)
        log.info('Writing {}'.format(args.coaddfile))
        io.write_spectra(args.coaddfile, spectra)

    log.info('Done at {}'.format(time.asctime()))

    return 0

