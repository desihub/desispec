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

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args

def main(args=None):

    log = get_logger()

    if args is None:
        args = parse()

    if args.inframes is None and args.expfile is None:
        log.critical('Must provide --inframes or --expfile')
        sys.exit(1)

    header = dict()
    if args.header is not None:
        for keyval in args.header:
            key, value = keyval.split('=', maxsplit=1)
            try:
                header[key] = int(value)
            except ValueError:
                header[key] = value

    #- Combining a set of frame files instead of a healpix?
    if args.inframes is not None:
        log.info('Starting at {}'.format(time.asctime()))
        log.info('Reading {} frame files'.format(len(args.inframes)))
        frames = dict()
        for filename in args.inframes:
            frame = FrameLite.read(filename)
            night = frame.meta['NIGHT']
            expid = frame.meta['EXPID']
            camera = frame.meta['CAMERA']
            frames[(night, expid, camera)] = frame

        log.info('Combining into spectra')
        spectra = frames2spectra(frames)

        log.info('Writing {}'.format(args.outfile))
        spectra.write(args.outfile, header=header)
        log.info('Done at {}'.format(time.asctime()))

        return 0

    #- otherwise args.expfile must be set
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

    frames = dict()
    for night, expid, spectro in nightexp['NIGHT', 'EXPID', 'SPECTRO']:
        for band in ['b', 'r', 'z']:
            camera = band+str(spectro)
            framefile = io.findfile('cframe', night, expid, camera,
                specprod_dir=args.reduxdir)
            if os.path.exists(framefile):
                frames[(night, expid, camera)] = FrameLite.read(framefile)
            else:
                log.warning(f'Missing {framefile}')

    log.info('Combining into spectra')
    spectra = frames2spectra(frames, pix=args.healpix, nside=args.nside)

    log.info('Writing {}'.format(args.outfile))
    spectra.write(args.outfile, header=header)
    log.info('Done at {}'.format(time.asctime()))

    return 0
