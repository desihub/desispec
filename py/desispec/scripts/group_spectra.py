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
    parser.add_argument("--expfile", type=str,
            help="File with NIGHT and EXPID  to use (fits, csv, or ecsv)")
    parser.add_argument("--survey", type=str,
            help="filter by SURVEY (or FA_SURV if SURVEY is missing in inputs)")
    parser.add_argument("--faprogram", type=str,
            help="filter by FAPRGRM.lower() (or FAFLAVOR mapped to a program for sv1")
    parser.add_argument("--nside", type=int, default=64,
            help="input spectra healpix nside (default %(default)s)")
    parser.add_argument("--healpix", type=str,
            help="Comma separated list of healpix to generate")
    parser.add_argument("-o", "--outdir", type=str,
            help="output directory; all outputs in this directory")
    parser.add_argument("--outroot", type=str,
            help="output root directory; files in subdirectories of this dir")
    parser.add_argument("--mpi", action="store_true",
            help="Use MPI for parallelism")
    parser.add_argument("--inframes", type=str, nargs='*',
            help="input frame files; ignore --reduxdir, --nights, --nside")
    parser.add_argument("--outfile", type=str,
            help="output to this file; only used with --inframes")

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args

def main(args=None, comm=None):

    log = get_logger()

    if args is None:
        args = parse()

    login_node = ('NERSC_HOST' in os.environ) & \
                 ('SLURM_JOB_NAME' not in os.environ)

    if comm:
        rank = comm.rank
        size = comm.size
    elif args.mpi and not login_node: 
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 1

    if args.outroot is None and args.outdir is None:
        args.outroot = io.specprod_root()

    #- Combining a set of frame files instead of a healpix?
    if args.inframes is not None:
        if rank == 0:
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
            spectra.write(args.outfile)
            log.info('Done at {}'.format(time.asctime()))


        #- All done; all ranks exit
        return 0

    #- options check
    if args.outfile is not None:
        if rank == 0:
            log.error('Only use --outfile with --inframes options')
        return 1

    if args.expfile:
        if rank == 0:
            nightexp = Table.read(args.expfile)
        else:
            nightexp = None

        if comm is not None:
            nightexp = comm.bcast(nightexp, root=0)

        #- all ranks parse table so that all will fail if there is a problem
        if (('NIGHT' not in nightexp.colnames) or
                ('EXPID' not in nightexp.colnames)):
            msg = f'{args.explist} missing NIGHT and/or EXPID columns'
            log.critical(msg)
            raise ValueError(msg)

        nights = np.unique(nightexp['NIGHT'])
        expids = np.asarray(nightexp['EXPID'])

    elif args.nights:
        nights = [int(night) for night in args.nights.split(',')]
        expids = None
    else:
        nights = None
        expids = None

    if rank == 0:
        if args.survey is not None:
            log.info(f'Filtering by SURVEY={args.survey}')
        else:
            log.info(f'Not filtering by SURVEY')

        if args.faprogram is not None:
            log.info(f'Filtering by FAPRGRM={args.faprogram}')
        else:
            log.info(f'Not filtering by FAPRGRM')

    #- Get table NIGHT EXPID SPECTRO HEALPIX NTARGETS 
    t0 = time.time()
    exp2pix = get_exp2healpix_map(nights=nights, expids=expids, comm=comm,
                                  nside=args.nside, specprod_dir=args.reduxdir,
                                  survey=args.survey, faprogram=args.faprogram)
    assert len(exp2pix) > 0
    if rank == 0:
        dt = time.time() - t0
        log.debug('Exposure to healpix mapping took {:.1f} sec'.format(dt))
        sys.stdout.flush()

    npix = len(np.unique(exp2pix['HEALPIX']))
    log.info(f'{npix} healpix found on {len(exp2pix)} x3 frames')
    if args.healpix is not None:
        keeppix = [int(tmp) for tmp in args.healpix.split(',')]
        log.info(f'Processing healpix {keeppix}')
        keep = np.isin(exp2pix['HEALPIX'], keeppix)
        exp2pix = exp2pix[keep]

    allpix = np.unique(exp2pix[['SURVEY', 'FAPROGRAM', 'HEALPIX']])
    mypix = np.array_split(allpix, size)[rank]
    log.info('Rank {} will process {} pixels'.format(rank, len(mypix)))
    sys.stdout.flush()

    frames = dict()
    for survey, faprogram, pix in mypix:
        keep = (exp2pix['SURVEY'] == survey)
        keep &= (exp2pix['FAPROGRAM'] == faprogram)
        keep &= (exp2pix['HEALPIX'] == pix)
        iipix = np.where(keep)[0]
        ntargets = np.sum(exp2pix['NTARGETS'][iipix])
        log.info('Rank {} pix {} with {} targets on {} frames'.format(
            rank, pix, ntargets, len(iipix)))
        sys.stdout.flush()
        framekeys = list()
        for i in iipix:
            night = exp2pix['NIGHT'][i]
            expid = exp2pix['EXPID'][i]
            spectro = exp2pix['SPECTRO'][i]
            for band in ['b', 'r', 'z']:
                camera = band + str(spectro)
                framefile = io.findfile('cframe', night, expid, camera,
                        specprod_dir=args.reduxdir)
                if os.path.exists(framefile):
                    framekeys.append((night, expid, camera))
                else:
                    #- print warning if file is missing, but proceed;
                    #- will use add_missing_frames later.
                    log.warning('missing {}; will use blank data'.format(framefile))

        #- Identify any frames that are already in pre-existing output file
        specfile = io.findfile('spectra', nside=args.nside, groupname=pix,
                survey=survey, faprogram=faprogram,
                specprod_dir=args.outroot)
        if args.outdir:
            specfile = os.path.join(args.outdir, os.path.basename(specfile))

        oldspectra = None
        if os.path.exists(specfile):
            oldspectra = SpectraLite.read(specfile)
            fm = oldspectra.fibermap
            for night, expid, spectro in set(zip(fm['NIGHT'], fm['EXPID'], fm['PETAL_LOC'])):
                for band in ['b', 'r', 'z']:
                    camera = band + str(spectro)
                    if (night, expid, camera) in framekeys:
                        framekeys.remove((night, expid, camera))

        if len(framekeys) == 0:
            log.info('pix {} already has all exposures; moving on'.format(pix))
            continue

        #- Load new frames to add
        log.info('pix {} has {} frames to add'.format(pix, len(framekeys)))
        update_frame_cache(frames, framekeys, specprod_dir=args.reduxdir)

        #- convert individual FrameLite objects into SpectraLite
        newspectra = frames2spectra(frames, pix, nside=args.nside)

        #- Combine with any previous spectra if needed
        if oldspectra:
            spectra = oldspectra + newspectra
        else:
            spectra = newspectra

        #- Write new spectra file
        header = dict(HPXNSIDE=args.nside, HPXPIXEL=pix, HPXNEST=True)
        spectra.write(specfile, header=header)
    
    dt = time.time() - t0
    log.info('Rank {} done in {:.1f} minutes'.format(rank, dt/60))

