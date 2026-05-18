"""
desispec.scripts.healpix_redshifts
==================================

Script for running healpix-based coadds+redshifts, assuming that the
spectral regrouping into healpix has already happened.
"""

import os, sys, time
import json
import subprocess
import numpy as np
import fitsio

from desiutil.log import get_logger

from desispec.workflow.redshifts import create_desi_zproc_batch_script, get_zpix_script_pathname
from desispec import io
from desispec.io.util import get_tempfilename
from desispec.pixgroup import get_exp2uniqpix_map, get_hpix2upix_map
from desispec.workflow import batch

def parse(options=None):
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--survey', type=str, required=True,
            help='survey (e.g. sv3, main)')
    p.add_argument('--program', type=str, required=True,
            help='program (e.g. dark, bright, backup)')
    p.add_argument('--expfile', type=str,
            help='exposure summary file with columns NIGHT,EXPID,TILEID,SURVEY,FAFLAVOR')
    p.add_argument('--uniqpix', type=str, required=False,
            help='uniqpix numbers to run (comma separated)')
    p.add_argument('--bundle-pix', type=int, default=5,
                help='bundle N pixels into a single job (default %(default)s)')
    p.add_argument('--nside-max', type=int, default=256, required=False,
            help='maximum nside to use for healpix->uniqpix map file')
    p.add_argument("--nosubmit", action="store_true",
            help="generate scripts but don't submit batch jobs")
    p.add_argument("--noafterburners", action="store_true",
            help="Do not run afterburners (like QSO fits)")
    p.add_argument("--batch-queue", type=str, default='regular',
            help="batch queue name")
    p.add_argument("--batch-reservation", type=str,
            help="batch reservation name")
    p.add_argument("--batch-dependency", type=str,
            help="job dependencies passed to sbatch --dependency")
    p.add_argument("--system-name", type=str, default=batch.default_system(),
            help="batch system name, e.g. cori-haswell, cori-knl, perlmutter-gpu")
    p.add_argument("--redrock-nodes", type=int, default=1,
            help="Number of nodes per redrock call (default 1)")
    p.add_argument("--redrock-cores-per-rank", type=int, default=1,
            help="cores per rank for redrock; use >1 for more memory per rank")
    p.add_argument("--dry-run-level", type=int, default=0, required=False,
            help="""If nonzero, this is a simulated run with no jobs submitted.
                    If level>=3, no output files are written at all.
                    Lower non-zero levels will create files but not submit jobs.
                    Logging will remain the same for testing as though scripts are being submitted.
                    Default is 0 (i.e. not dry run, submit jobs).""")

    args = p.parse_args(options)
    return args

def main(args):

    t0 = time.time()
    log = get_logger()

    log.info(f'Starting {args.survey} {args.program} uniqpix job submission at {time.asctime()}')
    if args.dry_run_level > 0:
        log.info(f"Dry run set: {args.dry_run_level=}; no actual jobs will be submitted")

    if args.expfile is None:
        args.expfile = io.findfile('exposures')
        if not os.path.exists(args.expfile):
            msg = f'Missing {args.expfile}; please create or specify --expfile'
            log.critical(msg)
            sys.exit(1)

    frames = fitsio.read(args.expfile, 'FRAMES')

    ztilefile = io.findfile('zcat_tile', survey=args.survey, faprogram=args.program, version='v2')
    zcat = fitsio.read(ztilefile, 'ZCATALOG')

    exppix = get_exp2uniqpix_map(zcat, frames)

    reduxdir = io.specprod_root()
    if args.uniqpix is not None:
        allpixels = [int(p) for p in args.uniqpix.split(',')]
    else:
        allpixels = np.unique(np.asarray(exppix['UNIQPIX']))

    #- Save mapping of healpix to uniqpix as the maximum nside in uniqpix
    uniqpix_for_map = np.unique(exppix['UNIQPIX'])
    hpix2upix_map, nside_max = get_hpix2upix_map(uniqpix_for_map, args.nside_max)
    outdir = io.findfile('spectra_base', survey=args.survey, faprogram=args.program)
    header = dict(
            NSIDE = nside_max,
            HPXNSIDE = nside_max, # same as NSIDE, but consistent with other files
            HPXNEST = True,
            SURVEY = args.survey,
            PROGRAM = args.program,
            SPECPROD = os.getenv('SPECPROD', 'unknown'),
            )
    hpixmapfile = f'{outdir}/hpix2upix-{args.survey}-{args.program}.fits'
    tmpfile = get_tempfilename(hpixmapfile)
    with fitsio.FITS(tmpfile, 'rw', clobber=True) as fits:
        fits.write(hpix2upix_map, header=header, extname='HPIX2UPIX')
        fits[0].write_comment(f'HPIX2UPIX[i] is the {args.survey}/{args.program} UNIQPIX')
        fits[0].write_comment(f'    that covers nested NSIDE={nside_max} HEALPIX=i')
    os.rename(tmpfile, hpixmapfile)
    log.info(f'Wrote healpix to uniqpix map for {args.survey} {args.program} to {hpixmapfile}')

    #- also save in json format; augment header with hpix2upix_map array
    header['HPIX2UPIX'] = hpix2upix_map.tolist()
    hpixmapfile = f'{outdir}/hpix2upix-{args.survey}-{args.program}.json'
    tmpfile = get_tempfilename(hpixmapfile)
    with open(tmpfile, 'w') as jsonfile:
        json.dump(header, jsonfile)
    os.rename(tmpfile, hpixmapfile)
    log.info(f'Wrote healpix to uniqpix map for {args.survey} {args.program} to {hpixmapfile}')

    npix = len(allpixels)
    nscripts = 0
    log.info(f'Submitting jobs for {npix} pixels')
    for i in range(0, len(allpixels), args.bundle_pix):
        pixels = allpixels[i:i+args.bundle_pix]
        pixexpfiles = list()
        ntilepetals = 0
        for pix in pixels:
            #- outdir is relative to specprod
            pixexpfile = io.findfile('pixexp', uniqpix=pix, survey=args.survey, faprogram=args.program)
            outdir = os.path.dirname(pixexpfile)
            ## For none dry_run, dry_run_levels 1 or 2, make the directories and csv files
            if args.dry_run_level < 3:
                os.makedirs(outdir, exist_ok=True)
            else:
                log.info(f"Dry run so not making directory: {outdir}")
            ii = exppix['UNIQPIX'] == pix
            ## For none dry_run, dry_run_levels 1 or 2, make the directories and csv files
            if args.dry_run_level < 3:
                exppix[ii].write(pixexpfile, overwrite=True)
            else:
                log.info(f"Dry run so not making the pixexp file: {pixexpfile}")
            ntilepetals += len(set(list(zip(exppix['TILEID'][ii], exppix['SPECTRO'][ii]))))
            pixexpfiles.append(pixexpfile)

        cmdline = [
            'desi_zproc',
            '--groupname', 'uniqpix',
            '--survey', args.survey,
            '--program', args.program,
            ]
        cmdline.append('--uniqpix')
        cmdline.extend( [str(p) for p in pixels] )
        cmdline.append('--expfiles')
        cmdline.extend(pixexpfiles)

        #- very roughly, one minute per input tile-petal with min/max applied
        runtime = max(20, min(ntilepetals, 120))

        ## For none dry_run, dry_run_levels 1 or 2, make the directories and csv files
        if args.dry_run_level < 2:
            batchscript = create_desi_zproc_batch_script(
                group='uniqpix',
                uniqpix=pixels,
                survey=args.survey,
                program=args.program,
                queue=args.batch_queue,
                system_name=args.system_name,
                cmdline=cmdline,
                runtime=runtime,
            )
        else:
            batchscript = get_zpix_script_pathname(pixels, args.survey,
                                                   args.program)
            log.info(f"Dry run so not creating the batch script: {batchscript}"
                     + f"\tfor {pixels=}, {args.survey=}, {args.program=}")

        ### cmd = ['sbatch', '--kill-on-invalid-dep=yes']
        cmd = ['sbatch', ]
        if args.batch_reservation:
            cmd.extend(['--reservation', args.batch_reservation])
        if args.batch_dependency:
            cmd.extend(['--dependency', args.batch_dependency])

        # - sbatch requires the script to be last, after all options
        cmd.append(batchscript)
        nscripts += 1

        if not args.nosubmit and args.dry_run_level == 0:
            err = subprocess.call(cmd)
            basename = os.path.basename(batchscript)
            if err == 0:
                log.info(f'submitted {basename}')
            else:
                log.error(f'Error {err} submitting {basename}')

            time.sleep(0.1)
        else:
            log.info(f"Dry run so not submitting command: {cmd=}")

    if not args.nosubmit and args.dry_run_level == 0:
        log.info(f'Submitted {nscripts} batch scripts')
    else:
        log.info(f'Dry run: would have submitted {nscripts} batch scripts')

    dt = time.time() - t0
    log.info(f'All done at {time.asctime()}; duration {dt/60:.2f} minutes')
