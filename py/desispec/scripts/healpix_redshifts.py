"""
script for running healpix-based coadds+redshifts, assuming that the
spectral regrouping into healpix has already happened
"""

import os, sys, time
import subprocess
import numpy as np

from desiutil.log import get_logger

from desispec.workflow.redshifts import create_desi_zproc_batch_script
from desispec import io
from desispec.pixgroup import get_exp2healpix_map
from desispec.workflow import batch

def parse(options=None):
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--healpix', type=str, required=False,
            help='nested healpix numbers to run (comma separated)')
    p.add_argument('--survey', type=str, required=True,
            help='survey (e.g. sv3, main)')
    p.add_argument('--program', type=str, required=True,
            help='program (e.g. dark, bright, backup)')
    p.add_argument('--nside', type=int, default=64,
            help='healpix nside (default 64)')
    p.add_argument('--exptabfile', type=str, required=False,
            help='input production exposures file')
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

    args = p.parse_args(options)
    return args

def main(args):

    log = get_logger()

    exppix = get_exp2healpix_map(survey=args.survey, program=args.program,
            exptabfile=args.exptabfile)

    reduxdir = io.specprod_root()
    if args.healpix is not None:
        pixels = [int(p) for p in args.healpix.split(',')]
    else:
        pixels = np.unique(exppix['HEALPIX'])

    npix = len(pixels)
    log.info(f'Submitting jobs for {npix} healpix')
    for healpix in pixels:
        #- outdir is relative to specprod
        subdir = f'healpix/{args.survey}/{args.program}/{healpix//100}'
        outdir = f'{reduxdir}/{subdir}/{healpix}'
        scriptdir = f'{reduxdir}/run/scripts/{subdir}'
        suffix = f'{args.program}-{healpix}'
        jobname = f'zpix-{args.survey}-{suffix}'

        os.makedirs(scriptdir, exist_ok=True)
        os.makedirs(outdir, exist_ok=True)

        ii = exppix['HEALPIX'] == healpix
        expfile = f'{outdir}/hpixexp-{args.survey}-{args.program}-{healpix}.csv'
        exppix[ii].write(expfile, overwrite=True)

        cmdline = [
            'desi_zproc',
            '--groupname', 'healpix',
            '--healpix', healpix,
            '--expfile', expfile,
            '--survey', args.survey,
            '--program', args.program,
            ]

        batchscript = create_desi_zproc_batch_script(
                group='healpix',
                healpix=healpix,
                survey=args.survey,
                program=args.program,
                queue=args.batch_queue,
                system_name=args.system_name,
                cmdline=cmdline,
                )

        if not args.nosubmit:
            cmd = ['sbatch' ,]
            if args.batch_reservation:
                cmd.extend(['--reservation', args.batch_reservation])
            if args.batch_dependency:
                cmd.extend(['--dependency', args.batch_dependency])

            # - sbatch requires the script to be last, after all options
            cmd.append(batchscript)

            err = subprocess.call(cmd)
            basename = os.path.basename(batchscript)
            if err == 0:
                log.info(f'submitted {basename}')
            else:
                log.error(f'Error {err} submitting {basename}')

            time.sleep(0.1)
