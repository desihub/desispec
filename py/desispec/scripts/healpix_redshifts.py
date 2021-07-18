"""
script for running healpix-based coadds+redshifts, assuming that the
spectral regrouping into healpix has already happened
"""

import os, sys
import subprocess
import numpy as np

from desiutil.log import get_logger

from desispec.scripts.tile_redshifts import write_redshift_script
from desispec import io

def parse(options=None):
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--healpix', type=int, required=True,
            help='nested healpix number to run')
    p.add_argument('--survey', type=str, required=True,
            help='survey (e.g. sv3, main)')
    p.add_argument('--faprogram', type=str, required=True,
            help='survey (e.g. dark, bright, backup)')
    p.add_argument('--nside', type=int, default=64,
            help='healpix nside (default 64)')
    p.add_argument("--nosubmit", action="store_true",
            help="generate scripts but don't submit batch jobs")
    p.add_argument("--noafterburners", action="store_true",
            help="Do not run afterburners (like QSO fits)")
    p.add_argument("--batch-queue", type=str, default='realtime',
            help="batch queue name")
    p.add_argument("--batch-reservation", type=str,
            help="batch reservation name")
    p.add_argument("--batch-dependency", type=str,
            help="job dependencies passed to sbatch --dependency")
    p.add_argument("--system-name", type=str,
            help="batch system name, e.g. cori-haswell, cori-knl, perlmutter-gpu")
    p.add_argument("--redrock-nodes", type=int, default=1,
            help="Number of nodes per redrock call (default 1)")
    p.add_argument("--redrock-cores-per-rank", type=int, default=1,
            help="cores per rank for redrock; use >1 for more memory per rank")

    args = p.parse_args(options)
    return args

def main(args):

    log = get_logger()

    specfile = io.findfile('spectra', nside=args.nside, groupname=args.healpix,
            survey=args.survey, faprogram=args.faprogram)
    if not os.path.exists(specfile):
        msg = f'missing {specfile}'
        log.critical(msg)
        raise ValueError(msg)

    #- outdir is relative to specprod
    outdir = f'healpix/{args.survey}/{args.faprogram}/{args.healpix//100}/{args.healpix}'
    suffix = f'{args.faprogram}-{args.healpix}'
    reduxdir = io.specprod_root()
    scriptdir = f'{reduxdir}/run/scripts/healpix/{args.healpix//100}'
    jobname = f'coz-hpix-{args.healpix}'
    batchscript = f'{scriptdir}/{jobname}.slurm'

    os.makedirs(scriptdir, exist_ok=True)

    write_redshift_script(
            batchscript=batchscript,
            outdir=outdir,
            jobname=jobname,
            num_nodes=args.redrock_nodes,
            group='healpix',
            spectro_string=args.survey,
            suffix=suffix,
            frame_glob=None,
            queue=args.batch_queue,
            system_name=args.system_name,
            onetile=False,
            run_zmtl=False,
            noafterburners=args.noafterburners,
            redrock_nodes=args.redrock_nodes,
            redrock_cores_per_rank=args.redrock_cores_per_rank,
            )

    log.info(f'Wrote {batchscript}')
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
