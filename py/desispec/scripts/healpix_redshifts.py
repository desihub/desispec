"""
desispec.scripts.healpix_redshifts
==================================

Script for running healpix-based coadds+redshifts, assuming that the
spectral regrouping into healpix has already happened.
"""

import os, sys, time
import subprocess
import numpy as np

from desiutil.log import get_logger

from desispec.workflow.redshifts import create_desi_zproc_batch_script, get_zpix_redshift_script_pathname
from desispec import io
from desispec.pixgroup import get_exp2healpix_map
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
    p.add_argument('--healpix', type=str, required=False,
            help='nested healpix numbers to run (comma separated)')
    p.add_argument('--bundle-healpix', type=int, default=5,
                help='bundle N healpix into a single job (default %(default)s)')
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
            help="""If nonzero, this is a simulated run.
                    if level>=3, no output files are written at all.
                    Logging will remain the same for testing as though scripts are being submitted.
                    Default is 0 (false).""")

    args = p.parse_args(options)
    return args

def main(args):

    log = get_logger()

    log.info(f"Dry run set: {args.dry_run_level=}")
    if args.expfile is None:
        args.expfile = io.findfile('exposures')
        if not os.path.exists(args.expfile):
            msg = f'Missing {args.expfile}; please create or specify --expfile'
            log.critical(msg)
            sys.exit(1)

    exppix = get_exp2healpix_map(args.expfile, survey=args.survey, program=args.program)

    reduxdir = io.specprod_root()
    if args.healpix is not None:
        allpixels = [int(p) for p in args.healpix.split(',')]
    else:
        allpixels = np.unique(exppix['HEALPIX'])

    npix = len(allpixels)
    log.info(f'Submitting jobs for {npix} healpix')
    for i in range(0, len(allpixels), args.bundle_healpix):
        healpixels = allpixels[i:i+args.bundle_healpix]
        hpixexpfiles = list()
        ntilepetals = 0
        for healpix in healpixels:
            #- outdir is relative to specprod
            rrfile = io.findfile('redrock', healpix=healpix, survey=args.survey, faprogram=args.program)
            outdir = os.path.dirname(rrfile)
            ## For none dry_run, dry_run_levels 1 or 2, make the directories and csv files
            if args.dry_run_level < 3:
                os.makedirs(outdir, exist_ok=True)
            else:
                log.info(f"Dry run so not making directory: {outdir}")
            ii = exppix['HEALPIX'] == healpix
            hpixexpfile = f'{outdir}/hpixexp-{args.survey}-{args.program}-{healpix}.csv'
            ## For none dry_run, dry_run_levels 1 or 2, make the directories and csv files
            if args.dry_run_level < 3:
                exppix[ii].write(hpixexpfile, overwrite=True)
            else:
                log.info(f"Dry run so not making the hpiexp file: {hpixexpfile}")
            ntilepetals += len(set(list(zip(exppix['TILEID'][ii], exppix['SPECTRO'][ii]))))
            hpixexpfiles.append(hpixexpfile)

        cmdline = [
            'desi_zproc',
            '--groupname', 'healpix',
            '--survey', args.survey,
            '--program', args.program,
            ]
        cmdline.append('--healpix')
        cmdline.extend( [str(hp) for hp in healpixels] )
        cmdline.append('--expfiles')
        cmdline.extend(hpixexpfiles)

        #- very roughly, one minute per input tile-petal with min/max applied
        runtime = max(20, min(ntilepetals, 120))

        ## For none dry_run, dry_run_levels 1 or 2, make the directories and csv files
        if args.dry_run_level < 2:
            batchscript = create_desi_zproc_batch_script(
                group='healpix',
                healpix=healpixels,
                survey=args.survey,
                program=args.program,
                queue=args.batch_queue,
                system_name=args.system_name,
                cmdline=cmdline,
                runtime=runtime,
            )
        else:
            batchscript = get_zpix_redshift_script_pathname(healpixels, args.survey,
                                                            args.program)
            log.info(f"Dry run so not creating the batch script: {batchscript}"
                     + f"\tfor {healpixels=}, {args.survey=}, {args.program=}")

        cmd = ['sbatch' ,]
        if args.batch_reservation:
            cmd.extend(['--reservation', args.batch_reservation])
        if args.batch_dependency:
            cmd.extend(['--dependency', args.batch_dependency])

        # - sbatch requires the script to be last, after all options
        cmd.append(batchscript)

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
