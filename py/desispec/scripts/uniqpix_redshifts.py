"""
desispec.scripts.healpix_redshifts
==================================

Script to submit jobs to run uniqpix-based combine+coadd+redshift
"""

import os, sys, time
import json
import subprocess
import datetime
import hashlib
import numpy as np
import fitsio
from astropy.table import Table

from desiutil.log import get_logger

from desispec.workflow.redshifts import create_desi_zproc_batch_script, get_zpix_script_pathname
from desispec import io
from desispec.io.util import get_tempfilename
from desispec.pixgroup import get_exp2uniqpix_map, get_hpix2upix_map, group_nspectra
from desispec.workflow import batch

def parse(options=None):
    import argparse

    p = argparse.ArgumentParser(
            description='Submit batch jobs to run redshifts grouped by unique pixel (adaptive sized healpix)')
    p.add_argument('-s', '--survey', type=str, required=True,
            help='survey (e.g. sv3, main)')
    p.add_argument('-p', '--program', type=str, required=True,
            help='program (e.g. dark, bright, backup)')
    p.add_argument('--expfile', type=str,
            help='exposure summary file with columns NIGHT,EXPID,TILEID,SURVEY,FAFLAVOR')
    p.add_argument('--uniqpix', type=str, required=False,
            help='uniqpix numbers to run (comma separated)')
    p.add_argument('--bundle-pix', type=int, default=16,
                help='bundle at most N pixels into a single job (default %(default)s)')
    p.add_argument('--nside-max', type=int, default=256, required=False,
            help='maximum nside to use for healpix->uniqpix map file (default %(default)s)')
    p.add_argument('--ntargets-max', type=int, default=5000, required=False,
            help='maximum number of targets per pixel before splitting into smaller pixels (default %(default)s)')
    p.add_argument('--nspectra-per-job', type=int, default=50000, required=False,
            help='maximum number of input spectra per job (default %(default)s)')
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
    p.add_argument("--system-name", type=str, default=None,
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

    # Lookup system after parsing args, to enable --help to work on any system,
    # even if the default system can't be determined (which raises an exception).
    if args.system_name is None:
        args.system_name = batch.default_system()

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

    exppix, upix_ntargets, hpix_ntargets = get_exp2uniqpix_map(zcat, frames,
                                                               nmax=args.ntargets_max, nside_max=args.nside_max)

    reduxdir = io.specprod_root()

    #- Save mapping of healpix to uniqpix as the maximum nside in uniqpix
    header = dict(
            NSIDE = args.nside_max,
            HPXNSIDE = args.nside_max, # same as NSIDE, but consistent with other files
            HPXNEST = True,
            SURVEY = args.survey,
            PROGRAM = args.program,
            SPECPROD = os.getenv('SPECPROD', 'unknown'),
            )
    hpixmapfile = io.findfile('hpix2upix', survey=args.survey, faprogram=args.program)
    basedir = os.path.dirname(hpixmapfile)
    os.makedirs(basedir, exist_ok=True)
    tmpfile = get_tempfilename(hpixmapfile)
    with fitsio.FITS(tmpfile, 'rw', clobber=True) as fits:
        fits.write(hpix_ntargets['UNIQPIX'], header=header, extname='HPIX2UPIX')
        fits[0].write_comment(f'HPIX2UPIX[i] is the {args.survey}/{args.program} UNIQPIX')
        fits[0].write_comment(f'    that covers nested NSIDE={args.nside_max} HEALPIX=i')
        fits.write(hpix_ntargets['NTARGETS'], header=header, extname='HPIX_NTARGETS')
        fits[0].write_comment(f'HPIX_NTARGETS[i] is the number of {args.survey}/{args.program} targets')
        fits[0].write_comment(f'    covered by nested NSIDE={args.nside_max} HEALPIX=i')
    os.rename(tmpfile, hpixmapfile)
    log.info(f'Wrote healpix to uniqpix map for {args.survey} {args.program} to {hpixmapfile}')

    #- also save in json format; augment header with hpix2upix_map array
    header['HPIX2UPIX'] = hpix_ntargets['UNIQPIX'].tolist()
    header['HPIX_NTARGETS'] = hpix_ntargets['NTARGETS'].tolist()
    hpixmapfile = io.findfile('hpix2upix_json', survey=args.survey, faprogram=args.program)
    tmpfile = get_tempfilename(hpixmapfile)
    with open(tmpfile, 'w') as jsonfile:
        json.dump(header, jsonfile)
    os.rename(tmpfile, hpixmapfile)
    log.info(f'Wrote healpix to uniqpix map for {args.survey} {args.program} to {hpixmapfile}')

    #- One more summary table: for each uniqpix, the number of targets from upix_ntargets
    upix_ntargets.meta['EXTNAME'] = 'UNIQPIX'
    upix_ntargets.meta['SURVEY'] = args.survey
    upix_ntargets.meta['PROGRAM'] = args.program
    upix_ntargets.meta['SPECPROD'] = os.getenv('SPECPROD', 'unknown')
    upixfile = io.findfile('upix_ntargets', survey=args.survey, faprogram=args.program)
    tmpfile = get_tempfilename(upixfile)
    upix_ntargets.write(tmpfile, overwrite=True)
    os.rename(tmpfile, upixfile)

    #- Trim to just the requested pixels to process
    if args.uniqpix is not None:
        todo_pixels = [int(p) for p in args.uniqpix.split(',')]
        keep = np.isin(exppix['UNIQPIX'], todo_pixels)
        exppix = exppix[keep]
        keep = np.isin(upix_ntargets['UNIQPIX'], todo_pixels)
        upix_ntargets = upix_ntargets[keep]
        keep = np.isin(hpix_ntargets['UNIQPIX'], todo_pixels)
        hpix_ntargets = hpix_ntargets[keep]

    group_indices = group_nspectra(upix_ntargets['NSPECTRA'], nmax=args.nspectra_per_job, max_groupsize=args.bundle_pix)

    npix = len(upix_ntargets)
    nscripts = 0
    jobtracker = list()
    log.info(f'Submitting jobs for {npix} pixels')
    for group in group_indices:
        pixels = upix_ntargets['UNIQPIX'][group].tolist()
        ntargets = int(np.sum(upix_ntargets['NTARGETS'][group]))
        nspectra = int(np.sum(upix_ntargets['NSPECTRA'][group]))
        npix = len(pixels)
        log.info(f'Group {nscripts}: {npix=} {ntargets=} {nspectra=} pixels={list(pixels)}')
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
            batchscript, jobhash = create_desi_zproc_batch_script(
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
            batchscript, jobhash = get_zpix_script_pathname(pixels, args.survey,
                                                   args.program)
            log.info(f"Dry run so not creating the batch script: {batchscript}"
                     + f"\tfor {pixels=}, {args.survey=}, {args.program=}")

        ### cmd = ['sbatch', '--kill-on-invalid-dep=yes']
        cmd = ['sbatch', '--parsable']
        if args.batch_reservation:
            cmd.extend(['--reservation', args.batch_reservation])
        if args.batch_dependency:
            cmd.extend(['--dependency', args.batch_dependency])

        # - sbatch requires the script to be last, after all options
        cmd.append(batchscript)
        nscripts += 1

        if not args.nosubmit and args.dry_run_level == 0:
            basename = os.path.basename(batchscript)
            try:
                qid = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                qid = int(qid.strip(' \t\n'))
                log.info(f'submitted {basename}')
            except CalledProcessError as err:
                qid = -1
                log.error(f'Error submitting {basename} at {datetime.datetime.now()}')
                log.error(f'{basename} {err.output=}')

            time.sleep(0.1)
        else:
            qid = -1
            log.info(f"Dry run so not submitting command: {cmd=}")

        pixstr = '|'.join([str(p) for p in pixels])
        jobtracker.append( (jobhash, qid, npix, ntargets, nspectra, pixstr) )

    if not args.nosubmit and args.dry_run_level == 0:
        log.info(f'Submitted {nscripts} batch scripts')
    else:
        log.info(f'Dry run: would have submitted {nscripts} batch scripts')

    jobtracker = Table(rows=jobtracker, names=['JOBHASH', 'QID', 'NPIX', 'NTARGETS', 'NSPECTRA', 'UNIQPIX'])
    scriptbase = os.path.dirname(os.path.dirname(batchscript))
    jobtracker.write(f'{scriptbase}/pixjobs-{args.survey}-{args.program}.csv', format='csv', overwrite=True)

    dt = time.time() - t0
    log.info(f'All done at {time.asctime()}; duration {dt/60:.2f} minutes')
