"""
desispec.scripts.zproc
======================

One stop shopping for redshifting  DESI spectra
"""

import time
start_imports = time.time()

#- python imports
import datetime
import sys, os, argparse, re
import subprocess
from copy import deepcopy
import json
import glob

#- external 3rd party imports
import numpy as np
import fitsio
from astropy.io import fits
from astropy.table import Table,vstack

#- external desi imports
from redrock.external import desi
import desiutil.timer
from desiutil.log import get_logger, DEBUG, INFO
import desiutil.iers

from desispec.io.meta import get_nights_up_to_date
from desispec.workflow.redshifts import create_desi_zproc_batch_script

#- internal desispec imports
import desispec.io
from desispec.io import findfile, specprod_root, replace_prefix, shorten_filename, get_readonly_filepath
from desispec.io.util import create_camword, decode_camword, parse_cameras, \
    camword_to_spectros, columns_to_goodcamword, difference_camwords
from desispec.io.util import validate_badamps, get_tempfilename, backup_filename
from desispec.util import runcmd
from desispec.scripts import group_spectra
from desispec.parallel import stdouterr_redirected
from desispec.workflow import batch
from desispec.workflow.exptable import get_exposure_table_pathname, \
    read_minimal_science_exptab_cols
from desispec.workflow.desi_proc_funcs import assign_mpi, update_args_with_headers, log_timer
from desispec.workflow.desi_proc_funcs import determine_resources, create_desi_proc_batch_script

stop_imports = time.time()

#########################################
######## Begin Body of the Code #########
#########################################
def parse(options=None):
    """
    Parses an argparser object for use with desi_zproc and returns arguments
    """
    parser = argparse.ArgumentParser(usage="{prog} [options]")

    parser.add_argument("-g", "--groupname", type=str,
                        help="Redshift grouping type: cumulative, perexp, pernight, healpix")

    #- Options for tile-based redshifts
    tiles_options = parser.add_argument_group("tile-based options (--groupname perexp, pernight, or cumulative)")
    tiles_options.add_argument("-t", "--tileid", type=int, default=None,
                        help="Tile ID")
    tiles_options.add_argument("-n", "--nights", type=int, nargs='*', default=None,
                        help="YEARMMDD night")
    tiles_options.add_argument("-e", "--expids", type=int, nargs='*', default=None,
                        help="Exposure IDs")
    tiles_options.add_argument("--thrunight", type=int, default=None,
                        help="Last night to include (YEARMMDD night) for "
                             + "cumulative redshift jobs. Used instead of nights.")
    tiles_options.add_argument("-c", "--cameras", type=str,
                        help="Subset of cameras to process, either as a camword (e.g. a012)" +
                             "Or a comma separated list (e.g. b0,r0,z0).")

    #- Options for healpix-based redshifts
    healpix_options = parser.add_argument_group("healpix-based options (--groupname healpix)")
    healpix_options.add_argument("-p", "--healpix", type=int, nargs='*', default=None,
            help="healpix pixels (nested nside=64")
    healpix_options.add_argument("--survey", help="survey name, e.g. main,sv1,sv3")
    healpix_options.add_argument("--program", help="program name, e.g. dark,bright,backup,other")
    healpix_options.add_argument("--expfiles", nargs='*',
                        help="csv files with NIGHT,EXPID,SPECTRO,HEALPIX")
    healpix_options.add_argument("--prodexpfile", help="production summary exposure file (using pre-generated --expfiles is more efficient)")

    #- Processing options
    processing_options = parser.add_argument_group('processing options')
    processing_options.add_argument("--mpi", action="store_true",
                        help="Use MPI parallelism")
    processing_options.add_argument("--no-gpu", action="store_true",
                        help="Don't use gpus")
    processing_options.add_argument("--max-gpuprocs", type=int, default=4,
                        help="Number of GPU prcocesses per node")
    processing_options.add_argument("--run-zmtl", action="store_true",
                        help="Whether to run zmtl or not")
    processing_options.add_argument("--no-afterburners", action="store_true",
                        help="Set if you don't want to run afterburners")
    processing_options.add_argument("--starttime", type=float,
                        help='start time; use "--starttime $(date +%%s)"')
    processing_options.add_argument("--timingfile", type=str,
                        help='save runtime info to this json file; augment if pre-existing')
    processing_options.add_argument("-d", "--dryrun", action="store_true",
                        help="show commands only, do not run")

    #- Batch submission options
    batch_options = parser.add_argument_group("batch queue options")
    batch_options.add_argument("--batch", action="store_true",
                        help="Submit a batch job to process this exposure")
    batch_options.add_argument("--nosubmit", action="store_true",
                        help="Create batch script but don't submit")
    batch_options.add_argument("-q", "--queue", type=str, default="realtime",
                        help="batch queue to use (default %(default)s)")
    batch_options.add_argument("--batch-opts", type=str, default=None,
                        help="additional batch commands")
    batch_options.add_argument("--batch-reservation", type=str,
                        help="batch reservation name")
    batch_options.add_argument("--batch-dependency", type=str,
                        help="job dependencies passed to sbatch --dependency")
    batch_options.add_argument("--runtime", type=int, default=None,
                        help="batch runtime in minutes")
    batch_options.add_argument("--system-name", type=str,
                        default=None,
                        help='Batch system name (cori-haswell, perlmutter-gpu, ...)')

    if options is not None:
        args = parser.parse_args(options)
    else:
        args = parser.parse_args()

    return args


def main(args=None, comm=None):
    if not isinstance(args, argparse.Namespace):
        args = parse(options=args)

    if args.starttime is not None:
        start_time = args.starttime
    else:
        start_time = time.time()

    log = get_logger()

    start_mpi_connect = time.time()
    if comm is not None:
        ## Use the provided comm to determine rank and size
        rank = comm.rank
        size = comm.size
    else:
        ## Check MPI flags and determine the comm, rank, and size given the arguments
        comm, rank, size = assign_mpi(do_mpi=args.mpi, do_batch=args.batch, log=log)
    stop_mpi_connect = time.time()

    #- set default groupname if needed (cumulative for tiles, otherwise healpix)
    if args.groupname is None:
        if args.tileid is not None:
            args.groupname = 'cumulative'
        elif args.healpix is not None:
            args.groupname = 'healpix'
        else:
            msg = 'Must specify --tileid or --healpix'
            log.critical(msg)
            raise ValueError(msg)

    #- consistency of options
    if args.groupname == 'healpix':
        assert args.healpix is not None, "--groupname healpix requires setting --healpix too"
        assert args.nights is None, f"--groupname healpix doesn't use --nights {args.nights}"
        assert args.expids is None, f"--groupname healpix doesn't use --expids {args.expids}"
        assert args.thrunight is None, f"--groupname healpix doesn't use --thrunight {args.thrunight}"
        assert args.cameras is None, f"--groupname healpix doesn't use --cameras {args.cameras}"
        assert (args.expfiles is None) or (args.prodexpfile is None), \
                "--groupname healpix use --expfiles OR --prodexpfile but not both"
    else:
        assert args.tileid is not None, f"--groupname {args.groupname} requires setting --tileid too"
        if args.cameras is None:
            args.cameras = 'a0123456789'

    if args.expfiles is not None:
        if args.nights is not None or args.expids is not None:
            msg = "use --expfiles OR --nights and --expids, but not both"
            log.error(msg)
            raise ValueError(msg)

    today = int(time.strftime('%Y%m%d'))
    if args.thrunight is not None:
        if args.groupname not in ['cumulative',]:
            msg = f"--thrunight only valid for cumulative redshifts."
            log.error(msg)
            raise ValueError(msg)
        #- very early data isn't supported, and future dates aren't supported
        #- because that implies data are included that don't yet exist
        elif args.thrunight < 20200214 or args.thrunight > today:
            msg = f"--thrunight must be between 20200214 and today"
            log.error(msg)
            raise ValueError(msg)

    if args.expids is not None:
        if args.nights is None:
            msg = f"Must specify --nights if specifying --expids."
            log.error(msg)
            raise ValueError(msg)
        else:
            if rank == 0:
                msg = f"Only using exposures specified with --expids {args.expids}"
                log.info(msg)

    if args.groupname in ['perexp', 'pernight'] and args.nights is not None:
        if len(args.nights) > 1:
            msg = f"Only expect one night for groupname {args.groupname}" \
                  + f" but received nights={args.nights}."
            log.error(msg)
            raise ValueError(msg)

    if (args.groupname == 'healpix') and (args.expfiles is None) and (args.prodexpfile is None):
        args.prodexpfile = findfile('exposures')
        if rank == 0:
            log.info(f'Using default --prodexpfile {args.prodexpfile}')
        if not os.path.exists(args.prodexpfile):
            msg = f'Missing {args.prodexpfile}; please create with desi_tsnr_afterburner or specify different --prodexpfile'
            if rank == 0:
                log.critical(msg)

            raise ValueError(msg)

    #- redrock non-MPI mode isn't compatible with GPUs,
    #- so if zproc is running in non-MPI mode, force --no-gpu
    #- https://github.com/desihub/redrock/issues/223
    if (args.mpi == False) and (args.no_gpu == False) and (not args.batch):
        log.warning("Redrock+GPU currently only works with MPI; since this is non-MPI, forcing --no-gpu")
        log.warning("See https://github.com/desihub/redrock/issues/223")
        args.no_gpu = True

    error_count = 0

    if rank == 0:
        thisfile=os.path.dirname(os.path.abspath(__file__))
        thistime=datetime.datetime.fromtimestamp(start_imports).isoformat()
        log.info(f'rank 0 started {thisfile} at {thistime}')
    ## Start timer; only print log messages from rank 0 (others are silent)
    timer = desiutil.timer.Timer(silent=rank>0)

    ## Fill in timing information for steps before we had the timer created
    if args.starttime is not None:
        timer.start('startup', starttime=args.starttime)
        timer.stop('startup', stoptime=start_imports)

    timer.start('imports', starttime=start_imports)
    timer.stop('imports', stoptime=stop_imports)

    timer.start('mpi_connect', starttime=start_mpi_connect)
    timer.stop('mpi_connect', stoptime=stop_mpi_connect)

    ## Freeze IERS after parsing args so that it doesn't bother if only --help
    timer.start('freeze_iers')
    ## Redirect all of the freeze_iers messages to /dev/null
    with stdouterr_redirected(comm=comm):
        desiutil.iers.freeze_iers()
    if rank == 0:
        log.info("Froze iers for all ranks")
    timer.stop('freeze_iers')


    timer.start('preflight')

    ## Derive the available cameras
    if args.groupname == 'healpix':
        camword = 'a0123456789'
    elif isinstance(args.cameras, str):
        if rank == 0:
            camword = parse_cameras(args.cameras)
        else:
            camword = parse_cameras(args.cameras, loglevel='ERROR')
    else:
        camword = create_camword(args.cameras)

    ## Unpack arguments for shorter names (tileid might be None, ok)
    tileid, groupname = args.tileid, args.groupname

    known_groups = ['cumulative', 'pernight', 'perexp', 'healpix']
    if groupname not in known_groups:
        msg = 'obstype {} not in {}'.format(groupname, known_groups)
        log.error(msg)
        raise ValueError(msg)

    if args.batch:
        err = 0
        #-------------------------------------------------------------------------
        ## Create and submit a batch job if requested
        if rank == 0:
            ## create the batch script
            cmdline = list(sys.argv).copy()
            scriptfile = create_desi_zproc_batch_script(group=groupname,
                                                        tileid=tileid,
                                                        cameras=camword,
                                                        thrunight=args.thrunight,
                                                        nights=args.nights,
                                                        expids=args.expids,
                                                        healpix=args.healpix,
                                                        survey=args.survey,
                                                        program=args.program,
                                                        queue=args.queue,
                                                        runtime=args.runtime,
                                                        batch_opts=args.batch_opts,
                                                        timingfile=args.timingfile,
                                                        system_name=args.system_name,
                                                        no_gpu=args.no_gpu,
                                                        max_gpuprocs=args.max_gpuprocs,
                                                        cmdline=cmdline)

            log.info("Generating batch script and exiting.")

            if not args.nosubmit and not args.dryrun:
                err = subprocess.call(['sbatch', scriptfile])

        ## All ranks need to exit if submitted batch
        if comm is not None:
            err = comm.bcast(err, root=0)
        sys.exit(err)

    exposure_table = None
    hpixexp = None
    if rank == 0:

        if groupname != 'healpix' and args.expfiles is not None:
            tmp = vstack([Table.read(fn) for fn in args.expfiles])
            args.expids = list(tmp['EXPID'])
            args.nights = list(tmp['NIGHT'])

        if groupname == 'healpix':
            if args.expfiles is not None:
                hpixexp = vstack([Table.read(fn) for fn in args.expfiles])
            else:
                from desispec.pixgroup import get_exp2healpix_map
                hpixexp = get_exp2healpix_map(args.prodexpfile, survey=args.survey, program=args.program)

            keep = np.isin(hpixexp['HEALPIX'], args.healpix)
            hpixexp = hpixexp[keep]

        elif groupname == 'perexp' and args.nights is not None \
                and args.cameras is not None and args.expids is not None:
            assert len(args.expids) == 1, "perexp job should only have one exposure"
            assert len(args.nights) == 1, "perexp job should only have one night"
            exposure_table = Table([Table.Column(name='EXPID', data=args.expids),
                                    Table.Column(name='NIGHT', data=args.nights),
                                    Table.Column(name='CAMWORD', data=[camword]),
                                    Table.Column(name='BADCAMWORD', data=[''])])
        else:
            if args.nights is not None:
                nights = args.nights
            elif args.thrunight is None:
                ## None will glob for all nights
                nights = None
            else:
                ## Get list of only nights up to date of thrunight
                nights = get_nights_up_to_date(args.thrunight)

            exposure_table = read_minimal_science_exptab_cols(nights=nights,
                                                              tileids=[tileid])
            if args.expids is not None:
                exposure_table = exposure_table[np.isin(exposure_table['EXPID'],
                                                        args.expids)]
            exposure_table.sort(keys=['EXPID'])

    ## Should remove, just nice for printouts while performance isn't important
    if comm is not None:
        comm.barrier()
        if groupname == 'healpix':
            hpixexp = comm.bcast(hpixexp, root=0)
        else:
            exposure_table = comm.bcast(exposure_table, root=0)

    if groupname != 'healpix':
        if len(exposure_table) == 0:
            msg = f"Didn't find any exposures!"
            log.error(msg)
            raise ValueError(msg)

        ## Get night and expid information
        expids = np.unique(exposure_table['EXPID'].data)
        nights = np.unique(exposure_table['NIGHT'].data)
        thrunight = np.max(nights)
    else:
        expids = None
        nights = None
        thrunight = None

    #------------------------------------------------------------------------#
    #------------------------ Proceed with running --------------------------#
    #------------------------------------------------------------------------#

    ## Print a summary of what we're going to do
    if rank == 0:
        log.info('------------------------------')
        log.info('Groupname {}'.format(groupname))
        if args.healpix is not None:
            log.info(f'Healpixels {args.healpix}')
        else:
            log.info(f'Tileid={tileid} nights={nights} expids={expids}')

        log.info(f'Supplied camword: {camword}')
        log.info('Output root {}'.format(desispec.io.specprod_root()))
        if args.run_zmtl:
            log.info(f'Will be running zmtl')
        if not args.no_afterburners:
            log.info(f'Will be running aferburners')
        log.info('------------------------------')

    if comm is not None:
        comm.barrier()

    ## Derive the available spectrographs
    if groupname == 'healpix':
        all_subgroups = args.healpix
    else:
        ## Find nights, exposures, and camwords
        expnight_dict = dict()
        complete_cam_set = set()

        camword_set = set(decode_camword(camword))
        for erow in exposure_table:
            key = (erow['EXPID'],erow['NIGHT'])
            val = set(decode_camword(difference_camwords(erow['CAMWORD'],
                                                         erow['BADCAMWORD'],
                                                         suppress_logging=True)))
            if camword != 'a0123456789':
                val = camword_set.intersection(val)

            complete_cam_set = complete_cam_set.union(val)
            expnight_dict[key] = val

        all_subgroups = camword_to_spectros(create_camword(list(complete_cam_set)),
                                           full_spectros_only=False)

        if len(all_subgroups) == 0:
            msg = f"Didn't find any spectrographs! complete_cam_set={complete_cam_set}"
            log.error(msg)
            raise ValueError(msg)

    ## options to be used by findfile for all output files
    if groupname == 'healpix':
        findfileopts = dict(groupname=groupname, survey=args.survey, faprogram=args.program)
    else:
        findfileopts = dict(night=thrunight, tile=tileid, groupname=groupname)
        if groupname == 'perexp':
            assert len(expids) == 1
            findfileopts['expid'] = expids[0]

    timer.stop('preflight')

    #-------------------------------------------------------------------------
    ## Do spectral grouping and coadding
    timer.start('groupspec')

    nblocks, block_size, block_rank, block_num = \
        distribute_ranks_to_blocks(len(all_subgroups), rank=rank, size=size, log=log)

    if rank == 0:
        if groupname == 'healpix':
            for hpix in args.healpix:
                findfileopts['healpix'] = hpix
                splog = findfile('spectra', spectrograph=0, logfile=True, **findfileopts)
                os.makedirs(os.path.dirname(splog), exist_ok=True)
        else:
            splog = findfile('spectra', spectrograph=0, logfile=True, **findfileopts)
            os.makedirs(os.path.dirname(splog), exist_ok=True)

    if comm is not None:
        comm.barrier()

    if block_rank == 0:
        for i in range(block_num, len(all_subgroups), nblocks):
            result, success = 0, True
            if groupname == 'healpix':
                healpix = all_subgroups[i]
                log.info(f'Coadding spectra for healpix {healpix}')
                findfileopts['healpix'] = healpix

                cframes = []
                ii = hpixexp['HEALPIX'] == healpix
                for night, expid, spectro in hpixexp['NIGHT', 'EXPID', 'SPECTRO'][ii]:
                    for band in ('b', 'r', 'z'):
                        camera = band+str(spectro)
                        filename = findfile('cframe', night=night, expid=expid, camera=camera,
                                            readonly=True)
                        if os.path.exists(filename):
                            cframes.append(filename)
                        else:
                            log.warning(f'Missing {filename}')

                if len(cframes) < 3:
                    log.error(f'healpix {healpix} only has {len(cframes)} cframes; skipping')
                    error_count += 1
                    continue

            else:
                spectro = all_subgroups[i]
                log.info(f'Coadding spectra for spectrograph {spectro}')
                findfileopts['spectrograph'] = spectro

                # generate list of cframes from dict of exposures, nights, and cameras
                cframes = []
                for (expid, night), cameras in expnight_dict.items():
                    for camera in cameras:
                        if int(spectro) == int(camera[1]):
                            cframes.append(findfile('cframe', night=night,
                                                    expid=expid, camera=camera,
                                                    readonly=True))

            spectrafile = findfile('spectra', **findfileopts)
            splog = findfile('spectra', logfile=True, **findfileopts)
            coaddfile = findfile('coadd', **findfileopts)

            cmd = f"desi_group_spectra --inframes {' '.join(cframes)} " \
                  + f"--outfile {spectrafile} " \
                  + f"--coaddfile {coaddfile} "

            if groupname == 'healpix':
                cmd += f"--healpix {healpix} "
                cmd += f"--header SURVEY={args.survey} PROGRAM={args.program} "
            else:
                cmd += "--onetile "
                cmd += (f"--header SPGRP={groupname} SPGRPVAL={thrunight} "
                        f"NIGHT={thrunight} TILEID={tileid} SPECTRO={spectro} PETAL={spectro} ")

                if groupname == 'perexp':
                    cmd += f'EXPID={expids[0]} '

            cmdargs = cmd.split()[1:]
            if args.dryrun:
                if rank == 0:
                    log.info(f"dryrun: Would have run {cmd}")
            else:
                with stdouterr_redirected(splog):
                    result, success = runcmd(group_spectra.main,
                                             args=cmdargs, inputs=cframes,
                                             outputs=[spectrafile, coaddfile])

            if not success:
                log.error(f'desi_group_spectra petal {spectro} failed; see {splog}')
                error_count += 1

    timer.stop('groupspec')

    if comm is not None:
        comm.barrier()

    if rank == 0:
        log.info("Done with spectra")

    #-------------------------------------------------------------------------
    ## Do redshifting
    timer.start('redrock')
    for subgroup in all_subgroups:
        result, success = 0, True

        if groupname == 'healpix':
            findfileopts['healpix'] = subgroup
        else:
            findfileopts['spectrograph'] = subgroup

        coaddfile = findfile('coadd', **findfileopts)
        rrfile = findfile('redrock', **findfileopts)
        rdfile = findfile('rrdetails', **findfileopts)
        rmfile = findfile('rrmodel', **findfileopts)
        rrlog = findfile('redrock', logfile=True, **findfileopts)

        cmd = f"rrdesi_mpi -i {coaddfile} -o {rrfile} -d {rdfile} --model {rmfile}"
        if not args.no_gpu:
            cmd += f' --gpu --max-gpuprocs {args.max_gpuprocs}'

        cmdargs = cmd.split()[1:]
        if args.dryrun:
            if rank == 0:
                log.info(f"dryrun: Would have run {cmd}")
        else:
            with stdouterr_redirected(rrlog, comm=comm):
                result, success = runcmd(desi.rrdesi, comm=comm, args=cmdargs,
                                         inputs=[coaddfile], outputs=[rrfile, rdfile, rmfile])

        ## Since all ranks running redrock, only count failure/success once
        if rank == 0 and not success:
            log.error(f'Redrock petal/healpix {subgroup} failed; see {rrlog}')
            error_count += 1

        ## Since all ranks running redrock, ensure we're all moving on to next
        ## iteration together
        if comm is not None:
            comm.barrier()

    if comm is not None:
        comm.barrier()

    timer.stop('redrock')

    if rank == 0:
        log.info("Done with redrock")

    #-------------------------------------------------------------------------
    ## Do tileqa if a tile (i.e. not for healpix)
    timer.start('tileqa')

    if rank == 0 and groupname in ['pernight', 'cumulative']:
        from desispec.scripts import tileqa

        result, success = 0, True
        qafile = findfile('tileqa', **findfileopts)
        qapng = findfile('tileqapng', **findfileopts)
        qalog = findfile('tileqa', logfile=True, **findfileopts)
        ## requires all coadd and redrock outputs in addition to exposureqa
        infiles = []
        for expid, night in zip(expids, nights):
            infiles.append(findfile('exposureqa', expid=expid, night=night, readonly=True))
        for spectro in all_subgroups:
            findfileopts['spectrograph'] = spectro
            infiles.append(findfile('coadd', **findfileopts))
            infiles.append(findfile('redrock', **findfileopts))
        cmd = f"desi_tile_qa -g {groupname} -n {thrunight} -t {tileid}"
        cmdargs = cmd.split()[1:]
        if args.dryrun:
            log.info(f"dryrun: Would have run {cmd} with"
                     + f"outputs {qafile}, {qapng}")
        else:
            with stdouterr_redirected(qalog):
                result, success = runcmd(tileqa.main, args=cmdargs,
                                         inputs=infiles, outputs=[qafile, qapng])

            ## count failure/success
            if not success:
                log.error(f'tileqa failed; see {qalog}')
                error_count += 1

        log.info("Done with tileqa")

    timer.stop('tileqa')

    if comm is not None:
        comm.barrier()

    #-------------------------------------------------------------------------
    ## Do zmtl if asked to
    if args.run_zmtl:
        from desispec.scripts import makezmtl

        timer.start('zmtl')
        if block_rank == 0:
            for i in range(block_num, len(all_subgroups), nblocks):
                result, success = 0, True
                if groupname == 'healpix':
                    findfileopts['healpix'] = all_subgroups[i]
                else:
                    findfileopts['spectrograph'] = all_subgroups[i]

                rrfile = findfile('redrock', **findfileopts)
                zmtlfile = findfile('zmtl', **findfileopts)
                zmtllog = findfile('zmtl', logfile=True, **findfileopts)
                cmd = f"make_zmtl_files --input_file {rrfile} --output_file {zmtlfile}"
                cmdargs = cmd.split()[1:]
                if args.dryrun:
                    if rank == 0:
                        log.info(f"dryrun: Would have run {cmd}")
                else:
                    with stdouterr_redirected(zmtllog):
                        result, success = runcmd(makezmtl.main, args=cmdargs,
                                                 inputs=[rrfile],
                                                 outputs=[zmtlfile])
                if not success:
                    log.error(f'zmtl petal/healpix {all_subgroups[i]} failed; see {zmtllog}')
                    error_count += 1

        if rank == 0:
            log.info("Done with zmtl")

        timer.stop('zmtl')

    if comm is not None:
        comm.barrier()

    #-------------------------------------------------------------------------
    ## Do afterburners if asked to
    if not args.no_afterburners:
        from desispec.scripts import qsoqn, qsomgii, emlinefit
        """
        for SPECTRO in 0 1 2 3 4 5 6 7 8 9; do
            coadd=tiles/cumulative/2288/20220918/coadd-$SPECTRO-2288-thru20220918.fits
            redrock=tiles/cumulative/2288/20220918/redrock-$SPECTRO-2288-thru20220918.fits
            qsomgii=tiles/cumulative/2288/20220918/qso_mgii-$SPECTRO-2288-thru20220918.fits
            qsoqn=tiles/cumulative/2288/20220918/qso_qn-$SPECTRO-2288-thru20220918.fits
            emfit=tiles/cumulative/2288/20220918/emline-$SPECTRO-2288-thru20220918.fits
            qsomgiilog=tiles/cumulative/2288/20220918/logs/qso_mgii-$SPECTRO-2288-thru20220918.log
            qsoqnlog=tiles/cumulative/2288/20220918/logs/qso_qn-$SPECTRO-2288-thru20220918.log
            emfitlog=tiles/cumulative/2288/20220918/logs/emline-$SPECTRO-2288-thru20220918.log
            cmd="srun -N 1 -n 1 -c 64 --cpu-bind=none desi_qso_mgii_afterburner --coadd $coadd --redrock $redrock --output $qsomgii --target_selection all --save_target all"
            cmd="srun -N 1 -n 1 -c 64 --cpu-bind=none desi_qso_qn_afterburner --coadd $coadd --redrock $redrock --output $qsoqn --target_selection all --save_target all"
            cmd="srun -N 1 -n 1 -c 64 --cpu-bind=none desi_emlinefit_afterburner --coadd $coadd --redrock $redrock --output $emfit"
        """
        timer.start('afterburners')
        nafterburners = 3
        nsubgroups = len(all_subgroups)
        ntasks = nafterburners * nsubgroups

        # TODO: for 64//3, this creates 4 blocks, with rank 63/64 being block 3 block_rank==0,
        # which ends up running mgii afterburner twice
        nblocks, block_size, block_rank, block_num = \
            distribute_ranks_to_blocks(ntasks, rank=rank, size=size, log=log)

        #- Create a subcommunicator with just this rank, e.g. for
        #- qsoqn afterburner that needs a communicator to pass to
        #- redrock, but is otherwise only a single rank.
        if comm is not None:
            monocomm = comm.Split(color=comm.rank)
        else:
            monocomm = None

        if block_rank == 0:
            ## If running mutiple afterburners at once, wait some time so
            ## I/O isn't hit all at once
            ## afterburner 2 runs with 10s delay, 3 with 20s delay
            time.sleep(0.2*block_num)
            for i in range(block_num, ntasks, nblocks):
                result, success = 0, True
                ## If running mutiple afterburners at once, wait some time so
                ## I/O isn't hit all at once
                ## afterburner 2 runs with 10s delay, 3 with 20s delay
                #time.sleep(0.2*i)
                subgroup = all_subgroups[i % nsubgroups]
                if groupname == 'healpix':
                    findfileopts['healpix'] = subgroup
                else:
                    findfileopts['spectrograph'] = subgroup

                coaddfile = findfile('coadd', **findfileopts)
                rrfile = findfile('redrock', **findfileopts)
                ## First set of nsubgroups ranks go to desi_qso_mgii_afterburner
                if i // nsubgroups == 0:
                    log.info(f"rank {rank}, block_rank {block_rank}, block_num {block_num}, is running spectro/healpix {subgroup} for qso mgii")
                    mgiifile = findfile('qso_mgii', **findfileopts)
                    mgiilog = findfile('qso_mgii', logfile=True, **findfileopts)
                    cmd = f"desi_qso_mgii_afterburner --coadd {coaddfile} " \
                          + f"--redrock {rrfile} --output {mgiifile} " \
                          + f"--target_selection all --save_target all"
                    cmdargs = cmd.split()[1:]
                    if args.dryrun:
                        if rank == 0:
                            log.info(f"dryrun: Would have run {cmd}")
                    else:
                        with stdouterr_redirected(mgiilog):
                            result, success = runcmd(qsomgii.main, args=cmdargs,
                                                     inputs=[coaddfile, rrfile],
                                                     outputs=[mgiifile])

                        if not success:
                            log.error(f'qsomgii afterburner petal/healpix {subgroup} failed; see {mgiilog}')

                ## Second set of nsubgroups ranks go to desi_qso_qn_afterburner
                elif i // nsubgroups == 1:
                    log.info(f"rank {rank}, block_rank {block_rank}, block_num {block_num}, is running spectro/healpix {subgroup} for qso qn")
                    qnfile = findfile('qso_qn', **findfileopts)
                    qnlog = findfile('qso_qn', logfile=True, **findfileopts)
                    cmd = f"desi_qso_qn_afterburner --coadd {coaddfile} " \
                          + f"--redrock {rrfile} --output {qnfile} " \
                          + f"--target_selection all --save_target all"
                    cmdargs = cmd.split()[1:]
                    if args.dryrun:
                        if rank == 0:
                            log.info(f"dryrun: Would have run {cmd}")
                    else:
                        with stdouterr_redirected(qnlog):
                            result, success = runcmd(qsoqn.main, args=cmdargs,
                                                     inputs=[coaddfile, rrfile],
                                                     outputs=[qnfile], comm=monocomm)

                        if not success:
                            log.error(f'qsoqn afterburner petal/healpix {subgroup} failed; see {qnlog}')

                ## Third set of nsubgroups ranks go to desi_emlinefit_afterburner
                elif i // nsubgroups == 2:
                    log.info(f"rank {rank}, block_rank {block_rank}, block_num {block_num}, is running spectro/healpix {subgroup} for emlinefit")
                    emfile = findfile('emline', **findfileopts)
                    emlog = findfile('emline', logfile=True, **findfileopts)
                    cmd = f"desi_emlinefit_afterburner --coadd {coaddfile} " \
                          + f"--redrock {rrfile} --output {emfile}"
                    cmdargs = cmd.split()[1:]
                    if args.dryrun:
                        if rank == 0:
                            log.info(f"dryrun: Would have run {cmd}")
                    else:
                        with stdouterr_redirected(emlog):
                            result, success = runcmd(emlinefit.main, args=cmdargs,
                                                     inputs=[coaddfile, rrfile],
                                                     outputs=[emfile])

                        if not success:
                            log.error(f'emlinefit afterburner petal/healpix {subgroup} failed; see {emlog}')

                ## For now only 3 afterburners, so shout if we loop goes higher than that
                else:
                    log.error(f"Index i={i} // nsubgroups={nsubgroups} should " \
                              + f"be between 0 and {nafterburners-1}!")

                if not success:
                    error_count += 1

        if rank == 0:
            log.info("Done with afterburners")

        timer.stop('afterburners')

    if comm is not None:
        comm.barrier()

    #-------------------------------------------------------------------------
    ## Collect error count and wrap up

    if comm is not None:
        all_error_counts = comm.gather(error_count, root=0)
        if rank == 0:
            final_error_count = int(np.sum(all_error_counts))
        else:
            final_error_count = 0
        error_count = comm.bcast(final_error_count, root=0)

    #- save / print timing information
    log_timer(timer, args.timingfile, comm=comm)

    if rank == 0:
        duration_seconds = time.time() - start_time
        mm = int(duration_seconds) // 60
        ss = int(duration_seconds - mm*60)
        goodbye = f'All done at {time.asctime()}; duration {mm}m{ss}s'

        if error_count > 0:
            log.error(f'{error_count} processing errors; see logs above')
            log.error(goodbye)
        else:
            log.info(goodbye)

    if error_count > 0:
        sys.exit(int(error_count))
    else:
        return 0


def distribute_ranks_to_blocks(nblocks, rank=None, size=None, comm=None,
                               log=None, split_comm=False):
    """
    Function to split up a set of ranks of size 'size' into nblock number
    of blocks or roughly equal size.

    Args:
        nblocks (int): the number of blocks to split the ranks into
        rank (int): the MPI world rank
        size (int): the number of world MPI ranks
        comm (object): MPI communicator
        log (object): logger
        split_comm (bool): whether to split the world communicator into blocks and return
            the block communicator

    Returns:
        tuple: A tuple containing:

        * nblocks, int: the achievable number of block based on size
        * block_size, int: the number of ranks in the assigned block of current rank
        * block_rank. int: the rank in the assigned block of the current rank
        * block_num, int: the block number (of nblocks blocks) in which the rank
          was assigned
        * block_comm (optional): if split_comm is true, returns a communicator of
          only the ranks in the current block. Splits from
          the world communicator
    """
    if rank is None or size is None:
        if comm is not None:
            # - Use the provided comm to determine rank and size
            rank = comm.rank
            size = comm.size
        else:
            msg = 'Either rank and size or comm must be defined. '
            msg += f'Received rank={rank}, size={size}, comm={comm}'
            if log is None:
                log = get_logger()
            log.error(msg)
            raise ValueError(msg)

    if log is not None and rank == 0:
        log.info(f"Attempting to split MPI ranks of size {size} into " +
                 f"{nblocks} blocks")

    if size <= nblocks:
        nblocks = size
        block_size = 1
        block_rank = 0
        block_num = rank
        if split_comm:
            block_comm = comm
    else:
        # nblocks = nblocks
        block_num = int(rank / (size/nblocks))
        block_rank = int(rank % (size/nblocks))

        # Calculate assignment for all ranks to be able to calculate
        # how many other ranks are in this same block
        all_ranks = np.arange(size)
        all_block_num = (all_ranks / (size/nblocks)).astype(int)
        assert all_block_num[rank] == block_num
        ii_this_block = all_block_num == block_num
        block_size = np.sum(ii_this_block)

        if split_comm:
            if comm is not None:
                block_comm = comm.Split(block_num, block_rank)
                assert block_rank == block_comm.Get_rank()
            else:
                block_comm = comm

    if log is not None:
        log.info(f"World rank/size: {rank}/{size} mapped to: Block #{block_num}, " +
                 f"block_rank/block_size: {block_rank}/{block_size}")

    if split_comm:
        return nblocks, block_size, block_rank, block_num, block_comm
    else:
        return nblocks, block_size, block_rank, block_num
