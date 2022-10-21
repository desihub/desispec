"""
One stop shopping for redshifting  DESI spectra

"""

import time, datetime

from desispec.io.meta import get_nights_up_to_date
from desispec.workflow.redshifts import read_minimal_exptables_columns, \
    create_desi_zproc_batch_script

start_imports = time.time()

import sys, os, argparse, re
import subprocess
from copy import deepcopy
import json

import numpy as np
import fitsio
from astropy.io import fits

from astropy.table import Table,vstack

import glob

from redrock.external import desi

import desiutil.timer
import desispec.io
from desispec.io import findfile, specprod_root, replace_prefix, shorten_filename, get_readonly_filepath
from desispec.io.util import create_camword, decode_camword, parse_cameras, \
    camword_to_spectros, columns_to_goodcamword, difference_camwords
from desispec.io.util import validate_badamps, get_tempfilename
from desispec.util import runcmd
from desispec.scripts import group_spectra
from desiutil.log import get_logger, DEBUG, INFO
import desiutil.iers
from desispec.parallel import stdouterr_redirected
from desispec.workflow import batch
from desispec.workflow.exptable import get_exposure_table_pathname
from desispec.workflow.desi_proc_funcs import assign_mpi, update_args_with_headers, _log_timer
from desispec.workflow.desi_proc_funcs import determine_resources, create_desi_proc_batch_script

stop_imports = time.time()

#########################################
######## Begin Body of the Code #########
#########################################

def parse(options=None):
    """
    Create an argparser object for use with desi_proc AND desi_proc_joint_fit based on arguments from sys.argv
    """
    # TODO - finalize the arguments we need
    parser = argparse.ArgumentParser(usage="{prog} [options]")

    parser.add_argument("-t", "--tileid", type=str, default=None, help="Tile ID")
    # parser.add_argument("-p", "--healpix", type=str, default=None, help="Healpix")
    parser.add_argument("-n", "--nights", type=int, nargs='*', default=None,
                        help="YEARMMDD night")
    parser.add_argument("-e", "--expids", type=int, nargs='*', default=None,
                        help="Exposure IDs")
    parser.add_argument("--thrunight", type=int, default=None,
                        help="Last night to include (YEARMMDD night) for "
                             + "cumulative redshift jobs. Used instead of nights.")
    parser.add_argument("-g", "--groupname", type=str, default='cumulative',
                        help="Redshift grouping type: cumulative, perexp, pernight")
    parser.add_argument("--cameras", type=str, default="a0123456789",
                        help="Explicitly define the spectrographs for which you want" +
                             " to reduce the data. Should be a comma separated list." +
                             " Numbers only assumes you want to reduce R, B, and Z " +
                             "for that spectrograph. Otherwise specify separately [BRZ|brz][0-9].")
    parser.add_argument("--mpi", action="store_true",
                        help="Use MPI parallelism")
    parser.add_argument("--nogpu", action="store_true",
                        help="Don't use gpus")
    parser.add_argument("--max-gpuprocs", type=int, default=4,
                        help="Number of GPU prcocesses per node")
    parser.add_argument("--run-zmtl", action="store_true",
                        help="Whether to run zmtl or not")
    parser.add_argument("--noafterburners", action="store_true",
                        help="Set if you don't want to run afterburners")
    parser.add_argument("--batch", action="store_true",
                        help="Submit a batch job to process this exposure")
    parser.add_argument("--nosubmit", action="store_true",
                        help="Create batch script but don't submit")
    parser.add_argument("-q", "--queue", type=str, default="realtime",
                        help="batch queue to use")
    parser.add_argument("--batch-opts", type=str, default=None,
                        help="additional batch commands")
    parser.add_argument("--batch-reservation", type=str,
                   help="batch reservation name")
    parser.add_argument("--batch-dependency", type=str,
                   help="job dependencies passed to sbatch --dependency")
    parser.add_argument("--runtime", type=int, default=None,
                        help="batch runtime in minutes")
    parser.add_argument("--starttime", type=str,
                        help='start time; use "--starttime `date +%%s`"')
    parser.add_argument("--timingfile", type=str,
                        help='save runtime info to this json file; augment if pre-existing')
    parser.add_argument("--system-name", type=str, default=batch.default_system(),
                        help='Batch system name (cori-haswell, perlmutter-gpu, ...)')
    parser.add_argument("-d", "--dryrun", action="store_true",
                        help="show commands only, do not run")
    if options is not None:
        args = parser.parse_args(options)
    else:
        args = parser.parse_args()

    return args


def main(args=None, comm=None):
    if not isinstance(args, argparse.Namespace):
        args = parse(options=args)

    log = get_logger()

    # if args.healpix is not None:
    #     msg = f"Healpix redshifts is not yet implemented"
    #     log.error(msg)
    #     raise NotImplementedError(msg)
    if args.tileid is None:
        msg = "Must specify tile"
        log.error(msg)
        raise ValueError(msg)

    today = int(time.strftime('%Y%m%d'))
    if args.thrunight is not None:
        ## change if implementing healpix
        if args.groupname not in ['cumulative']:
            msg = f"--thrunight only valid for cumulative redshifts."
            log.error(msg)
            raise ValueError(msg)
        elif args.thrunight < 20200214 or args.thrunight > today:
            msg = f"--thrunight must be between 20200214 and today."
            log.error(msg)
            raise ValueError(msg)

    if args.expids is not None:
        if args.nights is None:
            msg = f"Must specify --nights if specifying --expids."
            log.error(msg)
            raise ValueError(msg)
        else:
            msg = f"Only using exposures specified with --expids"
            log.info(msg)

    if args.groupname in ['perexp', 'pernight'] and args.nights is not None:
        if len(args.nights) > 1:
            msg = f"Only expect one night for groupname {args.groupname}" \
                  + f" but received nights={args.nights}."
            log.error(msg)
            raise ValueError(msg)

    start_time = time.time()
    error_count = 0

    start_mpi_connect = time.time()
    if comm is not None:
        ## Use the provided comm to determine rank and size
        rank = comm.rank
        size = comm.size
    else:
        ## Check MPI flags and determine the comm, rank, and size given the arguments
        comm, rank, size = assign_mpi(do_mpi=args.mpi, do_batch=args.batch, log=log)
    stop_mpi_connect = time.time()

    if rank == 0:
        thisfile=os.path.dirname(os.path.abspath(__file__))
        thistime=datetime.datetime.fromtimestamp(start_imports).isoformat()
        log.info(f'rank 0 started {thisfile} at {thistime}')
    ## Start timer; only print log messages from rank 0 (others are silent)
    timer = desiutil.timer.Timer(silent=(rank>0))

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
    if comm is not None:
        args = comm.bcast(args, root=0)

    ## Derive the available cameras
    if isinstance(args.cameras, str):
        camword = parse_cameras(args.cameras)
        cameras = decode_camword(camword)
    else:
        camword = create_camword(args.cameras)
        cameras = args.cameras

    ## Unpack arguments for shorter names
    tileid, groupname = args.tileid, args.groupname

    known_groups = ['cumulative', 'pernight', 'perexp']
    if groupname not in known_groups:
        raise ValueError('obstype {} not in {}'.format(groupname, known_groups))

    exposure_table = Table()
    err = 0
    if rank == 0:
        if groupname == 'perexp' and args.nights is not None \
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

            exposure_table = read_minimal_exptables_columns(nights=nights,
                                                            tileids=[tileid])
            if args.expids is not None:
                exposure_table = exposure_table[np.isin(exposure_table['EXPID'],
                                                        args.expids)]
            exposure_table.sort(keys=['EXPID'])

        #-------------------------------------------------------------------------
        ## Create and submit a batch job if requested

        if args.batch:
            ## create the batch script
            expids = np.unique(exposure_table['EXPID'])
            nights = np.unique(exposure_table['NIGHT'])
            scriptfile = create_desi_zproc_batch_script(tileid=tileid,
                                                        # healpix=healpix,
                                                        nights=nights,
                                                        expids=expids,
                                                        cameras=camword,
                                                        jobdesc=groupname,
                                                        queue=args.queue,
                                                        runtime=args.runtime,
                                                        batch_opts=args.batch_opts,
                                                        timingfile=args.timingfile,
                                                        system_name=args.system_name,
                                                        nogpu=args.nogpu,
                                                        max_gpuprocs=args.max_gpuprocs,
                                                        run_zmtl=args.run_zmtl,
                                                        noafterburners=args.noafterburners)

            log.info("Generating batch script and exiting.")

            if not args.nosubmit and not args.dryrun:
                err = subprocess.call(['sbatch', scriptfile])

    ## All ranks need to exit if submitted batch
    if args.batch:
        if comm is not None:
            err = comm.bcast(err, root=0)
        sys.exit(err)

    ## Should remove, just nice for printouts while performance isn't important
    if comm is not None:
        comm.barrier()
        exposure_table = comm.bcast(exposure_table, root=0)

    ## Get night and expid information
    expids = np.unique(exposure_table['EXPID'])
    nights = np.unique(exposure_table['NIGHT'])
    thrunight = np.max(nights)

    #------------------------------------------------------------------------#
    #------------------------ Proceed with running --------------------------#
    #------------------------------------------------------------------------#

    ## Print a summary of what we're going to do
    if rank == 0:
        log.info('------------------------------')
        log.info('Groupname {}'.format(groupname))
        # if healpix is not None:
        #     log.info(f'Healpix {healpix} night {thrunight} expids {args.expids}')
        # else:
        #     log.info(f'Tileid {tileid} night {thrunight} expids {args.expids}')
        log.info(f'Tileid={tileid} nights={nights} expids={expids}')

        log.info(f'Supplied camword: {camword}')
        #log.info(f'All Spectros:  {all_spectros}')
        log.info('Output root {}'.format(desispec.io.specprod_root()))
        if args.run_zmtl:
            log.info(f'Will be running zmtl')
        if not args.noafterburners:
            log.info(f'Will be running aferburners')
        log.info('------------------------------')

    if comm is not None:
        comm.barrier()

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


    ## Derive the available spectrographs
    all_spectros = camword_to_spectros(create_camword(list(complete_cam_set)),
                                       full_spectros_only=False)
    # full_spectros = camword_to_spectros(create_camword(list(complete_cam_set)),
    #                                            full_spectros_only=True)

    if len(expnight_dict) == 0:
        msg = f"Didn't find any exposures!"
        log.error(msg)
        raise ValueError(msg)
    elif len(all_spectros) == 0:
        msg = f"Didn't find any spectrographs!"
        log.error(msg)
        raise ValueError(msg)

    timer.stop('preflight')

    #-------------------------------------------------------------------------
    ## Do spectral grouping and coadding
    timer.start('groupspec')

    nblocks, block_size, block_rank, block_num = \
        distribute_ranks_to_blocks(len(all_spectros), rank=rank, size=size, log=log)

    if comm is not None:
        comm.barrier()

    if block_rank == 0:
        for i in range(block_num, len(all_spectros), nblocks):
            result, success = 0, True
            spectro = all_spectros[i]
            spectrafile = findfile('spectra', night=thrunight, tile=tileid,
                                   groupname=groupname, spectrograph=spectro)
            splog = findfile('spectra', night=thrunight, tile=tileid,
                             groupname=groupname, spectrograph=spectro,
                             log=True)
            coaddfile = findfile('coadd', night=thrunight, tile=tileid,
                                 groupname=groupname, spectrograph=spectro)
            # generate list of cframes from dict of exposures, nights, and cameras
            cframes = []
            for (expid, night), cameras in expnight_dict.items():
                for camera in cameras:
                    if int(spectro) == int(camera[1]):
                        cframes.append(findfile('cframe', night=night,
                                                expid=expid, camera=camera))

            cmd = f"desi_group_spectra --inframes {' '.join(cframes)} " \
                  + f"--outfile {spectrafile} " \
                  + f"--coaddfile {coaddfile} " \
                  + f"--header SPGRP={groupname} SPGRPVAL={thrunight} " \
                  + f"NIGHT={thrunight} TILEID={tileid} SPECTRO={spectro} " \
                  + f"PETAL={spectro}"
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
                error_count += 1

    if rank == 0:
        log.info("Done with spectra")
    timer.stop('groupspec')

    if comm is not None:
        comm.barrier()

    #-------------------------------------------------------------------------
    ## Do redshifting
    timer.start('redrock')
    for spectro in all_spectros:
        result, success = 0, True

        coaddfile = findfile('coadd', night=thrunight, tile=tileid,
                             groupname=groupname, spectrograph=spectro)
        rrfile = findfile('redrock', night=thrunight, tile=tileid,
                          groupname=groupname, spectrograph=spectro)
        rdfile = findfile('rrdetails', night=thrunight, tile=tileid,
                          groupname=groupname, spectrograph=spectro)
        rrlog = findfile('redrock', night=thrunight, tile=tileid,
                         groupname=groupname, spectrograph=spectro,
                         log=True)
        redrock_cmd = "rrdesi_mpi"

        cmd = f"rrdesi_mpi -i {coaddfile} -o {rrfile} -d {rdfile}"
        if not args.nogpu:
            cmd += f' --gpu --max-gpuprocs {args.max_gpuprocs}'

        cmdargs = cmd.split()[1:]
        if args.dryrun:
            if rank == 0:
                log.info(f"dryrun: Would have run {cmd}")
        else:
            with stdouterr_redirected(rrlog, comm=comm):
                result, success = runcmd(desi.rrdesi, comm=comm, args=cmdargs,
                                         inputs=[coaddfile], outputs=[rrfile, rdfile])

        ## Since all ranks running redrock, only count failure/success once
        if rank == 0 and not success:
            error_count += 1
        ## Since all ranks running redrock, ensure we're all moving on to next
        ## iteration together
        if comm is not None:
            comm.barrier()

    if rank == 0:
        log.info("Done with redrock")

    timer.stop('redrock')

    #-------------------------------------------------------------------------
    ## Do tileqa if a tile
    if groupname in ['pernight', 'cumulative']:
        from desispec.scripts import tileqa

        timer.start('tileqa')
        if rank == 0:
            result, success = 0, True
            qafile = findfile('tileqa', night=thrunight,
                              tile=tileid, groupname=groupname)
            qapng = findfile('tileqapng', night=thrunight, tile=tileid,
                             groupname=groupname)
            qalog = findfile('tileqa', night=thrunight, tile=tileid,
                             groupname=groupname, log=True)
            cmd = f"desi_tile_qa -g {groupname} -n {thrunight} -t {tileid}"
            cmdargs = cmd.split()[1:]
            if args.dryrun:
                if rank == 0:
                    log.info(f"dryrun: Would have run {cmd} with"
                             + f"outputs {qafile}, {qapng}")
            else:
                with stdouterr_redirected(qalog, comm=comm):
                    result, success = runcmd(tileqa.main, comm=comm, args=cmdargs,
                                             inputs=[], outputs=[qafile, qapng])

            ## Since all ranks running redrock, only count failure/success once
            if not success:
                error_count += 1
            ## Since all ranks running redrock, ensure we're all moving on to next
            ## iteration together

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
            for i in range(block_num, len(all_spectros), nblocks):
                result, success = 0, True
                spectro = all_spectros[i]
                # run_mock_func('spectra', 'cumulative', spectro=spectro, comm=comm)
                rrfile = findfile('redrock', night=thrunight, tile=tileid,
                                       groupname=groupname, spectrograph=spectro)
                zmtlfile = findfile('zmtl', night=thrunight, tile=tileid,
                                    groupname=groupname, spectrograph=spectro)
                zmtllog = findfile('zmtl', night=thrunight, tile=tileid,
                                   groupname=groupname, spectrograph=spectro,
                                   log=True)
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
                    error_count += 1

        if rank == 0:
            log.info("Done with zmtl")

        timer.stop('zmtl')

        if comm is not None:
            comm.barrier()

    #-------------------------------------------------------------------------
    ## Do afterburners if asked to
    if not args.noafterburners:
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
        nspectros = len(all_spectros)
        ntasks = nafterburners * nspectros

        nblocks, block_size, block_rank, block_num = \
            distribute_ranks_to_blocks(ntasks, rank=rank, size=size, log=log)

        if block_rank == 0:
            for i in range(block_num, ntasks, nblocks):
                result, success = 0, True
                ## If running mutiple afterburners at once, wait some time so
                ## I/O isn't hit all at once
                ## afterburner 2 runs with 10s delay, 3 with 20s delay
                if size > len(all_spectros):
                    time.sleep(10*(i % nspectros))
                spectro = all_spectros[i % nspectros]
                coaddfile = findfile('coadd', night=thrunight, tile=tileid,
                                     groupname=groupname, spectrograph=spectro)
                rrfile = findfile('redrock', night=thrunight, tile=tileid,
                                       groupname=groupname, spectrograph=spectro)
                ## First set of nspectros ranks go to desi_qso_mgii_afterburner
                if i // nspectros == 0:
                    mgiifile = findfile('qso_mgii', night=thrunight, tile=tileid,
                                        groupname=groupname,
                                        spectrograph=spectro)
                    mgiilog = findfile('qso_mgii', night=thrunight, tile=tileid,
                                       groupname=groupname, spectrograph=spectro,
                                       log=True)
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
                ## Second set of nspectros ranks go to desi_qso_qn_afterburner
                elif i // nspectros == 1:
                    qnfile = findfile('qso_qn', night=thrunight, tile=tileid,
                                      groupname=groupname,
                                      spectrograph=spectro)
                    qnlog = findfile('qso_qn', night=thrunight, tile=tileid,
                                     groupname=groupname, spectrograph=spectro,
                                     log=True)
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
                                                     outputs=[qnfile])
                ## Third set of nspectros ranks go to desi_emlinefit_afterburner
                elif i // nspectros == 2:
                    emfile = findfile('emline', night=thrunight, tile=tileid,
                                      groupname=groupname,
                                      spectrograph=spectro)
                    emlog = findfile('emline', night=thrunight, tile=tileid,
                                     groupname=groupname, spectrograph=spectro,
                                     log=True)
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
                ## For now only 3 afterburners, so shout if we loop goes higher than that
                else:
                    log.error(f"Index i={i} // nspectros={nspectros} should " \
                              + f"be between 0 and {nafterburners-1}!")

                if not success:
                    error_count += 1

        if rank == 0:
            log.info("Done with afterburners")

        timer.stop('afterburners')

        if comm is not None:
            comm.barrier()

    #-------------------------------------------------------------------------
    ## Collect error count
    if comm is not None:
        all_error_counts = comm.gather(error_count, root=0)
        error_count = int(comm.bcast(np.sum(all_error_counts), root=0))

    if rank == 0 and error_count > 0:
        log.error(f'{error_count} processing errors; see logs above')

    #-------------------------------------------------------------------------
    ## Wrap up

    _log_timer(timer, args.timingfile, comm=comm)
    if rank == 0:
        duration_seconds = time.time() - start_time
        mm = int(duration_seconds) // 60
        ss = int(duration_seconds - mm*60)

        log.info('All done at {}; duration {}m{}s'.format(
            time.asctime(), mm, ss))

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
        nblocks, int: the number of blocks to split the ranks into
        rank, int: the MPI world rank
        size, int: the number of world MPI ranks
        comm: MPI communicator
        log: logger
        split_comm, bool: whether to split the world communicator into blocks and return
                          the block communicator

    Returns:
        nblocks, int: the achievable number of block based on size
        block_size, int: the number of ranks in the assigned block of current rank
        block_rank. int: the rank in the assigned block of the current rank
        block_num, int: the block number (of nblocks blocks) in which the rank
                        was assigned
        block_comm (optional): if split_comm is true, returns a communicator of
                               only the ranks in the current block. Splits from
                               the world communicator
    """
    if rank is None or size in None:
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
        block_comm = comm
    elif comm is not None:
        # nblocks = nblocks
        block_size = int(size / nblocks)
        block_num = int(rank / block_size)
        block_rank = int(rank % block_size)
        block_comm = comm.Split(block_num, block_rank)
        assert block_rank == block_comm.Get_rank()
    else:
        nblocks = block_size = 1
        block_rank = block_num = 0
        block_comm = comm

    if log is not None:
        log.info(f"World rank/size: {rank}/{size} mapped to: Block #{block_num}, " +
                 f"block_rank/block_size: {block_rank}/{block_size}")

    if split_comm:
        return nblocks, block_size, block_rank, block_num, block_comm
    else:
        return nblocks, block_size, block_rank, block_num