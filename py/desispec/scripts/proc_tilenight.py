"""
Script for science processing of a given DESI tile and night
"""
import sys
import subprocess
import time, datetime
import numpy as np

from os.path import dirname, abspath
from desiutil.log import get_logger, DEBUG, INFO

from desispec.io import specprod_root
from desispec.workflow.exptable import get_exposure_table_pathname
from desispec.workflow.tableio import load_table
from desispec.io.util import decode_camword, create_camword, camword_union, difference_camwords

import desispec.scripts.proc as proc
import desispec.scripts.proc_joint_fit as proc_joint_fit

from desispec.workflow.desi_proc_funcs import assign_mpi, get_desi_proc_tilenight_parser
from desispec.workflow.desi_proc_funcs import update_args_with_headers, create_desi_proc_tilenight_batch_script

import desispec.gpu

#########################################
######## Begin Body of the Code #########
#########################################

def parse(options=None):
    parser = get_desi_proc_tilenight_parser()
    args = parser.parse_args(options)
    return args

def main(args=None, comm=None):
    if args is None:
        args = parse()

    log = get_logger()
    start_time = time.time()
    error_count = 0

    if comm is not None:
        #- Use the provided comm to determine rank and size
        rank = comm.rank
        size = comm.size
    else:
        #- Check MPI flags and determine the comm, rank, and size given the arguments
        comm, rank, size = assign_mpi(do_mpi=args.mpi, do_batch=args.batch, log=log)

    if rank == 0:
        thisfile=dirname(abspath(__file__))
        thistime=datetime.datetime.fromtimestamp(start_time).isoformat()
        log.info(f'Tilenight started main in {thisfile} at {thistime}')

    #- Read + broadcast exposure table for this night
    exptable = None
    if rank == 0:
        exptable_file = get_exposure_table_pathname(args.night)
        log.info(f'Reading exptable in {exptable_file}')
        exptable = load_table(exptable_file)

    if comm is not None:
        exptable = comm.bcast(exptable, root=0)

    #- Determine expids and cameras for a tile night
    keep  = exptable['OBSTYPE'] == 'science'
    keep &= exptable['TILEID']  == int(args.tileid)
    exptable = exptable[keep]

    expids = list(exptable['EXPID'])
    prestdstar_expids = []
    stdstar_expids = []
    poststdstar_expids = []
    camwords = dict()
    badamps = dict()
    for i in range(len(expids)):
        expid=expids[i]
        camword=exptable['CAMWORD'][i]
        badcamword=exptable['BADCAMWORD'][i]
        badamps[expid] = exptable['BADAMPS'][i]
        if isinstance(badcamword, str):
            camwords[expids[i]] = difference_camwords(camword,badcamword,suppress_logging=True)
        else:
            camwords[expids[i]] = camword
        laststep = exptable['LASTSTEP'][i]
        if laststep == 'all' or laststep == 'skysub':
            prestdstar_expids.append(expid)
        if laststep == 'all':
            stdstar_expids.append(expid)
            poststdstar_expids.append(expid)
    joint_camwords = camword_union(list(camwords.values()), full_spectros_only=True) 

    if len(prestdstar_expids) == 0:
        if rank == 0:
            log.critical(f'No good exposures found for tile {args.tileid} night {args.night}')
        sys.exit(1)

    #-------------------------------------------------------------------------
    #- Create and submit a batch job if requested

    if args.batch:
        ncameras = len(decode_camword(joint_camwords))
        scriptfile = create_desi_proc_tilenight_batch_script(night=args.night,
                                                   exp=expids,
                                                   ncameras=ncameras,
                                                   tileid=args.tileid,
                                                   queue=args.queue,
                                                   system_name=args.system_name,
                                                   mpistdstars=args.mpistdstars,
                                                   use_specter=args.use_specter,
                                                   no_gpu=args.no_gpu,
                                                   )
        err = 0
        if not args.nosubmit:
            err = subprocess.call(['sbatch', scriptfile])
        sys.exit(err)

    #-------------------------------------------------------------------------
    #- Proceeding with running

    #- What are we going to do?
    if rank == 0:
        log.info('----------')
        log.info('Tile {} night {} exposures {}'.format(
            args.tileid, args.night, prestdstar_expids))
        log.info('Output root {}'.format(specprod_root()))
        log.info('----------')
    
    #- common arguments
    common_args = f'--night {args.night}'
    if args.no_gpu:
        common_args += ' --no-gpu'

    #- extract options
    extract_args=''
    if args.use_specter:
        extract_args += ' --use-specter'

    #- mpi options
    mpi_args=''
    if args.mpi:
        mpi_args += ' --mpistdstars'

    #- run desiproc prestdstar over exps
    for expid in prestdstar_expids:
        prestdstar_args = common_args + extract_args
        prestdstar_args += f' --nostdstarfit --nofluxcalib --expid {expid} --cameras {camwords[expid]}'
        if len(badamps[expid]) > 0:
            prestdstar_args += f' --badamps {badamps[expid]}'
        if rank==0:
            log.info(f'running desi_proc {prestdstar_args}')
        prestdstar_args_parsed = proc.parse(prestdstar_args.split())
        if not args.dryrun:
            try:
                errcode = proc.main(prestdstar_args_parsed,comm)
                if errcode != 0:
                    error_count += 1
            except (BaseException, Exception) as e:
                import traceback
                lines = traceback.format_exception(*sys.exc_info())
                log.error(f"prestdstar step using desi_proc with args {prestdstar_args} raised an exception:")
                print("".join(lines))
                log.warning(f'continuing to next prestdstar exposure, if any')
                error_count += 1

    #- collect post prestdstar error count
    if comm is not None:
        all_error_counts = comm.gather(error_count, root=0)
        error_count_pre = int(comm.bcast(np.sum(all_error_counts), root=0))
    else:
        error_count_pre = error_count

    continue_tilenight=True
    if error_count_pre > 0:
        if rank == 0:
            log.info(f'prestdstar failed with {error_count_pre} errors, skipping rest of tilenight steps')
        continue_tilenight=False

    #- Cleanup GPU memory and rank assignments before continuing
    desispec.gpu.redistribute_gpu_ranks(comm)

    if continue_tilenight:
        #- run joint stdstar fit using all exp for this tile night
        stdstar_args  = common_args + mpi_args
        stdstar_args += f' --obstype science --expids {",".join(map(str, stdstar_expids))} --cameras {joint_camwords}'
        if rank==0:
            log.info(f'running desi_proc_joint_fit {stdstar_args}')
        stdstar_args_parsed = proc_joint_fit.parse(stdstar_args.split())
        if not args.dryrun:
            try:
                errcode = proc_joint_fit.main(stdstar_args_parsed, comm)
                if errcode != 0:
                    error_count += 1
            except (BaseException, Exception) as e:
                import traceback
                lines = traceback.format_exception(*sys.exc_info())
                log.error(f"joint fit step using desi_proc_joint_fit with args {stdstar_args} raised an exception:")
                print("".join(lines))
                error_count += 1

        #- collect post joint fit error count
        if comm is not None:
            all_error_counts = comm.gather(error_count, root=0)
            error_count_jnt = int(comm.bcast(np.sum(all_error_counts), root=0))
        else:
            error_count_jnt = error_count

        if error_count_jnt > 0:
            if rank == 0:
                log.info(f'joint fitting failed with {error_count_jnt} errors, skipping rest of tilenight steps')
            continue_tilenight=False

    #- Cleanup GPU memory and rank assignments before continuing
    desispec.gpu.redistribute_gpu_ranks(comm)

    if continue_tilenight:
        #- run desiproc poststdstar over exps
        for expid in poststdstar_expids:
            poststdstar_args  = common_args
            poststdstar_args += f' --nostdstarfit --noprestdstarfit --expid {expid} --cameras {camwords[expid]}'
            if len(badamps[expid]) > 0:
                poststdstar_args += f' --badamps {badamps[expid]}'
            if rank==0:
                log.info(f'running desi_proc {poststdstar_args}')
            poststdstar_args_parsed = proc.parse(poststdstar_args.split())
            if not args.dryrun:
                try:
                    errcode = proc.main(poststdstar_args_parsed, comm)
                    if errcode != 0:
                        error_count += 1
                except (BaseException, Exception) as e:
                    import traceback
                    lines = traceback.format_exception(*sys.exc_info())
                    log.error(f"joint fit step using desi_proc_joint_fit with args {poststdstar_args} raised an exception:")
                    print("".join(lines))
                    error_count += 1

    #-------------------------------------------------------------------------
    #- Collect error count
    if comm is not None:
        all_error_counts = comm.gather(error_count, root=0)
        error_count = int(comm.bcast(np.sum(all_error_counts), root=0))

    if rank == 0 and error_count > 0:
        log.error(f'{error_count} processing errors in tilenight; see logs above')

    #-------------------------------------------------------------------------
    #- Done

    if rank == 0:
        duration_seconds = time.time() - start_time
        mm = int(duration_seconds) // 60
        ss = int(duration_seconds - mm*60)

        log.info(f'Tilenight main in {thisfile} returned at {time.asctime()}; duration {mm}m{ss}s')

    return error_count
