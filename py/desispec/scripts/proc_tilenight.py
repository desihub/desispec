"""
Script for science processing of a given DESI tile and night
"""
import time, datetime
import numpy as np

from os.path import dirname, abspath
from desiutil.log import get_logger, DEBUG, INFO

from desispec.io import specprod_root
from desispec.workflow.tableio import load_table
from desispec.io.util import decode_camword, create_camword, camword_union

import desispec.scripts.proc as proc
import desispec.scripts.proc_joint_fit as proc_joint_fit

from desispec.workflow.desi_proc_funcs import assign_mpi, get_desi_proc_tilenight_parser
from desispec.workflow.desi_proc_funcs import update_args_with_headers

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

    #- What are we going to do?
    if rank == 0:
        log.info('----------')
        log.info('Tile {} night {}'.format(args.tileid, args.night))
        log.info('Output root {}'.format(specprod_root()))
        log.info('----------')

    #- Determine expids and cameras for a tile night
    reduxdir = specprod_root()
    exptable_file = f'{reduxdir}/exposure_tables/{str(args.night)[0:6]}/exposure_table_{args.night}.csv'
    log.info(f'Reading exptable in {exptable_file}')

    exptable = load_table(exptable_file)

    keep  = exptable['OBSTYPE'] == 'science'
    keep &= exptable['TILEID']  == int(args.tileid)
    exptable = exptable[keep]

    expids = list(exptable['EXPID'])
    prestdstar_expids = []
    stdstar_expids = []
    poststdstar_expids = []
    cameras = dict()
    for i in range(len(expids)):
        expid=expids[i]
        camword=exptable['CAMWORD'][i]
        badcamword=exptable['BADCAMWORD'][i]
        if isinstance(badcamword, str):
            full_cameras = decode_camword(camword)
            bad_cameras = decode_camword(badcamword)
            for cam in bad_cameras:
                if cam in full_cameras:
                    full_cameras.remove(cam)
            cameras[expids[i]] = create_camword(full_cameras)
        else:
            cameras[expids[i]] = camword
        laststep = exptable['LASTSTEP'][i]
        if laststep == 'all' or laststep == 'skysub':
            prestdstar_expids.append(expid)
        if laststep == 'all':
            stdstar_expids.append(expid)
            poststdstar_expids.append(expid)
    cameras_union = camword_union(list(cameras.values()), full_spectros_only=True) 

    #-------------------------------------------------------------------------
    #- Proceeding with running
    
    #- common arguments
    common_args = f'--traceshift --night {args.night}'

    #- gpu options
    gpu_args=''
    if args.gpuspecter:
        gpu_args += ' --gpuspecter'
    if args.gpuextract:
        gpu_args += ' --gpuextract'

    #- mpi options
    mpi_args=''
    if args.mpistdstars and args.mpi:
        mpi_args += ' --mpistdstars'

    #- run desiproc prestdstar over exps
    for expid in prestdstar_expids:
        prestdstar_args = common_args + gpu_args
        prestdstar_args += f' --nostdstarfit --nofluxcalib --expid {expid} --cameras {cameras[expid]}'
        if rank==0:
            log.info(f'running desi_proc {prestdstar_args}')
        prestdstar_args = proc.parse(prestdstar_args.split())
        if not args.dryrun:
            error_count += proc.main(prestdstar_args,comm)

    #- run joint stdstar fit using all exp for this tile night
    stdstar_args  = common_args + mpi_args
    stdstar_args += f' --obstype science --mpistdstars --expids {",".join(map(str, stdstar_expids))} --cameras {cameras_union}'
    if rank==0:
        log.info(f'running desi_proc_joint_fit {stdstar_args}')
    stdstar_args = proc_joint_fit.parse(stdstar_args.split())
    if not args.dryrun:
        error_count += proc_joint_fit.main(stdstar_args, comm)   

    #- run desiproc poststdstar over exps
    for expid in poststdstar_expids:
        poststdstar_args  = common_args
        poststdstar_args += f' --nostdstarfit --noprestdstarfit --expid {expid} --cameras {cameras[expid]}'
        if rank==0:
            log.info(f'running desi_proc {poststdstar_args}')
        poststdstar_args = proc.parse(poststdstar_args.split())
        if not args.dryrun:
            error_count += proc.main(poststdstar_args, comm)

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
