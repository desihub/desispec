"""
Script for science processing of a given DESI tile and night
"""

import time
start_imports = time.time()

import sys, os, argparse, re
import subprocess
from copy import deepcopy
import json
import pandas as pd
import numpy as np
import fitsio
from astropy.io import fits

from astropy.table import Table,vstack

from desitarget.targetmask import desi_mask

from desiutil.log import get_logger, DEBUG, INFO
import desiutil.iers

import desispec.io
from desispec.io.util import decode_camword, create_camword, camword_union
import desispec.scripts.proc as proc
import desispec.scripts.proc_joint_fit as proc_joint_fit

from desispec.workflow.desi_proc_funcs import assign_mpi, get_desi_proc_tilenight_parser, update_args_with_headers, \
    find_most_recent
from desispec.workflow.desi_proc_funcs import determine_resources

stop_imports = time.time()

#########################################
# TEMPORARY CONVENIENCE FUNCTIONS

def difference_camwords(fullcamword, badcamword):
    '''Borrowed from desispec to remove noisy log message

    See desispec.io.util.difference_camwords
    '''
    full_cameras = decode_camword(fullcamword)
    bad_cameras = decode_camword(badcamword)
    for cam in bad_cameras:
        if cam in full_cameras:
            full_cameras.remove(cam)
        # else:
        #     log.info(f"Can't remove {cam}: not in the fullcamword. fullcamword={fullcamword}, badcamword={badcamword}")
    return create_camword(full_cameras)

# END TEMPORARY CONVENIENCE FUNCTIONS
#########################################

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
    error_count = 0

    if comm is not None:
        #- Use the provided comm to determine rank and size
        rank = comm.rank
        size = comm.size
    else:
        #- Check MPI flags and determine the comm, rank, and size given the arguments
        comm, rank, size = assign_mpi(do_mpi=args.mpi, do_batch=args.batch, log=log)

    #- What are we going to do?
    if rank == 0:
        log.info('----------')
        log.info('Tile {} night {}'.format(args.tileid, args.night))
        log.info('Output root {}'.format(desispec.io.specprod_root()))
        log.info('----------')
        
    #- Determine expids and cameras for a tile night
    reduxdir = desispec.io.specprod_root()
    exptable_file = f'{reduxdir}/exposure_tables/{str(args.night)[0:6]}/exposure_table_{args.night}.csv'
    log.info(f'Reading exptable in {exptable_file}')

    exptable = pd.read_csv(exptable_file)

    keep  = exptable['OBSTYPE'] == 'science'
    keep &= exptable['TILEID'].isin([int(args.tileid)])
    exptable = exptable[keep]

    expids = list(exptable['EXPID'])
    cameras = dict()
    for i in range(len(expids)):
        cameras[expids[i]] = difference_camwords(
          exptable.iloc[i]['CAMWORD'],exptable.iloc[i]['BADCAMWORD']
        )
    cameras_union = camword_union(list(cameras.values()), full_spectros_only=True) 

    #-------------------------------------------------------------------------
    #- Proceeding with running
    
    #- common arguments
    common_args = f'--traceshift --night {args.night}'

    #- gpu options
    gpu_args=''
    if args.gpuspecter:
        gpu_args += ' --gpu_specter'
    if args.gpuextract:
        gpu_args += ' --gpuextract'

    #- run desiproc prestdstar over exps
    for expid in expids:
        prestdstar_args = common_args + gpu_args
        prestdstar_args += f' --nostdstarfit --nofluxcalib --expid {expid} --cameras {cameras[expid]}'
        prestdstar_args = proc.parse(prestdstar_args.split())
        error_count += proc.main(prestdstar_args,comm)

    #- run joint stdstar fit using all exp for this tile night
    stdstar_args  = common_args
    stdstar_args += f' --obstype science --mpistdstars --expids {",".join(map(str, expids))} --cameras {cameras_union}'
    stdstar_args = proc_joint_fit.parse(stdstar_args.split())
    error_count += proc_joint_fit.main(stdstar_args, comm)

    #- run desiproc poststdstar over exps
    for expid in expids:
        poststdstar_args  = common_args
        poststdstar_args += f' --nostdstarfit --noprestdstarfit --expid {expid} --cameras {cameras[expid]}'
        poststdstar_args = proc.parse(poststdstar_args.split())
        error_count += proc.main(poststdstar_args, comm)

    #-------------------------------------------------------------------------
    #- Collect error count
    if comm is not None:
        all_error_counts = comm.gather(error_count, root=0)
        error_count = int(comm.bcast(np.sum(all_error_counts), root=0))

    if rank == 0 and error_count > 0:
        log.error(f'{error_count} processing errors; see logs above')

    #-------------------------------------------------------------------------
    #- Done

    return error_count
