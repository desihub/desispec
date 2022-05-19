"""
Script for science processing of a given DESI tile and night
"""

import time
start_imports = time.time()

import sys, os, argparse, re
import subprocess
from copy import deepcopy
import json

import numpy as np
import fitsio
from astropy.io import fits

from astropy.table import Table,vstack

from desitarget.targetmask import desi_mask

from desiutil.log import get_logger, DEBUG, INFO
import desiutil.iers

import desispec.scripts.proc as proc
import desispec.scripts.proc_joint_fit as proc_joint_fit

from desispec.workflow.desi_proc_funcs import assign_mpi, get_desi_proc_tnight_parser, update_args_with_headers, \
    find_most_recent
from desispec.workflow.desi_proc_funcs import determine_resources

stop_imports = time.time()

#########################################
######## Begin Body of the Code #########
#########################################

def parse(options=None):
    parser = get_desi_proc_tnight_parser()
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
        log.info('Input {}'.format(args.input))
        log.info('Tile {} night {}'.format(args.tileid, args.night))
        log.info('Cameras {}'.format(args.cameras))
        log.info('Output root {}'.format(desispec.io.specprod_root()))
        log.info('----------')
        
    #-------------------------------------------------------------------------
    #- Proceeding with running
    
    #- common arguments
    common_args = f'--traceshift --night {night} --cameras {cameras}'

    #- get expids (wip)

    #- run desiproc prestdstar over exps
    for expid in expids:
        prestdstar_args = common_args
        prestdstar_args += f' --nostdstarfit --nofluxcalib --expid {expids[0]}'
        if args.gpu_specter:
            prestdstar_args += '--gpu_specter'
        if args.gpuextract:
            prestdstar_args += '--gpuextract'
        prestdstar_args = proc.parse(prestdstar_args)
        err = proc.main(prestdstar_args,comm)

    #- run joint stdstar fit using all exp for this night-tile
    stdstar_args  = common_args
    stdstar_args += f' --obstype science --mpistdstars --expids {",".join(map(str, expids))}'
    stdstar_args = proc_joint_fit.parse(stdstar_args)
    err = proc_joint_fit.main(stdstar_args, comm)

    #- run desiproc poststdstar over exps
    for expid in expids:
        poststdstar_args  = common_args
        poststdstar_args += f' --nostdstarfit --noprestdstarfit --expid {expids[0]}'
        poststdstar_args = proc.parse(poststdstar_args)
        err = proc.main(poststdstar_args, comm)

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
