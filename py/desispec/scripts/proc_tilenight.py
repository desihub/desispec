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

from desispec.workflow.desi_proc_funcs import assign_mpi, get_desi_proc_tnight_parser, update_args_with_headers, \
    find_most_recent
from desispec.workflow.desi_proc_funcs import determine_resources, create_desi_proc_tnight_batch_script, launch_desi_proc

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

    #-------------------------------------------------------------------------
    #- Create and submit a batch job if requested

    if args.batch:
        # create_desi_proc_tnight_batch_script not implemented yet
        scriptfile = create_desi_proc_tnight_batch_script(night=args.night, tileid=args.tileid,
                                                          cameras=args.cameras,
                                                          jobdesc=jobdesc, queue=args.queue,
                                                          runtime=args.runtime,
                                                          batch_opts=args.batch_opts,
                                                          timingfile=args.timingfile,
                                                          system_name=args.system_name)
        err = 0
        if not args.nosubmit:
            err = subprocess.call(['sbatch', scriptfile])
        sys.exit(err)

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
    
    #- run desiproc prestdstar over exps
    for expid in expids:
        launch_desi_proc(
            comm, proc, 'prestdstar', args.night, [expid], args.cameras,
            gpuspecter=args.gpuspecter, gpuextract=args.gpuextract
        )
    #- run joint stdstar fit using all exp for this night-tile
    launch_desi_proc(
        comm, proc_joint_fit, 'stdstarfit', args.night, expids, camword,
        timingsuffix=self.timingsuffix, gpuextract=self.gpuextract
    )
    #- run desiproc poststdstar over exps
    for expid in expids:
        launch_desi_proc(
            comm, proc, 'poststdstar', night, [expid], camword,
            dryrun=args.dry_run, gpuextract=False
        )

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
