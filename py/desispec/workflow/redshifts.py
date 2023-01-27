"""
desispec.workflow.redshifts
===========================

"""
import sys, os, glob
import re
import subprocess
import argparse
import numpy as np
from astropy.table import Table, vstack, Column

from desispec.io.util import parse_cameras, decode_camword
from desispec.workflow.desi_proc_funcs import determine_resources
from desiutil.log import get_logger

import desispec.io
from desispec.workflow.exptable import get_exposure_table_path, get_exposure_table_name, \
                                       get_exposure_table_pathname
from desispec.workflow.tableio import load_table
from desispec.workflow import batch
from desispec.util import parse_int_args



def get_ztile_relpath(tileid,group,night=None,expid=None):
    """
    Determine the relative output directory of the tile redshift batch script for spectra+coadd+redshifts for a tile

    Args:
        tileid (int): Tile ID
        group (str): cumulative, pernight, perexp, or a custom name
        night (int): Night
        expid (int): Exposure ID

    Returns:
        outdir (str): the relative path of output directory of the batch script from the specprod/run/scripts
    """
    log = get_logger()
    # - output directory relative to reduxdir
    if group == 'cumulative':
        outdir = f'tiles/{group}/{tileid}/{night}'
    elif group == 'pernight':
        outdir = f'tiles/{group}/{tileid}/{night}'
    elif group == 'perexp':
        outdir = f'tiles/{group}/{tileid}/{expid:08d}'
    elif group == 'pernight-v0':
        outdir = f'tiles/{tileid}/{night}'
    else:
        outdir = f'tiles/{group}/{tileid}'
        log.warning(f'Non-standard tile group={group}; writing outputs to {outdir}/*')
    return outdir

def get_ztile_script_pathname(tileid,group,night=None,expid=None):
    """
    Generate the pathname of the tile redshift batch script for spectra+coadd+redshifts for a tile

    Args:
        tileid (int): Tile ID
        group (str): cumulative, pernight, perexp, or a custom name
        night (int): Night
        expid (int): Exposure ID

    Returns:
        (str): the pathname of the tile redshift batch script
    """
    reduxdir = desispec.io.specprod_root()
    outdir = get_ztile_relpath(tileid,group,night=night,expid=expid)
    scriptdir = f'{reduxdir}/run/scripts/{outdir}'
    suffix = get_ztile_script_suffix(tileid,group,night=night,expid=expid)
    batchscript = f'ztile-{suffix}.slurm'
    return os.path.join(scriptdir, batchscript)

def get_ztile_script_suffix(tileid,group,night=None,expid=None):
    """
    Generate the suffix of the tile redshift batch script for spectra+coadd+redshifts for a tile

    Args:
        tileid (int): Tile ID
        group (str): cumulative, pernight, perexp, or a custom name
        night (int): Night
        expid (int): Exposure ID

    Returns:
        suffix (str): the suffix of the batch script
    """
    log = get_logger()
    if group == 'cumulative':
        suffix = f'{tileid}-thru{night}'
    elif group == 'pernight':
        suffix = f'{tileid}-{night}'
    elif group == 'perexp':
        suffix = f'{tileid}-exp{expid:08d}'
    elif group == 'pernight-v0':
        suffix = f'{tileid}-{night}'
    else:
        suffix = f'{tileid}-{group}'
        log.warning(f'Non-standard tile group={group}; writing outputs to {suffix}.*')
    return suffix

def get_zpix_redshift_script_pathname(healpix, survey, program):
    """Return healpix-based coadd+redshift+afterburner script pathname

    Args:
        healpix (int or array-like): healpixel(s)
        survey (str): DESI survey, e.g. main, sv1, sv3
        program (str): survey program, e.g. dark, bright, backup

    Returns:
        zpix_script_pathname
    """
    if np.isscalar(healpix):
        healpix = [healpix,]

    hpixmin = np.min(healpix)
    hpixmax = np.max(healpix)
    if len(healpix) == 1:
        scriptname = f'zpix-{survey}-{program}-{healpix[0]}.slurm'
    else:
        scriptname = f'zpix-{survey}-{program}-{hpixmin}-{hpixmax}.slurm'

    reduxdir = desispec.io.specprod_root()
    return os.path.join(reduxdir, 'run', 'scripts', 'healpix',
                        survey, program, str(hpixmin//100), scriptname)

def create_desi_zproc_batch_script(group,
                                   tileid=None, cameras=None,
                                   thrunight=None, nights=None, expids=None,
                                   healpix=None, survey=None, program=None,
                                   queue='regular', batch_opts=None,
                                   runtime=None, timingfile=None, batchdir=None,
                                   jobname=None, cmdline=None, system_name=None,
                                   max_gpuprocs=None, no_gpu=False, run_zmtl=False,
                                   no_afterburners=False):
    """
    Generate a SLURM batch script to be submitted to the slurm scheduler to run desi_proc.

    Args:
        group (str): Description of the job to be performed. zproc options include:
                     'perexp', 'pernight', 'cumulative'.
        tileid (int), optional: The tile id for the data.
        cameras (str or list of str), optional: List of cameras to include in the processing
                                      or a camword.
        thrunight (int), optional: For group=cumulative, include exposures through this night
        nights (list of int), optional: The nights the data was acquired.
        expids (list of int), optional: The exposure id(s) for the data.
        healpix (list of int), optional: healpixels to process (group='healpix')
        queue (str), optional: Queue to be used.
        batch_opts (str), optional: Other options to give to the slurm batch scheduler (written into the script).
        runtime (str), optional: Timeout wall clock time in minutes.
        timingfile (str), optional: Specify the name of the timing file.
        batchdir (str), optional: can define an alternative location to write the file. The default
                  is to SPECPROD under run/scripts/tiles/GROUP/TILE/NIGHT
        jobname (str), optional: name to save this batch script file as and the name of the eventual log file. Script is save  within
                 the batchdir directory.
        cmdline (str or list of str), optional: Complete command as would be given in terminal to run the desi_zproc,
                 or list of args.  Can be used instead of reading from argv.
        system_name (str), optional: name of batch system, e.g. cori-haswell, cori-knl
        max_gpuprocs (int), optional: Number of gpu processes
        no_gpu (bool), optional: Default false. If true it doesn't use GPU's even if available.
        run_zmtl (bool), optional: Default false. If true it runs zmtl.
        no_afterburners (bool), optional: Default false. If true it doesn't run afterburners.

    Returns:
        scriptfile: the full path name for the script written.

    Note:
        batchdir and jobname can be used to define an alternative pathname, but
           may not work with assumptions in the spectro pipeline.
    """
    log = get_logger()
    if nights is not None:
        nights = np.asarray(nights, dtype=int)
        night = np.max(nights)
    elif thrunight is not None:
        night = thrunight
    elif group == 'healpix':
        if (healpix is None or survey is None or program is None):
            msg = f"group='healpix' must define healpix,survey,program (got {healpix},{survey},{program})"
            log.error(msg)
            raise ValueError(msg)
    else:
        msg = f"group {group} must define either nights or thrunight"
        log.error(msg)
        raise ValueError(msg)

    expid = None
    if group == 'perexp':
        if expids is None:
            msg = f"Must define expids for perexp exposure"
            log.error(msg)
            raise ValueError(msg)
        else:
            expid = expids[0]

    if group == 'healpix':
        scriptpath = get_zpix_redshift_script_pathname(healpix, survey, program)
    else:
        scriptpath = get_ztile_script_pathname(tileid, group=group,
                                                   night=night, expid=expid)

    if cameras is None:
        cameras = decode_camword('a0123456789')
    elif np.isscalar(cameras):
        camword = parse_cameras(cameras)
        cameras = decode_camword(camword)

    if batchdir is None:
        batchdir = os.path.dirname(scriptpath)

    os.makedirs(batchdir, exist_ok=True)

    if jobname is None:
        jobname = os.path.basename(scriptpath).removesuffix('.slurm')

    if timingfile is None:
        timingfile = f'{jobname}-timing-$SLURM_JOBID.json'
        timingfile = os.path.join(batchdir, timingfile)

    scriptfile = os.path.join(batchdir, jobname + '.slurm')

    ## If system name isn't specified, guess it
    if system_name is None:
        system_name = batch.default_system(jobdesc=group, no_gpu=no_gpu)

    batch_config = batch.get_config(system_name)
    threads_per_core = batch_config['threads_per_core']
    gpus_per_node = batch_config['gpus_per_node']
    if max_gpuprocs is not None and max_gpuprocs < gpus_per_node:
        gpus_per_node = max_gpuprocs

    ncameras = len(cameras)
    nexps = 1
    if expids is not None and type(expids) is not str:
        nexps = len(expids)

    if cmdline is None:
        inparams = list(sys.argv).copy()
    elif np.isscalar(cmdline):
        inparams = []
        for param in cmdline.split(' '):
            for subparam in param.split("="):
                inparams.append(subparam)
    else:
        # ensure all elements are str (not int or float)
        inparams = [str(x) for x in cmdline]

    for parameter in ['--queue', '-q', '--batch-opts']:
        ## If a parameter is in the list, remove it and its argument
        ## Elif it is a '--' command, it might be --option=value, which won't be split.
        ##      check for that and remove the whole "--option=value"
        if parameter in inparams:
            loc = np.where(np.array(inparams) == parameter)[0][0]
            # Remove the command
            inparams.pop(loc)
            # Remove the argument of the command (now in the command location after pop)
            inparams.pop(loc)
        elif '--' in parameter:
            for ii, inparam in enumerate(inparams.copy()):
                if parameter in inparam:
                    inparams.pop(ii)
                    break

    cmd = ' '.join(inparams)
    cmd = cmd.replace(' --batch', ' ').replace(' --nosubmit', ' ')

    srun_rr_gpu_opts = ''
    if not no_gpu:
        if system_name == 'perlmutter-gpu':
            if '--max-gpuprocs' not in cmd:
                cmd += f' --max-gpuprocs {gpus_per_node}'
            gpumap = ','.join(np.arange(gpus_per_node).astype(str)[::-1])
            srun_rr_gpu_opts = f' --gpu-bind=map_gpu:{gpumap}'
        else:
            ## no_gpu isn't set, but we want it set since not perlmutter-gpu
            cmd += ' --no-gpu'

    if run_zmtl and '--run-zmtl' not in cmd:
        cmd += ' --run-zmtl'
    if no_afterburners and '--no-afterburners' not in cmd:
        cmd += ' --no-afterburners'

    cmd += ' --starttime $(date +%s)'
    cmd += f' --timingfile {timingfile}'

    if '--mpi' not in cmd:
        cmd += ' --mpi'

    ncores, nodes, runtime = determine_resources(
            ncameras, group.upper(), queue=queue, nexps=nexps,
            forced_runtime=runtime, system_name=system_name)

    runtime_hh = int(runtime // 60)
    runtime_mm = int(runtime % 60)

    with open(scriptfile, 'w') as fx:
        fx.write('#!/bin/bash -l\n\n')
        fx.write('#SBATCH -N {}\n'.format(nodes))
        fx.write('#SBATCH --qos {}\n'.format(queue))
        for opts in batch_config['batch_opts']:
            fx.write('#SBATCH {}\n'.format(opts))
        if batch_opts is not None:
            fx.write('#SBATCH {}\n'.format(batch_opts))
        if system_name == 'perlmutter-gpu' and not no_gpu:
            # perlmutter-gpu requires projects name with "_g" appended
            fx.write('#SBATCH --account desi_g\n')
        else:
            fx.write('#SBATCH --account desi\n')
        fx.write('#SBATCH --job-name {}\n'.format(jobname))
        fx.write('#SBATCH --output {}/{}-%j.log\n'.format(batchdir, jobname))
        fx.write('#SBATCH --time={:02d}:{:02d}:00\n'.format(runtime_hh, runtime_mm))
        fx.write('#SBATCH --exclusive\n')
        fx.write('\n')

        # batch-friendly matplotlib backend
        fx.write('export MPLBACKEND=agg\n')

        # fx.write("export OMP_NUM_THREADS={}\n".format(threads_per_core))
        fx.write("export OMP_NUM_THREADS=1\n")

        #- Special case CFS readonly mount at NERSC
        if 'DESI_ROOT_READONLY' in os.environ:
            readonlydir = os.environ['DESI_ROOT_READONLY']
        elif os.environ['DESI_ROOT'].startswith('/global/cfs/cdirs'):
            readonlydir = os.environ['DESI_ROOT'].replace(
                    '/global/cfs/cdirs', '/dvs_ro/cfs/cdirs', 1)
        else:
            readonlydir = None

        if readonlydir is not None:
            fx.write(f'export DESI_ROOT_READONLY={readonlydir}\n\n')

        fx.write(f'# using {ncores} cores on {nodes} nodes\n\n')

        fx.write('echo Starting at $(date)\n')

        ## Don't currently need for 1 node jobs
        # mps_wrapper=''
        # if system_name == 'perlmutter-gpu':
        #     fx.write("export MPICH_GPU_SUPPORT_ENABLED=1\n")
        #     mps_wrapper='desi_mps_wrapper'

        srun = f"srun -N {nodes} -n {ncores} -c {threads_per_core}" \
               + f"{srun_rr_gpu_opts} --cpu-bind=cores {cmd}"
        fx.write(f"echo RUNNING {srun}\n")
        fx.write(f'{srun}\n')

        fx.write('\nif [ $? -eq 0 ]; then\n')
        fx.write('  echo SUCCESS: done at $(date)\n')
        fx.write('else\n')
        fx.write('  echo FAILED: done at $(date)\n')
        fx.write('  exit 1\n')
        fx.write('fi\n')

    print('Wrote {}'.format(scriptfile))
    print('logfile will be {}/{}-JOBID.log\n'.format(batchdir, jobname))

    return scriptfile


def read_minimal_exptables_columns(nights=None, tileids=None):
    """
    Read exposure tables while handling evolving formats

    Args:
        nights (list of int): nights to include (default all nights found)
        tileids (list of int): tileids to include (default all tiles found)

    Returns exptable with just columns TILEID, NIGHT, EXPID, 'CAMWORD',
        'BADCAMWORD', filtered by science
        exposures with LASTSTEP='all' and TILEID>=0

    Note: the returned table is the full pipeline exposures table. It is trimmed
          to science exposures that have LASTSTEP=='all'
    """
    log = get_logger()
    if nights is None:
        exptab_path = get_exposure_table_path(night=None)
        monthglob = '202???'
        globname = get_exposure_table_name(night='202?????')
        etab_files = glob.glob(os.path.join(exptab_path, monthglob, globname))
    else:
        etab_files = list()
        for night in nights:
            etab_file = get_exposure_table_pathname(night)
            if os.path.exists(etab_file):
                etab_files.append(etab_file)
            elif night >= 20201201:
                log.error(f"Exposure table missing for night {night}")
            else:
                # - these are expected for the daily run, ok
                log.debug(f"Exposure table missing for night {night}")

    etab_files = sorted(etab_files)
    exptables = list()
    for etab_file in etab_files:
        ## correct way but slower and we don't need multivalue columns
        #t = load_table(etab_file, tabletype='etable')
        t = Table.read(etab_file, format='ascii.csv')
        ## For backwards compatibility if BADCAMWORD column does not
        ## exist then add a blank one
        if 'BADCAMWORD' not in t.colnames:
            t.add_column(Table.Column(['' for i in range(len(t))], dtype='S36', name='BADCAMWORD'))
        keep = (t['OBSTYPE'] == 'science') & (t['TILEID'] >= 0)
        if 'LASTSTEP' in t.colnames:
            keep &= (t['LASTSTEP'] == 'all')
        if tileids is not None:
            # Default false
            keep &= np.isin(t['TILEID'], tileids)
        t = t[keep]
        ## Need to ensure that the string columns are consistent
        for col in ['CAMWORD', 'BADCAMWORD']:
            ## Masked arrays need special handling
            ## else just reassign with consistent dtype
            if isinstance(t[col], Table.MaskedColumn):
                ## If compeltely empty it's loaded as type int
                ## otherwise fill masked with ''
                if t[col].dtype == int:
                    t[col] = Table.Column(['' for i in range(len(t))], dtype='S36', name=col)
                else:
                    t[col] = Table.Column(t[col].filled(fill_value=''), dtype='S36', name=col)
            else:
                t[col] = Table.Column(t[col], dtype='S36', name=col)
        exptables.append(t['TILEID', 'NIGHT', 'EXPID', 'CAMWORD', 'BADCAMWORD'])

    return vstack(exptables)
