"""
desispec.workflow.batch_writer
=======================

Utilities for writing slurm batch scripts.
"""

import os
import sys
from desispec.workflow.batch import determine_resources
import numpy as np
from desispec.io import findfile
from desispec.io.util import decode_camword, parse_cameras
from desispec.workflow import batch
from desiutil.log import get_logger


def get_desi_proc_batch_file_name(night, exp, jobdesc, cameras):
    """
    Returns the default directory location to store a batch script file given a night

    Args:
        night: str or int, defines the night (should be 8 digits)
        exp: str, int, or array of ints, defines the exposure id(s) relevant to the job
        jobdesc: str, type of data being processed
        cameras: str or list of str. If str, must be camword, If list, must be list of cameras to include in the processing.

    Returns:
        pathname: str, the default script name for a desi_proc batch script file
    """
    camword = parse_cameras(cameras)
    if jobdesc.lower() == 'linkcal':
        expstr = ""
    elif type(exp) is not str:
        if exp is None:
            expstr = 'none'
        elif np.isscalar(exp):
            expstr = '{:08d}'.format(exp)
        else:
            #expstr = '-'.join(['{:08d}'.format(curexp) for curexp in exp])
            expstr = '{:08d}'.format(exp[0])
    else:
        expstr = exp

    if expstr != "":
        expstr = "-" + expstr

    jobname = f'{jobdesc.lower()}-{night}{expstr}-{camword}'
    return jobname


def get_desi_proc_batch_file_path(night,reduxdir=None):
    """
    Returns the default directory location to store a batch script file given a night

    Args:
        night (str or int): defines the night (should be 8 digits)
        reduxdir (str, optional): define the base directory where the /run/scripts directory should or does live.

    Returns:
        str: the default location where a batch script file should be written
    """
    if reduxdir is None:
        from desispec.io import specprod_root
        reduxdir = specprod_root()
    batchdir = os.path.join(reduxdir, 'run', 'scripts', 'night', str(night))
    return batchdir


def get_desi_proc_batch_file_pathname(night, exp, jobdesc, cameras,
                                      reduxdir=None):
    """
    Returns the default directory location to store a batch script file given a night

    Args:
        night: str or int, defines the night (should be 8 digits)
        exp: str, int, or array of ints, defines the exposure id(s) relevant to the job
        jobdesc: str, type of data being processed
        cameras: str or list of str. If str, must be camword, If list, must be list of cameras to include in the processing.
        reduxdir: str (optional), define the base directory where the /run/scripts directory should or does live

    Returns:
        pathname: str, the default location and script name for a desi_proc batch script file
    """
    path = get_desi_proc_batch_file_path(night, reduxdir=reduxdir)
    name = get_desi_proc_batch_file_name(night, exp, jobdesc, cameras)
    return os.path.join(path, name)


def get_desi_proc_tilenight_batch_file_name(night, tileid):
    """
    Returns the filename for a tilenight batch script file given a night and tileid

    Args:
        night: str or int, defines the night (should be 8 digits)
        tileid: str or int, defines the tile id relevant to the job

    Returns:
        pathname: str, the default script name for a desi_proc_tilenight batch script file
    """
    if type(tileid) is not str:
        if np.isscalar(tileid):
            tileid = '{}'.format(tileid)
        else:
            raise RuntimeError('tileid should be either int or str')

    jobname = 'tilenight-{}-{}'.format(night, tileid)
    return jobname


def get_desi_proc_tilenight_batch_file_pathname(night, tileid, reduxdir=None):
    """
    Returns the default directory location to store a tilenight batch script file given a night and tileid

    Args:
        night: str or int, defines the night (should be 8 digits)
        tileid: str or int, defines the tile id relevant to the job
        reduxdir: str (optional), define the base directory where the /run/scripts directory should or does live

    Returns:
        pathname: str, the default location and script name for a desi_proc_tilenight batch script file
    """
    path = get_desi_proc_batch_file_path(night,reduxdir=reduxdir)
    name = get_desi_proc_tilenight_batch_file_name(night,tileid)
    return os.path.join(path, name)


def wrap_command_for_script(cmd, nodes, ntasks, threads_per_task):
    """
    Wraps a command for execution in a bash script using srun.

    Args:
        cmd (str): The command to be executed.
        nodes (int): Number of nodes to use.
        ntasks (int): Total number of tasks to use.
        threads_per_task (int): Number of threads per core.

    Returns:
        str: The wrapped command ready for inclusion in a bash script.
    """
    srun = f'srun -N {nodes} -n {ntasks} -c {threads_per_task} --cpu-bind=cores {cmd}'
    wrapped_cmd =  f'echo Running {srun}\n'
    wrapped_cmd += f'{srun}\n\n'

    wrapped_cmd += 'if [ $? -eq 0 ]; then\n'
    wrapped_cmd += '\techo pdark succeeded at $(date)\n'
    wrapped_cmd += 'else\n'
    wrapped_cmd += '\techo FAILED: pdark failed, stopping at $(date)\n'
    wrapped_cmd += '\texit 1\n'
    wrapped_cmd += 'fi\n'
    return wrapped_cmd
    

def create_linkcal_batch_script(newnight, queue, cameras=None, runtime=None,
                                batch_opts=None, timingfile=None,
                                batchdir=None, jobname=None, cmd=None,
                                system_name=None):
    """
    Generate a batch script to be submitted to the slurm scheduler to run
    desi_link_calibnight.

    Args:
        newnight (str or int): The night in calibnight where the links will
        queue (str): Queue to be used.
        cameras (str or list of str): List of cameras to include in the processing.
        runtime (str, optional): Timeout wall clock time.
        batch_opts (str, optional): Other options to give to the slurm batch scheduler (written into the script).
        timingfile (str, optional): Specify the name of the timing file.
        batchdir (str, optional): Specify where the batch file will be written.
        jobname (str, optional): Specify the name of the slurm script written.
        cmd (str, optional): Complete command as would be given in terminal to
            run desi_link_calibnight.
        system_name (str, optional): name of batch system, e.g. cori-haswell, cori-knl

    Returns:
        scriptfile: the full path name for the script written.

    Note:
        batchdir and jobname can be used to define an alternative pathname, but may not work with assumptions in desi_proc.
        These optional arguments should be used with caution and primarily for debugging.
    """
    jobdesc = 'linkcal'

    if cameras is None or np.isscalar(cameras):
        camword = cameras
        cameras = decode_camword(camword)

    if batchdir is None:
        batchdir = get_desi_proc_batch_file_path(newnight)

    os.makedirs(batchdir, exist_ok=True)

    if jobname is None:
        jobname = get_desi_proc_batch_file_name(night=newnight, exp="",
                                                jobdesc=jobdesc, cameras=cameras)

    if timingfile is None:
        timingfile = f'{jobname}-timing-$SLURM_JOBID.json'
        timingfile = os.path.join(batchdir, timingfile)

    scriptfile = os.path.join(batchdir, jobname + '.slurm')

    ## If system name isn't specified, pick it based upon jobdesc
    if system_name is None:
        system_name = batch.default_system(jobdesc=jobdesc)

    batch_config = batch.get_config(system_name)
    threads_per_core = batch_config['threads_per_core']
    gpus_per_node = batch_config['gpus_per_node']
    ncameras = len(cameras)


    ncores, nodes, runtime = determine_resources(ncameras, jobdesc.upper(),
                                                 forced_runtime=runtime,
                                                 system_name=system_name)

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
        if system_name == 'perlmutter-gpu':
            # perlmutter-gpu requires projects name with "_g" appended
            fx.write('#SBATCH --account desi_g\n')
        else:
            fx.write('#SBATCH --account desi\n')
        fx.write('#SBATCH --job-name {}\n'.format(jobname))
        fx.write('#SBATCH --output {}/{}-%j.log\n'.format(batchdir, jobname))
        fx.write('#SBATCH --time={:02d}:{:02d}:00\n'.format(runtime_hh, runtime_mm))
        #fx.write('#SBATCH --exclusive\n')

        fx.write('\n')

        fx.write(f'# {jobdesc} with {ncameras} cameras\n')
        fx.write(f'# using {ncores} cores on {nodes} nodes\n\n')

        fx.write('echo Starting job $SLURM_JOB_ID on $(hostname) at $(date)\n')
        # fx.write("export OMP_NUM_THREADS=1\n")

        fx.write(f'\n# Link refnight to new night\n')
        fx.write(wrap_command_for_script(cmd, nodes, ntasks=ncores, threads_per_task=threads_per_core))

    print('Wrote {}'.format(scriptfile))
    print('logfile will be {}/{}-JOBID.log\n'.format(batchdir, jobname))

    return scriptfile


def create_biaspdark_batch_script(night, expids,
                                 jobsdesc=None, camword='a0123456789',
                                 do_biasnight=False, do_pdark=False, 
                                 queue='regular', system_name=None):
    """
    Generate a SLURM batch script to be submitted to the slurm scheduler to run biasnight
    and then preproc darks script.

    Args:
        night (str or int): The night in which the biaspdark script will be run.
        expids (list of int or np.array): The exposure id(s) for the data.
        jobsdesc (str, optional): Description of the job to be performed. If None, will 
            default to 'biaspdark' or 'pdark' depending on do_biasnight and do_pdark.
        camword (str): Camword of cameras to include in the processing.
        do_biasnight (bool): If True, run the nightly bias script first.
        do_pdark (bool): If True, run the preproc darks script.
        queue (str): Queue to be used.
        system_name (str, optional): name of batch system, e.g. cori-haswell, perlmutter

    Returns:
        scriptpathname (str): The full path name for the biaspdark batch script file.
    """
    log = get_logger()
    if jobsdesc is None:
        if do_biasnight:
            jobdesc = 'biaspdark'
        elif do_pdark:
            jobdesc = 'pdark'
        else:
            log.error('Must specify at least one of do_biasnight or do_pdark')
            raise ValueError('Must specify at least one of do_biasnight or do_pdark')
    
    scriptpathname = get_desi_proc_batch_file_pathname(night=night, exp=expids, 
                                                   jobdesc=jobdesc, cameras=camword)

    cameras = decode_camword(camword)
    ncameras = len(cameras)
    nexps = len(expids) if expids is not None else 1
    batchdir = os.path.dirname(scriptpathname)
    os.makedirs(batchdir, exist_ok=True)
    jobname = os.path.basename(scriptpathname).removesuffix('.slurm')

    ## If system name isn't specified, guess it
    if system_name is None:
        system_name = batch.default_system(jobdesc=jobdesc)

    batch_config = batch.get_config(system_name)

    dark_ncores, nodes, runtime = determine_resources(ncameras, jobdesc='pdark', 
                                                        queue=queue, nexps=nexps, 
                                                        system_name=system_name)
    script_body = ""
    # Run nightlybias first  
    if do_biasnight: 
        tot_threads = batch_config['threads_per_core'] * batch_config['cores_per_node']
        bias_threads_per_task = tot_threads // 8
        bias_ntasks, nodes, bias_runtime = determine_resources(ncameras, jobdesc='biasnight', 
                                                                    queue=queue, nexps=1, 
                                                                    system_name=system_name)
        runtime += bias_runtime
        cmd = f'desi_proc --cameras {camword} -n {night} --nightlybias --mpi'
        script_body += wrap_command_for_script(cmd, nodes, ntasks=bias_ntasks, threads_per_task=bias_threads_per_task)

    # Then pdarks  
    if do_pdark: 
        ## if do_biasnight is True, then we need to run pdark with the same number of nodes
        if do_biasnight:
            dark_ntasks = min([ncameras*nexps, nodes*batch_config['cores_per_node']])

        cmd = f'desi_preproc_darks -n {night} --expids={",".join(expids)} --camword={camword} --mpi'
        script_body += wrap_command_for_script(cmd, nodes, ntasks=dark_ntasks, threads_per_task=batch_config['threads_per_core'])

    runtime_hh = int(runtime // 60)
    runtime_mm = int(runtime % 60)

    with open(scriptpathname, 'w') as fx:
        fx.write('#!/bin/bash -l\n\n')
        fx.write('#SBATCH -N {}\n'.format(nodes))
        fx.write('#SBATCH --qos {}\n'.format(queue))
        for opts in batch_config['batch_opts']:
            fx.write('#SBATCH {}\n'.format(opts))
        fx.write('#SBATCH --account desi\n')
        fx.write('#SBATCH --job-name {}\n'.format(jobname))
        fx.write('#SBATCH --output {}/{}-%j.log\n'.format(batchdir, jobname))
        fx.write('#SBATCH --time={:02d}:{:02d}:00\n'.format(runtime_hh, runtime_mm))
        fx.write('#SBATCH --exclusive\n')
        fx.write('\n')

        # batch-friendly matplotlib backend
        fx.write('export MPLBACKEND=agg\n')

        ## we're using MPI for this job, so set OMP_NUM_THREADS to 1
        fx.write("export OMP_NUM_THREADS=1\n")
        fx.write(f'# using {nodes*batch_config["cores_per_node"]} cores on {nodes} nodes\n\n')

        fx.write('echo Starting at $(date)\n')

        fx.write(script_body)

    print('Wrote {}'.format(scriptpathname))
    print('logfile will be {}/{}-JOBID.log\n'.format(batchdir, jobname))

    return scriptpathname


def create_ccdcalib_batch_script(night, expids, camword='a0123456789', 
                                 do_darknight=False, do_badcolumn=False, 
                                 do_ctecorr=False, n_nights_before=None, n_nights_after=None,
                                 dark_expid=None, cte_expids=None,
                                 queue='regular', system_name=None):
    """
    Generate a SLURM batch script to be submitted to the slurm scheduler to run the 
    requested CCD calibration tasks

    Args:
        night (str or int): The night in which the ccdcalib script will be run.
        expids (list of int or np.array): The exposure id(s) for the data.
        camword (str): Camword of cameras to include in the processing.
        do_darknight (bool): If True, run the darknight script first.
        do_badcolumn (bool): If True, run the badcolumn script.
        do_ctecorr (bool): If True, run the ctecorr script.
        n_nights_before (int, optional): Number of nights before the current night to include in the darknight script.
        n_nights_after (int, optional): Number of nights after the current night to include in the darknight script.
        dark_expid (int, optional): The exposure id to use for the darknight script. If None, will use the first expid.
        cte_expids (list of int, optional): The exposure ids to use for the ctecorr script. If None, will use all expids except the first.
        queue (str): Queue to be used.
        system_name (str, optional): name of batch system, e.g. cori-haswell, perlmutter

    Returns:
        scriptpathname (str): The full path name for the ccdcalib batch script file.
    """
    log = get_logger()
    if not (do_darknight or do_badcolumn or do_ctecorr):
        log.critical('Must specify at least one of do_darknight, do_badcolumn, or do_ctecorr')
        raise ValueError('Must specify at least one of do_darknight, do_badcolumn, or do_ctecorr')
    jobdesc = 'ccdcalib'
        
    scriptpathname = get_desi_proc_batch_file_pathname(night=night, exp=expids, 
                                                   jobdesc=jobdesc, cameras=camword)

    cameras = decode_camword(camword)
    ncameras = len(cameras)
    nexps = len(expids) if expids is not None else 1
    batchdir = os.path.dirname(scriptpathname)
    os.makedirs(batchdir, exist_ok=True)
    jobname = os.path.basename(scriptpathname).removesuffix('.slurm')

    ## If system name isn't specified, guess it
    if system_name is None:
        system_name = batch.default_system(jobdesc=jobdesc)

    batch_config = batch.get_config(system_name)
    threads_per_core = batch_config['threads_per_core']
    ntasks, nodes, runtime = determine_resources(ncameras, jobdesc='ccdcalib', 
                                                 queue=queue, nexps=nexps, 
                                                 system_name=system_name)
    threads_per_task = threads_per_core*nodes*(batch_config['cores_per_node'] // ntasks)
    script_body = ""
    # Run nightlybias first  
    if do_darknight: 
        cmd = f'desi_compute_dark_night --reference_night={night} --camword={camword}'
        if n_nights_before is not None:
            cmd += f' --before={n_nights_before}'
        if n_nights_after is not None:
            cmd += f' --after={n_nights_after}'
        cmd += ' --mpi'
        script_body += wrap_command_for_script(cmd, nodes, ntasks=ntasks, threads_per_task=threads_per_task)

    # Then pdarks  
    if do_badcolumn: 
        if dark_expid is None:
            dark_expid = expids[0]
        cmd = f'desi_proc -n {night} -c {camword} -e {dark_expid} --mpi'
        script_body += wrap_command_for_script(cmd, nodes, ntasks=ntasks, threads_per_task=threads_per_task)

    if do_ctecorr:
        if cte_expids is None:
            cte_expids = expids[1:]
        cmd = f"desi_fit_cte_night -n {night} -c {camword} -e {cte_expids}"
        script_body += wrap_command_for_script(cmd, nodes, ntasks=ntasks, threads_per_task=threads_per_task)

    runtime_hh = int(runtime // 60)
    runtime_mm = int(runtime % 60)

    with open(scriptpathname, 'w') as fx:
        fx.write('#!/bin/bash -l\n\n')
        fx.write('#SBATCH -N {}\n'.format(nodes))
        fx.write('#SBATCH --qos {}\n'.format(queue))
        for opts in batch_config['batch_opts']:
            fx.write('#SBATCH {}\n'.format(opts))
        fx.write('#SBATCH --account desi\n')
        fx.write('#SBATCH --job-name {}\n'.format(jobname))
        fx.write('#SBATCH --output {}/{}-%j.log\n'.format(batchdir, jobname))
        fx.write('#SBATCH --time={:02d}:{:02d}:00\n'.format(runtime_hh, runtime_mm))
        fx.write('#SBATCH --exclusive\n')
        fx.write('\n')

        # batch-friendly matplotlib backend
        fx.write('export MPLBACKEND=agg\n')

        ## we're using MPI for this job, so set OMP_NUM_THREADS to 1
        fx.write("export OMP_NUM_THREADS=1\n")
        fx.write(f'# using {nodes*batch_config["cores_per_node"]} cores on {nodes} nodes\n\n')

        fx.write('echo Starting at $(date)\n')

        fx.write(script_body)

    print('Wrote {}'.format(scriptpathname))
    print('logfile will be {}/{}-JOBID.log\n'.format(batchdir, jobname))

    return scriptpathname


def create_desi_proc_batch_script(night, exp, cameras, jobdesc, queue,
                                  runtime=None, batch_opts=None, timingfile=None,
                                  batchdir=None, jobname=None, cmdline=None,
                                  system_name=None, use_specter=False,
                                  no_gpu=False, nightlybias=None,
                                  nightlycte=None, cte_expids=None):
    """
    Generate a SLURM batch script to be submitted to the slurm scheduler to run desi_proc.

    Args:
        night (str or int): The night the data was acquired
        exp (str, int, or list of int): The exposure id(s) for the data.
        cameras (str or list of str): List of cameras to include in the processing.
        jobdesc (str): Description of the job to be performed. Used to determine requested resources
            and whether to operate in a more mpi parallelism (all except poststdstar) or less (only poststdstar).
            Directly relate to the obstype, with science exposures being split into two (pre, post)-stdstar,
            and adding joint fit categories stdstarfit, psfnight, and nightlyflat.
            Options include: 'prestdstar', 'poststdstar', 'stdstarfit', 'arc', 'flat', 'psfnight', 'nightlyflat'
        queue (str): Queue to be used.
        runtime (str, optional): Timeout wall clock time.
        batch_opts (str, optional): Other options to give to the slurm batch scheduler (written into the script).
        timingfile (str, optional): Specify the name of the timing file.
        batchdir (str, optional): Specify where the batch file will be written.
        jobname (str, optional): Specify the name of the slurm script written.
        cmdline (str, optional): Complete command as would be given in terminal to run the desi_proc. Can be used instead
            of reading from argv.
        system_name (str, optional): name of batch system, e.g. cori-haswell, cori-knl
        use_specter (bool, optional): Use classic specter instead of gpu_specter for extractions
        no_gpu (bool, optional): Do not use GPU even if available
        nightlybias (bool): Create nightly bias model from ZEROs
        nightlycte (bool): Fit CTE model from LED exposures
        cte_expids (list): Explicitly name expids of the cte flat and flat to use for cte model

    Returns:
        scriptfile: the full path name for the script written.

    Note:
        batchdir and jobname can be used to define an alternative pathname, but may not work with assumptions in desi_proc.
        These optional arguments should be used with caution and primarily for debugging.
    """
    log = get_logger()
    if np.isscalar(cameras):
        camword = cameras
        cameras = decode_camword(camword)

    if batchdir is None:
        batchdir = get_desi_proc_batch_file_path(night)

    os.makedirs(batchdir, exist_ok=True)

    if jobname is None:
        jobname = get_desi_proc_batch_file_name(night, exp, jobdesc, cameras)

    if timingfile is None:
        timingfile = f'{jobname}-timing-$SLURM_JOBID.json'
        timingfile = os.path.join(batchdir, timingfile)

    scriptfile = os.path.join(batchdir, jobname + '.slurm')

    ## If system name isn't specified, pick it based upon jobdesc
    if system_name is None:
        system_name = batch.default_system(jobdesc=jobdesc)

    batch_config = batch.get_config(system_name)
    threads_per_core = batch_config['threads_per_core']
    gpus_per_node = batch_config['gpus_per_node']
    ncameras = len(cameras)
    nexps = 1
    if exp is not None and not np.isscalar(exp) and type(exp) is not str:
        nexps = len(exp)

    ncores, nodes, runtime = determine_resources(
            ncameras, jobdesc.upper(), queue=queue, nexps=nexps,
            forced_runtime=runtime, system_name=system_name)

    ## derive from cmdline or sys.argv whether this is a nightlybias job
    ## if not explicitly defined
    if nightlybias is None:
        nightlybias = False
        if cmdline is not None:
            if '--nightlybias' in cmdline:
                nightlybias = True
        elif '--nightlybias' in sys.argv:
            nightlybias = True

    #- nightlybias jobs are memory limited, so throttle number of ranks
    if nightlybias:
        tot_threads = batch_config['threads_per_core'] * batch_config['cores_per_node']
        bias_threads_per_core = tot_threads // 8

        bias_cores, bias_nodes, bias_runtime = determine_resources(
                ncameras, 'NIGHTLYBIAS', queue=queue, nexps=nexps,
                system_name=system_name)

        nodes = max(nodes, bias_nodes)
        runtime += bias_runtime

    ## derive from cmdline or sys.argv whether this is a nightlycte job
    ## if not explicitly defined
    if nightlycte is None:
        nightlycte = False
        if cmdline is not None:
            if '--nightlycte' in cmdline:
                nightlycte = True
        elif '--nightlycte' in sys.argv:
            nightlycte = True

    ## nightlycte jobs add time to the job
    ## hardcoding a runtime for nightlycte.
    ## TODO should be moved into determine_resources()
    if nightlycte:
        cte_runtime = 5
        runtime += cte_runtime

    #- arc fits require 3.2 GB of memory per bundle, so increase nodes as needed
    if jobdesc.lower() == 'arc':
        cores_per_node = (ncores-1) // nodes + ((ncores-1) % nodes > 0)
        mem_per_node = float(batch_config['memory'])
        mem_per_core = mem_per_node / cores_per_node
        while mem_per_core < 3.2:
            nodes += 1
            cores_per_node = (ncores-1) // nodes + ((ncores-1) % nodes > 0)
            mem_per_core = mem_per_node / cores_per_node
        threads_per_node = batch_config['threads_per_core'] * batch_config['cores_per_node']
        threads_per_core = (threads_per_node * nodes) // ncores

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
        if system_name == 'perlmutter-gpu':
            # perlmutter-gpu requires projects name with "_g" appended
            fx.write('#SBATCH --account desi_g\n')
        else:
            fx.write('#SBATCH --account desi\n')
        fx.write('#SBATCH --job-name {}\n'.format(jobname))
        fx.write('#SBATCH --output {}/{}-%j.log\n'.format(batchdir, jobname))
        fx.write('#SBATCH --time={:02d}:{:02d}:00\n'.format(runtime_hh, runtime_mm))
        fx.write('#SBATCH --exclusive\n')

        fx.write('\n')

        #- Special case CFS readonly mount at NERSC
        #- SB 2023-01-27: disable this since Perlmutter might deprecate /dvs_ro;
        #- inherit it from the environment but don't hardcode into script itself
        # if 'DESI_ROOT_READONLY' in os.environ:
        #     readonlydir = os.environ['DESI_ROOT_READONLY']
        # elif os.environ['DESI_ROOT'].startswith('/global/cfs/cdirs'):
        #     readonlydir = os.environ['DESI_ROOT'].replace(
        #             '/global/cfs/cdirs', '/dvs_ro/cfs/cdirs', 1)
        # else:
        #     readonlydir = None
        #
        # if readonlydir is not None:
        #     fx.write(f'export DESI_ROOT_READONLY={readonlydir}\n\n')

        if cmdline is None:
            inparams = list(sys.argv).copy()
        elif np.isscalar(cmdline):
            inparams = []
            for param in cmdline.split(' '):
                for subparam in param.split("="):
                    inparams.append(subparam)
        else:
            inparams = list(cmdline)
        for parameter in ['--queue', '-q', '--batch-opts', '--cte-expids']:
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
                for ii,inparam in enumerate(inparams.copy()):
                    if parameter in inparam:
                        inparams.pop(ii)
                        break

        cmd = ' '.join(inparams)
        cmd = cmd.replace(' --batch', ' ').replace(' --nosubmit', ' ')
        cmd = cmd.replace(' --nightlycte', ' ')
        if '--mpi' not in cmd:
            cmd += ' --mpi'

        if jobdesc.lower() == 'stdstarfit':
            cmd += ' --mpistdstars'

        if no_gpu and '--no-gpu' not in cmd:
            cmd += ' --no-gpu'

        if (use_specter and ('--use-specter' not in cmd) and
                jobdesc.lower() in ['flat', 'science', 'prestdstar', 'tilenight']):
            cmd += ' --use-specter'

        cmd += ' --starttime $(date +%s)'
        cmd += f' --timingfile {timingfile}'

        fx.write(f'# {jobdesc} exposure with {ncameras} cameras\n')
        fx.write(f'# using {ncores} cores on {nodes} nodes\n\n')

        fx.write('echo Starting job $SLURM_JOB_ID on $(hostname) at $(date)\n')

        mps_wrapper=''
        if jobdesc.lower() == 'arc':
            fx.write("export OMP_NUM_THREADS={}\n".format(threads_per_core))
        else:
            fx.write("export OMP_NUM_THREADS=1\n")
        if system_name == 'perlmutter-gpu' and jobdesc.lower() not in ['arc']:
            fx.write("export MPICH_GPU_SUPPORT_ENABLED=1\n")
            mps_wrapper='desi_mps_wrapper'

        if jobdesc.lower() not in ['science', 'prestdstar', 'stdstarfit', 'poststdstar']:
            if nightlybias:
                tmp = cmd.split()
                has_expid = False
                if '-e' in tmp:
                    has_expid = True
                    i = tmp.index('-e')
                    tmp.pop(i)  # -e
                    tmp.pop(i)  # EXPID
                if '--expid' in tmp:
                    has_expid = True
                    i = tmp.index('--expid')
                    tmp.pop(i)  # --expid
                    tmp.pop(i)  # EXPID
                bias_cmd = ' '.join(tmp)

                fx.write('\n# Run nightlybias first\n')
                srun=f'srun -N {bias_nodes} -n {bias_cores} -c {bias_threads_per_core} {bias_cmd}'
                fx.write('echo Running {}\n'.format(srun))
                fx.write('{}\n'.format(srun))

                if has_expid:
                    fx.write('\nif [ $? -eq 0 ]; then\n')
                    fx.write('  echo nightlybias succeeded at $(date)\n')
                    fx.write('else\n')
                    fx.write('  echo FAILED: nightlybias failed; stopping at $(date)\n')
                    fx.write('  exit 1\n')
                    fx.write('fi\n')

            if ' -e ' in cmd or ' --expid ' in cmd:
                fx.write('\n# Process exposure\n')
                cmd = cmd.replace(' --nightlybias', '')
                cmd = cmd.replace(' --nightlycte', '')
                srun=(f'srun -N {nodes} -n {ncores} -c {threads_per_core} --cpu-bind=cores '
                    +mps_wrapper+f' {cmd}')
                fx.write('echo Running {}\n'.format(srun))
                fx.write('{}\n'.format(srun))

            #- nightlybias implies that this is a ccdcalib job,
            #- where we will also run CTE fitting
            if nightlybias:

                #- first check if previous command failed
                fx.write('\nif [ $? -eq 0 ]; then\n')
                fx.write('  echo command succeeded at $(date)\n')
                fx.write('else\n')
                fx.write('  echo FAILED: processing failed; stopping at $(date)\n')
                fx.write('  exit 1\n')
                fx.write('fi\n')

            if nightlycte:
                #- then proceed with desi_fit_cte_night command
                camword = parse_cameras(cameras)
                fx.write('\n# Fit CTE parameters from flats if needed\n')
                cmd = f'desi_fit_cte_night -n {night} -c {camword}'
                if cte_expids is not None:
                    cmd += f' -e ' + ','.join(np.atleast_1d(cte_expids).astype(str))
                ctecorrfile = findfile('ctecorrnight', night=night)
                fname = os.path.basename(ctecorrfile)
                fx.write(f'if [ -f {ctecorrfile} ]; then\n')
                fx.write(f'  echo Already have {fname}\n')
                fx.write(f'else\n')
                fx.write(f'  echo running {cmd}\n')
                fx.write(f'  {cmd}\n')
                fx.write(f'fi\n')

        else:
            if jobdesc.lower() in ['science', 'prestdstar', 'stdstarfit']:
                fx.write('\n# Do steps through stdstarfit at full MPI parallelism\n')
                srun = (f' srun -N {nodes} -n {ncores} -c {threads_per_core} --cpu-bind=cores '
                    +mps_wrapper+f' {cmd}')
                if jobdesc.lower() in ['science', 'prestdstar']:
                    srun += ' --nofluxcalib'
                fx.write('echo Running {}\n'.format(srun))
                fx.write('{}\n'.format(srun))

            if jobdesc.lower() in ['science', 'poststdstar']:
                ntasks=ncameras

                tot_threads = nodes * batch_config['cores_per_node'] * batch_config['threads_per_core']
                threads_per_task = max(int(tot_threads / ntasks), 1)
                fx.write('\n# Use less MPI parallelism for fluxcalib MP parallelism\n')
                fx.write('# This should quickly skip over the steps already done\n')
                #- fluxcalib multiprocessing parallelism needs --cpu-bind=none (or at least not "cores")
                srun = f'srun -N {nodes} -n {ntasks} -c {threads_per_task} --cpu-bind=none {cmd} '
                fx.write('if [ $? -eq 0 ]; then\n')
                fx.write('  echo Running {}\n'.format(srun))
                fx.write('  {}\n'.format(srun))
                fx.write('else\n')
                fx.write('  echo FAILED: done at $(date)\n')
                fx.write('  exit 1\n')
                fx.write('fi\n')

        fx.write('\nif [ $? -eq 0 ]; then\n')
        fx.write('  echo SUCCESS: done at $(date)\n')
        fx.write('else\n')
        fx.write('  echo FAILED: done at $(date)\n')
        fx.write('  exit 1\n')
        fx.write('fi\n')

    print('Wrote {}'.format(scriptfile))
    print('logfile will be {}/{}-JOBID.log\n'.format(batchdir, jobname))

    return scriptfile


def create_desi_proc_tilenight_batch_script(night, exp, tileid, ncameras, queue, runtime=None, batch_opts=None,
                                  system_name=None, mpistdstars=True, use_specter=False,
                                  no_gpu=False, laststeps=None, cameras=None
                                  ):
    """
    Generate a SLURM batch script to be submitted to the slurm scheduler to run desi_proc.

    Args:
        night: str or int. The night the data was acquired.
        exp: int, or list of ints. The exposure id(s) for the data.
        tileid: str or int. The tile id for the data.
        ncameras: int. The number of cameras used for joint fitting.
        queue: str. Queue to be used.

    Options:
        runtime: str. Timeout wall clock time.
        batch_opts: str. Other options to give to the slurm batch scheduler (written into the script).
        system_name: name of batch system, e.g. cori-haswell, cori-knl.
        mpistdstars: bool. Whether to use MPI for stdstar fitting.
        use_specter: bool. Use classic specter instead of gpu_specter for extractions
        no_gpu: bool. Do not use GPU even if available
        laststeps: list of str. A list of laststeps to pass as the laststeps argument to tilenight
        cameras: str, must be camword.

    Returns:
        scriptfile: the full path name for the script written.

    """

    batchdir = get_desi_proc_batch_file_path(night)
    os.makedirs(batchdir, exist_ok=True)

    nexps = 1
    if exp is not None and not np.isscalar(exp):
        nexps = len(exp)

    jobname = get_desi_proc_tilenight_batch_file_name(night, tileid)

    timingfile = f'{jobname}-timing-$SLURM_JOBID.json'
    timingfile = os.path.join(batchdir, timingfile)

    scriptfile = os.path.join(batchdir, jobname + '.slurm')

    ## If system name isn't specified, pick it based upon jobdesc
    if system_name is None:
        system_name = batch.default_system(jobdesc='tilenight')

    batch_config = batch.get_config(system_name)
    threads_per_core = batch_config['threads_per_core']
    gpus_per_node = batch_config['gpus_per_node']

    ncores, nodes, runtime = determine_resources(ncameras,'TILENIGHT',
        queue=queue, nexps=nexps, system_name=system_name,forced_runtime=runtime)

    if runtime is None:
        runtime = 30

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
        if system_name == 'perlmutter-gpu':
            # perlmutter-gpu requires projects name with "_g" appended
            fx.write('#SBATCH --account desi_g\n')
        else:
            fx.write('#SBATCH --account desi\n')
        fx.write('#SBATCH --job-name {}\n'.format(jobname))
        fx.write('#SBATCH --output {}/{}-%j.log\n'.format(batchdir, jobname))
        fx.write('#SBATCH --time={:02d}:{:02d}:00\n'.format(runtime_hh, runtime_mm))
        fx.write('#SBATCH --exclusive\n')

        fx.write('\n')

        #- Special case CFS readonly mount at NERSC
        #- SB 2023-01-27: disable this since Perlmutter might deprecate /dvs_ro;
        #- inherit it from the environment but don't hardcode into script itself
        # if 'DESI_ROOT_READONLY' in os.environ:
        #     readonlydir = os.environ['DESI_ROOT_READONLY']
        # elif os.environ['DESI_ROOT'].startswith('/global/cfs/cdirs'):
        #     readonlydir = os.environ['DESI_ROOT'].replace(
        #             '/global/cfs/cdirs', '/dvs_ro/cfs/cdirs', 1)
        # else:
        #     readonlydir = None
        #
        # if readonlydir is not None:
        #     fx.write(f'export DESI_ROOT_READONLY={readonlydir}\n\n')
        #
        # fx.write('\n')

        cmd = 'desi_proc_tilenight'
        cmd += f' -n {night}'
        cmd += f' -t {tileid}'
        cmd += f' --mpi'
        if cameras is not None:
            cmd += f' --cameras {cameras}'
        else:
            cmd += f' --cameras a0123456789'
        if mpistdstars:
            cmd += f' --mpistdstars'
        if no_gpu:
            cmd += f' --no-gpu'
        elif use_specter:
            cmd += f' --use-specter'
        if laststeps is not None:
            cmd += f' --laststeps="{",".join(laststeps)}"'

        cmd += f' --timingfile {timingfile}'

        fx.write(f'# running a tile-night\n')
        fx.write(f'# using {ncores} cores on {nodes} nodes\n\n')

        fx.write('echo Starting job $SLURM_JOB_ID on $(hostname) at $(date)\n')

        mps_wrapper=''
        if system_name == 'perlmutter-gpu':
            fx.write("export MPICH_GPU_SUPPORT_ENABLED=1\n")
            mps_wrapper='desi_mps_wrapper'

        fx.write('\n# Do steps through stdstarfit at full MPI parallelism\n')
        srun = (f' srun -N {nodes} -n {ncores} -c {threads_per_core} --cpu-bind=cores '
                +mps_wrapper+f' {cmd}')
        fx.write('echo Running {}\n'.format(srun))
        fx.write('{}\n'.format(srun))

        fx.write('\nif [ $? -eq 0 ]; then\n')
        fx.write('  echo SUCCESS: done at $(date)\n')
        fx.write('else\n')
        fx.write('  echo FAILED: done at $(date)\n')
        fx.write('  exit 1\n')
        fx.write('fi\n')

    print('Wrote {}'.format(scriptfile))
    print('logfile will be {}/{}-JOBID.log\n'.format(batchdir, jobname))

    return scriptfile
