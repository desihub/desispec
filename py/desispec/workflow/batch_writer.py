"""
desispec.workflow.batch_writer
==============================

Utilities for writing slurm batch scripts.
"""

import os
import sys
from desispec.workflow.batch import determine_resources, max_nodes_for_jobdesc
import numpy as np
from desispec.io import findfile
from desispec.io.util import decode_camword, parse_cameras
from desispec.workflow import batch
from desiutil.log import get_logger

## Minimum memory in GB required per MPI rank of an arc (PSF) fit
_ARC_MEMORY_PER_RANK = 3.2

## desispec.workflow.schedule.Schedule consumes ranks in groups of this size
## plus one scheduler rank, so arc rank counts must be 20*k + 1
_ARC_RANKS_PER_GROUP = 20


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


def adjust_arc_resources(ncores, nodes, batch_config):
    """
    Adjust node count and thread count for an arc (PSF fit) job.

    Arc fits require _ARC_MEMORY_PER_RANK GB of memory per bundle, so the node
    count is increased until that is satisfied.

    Args:
        ncores (int): number of MPI ranks requested, including the one extra
            desispec.workflow.schedule scheduler rank.
        nodes (int): number of nodes currently requested.
        batch_config (dict): batch configuration from batch.get_config().

    Returns:
        tuple: A tuple containing:

        * nodes: int, number of nodes, possibly increased to satisfy the memory
          requirement.
        * ranks_per_node: int, the largest number of worker ranks that will land
          on a single node.
        * threads_per_task: int, number of threads to give each rank.
    """
    ranks_per_node = (ncores - 1) // nodes + ((ncores - 1) % nodes > 0)
    mem_per_node = float(batch_config['memory'])
    mem_per_rank = mem_per_node / ranks_per_node
    while mem_per_rank < _ARC_MEMORY_PER_RANK:
        nodes += 1
        ranks_per_node = (ncores - 1) // nodes + ((ncores - 1) % nodes > 0)
        mem_per_rank = mem_per_node / ranks_per_node
    threads_per_node = batch_config['threads_per_core'] * batch_config['cores_per_node']
    threads_per_task = (threads_per_node * nodes) // ncores
    return nodes, ranks_per_node, threads_per_task


def wrap_command_for_script(cmd, nodes, ntasks, threads_per_task, stepname='step'):
    """
    Wraps a command for execution in a bash script using srun.

    Args:
        cmd (str): The command to be executed.
        nodes (int): Number of nodes to use.
        ntasks (int): Total number of tasks to use.
        threads_per_task (int): Number of threads per core.
        stepname (str): Short name of command step for logging purposes only

    Returns:
        str: The wrapped command ready for inclusion in a bash script.
    """
    srun = f'srun -N {nodes} -n {ntasks} -c {threads_per_task} --cpu-bind=cores {cmd}'
    wrapped_cmd =  f'\necho Running {srun}\n'
    wrapped_cmd += f'{srun}\n\n'

    wrapped_cmd += 'if [ $? -eq 0 ]; then\n'
    wrapped_cmd += f'    echo {stepname} succeeded at $(date)\n'
    wrapped_cmd += 'else\n'
    wrapped_cmd += f'    echo FAILED: {stepname} failed, stopping at $(date)\n'
    wrapped_cmd += '    exit 1\n'
    wrapped_cmd += 'fi\n'
    return wrapped_cmd

def wrapup_for_script():
    """
    Give the boiler plate ending to a DESI slurm script echo'ing that the job succeeded or failed
    """
    wrapped_cmd  = "\n\n"
    wrapped_cmd += 'if [ $? -eq 0 ]; then\n'
    wrapped_cmd += '    echo All done at $(date)\n'
    wrapped_cmd += 'else\n'
    wrapped_cmd += '    echo FAILED: Script failed, stopping at $(date)\n'
    wrapped_cmd += '    exit 1\n'
    wrapped_cmd += 'fi\n'
    return wrapped_cmd


def create_linkcal_batch_script(newnight, queue, cameras=None, runtime=None,
                                batch_opts=None, timingfile=None,
                                batchdir=None, jobname=None, cmd=None,
                                system_name=None, biascmd=None):
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
        biascmd (str, optional): a second command just for biasnight linking if it has a different camword

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
        fx.write(f'cd {batchdir}\n')

        if biascmd is not None:
            fx.write(f'\n# Link refnight to new night for biasnights')
            fx.write(wrap_command_for_script(biascmd, nodes, ntasks=ncores, threads_per_task=threads_per_core, stepname='biasnight linking'))

        fx.write(f'\n# Link refnight to new night')    
        fx.write(wrap_command_for_script(cmd, nodes, ntasks=ncores, threads_per_task=threads_per_core, stepname='job linking'))
        fx.write(wrapup_for_script())

    print('Wrote {}'.format(scriptfile))
    print('logfile will be {}/{}-JOBID.log\n'.format(batchdir, jobname))

    return scriptfile


def create_biaspdark_batch_script(night, expids,
                                 jobdesc=None, camword='a0123456789',
                                 do_biasnight=False, do_pdark=False,
                                 queue=None, system_name=None):
    """
    Generate a SLURM batch script to be submitted to the slurm scheduler to run biasnight
    and then preproc darks script.

    Args:
        night (str or int): The night in which the biaspdark script will be run.
        expids (list of int or np.array): The exposure id(s) for the data. These are the
            dark expids if pdark or biaspdark is being run. Otherwise it is a zero expid.
        jobdesc (str, optional): Description of the job to be performed. If None, will
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
    if jobdesc is None:
        if do_biasnight:
            if do_pdark:
                jobdesc = 'biaspdark'
            else:
                jobdesc = 'biasnight'
        elif do_pdark:
            jobdesc = 'pdark'
        else:
            log.error('Must specify at least one of do_biasnight or do_pdark')
            raise ValueError('Must specify at least one of do_biasnight or do_pdark')

    ## Default to regular queue
    if queue is None:
        queue = 'regular'

    scriptpathname = get_desi_proc_batch_file_pathname(night=night, exp=expids,
                                                   jobdesc=jobdesc, cameras=camword)
    scriptpathname += '.slurm'
    cameras = decode_camword(camword)
    ncameras = len(cameras)
    nexps = len(expids) if expids is not None else 1
    expids = np.array(expids) if expids is not None else None
    batchdir = os.path.dirname(scriptpathname)
    os.makedirs(batchdir, exist_ok=True)
    jobname = os.path.basename(scriptpathname).removesuffix('.slurm')
    timingfile = f'{jobname}-timing-$SLURM_JOBID.json'

    if do_pdark and expids is None:
        log.error('Must provide exposure ids if requesting pdark')
        raise ValueError('Must provide exposure ids if requesting pdark')

    ## If system name isn't specified, guess it
    if system_name is None:
        system_name = batch.default_system(jobdesc=jobdesc)

    batch_config = batch.get_config(system_name)

    ## Get number of mpi workers
    nranks, nodes, runtime = determine_resources(ncameras, jobdesc=jobdesc,
                                                 queue=queue, nexps=nexps,
                                                 system_name=system_name)

    threads_on_node = batch_config['cores_per_node'] * batch_config['threads_per_core']
    script_body = ""
    # Run nightlybias first
    if do_biasnight:
        ## One rank for each camera
        bias_nranks = ncameras
        ## srun won't split a ranks across nodes, so for ranks that aren't evenly split
        ## across nodes, make sure largest rank count with number of threads
        ## will still fit in a single node
        if nodes > 1 and bias_nranks % nodes != 0:
            largest_nranks_on_node = np.ceil(float(bias_nranks)/float(nodes))
            bias_threads_per_rank = int(np.floor(threads_on_node / largest_nranks_on_node))
        else:
            tot_threads = nodes * threads_on_node
            bias_threads_per_rank = int(np.floor(tot_threads // bias_nranks))

        if bias_nranks * bias_threads_per_rank > nodes * threads_on_node:
            assertstring = f"Requested {bias_nranks} ranks with {bias_threads_per_rank} threads per rank on " \
                           + f"{nodes} nodes with {threads_on_node} threads per node exceeds available threads ({nodes*threads_on_node})"
            log.critical(assertstring)
            raise AssertionError(assertstring)

        cmd = f'desi_proc --cameras {camword} -n {night} --nightlybias --mpi'
        cmd += f' --starttime $(date +%s) --timingfile {timingfile}'

        script_body += wrap_command_for_script(cmd, nodes, ntasks=bias_nranks, threads_per_task=bias_threads_per_rank, stepname='biasnight')

    # Then pdarks
    if do_pdark:
        ## if fewer than one-to-one assign more than one core to each rank (min of batch_config['threads_per_core']
        ## since we don't use threads)
        ## srun won't split a rank across nodes, so for ranks that aren't evenly split
        ## across nodes, make sure largest rank count with number of threads
        ## will still fit in a single node
        if nodes > 1 and nranks % nodes != 0:
            largest_nranks_on_node = np.ceil(float(nranks)/float(nodes))
            dark_threads_per_rank = int(np.floor(threads_on_node / largest_nranks_on_node))
        else:
            tot_threads = nodes * threads_on_node
            dark_threads_per_rank = int(np.floor(nodes*batch_config['cores_per_node']*batch_config['threads_per_core'] // nranks))

        if nranks * dark_threads_per_rank > nodes * threads_on_node:
            assertstring = f"Requested {nranks} ranks with {dark_threads_per_rank} threads per rank on " \
                           + f"{nodes} nodes with {threads_on_node} threads per node exceeds available threads ({nodes*threads_on_node})"
            log.critical(assertstring)
            raise AssertionError(assertstring)

        cmd = f'desi_preproc_darks -n {night} --expids={",".join(expids.astype(str))} --camword={camword} --mpi'
        script_body += wrap_command_for_script(cmd, nodes, ntasks=nranks, threads_per_task=dark_threads_per_rank, stepname='pdark')

    script_body += wrapup_for_script()
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
        fx.write(f'cd {batchdir}\n')

        fx.write(script_body)

    print('Wrote {}'.format(scriptpathname))
    print('logfile will be {}/{}-JOBID.log\n'.format(batchdir, jobname))

    return scriptpathname


def create_ccdcalib_batch_script(night, expids, camword='a0123456789',
                                 do_darknight=False, do_badcolumn=False,
                                 do_ctecorr=False, n_nights_before=None, n_nights_after=None,
                                 dark_expid=None, cte_expids=None,
                                 queue=None, system_name=None):
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

    ## Default to regular queue
    if queue is	None:
        queue =	'regular'

    scriptpathname = get_desi_proc_batch_file_pathname(night=night, exp=expids,
                                                   jobdesc=jobdesc, cameras=camword)
    scriptpathname += '.slurm'
    cameras = decode_camword(camword)
    ncameras = len(cameras)
    nexps = len(expids) if expids is not None else 1
    batchdir = os.path.dirname(scriptpathname)
    os.makedirs(batchdir, exist_ok=True)
    jobname = os.path.basename(scriptpathname).removesuffix('.slurm')
    timingfile = f'{jobname}-timing-$SLURM_JOBID.json'

    ## If system name isn't specified, guess it
    if system_name is None:
        system_name = batch.default_system(jobdesc=jobdesc)

    batch_config = batch.get_config(system_name)
    ntasks, nodes, runtime = determine_resources(ncameras, jobdesc='ccdcalib',
                                                 queue=queue, nexps=nexps,
                                                 system_name=system_name)
    threads_on_node = batch_config['cores_per_node'] * batch_config['threads_per_core']
    threads_per_task = int(np.floor((nodes*threads_on_node) / ntasks))
    script_body = ""
    # Run nightlybias first
    if do_darknight:
        cmd = f'desi_compute_dark_night --reference-night={night} --camword={camword}'
        if n_nights_before is not None:
            cmd += f' --before={n_nights_before}'
        if n_nights_after is not None:
            cmd += f' --after={n_nights_after}'
        cmd += ' --mpi'
        ## darknight will hit memory limits if more than 10 are done on a
        ## single node simultaneously
        max_ranks_per_node = 10
        if float(ntasks)/float(nodes) > max_ranks_per_node:
            ## will need to run in multiple batches, so reduce the ntasks and add more runtime
            dn_ntasks = max_ranks_per_node*nodes #  concurrent ranks that won't hit memory limit issues
            dn_threads_per_task = int(np.floor(threads_on_node / max_ranks_per_node))
        else:
            dn_ntasks, dn_threads_per_task = ntasks, threads_per_task
        runtime += 7.*np.ceil(float(ntasks)/float(dn_ntasks)) ## each loop takes about 3-5 minutes, but add 7 each for contingency
        script_body += wrap_command_for_script(cmd, nodes, ntasks=dn_ntasks, threads_per_task=dn_threads_per_task, stepname='darknight')

    # Then pdarks
    if do_badcolumn:
        if dark_expid is None:
            dark_expid = expids[0]
        cmd = f'desi_proc -n {night} --cameras {camword} -e {dark_expid} --mpi'
        cmd += f' --starttime $(date +%s) --timingfile {timingfile}'
        script_body += wrap_command_for_script(cmd, nodes, ntasks=ntasks, threads_per_task=threads_per_task, stepname='badcolumn')

    if do_ctecorr:
        if cte_expids is None:
            if do_darknight or do_badcolumn:
                cte_expids = expids[1:]
            else:
                cte_expids = expids
        cte_expstr = ','.join(np.array(cte_expids).astype(str))
        cmd = f"desi_fit_cte_night -n {night} -c {camword} -e {cte_expstr}"
        script_body += wrap_command_for_script(cmd, nodes, ntasks=ntasks, threads_per_task=threads_per_task, stepname='ctecorr')

    script_body += wrapup_for_script()
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
        fx.write(f'cd {batchdir}\n')

        fx.write(script_body)

    print('Wrote {}'.format(scriptpathname))
    print('logfile will be {}/{}-JOBID.log\n'.format(batchdir, jobname))

    return scriptpathname


def get_calibration_bundle_step_resources(step_jobdesc, ncameras, queue=None,
                                          system_name=None):
    """
    Determine the resources for one exposure step inside a calibration bundle.

    Every bundle step runs on a single node so that the bundle's allocation is
    simply one node per concurrent exposure.

    Args:
        step_jobdesc (str): resource class of the step, 'ARC', 'FLAT', or
            'CTEFLAT'.
        ncameras (int): number of cameras this step will process.
        queue (str, optional): the Slurm queue that will be used.
        system_name (str, optional): name of batch system, e.g. perlmutter-cpu.

    Returns:
        dict: The resources of this step, with keys:

        * 'nodes': int, number of nodes for this step, always 1.
        * 'ntasks': int, number of MPI ranks for this step.
        * 'threads_per_task': int, number of threads per rank.
        * 'runtime': float, estimated runtime of this step in minutes.
    """
    step_jobdesc = step_jobdesc.upper()
    ## determine_resources() guesses the system from an already uppercased
    ## jobdesc, which never matches its lowercase CPU-only list, so resolve the
    ## system here from the lowercase name instead
    if system_name is None:
        system_name = batch.default_system(jobdesc=step_jobdesc.lower())
    batch_config = batch.get_config(system_name)
    ncores, nodes, runtime = determine_resources(ncameras, step_jobdesc,
                                                 queue=queue,
                                                 system_name=system_name)
    threads_per_node = batch_config['threads_per_core'] * batch_config['cores_per_node']
    if step_jobdesc == 'ARC':
        ## determine_resources sizes an arc for a multi-node job. Reshape that
        ## layout onto a single node by giving the step the number of ranks
        ## that would have landed on one node, rounded down to a valid
        ## 20*k + 1 count for desispec.workflow.schedule.Schedule.
        ## NOTE: this gives fewer concurrent camera groups, and therefore more
        ## waves, than the single-exposure arc job the 45 minute ARC runtime
        ## constant was calibrated against. See the rollout notes before
        ## enabling bundles in the realtime queue.
        nodes, ranks_per_node, threads_per_task = adjust_arc_resources(
                ncores, nodes, batch_config)
        ntasks = (ranks_per_node // _ARC_RANKS_PER_GROUP) * _ARC_RANKS_PER_GROUP + 1
        threads_per_task = max(threads_per_node // ntasks, 1)
    else:
        ntasks = ncores
        threads_per_task = batch_config['threads_per_core']
    ## nodes is always 1; the multi-node layout determine_resources() returned
    ## has already been reshaped onto a single node above
    return {'nodes': 1, 'ntasks': ntasks,
            'threads_per_task': threads_per_task, 'runtime': runtime}


def _calibration_bundle_step_filenames(step_jobdesc, night, expid, camword,
                                       jobid_var='$SLURM_JOBID'):
    """
    Return the timing and log filenames of one calibration bundle exposure step.

    Args:
        step_jobdesc (str): the step's own descriptor, 'arc', 'flat', or
            'cteflat'. Note this is not the bundle's JOBDESC.
        night (str or int): the night the data was acquired.
        expid (int): the exposure id of this step.
        camword (str): the camword this step will process.
        jobid_var (str): shell expression giving the Slurm job id.

    Returns:
        tuple: (timingfile, logfile) basenames, both relative to the batch dir.
    """
    base = '{}-{}-{:08d}-{}'.format(step_jobdesc, night, int(expid), camword)
    return f'{base}-timing-{jobid_var}.json', f'{base}-{jobid_var}.log'


def create_calibration_bundle_batch_script(night, jobdesc, expids, camword,
                                           steps, joint_cmd=None,
                                           joint_camword=None, queue=None,
                                           system_name=None, runtime=None,
                                           batch_opts=None, concurrency=None,
                                           reduxdir=None):
    """
    Generate a SLURM batch script that processes every exposure of a nightly
    calibration bundle within a single allocation.

    Arc bundles launch one backgrounded srun per exposure and then run
    psfnight, normal-flat bundles throttle their exposures with GNU parallel
    and then run nightlyflat, and CTE-flat bundles run their exposures
    sequentially and have no joint fit.

    In all three cases every selected exposure is attempted even if a sibling
    fails, but a joint fit is never run over an incomplete set of exposures.

    Args:
        night (str or int): The night the data was acquired.
        jobdesc (str): The bundle job description: 'psfnight', 'nightlyflat',
            or 'cteflat'.
        expids (list of int or np.array): All exposure ids of the bundle, used
            for the script pathname. This is the full selected set, not the
            possibly smaller set of exposures still needing to be processed.
        camword (str): The full camword of the bundle, used for the script and
            job name so the pathname is stable under camera pruning.
        steps (list of dict): One entry per exposure that still needs to be
            processed, each with keys 'expid' (int), 'camword' (str), and
            'cmd' (str). The command is a complete desi_proc command line
            without the srun prefix, --starttime, or --timingfile.
        joint_cmd (str, optional): Complete desi_proc_joint_fit command line
            without the srun prefix, --starttime, or --timingfile. Required
            for 'psfnight' and 'nightlyflat', must be None for 'cteflat'.
        joint_camword (str, optional): The camword the joint fit will process.
            Defaults to camword.
        queue (str, optional): Queue to be used. Default is 'regular'.
        system_name (str, optional): name of batch system, e.g. perlmutter-cpu.
        runtime (int, optional): Force the wall clock request in minutes rather
            than deriving it from determine_resources().
        batch_opts (str, optional): Other options to give to the slurm batch
            scheduler (written into the script).
        concurrency (int, optional): Number of normal-flat exposure steps to
            run at once. Ignored by the other two bundle types. Default is
            the number of remaining steps divided by three.
        reduxdir (str, optional): base directory where run/scripts lives.

    Returns:
        str: The full path name for the batch script file written.
    """
    log = get_logger()
    jobdesc = str(jobdesc).lower()
    if jobdesc not in ('psfnight', 'nightlyflat', 'cteflat'):
        msg = f'Unknown calibration bundle jobdesc={jobdesc}'
        log.critical(msg)
        raise ValueError(msg)
    if jobdesc == 'cteflat' and joint_cmd is not None:
        msg = 'cteflat bundles have no joint fit, but a joint_cmd was given'
        log.critical(msg)
        raise ValueError(msg)
    if jobdesc != 'cteflat' and joint_cmd is None:
        msg = f'{jobdesc} bundles require a joint fit command'
        log.critical(msg)
        raise ValueError(msg)
    if len(steps) == 0 and joint_cmd is None:
        msg = f'{jobdesc} bundle has no exposure steps and no joint fit to run'
        log.critical(msg)
        raise ValueError(msg)

    ## Default to regular queue
    if queue is None:
        queue = 'regular'

    if joint_camword is None:
        joint_camword = camword

    ## If system name isn't specified, pick it based upon jobdesc
    if system_name is None:
        system_name = batch.default_system(jobdesc=jobdesc)

    batch_config = batch.get_config(system_name)

    ## Resource class of each exposure step, and the descriptor used in the
    ## per-step filenames. Note the step descriptor is not the bundle's JOBDESC.
    if jobdesc == 'psfnight':
        step_jobdesc, step_class, joint_class = 'arc', 'ARC', 'PSFNIGHT'
    elif jobdesc == 'nightlyflat':
        step_jobdesc, step_class, joint_class = 'flat', 'FLAT', 'NIGHTLYFLAT'
    else:
        step_jobdesc, step_class, joint_class = 'cteflat', 'CTEFLAT', None

    ## Deterministic exposure ordering
    steps = sorted(steps, key=lambda step: int(step['expid']))
    nsteps = len(steps)

    scriptpathname = get_desi_proc_batch_file_pathname(night=night, exp=expids,
                                                       jobdesc=jobdesc,
                                                       cameras=camword,
                                                       reduxdir=reduxdir)
    scriptpathname += '.slurm'
    batchdir = os.path.dirname(scriptpathname)
    os.makedirs(batchdir, exist_ok=True)
    jobname = os.path.basename(scriptpathname).removesuffix('.slurm')

    ## Resources of each remaining exposure step. These are computed per step
    ## because camera pruning can leave the steps with different camwords.
    step_resources = []
    for step in steps:
        ncam = len(decode_camword(step['camword']))
        step_resources.append(get_calibration_bundle_step_resources(
                step_class, ncam, queue=queue, system_name=system_name))

    nodes_per_step = max([res['nodes'] for res in step_resources], default=1)
    step_runtimes = [res['runtime'] for res in step_resources]

    ## Resources of the joint fit, if there is one
    joint_nodes, joint_ntasks, joint_runtime = 1, 0, 0.
    joint_threads = batch_config['threads_per_core']
    if joint_cmd is not None:
        njointcams = len(decode_camword(joint_camword))
        joint_ntasks, joint_nodes, joint_runtime = determine_resources(
                njointcams, joint_class, queue=queue, system_name=system_name)

    ## Size the allocation. Every bundle requests one node per concurrent
    ## exposure step, with at least enough nodes for the joint fit.
    if jobdesc == 'cteflat':
        ## CTE flats run one at a time on a single node
        nodes, concurrency = nodes_per_step, 1
        npasses = nsteps
        est_runtime = float(np.sum(step_runtimes))
    else:
        cap = max_nodes_for_jobdesc(joint_class)
        if joint_nodes > cap:
            msg = (f'{jobdesc} joint fit requires {joint_nodes} nodes, which '
                   + f'exceeds the {cap} node cap for {joint_class}')
            log.critical(msg)
            raise ValueError(msg)
        if jobdesc == 'psfnight':
            ## every selected arc should be processed at the same time
            requested_concurrency = max(nsteps, 1)
        elif concurrency is not None:
            requested_concurrency = max(int(concurrency), 1)
        else:
            ## throttle flats so that the allocation stays small enough for
            ## two jobs to run together in the 10 node realtime queue
            requested_concurrency = max(1, nsteps // 3)
        nodes = min(max(requested_concurrency * nodes_per_step, joint_nodes), cap)
        concurrency = max(1, min(requested_concurrency, nodes // nodes_per_step))
        if 0 < concurrency < nsteps and jobdesc == 'psfnight':
            log.warning(f'{jobdesc} bundle has {nsteps} arc exposures but the '
                        + f'{cap} node cap for {joint_class} only allows '
                        + f'{concurrency} at a time, so the arcs will be '
                        + 'processed in multiple passes.')
        npasses = int(np.ceil(float(nsteps) / float(concurrency)))
        est_runtime = npasses * max(step_runtimes, default=0.)
        est_runtime += joint_runtime

    if runtime is not None:
        est_runtime = runtime

    ## Never request a degenerate wall clock, e.g. for a bundle that somehow
    ## has neither exposure steps nor a joint fit runtime
    est_runtime = max(float(est_runtime), 5.)
    runtime_hh = int(est_runtime // 60)
    runtime_mm = int(est_runtime % 60)

    ## GPU systems run the exposure steps through the MPS wrapper and request
    ## GPUs per step; the joint step inherits the allocation's GPUs
    mps_wrapper, step_gpu_opt = '', ''
    if system_name == 'perlmutter-gpu':
        mps_wrapper = 'desi_mps_wrapper'
        step_gpu_opt = '--gpus-per-node={} '.format(batch_config['gpus_per_node'])

    def _step_srun(step, resources, jobid_var='$SLURM_JOBID'):
        """Return (srun_command, logfile) for one exposure step"""
        stepnodes = resources['nodes']
        ntasks = resources['ntasks']
        threads = resources['threads_per_task']
        timingfile, logfile = _calibration_bundle_step_filenames(
                step_jobdesc, night, step['expid'], step['camword'],
                jobid_var=jobid_var)
        cmd = step['cmd']
        if jobdesc == 'nightlyflat':
            ## STARTTIMESTR is expanded by the shell parallel spawns per job
            cmd += ' ${STARTTIMESTR}'
        else:
            cmd += ' --starttime $(date +%s)'
        cmd += f' --timingfile {timingfile}'
        srun = (f'srun {step_gpu_opt}-N {stepnodes} -n {ntasks} -c {threads} '
                + '--cpu-bind=cores ' + mps_wrapper + f' {cmd}')
        return srun, logfile

    script_body = ''
    if nsteps == 0:
        ## Every exposure already has its per-exposure products, so the script
        ## contains only the joint fit that creates the nightly product
        script_body += '\n## All individual exposures already have their expected outputs,\n'
        script_body += f'## so this script only runs {jobdesc}\n'
    elif jobdesc == 'psfnight':
        script_body += '\n## Launch every arc exposure at once, one node each, and collect the\n'
        script_body += '## PIDs so that each one can be waited on individually below\n'
        script_body += 'pids=""\n'
        for step, resources in zip(steps, step_resources):
            srun, logfile = _step_srun(step, resources)
            script_body += f"\n# Process arc exposure {step['expid']}\n"
            threads = resources['threads_per_task']   # could be different per exposure if different number of cameras
            script_body += f'export OMP_NUM_THREADS={threads}\n'
            script_body += f'echo Running {srun}\n'
            script_body += f'echo Logging to {logfile}\n'
            script_body += f'{srun} > {logfile} 2>&1 &\n'
            script_body += 'pids="$pids $!"\n'
        script_body += '\n## Wait for every exposure, counting failures. A failed arc must not cut\n'
        script_body += '## its siblings short, but psfnight must not be built from a subset.\n'
        script_body += 'nfail=0\n'
        script_body += 'for pid in $pids; do\n'
        script_body += '    wait $pid || nfail=$((nfail+1))\n'
        script_body += 'done\n'
        script_body += 'if [ $nfail -ne 0 ]; then\n'
        script_body += f'  echo FAILED: $nfail of {nsteps} arc exposures failed,' \
                       + ' not running psfnight at $(date)\n'
        script_body += '  exit 1\n'
        script_body += 'fi\n'
        script_body += 'echo Successfully completed arcs at $(date)\n'
        script_body += '\n# Switch back to num threads of 1\n'
        script_body += 'export OMP_NUM_THREADS=1\n'
    elif jobdesc == 'nightlyflat':
        joblog = 'joblog-flats-{}-{:08d}-$SLURM_JOBID.log'.format(
                night, int(np.min(expids)))
        script_body += '\n# Process individual flat exposures\n'
        script_body += '## Run with "$SLURM_JOB_NUM_NODES" workers, each with 1 node of resources\n'
        script_body += '## -v prints the command before executing it, -j is the number of workers\n'
        script_body += '## --joblog saves basic timing of the jobs\n'
        script_body += "## after ':::' is the list of commands to be run\n"
        script_body += '## STARTTIMESTR is single quoted so that $(date +%s) reaches parallel as a\n'
        script_body += '## literal and is evaluated by the shell parallel spawns for each job,\n'
        script_body += '## giving every exposure its own start time\n'
        script_body += "STARTTIMESTR='--starttime $(date +%s)'\n"
        script_body += f'parallel -v -j "$SLURM_JOB_NUM_NODES" --joblog "{joblog}" ::: \\\n'
        srun_strings = []
        for step, resources in zip(steps, step_resources):
            srun, logfile = _step_srun(step, resources,
                                       jobid_var='${SLURM_JOBID}')
            ## parallel runs each string through $PARALLEL_SHELL/$SHELL, which
            ## isn't guaranteed to be bash, so use POSIX redirection
            srun_strings.append(f'"{srun} > {logfile} 2>&1"')
        script_body += ' \\\n  '.join(srun_strings) + '\n'
        script_body += 'nfail=$?\n'
        script_body += '\n## By default parallel exits with the number of failed jobs. Capture that\n'
        script_body += '## into nfail before the test below, since [ ] would overwrite $?.\n'
        script_body += '## Every flat is attempted, but nightlyflat must not be built from a subset.\n'
        script_body += 'if [ $nfail -ne 0 ]; then\n'
        script_body += '  echo FAILED to process $nfail individual flats,' \
                       + ' not running nightlyflat at $(date)\n'
        script_body += '  exit 1\n'
        script_body += 'fi\n'
        script_body += 'echo Successfully completed flats at $(date)\n'
    else:
        script_body += '\n## Note each exposure uses its own camword. CTE flats have no joint fit,\n'
        script_body += '## so there is no common camera set to intersect against; a camera missing\n'
        script_body += '## from one CTE exposure must not remove it from the others.\n'
        script_body += '## A failed exposure must not stop the ones after it, so accumulate the\n'
        script_body += '## failures rather than exiting between steps.\n'
        script_body += 'nfail=0\n'
        for step, resources in zip(steps, step_resources):
            srun, logfile = _step_srun(step, resources)
            script_body += f"\n# Process exposure {step['expid']}\n"
            script_body += f'echo Running {srun}\n'
            script_body += f'{srun} > {logfile} 2>&1\n'
            script_body += 'if [ $? -ne 0 ]; then\n'
            script_body += '  nfail=$((nfail+1))\n'
            script_body += f"  echo FAILED: cteflat {step['expid']} at $(date)\n"
            script_body += 'else\n'
            script_body += f"  echo cteflat {step['expid']} succeeded at $(date)\n"
            script_body += 'fi\n'

    if joint_cmd is not None:
        joint_timingfile = f'{jobname}-timing-$SLURM_JOBID.json'
        cmd = joint_cmd + ' --starttime $(date +%s)'
        cmd += f' --timingfile {joint_timingfile}'
        srun = (f'srun -N {joint_nodes} -n {joint_ntasks} -c {joint_threads} '
                + '--cpu-bind=cores ' + mps_wrapper + f' {cmd}')
        script_body += f'\n# Process {jobdesc}\n'
        script_body += f'echo Running {srun}\n'
        script_body += f'{srun}\n'
        script_body += '\nif [ $? -eq 0 ]; then\n'
        script_body += '  echo SUCCESS: done at $(date)\n'
        script_body += 'else\n'
        script_body += '  echo FAILED: done at $(date)\n'
        script_body += '  exit 1\n'
        script_body += 'fi\n'
    else:
        script_body += '\n## No joint fit for CTE flats; the accumulated failure count is the only gate.\n'
        script_body += 'if [ $nfail -eq 0 ]; then\n'
        script_body += '  echo SUCCESS: done at $(date)\n'
        script_body += 'else\n'
        script_body += f'  echo FAILED: $nfail of {nsteps} CTE exposures failed:' \
                       + ' done at $(date)\n'
        script_body += '  exit 1\n'
        script_body += 'fi\n'

    with open(scriptpathname, 'w') as fx:
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

        if jobdesc == 'psfnight':
            fx.write('## Processing individual arcs, one node each, then psfnight\n')
        elif jobdesc == 'nightlyflat':
            fx.write('## Processing individual flats, {} at a time, then nightlyflat\n'.format(concurrency))
        else:
            fx.write('## Processing CTE flats serially on a single node.\n')
            fx.write('## Unlike the arc and normal-flat bundles there is no joint fit, and\n')
            fx.write('## exposures are run one at a time rather than concurrently, so the whole\n')
            fx.write('## job needs only the resources of a single exposure.\n')

        fx.write('echo Starting job $SLURM_JOB_ID on $(hostname) at $(date)\n')
        fx.write(f'cd {batchdir}\n')
        fx.write('export OMP_NUM_THREADS=1\n')
        if system_name == 'perlmutter-gpu':
            fx.write('export MPICH_GPU_SUPPORT_ENABLED=1\n')

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
        nodes, cores_per_node, threads_per_core = adjust_arc_resources(
                ncores, nodes, batch_config)

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
        fx.write(f'cd {batchdir}\n')

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
                srun = (f'srun -N {nodes} -n {ncores} -c {threads_per_core} --cpu-bind=cores '
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
        fx.write(f'cd {batchdir}\n')

        mps_wrapper=''
        if system_name == 'perlmutter-gpu':
            fx.write("export MPICH_GPU_SUPPORT_ENABLED=1\n")
            mps_wrapper='desi_mps_wrapper'

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
