
import sys, os, glob
import json
from astropy.io import fits
from astropy.table import Table, join
import numpy as np

import time, datetime
from collections import OrderedDict
import subprocess
from copy import deepcopy

from desispec.scripts.tile_redshifts import generate_tile_redshift_scripts, get_tile_redshift_script_pathname, \
                                            get_tile_redshift_relpath, get_tile_redshift_script_suffix
from desispec.workflow.queue import get_resubmission_states, update_from_queue
from desispec.workflow.timing import what_night_is_it
from desispec.workflow.desi_proc_funcs import get_desi_proc_batch_file_pathname, create_desi_proc_batch_script, \
                                              get_desi_proc_batch_file_path, get_desi_proc_tilenight_batch_file_pathname, \
                                              create_desi_proc_tilenight_batch_script
from desispec.workflow.utils import pathjoin, sleep_and_report
from desispec.workflow.tableio import write_table
from desispec.workflow.proctable import table_row_to_dict
from desiutil.log import get_logger

from desispec.io import findfile, specprod_root
from desispec.io.util import decode_camword, create_camword, difference_camwords, \
                             camword_to_spectros, camword_union

#################################################
############## Misc Functions ###################
#################################################
def night_to_starting_iid(night=None):
    """
    Creates an internal ID for a given night. The resulting integer is an 8 digit number.
    The digits are YYMMDDxxx where YY is the years since 2000, MM and DD are the month and day. xxx are 000,
    and are incremented for up to 1000 unique job ID's for a given night.

    Args:
        night, str or int. YYYYMMDD of the night to get the starting internal ID for.

    Returns:
        internal_id, int. 9 digit number consisting of YYMMDD000. YY is years after 2000, MMDD is month and day.
                          000 being the starting job number (0).
    """
    if night is None:
        night = what_night_is_it()
    night = int(night)
    internal_id = (night - 20000000) * 1000
    return internal_id



#################################################
############ Script Functions ###################
#################################################
def batch_script_name(prow):
    """
    Wrapper script that takes a processing table row (or dictionary with NIGHT, EXPID, JOBDESC, PROCCAMWORD defined)
    and determines the script file pathname as defined by desi_proc's helper functions.

    Args:
        prow, Table.Row or dict. Must include keyword accessible definitions for 'NIGHT', 'EXPID', 'JOBDESC', and 'PROCCAMWORD'.

    Returns:
        scriptfile, str. The complete pathname to the script file, as it is defined within the desi_proc ecosystem.
    """
    expids = prow['EXPID']
    if len(expids) == 0:
        expids = None
    if prow['JOBDESC'] == 'tilenight':
        pathname = get_desi_proc_tilenight_batch_file_pathname(night = prow['NIGHT'], tileid=prow['TILEID'])
    else:
        pathname = get_desi_proc_batch_file_pathname(night = prow['NIGHT'], exp=expids, \
                                             jobdesc=prow['JOBDESC'], cameras=prow['PROCCAMWORD'])
    scriptfile =  pathname + '.slurm'
    return scriptfile

def check_for_outputs_on_disk(prow, resubmit_partial_complete=True):
    """
    Args:
        prow, Table.Row or dict. Must include keyword accessible definitions for processing_table columns found in
                                 desispect.workflow.proctable.get_processing_table_column_defs()
        resubmit_partial_complete, bool. Default is True. Must be used with check_for_outputs=True. If this flag is True,
                                         jobs with some prior data are pruned using PROCCAMWORD to only process the
                                         remaining cameras not found to exist.

    Returns:
        prow, Table.Row or dict. The same prow type and keywords as input except with modified values updated to reflect
                                 the change in job status after creating and submitting the job for processing.
    """
    prow['STATUS'] = 'UNKNOWN'
    log = get_logger()

    job_to_file_map = {
            'prestdstar': 'sframe',
            'stdstarfit': 'stdstars',
            'poststdstar': 'cframe',
            'nightlybias': 'biasnight',
            'ccdcalib': 'badcolumns',
            'badcol': 'badcolumns',
            'arc': 'fitpsf',
            'flat': 'fiberflat',
            'psfnight': 'psfnight',
            'nightlyflat': 'fiberflatnight',
            'spectra': 'spectra_tile',
            'coadds': 'coadds_tile',
            'redshift': 'redrock_tile',
            }

    night = prow['NIGHT']
    if prow['JOBDESC'] in ['cumulative','pernight-v0','pernight','perexp']:
        filetype = 'redrock_tile'
    else:
        filetype = job_to_file_map[prow['JOBDESC']]
    orig_camword = prow['PROCCAMWORD']

    ## if spectro based, look for spectros, else look for cameras
    if prow['JOBDESC'] in ['stdstarfit','spectra','coadds','redshift']:
        ## Spectrograph based
        spectros = camword_to_spectros(prow['PROCCAMWORD'])
        n_desired = len(spectros)
        ## Suppress outputs about using tile based files in findfile if only looking for stdstarfits
        if prow['JOBDESC'] == 'stdstarfit':
            tileid = None
        else:
            tileid = prow['TILEID']
        expid = prow['EXPID'][0]
        existing_spectros = []
        for spectro in spectros:
            if os.path.exists(findfile(filetype=filetype, night=night, expid=expid, spectrograph=spectro, tile=tileid)):
                existing_spectros.append(spectro)
        completed = (len(existing_spectros) == n_desired)
        if not completed and resubmit_partial_complete and len(existing_spectros) > 0:
            existing_camword = 'a' + ''.join([str(spec) for spec in sorted(existing_spectros)])
            prow['PROCCAMWORD'] = difference_camwords(prow['PROCCAMWORD'],existing_camword)
    elif prow['JOBDESC'] in ['cumulative','pernight-v0','pernight','perexp']:
        ## Spectrograph based
        spectros = camword_to_spectros(prow['PROCCAMWORD'])
        n_desired = len(spectros)
        ## Suppress outputs about using tile based files in findfile if only looking for stdstarfits
        tileid = prow['TILEID']
        expid = prow['EXPID'][0]
        redux_dir = specprod_root()
        outdir = os.path.join(redux_dir,get_tile_redshift_relpath(tileid,group=prow['JOBDESC'],night=night,expid=expid))
        suffix = get_tile_redshift_script_suffix(tileid, group=prow['JOBDESC'], night=night, expid=expid)
        existing_spectros = []
        for spectro in spectros:
            if os.path.exists(os.path.join(outdir, f"redrock-{spectro}-{suffix}.fits")):
                existing_spectros.append(spectro)
        completed = (len(existing_spectros) == n_desired)
        if not completed and resubmit_partial_complete and len(existing_spectros) > 0:
            existing_camword = 'a' + ''.join([str(spec) for spec in sorted(existing_spectros)])
            prow['PROCCAMWORD'] = difference_camwords(prow['PROCCAMWORD'],existing_camword)
    else:
        ## Otheriwse camera based
        cameras = decode_camword(prow['PROCCAMWORD'])
        n_desired = len(cameras)
        if len(prow['EXPID']) > 0:
            expid = prow['EXPID'][0]
        else:
            expid = None
        if len(prow['EXPID']) > 1 and prow['JOBDESC'] not in ['psfnight','nightlyflat']:
            log.warning(f"{prow['JOBDESC']} job with exposure(s) {prow['EXPID']}. This job type only makes " +
                     f"sense with a single exposure. Proceeding with {expid}.")
        missing_cameras = []
        for cam in cameras:
            if not os.path.exists(findfile(filetype=filetype, night=night, expid=expid, camera=cam)):
                missing_cameras.append(cam)
        completed = (len(missing_cameras) == 0)
        if not completed and resubmit_partial_complete and len(missing_cameras) < n_desired:
            prow['PROCCAMWORD'] = create_camword(missing_cameras)

    if completed:
        prow['STATUS'] = 'COMPLETED'
        log.info(f"{prow['JOBDESC']} job with exposure(s) {prow['EXPID']} already has " +
                 f"the desired {n_desired} {filetype}'s. Not submitting this job.")
    elif resubmit_partial_complete and orig_camword != prow['PROCCAMWORD']:
        log.info(f"{prow['JOBDESC']} job with exposure(s) {prow['EXPID']} already has " +
                 f"some {filetype}'s. Submitting smaller camword={prow['PROCCAMWORD']}.")
    elif not resubmit_partial_complete:
        log.info(f"{prow['JOBDESC']} job with exposure(s) {prow['EXPID']} doesn't have all " +
                 f"{filetype}'s and resubmit_partial_complete=False. "+
                 f"Submitting full camword={prow['PROCCAMWORD']}.")
    else:
        log.info(f"{prow['JOBDESC']} job with exposure(s) {prow['EXPID']} has no " +
                 f"existing {filetype}'s. Submitting full camword={prow['PROCCAMWORD']}.")
    return prow

def create_and_submit(prow, queue='realtime', reservation=None, dry_run=0, joint=False,
                      strictly_successful=False, check_for_outputs=True, resubmit_partial_complete=True,
                      system_name=None,use_specter=False):
    """
    Wrapper script that takes a processing table row and three modifier keywords, creates a submission script for the
    compute nodes, and then submits that script to the Slurm scheduler with appropriate dependencies.

    Args:
        prow, Table.Row or dict. Must include keyword accessible definitions for processing_table columns found in
                                 desispect.workflow.proctable.get_processing_table_column_defs()
        queue, str. The name of the NERSC Slurm queue to submit to. Default is the realtime queue.
        reservation: str. The reservation to submit jobs to. If None, it is not submitted to a reservation.
        dry_run, int. If nonzero, this is a simulated run. If dry_run=1 the scripts will be written or submitted. If
                      dry_run=2, the scripts will not be writter or submitted. Logging will remain the same
                      for testing as though scripts are being submitted. Default is 0 (false).
        joint, bool. Whether this is a joint fitting job (the job involves multiple exposures) and therefore needs to be
                     run with desi_proc_joint_fit. Default is False.
        strictly_successful, bool. Whether all jobs require all inputs to have succeeded. For daily processing, this is
                                   less desirable because e.g. the sciences can run with SVN default calibrations rather
                                   than failing completely from failed calibrations. Default is False.
        check_for_outputs, bool. Default is True. If True, the code checks for the existence of the expected final
                                 data products for the script being submitted. If all files exist and this is True,
                                 then the script will not be submitted. If some files exist and this is True, only the
                                 subset of the cameras without the final data products will be generated and submitted.
        resubmit_partial_complete, bool. Default is True. Must be used with check_for_outputs=True. If this flag is True,
                                         jobs with some prior data are pruned using PROCCAMWORD to only process the
                                         remaining cameras not found to exist.
        system_name (str): batch system name, e.g. cori-haswell or perlmutter-gpu
        use_specter, bool, optional. Default is False. If True, use specter, otherwise use gpu_specter by default.

    Returns:
        prow, Table.Row or dict. The same prow type and keywords as input except with modified values updated to reflect
                                 the change in job status after creating and submitting the job for processing.

    Note:
        This modifies the input. Though Table.Row objects are generally copied on modification, so the change to the
        input object in memory may or may not be changed. As of writing, a row from a table given to this function will
        not change during the execution of this function (but can be overwritten explicitly with the returned row if desired).
    """
    orig_prow = prow.copy()
    if check_for_outputs:
        prow = check_for_outputs_on_disk(prow, resubmit_partial_complete)
        if prow['STATUS'].upper() == 'COMPLETED':
            return prow
    prow = create_batch_script(prow, queue=queue, dry_run=dry_run, joint=joint, system_name=system_name, use_specter=use_specter)
    prow = submit_batch_script(prow, reservation=reservation, dry_run=dry_run, strictly_successful=strictly_successful)
    ## If resubmitted partial, the PROCCAMWORD and SCRIPTNAME will correspond to the pruned values. But we want to
    ## retain the full job's value, so get those from the old job.
    if resubmit_partial_complete:
        prow['PROCCAMWORD'] = orig_prow['PROCCAMWORD']
        prow['SCRIPTNAME'] = orig_prow['SCRIPTNAME']
    return prow

def desi_proc_command(prow, queue=None, system_name, use_specter):
    """
    Wrapper script that takes a processing table row (or dictionary with NIGHT, EXPID, OBSTYPE, JOBDESC, PROCCAMWORD defined)
    and determines the proper command line call to process the data defined by the input row/dict.

    Args:
        prow, Table.Row or dict. Must include keyword accessible definitions for 'NIGHT', 'EXPID', 'JOBDESC', and 'PROCCAMWORD'.
        queue, str. The name of the NERSC Slurm queue to submit to. Default is None (which leaves it to the desi_proc default).
        system_name: batch system name, e.g. cori-haswell, cori-knl, perlmutter-gpu
        use_specter, bool, optional. Default is False. If True, use specter, otherwise use gpu_specter by default.

    Returns:
        cmd, str. The proper command to be submitted to desi_proc to process the job defined by the prow values.
    """
    cmd = 'desi_proc'
    cmd += ' --batch'
    cmd += ' --nosubmit'
    if queue is not None:
        cmd += f' -q {queue}'
    if prow['OBSTYPE'].lower() == 'science':
        if prow['JOBDESC'] == 'prestdstar':
            cmd += ' --nostdstarfit --nofluxcalib'
        elif prow['JOBDESC'] == 'poststdstar':
            cmd += ' --noprestdstarfit --nostdstarfit'
    elif prow['JOBDESC'] in ['nightlybias', 'ccdcalib']:
        cmd += ' --nightlybias'
    elif prow['JOBDESC'] in ['flat'] and not use_specter:
        cmd += ' --gpuspecter'
        if system_name=="perlmutter-gpu":
            cmd += ' --gpuextract'
    pcamw = str(prow['PROCCAMWORD'])
    cmd += f" --cameras={pcamw} -n {prow['NIGHT']}"
    if len(prow['EXPID']) > 0:
        cmd += f" -e {prow['EXPID'][0]}"
    if prow['BADAMPS'] != '':
        cmd += ' --badamps={}'.format(prow['BADAMPS'])
    return cmd

def desi_proc_joint_fit_command(prow, queue=None):
    """
    Wrapper script that takes a processing table row (or dictionary with NIGHT, EXPID, OBSTYPE, PROCCAMWORD defined)
    and determines the proper command line call to process the data defined by the input row/dict.

    Args:
        prow, Table.Row or dict. Must include keyword accessible definitions for 'NIGHT', 'EXPID', 'JOBDESC', and 'PROCCAMWORD'.
        queue, str. The name of the NERSC Slurm queue to submit to. Default is None (which leaves it to the desi_proc default).

    Returns:
        cmd, str. The proper command to be submitted to desi_proc_joint_fit to process the job defined by the prow values.
    """
    cmd = 'desi_proc_joint_fit'
    cmd += ' --batch'
    cmd += ' --nosubmit'
    if queue is not None:
        cmd += f' -q {queue}'

    descriptor = prow['OBSTYPE'].lower()

    night = prow['NIGHT']
    specs = str(prow['PROCCAMWORD'])
    expid_str = ','.join([str(eid) for eid in prow['EXPID']])

    cmd += f' --obstype {descriptor}'
    cmd += f' --cameras={specs} -n {night}'
    if len(expid_str) > 0:
        cmd += f' -e {expid_str}'
    return cmd

def create_batch_script(prow, queue='realtime', dry_run=0, joint=False, system_name=None, use_specter=False):
    """
    Wrapper script that takes a processing table row and three modifier keywords and creates a submission script for the
    compute nodes.

    Args:
        prow, Table.Row or dict. Must include keyword accessible definitions for processing_table columns found in
                                 desispect.workflow.proctable.get_processing_table_column_defs()
        queue, str. The name of the NERSC Slurm queue to submit to. Default is the realtime queue.
        dry_run, int. If nonzero, this is a simulated run. If dry_run=1 the scripts will be written but not submitted.
                      If dry_run=2, the scripts will not be written nor submitted. Logging will remain the same
                      for testing as though scripts are being submitted. Default is 0 (false).
        joint, bool. Whether this is a joint fitting job (the job involves multiple exposures) and therefore needs to be
                     run with desi_proc_joint_fit when not using tilenight. Default is False.
        system_name (str): batch system name, e.g. cori-haswell or perlmutter-gpu
        use_specter, bool, optional. Default is False. If True, use specter, otherwise use gpu_specter by default.

    Returns:
        prow, Table.Row or dict. The same prow type and keywords as input except with modified values updated values for
                                 scriptname.

    Note:
        This modifies the input. Though Table.Row objects are generally copied on modification, so the change to the
        input object in memory may or may not be changed. As of writing, a row from a table given to this function will
        not change during the execution of this function (but can be overwritten explicitly with the returned row if desired).
    """
    log = get_logger()
    if prow['JOBDESC'] in ['perexp','pernight','pernight-v0','cumulative']:
        if dry_run > 1:
            scriptpathname = get_tile_redshift_script_pathname(tileid=prow['TILEID'],group=prow['JOBDESC'],
                                                               night=prow['NIGHT'], expid=prow['EXPID'][0])

            log.info("Output file would have been: {}".format(scriptpathname))
        else:
            #- run zmtl for cumulative redshifts but not others
            run_zmtl = (prow['JOBDESC'] == 'cumulative')

            scripts, failed_scripts = generate_tile_redshift_scripts(tileid=prow['TILEID'], group=prow['JOBDESC'],
                                                                     night=[prow['NIGHT']], expid=prow['EXPID'],
                                                                     run_zmtl=run_zmtl,
                                                                     batch_queue=queue, system_name=system_name,
                                                                     nosubmit=True)
            if len(failed_scripts) > 0:
                log.error(f"Redshifts failed for group={prow['JOBDESC']}, night={prow['NIGHT']}, "+
                          f"tileid={prow['TILEID']}, expid={prow['EXPID']}.")
                log.info(f"Returned failed scriptname is {failed_scripts}")
            elif len(scripts) > 1:
                log.error(f"More than one redshifts returned for group={prow['JOBDESC']}, night={prow['NIGHT']}, "+
                          f"tileid={prow['TILEID']}, expid={prow['EXPID']}.")
                log.info(f"Returned scriptnames were {scripts}")
            else:
                scriptpathname = scripts[0]
    else:
        if prow['JOBDESC'] != 'tilenight':
            if joint:
                cmd = desi_proc_joint_fit_command(prow, queue=queue)
            else:
                cmd = desi_proc_command(prow, queue=queue, system_name, use_specter)
        if dry_run > 1:
            scriptpathname = batch_script_name(prow)
            log.info("Output file would have been: {}".format(scriptpathname))
            if prow['JOBDESC'] != 'tilenight':
                log.info("Command to be run: {}".format(cmd.split()))
        else:
            expids = prow['EXPID']
            if len(expids) == 0:
                expids = None
            gpuspecter = ((not use_specter) and prow['JOBDESC'] in ['science', 'prestdstar', 'tilenight'])
            gpuextract = (gpuspecter and system_name=="perlmutter-gpu")
            if prow['JOBDESC'] == 'tilenight':
                log.info("Creating tilenight script for tile {}".format(prow['TILEID']))
                ncameras = len(decode_camword(prow['PROCCAMWORD']))
                scriptpathname = create_desi_proc_tilenight_batch_script(
                                                               night=prow['NIGHT'], exp=expids,
                                                               tileid=prow['TILEID'],
                                                               ncameras=ncameras,
                                                               queue=queue,
                                                               mpistdstars=True,
                                                               gpuspecter=gpuspecter,
                                                               gpuextract=gpuextract,
                                                               system_name=system_name)
            else:
                log.info("Running: {}".format(cmd.split()))
                scriptpathname = create_desi_proc_batch_script(
                                                               night=prow['NIGHT'], exp=expids,
                                                               cameras=prow['PROCCAMWORD'],
                                                               jobdesc=prow['JOBDESC'],
                                                               queue=queue, cmdline=cmd,
                                                               gpuspecter=gpuspecter,
                                                               gpuextract=gpuextract,
                                                               system_name=system_name)
    log.info("Outfile is: {}".format(scriptpathname))
    prow['SCRIPTNAME'] = os.path.basename(scriptpathname)
    return prow


def submit_batch_script(prow, dry_run=0, reservation=None, strictly_successful=False):
    """
    Wrapper script that takes a processing table row and three modifier keywords and submits the scripts to the Slurm
    scheduler.

    Args:
        prow, Table.Row or dict. Must include keyword accessible definitions for processing_table columns found in
                                 desispect.workflow.proctable.get_processing_table_column_defs()
        dry_run, int. If nonzero, this is a simulated run. If dry_run=1 the scripts will be written or submitted. If
                      dry_run=2, the scripts will not be writter or submitted. Logging will remain the same
                      for testing as though scripts are being submitted. Default is 0 (false).
        reservation: str. The reservation to submit jobs to. If None, it is not submitted to a reservation.
        strictly_successful, bool. Whether all jobs require all inputs to have succeeded. For daily processing, this is
                                   less desirable because e.g. the sciences can run with SVN default calibrations rather
                                   than failing completely from failed calibrations. Default is False.

    Returns:
        prow, Table.Row or dict. The same prow type and keywords as input except with modified values updated values for
                                 scriptname.

    Note:
        This modifies the input. Though Table.Row objects are generally copied on modification, so the change to the
        input object in memory may or may not be changed. As of writing, a row from a table given to this function will
        not change during the execution of this function (but can be overwritten explicitly with the returned row if desired).
    """
    log = get_logger()
    dep_qids = prow['LATEST_DEP_QID']
    dep_list, dep_str = '', ''

    if len(dep_qids) > 0:
        jobtype = prow['JOBDESC']
        if strictly_successful:
            depcond = 'afterok'
        elif jobtype in ['arc', 'psfnight', 'prestdstar', 'stdstarfit']:
            ## (though psfnight and stdstarfit will require some inputs otherwise they'll go up in flames)
            depcond = 'afterany'
        else:
            ## if 'flat','nightlyflat','poststdstar', or any type of redshift, require strict success of inputs
            depcond = 'afterok'

        dep_str = f'--dependency={depcond}:'

        if np.isscalar(dep_qids):
            dep_list = str(dep_qids).strip(' \t')
            if dep_list == '':
                dep_str = ''
            else:
                dep_str += dep_list
        else:
            if len(dep_qids)>1:
                dep_list = ':'.join(np.array(dep_qids).astype(str))
                dep_str += dep_list
            elif len(dep_qids) == 1 and dep_qids[0] not in [None, 0]:
                dep_str += str(dep_qids[0])
            else:
                dep_str = ''

    # script = f'{jobname}.slurm'
    # script_path = pathjoin(batchdir, script)
    if prow['JOBDESC'] in ['pernight-v0','pernight','perexp','cumulative']:
        script_path = get_tile_redshift_script_pathname(tileid=prow['TILEID'],group=prow['JOBDESC'],
                                                        night=prow['NIGHT'], expid=np.min(prow['EXPID']))
        jobname = os.path.split(script_path)[-1]
    else:
        batchdir = get_desi_proc_batch_file_path(night=prow['NIGHT'])
        jobname = batch_script_name(prow)
        script_path = pathjoin(batchdir, jobname)

    batch_params = ['sbatch', '--parsable']
    if dep_str != '':
        batch_params.append(f'{dep_str}')
    if reservation is not None:
        batch_params.append(f'--reservation={reservation}')
    batch_params.append(f'{script_path}')

    if dry_run:
        ## in dry_run, mock Slurm ID's are generated using CPU seconds. Wait one second so we have unique ID's
        current_qid = int(time.time() - 1.6e9)
        time.sleep(1)
    else:
        #- sbatch sometimes fails; try several times before giving up
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                current_qid = subprocess.check_output(batch_params, stderr=subprocess.STDOUT, text=True)
                current_qid = int(current_qid.strip(' \t\n'))
                break
            except subprocess.CalledProcessError as err:
                log.error(f'{jobname} submission failed: {batch_params}')
                log.error(f'{jobname} {err.output=}')
                if attempt < max_attempts - 1:
                    log.info('Sleeping 60 seconds then retrying')
                    time.sleep(60)
        else:  #- for/else happens if loop doesn't succeed
            msg = f'{jobname} submission failed {max_attempts} times; exiting'
            log.critical(msg)
            raise RuntimeError(msg)

    log.info(batch_params)
    log.info(f'Submitted {jobname} with dependencies {dep_str} and reservation={reservation}. Returned qid: {current_qid}')

    prow['LATEST_QID'] = current_qid
    prow['ALL_QIDS'] = np.append(prow['ALL_QIDS'],current_qid)
    prow['STATUS'] = 'SUBMITTED'
    prow['SUBMIT_DATE'] = int(time.time())

    return prow


#############################################
##########   Row Manipulations   ############
#############################################
def define_and_assign_dependency(prow, calibjobs, use_tilenight=False):
    """
    Given input processing row and possible calibjobs, this defines the
    JOBDESC keyword and assigns the dependency appropriate for the job type of
    prow.

    Args:
        prow, Table.Row or dict. Must include keyword accessible definitions for
                                 'OBSTYPE'. A row must have column names for
                                 'JOBDESC', 'INT_DEP_IDS', and 'LATEST_DEP_ID'.
        calibjobs, dict. Dictionary containing 'nightlybias', 'ccdcalib', 'psfnight'
                       and 'nightlyflat'. Each key corresponds to a Table.Row or
                       None. The table.Row() values are for the corresponding
                       calibration job. Each value that isn't None must contain
                       'INTID', and 'LATEST_QID'. If None, it assumes the
                       dependency doesn't exist and no dependency is assigned.
        use_tilenight, bool. Default is False. If True, use desi_proc_tilenight
                             for prestdstar, stdstar,and poststdstar steps for
                             science exposures.
    Returns:
        prow, Table.Row or dict. The same prow type and keywords as input except
                                 with modified values updated values for
                                 'JOBDESC', 'INT_DEP_IDS'. and 'LATEST_DEP_ID'.

    Note:
        This modifies the input. Though Table.Row objects are generally copied
        on modification, so the change to the input object in memory may or may
        not be changed. As of writing, a row from a table given to this function
        will not change during the execution of this function (but can be
        overwritten explicitly with the returned row if desired).
    """
    if prow['OBSTYPE'] in ['science', 'twiflat']:
        if calibjobs['nightlyflat'] is not None:
            dependency = calibjobs['nightlyflat']
        elif calibjobs['psfnight'] is not None:
            dependency = calibjobs['psfnight']
        elif calibjobs['ccdcalib'] is not None:
            dependency = calibjobs['ccdcalib']
        else:
            dependency = calibjobs['nightlybias']
        if not use_tilenight:
            prow['JOBDESC'] = 'prestdstar'
    elif prow['OBSTYPE'] == 'flat':
        if calibjobs['psfnight'] is not None:
            dependency = calibjobs['psfnight']
        elif calibjobs['ccdcalib'] is not None:
            dependency = calibjobs['ccdcalib']
        else:
            dependency = calibjobs['nightlybias']
    elif prow['OBSTYPE'] == 'arc':
        if calibjobs['ccdcalib'] is not None:
            dependency = calibjobs['ccdcalib']
        elif calibjobs['nightlybias'] is not None:
            dependency = calibjobs['nightlybias']
        elif calibjobs['badcol'] is not None:
            dependency = calibjobs['badcol']
        else:
            # arc job, but no ZEROs or DARKs to process ahead of time
            dependency = None
    else:
        dependency = None

    prow = assign_dependency(prow, dependency)

    return prow


def assign_dependency(prow, dependency):
    """
    Given input processing row and possible arcjob (processing row for psfnight) and flatjob (processing row for
    nightlyflat), this defines the JOBDESC keyword and assigns the dependency appropriate for the job type of prow.

    Args:
        prow, Table.Row or dict. Must include keyword accessible definitions for 'OBSTYPE'. A row must have column names for
                                 'JOBDESC', 'INT_DEP_IDS', and 'LATEST_DEP_ID'.
        dependency, NoneType or scalar/list/array of Table.Row, dict. Processing row corresponding to the required input
                                                                      for the job in prow. This must contain keyword
                                                                      accessible values for 'INTID', and 'LATEST_QID'.
                                                                      If None, it assumes the dependency doesn't exist
                                                                      and no dependency is assigned.

    Returns:
        prow, Table.Row or dict. The same prow type and keywords as input except with modified values updated values for
                                 'JOBDESC', 'INT_DEP_IDS'. and 'LATEST_DEP_ID'.

    Note:
        This modifies the input. Though Table.Row objects are generally copied on modification, so the change to the
        input object in memory may or may not be changed. As of writing, a row from a table given to this function will
        not change during the execution of this function (but can be overwritten explicitly with the returned row if desired).
    """
    prow['INT_DEP_IDS'] = np.ndarray(shape=0).astype(int)
    prow['LATEST_DEP_QID'] = np.ndarray(shape=0).astype(int)
    if dependency is not None:
        if type(dependency) in [list, np.array]:
            ids, qids = [], []
            for curdep in dependency:
                if still_a_dependency(curdep):
                    ids.append(curdep['INTID'])
                    qids.append(curdep['LATEST_QID'])
            prow['INT_DEP_IDS'] = np.array(ids, dtype=int)
            prow['LATEST_DEP_QID'] = np.array(qids, dtype=int)
        elif type(dependency) in [dict, OrderedDict, Table.Row] and still_a_dependency(dependency):
            prow['INT_DEP_IDS'] = np.array([dependency['INTID']], dtype=int)
            prow['LATEST_DEP_QID'] = np.array([dependency['LATEST_QID']], dtype=int)
    return prow

def still_a_dependency(dependency):
    """
    Defines the criteria for which a dependency is deemed complete (and therefore no longer a dependency).

     Args:
        dependency, Table.Row or dict. Processing row corresponding to the required input for the job in prow.
                                     This must contain keyword accessible values for 'STATUS', and 'LATEST_QID'.

    Returns:
        bool. False if the criteria indicate that the dependency is completed and no longer a blocking factor (ie no longer
              a genuine dependency). Returns True if the dependency is still a blocking factor such that the slurm
              scheduler needs to be aware of the pending job.

    """
    return dependency['LATEST_QID'] > 0 and dependency['STATUS'] != 'COMPLETED'

def get_type_and_tile(erow):
    """
    Trivial function to return the OBSTYPE and the TILEID from an exposure table row

    Args:
        erow, Table.Row or dict. Must contain 'OBSTYPE' and 'TILEID' as keywords.

    Returns:
        tuple (str, str), corresponding to the OBSTYPE and TILEID values of the input erow.
    """
    return str(erow['OBSTYPE']).lower(), erow['TILEID']


#############################################
#########   Table manipulators   ############
#############################################
def parse_previous_tables(etable, ptable, night):
    """
    This takes in the exposure and processing tables and regenerates all the working memory variables needed for the
    daily processing script.

    Used by the daily processing to define most of its state-ful variables into working memory.
    If the processing table is empty, these are simply declared and returned for use.
    If the code had previously run and exited (or crashed), however, this will all the code to
    re-establish itself by redefining these values.

    Args:
        etable, Table, Exposure table of all exposures that have been dealt with thus far.
        ptable, Table, Processing table of all exposures that have been processed.
        night, str or int, the night the data was taken.

    Returns:
        arcs, list of dicts, list of the individual arc jobs used for the psfnight (NOT all
                                   the arcs, if multiple sets existed)
        flats, list of dicts, list of the individual flat jobs used for the nightlyflat (NOT
                                    all the flats, if multiple sets existed)
        sciences, list of dicts, list of the most recent individual prestdstar science exposures
                                       (if currently processing that tile)
        calibjobs, dict. Dictionary containing 'nightlybias', 'ccdcalib', 'badcol', 'psfnight'
                       and 'nightlyflat'. Each key corresponds to a Table.Row or
                       None. The table.Row() values are for the corresponding
                       calibration job.
        curtype, None, the obstype of the current job being run. Always None as first new job will define this.
        lasttype, str or None, the obstype of the last individual exposure row to be processed.
        curtile, None, the tileid of the current job (if science). Otherwise None. Always None as first
                       new job will define this.
        lasttile, str or None, the tileid of the last job (if science). Otherwise None.
        internal_id, int, an internal identifier unique to each job. Increments with each new job. This
                          is the latest unassigned value.
    """
    log = get_logger()
    arcs, flats, sciences = [], [], []
    calibjobs = {'nightlybias': None, 'ccdcalib': None, 'badcol': None, 'psfnight': None,
                 'nightlyflat': None}
    curtype,lasttype = None,None
    curtile,lasttile = None,None

    if len(ptable) > 0:
        prow = ptable[-1]
        internal_id = int(prow['INTID'])+1
        lasttype,lasttile = get_type_and_tile(ptable[-1])
        jobtypes = ptable['JOBDESC']

        if 'nightlybias' in jobtypes:
            calibjobs['nightlybias'] = table_row_to_dict(ptable[jobtypes=='nightlybias'][0])
            log.info("Located nightlybias job in exposure table: {}".format(calibjobs['nightlybias']))

        if 'ccdcalib' in jobtypes:
            calibjobs['ccdcalib'] = table_row_to_dict(ptable[jobtypes=='ccdcalib'][0])
            log.info("Located ccdcalib job in exposure table: {}".format(calibjobs['ccdcalib']))

        if 'psfnight' in jobtypes:
            calibjobs['psfnight'] = table_row_to_dict(ptable[jobtypes=='psfnight'][0])
            log.info("Located joint fit psfnight job in exposure table: {}".format(calibjobs['psfnight']))
        elif lasttype == 'arc':
            seqnum = 10
            for row in ptable[::-1]:
                erow = etable[etable['EXPID']==row['EXPID'][0]]
                if row['OBSTYPE'].lower() == 'arc' and int(erow['SEQNUM'])<seqnum:
                    arcs.append(table_row_to_dict(row))
                    seqnum = int(erow['SEQNUM'])
                else:
                    break
            ## Because we work backword to fill in, we need to reverse them to get chronological order back
            arcs = arcs[::-1]

        if 'nightlyflat' in jobtypes:
            calibjobs['nightlyflat'] = table_row_to_dict(ptable[jobtypes=='nightlyflat'][0])
            log.info("Located joint fit nightlyflat job in exposure table: {}".format(calibjobs['nightlyflat']))
        elif lasttype == 'flat':
            for row in ptable[::-1]:
                erow = etable[etable['EXPID']==row['EXPID'][0]]
                if row['OBSTYPE'].lower() == 'flat' and int(erow['SEQTOT']) < 5:
                    if float(erow['EXPTIME']) > 100.:
                        flats.append(table_row_to_dict(row))
                else:
                    break
            flats = flats[::-1]

        if lasttype.lower() == 'science':
            for row in ptable[::-1]:
                if row['OBSTYPE'].lower() == 'science' and row['TILEID'] == lasttile and \
                   row['JOBDESC'] == 'prestdstar' and row['LASTSTEP'] != 'skysub':
                    sciences.append(table_row_to_dict(row))
                else:
                    break
            sciences = sciences[::-1]
    else:
        internal_id = night_to_starting_iid(night)

    return arcs,flats,sciences, \
           calibjobs, \
           curtype, lasttype, \
           curtile, lasttile,\
           internal_id


def update_and_recurvsively_submit(proc_table, submits=0, resubmission_states=None,
                                   ptab_name=None, dry_run=0,reservation=None):
    """
    Given an processing table, this loops over job rows and resubmits failed jobs (as defined by resubmission_states).
    Before submitting a job, it checks the dependencies for failures. If a dependency needs to be resubmitted, it recursively
    follows dependencies until it finds the first job without a failed dependency and resubmits that. Then resubmits the
    other jobs with the new Slurm jobID's for proper dependency coordination within Slurm.

    Args:
        proc_table, Table, the processing table with a row per job.
        submits, int, the number of submissions made to the queue. Used for saving files and in not overloading the scheduler.
        resubmission_states, list or array of strings, each element should be a capitalized string corresponding to a
                                                       possible Slurm scheduler state, where you wish for jobs with that
                                                       outcome to be resubmitted
        ptab_name, str, the full pathname where the processing table should be saved.
        dry_run, int, If nonzero, this is a simulated run. If dry_run=1 the scripts will be written or submitted. If
                      dry_run=2, the scripts will not be writter or submitted. Logging will remain the same
                      for testing as though scripts are being submitted. Default is 0 (false).
        reservation: str. The reservation to submit jobs to. If None, it is not submitted to a reservation.
    Returns:
        proc_table: Table, a table with the same rows as the input except that Slurm and jobid relevant columns have
                           been updated for those jobs that needed to be resubmitted.
        submits: int, the number of submissions made to the queue. This is incremented from the input submits, so it is
                      the number of submissions made from this function call plus the input submits value.

    Note:
        This modifies the inputs of both proc_table and submits and returns them.
    """
    log = get_logger()
    if resubmission_states is None:
        resubmission_states = get_resubmission_states()
    log.info(f"Resubmitting jobs with current states in the following: {resubmission_states}")
    proc_table = update_from_queue(proc_table, dry_run=False)
    log.info("Updated processing table queue information:")
    cols = ['INTID', 'INT_DEP_IDS', 'EXPID', 'TILEID',
            'OBSTYPE', 'JOBDESC', 'LATEST_QID', 'STATUS']
    print(np.array(cols))
    for row in proc_table:
        print(np.array(row[cols]))
    print("\n")
    id_to_row_map = {row['INTID']: rown for rown, row in enumerate(proc_table)}
    for rown in range(len(proc_table)):
        if proc_table['STATUS'][rown] in resubmission_states:
            proc_table, submits = recursive_submit_failed(rown, proc_table, submits,
                                                          id_to_row_map, ptab_name,
                                                          resubmission_states,
                                                          reservation, dry_run)
    return proc_table, submits

def recursive_submit_failed(rown, proc_table, submits, id_to_row_map, ptab_name=None,
                            resubmission_states=None, reservation=None, dry_run=0):
    """
    Given a row of a processing table and the full processing table, this resubmits the given job.
    Before submitting a job, it checks the dependencies for failures in the processing table. If a dependency needs to
    be resubmitted, it recursively follows dependencies until it finds the first job without a failed dependency and
    resubmits that. Then resubmits the other jobs with the new Slurm jobID's for proper dependency coordination within Slurm.

    Args:
        rown, Table.Row, the row of the processing table that you want to resubmit.
        proc_table, Table, the processing table with a row per job.
        submits, int, the number of submissions made to the queue. Used for saving files and in not overloading the scheduler.
        id_to_row_map, dict, lookup dictionary where the keys are internal ids (INTID's) and the values are the row position
                             in the processing table.
        ptab_name, str, the full pathname where the processing table should be saved.
        resubmission_states, list or array of strings, each element should be a capitalized string corresponding to a
                                                       possible Slurm scheduler state, where you wish for jobs with that
                                                       outcome to be resubmitted
        reservation: str. The reservation to submit jobs to. If None, it is not submitted to a reservation.
        dry_run, int, If nonzero, this is a simulated run. If dry_run=1 the scripts will be written or submitted. If
                      dry_run=2, the scripts will not be writter or submitted. Logging will remain the same
                      for testing as though scripts are being submitted. Default is 0 (false).
    Returns:
        proc_table: Table, a table with the same rows as the input except that Slurm and jobid relevant columns have
                           been updated for those jobs that needed to be resubmitted.
        submits: int, the number of submissions made to the queue. This is incremented from the input submits, so it is
                      the number of submissions made from this function call plus the input submits value.

    Note:
        This modifies the inputs of both proc_table and submits and returns them.
    """
    log = get_logger()
    row = proc_table[rown]
    log.info(f"Identified row {row['INTID']} as needing resubmission.")
    log.info(f"{row['INTID']}: Expid(s): {row['EXPID']}  Job: {row['JOBDESC']}")
    if resubmission_states is None:
        resubmission_states = get_resubmission_states()
    ideps = proc_table['INT_DEP_IDS'][rown]
    if ideps is None:
        proc_table['LATEST_DEP_QID'][rown] = np.ndarray(shape=0).astype(int)
    else:
        all_valid_states = list(resubmission_states.copy())
        all_valid_states.extend(['RUNNING','PENDING','SUBMITTED','COMPLETED'])
        for idep in np.sort(np.atleast_1d(ideps)):
            if proc_table['STATUS'][id_to_row_map[idep]] not in all_valid_states:
                log.warning(f"Proc INTID: {proc_table['INTID'][rown]} depended on" +
                            f" INTID {proc_table['INTID'][id_to_row_map[idep]]}" +
                            f" but that exposure has state" +
                            f" {proc_table['STATUS'][id_to_row_map[idep]]} that" +
                            f" isn't in the list of resubmission states." +
                            f" Exiting this job's resubmission attempt.")
                proc_table['STATUS'][rown] = "DEP_NOT_SUBD"
                return proc_table, submits
        qdeps = []
        for idep in np.sort(np.atleast_1d(ideps)):
            if proc_table['STATUS'][id_to_row_map[idep]] in resubmission_states:
                proc_table, submits = recursive_submit_failed(id_to_row_map[idep],
                                                              proc_table, submits,
                                                              id_to_row_map,
                                                              reservation=reservation,
                                                              dry_run=dry_run)
            qdeps.append(proc_table['LATEST_QID'][id_to_row_map[idep]])

        qdeps = np.atleast_1d(qdeps)
        if len(qdeps) > 0:
            proc_table['LATEST_DEP_QID'][rown] = qdeps
        else:
            log.error(f"number of qdeps should be 1 or more: Rown {rown}, ideps {ideps}")

    proc_table[rown] = submit_batch_script(proc_table[rown], reservation=reservation,
                                           strictly_successful=True, dry_run=dry_run)
    submits += 1

    if not dry_run:
        sleep_and_report(1, message_suffix=f"after submitting job to queue")
        if submits % 10 == 0:
            if ptab_name is None:
                write_table(proc_table, tabletype='processing', overwrite=True)
            else:
                write_table(proc_table, tablename=ptab_name, overwrite=True)
            sleep_and_report(2, message_suffix=f"after writing to disk")
        if submits % 100 == 0:
            proc_table = update_from_queue(proc_table)
            if ptab_name is None:
                write_table(proc_table, tabletype='processing', overwrite=True)
            else:
                write_table(proc_table, tablename=ptab_name, overwrite=True)
            sleep_and_report(10, message_suffix=f"after updating queue and writing to disk")
    return proc_table, submits


#########################################
########     Joint fit     ##############
#########################################
def joint_fit(ptable, prows, internal_id, queue, reservation, descriptor, z_submit_types=None,
              dry_run=0, strictly_successful=False, check_for_outputs=True, resubmit_partial_complete=True,
              system_name=None):
    """
    Given a set of prows, this generates a processing table row, creates a batch script, and submits the appropriate
    joint fitting job given by descriptor. If the joint fitting job is standard star fitting, the post standard star fits
    for all the individual exposures also created and submitted. The returned ptable has all of these rows added to the
    table given as input.

    Args:
        ptable, Table. The processing table where each row is a processed job.
        prows, list or array of dicts. The rows corresponding to the individual exposure jobs that are
                                                     inputs to the joint fit.
        internal_id, int, the next internal id to be used for assignment (already incremented up from the last used id number used).
        queue, str. The name of the queue to submit the jobs to. If None is given the current desi_proc default is used.
        reservation: str. The reservation to submit jobs to. If None, it is not submitted to a reservation.
        descriptor, str. Description of the joint fitting job. Can either be 'science' or 'stdstarfit', 'arc' or 'psfnight',
                         or 'flat' or 'nightlyflat'.
        z_submit_types: list of str's. The "group" types of redshifts that should be submitted with each
                                        exposure. If not specified or None, then no redshifts are submitted.
        dry_run, int, If nonzero, this is a simulated run. If dry_run=1 the scripts will be written or submitted. If
                      dry_run=2, the scripts will not be writter or submitted. Logging will remain the same
                      for testing as though scripts are being submitted. Default is 0 (false).
        strictly_successful, bool. Whether all jobs require all inputs to have succeeded. For daily processing, this is
                                   less desirable because e.g. the sciences can run with SVN default calibrations rather
                                   than failing completely from failed calibrations. Default is False.
        check_for_outputs, bool. Default is True. If True, the code checks for the existence of the expected final
                                 data products for the script being submitted. If all files exist and this is True,
                                 then the script will not be submitted. If some files exist and this is True, only the
                                 subset of the cameras without the final data products will be generated and submitted.
        resubmit_partial_complete, bool. Default is True. Must be used with check_for_outputs=True. If this flag is True,
                                         jobs with some prior data are pruned using PROCCAMWORD to only process the
                                         remaining cameras not found to exist.
        system_name (str): batch system name, e.g. cori-haswell or perlmutter-gpu

    Returns:
        ptable, Table. The same processing table as input except with added rows for the joint fit job and, in the case
                       of a stdstarfit, the poststdstar science exposure jobs.
        joint_prow, dict. Row of a processing table corresponding to the joint fit job.
        internal_id, int, the next internal id to be used for assignment (already incremented up from the last used id number used).
    """
    log = get_logger()
    if len(prows) < 1:
        return ptable, None, internal_id

    if descriptor is None:
        return ptable, None
    elif descriptor == 'arc':
        descriptor = 'psfnight'
    elif descriptor == 'flat':
        descriptor = 'nightlyflat'
    elif descriptor == 'science':
        if z_submit_types is None or len(z_submit_types) == 0:
            descriptor = 'stdstarfit'

    if descriptor not in ['psfnight', 'nightlyflat', 'science','stdstarfit']:
        return ptable, None, internal_id

    log.info(" ")
    log.info(f"Joint fit criteria found. Running {descriptor}.\n")

    if descriptor == 'science':
        joint_prow = make_joint_prow(prows, descriptor='stdstarfit', internal_id=internal_id)
    else:
        joint_prow = make_joint_prow(prows, descriptor=descriptor, internal_id=internal_id)
    internal_id += 1
    joint_prow = create_and_submit(joint_prow, queue=queue, reservation=reservation, joint=True, dry_run=dry_run,
                                   strictly_successful=strictly_successful, check_for_outputs=check_for_outputs,
                                   resubmit_partial_complete=resubmit_partial_complete, system_name=system_name)
    ptable.add_row(joint_prow)

    if descriptor in ['science','stdstarfit']:
        if descriptor == 'science':
            zprows = []
        log.info(" ")
        log.info(f"Submitting individual science exposures now that joint fitting of standard stars is submitted.\n")
        for row in prows:
            if row['LASTSTEP'] == 'stdstarfit':
                continue
            row['JOBDESC'] = 'poststdstar'
            row['INTID'] = internal_id
            internal_id += 1
            row['ALL_QIDS'] = np.ndarray(shape=0).astype(int)
            row = assign_dependency(row, joint_prow)
            row = create_and_submit(row, queue=queue, reservation=reservation, dry_run=dry_run,
                                    strictly_successful=strictly_successful, check_for_outputs=check_for_outputs,
                                    resubmit_partial_complete=resubmit_partial_complete, system_name=system_name)
            ptable.add_row(row)
            if descriptor == 'science' and row['LASTSTEP'] == 'all':
                zprows.append(row)

    ## Now run redshifts
    if descriptor == 'science' and len(zprows) > 0 and z_submit_types is not None:
        prow_selection = (  (ptable['OBSTYPE'] == 'science')
                          & (ptable['LASTSTEP'] == 'all')
                          & (ptable['JOBDESC'] == 'poststdstar')
                          & (ptable['TILEID'] == int(zprows[0]['TILEID'])) )
        nightly_zprows = []
        if np.sum(prow_selection) == len(zprows):
            nightly_zprows = zprows.copy()
        else:
            for prow in ptable[prow_selection]:
                nightly_zprows.append(table_row_to_dict(prow))

        for zsubtype in z_submit_types:
            if zsubtype == 'perexp':
                for zprow in zprows:
                    log.info(" ")
                    log.info(f"Submitting redshift fit of type {zsubtype} for TILEID {zprow['TILEID']} and EXPID {zprow['EXPID']}.\n")
                    joint_prow = make_joint_prow([zprow], descriptor=zsubtype, internal_id=internal_id)
                    internal_id += 1
                    joint_prow = create_and_submit(joint_prow, queue=queue, reservation=reservation, joint=True, dry_run=dry_run,
                                                   strictly_successful=strictly_successful, check_for_outputs=check_for_outputs,
                                                   resubmit_partial_complete=resubmit_partial_complete, system_name=system_name)
                    ptable.add_row(joint_prow)
            else:
                log.info(" ")
                log.info(f"Submitting joint redshift fits of type {zsubtype} for TILEID {nightly_zprows[0]['TILEID']}.")
                expids = [prow['EXPID'][0] for prow in nightly_zprows]
                log.info(f"Expids: {expids}.\n")
                joint_prow = make_joint_prow(nightly_zprows, descriptor=zsubtype, internal_id=internal_id)
                internal_id += 1
                joint_prow = create_and_submit(joint_prow, queue=queue, reservation=reservation, joint=True, dry_run=dry_run,
                                               strictly_successful=strictly_successful, check_for_outputs=check_for_outputs,
                                               resubmit_partial_complete=resubmit_partial_complete, system_name=system_name)
                ptable.add_row(joint_prow)

    if descriptor in ['psfnight', 'nightlyflat']:
        log.info(f"Setting the calibration exposures as calibrators in the processing table.\n")
        ptable = set_calibrator_flag(prows, ptable)

    return ptable, joint_prow, internal_id

#########################################
########     Redshifts     ##############
#########################################
def submit_redshifts(ptable, prows, tnight, internal_id, queue, reservation,
              dry_run=0, strictly_successful=False,
              check_for_outputs=True, resubmit_partial_complete=True,
              z_submit_types=None, system_name=None):
    """
    Given a set of prows, this generates a processing table row, creates a batch script, and submits the appropriate
    tilenight job given by descriptor. The returned ptable has all of these rows added to the
    table given as input.

    Args:
        ptable, Table. The processing table where each row is a processed job.
        prows list or array of dicts. Unsubmitted prestdstar jobs that are first steps in tilenight.
        tnight, Table.Row. The processing table row of the tilenight job on which the redshifts depend.
        internal_id, int, the next internal id to be used for assignment (already incremented up from the last used id number used).
        queue, str. The name of the queue to submit the jobs to. If None is given the current desi_proc default is used.
        reservation: str. The reservation to submit jobs to. If None, it is not submitted to a reservation.
        dry_run, int, If nonzero, this is a simulated run. If dry_run=1 the scripts will be written or submitted. If
                      dry_run=2, the scripts will not be writter or submitted. Logging will remain the same
                      for testing as though scripts are being submitted. Default is 0 (false).
        strictly_successful, bool. Whether all jobs require all inputs to have succeeded. For daily processing, this is
                                   less desirable because e.g. the sciences can run with SVN default calibrations rather
                                   than failing completely from failed calibrations. Default is False.
        check_for_outputs, bool. Default is True. If True, the code checks for the existence of the expected final
                                 data products for the script being submitted. If all files exist and this is True,
                                 then the script will not be submitted. If some files exist and this is True, only the
                                 subset of the cameras without the final data products will be generated and submitted.
        resubmit_partial_complete, bool. Default is True. Must be used with check_for_outputs=True. If this flag is True,
                                         jobs with some prior data are pruned using PROCCAMWORD to only process the
                                         remaining cameras not found to exist.
        z_submit_types: list of str's. The "group" types of redshifts that should be submitted with each
                                        exposure. If not specified or None, then no redshifts are submitted.
        system_name (str): batch system name, e.g. cori-haswell or perlmutter-gpu

    Returns:
        ptable, Table. The same processing table as input except with added rows for the joint fit job.
        internal_id, int, the next internal id to be used for assignment (already incremented up from the last used id number used).
    """
    log = get_logger()
    if len(prows) < 1 or z_submit_types == None:
        return ptable, internal_id

    log.info(" ")
    log.info(f"Running redshifts.\n")

    ## Now run redshifts
    zprows = []
    for row in prows:
        if row['LASTSTEP'] == 'all':
            zprows.append(row)

    if len(zprows) > 0:
        for zsubtype in z_submit_types:
            if zsubtype == 'perexp':
                for zprow in zprows:
                    log.info(" ")
                    log.info(f"Submitting redshift fit of type {zsubtype} for TILEID {zprow['TILEID']} and EXPID {zprow['EXPID']}.\n")
                    redshift_prow = make_redshift_prow([zprow], tnight, descriptor=zsubtype, internal_id=internal_id)
                    internal_id += 1
                    redshift_prow = create_and_submit(redshift_prow, queue=queue, reservation=reservation, joint=True, dry_run=dry_run,
                                                   strictly_successful=strictly_successful, check_for_outputs=check_for_outputs,
                                                   resubmit_partial_complete=resubmit_partial_complete, system_name=system_name)
                    ptable.add_row(redshift_prow)
            else:
                log.info(" ")
                log.info(f"Submitting joint redshift fits of type {zsubtype} for TILEID {zprows[0]['TILEID']}.")
                expids = [prow['EXPID'][0] for prow in zprows]
                log.info(f"Expids: {expids}.\n")
                redshift_prow = make_redshift_prow(zprows, tnight, descriptor=zsubtype, internal_id=internal_id)
                internal_id += 1
                redshift_prow = create_and_submit(redshift_prow, queue=queue, reservation=reservation, joint=True, dry_run=dry_run,
                                               strictly_successful=strictly_successful, check_for_outputs=check_for_outputs,
                                               resubmit_partial_complete=resubmit_partial_complete, system_name=system_name)
                ptable.add_row(redshift_prow)

    return ptable, internal_id

#########################################
########     Tilenight     ##############
#########################################
def submit_tilenight(ptable, prows, calibjobs, internal_id, queue, reservation,
              dry_run=0, strictly_successful=False, resubmit_partial_complete=True,
              system_name=None,use_specter=False):
    """
    Given a set of prows, this generates a processing table row, creates a batch script, and submits the appropriate
    tilenight job given by descriptor. The returned ptable has all of these rows added to the
    table given as input.

    Args:
        ptable, Table. The processing table where each row is a processed job.
        prows list or array of dicts. Unsubmitted prestdstar jobs that are first steps in tilenight.
        calibjobs, dict. Dictionary containing 'nightlybias', 'ccdcalib', 'psfnight'
                       and 'nightlyflat'. Each key corresponds to a Table.Row or
                       None. The table.Row() values are for the corresponding
                       calibration job.
        internal_id, int, the next internal id to be used for assignment (already incremented up from the last used id number used).
        queue, str. The name of the queue to submit the jobs to. If None is given the current desi_proc default is used.
        reservation: str. The reservation to submit jobs to. If None, it is not submitted to a reservation.
        dry_run, int, If nonzero, this is a simulated run. If dry_run=1 the scripts will be written or submitted. If
                      dry_run=2, the scripts will not be writter or submitted. Logging will remain the same
                      for testing as though scripts are being submitted. Default is 0 (false).
        strictly_successful, bool. Whether all jobs require all inputs to have succeeded. For daily processing, this is
                                   less desirable because e.g. the sciences can run with SVN default calibrations rather
                                   than failing completely from failed calibrations. Default is False.
        resubmit_partial_complete, bool. Default is True. Must be used with check_for_outputs=True. If this flag is True,
                                         jobs with some prior data are pruned using PROCCAMWORD to only process the
                                         remaining cameras not found to exist.
        system_name (str): batch system name, e.g. cori-haswell or perlmutter-gpu
        use_specter, bool, optional. Default is False. If True, use specter, otherwise use gpu_specter by default.

    Returns:
        ptable, Table. The same processing table as input except with added rows for the joint fit job.
        tnight_prow, dict. Row of a processing table corresponding to the tilenight job.
        internal_id, int, the next internal id to be used for assignment (already incremented up from the last used id number used).
    """
    log = get_logger()
    if len(prows) < 1:
        return ptable, None, internal_id

    log.info(" ")
    log.info(f"Running tilenight.\n")

    tnight_prow = make_tnight_prow(prows, calibjobs, internal_id=internal_id)
    internal_id += 1
    tnight_prow = create_and_submit(tnight_prow, queue=queue, reservation=reservation, dry_run=dry_run,
                                   strictly_successful=strictly_successful, check_for_outputs=False,
                                   resubmit_partial_complete=resubmit_partial_complete, system_name=system_name,
                                   use_specter=use_specter)
    ptable.add_row(tnight_prow)

    return ptable, tnight_prow, internal_id

## wrapper functions for joint fitting
def science_joint_fit(ptable, sciences, internal_id, queue='realtime', reservation=None,
                      z_submit_types=None, dry_run=0, strictly_successful=False,
                      check_for_outputs=True, resubmit_partial_complete=True,
                      system_name=None):
    """
    Wrapper function for desiproc.workflow.procfuns.joint_fit specific to the stdstarfit joint fit and redshift fitting.

    All variables are the same except:
        Arg 'sciences' is mapped to the prows argument of joint_fit.
        The joint_fit argument descriptor is pre-defined as 'science'.
    """
    return joint_fit(ptable=ptable, prows=sciences, internal_id=internal_id, queue=queue, reservation=reservation,
                     descriptor='science', z_submit_types=z_submit_types, dry_run=dry_run,
                     strictly_successful=strictly_successful, check_for_outputs=check_for_outputs,
                     resubmit_partial_complete=resubmit_partial_complete, system_name=system_name)


def flat_joint_fit(ptable, flats, internal_id, queue='realtime',
                   reservation=None, dry_run=0, strictly_successful=False,
                   check_for_outputs=True, resubmit_partial_complete=True,
                   system_name=None):
    """
    Wrapper function for desiproc.workflow.procfuns.joint_fit specific to the nightlyflat joint fit.

    All variables are the same except:
        Arg 'flats' is mapped to the prows argument of joint_fit.
        The joint_fit argument descriptor is pre-defined as 'nightlyflat'.
    """
    return joint_fit(ptable=ptable, prows=flats, internal_id=internal_id, queue=queue, reservation=reservation,
                     descriptor='nightlyflat', dry_run=dry_run, strictly_successful=strictly_successful,
                     check_for_outputs=check_for_outputs, resubmit_partial_complete=resubmit_partial_complete,
                     system_name=system_name)


def arc_joint_fit(ptable, arcs, internal_id, queue='realtime',
                  reservation=None, dry_run=0, strictly_successful=False,
                  check_for_outputs=True, resubmit_partial_complete=True,
                  system_name=None):
    """
    Wrapper function for desiproc.workflow.procfuns.joint_fit specific to the psfnight joint fit.

    All variables are the same except:
        Arg 'arcs' is mapped to the prows argument of joint_fit.
        The joint_fit argument descriptor is pre-defined as 'psfnight'.
    """
    return joint_fit(ptable=ptable, prows=arcs, internal_id=internal_id, queue=queue, reservation=reservation,
                     descriptor='psfnight', dry_run=dry_run, strictly_successful=strictly_successful,
                     check_for_outputs=check_for_outputs, resubmit_partial_complete=resubmit_partial_complete,
                     system_name=system_name)


def make_joint_prow(prows, descriptor, internal_id):
    """
    Given an input list or array of processing table rows and a descriptor, this creates a joint fit processing job row.
    It starts by copying the first input row, overwrites relevant columns, and defines the new dependencies (based on the
    input prows).

    Args:
        prows, list or array of dicts. The rows corresponding to the individual exposure jobs that are
                                                     inputs to the joint fit.
        descriptor, str. Description of the joint fitting job. Can either be 'stdstarfit', 'psfnight', or 'nightlyflat'.
        internal_id, int, the next internal id to be used for assignment (already incremented up from the last used id number used).

    Returns:
        joint_prow, dict. Row of a processing table corresponding to the joint fit job.
    """
    first_row = prows[0]
    joint_prow = first_row.copy()

    joint_prow['INTID'] = internal_id
    joint_prow['JOBDESC'] = descriptor
    joint_prow['LATEST_QID'] = -99
    joint_prow['ALL_QIDS'] = np.ndarray(shape=0).astype(int)
    joint_prow['SUBMIT_DATE'] = -99
    joint_prow['STATUS'] = 'U'
    joint_prow['SCRIPTNAME'] = ''
    joint_prow['EXPID'] = np.array([currow['EXPID'][0] for currow in prows], dtype=int)
    if descriptor == 'stdstarfit':
        pcamwords = [prow['PROCCAMWORD'] for prow in prows]
        joint_prow['PROCCAMWORD'] = camword_union(pcamwords, full_spectros_only=True)

    joint_prow = assign_dependency(joint_prow,dependency=prows)
    return joint_prow

def make_tnight_prow(prows, calibjobs, internal_id):
    """
    Given an input list or array of processing table rows and a descriptor, this creates a joint fit processing job row.
    It starts by copying the first input row, overwrites relevant columns, and defines the new dependencies (based on the
    input prows).

    Args:
        prows, list or array of dicts. Unsumbitted rows corresponding to the individual prestdstar jobs that are
                                                     the first steps of tilenight.
        calibjobs, dict. Dictionary containing keys that each corresponds to a Table.Row or
                       None, with each table.Row() value corresponding to a calibration job
                       on which the tilenight job depends.
        internal_id, int, the next internal id to be used for assignment (already incremented up from the last used id number used).

    Returns:
        tnight_prow, dict. Row of a processing table corresponding to the tilenight job.
    """
    first_row = prows[0]
    joint_prow = first_row.copy()

    joint_prow['INTID'] = internal_id
    joint_prow['JOBDESC'] = 'tilenight'
    joint_prow['LATEST_QID'] = -99
    joint_prow['ALL_QIDS'] = np.ndarray(shape=0).astype(int)
    joint_prow['SUBMIT_DATE'] = -99
    joint_prow['STATUS'] = 'U'
    joint_prow['SCRIPTNAME'] = ''
    joint_prow['EXPID'] = np.array([currow['EXPID'][0] for currow in prows], dtype=int)

    joint_prow = define_and_assign_dependency(joint_prow,calibjobs,use_tilenight=True)

    return joint_prow

def make_redshift_prow(prows, tnight, descriptor, internal_id):
    """
    Given an input list or array of processing table rows and a descriptor, this creates a joint fit processing job row.
    It starts by copying the first input row, overwrites relevant columns, and defines the new dependencies (based on the
    input prows).

    Args:
        prows, list or array of dicts. Unsumbitted rows corresponding to the individual prestdstar jobs that are
                                                     the first steps of tilenight.
        tnight, Table.Row object. Row corresponding to the tilenight job on which the redshift job depends.
        internal_id, int, the next internal id to be used for assignment (already incremented up from the last used id number used).

    Returns:
        tnight_prow, dict. Row of a processing table corresponding to the tilenight job.
    """
    first_row = prows[0]
    redshift_prow = first_row.copy()

    redshift_prow['INTID'] = internal_id
    redshift_prow['JOBDESC'] = descriptor
    redshift_prow['LATEST_QID'] = -99
    redshift_prow['ALL_QIDS'] = np.ndarray(shape=0).astype(int)
    redshift_prow['SUBMIT_DATE'] = -99
    redshift_prow['STATUS'] = 'U'
    redshift_prow['SCRIPTNAME'] = ''
    redshift_prow['EXPID'] = np.array([currow['EXPID'][0] for currow in prows], dtype=int)

    redshift_prow = assign_dependency(redshift_prow,dependency=tnight)

    return redshift_prow

def checkfor_and_submit_joint_job(ptable, arcs, flats, sciences, calibjobs,
                                  lasttype, internal_id, z_submit_types=None, dry_run=0,
                                  queue='realtime', reservation=None, strictly_successful=False,
                                  check_for_outputs=True, resubmit_partial_complete=True,
                                  system_name=None):
    """
    Takes all the state-ful data from daily processing and determines whether a joint fit needs to be submitted. Places
    the decision criteria into a single function for easier maintainability over time. These are separate from the
    new standard manifest*.json method of indicating a calibration sequence is complete. That is checked independently
    elsewhere and doesn't interact with this.

    Args:
        ptable, Table, Processing table of all exposures that have been processed.
        arcs, list of dicts, list of the individual arc jobs to be used for the psfnight (NOT all
                                   the arcs, if multiple sets existed). May be empty if none identified yet.
        flats, list of dicts, list of the individual flat jobs to be used for the nightlyflat (NOT
                                    all the flats, if multiple sets existed). May be empty if none identified yet.
        sciences, list of dicts, list of the most recent individual prestdstar science exposures
                                       (if currently processing that tile). May be empty if none identified yet.
        calibjobs, dict. Dictionary containing 'nightlybias', 'ccdcalib', 'psfnight'
                       and 'nightlyflat'. Each key corresponds to a Table.Row or
                       None. The table.Row() values are for the corresponding
                       calibration job.
        lasttype, str or None, the obstype of the last individual exposure row to be processed.
        internal_id, int, an internal identifier unique to each job. Increments with each new job. This
                          is the smallest unassigned value.
        z_submit_types: list of str's. The "group" types of redshifts that should be submitted with each
                                        exposure. If not specified or None, then no redshifts are submitted.
        dry_run, int, If nonzero, this is a simulated run. If dry_run=1 the scripts will be written or submitted. If
                      dry_run=2, the scripts will not be writter or submitted. Logging will remain the same
                      for testing as though scripts are being submitted. Default is 0 (false).
        queue, str. The name of the queue to submit the jobs to. If None is given the current desi_proc default is used.
        reservation: str. The reservation to submit jobs to. If None, it is not submitted to a reservation.
        strictly_successful, bool. Whether all jobs require all inputs to have succeeded. For daily processing, this is
                                   less desirable because e.g. the sciences can run with SVN default calibrations rather
                                   than failing completely from failed calibrations. Default is False.
        check_for_outputs, bool. Default is True. If True, the code checks for the existence of the expected final
                                 data products for the script being submitted. If all files exist and this is True,
                                 then the script will not be submitted. If some files exist and this is True, only the
                                 subset of the cameras without the final data products will be generated and submitted.
        resubmit_partial_complete, bool. Default is True. Must be used with check_for_outputs=True. If this flag is True,
                                         jobs with some prior data are pruned using PROCCAMWORD to only process the
                                         remaining cameras not found to exist.
        system_name (str): batch system name, e.g. cori-haswell, cori-knl, permutter-gpu

    Returns:
        ptable, Table, Processing table of all exposures that have been processed.
        calibjobs, dict. Dictionary containing 'nightlybias', 'ccdcalib', 'psfnight'
                       and 'nightlyflat'. Each key corresponds to a Table.Row or
                       None. The table.Row() values are for the corresponding
                       calibration job.
        sciences, list of dicts, list of the most recent individual prestdstar science exposures
                                       (if currently processing that tile). May be empty if none identified yet or
                                       we just submitted them for processing.
        internal_id, int, if no job is submitted, this is the same as the input, otherwise it is incremented upward from
                          from the input such that it represents the smallest unused ID.
    """
    if lasttype == 'science' and len(sciences) > 0:
        log = get_logger()
        skysubonly = np.array([sci['LASTSTEP'] == 'skysub' for sci in sciences])
        if np.all(skysubonly):
            log.error("Identified all exposures in joint fitting request as skysub-only. Not submitting")
            sciences = []
            return ptable, calibjobs, sciences, internal_id

        if np.any(skysubonly):
            log.error("Identified skysub-only exposures in joint fitting request")
            log.info("Expid's: {}".format([row['EXPID'] for row in sciences]))
            log.info("LASTSTEP's: {}".format([row['LASTSTEP'] for row in sciences]))
            sciences = (np.array(sciences,dtype=object)[~skysubonly]).tolist()
            log.info("Removed skysub only exposures in joint fitting:")
            log.info("Expid's: {}".format([row['EXPID'] for row in sciences]))
            log.info("LASTSTEP's: {}".format([row['LASTSTEP'] for row in sciences]))

        from collections import Counter
        tiles = np.array([sci['TILEID'] for sci in sciences])
        counts = Counter(tiles)
        if len(counts.most_common()) > 1:
            log.error("Identified more than one tile in a joint fitting request")
            log.info("Expid's: {}".format([row['EXPID'] for row in sciences]))
            log.info("Tileid's: {}".format(tiles))
            log.info("Returning without joint fitting any of these exposures.")
            # most_common, nmost_common = counts.most_common()[0]
            # if most_common == -99:
            #     most_common, nmost_common = counts.most_common()[1]
            # log.warning(f"Given multiple tiles to jointly fit: {counts}. "+
            #             "Only processing the most common non-default " +
            #             f"tile: {most_common} with {nmost_common} exposures")
            # sciences = (np.array(sciences,dtype=object)[tiles == most_common]).tolist()
            # log.info("Tiles and exposure id's being submitted for joint fitting:")
            # log.info("Expid's: {}".format([row['EXPID'] for row in sciences]))
            # log.info("Tileid's: {}".format([row['TILEID'] for row in sciences]))
            sciences = []
            return ptable, calibjobs, sciences, internal_id

        ptable, tilejob, internal_id = science_joint_fit(ptable, sciences, internal_id, z_submit_types=z_submit_types,
                                                         dry_run=dry_run, queue=queue, reservation=reservation,
                                                         strictly_successful=strictly_successful,
                                                         check_for_outputs=check_for_outputs,
                                                         resubmit_partial_complete=resubmit_partial_complete,
                                                         system_name=system_name)
        if tilejob is not None:
            sciences = []

    elif lasttype == 'flat' and calibjobs['nightlyflat'] is None and len(flats) == 12:
        ## Note here we have an assumption about the number of expected flats being greater than 11
        ptable, calibjobs['nightlyflat'], internal_id \
            = flat_joint_fit(ptable, flats, internal_id, dry_run=dry_run, queue=queue,
                             reservation=reservation, strictly_successful=strictly_successful,
                             check_for_outputs=check_for_outputs,
                             resubmit_partial_complete=resubmit_partial_complete,
                             system_name=system_name
                            )

    elif lasttype == 'arc' and calibjobs['psfnight'] is None and len(arcs) == 5:
        ## Note here we have an assumption about the number of expected arcs being greater than 4
        ptable, calibjobs['psfnight'], internal_id \
            = arc_joint_fit(ptable, arcs, internal_id, dry_run=dry_run, queue=queue,
                            reservation=reservation, strictly_successful=strictly_successful,
                            check_for_outputs=check_for_outputs,
                            resubmit_partial_complete=resubmit_partial_complete,
                            system_name=system_name
                            )
    return ptable, calibjobs, sciences, internal_id

def submit_tilenight_and_redshifts(ptable, sciences, calibjobs, lasttype, internal_id, dry_run=0,
                                  queue='realtime', reservation=None, strictly_successful=False,
                                  check_for_outputs=True, resubmit_partial_complete=True,
                                  z_submit_types=None, system_name=None,use_specter=False):
    """
    Takes all the state-ful data from daily processing and determines whether a tilenight job needs to be submitted.

    Args:
        ptable, Table, Processing table of all exposures that have been processed.
        sciences, list of dicts, list of the most recent individual prestdstar science exposures
                                       (if currently processing that tile). May be empty if none identified yet.
        lasttype, str or None, the obstype of the last individual exposure row to be processed.
        internal_id, int, an internal identifier unique to each job. Increments with each new job. This
                          is the smallest unassigned value.
        dry_run, int, If nonzero, this is a simulated run. If dry_run=1 the scripts will be written or submitted. If
                      dry_run=2, the scripts will not be writter or submitted. Logging will remain the same
                      for testing as though scripts are being submitted. Default is 0 (false).
        queue, str. The name of the queue to submit the jobs to. If None is given the current desi_proc default is used.
        reservation: str. The reservation to submit jobs to. If None, it is not submitted to a reservation.
        strictly_successful, bool. Whether all jobs require all inputs to have succeeded. For daily processing, this is
                                   less desirable because e.g. the sciences can run with SVN default calibrations rather
                                   than failing completely from failed calibrations. Default is False.
        check_for_outputs, bool. Default is True. If True, the code checks for the existence of the expected final
                                 data products for the script being submitted. If all files exist and this is True,
                                 then the script will not be submitted. If some files exist and this is True, only the
                                 subset of the cameras without the final data products will be generated and submitted.
        resubmit_partial_complete, bool. Default is True. Must be used with check_for_outputs=True. If this flag is True,
                                         jobs with some prior data are pruned using PROCCAMWORD to only process the
                                         remaining cameras not found to exist.
        z_submit_types: list of str's. The "group" types of redshifts that should be submitted with each
                                        exposure. If not specified or None, then no redshifts are submitted.
        system_name (str): batch system name, e.g. cori-haswell, cori-knl, permutter-gpu
        use_specter, bool, optional. Default is False. If True, use specter, otherwise use gpu_specter by default.

    Returns:
        ptable, Table, Processing table of all exposures that have been processed.
        sciences, list of dicts, list of the most recent individual prestdstar science exposures
                                       (if currently processing that tile). May be empty if none identified yet or
                                       we just submitted them for processing.
        internal_id, int, if no job is submitted, this is the same as the input, otherwise it is incremented upward from
                          from the input such that it represents the smallest unused ID.
    """
    ptable, tnight, internal_id = submit_tilenight(ptable, sciences, calibjobs, internal_id,
                                             queue=queue, reservation=reservation,
                                             dry_run=dry_run, strictly_successful=strictly_successful,
                                             resubmit_partial_complete=resubmit_partial_complete,
                                             system_name=system_name,use_specter=use_specter
                                             )

    ptable, internal_id = submit_redshifts(ptable, sciences, tnight, internal_id,
                                    queue=queue, reservation=reservation,
                                    dry_run=dry_run, strictly_successful=strictly_successful,
                                    check_for_outputs=check_for_outputs,
                                    resubmit_partial_complete=resubmit_partial_complete,
                                    z_submit_types=z_submit_types, system_name=system_name
                                    )

    if tnight is not None:
        sciences = []

    return ptable, sciences, internal_id

def set_calibrator_flag(prows, ptable):
    """
    Sets the "CALIBRATOR" column of a procesing table row to 1 (integer representation of True)
     for all input rows. Used within joint fitting code to flag the exposures that were input
     to the psfnight or nightlyflat for later reference.

    Args:
        prows, list or array of Table.Rows or dicts. The rows corresponding to the individual exposure jobs that are
                                                     inputs to the joint fit.
        ptable, Table. The processing table where each row is a processed job.

    Returns:
        ptable, Table. The same processing table as input except with added rows for the joint fit job and, in the case
                       of a stdstarfit, the poststdstar science exposure jobs.
    """
    for prow in prows:
        ptable['CALIBRATOR'][ptable['INTID'] == prow['INTID']] = 1
    return ptable
