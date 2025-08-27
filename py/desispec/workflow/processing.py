"""
desispec.workflow.processing
============================

"""
import sys, os, glob
import json
from astropy.io import fits
from astropy.table import Table, join
import numpy as np

import time, datetime
from collections import OrderedDict
import subprocess

from desispec.scripts.link_calibnight import derive_include_exclude
from desispec.scripts.tile_redshifts import generate_tile_redshift_scripts
from desispec.workflow.redshifts import get_ztile_script_pathname, \
    get_ztile_relpath, \
    get_ztile_script_suffix
from desispec.workflow.exptable import read_minimal_science_exptab_cols
from desispec.workflow.queue import get_resubmission_states, update_from_queue, \
    queue_info_from_qids, get_queue_states_from_qids, update_queue_state_cache, \
    get_non_final_states
from desispec.workflow.timing import what_night_is_it
from desispec.workflow.desi_proc_funcs import get_desi_proc_batch_file_pathname, \
    create_desi_proc_batch_script, \
    get_desi_proc_batch_file_path, \
    get_desi_proc_tilenight_batch_file_pathname, \
    create_desi_proc_tilenight_batch_script, create_linkcal_batch_script
from desispec.workflow.batch import parse_reservation
from desispec.workflow.utils import pathjoin, sleep_and_report, \
    load_override_file
from desispec.workflow.tableio import write_table, load_table
from desispec.workflow.proctable import table_row_to_dict, erow_to_prow, \
    read_minimal_tilenight_proctab_cols, read_minimal_full_proctab_cols, \
    update_full_ptab_cache, default_prow, get_default_qid
from desiutil.log import get_logger

from desispec.io import findfile, specprod_root
from desispec.io.util import decode_camword, create_camword, \
    difference_camwords, \
    camword_to_spectros, camword_union, camword_intersection, parse_badamps


#################################################
############## Misc Functions ###################
#################################################
def night_to_starting_iid(night=None):
    """
    Creates an internal ID for a given night. The resulting integer is an 8 digit number.
    The digits are YYMMDDxxx where YY is the years since 2000, MM and DD are the month and day. xxx are 000,
    and are incremented for up to 1000 unique job ID's for a given night.

    Args:
        night (str or int): YYYYMMDD of the night to get the starting internal ID for.

    Returns:
        int: 9 digit number consisting of YYMMDD000. YY is years after 2000, MMDD is month and day.
        000 being the starting job number (0).
    """
    if night is None:
        night = what_night_is_it()
    night = int(night)
    internal_id = (night - 20000000) * 1000
    return internal_id

class ProcessingParams():
    def __init__(self, dry_run_level=0, queue='realtime',
                 reservation=None, strictly_successful=True,
                 check_for_outputs=True,
                 resubmit_partial_complete=True,
                 system_name='perlmutter', use_specter=True):

        self.dry_run_level = dry_run_level
        self.system_name = system_name
        self.queue = queue
        self.reservation = reservation
        self.strictly_successful = strictly_successful
        self.check_for_outputs = check_for_outputs
        self.resubmit_partial_complete = resubmit_partial_complete
        self.use_specter = use_specter

#################################################
############ Script Functions ###################
#################################################
def batch_script_name(prow):
    """
    Wrapper script that takes a processing table row (or dictionary with NIGHT, EXPID, JOBDESC, PROCCAMWORD defined)
    and determines the script file pathname as defined by desi_proc's helper functions.

    Args:
        prow (Table.Row or dict): Must include keyword accessible definitions for 'NIGHT', 'EXPID', 'JOBDESC', and 'PROCCAMWORD'.

    Returns:
        str: The complete pathname to the script file, as it is defined within the desi_proc ecosystem.
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

def get_jobdesc_to_file_map():
    """
    Returns a mapping of job descriptions to the filenames of the output files

    Args:
        None

    Returns:
        dict. Dictionary with keys as lowercase job descriptions and to the
            filename of their expected outputs.

    """
    return {'prestdstar': 'sframe',
            'stdstarfit': 'stdstars',
            'poststdstar': 'cframe',
            'nightlybias': 'biasnight',
            # 'ccdcalib': 'badcolumns',
            'badcol': 'badcolumns',
            'arc': 'fitpsf',
            'flat': 'fiberflat',
            'psfnight': 'psfnight',
            'nightlyflat': 'fiberflatnight',
            'spectra': 'spectra_tile',
            'coadds': 'coadds_tile',
            'redshift': 'redrock_tile'}

def get_file_to_jobdesc_map():
    """
    Returns a mapping of output filenames to job descriptions

    Args:
        None

    Returns:
        dict. Dictionary with keys as filename of their expected outputs to
            the lowercase job descriptions
            .

    """
    job_to_file_map = get_jobdesc_to_file_map()
    job_to_file_map.pop('badcol') # these files can also be in a ccdcalib job
    job_to_file_map.pop('nightlybias') # these files can also be in a ccdcalib job
    return {value: key for key, value in job_to_file_map.items()}

def check_for_outputs_on_disk(prow, resubmit_partial_complete=True):
    """
    Args:
        prow (Table.Row or dict): Must include keyword accessible definitions for processing_table columns found in
            desispect.workflow.proctable.get_processing_table_column_defs()
        resubmit_partial_complete (bool, optional): Default is True. Must be used with check_for_outputs=True. If this flag is True,
            jobs with some prior data are pruned using PROCCAMWORD to only process the
            remaining cameras not found to exist.

    Returns:
        Table.Row or dict: The same prow type and keywords as input except with modified values updated to reflect
        the change in job status after creating and submitting the job for processing.
    """
    prow['STATUS'] = 'UNKNOWN'
    log = get_logger()

    if prow['JOBDESC'] in ['linkcal', 'ccdcalib']:
        log.info(f"jobdesc={prow['JOBDESC']} has indeterminated outputs, so "
                + "not checking for files on disk.")
        return prow

    job_to_file_map = get_jobdesc_to_file_map()

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
        outdir = os.path.join(redux_dir,get_ztile_relpath(tileid,group=prow['JOBDESC'],night=night,expid=expid))
        suffix = get_ztile_script_suffix(tileid, group=prow['JOBDESC'], night=night, expid=expid)
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

def create_and_submit(prow, queue='realtime', reservation=None, dry_run=0,
                      joint=False, strictly_successful=False,
                      check_for_outputs=True, resubmit_partial_complete=True,
                      system_name=None, use_specter=False,
                      extra_job_args=None):
    """
    Wrapper script that takes a processing table row and three modifier keywords, creates a submission script for the
    compute nodes, and then submits that script to the Slurm scheduler with appropriate dependencies.

    Args:
        prow (Table.Row or dict): Must include keyword accessible definitions for processing_table columns found in
            desispect.workflow.proctable.get_processing_table_column_defs()
        queue (str, optional): The name of the NERSC Slurm queue to submit to. Default is the realtime queue.
        reservation: str. The reservation to submit jobs to. If None, it is not submitted to a reservation.
        dry_run (int, optional): If nonzero, this is a simulated run. Default is 0.
            0 which runs the code normally.
            1 writes all files but doesn't submit any jobs to Slurm.
            2 writes tables but doesn't write scripts or submit anything.
            3 Doesn't write or submit anything but queries Slurm normally for job status.
            4 Doesn't write, submit jobs, or query Slurm.
            5 Doesn't write, submit jobs, or query Slurm; instead it makes up the status of the jobs.
        joint (bool, optional): Whether this is a joint fitting job (the job involves multiple exposures) and therefore needs to be
            run with desi_proc_joint_fit. Default is False.
        strictly_successful (bool, optional): Whether all jobs require all inputs to have succeeded. For daily processing, this is
            less desirable because e.g. the sciences can run with SVN default calibrations rather
            than failing completely from failed calibrations. Default is False.
        check_for_outputs (bool, optional): Default is True. If True, the code checks for the existence of the expected final
            data products for the script being submitted. If all files exist and this is True,
            then the script will not be submitted. If some files exist and this is True, only the
            subset of the cameras without the final data products will be generated and submitted.
        resubmit_partial_complete (bool, optional): Default is True. Must be used with check_for_outputs=True. If this flag is True,
            jobs with some prior data are pruned using PROCCAMWORD to only process the
            remaining cameras not found to exist.
        system_name (str): batch system name, e.g. cori-haswell or perlmutter-gpu
        use_specter (bool, optional): Default is False. If True, use specter, otherwise use gpu_specter by default.
        extra_job_args (dict): Dictionary with key-value pairs that specify additional
            information used for a specific type of job. Examples include refnight
            and include/exclude lists for linkcals, laststeps for for tilenight, etc.

    Returns:
        Table.Row or dict: The same prow type and keywords as input except with modified values updated to reflect
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

    prow = create_batch_script(prow, queue=queue, dry_run=dry_run, joint=joint,
                               system_name=system_name, use_specter=use_specter,
                               extra_job_args=extra_job_args)
    prow = submit_batch_script(prow, reservation=reservation, dry_run=dry_run,
                               strictly_successful=strictly_successful)

    ## If resubmitted partial, the PROCCAMWORD and SCRIPTNAME will correspond
    ## to the pruned values. But we want to
    ## retain the full job's value, so get those from the old job.
    if resubmit_partial_complete:
        prow['PROCCAMWORD'] = orig_prow['PROCCAMWORD']
        prow['SCRIPTNAME'] = orig_prow['SCRIPTNAME']
    return prow

def desi_link_calibnight_command(prow, refnight, include=None):
    """
    Wrapper script that takes a processing table row (or dictionary with
    REFNIGHT, NIGHT, PROCCAMWORD defined) and determines the proper command
    line call to link data defined by the input row/dict.

    Args:
        prow (Table.Row or dict): Must include keyword accessible definitions
            for 'NIGHT', 'REFNIGHT', and 'PROCCAMWORD'.
        refnight (str or int): The night with a valid set of calibrations
            be created.
        include (list): The filetypes to include in the linking.
    Returns:
        str: The proper command to be submitted to desi_link_calibnight
            to process the job defined by the prow values.
    """
    cmd = 'desi_link_calibnight'
    cmd += f' --refnight={refnight}'
    cmd += f' --newnight={prow["NIGHT"]}'
    cmd += f' --cameras={prow["PROCCAMWORD"]}'
    if include is not None:
        cmd += f' --include=' + ','.join(list(include))
    return cmd

def desi_proc_command(prow, system_name, use_specter=False, queue=None):
    """
    Wrapper script that takes a processing table row (or dictionary with NIGHT, EXPID, OBSTYPE, JOBDESC, PROCCAMWORD defined)
    and determines the proper command line call to process the data defined by the input row/dict.

    Args:
        prow (Table.Row or dict): Must include keyword accessible definitions for 'NIGHT', 'EXPID', 'JOBDESC', and 'PROCCAMWORD'.
        system_name (str): batch system name, e.g. cori-haswell, cori-knl, perlmutter-gpu
        queue (str, optional): The name of the NERSC Slurm queue to submit to. Default is None (which leaves it to the desi_proc default).
        use_specter (bool, optional): Default is False. If True, use specter, otherwise use gpu_specter by default.

    Returns:
        str: The proper command to be submitted to desi_proc to process the job defined by the prow values.
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

        if use_specter:
            cmd += ' --use-specter'
    elif prow['JOBDESC'] in ['flat', 'prestdstar'] and use_specter:
        cmd += ' --use-specter'
    pcamw = str(prow['PROCCAMWORD'])
    cmd += f" --cameras={pcamw} -n {prow['NIGHT']}"
    if len(prow['EXPID']) > 0:
        ## If ccdcalib job without a dark exposure, don't assign the flat expid
        ## since it would incorrectly process the flat using desi_proc
        if prow['OBSTYPE'].lower() != 'flat' or prow['JOBDESC'] != 'ccdcalib':
            cmd += f" -e {prow['EXPID'][0]}"
    if prow['BADAMPS'] != '':
        cmd += ' --badamps={}'.format(prow['BADAMPS'])
    return cmd

def desi_proc_joint_fit_command(prow, queue=None):
    """
    Wrapper script that takes a processing table row (or dictionary with NIGHT, EXPID, OBSTYPE, PROCCAMWORD defined)
    and determines the proper command line call to process the data defined by the input row/dict.

    Args:
        prow (Table.Row or dict): Must include keyword accessible definitions for 'NIGHT', 'EXPID', 'JOBDESC', and 'PROCCAMWORD'.
        queue (str): The name of the NERSC Slurm queue to submit to. Default is None (which leaves it to the desi_proc default).

    Returns:
        str: The proper command to be submitted to desi_proc_joint_fit
            to process the job defined by the prow values.
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


def create_batch_script(prow, queue='realtime', dry_run=0, joint=False,
                        system_name=None, use_specter=False, extra_job_args=None):
    """
    Wrapper script that takes a processing table row and three modifier keywords and creates a submission script for the
    compute nodes.

    Args:
        prow, Table.Row or dict. Must include keyword accessible definitions for processing_table columns found in
            desispect.workflow.proctable.get_processing_table_column_defs()
        queue, str. The name of the NERSC Slurm queue to submit to. Default is the realtime queue.
        dry_run (int, optional): If nonzero, this is a simulated run. Default is 0.
            0 which runs the code normally.
            1 writes all files but doesn't submit any jobs to Slurm.
            2 writes tables but doesn't write scripts or submit anything.
            3 Doesn't write or submit anything but queries Slurm normally for job status.
            4 Doesn't write, submit jobs, or query Slurm.
            5 Doesn't write, submit jobs, or query Slurm; instead it makes up the status of the jobs.
        joint, bool. Whether this is a joint fitting job (the job involves multiple exposures) and therefore needs to be
            run with desi_proc_joint_fit when not using tilenight. Default is False.
        system_name (str): batch system name, e.g. cori-haswell or perlmutter-gpu
        use_specter, bool, optional. Default is False. If True, use specter, otherwise use gpu_specter by default.
        extra_job_args (dict): Dictionary with key-value pairs that specify additional
            information used for a specific type of job. Examples include refnight
            and include/exclude lists for linkcal, laststeps for tilenight, etc.

    Returns:
        Table.Row or dict: The same prow type and keywords as input except with modified values updated values for
        scriptname.

    Note:
        This modifies the input. Though Table.Row objects are generally copied on modification, so the change to the
        input object in memory may or may not be changed. As of writing, a row from a table given to this function will
        not change during the execution of this function (but can be overwritten explicitly with the returned row if desired).
    """
    log = get_logger()

    if extra_job_args is None:
        extra_job_args = {}

    if prow['JOBDESC'] in ['perexp','pernight','pernight-v0','cumulative']:
        if dry_run > 1:
            scriptpathname = get_ztile_script_pathname(tileid=prow['TILEID'],group=prow['JOBDESC'],
                                                               night=prow['NIGHT'], expid=prow['EXPID'][0])

            log.info("Output file would have been: {}".format(scriptpathname))
        else:
            #- run zmtl for cumulative redshifts but not others
            run_zmtl = (prow['JOBDESC'] == 'cumulative')
            no_afterburners = False
            print(f"entering tileredshiftscript: {prow}")
            scripts, failed_scripts = generate_tile_redshift_scripts(tileid=prow['TILEID'], group=prow['JOBDESC'],
                                                                     nights=[prow['NIGHT']], expids=prow['EXPID'],
                                                                     batch_queue=queue, system_name=system_name,
                                                                     run_zmtl=run_zmtl,
                                                                     no_afterburners=no_afterburners,
                                                                     nosubmit=True)
            if len(failed_scripts) > 0:
                log.error(f"Redshifts failed for group={prow['JOBDESC']}, night={prow['NIGHT']}, "+
                          f"tileid={prow['TILEID']}, expid={prow['EXPID']}.")
                log.info(f"Returned failed scriptname is {failed_scripts}")
            elif len(scripts) > 1:
                log.error(f"More than one redshifts returned for group={prow['JOBDESC']}, night={prow['NIGHT']}, "+
                          f"tileid={prow['TILEID']}, expid={prow['EXPID']}.")
                log.info(f"Returned scriptnames were {scripts}")
            elif len(scripts) == 0:
                msg = f'No scripts were generated for {prow=}'
                log.critical(prow)
                raise ValueError(msg)
            else:
                scriptpathname = scripts[0]

    elif prow['JOBDESC'] == 'linkcal':
        refnight, include, exclude = -99, None, None
        if 'refnight' in extra_job_args:
            refnight = extra_job_args['refnight']
        if 'include' in extra_job_args:
            include = extra_job_args['include']
        if 'exclude' in extra_job_args:
            exclude = extra_job_args['exclude']
        include, exclude = derive_include_exclude(include, exclude)
        ## Fiberflatnights shouldn't to be generated with psfs from same time, so
        ## shouldn't link psfs without also linking fiberflatnight
        ## However, this should be checked at a higher level. If set here,
        ## go ahead and do it
        # if 'psfnight' in include and not 'fiberflatnight' in include:
        #     err = "Must link fiberflatnight if linking psfnight"
        #     log.error(err)
        #     raise ValueError(err)
        if dry_run > 1:
            scriptpathname = batch_script_name(prow)
            log.info("Output file would have been: {}".format(scriptpathname))
            cmd = desi_link_calibnight_command(prow, refnight, include)
            log.info("Command to be run: {}".format(cmd.split()))
        else:
            if refnight == -99:
                err = f'For {prow=} asked to link calibration but not given' \
                      + ' a valid refnight'
                log.error(err)
                raise ValueError(err)

            cmd = desi_link_calibnight_command(prow, refnight, include)
            log.info(f"Running: {cmd.split()}")
            scriptpathname = create_linkcal_batch_script(newnight=prow['NIGHT'],
                                                        cameras=prow['PROCCAMWORD'],
                                                        queue=queue,
                                                        cmd=cmd,
                                                        system_name=system_name)
    else:
        if prow['JOBDESC'] != 'tilenight':
            nightlybias, nightlycte, cte_expids = False, False, None
            if 'nightlybias' in extra_job_args:
                nightlybias = extra_job_args['nightlybias']
            elif prow['JOBDESC'].lower() == 'nightlybias':
                nightlybias = True
            if 'nightlycte' in extra_job_args:
                nightlycte = extra_job_args['nightlycte']
            if 'cte_expids' in extra_job_args:
                cte_expids = extra_job_args['cte_expids']
            ## run known joint jobs as joint even if unspecified
            ## in the future we can eliminate the need for "joint"
            if joint or prow['JOBDESC'].lower() in ['psfnight', 'nightlyflat']:
                cmd = desi_proc_joint_fit_command(prow, queue=queue)
                ## For consistency with how we edit the other commands, do them
                ## here, but future TODO would be to move these into the command
                ## generation itself
                if 'extra_cmd_args' in extra_job_args:
                    cmd += ' ' + ' '.join(np.atleast_1d(extra_job_args['extra_cmd_args']))
            else:
                cmd = desi_proc_command(prow, system_name, use_specter, queue=queue)
                if nightlybias:
                    cmd += ' --nightlybias'
                if nightlycte:
                    cmd += ' --nightlycte'
                    if cte_expids is not None:
                        cmd += ' --cte-expids '
                        cmd += ','.join(np.atleast_1d(cte_expids).astype(str))
        if dry_run > 1:
            scriptpathname = batch_script_name(prow)
            log.info("Output file would have been: {}".format(scriptpathname))
            if prow['JOBDESC'] != 'tilenight':
                log.info("Command to be run: {}".format(cmd.split()))
        else:
            expids = prow['EXPID']
            if len(expids) == 0:
                expids = None

            if prow['JOBDESC'] == 'tilenight':
                log.info("Creating tilenight script for tile {}".format(prow['TILEID']))
                if 'laststeps' in extra_job_args:
                    laststeps = extra_job_args['laststeps']
                else:
                    err = f'{prow=} job did not specify last steps to tilenight'
                    log.error(err)
                    raise ValueError(err)
                ncameras = len(decode_camword(prow['PROCCAMWORD']))
                scriptpathname = create_desi_proc_tilenight_batch_script(
                                                               night=prow['NIGHT'], exp=expids,
                                                               tileid=prow['TILEID'],
                                                               ncameras=ncameras,
                                                               queue=queue,
                                                               mpistdstars=True,
                                                               use_specter=use_specter,
                                                               system_name=system_name,
                                                               laststeps=laststeps)
            else:
                if expids is not None and len(expids) > 1:
                    expids = expids[:1]
                log.info("Running: {}".format(cmd.split()))
                scriptpathname = create_desi_proc_batch_script(night=prow['NIGHT'], exp=expids,
                                                               cameras=prow['PROCCAMWORD'],
                                                               jobdesc=prow['JOBDESC'],
                                                               queue=queue, cmdline=cmd,
                                                               use_specter=use_specter,
                                                               system_name=system_name,
                                                               nightlybias=nightlybias,
                                                               nightlycte=nightlycte,
                                                               cte_expids=cte_expids)
    log.info("Outfile is: {}".format(scriptpathname))
    prow['SCRIPTNAME'] = os.path.basename(scriptpathname)
    return prow

_fake_qid = int(time.time() - 1.7e9)
def _get_fake_qid():
    """
    Return fake slurm queue jobid to use for dry-run testing
    """
    # Note: not implemented as a yield generator so that this returns a
    # genuine int, not a generator object
    global _fake_qid
    _fake_qid += 1
    return _fake_qid

def submit_batch_script(prow, dry_run=0, reservation=None, strictly_successful=False):
    """
    Wrapper script that takes a processing table row and three modifier keywords and submits the scripts to the Slurm
    scheduler.

    Args:
        prow, Table.Row or dict. Must include keyword accessible definitions for processing_table columns found in
            desispect.workflow.proctable.get_processing_table_column_defs()
        dry_run (int, optional): If nonzero, this is a simulated run. Default is 0.
            0 which runs the code normally.
            1 writes all files but doesn't submit any jobs to Slurm.
            2 writes tables but doesn't write scripts or submit anything.
            3 Doesn't write or submit anything but queries Slurm normally for job status.
            4 Doesn't write, submit jobs, or query Slurm.
            5 Doesn't write, submit jobs, or query Slurm; instead it makes up the status of the jobs.
        reservation: str. The reservation to submit jobs to. If None, it is not submitted to a reservation.
        strictly_successful, bool. Whether all jobs require all inputs to have succeeded. For daily processing, this is
            less desirable because e.g. the sciences can run with SVN default calibrations rather
            than failing completely from failed calibrations. Default is False.

    Returns:
        Table.Row or dict: The same prow type and keywords as input except with modified values updated values for
        scriptname.

    Note:
        This modifies the input. Though Table.Row objects are generally copied on modification, so the change to the
        input object in memory may or may not be changed. As of writing, a row from a table given to this function will
        not change during the execution of this function (but can be overwritten explicitly with the returned row if desired).
    """
    log = get_logger()
    dep_qids = prow['LATEST_DEP_QID']
    dep_list, dep_str = '', ''

    ## With desi_proc_night we now either resubmit failed jobs or exit, so this
    ## should no longer be necessary in the normal workflow.
    # workaround for sbatch --dependency bug not tracking jobs correctly
    # see NERSC TICKET INC0203024
    failed_dependency = False
    if len(dep_qids) > 0 and not dry_run:
        non_final_states = get_non_final_states()
        state_dict = get_queue_states_from_qids(dep_qids, dry_run_level=dry_run, use_cache=True)
        still_depids = []
        for depid in dep_qids:
            if depid in state_dict.keys():
                if state_dict[int(depid)] == 'COMPLETED':
                   log.info(f"removing completed jobid {depid}")
                elif state_dict[int(depid)] not in non_final_states:
                    failed_dependency = True
                    log.info("Found a dependency in a bad final state="
                             + f"{state_dict[int(depid)]} for depjobid={depid},"
                             + " not submitting this job.")
                    still_depids.append(depid)
                else:
                    still_depids.append(depid)
            else:
                still_depids.append(depid)
        dep_qids = np.array(still_depids)

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
        script_path = get_ztile_script_pathname(tileid=prow['TILEID'],group=prow['JOBDESC'],
                                                        night=prow['NIGHT'], expid=np.min(prow['EXPID']))
        jobname = os.path.basename(script_path)
    else:
        batchdir = get_desi_proc_batch_file_path(night=prow['NIGHT'])
        jobname = batch_script_name(prow)
        script_path = pathjoin(batchdir, jobname)

    batch_params = ['sbatch', '--parsable']
    if dep_str != '':
        batch_params.append(f'{dep_str}')

    reservation = parse_reservation(reservation, prow['JOBDESC'])
    if reservation is not None:
        batch_params.append(f'--reservation={reservation}')

    batch_params.append(f'{script_path}')
    submitted = True
    ## If dry_run give it a fake QID
    ## if a dependency has failed don't even try to submit the job because
    ## Slurm will refuse, instead just mark as unsubmitted.
    if dry_run:
        current_qid = _get_fake_qid()
    elif not failed_dependency:
        #- sbatch sometimes fails; try several times before giving up
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                current_qid = subprocess.check_output(batch_params, stderr=subprocess.STDOUT, text=True)
                current_qid = int(current_qid.strip(' \t\n'))
                break
            except subprocess.CalledProcessError as err:
                log.error(f'{jobname} submission failure at {datetime.datetime.now()}')
                log.error(f'{jobname}   {batch_params}')
                log.error(f'{jobname}   {err.output=}')
                if attempt < max_attempts - 1:
                    log.info('Sleeping 60 seconds then retrying')
                    time.sleep(60)
        else:  #- for/else happens if loop doesn't succeed
            msg = f'{jobname} submission failed {max_attempts} times.' \
                  + ' setting as unsubmitted and moving on'
            log.error(msg)
            current_qid = get_default_qid()
            submitted = False
    else:
        current_qid = get_default_qid()
        submitted = False

    ## Update prow with new information
    prow['LATEST_QID'] = current_qid

    ## If we didn't submit, don't say we did and don't add to ALL_QIDS
    if submitted:
        log.info(batch_params)
        log.info(f'Submitted {jobname} with dependencies {dep_str} and '
                 + f'reservation={reservation}. Returned qid: {current_qid}')

        ## Update prow with new information
        prow['ALL_QIDS'] = np.append(prow['ALL_QIDS'],current_qid)
        prow['STATUS'] = 'SUBMITTED'
        prow['SUBMIT_DATE'] = int(time.time())
    else:
        log.info(f"Would have submitted: {batch_params}")
        prow['STATUS'] = 'UNSUBMITTED'

        ## Update the Slurm jobid cache of job states
        update_queue_state_cache(qid=prow['LATEST_QID'], state=prow['STATUS'])

    return prow


#############################################
##########   Row Manipulations   ############
#############################################
def define_and_assign_dependency(prow, calibjobs, use_tilenight=False,
                                 refnight=None):
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
        refnight, int. The reference night for linking jobs

    Returns:
        Table.Row or dict: The same prow type and keywords as input except
        with modified values updated values for 'JOBDESC', 'INT_DEP_IDS'. and 'LATEST_DEP_ID'.

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
        elif calibjobs['nightlybias'] is not None:
            dependency = calibjobs['nightlybias']
        elif calibjobs['badcol'] is not None:
            dependency = calibjobs['badcol']
        else:
            dependency = calibjobs['linkcal']
        if not use_tilenight:
            prow['JOBDESC'] = 'prestdstar'
    elif prow['OBSTYPE'] == 'flat':
        if calibjobs['psfnight'] is not None:
            dependency = calibjobs['psfnight']
        elif calibjobs['ccdcalib'] is not None:
            dependency = calibjobs['ccdcalib']
        elif calibjobs['nightlybias'] is not None:
            dependency = calibjobs['nightlybias']
        elif calibjobs['badcol'] is not None:
            dependency = calibjobs['badcol']
        else:
            dependency = calibjobs['linkcal']
    elif prow['OBSTYPE'] == 'arc':
        if calibjobs['ccdcalib'] is not None:
            dependency = calibjobs['ccdcalib']
        elif calibjobs['nightlybias'] is not None:
            dependency = calibjobs['nightlybias']
        elif calibjobs['badcol'] is not None:
            dependency = calibjobs['badcol']
        else:
            dependency = calibjobs['linkcal']
    elif prow['JOBDESC'] in ['badcol', 'nightlybias', 'ccdcalib']:
        dependency = calibjobs['linkcal']
    elif prow['OBSTYPE'] == 'dark':
        if calibjobs['ccdcalib'] is not None:
            dependency = calibjobs['ccdcalib']
        elif calibjobs['nightlybias'] is not None:
            dependency = calibjobs['nightlybias']
        elif calibjobs['badcol'] is not None:
            dependency = calibjobs['badcol']
        else:
            dependency = calibjobs['linkcal']
    elif prow['JOBDESC'] == 'linkcal' and refnight is not None:
        dependency = None
        ## For link cals only, enable cross-night dependencies if available
        refproctable = findfile('proctable', night=refnight)
        if os.path.exists(refproctable):
            ptab = load_table(tablename=refproctable, tabletype='proctable')
            ## This isn't perfect because we may depend on jobs that aren't
            ## actually being linked
            ## Also allows us to proceed even if jobs don't exist yet
            deps = []
            for job in ['nightlybias', 'ccdcalib', 'psfnight', 'nightlyflat']:
                if job in ptab['JOBDESC']:
                    ## add prow to dependencies
                    deps.append(ptab[ptab['JOBDESC']==job][0])
            if len(deps) > 0:
                dependency = deps
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
        Table.Row or dict: The same prow type and keywords as input except with modified values updated values for
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
                ids.append(curdep['INTID'])
                if still_a_dependency(curdep):
                    # ids.append(curdep['INTID'])
                    qids.append(curdep['LATEST_QID'])
            prow['INT_DEP_IDS'] = np.array(ids, dtype=int)
            prow['LATEST_DEP_QID'] = np.array(qids, dtype=int)
        elif type(dependency) in [dict, OrderedDict, Table.Row]:
            prow['INT_DEP_IDS'] = np.array([dependency['INTID']], dtype=int)
            if still_a_dependency(dependency):
                prow['LATEST_DEP_QID'] = np.array([dependency['LATEST_QID']], dtype=int)
    return prow

def still_a_dependency(dependency):
    """
    Defines the criteria for which a dependency is deemed complete (and therefore no longer a dependency).

     Args:
        dependency, Table.Row or dict. Processing row corresponding to the required input for the job in prow.
            This must contain keyword accessible values for 'STATUS', and 'LATEST_QID'.

    Returns:
        bool: False if the criteria indicate that the dependency is completed and no longer a blocking factor (ie no longer
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
        tuple: A tuple containing:

        * arcs, list of dicts, list of the individual arc jobs used for the psfnight (NOT all
          the arcs, if multiple sets existed)
        * flats, list of dicts, list of the individual flat jobs used for the nightlyflat (NOT
          all the flats, if multiple sets existed)
        * sciences, list of dicts, list of the most recent individual prestdstar science exposures
          (if currently processing that tile)
        * calibjobs, dict. Dictionary containing 'nightlybias', 'ccdcalib', 'badcol', 'psfnight'
          and 'nightlyflat'. Each key corresponds to a Table.Row or
          None. The table.Row() values are for the corresponding
          calibration job.
        * curtype, None, the obstype of the current job being run. Always None as first new job will define this.
        * lasttype, str or None, the obstype of the last individual exposure row to be processed.
        * curtile, None, the tileid of the current job (if science). Otherwise None. Always None as first
          new job will define this.
        * lasttile, str or None, the tileid of the last job (if science). Otherwise None.
        * internal_id, int, an internal identifier unique to each job. Increments with each new job. This
          is the latest unassigned value.
    """
    log = get_logger()
    arcs, flats, sciences = [], [], []
    calibjobs = {'nightlybias': None, 'ccdcalib': None, 'badcol': None,
                 'psfnight': None, 'nightlyflat': None, 'linkcal': None,
                 'accounted_for': dict()}
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

def generate_calibration_dict(ptable, files_to_link=None):
    """
    This takes in a processing table and regenerates the working memory calibration
    dictionary for dependency tracking. Used by the daily processing to define 
    most of its state-ful variables into working memory.
    If the processing table is empty, these are simply declared and returned for use.
    If the code had previously run and exited (or crashed), however, this will all the code to
    re-establish itself by redefining these values.

    Args:
        ptable, Table, Processing table of all exposures that have been processed.
        files_to_link, set, Set of filenames that the linkcal job will link.

    Returns:
        calibjobs, dict. Dictionary containing 'nightlybias', 'badcol', 'ccdcalib',
            'psfnight', 'nightlyflat', 'linkcal', and 'completed'. Each key corresponds to a
            Table.Row or None. The table.Row() values are for the corresponding
            calibration job.
    """
    log = get_logger()
    job_to_file_map = get_jobdesc_to_file_map()
    accounted_for = {'biasnight': False, 'badcolumns': False,
                     'ctecorrnight': False, 'psfnight': False,
                     'fiberflatnight': False}
    calibjobs = {'nightlybias': None, 'ccdcalib': None, 'badcol': None,
                 'psfnight': None, 'nightlyflat': None, 'linkcal': None}

    ptable_jobtypes = ptable['JOBDESC']

    for jobtype in calibjobs.keys():
        if jobtype in ptable_jobtypes:
            calibjobs[jobtype] = table_row_to_dict(ptable[ptable_jobtypes==jobtype][0])
            log.info(f"Located {jobtype} job in exposure table: {calibjobs[jobtype]}")
            if jobtype == 'linkcal':
                if files_to_link is not None and len(files_to_link) > 0:
                    log.info(f"Assuming existing linkcal job processed "
                             + f"{files_to_link} since given in override file.")
                    accounted_for = update_accounted_for_with_linking(accounted_for,
                                                                  files_to_link)
                else:
                    err = f"linkcal job exists but no files given: {files_to_link=}"
                    log.error(err)
                    raise ValueError(err)
            elif jobtype == 'ccdcalib':
                possible_ccd_files = set(['biasnight', 'badcolumns', 'ctecorrnight'])
                if files_to_link is None:
                    files_accounted_for = possible_ccd_files
                else:
                    files_accounted_for = possible_ccd_files.difference(files_to_link)
                    ccd_files_linked = possible_ccd_files.intersection(files_to_link)
                    log.info(f"Assuming existing ccdcalib job processed "
                             + f"{files_accounted_for} since {ccd_files_linked} "
                             + f"are linked.")
                for fil in files_accounted_for:
                    accounted_for[fil] = True
            else:
                accounted_for[job_to_file_map[jobtype]] = True

    calibjobs['accounted_for'] = accounted_for
    return calibjobs

def update_accounted_for_with_linking(accounted_for, files_to_link):
    """
    This takes in a dictionary summarizing the calibration files accounted for
     and updates it based on the files_to_link, which are assumed to have
     already been linked such that those files already exist on disk and
     don't need ot be generated.

    Parameters
    ----------
        accounted_for: dict
            Dictionary containing 'biasnight', 'badcolumns', 'ctecorrnight',
            'psfnight', and 'fiberflatnight'. Each value is True if file is
            accounted for and False if it is not.
        files_to_link: set
            Set of filenames that the linkcal job will link.

    Returns
    -------
        accounted_for: dict
            Dictionary containing 'biasnight', 'badcolumns', 'ctecorrnight',
            'psfnight', and 'fiberflatnight'. Each value is True if file is
            accounted for and False if it is not.
    """
    log = get_logger()
    
    for fil in files_to_link:
        if fil in accounted_for:
            accounted_for[fil] = True
        else:
            err = f"{fil} doesn't match an expected filetype: "
            err += f"{accounted_for.keys()}"
            log.error(err)
            raise ValueError(err)

    return accounted_for

def all_calibs_submitted(accounted_for, do_cte_flats):
    """
    Function that returns the boolean logic to determine if the necessary
    calibration jobs have been submitted for calibration.

    Args:
        accounted_for, dict, Dictionary with keys corresponding to the calibration
            filenames and values of True or False.
        do_cte_flats, bool, whether ctecorrnight files are expected or not.

    Returns:
        bool, True if all necessary calibrations have been submitted or handled, False otherwise.
    """
    test_dict = accounted_for.copy()
    if not do_cte_flats:
        test_dict.pop('ctecorrnight')

    return np.all(list(test_dict.values()))

def update_and_recursively_submit(proc_table, submits=0, max_resubs=100,
                                  resubmission_states=None,
                                  no_resub_failed=False, ptab_name=None,
                                  dry_run_level=0, reservation=None,
                                  expids=None, tileids=None):
    """
    Given an processing table, this loops over job rows and resubmits failed jobs (as defined by resubmission_states).
    Before submitting a job, it checks the dependencies for failures. If a dependency needs to be resubmitted, it recursively
    follows dependencies until it finds the first job without a failed dependency and resubmits that. Then resubmits the
    other jobs with the new Slurm jobID's for proper dependency coordination within Slurm.

    Args:
        proc_table, Table, the processing table with a row per job.
        submits, int, the number of submissions made to the queue. Used for saving files and in not overloading the scheduler.
        max_resubs, int, the number of times a job should be resubmitted before giving up. Default is very high at 100.
        resubmission_states, list or array of strings, each element should be a capitalized string corresponding to a
            possible Slurm scheduler state, where you wish for jobs with that
            outcome to be resubmitted
        no_resub_failed: bool. Set to True if you do NOT want to resubmit
            jobs with Slurm status 'FAILED' by default. Default is False.
        ptab_name, str, the full pathname where the processing table should be saved.
        dry_run_level (int, optional): If nonzero, this is a simulated run. Default is 0.
            0 which runs the code normally.
            1 writes all files but doesn't submit any jobs to Slurm.
            2 writes tables but doesn't write scripts or submit anything.
            3 Doesn't write or submit anything but queries Slurm normally for job status.
            4 Doesn't write, submit jobs, or query Slurm.
            5 Doesn't write, submit jobs, or query Slurm; instead it makes up the status of the jobs.
        reservation: str. The reservation to submit jobs to. If None, it is not submitted to a reservation.
        expids: list of ints. The exposure ids to resubmit (along with the jobs they depend on).
        tileids: list of ints. The tile ids to resubmit (along with the jobs they depend on).

    Returns:
        tuple: A tuple containing:

        * proc_table: Table, a table with the same rows as the input except that Slurm and jobid relevant columns have
          been updated for those jobs that needed to be resubmitted.
        * submits: int, the number of submissions made to the queue. This is incremented from the input submits, so it is
          the number of submissions made from this function call plus the input submits value.

    Note:
        This modifies the inputs of both proc_table and submits and returns them.
    """
    log = get_logger()
    if tileids is not None and expids is not None:
        msg = f"Provided both expids and tilesids. Please only provide one."
        log.critical(msg)
        raise AssertionError(msg)
    elif tileids is not None:
        msg = f"Only resubmitting the following tileids and the jobs they depend on: {tileids=}"
        log.info(msg)
    elif expids is not None:
        msg = f"Only resubmitting the following expids and the jobs they depend on: {expids=}"
        log.info(msg)

    if resubmission_states is None:
        resubmission_states = get_resubmission_states(no_resub_failed=no_resub_failed)

    log.info(f"Resubmitting jobs with current states in the following: {resubmission_states}")
    proc_table = update_from_queue(proc_table, dry_run_level=dry_run_level)

    log.info("Updated processing table queue information:")
    cols = ['INTID', 'INT_DEP_IDS', 'EXPID', 'TILEID',
            'OBSTYPE', 'JOBDESC', 'LATEST_QID', 'STATUS']
    log.info(np.array(cols))
    for row in proc_table:
        log.info(np.array(row[cols]))

    ## If expids or tileids are given, subselect to the processing table rows
    ## that included those exposures or tiles otherwise just list all indices
    ## NOTE: Other rows can still be submitted if the selected rows depend on them
    ## we hand the entire table to recursive_submit_failed(), which will walk the
    ## entire dependency tree as necessary.
    if expids is not None:
        select_ptab_rows = np.where([np.any(np.isin(prow_eids, expids)) for prow_eids in proc_table['EXPID']])[0]
    elif tileids is not None:
        select_ptab_rows = np.where(np.isin(proc_table['TILEID'], tileids))[0]
    else:
        select_ptab_rows = np.arange(len(proc_table))

    log.info("\n")
    id_to_row_map = {row['INTID']: rown for rown, row in enumerate(proc_table)}
    ## Loop over all requested rows and resubmit those that have failed
    for rown in select_ptab_rows:
        if proc_table['STATUS'][rown] in resubmission_states:
            proc_table, submits = recursive_submit_failed(rown=rown, proc_table=proc_table,
                                                          submits=submits, max_resubs=max_resubs,
                                                          id_to_row_map=id_to_row_map,
                                                          ptab_name=ptab_name,
                                                          resubmission_states=resubmission_states,
                                                          reservation=reservation,
                                                          dry_run_level=dry_run_level)

    proc_table = update_from_queue(proc_table, dry_run_level=dry_run_level)

    return proc_table, submits

def recursive_submit_failed(rown, proc_table, submits, id_to_row_map, max_resubs=100, ptab_name=None,
                            resubmission_states=None, reservation=None, dry_run_level=0):
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
        max_resubs, int, the number of times a job should be resubmitted before giving up. Default is very high at 100.
        ptab_name, str, the full pathname where the processing table should be saved.
        resubmission_states, list or array of strings, each element should be a capitalized string corresponding to a
            possible Slurm scheduler state, where you wish for jobs with that
            outcome to be resubmitted
        reservation: str. The reservation to submit jobs to. If None, it is not submitted to a reservation.
        dry_run_level (int, optional): If nonzero, this is a simulated run. Default is 0.
            0 which runs the code normally.
            1 writes all files but doesn't submit any jobs to Slurm.
            2 writes tables but doesn't write scripts or submit anything.
            3 Doesn't write or submit anything but queries Slurm normally for job status.
            4 Doesn't write, submit jobs, or query Slurm.
            5 Doesn't write, submit jobs, or query Slurm; instead it makes up the status of the jobs.

    Returns:
        tuple: A tuple containing:

        * proc_table: Table, a table with the same rows as the input except that Slurm and jobid relevant columns have
          been updated for those jobs that needed to be resubmitted.
        * submits: int, the number of submissions made to the queue. This is incremented from the input submits, so it is
          the number of submissions made from this function call plus the input submits value.

    Note:
        This modifies the inputs of both proc_table and submits and returns them.
    """
    log = get_logger()
    row = proc_table[rown]
    log.info(f"Identified row {row['INTID']} as needing resubmission.")
    log.info(f"\t{row['INTID']}: Tileid={row['TILEID']}, Expid(s)={row['EXPID']}, Jobdesc={row['JOBDESC']}")
    if len(proc_table['ALL_QIDS'][rown]) > max_resubs:
        log.warning(f"Tileid={row['TILEID']}, Expid(s)={row['EXPID']}, "
                    + f"Jobdesc={row['JOBDESC']} has already been submitted "
                    + f"{max_resubs+1} times. Not resubmitting.")
        proc_table['STATUS'][rown] = "MAX_RESUB"
        return proc_table, submits
    if resubmission_states is None:
        resubmission_states = get_resubmission_states()
    ideps = proc_table['INT_DEP_IDS'][rown]
    if ideps is None or len(ideps)==0:
        proc_table['LATEST_DEP_QID'][rown] = np.ndarray(shape=0).astype(int)
    else:
        all_valid_states = list(resubmission_states.copy())
        good_states = ['RUNNING','PENDING','SUBMITTED','COMPLETED']
        all_valid_states.extend(good_states)
        othernight_idep_row_lookup = {}
        for idep in np.sort(np.atleast_1d(ideps)):
            if idep not in id_to_row_map:
                if idep // 1000 != row['INTID'] // 1000:
                    log.debug("Internal ID: %d not in id_to_row_map. "
                             + "This is expected since it is from another day. ", idep)
                    reference_night = 20000000 + (idep // 1000)
                    reftab = read_minimal_full_proctab_cols(nights=[reference_night])
                    if reftab is None:
                        msg = f"The dependency is from night={reference_night}" \
                              + f" but read_minimal_full_proctab_cols couldn't" \
                              + f" locate that processing table, this is a " \
                              +  f"fatal error."
                        log.critical(msg)
                        raise ValueError(msg)
                    reftab = update_from_queue(reftab, dry_run_level=dry_run_level)
                    entry = reftab[reftab['INTID'] == idep][0]
                    if entry['STATUS'] not in good_states:
                        msg = f"Internal ID: {idep} not in id_to_row_map. " \
                              + f"Since the dependency is from night={reference_night} " \
                              + f"and that job isn't in a good state this is an " \
                              + f"error we can't overcome."
                        log.error(msg)
                        proc_table['STATUS'][rown] = "DEP_NOT_SUBD"
                        return proc_table, submits
                    else:
                        ## otherwise if incomplete, just update the cache to use this
                        ## in the next stage
                        othernight_idep_row_lookup[idep] = entry
                        update_full_ptab_cache(reftab)
                else:
                    msg = f"Internal ID: {idep} not in id_to_row_map. " \
                         + f"Since the dependency is from the same night" \
                         + f" and we can't find it, this is a fatal error."
                    log.critical(msg)
                    raise ValueError(msg)
            elif proc_table['STATUS'][id_to_row_map[idep]] not in all_valid_states:
                log.error(f"Proc INTID: {proc_table['INTID'][rown]} depended on" +
                            f" INTID {proc_table['INTID'][id_to_row_map[idep]]}" +
                            f" but that exposure has state" +
                            f" {proc_table['STATUS'][id_to_row_map[idep]]} that" +
                            f" isn't in the list of resubmission states." +
                            f" Exiting this job's resubmission attempt.")
                proc_table['STATUS'][rown] = "DEP_NOT_SUBD"
                return proc_table, submits
        qdeps = []
        for idep in np.sort(np.atleast_1d(ideps)):
            if idep in id_to_row_map:
                if proc_table['STATUS'][id_to_row_map[idep]] in resubmission_states:
                    proc_table, submits = recursive_submit_failed(id_to_row_map[idep],
                                                                  proc_table, submits,
                                                                  id_to_row_map,
                                                                  reservation=reservation,
                                                                  dry_run_level=dry_run_level)
                ## Now that we've resubmitted the dependency if necessary,
                ## add the most recent QID to the list assuming it isn't COMPLETED
                if still_a_dependency(proc_table[id_to_row_map[idep]]):
                    qdeps.append(proc_table['LATEST_QID'][id_to_row_map[idep]])
                else:
                    log.info(f"{idep} is COMPLETED. Not submitting as a dependency.")
            else:
                ## Since we verified above that the cross night QID is still
                ## either pending or successful, add that to the list of QID's
                if still_a_dependency(othernight_idep_row_lookup[idep]):
                    qdeps.append(othernight_idep_row_lookup[idep]['LATEST_QID'])
                else:
                    log.info(f"{idep} is COMPLETED. Not submitting as a dependency.")

        qdeps = np.atleast_1d(qdeps)
        proc_table['LATEST_DEP_QID'][rown] = qdeps
        if len(qdeps) < len(ideps):
            log.warning(f"Number of internal dependencies was {len(ideps)} but number "
                        + f"of queue deps is {len(qdeps)} for Rown {rown}, ideps {ideps}."
                        + " This is expected if the ideps were status=COMPLETED")

    proc_table[rown] = submit_batch_script(proc_table[rown], reservation=reservation,
                                           strictly_successful=True, dry_run=dry_run_level)
    submits += 1

    if dry_run_level < 3:
        if ptab_name is None:
            write_table(proc_table, tabletype='processing', overwrite=True)
        else:
            write_table(proc_table, tablename=ptab_name, overwrite=True)
        sleep_and_report(0.1 + 0.1*(submits % 10 == 0),
                         message_suffix=f"after submitting job to queue and writing proctable")
    return proc_table, submits


#########################################
########     Joint fit     ##############
#########################################
def joint_fit(ptable, prows, internal_id, queue, reservation, descriptor, z_submit_types=None,
              dry_run=0, strictly_successful=False, check_for_outputs=True, resubmit_partial_complete=True,
              system_name=None):
    """
    DEPRECATED
    Given a set of prows, this generates a processing table row, creates a batch script, and submits the appropriate
    joint fitting job given by descriptor. If the joint fitting job is standard star fitting, the post standard star fits
    for all the individual exposures also created and submitted. The returned ptable has all of these rows added to the
    table given as input.

    Args:
        ptable (Table): The processing table where each row is a processed job.
        prows (list or array of dict): The rows corresponding to the individual exposure jobs that are
            inputs to the joint fit.
        internal_id (int): the next internal id to be used for assignment (already incremented up from the last used id number used).
        queue (str): The name of the queue to submit the jobs to. If None is given the current desi_proc default is used.
        reservation (str): The reservation to submit jobs to. If None, it is not submitted to a reservation.
        descriptor (str): Description of the joint fitting job. Can either be 'science' or 'stdstarfit', 'arc' or 'psfnight',
            or 'flat' or 'nightlyflat'.
        z_submit_types (list of str, optional): The "group" types of redshifts that should be submitted with each
            exposure. If not specified or None, then no redshifts are submitted.
        dry_run (int, optional): If nonzero, this is a simulated run. Default is 0.
            0 which runs the code normally.
            1 writes all files but doesn't submit any jobs to Slurm.
            2 writes tables but doesn't write scripts or submit anything.
            3 Doesn't write or submit anything but queries Slurm normally for job status.
            4 Doesn't write, submit jobs, or query Slurm.
            5 Doesn't write, submit jobs, or query Slurm; instead it makes up the status of the jobs.
        strictly_successful (bool, optional): Whether all jobs require all inputs to have succeeded. For daily processing, this is
            less desirable because e.g. the sciences can run with SVN default calibrations rather
            than failing completely from failed calibrations. Default is False.
        check_for_outputs (bool, optional): Default is True. If True, the code checks for the existence of the expected final
            data products for the script being submitted. If all files exist and this is True,
            then the script will not be submitted. If some files exist and this is True, only the
            subset of the cameras without the final data products will be generated and submitted.
        resubmit_partial_complete (bool, optional): Default is True. Must be used with check_for_outputs=True. If this flag is True,
            jobs with some prior data are pruned using PROCCAMWORD to only process the
            remaining cameras not found to exist.
        system_name (str): batch system name, e.g. cori-haswell or perlmutter-gpu

    Returns:
        tuple: A tuple containing:

        * ptable, Table. The same processing table as input except with added rows for the joint fit job and, in the case
          of a stdstarfit, the poststdstar science exposure jobs.
        * joint_prow, dict. Row of a processing table corresponding to the joint fit job.
        * internal_id, int, the next internal id to be used for assignment (already incremented up from the last used id number used).
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
        joint_prow, internal_id = make_joint_prow(prows, descriptor='stdstarfit', internal_id=internal_id)
    else:
        joint_prow, internal_id = make_joint_prow(prows, descriptor=descriptor, internal_id=internal_id)
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

            # poststdstar job can't process cameras not included in its stdstar joint fit
            stdcamword = joint_prow['PROCCAMWORD']
            thiscamword = row['PROCCAMWORD']
            proccamword = camword_intersection([stdcamword, thiscamword])
            if proccamword != thiscamword:
                dropcams = difference_camwords(thiscamword, proccamword)
                assert dropcams != ''  #- i.e. if they differ, we should be dropping something
                log.warning(f"Dropping exp {row['EXPID']} poststdstar cameras {dropcams} since they weren't included in stdstar fit {stdcamword}")
                row['PROCCAMWORD'] = proccamword

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
                    joint_prow, internal_id = make_joint_prow([zprow], descriptor=zsubtype, internal_id=internal_id)
                    joint_prow = create_and_submit(joint_prow, queue=queue, reservation=reservation, joint=True, dry_run=dry_run,
                                                   strictly_successful=strictly_successful, check_for_outputs=check_for_outputs,
                                                   resubmit_partial_complete=resubmit_partial_complete, system_name=system_name)
                    ptable.add_row(joint_prow)
            else:
                log.info(" ")
                log.info(f"Submitting joint redshift fits of type {zsubtype} for TILEID {nightly_zprows[0]['TILEID']}.")
                expids = [prow['EXPID'][0] for prow in nightly_zprows]
                log.info(f"Expids: {expids}.\n")
                joint_prow, internal_id = make_joint_prow(nightly_zprows, descriptor=zsubtype, internal_id=internal_id)
                joint_prow = create_and_submit(joint_prow, queue=queue, reservation=reservation, joint=True, dry_run=dry_run,
                                               strictly_successful=strictly_successful, check_for_outputs=check_for_outputs,
                                               resubmit_partial_complete=resubmit_partial_complete, system_name=system_name)
                ptable.add_row(joint_prow)

    if descriptor in ['psfnight', 'nightlyflat']:
        log.info(f"Setting the calibration exposures as calibrators in the processing table.\n")
        ptable = set_calibrator_flag(prows, ptable)

    return ptable, joint_prow, internal_id

def joint_cal_fit(descriptor, ptable, prows, internal_id, queue, reservation,
                  dry_run=0, strictly_successful=False, check_for_outputs=True,
                  resubmit_partial_complete=True, system_name=None):
    """
    Given a set of prows, this generates a processing table row, creates a batch script, and submits the appropriate
    joint fitting job given by descriptor. If the joint fitting job is standard star fitting, the post standard star fits
    for all the individual exposures also created and submitted. The returned ptable has all of these rows added to the
    table given as input.

    Args:
        descriptor (str): Description of the joint fitting job. Can either be 'science' or 'stdstarfit', 'arc' or 'psfnight',
            or 'flat' or 'nightlyflat'.
        prows (list or array of dict): The rows corresponding to the individual exposure jobs that are
            inputs to the joint fit.
        ptable (Table): The processing table where each row is a processed job.
        internal_id (int): the next internal id to be used for assignment (already incremented up from the last used id number used).
        queue (str): The name of the queue to submit the jobs to. If None is given the current desi_proc default is used.
        reservation (str): The reservation to submit jobs to. If None, it is not submitted to a reservation.
        dry_run (int, optional): If nonzero, this is a simulated run. Default is 0.
            0 which runs the code normally.
            1 writes all files but doesn't submit any jobs to Slurm.
            2 writes tables but doesn't write scripts or submit anything.
            3 Doesn't write or submit anything but queries Slurm normally for job status.
            4 Doesn't write, submit jobs, or query Slurm.
            5 Doesn't write, submit jobs, or query Slurm; instead it makes up the status of the jobs.
        strictly_successful (bool, optional): Whether all jobs require all inputs to have succeeded. For daily processing, this is
            less desirable because e.g. the sciences can run with SVN default calibrations rather
            than failing completely from failed calibrations. Default is False.
        check_for_outputs (bool, optional): Default is True. If True, the code checks for the existence of the expected final
            data products for the script being submitted. If all files exist and this is True,
            then the script will not be submitted. If some files exist and this is True, only the
            subset of the cameras without the final data products will be generated and submitted.
        resubmit_partial_complete (bool, optional): Default is True. Must be used with check_for_outputs=True. If this flag is True,
            jobs with some prior data are pruned using PROCCAMWORD to only process the
            remaining cameras not found to exist.
        system_name (str): batch system name, e.g. cori-haswell or perlmutter-gpu

    Returns:
        tuple: A tuple containing:

        * ptable, Table. The same processing table as input except with added rows for the joint fit job and, in the case
          of a stdstarfit, the poststdstar science exposure jobs.
        * joint_prow, dict. Row of a processing table corresponding to the joint fit job.
        * internal_id, int, the next internal id to be used for assignment (already incremented up from the last used id number used).
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

    if descriptor not in ['psfnight', 'nightlyflat']:
        return ptable, None, internal_id

    log.info(" ")
    log.info(f"Joint fit criteria found. Running {descriptor}.\n")

    joint_prow, internal_id = make_joint_prow(prows, descriptor=descriptor, internal_id=internal_id)
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
        ptable (Table): The processing table where each row is a processed job.
        prows (list or array of dict): Unsubmitted prestdstar jobs that are first steps in tilenight.
        tnight (Table.Row): The processing table row of the tilenight job on which the redshifts depend.
        internal_id (int): the next internal id to be used for assignment (already incremented up from the last used id number used).
        queue (str): The name of the queue to submit the jobs to. If None is given the current desi_proc default is used.
        reservation (str): The reservation to submit jobs to. If None, it is not submitted to a reservation.
        dry_run (int, optional): If nonzero, this is a simulated run. Default is 0.
            0 which runs the code normally.
            1 writes all files but doesn't submit any jobs to Slurm.
            2 writes tables but doesn't write scripts or submit anything.
            3 Doesn't write or submit anything but queries Slurm normally for job status.
            4 Doesn't write, submit jobs, or query Slurm.
            5 Doesn't write, submit jobs, or query Slurm; instead it makes up the status of the jobs.
        strictly_successful (bool, optional): Whether all jobs require all inputs to have succeeded. For daily processing, this is
            less desirable because e.g. the sciences can run with SVN default calibrations rather
            than failing completely from failed calibrations. Default is False.
        check_for_outputs (bool, optional): Default is True. If True, the code checks for the existence of the expected final
            data products for the script being submitted. If all files exist and this is True,
            then the script will not be submitted. If some files exist and this is True, only the
            subset of the cameras without the final data products will be generated and submitted.
        resubmit_partial_complete (bool, optional): Default is True. Must be used with check_for_outputs=True. If this flag is True,
            jobs with some prior data are pruned using PROCCAMWORD to only process the
            remaining cameras not found to exist.
        z_submit_types (list of str): The "group" types of redshifts that should be submitted with each
            exposure. If not specified or None, then no redshifts are submitted.
        system_name (str): batch system name, e.g. cori-haswell or perlmutter-gpu

    Returns:
        tuple: A tuple containing:

        * ptable, Table. The same processing table as input except with added rows for the joint fit job.
        * internal_id, int, the next internal id to be used for assignment (already incremented up from the last used id number used).
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
            log.info(" ")
            log.info(f"Submitting joint redshift fits of type {zsubtype} for TILEID {zprows[0]['TILEID']}.")
            if zsubtype == 'perexp':
                for zprow in zprows:
                    log.info(f"EXPID: {zprow['EXPID']}.\n")
                    redshift_prow = make_redshift_prow([zprow], tnight, descriptor=zsubtype, internal_id=internal_id)
                    internal_id += 1
                    redshift_prow = create_and_submit(redshift_prow, queue=queue, reservation=reservation, joint=True, dry_run=dry_run,
                                                   strictly_successful=strictly_successful, check_for_outputs=check_for_outputs,
                                                   resubmit_partial_complete=resubmit_partial_complete, system_name=system_name)
                    ptable.add_row(redshift_prow)
            elif zsubtype == 'cumulative':
                tileids = np.unique([prow['TILEID'] for prow in zprows])
                if len(tileids) > 1:
                    msg = f"Error, more than one tileid provided for cumulative redshift job: {tileids}"
                    log.critical(msg)
                    raise ValueError(msg)
                nights = np.unique([prow['NIGHT'] for prow in zprows])
                if len(nights) > 1:
                    msg = f"Error, more than one night provided for cumulative redshift job: {nights}"
                    log.critical(msg)
                    raise ValueError(msg)
                tileid, night = tileids[0], nights[0]
                ## For cumulative redshifts, get any existing processing rows for tile
                matched_prows = read_minimal_tilenight_proctab_cols(tileids=tileids)
                ## Identify the processing rows that should be assigned as dependecies
                ## tnight should be first such that the new job inherits the other metadata from it
                tnights = [tnight]
                if matched_prows is not None:
                    matched_prows = matched_prows[matched_prows['NIGHT'] <= night]
                    for prow in matched_prows:
                        if prow['INTID'] != tnight['INTID']:
                            tnights.append(prow)
                log.info(f"Internal Processing IDs: {[prow['INTID'] for prow in tnights]}.\n")
                ## Identify all exposures that should go into the fit
                expids = [prow['EXPID'][0] for prow in zprows]
                ## note we can actually get the full list of exposures, but for now
                ## we'll stay consistent with old processing where we only list exposures
                ## from the current night
                ## For cumulative redshifts, get valid expids from exptables
                #matched_erows = read_minimal_science_exptab_cols(tileids=tileids)
                #matched_erows = matched_erows[matched_erows['NIGHT']<=night]
                #expids = list(set([prow['EXPID'][0] for prow in zprows])+set(matched_erows['EXPID']))
                log.info(f"Expids: {expids}.\n")
                redshift_prow, internal_id = make_joint_prow(tnights, descriptor=zsubtype, internal_id=internal_id)
                redshift_prow['EXPID'] = expids
                redshift_prow = create_and_submit(redshift_prow, queue=queue, reservation=reservation, joint=True, dry_run=dry_run,
                                               strictly_successful=strictly_successful, check_for_outputs=check_for_outputs,
                                               resubmit_partial_complete=resubmit_partial_complete, system_name=system_name)
                ptable.add_row(redshift_prow)
            else: # pernight
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
              system_name=None, use_specter=False, extra_job_args=None):
    """
    Given a set of prows, this generates a processing table row, creates a batch script, and submits the appropriate
    tilenight job given by descriptor. The returned ptable has all of these rows added to the
    table given as input.

    Args:
        ptable (Table): The processing table where each row is a processed job.
        prows (list or array of dict): Unsubmitted prestdstar jobs that are first steps in tilenight.
        calibjobs (dict): Dictionary containing 'nightlybias', 'ccdcalib', 'psfnight'
            and 'nightlyflat'. Each key corresponds to a Table.Row or
            None. The table.Row() values are for the corresponding
            calibration job.
        internal_id (int): the next internal id to be used for assignment (already incremented up from the last used id number used).
        queue (str): The name of the queue to submit the jobs to. If None is given the current desi_proc default is used.
        reservation (str): The reservation to submit jobs to. If None, it is not submitted to a reservation.
        dry_run (int, optional): If nonzero, this is a simulated run. Default is 0.
            0 which runs the code normally.
            1 writes all files but doesn't submit any jobs to Slurm.
            2 writes tables but doesn't write scripts or submit anything.
            3 Doesn't write or submit anything but queries Slurm normally for job status.
            4 Doesn't write, submit jobs, or query Slurm.
            5 Doesn't write, submit jobs, or query Slurm; instead it makes up the status of the jobs.
        strictly_successful (bool, optional): Whether all jobs require all inputs to have succeeded. For daily processing, this is
            less desirable because e.g. the sciences can run with SVN default calibrations rather
            than failing completely from failed calibrations. Default is False.
        resubmit_partial_complete (bool, optional): Default is True. Must be used with check_for_outputs=True. If this flag is True,
            jobs with some prior data are pruned using PROCCAMWORD to only process the
            remaining cameras not found to exist.
        system_name (str): batch system name, e.g. cori-haswell or perlmutter-gpu
        use_specter (bool, optional): Default is False. If True, use specter, otherwise use gpu_specter by default.
        extra_job_args (dict): Dictionary with key-value pairs that specify additional
            information used for a specific type of job. Examples include
            laststeps for for tilenight, etc.

    Returns:
        tuple: A tuple containing:

        * ptable, Table. The same processing table as input except with added rows for the joint fit job.
        * tnight_prow, dict. Row of a processing table corresponding to the tilenight job.
        * internal_id, int, the next internal id to be used for assignment (already incremented up from the last used id number used).
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
                                   use_specter=use_specter, extra_job_args=extra_job_args)
    ptable.add_row(tnight_prow)

    return ptable, tnight_prow, internal_id

## wrapper functions for joint fitting
def science_joint_fit(ptable, sciences, internal_id, queue='realtime', reservation=None,
                      z_submit_types=None, dry_run=0, strictly_successful=False,
                      check_for_outputs=True, resubmit_partial_complete=True,
                      system_name=None):
    """
    Wrapper function for desiproc.workflow.processing.joint_fit specific to the stdstarfit joint fit and redshift fitting.

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
    Wrapper function for desiproc.workflow.processing.joint_fit specific to the nightlyflat joint fit.

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
    Wrapper function for desiproc.workflow.processing.joint_fit specific to the psfnight joint fit.

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
        dict: Row of a processing table corresponding to the joint fit job.
        internal_id, int, the next internal id to be used for assignment (already incremented up from the last used id number used).
    """
    log = get_logger()
    first_row = table_row_to_dict(prows[0])
    joint_prow = first_row.copy()

    joint_prow['INTID'] = internal_id
    internal_id += 1
    joint_prow['JOBDESC'] = descriptor
    joint_prow['LATEST_QID'] = -99
    joint_prow['ALL_QIDS'] = np.ndarray(shape=0).astype(int)
    joint_prow['SUBMIT_DATE'] = -99
    joint_prow['STATUS'] = 'UNSUBMITTED'
    joint_prow['SCRIPTNAME'] = ''
    joint_prow['EXPID'] = np.unique(np.concatenate([currow['EXPID'] for currow in prows])).astype(int)

    ## Assign the PROCCAMWORD based on the descriptor and the input exposures
    ## UPDATE 2024-04-24: badamps are now included in arc/flat joint fits,
    ## so grab all PROCCAMWORDs instead of filtering out BADAMP cameras
    ## For flats we want any camera that exists in all 12 exposures
    ## For arcs we want any camera that exists in at least 3 exposures
    pcamwords = [prow['PROCCAMWORD'] for prow in prows]
    if descriptor in 'stdstarfit':
        joint_prow['PROCCAMWORD'] = camword_union(pcamwords,
                                                  full_spectros_only=True)
    elif descriptor in ['pernight', 'cumulative']:
        joint_prow['PROCCAMWORD'] = camword_union(pcamwords,
                                                  full_spectros_only=False)
    elif descriptor == 'nightlyflat':
        joint_prow['PROCCAMWORD'] = camword_intersection(pcamwords,
                                                         full_spectros_only=False)
    elif descriptor == 'psfnight':
        ## Count number of exposures each camera is present for
        camcheck = {}
        for camword in pcamwords:
            for cam in decode_camword(camword):
                if cam in camcheck:
                    camcheck[cam] += 1
                else:
                    camcheck[cam] = 1
        ## if exists in 3 or more exposures, then include it
        goodcams = []
        for cam,camcount in camcheck.items():
            if camcount >= 3:
                goodcams.append(cam)
        joint_prow['PROCCAMWORD'] = create_camword(goodcams)
    else:
        log.warning("Warning asked to produce joint proc table row for unknown"
                    + f" job description {descriptor}")

    joint_prow = assign_dependency(joint_prow, dependency=prows)
    return joint_prow, internal_id

def make_exposure_prow(erow, int_id, calibjobs, jobdesc=None):
    prow = erow_to_prow(erow)
    prow['INTID'] = int_id
    int_id += 1
    if jobdesc is None:
        prow['JOBDESC'] = prow['OBSTYPE']
    else:
        prow['JOBDESC'] = jobdesc
    prow = define_and_assign_dependency(prow, calibjobs)
    return prow, int_id

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
        dict: Row of a processing table corresponding to the tilenight job.
    """
    first_row = table_row_to_dict(prows[0])
    joint_prow = first_row.copy()

    joint_prow['INTID'] = internal_id
    joint_prow['JOBDESC'] = 'tilenight'
    joint_prow['LATEST_QID'] = -99
    joint_prow['ALL_QIDS'] = np.ndarray(shape=0).astype(int)
    joint_prow['SUBMIT_DATE'] = -99
    joint_prow['STATUS'] = 'UNSUBMITTED'
    joint_prow['SCRIPTNAME'] = ''
    joint_prow['EXPID'] = np.array([currow['EXPID'][0] for currow in prows], dtype=int)

    joint_prow = define_and_assign_dependency(joint_prow, calibjobs, use_tilenight=True)

    return joint_prow

def make_redshift_prow(prows, tnights, descriptor, internal_id):
    """
    Given an input list or array of processing table rows and a descriptor, this creates a joint fit processing job row.
    It starts by copying the first input row, overwrites relevant columns, and defines the new dependencies (based on the
    input prows).

    Args:
        prows, list or array of dicts. Unsumbitted rows corresponding to the individual prestdstar jobs that are
            the first steps of tilenight.
        tnights, list or array of Table.Row objects. Rows corresponding to the tilenight jobs on which the redshift job depends.
        internal_id, int, the next internal id to be used for assignment (already incremented up from the last used id number used).

    Returns:
        dict: Row of a processing table corresponding to the tilenight jobs.
    """
    first_row = table_row_to_dict(prows[0])
    redshift_prow = first_row.copy()

    redshift_prow['INTID'] = internal_id
    redshift_prow['JOBDESC'] = descriptor
    redshift_prow['LATEST_QID'] = -99
    redshift_prow['ALL_QIDS'] = np.ndarray(shape=0).astype(int)
    redshift_prow['SUBMIT_DATE'] = -99
    redshift_prow['STATUS'] = 'UNSUBMITTED'
    redshift_prow['SCRIPTNAME'] = ''
    redshift_prow['EXPID'] = np.array([currow['EXPID'][0] for currow in prows], dtype=int)

    redshift_prow = assign_dependency(redshift_prow,dependency=tnights)

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
        ptable (Table): Processing table of all exposures that have been processed.
        arcs (list of dict): list of the individual arc jobs to be used for the psfnight (NOT all
            the arcs, if multiple sets existed). May be empty if none identified yet.
        flats (list of dict): list of the individual flat jobs to be used for the nightlyflat (NOT
            all the flats, if multiple sets existed). May be empty if none identified yet.
        sciences (list of dict): list of the most recent individual prestdstar science exposures
            (if currently processing that tile). May be empty if none identified yet.
        calibjobs (dict): Dictionary containing 'nightlybias', 'ccdcalib', 'psfnight'
            and 'nightlyflat'. Each key corresponds to a Table.Row or
            None. The table.Row() values are for the corresponding
            calibration job.
        lasttype (str or None): the obstype of the last individual exposure row to be processed.
        internal_id (int): an internal identifier unique to each job. Increments with each new job. This
            is the smallest unassigned value.
        z_submit_types (list of str): The "group" types of redshifts that should be submitted with each
            exposure. If not specified or None, then no redshifts are submitted.
        dry_run (int, optional): If nonzero, this is a simulated run. Default is 0.
            0 which runs the code normally.
            1 writes all files but doesn't submit any jobs to Slurm.
            2 writes tables but doesn't write scripts or submit anything.
            3 Doesn't write or submit anything but queries Slurm normally for job status.
            4 Doesn't write, submit jobs, or query Slurm.
            5 Doesn't write, submit jobs, or query Slurm; instead it makes up the status of the jobs.
        queue (str, optional): The name of the queue to submit the jobs to. If None is given the current desi_proc default is used.
        reservation (str, optional): The reservation to submit jobs to. If None, it is not submitted to a reservation.
        strictly_successful (bool, optional): Whether all jobs require all inputs to have succeeded. For daily processing, this is
            less desirable because e.g. the sciences can run with SVN default calibrations rather
            than failing completely from failed calibrations. Default is False.
        check_for_outputs (bool, optional): Default is True. If True, the code checks for the existence of the expected final
            data products for the script being submitted. If all files exist and this is True,
            then the script will not be submitted. If some files exist and this is True, only the
            subset of the cameras without the final data products will be generated and submitted.
        resubmit_partial_complete (bool, optional): Default is True. Must be used with check_for_outputs=True. If this flag is True,
            jobs with some prior data are pruned using PROCCAMWORD to only process the
            remaining cameras not found to exist.
        system_name (str): batch system name, e.g. cori-haswell, cori-knl, permutter-gpu

    Returns:
        tuple: A tuple containing:

        * ptable, Table, Processing table of all exposures that have been processed.
        * calibjobs, dict. Dictionary containing 'nightlybias', 'ccdcalib', 'psfnight'
          and 'nightlyflat'. Each key corresponds to a Table.Row or
          None. The table.Row() values are for the corresponding
          calibration job.
        * sciences, list of dicts, list of the most recent individual prestdstar science exposures
          (if currently processing that tile). May be empty if none identified yet or
          we just submitted them for processing.
        * internal_id, int, if no job is submitted, this is the same as the input, otherwise it is incremented upward from
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

def submit_tilenight_and_redshifts(ptable, sciences, calibjobs, internal_id, dry_run=0,
                                  queue='realtime', reservation=None, strictly_successful=False,
                                  check_for_outputs=True, resubmit_partial_complete=True,
                                  system_name=None,use_specter=False, extra_job_args=None):
    """
    Takes all the state-ful data from daily processing and determines whether a tilenight job needs to be submitted.

    Args:
        ptable (Table): Processing table of all exposures that have been processed.
        sciences (list of dict): list of the most recent individual prestdstar science exposures
            (if currently processing that tile). May be empty if none identified yet.
        internal_id (int): an internal identifier unique to each job. Increments with each new job. This
            is the smallest unassigned value.
        dry_run (int, optional): If nonzero, this is a simulated run. Default is 0.
            0 which runs the code normally.
            1 writes all files but doesn't submit any jobs to Slurm.
            2 writes tables but doesn't write scripts or submit anything.
            3 Doesn't write or submit anything but queries Slurm normally for job status.
            4 Doesn't write, submit jobs, or query Slurm.
            5 Doesn't write, submit jobs, or query Slurm; instead it makes up the status of the jobs.
        queue (str, optional): The name of the queue to submit the jobs to. If None is given the current desi_proc default is used.
        reservation (str, optional): The reservation to submit jobs to. If None, it is not submitted to a reservation.
        strictly_successful (bool, optional): Whether all jobs require all inputs to have succeeded. For daily processing, this is
            less desirable because e.g. the sciences can run with SVN default calibrations rather
            than failing completely from failed calibrations. Default is False.
        check_for_outputs (bool, optional): Default is True. If True, the code checks for the existence of the expected final
            data products for the script being submitted. If all files exist and this is True,
            then the script will not be submitted. If some files exist and this is True, only the
            subset of the cameras without the final data products will be generated and submitted.
        resubmit_partial_complete (bool, optional): Default is True. Must be used with check_for_outputs=True. If this flag is True,
            jobs with some prior data are pruned using PROCCAMWORD to only process the
            remaining cameras not found to exist.
        system_name (str): batch system name, e.g. cori-haswell, cori-knl, permutter-gpu
        use_specter (bool, optional): Default is False. If True, use specter, otherwise use gpu_specter by default.
        extra_job_args (dict, optional): Dictionary with key-value pairs that specify additional
            information used for a specific type of job. Examples include
            laststeps for tilenight, z_submit_types for redshifts, etc.

    Returns:
        tuple: A tuple containing:

        * ptable, Table, Processing table of all exposures that have been processed.
        * sciences, list of dicts, list of the most recent individual prestdstar science exposures
          (if currently processing that tile). May be empty if none identified yet or
          we just submitted them for processing.
        * internal_id, int, if no job is submitted, this is the same as the input, otherwise it is incremented upward from
          from the input such that it represents the smallest unused ID.
    """
    ptable, tnight, internal_id = submit_tilenight(ptable, sciences, calibjobs, internal_id,
                                             queue=queue, reservation=reservation,
                                             dry_run=dry_run, strictly_successful=strictly_successful,
                                             resubmit_partial_complete=resubmit_partial_complete,
                                             system_name=system_name,use_specter=use_specter,
                                             extra_job_args=extra_job_args)

    z_submit_types = None
    if 'z_submit_types'  in extra_job_args:
        z_submit_types = extra_job_args['z_submit_types']
        
    ptable, internal_id = submit_redshifts(ptable, sciences, tnight, internal_id,
                                    queue=queue, reservation=reservation,
                                    dry_run=dry_run, strictly_successful=strictly_successful,
                                    check_for_outputs=check_for_outputs,
                                    resubmit_partial_complete=resubmit_partial_complete,
                                    z_submit_types=z_submit_types,
                                    system_name=system_name)

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
        Table: The same processing table as input except with added rows for the joint fit job and, in the case
        of a stdstarfit, the poststdstar science exposure jobs.
    """
    for prow in prows:
        ptable['CALIBRATOR'][ptable['INTID'] == prow['INTID']] = 1
    return ptable
