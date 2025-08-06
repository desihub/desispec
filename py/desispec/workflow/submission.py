"""
desispec.workflow.submision
=======================

Utilities for working submitting jobs to slurm 
"""

import os
from importlib import resources
import yaml

from desispec.workflow.tableio import load_table, write_table
from astropy.table import Table, vstack
from desispec.scripts.compute_dark import compute_dark_parser, get_stacked_dark_exposure_table
from desispec.workflow.proctable import default_prow, get_pdarks_from_ptable
import vstack
import nunpy as np

from desispec.io.util import all_impacted_cameras, columns_to_goodcamword, erow_to_goodcamword
from desispec.scripts.link_calibnight import derive_include_exclude

from desispec.io.meta import findfile
from desispec.workflow.processing import create_and_submit, define_and_assign_dependency, generate_calibration_dict, night_to_starting_iid
from desispec.workflow.utils import load_override_file, sleep_and_report
from desiutil.log import get_logger


def submit_linkcal_jobs(night, ptable, cal_override=None, override_pathname=None,
                        psf_linking_without_fflat=False, proccamword='a0123456789',
                        dry_run_level=0, queue=None, reservation=None,
                        check_outputs=True, system_name=None):
    """
    Submit linkcal jobs for the current night.
    This function will read the override file, determine what calibrations
    have been done, and submit jobs to link the calibrations that are
    specified in the override file.
    It will also update the processing table with the new jobs.
    Args:
        night (int): The night to process, in YYYYMMDD format.
        ptable (Table): Processing table to update with new jobs.
        cal_override (dict, optional): Dictionary of calibration overrides.
                                       If None, will read from override file.
        override_pathname (str, optional): Path to the override file.
                                           If None, will search for it.
        psf_linking_without_fflat (bool, optional): If True, allows linking
                                                    psfnight without fiberflatnight.
        proccamword (str, optional): Camera word defining the cameras to process.
        dry_run_level (int, optional): Level of dry run to perform. Default is 0.
        queue (str, optional): Slurm queue to submit jobs to. Default is None.
        reservation (str, optional): Slurm reservation to use. Default is None.
        check_outputs (bool, optional): If True, checks for job outputs before submitting. Default is True.
        system_name (str, optional): Name of the system to use for batch submission. Default is None.

    Returns:
        ptable (Table): Updated processing table with new jobs.
        files_to_link (set): Set of calibration files that will be linked.
    """
    log = get_logger()

    if cal_override is None:
        ## Require cal_override to exist if explcitly specified
        if override_pathname is None:
            override_pathname = findfile('override', night=night)
        elif not os.path.exists(override_pathname):
            raise IOError(f"Specified override file: "
                        f"{override_pathname} not found. Exiting this night.")
        ## Load calibration_override_file
        overrides = load_override_file(filepathname=override_pathname)
        cal_override = {}
        if 'calibration' in overrides:
            cal_override = overrides['calibration']

    if len(ptable) > 0:
        int_id = np.max(ptable['INTID'])+1
    else:
        int_id = night_to_starting_iid(night=night)

    ## Determine calibrations that will be linked
    if 'linkcal' in cal_override:
        files_to_link, files_not_linked = None, None
        if 'include' in cal_override['linkcal']:
            files_to_link = cal_override['linkcal']['include']
        if 'exclude' in cal_override['linkcal']:
            files_not_linked = cal_override['linkcal']['exclude']
        files_to_link, files_not_linked = derive_include_exclude(files_to_link,                             
                                                                 files_not_linked)
        ## Fiberflatnights need to be generated with psfs from same time, so
        ## can't link psfs without also linking fiberflatnight
        if 'psfnight' in files_to_link and not 'fiberflatnight' in files_to_link \
                and not psf_linking_without_fflat:
            err = "Must link fiberflatnight if linking psfnight"
            log.error(err)
            raise ValueError(err)
    else:
        files_to_link = set()

    submitted = False
    if 'linkcal' in cal_override and 'linkcal' not in ptable['JOBDESC']:
        log.info("Linking calibration files listed in override files: "
                 + f"{files_to_link}")
        prow = default_prow()
        prow['INTID'] = int_id
        int_id += 1
        prow['JOBDESC'] = 'linkcal'
        prow['OBSTYPE'] = 'link'
        prow['CALIBRATOR'] = 1
        prow['NIGHT'] = night
        if 'refnight' in cal_override['linkcal']:
            refnight = int(cal_override['linkcal']['refnight'])
            prow = define_and_assign_dependency(prow, ptable, refnight=refnight)
        if 'camword' in cal_override['linkcal']:
            prow['PROCCAMWORD'] = cal_override['linkcal']['camword']
        else:
            ## If no camword is specified, use the provided camword,
            ## or if not provided, use default to all cameras
            prow['PROCCAMWORD'] = proccamword

        ## create dictionary to carry linking information
        linkcalargs = cal_override['linkcal']
        log.info(f"\nProcessing: {prow}\n")
        prow = create_and_submit(prow, dry_run=dry_run_level, queue=queue,
                                 reservation=reservation,
                                 strictly_successful=True,
                                 check_for_outputs=check_outputs,
                                 system_name=system_name,
                                 extra_job_args=linkcalargs)
        ## Add the processing row to the processing table
        ptable.add_row(prow)

        return ptable, files_to_link


def submit_biasnight_and_preproc_darks(night, dark_expids, proc_obstypes, 
                           camword, badcamword, badamps=None,
                           exp_table_path=None,
                           proc_table_path=None,
                           override_path=None,
                           psf_linking_without_fflat=False,
                           queue=None, reservation=None,
                           check_for_outputs=True, system_name=None,
                           specprod=None, path_to_data=None,
                           sub_wait_time=0.1, dry_run_level=0):
    """
    Submit a biasnight and/or preproc_darks jobs for the given night.
    This function will read the override file, determine what calibrations
    have been done, and submit jobs to process the bias and dark frames.
    
    Args:
        night (int): The night to process, in YYYYMMDD format.
        dark_expids (list): List of exposure IDs for the dark frames to process.
        proc_obstypes (list): List of obstypes to process.
        camword (str): Camera word defining the cameras to process.
        badcamword (str): Camera word defining the bad cameras.
        badamps (list, optional): List of bad amps to exclude. Default is None.
        exp_table_path (str, optional): Path to the exposure table file.
                                            If None, will search for it.
        proc_table_path (str, optional): Path to the processing table file.
                                            If None, will search for it.
        override_path (str, optional): Path to the override file.
                                            If None, will search for it.
        psf_linking_without_fflat (bool, optional): If True, allows linking
                                                    psfnight without fiberflatnight.
        queue (str, optional): Slurm queue to submit jobs to. Default is None.
        reservation (str, optional): Slurm reservation to use. Default is None.
        check_for_outputs (bool, optional): If True, checks for job outputs before submitting. Default is True.
        system_name (str, optional): Name of the system to use for batch submission. Default is None.
        specprod (str, optional): Name of the spectroscopic production. Default is None.
        path_to_data (str, optional): Path to the data directory. Default is None.
        sub_wait_time (float, optional): Time to wait between submissions. Default is 0.1 seconds.
        dry_run_level (int, optional): Level of dry run to perform. Default is 0.
 
    Returns:
        ptable (Table): Updated processing table with new jobs.
    """
    log = get_logger()
   
    ## Determine where the processing table will be written
    if proc_table_path is None:
        proc_table_path = os.path.dirname(proc_table_pathname)
    else:
        proc_table_name = os.path.basename(findfile('processing_table', night=night, readonly=True))
        proc_table_pathname = os.path.join(proc_table_path, proc_table_name)
    if dry_run_level < 3:
        os.makedirs(proc_table_path, exist_ok=True)

    ## Load in the files defined above
    ptable = load_table(tablename=proc_table_pathname, tabletype='proctable')

    dark_expid_to_process = np.array([], dtype=int)
    if len(ptable) > 0:
        processed_dark_expids = get_pdarks_from_ptable(ptable)
        dark_expid_to_process = np.setdiff1d(dark_expids, processed_dark_expids)

    if len(ptable) > 0 and 'biaspdark' in ptable['JOBDESC'] and len(dark_expid_to_process) == 0:
        log.info(f"Bias and all preproc darks are already accounted for on {night=}.")
        return ptable

    ## Determine where the exposure table will be written
    if exp_table_path is None:
        exp_table_pathname = findfile('exposure_table', night=night, readonly=True)
        exp_table_path = os.path.dirname(exp_table_pathname)
    else:
        exp_table_pathname = os.path.join(exp_table_path, f"{night//100}", f'exposure_table_{night}.csv')
    if not os.path.exists(exp_table_pathname):
        raise IOError(f"Exposure table: {exp_table_pathname} not found. Exiting this night.")
    
     ## Load in the files defined above
    etable = load_table(tablename=exp_table_pathname, tabletype='exptable')

    ## Require cal_override to exist if explcitly specified
    if override_path is None:
        override_pathname = findfile('override', night=night, readonly=True)
        override_path = os.path.dirname(override_pathname)
    else:
        override_pathname = os.path.join(override_path, f'override_{night}.yaml')
        if not os.path.exists(override_pathname):
            raise IOError(f"Specified override file: "
                        f"{override_pathname} not found. Exiting this night.")
        
    ## Load calibration_override_file
    overrides = load_override_file(filepathname=override_pathname)
    cal_override = {}
    if 'calibration' in overrides:
        cal_override = overrides['calibration']

    ## Identify what calibrations have been done
    if 'linkcal' in cal_override and 'linkcal' not in ptable['JOBDESC']:
        proccamword = difference_camwords(camword, badcamword)
        ptable, files_to_link = submit_linkcal_jobs(night, ptable, cal_override=cal_override,
                        psf_linking_without_fflat=psf_linking_without_fflat, proccamword=proccamword,
                        dry_run_level=dry_run_level, queue=queue, reservation=reservation,
                        check_outputs=check_for_outputs, system_name=system_name)
        if len(ptable) > 0 and dry_run_level < 3:
            write_table(ptable, tablename=proc_table_pathname, tabletype='proctable')
            sleep_and_report(sub_wait_time,
                             message_suffix=f"to slow down the queue submission rate",
                             dry_run=dry_run_level>0, logfunc=log.info)

    zero_expids = np.array(etable[etable['OBSTYPE'] == 'zero']['EXPID'].data, dtype=int)

    linked_bias = 'biasnight' in files_to_link
    dobias = (not linked_bias) and ('biaspdark' not in ptable['JOBTYPE']) and 'zero' in proc_obstypes and len(zero_expids) > 0
    dodarks = 'dark' in proc_obstypes and len(dark_expid_to_process) > 0 # 'darknight' not in files_to_link and

    # dep = None
    # if linked_bias:
    #     if 'linkcal' in ptable['JOBDESC']:
    #         dep = [ptable[ptable['JOBDESC'] == 'linkcal'][0]]
    #     else:
    #         log.critical("Bias is supposed to be linked, but no linkcal job found in processing table.")
    #         raise ValueError("Bias is supposed to be linked, but no linkcal job found in processing table.")
    # elif dodarks and not dobias:
    #     if 'biaspdark' in ptable['JOBDESC']:
    #         dep = [ptable[ptable['JOBDESC'] == 'biaspdark'][0]]
    #     else:
    #         log.critical("Darks are supposed to be preprocessed, but bias job can be accounted for.")
    #         raise ValueError("Darks are supposed to be preprocessed, but bias job can be accounted for.")

    extra_job_args = {'steps': []}
    if dobias:
        extra_job_args['steps'].append('biasnight')
    if dodarks:
        extra_job_args['steps'].append('pdark')

    if len(ptable) > 0:
        int_id = np.max(ptable['INTID'])+1
    else:
        int_id = night_to_starting_iid(night=night)
    prow = None
    ## If submit bias and darks, submit joint job
    if dobias and dodarks:
        log.info(f"Submitting biasnight and preproc_darks for night {night}.")
        prow = default_prow()
        prow['INTID'] = int_id
        prow['JOBDESC'] = 'biaspdark'
        prow['OBSTYPE'] = 'dark'
        prow['CALIBRATOR'] = 1
        prow['NIGHT'] = night
        prow['EXPID'] = dark_expid_to_process
        prow = define_and_assign_dependency(prow, ptable)
        prow['PROCCAMWORD'] = columns_to_goodcamword(camword, badcamword, badamps, 
                                                     suppress_logging=True, exclude_badamps=True)
    elif dobias:
        log.info(f"Submitting biaspdark for night {night}.")
        prow = default_prow()
        prow['INTID'] = int_id
        prow['JOBDESC'] = 'biaspdark'
        prow['OBSTYPE'] = 'zero'
        prow['CALIBRATOR'] = 1
        prow['NIGHT'] = night
        prow['EXPID'] = zero_expids[:1] # set as first zero expid
        prow = define_and_assign_dependency(prow, ptable)
        prow['PROCCAMWORD'] = columns_to_goodcamword(camword, badcamword, badamps, 
                                                     suppress_logging=True, exclude_badamps=True)   
    elif dodarks:
        log.info(f"Submitting pdark for night {night}.")
        prow = default_prow()
        prow['INTID'] = int_id
        prow['JOBDESC'] = 'pdark'
        prow['OBSTYPE'] = 'dark'
        prow['CALIBRATOR'] = 1
        prow['NIGHT'] = night
        prow['EXPID'] = dark_expid_to_process
        prow = define_and_assign_dependency(prow, ptable)
        prow['PROCCAMWORD'] = columns_to_goodcamword(camword, badcamword, badamps, 
                                                     suppress_logging=True, exclude_badamps=True) 

    if prow is not None:
        prow = create_and_submit(prow, dry_run=dry_run_level, queue=queue,
                                    reservation=reservation,
                                    strictly_successful=True,
                                    check_for_outputs=True,
                                    system_name=system_name,
                                    extra_job_args=extra_job_args)
        ## Add the processing row to the processing table
        ptable.add_row(prow)
        if len(ptable) > 0 and dry_run_level < 3:
            write_table(ptable, tablename=proc_table_pathname, tabletype='proctable')
        sleep_and_report(sub_wait_time,
                            message_suffix=f"to slow down the queue submission rate",
                            dry_run=(dry_run_level>0), logfunc=log.info)
        
        ## Add the processing row to the processing table
        ptable.add_row(prow)
        log.info(f"Successfully submitted {prow['JOBDESC']} job submitted for night {night}.")
    else:
        log.info(f"No biasnight or preproc_darks jobs submitted for night {night}.")
    
    return ptable


def submit_necessary_biasnights_and_preproc_darks(reference_night, proc_obstypes, 
                           camword, badcamword, badamps=None,
                           exp_table_pathname=None,
                           proc_table_pathname=None,
                           specprod=None, path_to_data=None,
                           sub_wait_time=0.1, dry_run_level=0,
                           n_nights_before=None, n_nights_after=None):
    """
    Submit biasnight and preproc_darks jobs for the given reference night.
    This function will read the override file, determine what calibrations
    have been done, and submit jobs to process the bias and dark frames.
    
    Args:
        reference_night (int): The reference night to process, in YYYYMMDD format.
        proc_obstypes (list): List of obstypes to process.
        camword (str): Camera word defining the cameras to process.
        badcamword (str): Camera word defining the bad cameras.
        badamps (list, optional): List of bad amps to exclude. Default is None.
        exp_table_pathname (str, optional): Path to the exposure table file.
                                            If None, will search for it.
        proc_table_pathname (str, optional): Path to the processing table file.
                                             If None, will search for it.
        specprod (str, optional): Name of the spectroscopic production. Default is None.
        path_to_data (str, optional): Path to the data directory. Default is None.
        sub_wait_time (float, optional): Time to wait between submissions. Default is 0.1 seconds.
        dry_run_level (int, optional): Level of dry run to perform. Default is 0.
        n_nights_before (int, optional): Number of nights before the reference night to process. Default is None.
        n_nights_after (int, optional): Number of nights after the reference night to process. Default is None.

    Returns:
        ptable (Table): Updated processing table with new jobs.
    """
    log = get_logger()
    compdarkparser = compute_dark_parser()
    options = ['--reference-night', str(reference_night)]
    if n_nights_before is not None:
        options.extend(['--before', str(n_nights_before)])
    if n_nights_after is not None:
        options.extend(['--after', str(n_nights_after)])
    compdarkargs = compdarkparser.parse_args(options=options)

    exptab_for_dark_night = get_stacked_dark_exposure_table(compdarkargs, skip_camera_check=True)
    
    refnight_ptable = None
    for night in np.unique(exptab_for_dark_night['NIGHT']):
        log.info(f"Processing night {night} for biasnight and preproc_darks.")
        dark_expids = np.array(exptab_for_dark_night[exptab_for_dark_night['NIGHT'] == night]['EXPID'].data, dtype=int)
        ptable = submit_biasnight_and_preproc_darks(
            night=night, dark_expids=dark_expids, proc_obstypes=proc_obstypes,
            camword=camword, badcamword=badcamword, badamps=badamps,
            exp_table_pathname=exp_table_pathname,
            proc_table_pathname=proc_table_pathname,
            specprod=specprod, path_to_data=path_to_data,
            sub_wait_time=sub_wait_time, dry_run_level=dry_run_level)
        if night == reference_night:
            refnight_ptable = ptable
    
    return refnight_ptable
