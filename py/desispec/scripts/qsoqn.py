"""
desispec.scripts.qsoqn
======================

"""

import os
import sys
import glob
import time
import argparse

import fitsio
import numpy as np
# import pandas as pd
from astropy.table import Table

from desitarget.targets import main_cmx_or_sv
from desitarget.targetmask import desi_mask
from desitarget.sv3.sv3_targetmask import desi_mask as sv3_mask
from desitarget.sv2.sv2_targetmask import desi_mask as sv2_mask
from desitarget.sv1.sv1_targetmask import desi_mask as sv1_mask
from desitarget.cmx.cmx_targetmask import cmx_mask

from desiutil.log import get_logger
import desiutil.depend

from desispec.io.util import get_tempfilename

from redrock.templates import find_templates
from redrock.external.desi import rrdesi

from quasarnp.io import load_model, load_desi_coadd
from quasarnp.utils import process_preds

def get_default_qso_templates():
    """Return list of default Redrock QSO template filenames to use"""
    log = get_logger()
    all_templates = find_templates()
    qso_templates = list()
    for filename in all_templates:
        if os.path.basename(filename).startswith('rrtemplate-QSO-'):
            qso_templates.append(filename)

    #- Expect HIZ and LOZ but not more than that
    #- otherwise log error but don't actually crash
    if len(qso_templates) != 2:
        log.error(f'Unexpected number of QSO templates found: {qso_templates}')
    else:
        log.debug('Using Redrock templates %s', qso_templates)

    return qso_templates


def parse(options=None):
    parser = argparse.ArgumentParser(description="Run QN and rerun RR (only for SPECTYPE != QSO or for SPECTYPE == QSO && |z_QN - z_RR| > 0.05) on a coadd file to find true quasars with correct redshift")

    parser.add_argument("--coadd", type=str, required=True,
                        help="coadd file containing spectra")
    parser.add_argument("--redrock", type=str, required=True,
                        help="redrock file associated (in the same folder) to the coadd file")
    parser.add_argument("--output", type=str, required=True,
                        help="output filename where the result of the QN will be saved")

    parser.add_argument("--target_selection", type=str, required=False, default="qso_targets",
                        help="on which sample the QN is performed: \
                             qso_targets (QSO targets) -- qso (works also) \
                             / all_targets (All targets in the coadd file) -- all (works also)")
    parser.add_argument("--save_target", type=str, required=False, default="restricted",
                        help="which objects will be saved in the output files: \
                             restricted (objects which are identify as QSO by QN and where we have a new run of RR) \
                              / all (All objects which are tested by QN \
                             --> To have coadd.size objects in the ouput file: set --target_selection all_targets --save_target all")

    # for QN
    parser.add_argument("--c_thresh", type=float, required=False, default=0.5,
                        help="For QN: is_qso_QN =  np.sum(c_line > c_thresh, axis=0) >= n_thresh")
    parser.add_argument("--n_thresh", type=int, required=False, default=1,
                        help="For QN: is_qso_QN =  np.sum(c_line > c_thresh, axis=0) >= n_thresh")

    # for RR
    parser.add_argument("--templates", type=str, nargs='+', required=False,
                        help="give the templates used during the new run of RR\
                              By default use the templates from redrock of the form rrtemplate-QSO-*.fits")
    parser.add_argument("--filename_priors", type=str, required=False, default=None,
                        help="filename for the RR prior file, by default use the directory of coadd file")
    parser.add_argument("--filename_output_rerun_RR", type=str, required=False, default=None,
                        help="filename for the output of the new run of RR, by default use the directory of coadd file")
    parser.add_argument("--filename_redrock_rerun_RR", type=str, required=False, default=None,
                        help="filename for the redrock file of the new run of RR, by default use the directory of coadd file")
    parser.add_argument("--delete_RR_output", type=str, required=False, default='True',
                        help="delete RR outputs: True or False, they are useless since everything usefull are saved in output, by defaut:True")

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    if args.templates is None:
        args.templates = get_default_qso_templates()

    return args


def collect_redshift_with_new_RR_run(spectra_name, targetid, z_qn, z_prior, param_RR, comm=None):
    """
    Wrapper to run Redrock on targetid (numpy array) from the spectra_name_file
    with z_prior using the template contained in template_file

    Parameters
    ----------
    spectra_name : str
        The name of the spectra file.
    targetid : int array
        Array of the targetid (contained in the spectra_name_file)
        on which RR will be rerun with prior and qso template.
    z_qn : float array
        Array of the same size as targetid with the
        redshift estimated by QN for the associated targetid
    z_prior : float array
        Array of the same size as targetid with the
        redshift prior for the associated targetid
    param_RR : dict
        Contains info to re-run RR as the template_filename,
        filename_priors, filename_output_rerun_RR, filename_redrock_rerun_RR
    comm : object, optional
        MPI communicator to pass to redrock; must be size=1

    Returns
    -------
    Table of redrock rerun results
    """
    log = get_logger()
    def write_prior_for_RR(targetid, z_qn, z_prior, filename_priors):
        """
        Write the prior file for RR associated to the targetid list

        Args:
            targetid (int array): array of the targetid
                on which RR will be rerun with prior and qso template.
            z_qn (float array): array of the same size as targetid with the
                redshift estimated by QN for the associated targetid
            z_prior (float array): Array of the same size as targetid with the
                redshift prior for the associated targetid
            filename_priors (str): name under which the file will be saved
        """
        function = np.array(['tophat'] * z_prior.size)  # need to be the same for every one
        # save
        out = fitsio.FITS(filename_priors, 'rw', clobber=True)
        data, names, extname = [targetid, function, z_qn, z_prior], ['TARGETID', 'FUNCTION', 'Z', 'SIGMA'], 'PRIORS'
        out.write(data, names=names, extname=extname)
        out.close()
        log.debug(f'Write prior file for RR with {z_prior.size} objects: {filename_priors}')
        return

    def extract_redshift_info_from_RR(filename_redrock, targetid):
        """
        extract information of the redrock file from the new RR run

        Args:
           filename_redrock (str): Name of the redrock file from the new run of RR
           targetid (int array): array of the targetid (contained in the spectra_name_file)
                on which RR will be rerun with prior and qso template.

        Returns:
            Redrock output table ordered by input `targetid` list
        """
        with fitsio.FITS(filename_redrock) as redrock:
            # 9 July 2021:
            # The new run of RR does not save the targetid in the correct order ...
            # The TARGETID from REDSHIFTS HDU and FIBERMAP HDU are not the same
            # To avoid any kind of problem in the future --> sort redrock

            rr = redrock['REDSHIFTS'].read()
            redrock_tgid = rr['TARGETID']

            # targetid.size is the number of target in new-run redrock file
            log.info('SANITY CHECK: Match the order of the REDSHIFTS HDU from new RR run with the original order of targetid')
            correct_index = np.zeros(targetid.size, dtype=int)
            for i, tgid in enumerate(targetid):
                correct_index[i] = int(np.where(redrock_tgid == tgid)[0])

            rr_reordered = rr[correct_index]

        return rr_reordered

    # define the output
    redshift, err_redshift, chi2, coeffs = np.zeros(targetid.size), np.zeros(targetid.size), np.inf * np.ones(targetid.size), np.zeros((targetid.size, 10))

    if len(param_RR['template_filenames']) == 0:
        msg = 'No Redrock templates provided'
        log.critical(msg)
        raise ValueError(msg)

    # make an independant run for each quasar templates to circumvent some problem from redrock:
    # 1] cannot give two templates in redrock as input, only one or a directory
    # 2] problem with prior and template redshift range .. --> give zero result and redrock stop

    filename_priors = param_RR['filename_priors']
    filename_output_rerun_RR = param_RR['filename_output_rerun_RR']
    filename_redrock_rerun_RR = param_RR['filename_redrock_rerun_RR']

    # Check that QN redshifts +- prior/2 is covered by Redrock templates.
    # This should never fail, but if it does exit with an informative
    # message so that we know what to fix
    zmin = 100
    zmax = -100
    for template_filename in param_RR['template_filenames']:
        redshift_template = fitsio.FITS(template_filename)['REDSHIFTS'][:]
        zmin = min(zmin, np.min(redshift_template))
        zmax = max(zmax, np.max(redshift_template))

    badz = (z_qn-z_prior/2 > zmax) | (z_qn+z_prior/2 < zmin)
    if np.any(badz):
        msg = f'Targets {targetid[badz]} have redshifts {z_qn[badz]} outside the Redrock coverage {zmin} <= z <= {zmax}'
        log.critical(msg)
        raise ValueError(msg)

    write_prior_for_RR(targetid, z_qn, z_prior, filename_priors)

    # Info: in the case where targetid is -7008279, we cannot use it at first element of the targetid list
    # otherwise RR required at least one argument for --targetids .. (it is a known problem in python this comes from -)
    # see for example https://webdevdesigner.com/q/can-t-get-argparse-to-read-quoted-string-with-dashes-in-it-47556/
    # To figure out with this, we just need to add a space before the -

    argument_for_rerun_RR  = ['--infiles', spectra_name]
    argument_for_rerun_RR += ['--templates',] + param_RR['template_filenames']
    argument_for_rerun_RR += ['--targetids', ' '+",".join(reversed(targetid.astype(str))),
                                             # see long comment above about need for preceeding space
                             '--priors', filename_priors,
                             '--details', filename_output_rerun_RR,
                             '--outfile', filename_redrock_rerun_RR]
    # NEW RUN RR
    log.info(f'Running redrock with priors on selected targets with {param_RR["template_filenames"]}')
    log.info('rrdesi '+' '.join(argument_for_rerun_RR))

    rrdesi(argument_for_rerun_RR, comm=comm)

    log.info('Done running redrock')

    # Extract information from the new run of RR
    ### redshift_tmp, err_redshift_tmp, chi2_tmp, coeffs_tmp = extract_redshift_info_from_RR(filename_redrock_rerun_RR, targetid[sel])
    redrock = extract_redshift_info_from_RR(filename_redrock_rerun_RR, targetid)

    if param_RR['delete_RR_output'] == 'True':
        log.debug("Remove output from the new run of RR")
        os.remove(filename_priors)
        os.remove(filename_output_rerun_RR)
        os.remove(filename_redrock_rerun_RR)

    return redrock


def selection_targets_with_QN(redrock, fibermap, sel_to_QN, DESI_TARGET, spectra_name, param_QN, param_RR, save_target, comm=None):
    """
    Run QuasarNet to the object with index_to_QN == True from spectra_name.
    Then, Re-Run RedRock for the targetids which are selected by QN as a QSO.

    Args:
        redrock: fitsio hdu 'REDSHIFTS' from redrock file
        fibermap:  fitsio hdu 'FIBERMAP' from redrock file
        sel_to_QN (bool array): Select on which objects QN will be apply (index based on redrock)
        DESI_TARGET (str): name of DESI_TARGET for the wanted version of the target selection
        spectra_name (str): The name of the spectra file
        param_QN (dict): contains info for QN as n_thresh and c_thresh
        param_RR (dict): contains info to re-run RR as the template_filename,
            filename_priors, filename_output_rerun_RR, filename_redrock_rerun_RR
        save_target (str) : restricted (save only IS_QSO_QN_NEW_RR==true targets) / all (save all the sample)
        comm, optional: MPI communicator to pass to redrock; must be size=1

    Returns:
        QSO_sel (pandas dataframe): contains all the information useful to build the QSO cat
    """
    log = get_logger()

    # INFO FOR QUASAR NET
    lines = ['LYA', 'CIV(1548)', 'CIII(1909)', 'MgII(2796)', 'Hbeta', 'Halpha']
    lines_bal = ['CIV(1548)']

    if 'QN_MODEL_FILE' in os.environ.keys():
        # check if the env variable QN_MODEL_FILE is defined, otherwise load boss_dr12/qn_train_coadd_indtrain_0_0_boss10.h5
        model_QN_path = os.environ['QN_MODEL_FILE']
    else:
        log.warning(
            "$QN_MODEL_FILE is not set in the current environment. Default path will be used: $DESI_ROOT/science/lya/qn_models/boss_dr12/qn_train_coadd_indtrain_0_0_boss10.h5")
        if 'DESI_ROOT' in os.environ.keys():
            DESI_ROOT = os.environ['DESI_ROOT']
            model_QN_path = os.path.join(DESI_ROOT,
                                         "science/lya/qn_models/boss_dr12/qn_train_coadd_indtrain_0_0_boss10.h5")
        else:  # if $DESI_ROOT is not set, exit the program.
            log.error(
                "$DESI_ROOT is not set in the current environment. Please set it before running this code.")
            raise KeyError("QN_MODEL_FILE and DESI_ROOT are not set in the current environment.")

    model_QN, wave_to_use = load_model(model_QN_path)
    data, index_with_QN = load_desi_coadd(spectra_name, sel_to_QN, out_grid=wave_to_use)

    # Code to calculate the scaling constant for the dynamic RR prior
    l_min = np.log10(wave_to_use[0])
    l_max = np.log10(wave_to_use[-1])

    # If this changes you must change it here. A future update to QuasarNP
    # might store nboxes as a model parameter but as of 0.2.0 this is not the case.
    nboxes = 13

    dl_bins = (l_max - l_min) / nboxes
    a = 2 * (10**(dl_bins) - 1)
    log.info(f"Using {a = } for redrock prior scaling")

    if len(index_with_QN) == 0:  # if there is no object for QN :(
        sel_QN = np.zeros(sel_to_QN.size, dtype='bool')
        index_with_QN_with_no_pb = sel_QN.copy()
        c_line, z_line, z_QN = np.array([]), np.array([]), np.array([])
    else:
        p = model_QN.predict(data[:, :, None])
        c_line, z_line, z_QN, c_line_bal, z_line_bal = process_preds(p, lines, lines_bal,
                                                                     verbose=False, wave=wave_to_use)  # c_line.size = index_with_QN.size and not index_to_QN !!

        # Selection QSO with QN :
        # sel_index_with_QN.size = z_QN.size = index_with_QN.size | is_qso_QN.size = index_to_QN.size | sel_QN.size = 500
        sel_index_with_QN = np.sum(c_line > param_QN['c_thresh'], axis=0) >= param_QN['n_thresh']
        log.info(f"We select QSO from QN with c_thresh={param_QN['c_thresh']} and n_thresh={param_QN['n_thresh']} --> {sel_index_with_QN.sum()} objects are QSO for QN")

        # Exclude sky fibers even if they were selected by QN as a QSO
        science_targets = (fibermap['OBJTYPE'][index_with_QN] == 'TGT')

        if np.any(sel_index_with_QN & ~science_targets):
            n = np.sum(sel_index_with_QN & ~science_targets)
            log.info(f'Excluding {n} non-target fibers selected by QN (e.g. sky fibers)')
            sel_index_with_QN &= science_targets

        is_qso_QN = np.zeros(sel_to_QN.sum(), dtype=bool)
        is_qso_QN[index_with_QN] = sel_index_with_QN
        sel_QN = sel_to_QN.copy()
        sel_QN[sel_to_QN] = is_qso_QN

        sel_QSO_RR_with_no_z_pb = (redrock['SPECTYPE'] == 'QSO')
        prior = a * (z_QN + 1) # Analytic prior width from QN box width
        sel_QSO_RR_with_no_z_pb[sel_QN] &= np.abs(redrock['Z'][sel_QN] - z_QN[sel_index_with_QN]) <= (prior[sel_index_with_QN] / 2)
        log.info(f"Remove {sel_QSO_RR_with_no_z_pb[sel_QN].sum()} objects with SPECTYPE==QSO and |z_RR - z_QN| < (prior_width / 2) (since even with the prior, RR will give the same result)")

        sel_QN &= ~sel_QSO_RR_with_no_z_pb
        index_with_QN_with_no_pb = sel_QN[sel_to_QN][index_with_QN]

        log.info(f"RUN RR on {sel_QN.sum()} targets")
        if sel_QN.sum() != 0:
            # Re-run Redrock with prior and with only qso templates to compute the redshift of QSO_QN
            redrock_rerun = collect_redshift_with_new_RR_run(spectra_name, redrock['TARGETID'][sel_QN], z_qn=z_QN[index_with_QN_with_no_pb], z_prior=prior[index_with_QN_with_no_pb], param_RR=param_RR, comm=comm)

    if save_target == 'restricted':
        index_to_save = sel_QN.copy()
        index_to_save_QN_result = sel_QN[sel_to_QN]
    elif save_target == 'all':
        index_to_save = sel_to_QN.copy()
        # save every object with nan value if it is necessary --> there are sel_to_QN.sum() objects to save
        # index_with_QN is size of sel_to_QN.sum()
        index_to_save_QN_result = np.ones(sel_to_QN.sum(), dtype=bool)
    else:
        # never happen since a test is performed before running this function in desi_qso_qn_afterburner
        log.error('**** CHOOSE CORRECT SAVE_TARGET FLAG (restricted / all) ****')

    QSO_sel = Table()
    QSO_sel['TARGETID'] = redrock['TARGETID'][index_to_save]
    QSO_sel['RA'] = fibermap['TARGET_RA'][index_to_save]
    QSO_sel['DEC'] = fibermap['TARGET_DEC'][index_to_save]
    QSO_sel[DESI_TARGET] = fibermap[DESI_TARGET][index_to_save]
    QSO_sel['IS_QSO_QN_NEW_RR'] = sel_QN[index_to_save]
    QSO_sel['SPECTYPE_RR'] = redrock['SPECTYPE'][index_to_save]
    QSO_sel['Z_RR'] = redrock['Z'][index_to_save]

    num_to_save = sel_to_QN.sum()
    assert len(QSO_sel) == num_to_save

    tmp_arr = np.nan * np.ones(sel_to_QN.sum())
    tmp_arr[index_with_QN] = z_QN
    QSO_sel['Z_QN'] = tmp_arr[index_to_save_QN_result]

    # tmp_arr = np.nan * np.ones((6, sel_to_QN.sum()))
    # tmp_arr[:, index_with_QN] = c_line
    # QSO_sel['C_LINES'] = tmp_arr.T[index_to_save_QN_result]
    # tmp_arr = np.nan * np.ones((6, sel_to_QN.sum()))
    # tmp_arr[:, index_with_QN] = z_line
    # QSO_sel['Z_LINES'] = tmp_arr.T[index_to_save_QN_result]

    #- TODO: indexing probably wrong for some options for subsets to run vs. save
    for i, linename in enumerate(lines):
        linename = linename.split('(')[0]  #- strip (wavelength) part of line names
        QSO_sel[f'C_{linename}'] = np.nan
        QSO_sel[f'C_{linename}'][index_with_QN] = c_line[i]
        QSO_sel[f'Z_{linename}'] = np.nan
        QSO_sel[f'Z_{linename}'][index_with_QN] = z_line[i]

    tmp_arr = np.nan * np.ones(sel_to_QN.sum())
    if index_with_QN_with_no_pb.sum() != 0:  # in case where sel_QN.sum() == 0 and redshift is so not defined
        tmp_arr[index_with_QN[index_with_QN_with_no_pb]] = redrock_rerun['Z']
    QSO_sel['Z_NEW'] = tmp_arr[index_to_save_QN_result]

    tmp_arr = np.nan * np.ones(sel_to_QN.sum())
    if index_with_QN_with_no_pb.sum() != 0:
        tmp_arr[index_with_QN[index_with_QN_with_no_pb]] = redrock_rerun['ZERR']
    QSO_sel['ZERR_NEW'] = tmp_arr[index_to_save_QN_result]

    tmp_arr = np.zeros(sel_to_QN.sum(), dtype=np.int)
    if index_with_QN_with_no_pb.sum() != 0:
        tmp_arr[index_with_QN[index_with_QN_with_no_pb]] = redrock_rerun['ZWARN']
    QSO_sel['ZWARN_NEW'] = tmp_arr[index_to_save_QN_result]

    tmp_arr = np.zeros(sel_to_QN.sum(), dtype='<U20')
    if index_with_QN_with_no_pb.sum() != 0:
        tmp_arr[index_with_QN[index_with_QN_with_no_pb]] = redrock_rerun['SUBTYPE']
    QSO_sel['SUBTYPE_NEW'] = tmp_arr[index_to_save_QN_result]

    tmp_arr = np.nan * np.ones((sel_to_QN.sum(), 10))
    if index_with_QN_with_no_pb.sum() != 0:
        ncoeff = redrock_rerun['COEFF'].shape[1]
        tmp_arr[index_with_QN[index_with_QN_with_no_pb], 0:ncoeff] = redrock_rerun['COEFF']
        tmp_arr[index_with_QN[index_with_QN_with_no_pb], ncoeff:] = 0.0
    QSO_sel['COEFFS'] = tmp_arr[index_to_save_QN_result].tolist()

    return QSO_sel


def save_to_fits(data, filename, DESI_TARGET, clobber=True, templatefiles=None):
    """
    Save info from data in a fits file

    Args:
        data (astropy Table): Table containing all the necessary QSO info
        filename (str):  name of the fits file
        DESI_TARGET (str): name of DESI_TARGET for the wanted version of the target selection

    Options:
        clobber (bool): overwrite the fits file defined by filename ?
        templatefiles (list): list of Redrock template filenames to record in header

    Returns:
        None
    """
    log = get_logger()

    #- lightweight copy so that we can update .meta header without altering original
    data = Table(data, copy=False)
    data.meta['EXTNAME'] = 'QN_RR'

    # Header to save provenance
    desiutil.depend.add_dependencies(data.meta)
    for key in ('QN_MODEL_FILE', 'RR_TEMPLATE_DIR'):
        desiutil.depend.setdep(data.meta, key, os.getenv(key, 'None'))

    if templatefiles is not None:
        for i, templatefilename in enumerate(templatefiles):
            key = f'RR_TEMPLATE_{i}'
            if 'RR_TEMPLATE_DIR' in os.environ and templatefilename.startswith(os.environ['RR_TEMPLATE_DIR']):
                templatefilename = os.path.basename(templatefilename)

            desiutil.depend.setdep(data.meta, key, templatefilename)
    else:
        log.warning('Not recording template filenames in output header')

    # Save file in temporary file to track when timeout error appears during the writing
    tmpfile = get_tempfilename(filename)
    data.write(tmpfile)
    os.rename(tmpfile, filename)

    log.info(f'wrote output in: {filename}')

    return


def main(args=None, comm=None):
    from desispec.io.util import replace_prefix

    log = get_logger()

    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    #- We need an MPI communicator to pass to redrock, but the rest of
    #- the qsoqn code isn't instrumented for MPI parallelism, so we require
    #- that the communicator is size=1
    if comm is not None and comm.size>1:
        raise ValueError(f'{comm.size=} not allowed, must be size=1')

    start = time.time()

    # Selection param for QuasarNet
    param_QN = {'c_thresh': args.c_thresh, 'n_thresh': args.n_thresh}

    # param for the new run of RR
    param_RR = {'template_filenames': args.templates}

    #- write all temporary files to output directory, not input directory
    outdir = os.path.dirname(os.path.abspath(args.output))
    outbase = os.path.join(outdir, os.path.basename(args.coadd))

    if args.filename_priors is None:
        param_RR['filename_priors'] = replace_prefix(outbase, 'coadd', 'priors_qn')
    else:
        param_RR['filename_priors'] = args.filename_priors
    if args.filename_output_rerun_RR is None:
        tmp = replace_prefix(outbase, 'coadd', 'rrdetails_qn')
        tmp = os.path.splitext(tmp)[0] + '.h5'  #- replace extension with .h5
        param_RR['filename_output_rerun_RR'] = tmp
    else:
        param_RR['filename_output_rerun_RR'] = args.filename_output_rerun_RR
    if (args.filename_redrock_rerun_RR is None):
        param_RR['filename_redrock_rerun_RR'] = replace_prefix(outbase, 'coadd', 'redrock_qn')
    else:
        param_RR['filename_redrock_rerun_RR'] = args.filename_redrock_rerun_RR

    param_RR['delete_RR_output'] = args.delete_RR_output

    if os.path.isfile(args.coadd) and os.path.isfile(args.redrock):
        # Testing if there are three cameras in the coadd file. If not create a warning file.
        if np.isin(['B_FLUX', 'R_FLUX', 'Z_FLUX'], [hdu.get_extname() for hdu in fitsio.FITS(args.coadd)]).sum() != 3:
            misscamera = os.path.splitext(args.output)[0] + '.misscamera.txt'
            with open(misscamera, "w") as miss:
                miss.write(f"At least one camera is missing from the coadd file: {args.coadd}.\n")
                miss.write("This is expected for the exposure directory.\n")
                miss.write('This is NOT expected for cumulative / healpix directory!\n')
            log.warning(f"At least one camera is missing from the coadd file; warning file {misscamera} has been written.")
        else:
            # open best fit file generated by redrock
            with fitsio.FITS(args.redrock) as redrock_file:
                redrock = redrock_file['REDSHIFTS'].read()
                fibermap = redrock_file['FIBERMAP'].read()

            # from everest REDSHIFTS hdu and FIBERMAP hdu have the same order (the indices match)
            if np.sum(redrock['TARGETID'] == fibermap['TARGETID']) == redrock['TARGETID'].size:
                log.info("SANITY CHECK: The indices of REDSHIFTS HDU and FIBERMAP HDU match.")
            else:
                log.error("**** The indices of REDROCK HDU AND FIBERMAP DHU do not match. This is not expected ! ****")
                return 1

            # Find which selection is used (SV1/ SV2 / SV3 / MAIN / ...)
            DESI_TARGET = main_cmx_or_sv(fibermap)[0][0]

            if args.target_selection.lower() in ('qso', 'qso_targets'):
                if DESI_TARGET == 'DESI_TARGET':
                    qso_mask_bit = desi_mask.mask('QSO')
                elif DESI_TARGET == 'SV3_DESI_TARGET':
                    qso_mask_bit = sv3_mask.mask('QSO')
                elif DESI_TARGET == 'SV2_DESI_TARGET':
                    qso_mask_bit = sv2_mask.mask('QSO')
                elif DESI_TARGET == 'SV1_DESI_TARGET':
                    qso_mask_bit = sv1_mask.mask('QSO')
                elif DESI_TARGET == 'CMX_TARGET':
                    qso_mask_bit = cmx_mask.mask('MINI_SV_QSO|SV0_QSO')
                else:
                    log.error("**** DESI_TARGET IS NOT CMX / SV1 / SV2 / SV3 / MAIN ****")
                    sys.exit(1)
                sel_to_QN = fibermap[DESI_TARGET] & qso_mask_bit != 0

            elif args.target_selection.lower() in ('all', 'all_targets'):
                sel_to_QN = np.ones(redrock['TARGETID'].size, dtype=bool)
            else:
                log.error("**** CHOOSE CORRECT TARGET_SELECTION FLAG (qso_targets / all_targets) ****")
                return 1

            # Check args.save_target to avoid a crash after the QN + new RR Run
            if not (args.save_target in ['restricted', 'all']):
                log.error('**** CHOOSE CORRECT SAVE_TARGET FLAG (restricted / all) ****')
                return 1

            log.info(f"Nbr objetcs for QN: {sel_to_QN.sum()}")
            QSO_from_QN = selection_targets_with_QN(redrock, fibermap, sel_to_QN, DESI_TARGET,
                                                    args.coadd, param_QN, param_RR, args.save_target, comm=comm)

            if len(QSO_from_QN) > 0:
                log.info(f"Number of targets saved : {len(QSO_from_QN)} -- "
                         f"Selected with QN + new RR: {QSO_from_QN['IS_QSO_QN_NEW_RR'].sum()}")
                save_to_fits(QSO_from_QN, args.output, DESI_TARGET, templatefiles=args.templates)
            else:
                file = open(os.path.splitext(args.output)[0] + '.notargets.txt', "w")
                file.write("No targets were selected by QN afterburner to be a QSO.")
                file.write(f"\nThis is done with the following parametrization : target_selection = {args.target_selection}\n")
                file.write("\nIN SOME CASE (BRIGHT TILE + target_selection=QSO), this file is expected !")
                file.close()
                log.warning(f"No objects selected to save; blank file {os.path.splitext(args.output)[0]+'.notargets.txt'} is written")

    else:  # file for the consider Tile / Night / petal does not exist
        log.error(f"**** Missing {args.coadd} or {args.redrock} ****")
        return 1

    log.info(f"EXECUTION TIME: {time.time() - start:3.2f} s.")
    return 0
