#!/usr/bin/env python
# coding: utf-8

# author:  edmond chaussidon (cea saclay)
# contact: edmond.chaussidon@cea.fr

import sys
import os
import glob

import tqdm

import fitsio
import numpy as np
import pandas as pd

from desitarget.targetmask import desi_mask
from desitarget.targets import main_cmx_or_sv

from desiutil.log import get_logger
from desiutil.dust import SFDMap, ext_odonnell
from desitarget.io import desitarget_resolve_dec
# to correct the photometry
from speclite import filters

decamwise = filters.load_filters('decam2014-g', 'decam2014-r','decam2014-z', 'wise2010-W1', 'wise2010-W2')
bassmzlswise = filters.load_filters('BASS-g', 'BASS-r', 'MzLS-z','wise2010-W1', 'wise2010-W2')

log = get_logger()


def extract_info_from_fibermap(fibermap, exp_fibermap, dataframe, sel):
    """
    Extract information from the fibermap. We save: TARGETID, RA, DEC, DESI_TARGET, MASKBIT,
    r, g, z, W1, W2, TILEID, LAST_NIGHT, PETAL, FIBER, LOCATION, ZWARN, COADD_FIBERSTATUS
    Args:
        fibermap:  fitsio hdu 'FIBERMAP' from redrock file
        exp_fibermap: fitsio hdu 'EXP_FIBERMAP' from redrock file
        dataframe: dataframe to fill with useful information extracted from the fibermap
        sel: mask to select which objects from the fibermap have to be saved. Order given by the redrock file
    Returns:
        QSO_cat (pandas dataframe): Dataframe containing all the information
    """

    def compute_grzW1W2(fibermap, sel):
        # Compute g, r, z, W1, w2 magnitudes corrected by the Milky Way transmission
        # Args:
        #     fibermap: fitsio hdu 'COADD_FIBERMAP' from redrock file
        #     sel: mask to select which objects from the fibermap have to be saved.
        # Returns:
        #     g, r, z, W1, W2: magnitudes corrected

        north = (fibermap['TARGET_RA'][sel] > 80) & (fibermap['TARGET_RA'][sel] < 300 ) & (fibermap['TARGET_DEC'][sel] > desitarget_resolve_dec())

        gflux  = fibermap['FLUX_G'][sel]
        rflux  = fibermap['FLUX_R'][sel]
        zflux  = fibermap['FLUX_Z'][sel]
        W1flux  = fibermap['FLUX_W1'][sel]
        W2flux  = fibermap['FLUX_W2'][sel]

        RV = 3.1
        EBV =  fibermap['EBV'][sel]

        mw_transmission = np.array([10**(-0.4 * EBV[i] * RV * ext_odonnell(bassmzlswise.effective_wavelengths.value, Rv=RV)) if north[i]
                                    else 10**(-0.4 * EBV[i] * RV * ext_odonnell(decamwise.effective_wavelengths.value, Rv=RV)) for i in range(EBV.size)])

        with np.errstate(divide='ignore', invalid='ignore'):
            g = np.where(gflux>0, 22.5-2.5*np.log10(gflux/mw_transmission[:, 0]), 0.)
            r = np.where(rflux>0, 22.5-2.5*np.log10(rflux/mw_transmission[:, 1]), 0.)
            z = np.where(zflux>0, 22.5-2.5*np.log10(zflux/mw_transmission[:, 2]), 0.)
            W1 = np.where(W1flux>0, 22.5-2.5*np.log10(W1flux/mw_transmission[:, 3]), 0.)
            W2 = np.where(W2flux>0, 22.5-2.5*np.log10(W2flux/mw_transmission[:, 4]), 0.)

        return g, r, z, W1, W2

    dataframe['TARGETID'] = fibermap['TARGETID'][sel]
    dataframe['RA'] = fibermap['TARGET_RA'][sel]
    dataframe['DEC'] = fibermap['TARGET_DEC'][sel]

    DESI_TARGET = main_cmx_or_sv(fibermap)[0][0]
    dataframe[DESI_TARGET] = fibermap[DESI_TARGET][sel]

    dataframe['COADD_FIBERSTATUS'] = fibermap['COADD_FIBERSTATUS'][sel]
    dataframe['MASKBITS'] = fibermap["MASKBITS"][sel]

    g, r, z, W1, W2 = compute_grzW1W2(fibermap, sel)
    dataframe['G_MAG'] = g
    dataframe['R_MAG'] = r
    dataframe['Z_MAG'] = z
    dataframe['W1_MAG'] = W1
    dataframe['W2_MAG'] = W2

    # with pixels there is no tile id in the fibermap for the moment ...
    if 'TILEID' in fibermap.dtype.names:
        dataframe['TILEID'] = fibermap['TILEID'][sel]

    last_night = np.zeros(sel.sum(), dtype=int)
    for i, targetid in enumerate(fibermap['TARGETID'][sel]):
        last_night[i] = np.max(exp_fibermap['NIGHT'][exp_fibermap['TARGETID'] == targetid])
    dataframe['LAST_NIGHT'] = last_night

    if 'PETAL_LOC' in fibermap.dtype.names:
        dataframe['PETAL_LOC'] = fibermap['PETAL_LOC'][sel]
    if 'FIBER' in fibermap.dtype.names:
        dataframe['FIBER'] = fibermap['FIBER'][sel]
    if 'LOCATION' in fibermap.dtype.names:
        dataframe['LOCATION'] = fibermap['LOCATION'][sel]

    return dataframe


def select_qso_with_RR(zbest, fibermap, exp_fibermap, qn_cat):
    """
    Apply the selection based on Redrock when there is no discrepancy with QuasarNet
    or if the objects are not selected by QuasarNet.
    Args:
        zbest: fitsio hdu 'REDSHIFTS' from redrock file
        fibermap:  fitsio hdu 'COADD_FIBERMAP' from redrock file
        exp_fibermap: fitsio hdu 'EXP_FIBERMAP' from redrock file
        qn_cat: fitsio hdu 'QN+RR' from qn file
    Returns:
        QSO_cat (pandas dataframe): Dataframe containing all the information
    """

    # find which objects from zbest are in the qn_cat
    is_in_qn_cat = np.isin(zbest['TARGETID'], qn_cat['TARGETID'])

    # sel objects with SPECTYPE == QSO and |z_RR - z_QN| <= 0.05 or not QSO for QN.
    sel = (zbest['SPECTYPE'] == 'QSO')
    sel[is_in_qn_cat] &= ~qn_cat['IS_QSO_QN_NEW_RR']

    # mask to select z_QN in qn_cat
    sel_in_qn_cat = (qn_cat['SPECTYPE'] == 'QSO') & ~qn_cat['IS_QSO_QN_NEW_RR']

    QSO_cat = pd.DataFrame()
    #to avoid error with np.where in extract_info_from_fibermap
    if sel.sum() != 0:
        QSO_cat = extract_info_from_fibermap(fibermap, exp_fibermap, QSO_cat, sel)

        QSO_cat.insert(3, 'Z', zbest['Z'][sel])
        QSO_cat.insert(4, 'ZERR', zbest['ZERR'][sel])
        QSO_cat.insert(5, 'SELECTION_METHOD', "sel:RR - z:RR")
        QSO_cat.insert(6, 'Z_RR_INI', np.NaN)
        QSO_cat.insert(7, 'ZWARN', zbest['ZWARN'][sel])

        # add qn info
        i = 8
        sel_in_qn_tmp = np.isin(zbest['TARGETID'][sel], qn_cat['TARGETID'][sel_in_qn_cat])
        for name in ['Z_QN', 'C_LYA', 'C_CIV', 'C_CIII', 'C_MgII', 'C_Hbeta', 'C_Halpha',
                             'Z_LYA', 'Z_CIV', 'Z_CIII', 'Z_MgII', 'Z_Hbeta', 'Z_Halpha']:
            qn_tmp = np.NaN * np.zeros(sel.sum())
            qn_tmp[sel_in_qn_tmp] = qn_cat[name][sel_in_qn_cat]
            QSO_cat.insert(i, name, qn_tmp)
            i += 1

    return QSO_cat


def select_qso_with_RR_new_run_RR(zbest, fibermap, exp_fibermap, qn_cat):
    """
    Apply the selection based on Redrock when there is a discrepancy with QuasarNet
    Choosing the redshift found with a new run of Redrock with prior
    Args:
        zbest: fitsio hdu 'REDSHIFTS' from redrock file
        fibermap:  fitsio hdu 'COADD_FIBERMAP' from redrock file
        exp_fibermap: fitsio hdu 'EXP_FIBERMAP' from redrock file
        qn_cat: fitsio hdu 'QN+RR' from qn file
    Returns:
        QSO_cat (pandas dataframe): Dataframe containing all the information
    """
    # find which objects from zbest are in the qn_cat
    is_in_qn_cat = np.isin(zbest['TARGETID'], qn_cat['TARGETID'])

    # sel objects with SPECTYPE == QSO and QSO for QN with|z_RR - z_QN| > 0.05
    sel = (zbest['SPECTYPE'] == 'QSO') & is_in_qn_cat
    sel[is_in_qn_cat] &= qn_cat['IS_QSO_QN_NEW_RR']

    sel_in_qn_cat = (qn_cat['SPECTYPE'] == 'QSO') & qn_cat['IS_QSO_QN_NEW_RR']

    QSO_cat = pd.DataFrame()
    #to avoid error with np.where in extract_info_from_fibermap
    if sel.sum() != 0:
        QSO_cat = extract_info_from_fibermap(fibermap, exp_fibermap, QSO_cat, sel)
        QSO_cat.insert(3, 'Z', qn_cat['Z_NEW'][sel_in_qn_cat])
        QSO_cat.insert(4, 'ZERR', qn_cat['ZERR_NEW'][sel_in_qn_cat])
        QSO_cat.insert(5, 'SELECTION_METHOD', "sel:RR - z:QN/RR")
        QSO_cat.insert(6, 'Z_RR_INI', zbest['Z'][sel])
        QSO_cat.insert(7, 'ZWARN', zbest['ZWARN'][sel])

        i = 8
        for name in ['Z_QN', 'C_LYA', 'C_CIV', 'C_CIII', 'C_MgII', 'C_Hbeta', 'C_Halpha',
                             'Z_LYA', 'Z_CIV', 'Z_CIII', 'Z_MgII', 'Z_Hbeta', 'Z_Halpha']:
            QSO_cat.insert(i, name, qn_cat[name][sel_in_qn_cat])
            i += 1

    return QSO_cat


def select_qso_with_MgII(zbest, fibermap, exp_fibermap, mgii_cat, qn_cat):
    """
    Apply the selection based on the MgII fitter, keeping the redshift estimated by Redrock
    Args:
        zbest: fitsio hdu 'REDSHIFTS' from redrock file
        fibermap:  fitsio hdu 'COADD_FIBERMAP' from redrock file
        exp_fibermap: fitsio hdu 'EXP_FIBERMAP' from redrock file
        mgii_cat: fitsio hdu 'MGII' from mgii file
    Returns:
        QSO_cat (pandas dataframe): Dataframe containing all the information
    """
    # find which objects from zbest are in the mgii_cat
    is_in_mgii_cat = np.isin(zbest['TARGETID'], mgii_cat['TARGETID'])

    # sel objects with SPECTYPE != QSO and selected by the MgII
    sel = (zbest['SPECTYPE'] != 'QSO') & is_in_mgii_cat
    sel[is_in_mgii_cat] &= mgii_cat['IS_QSO_MGII']

    sel_in_mgii_cat = (mgii_cat['SPECTYPE'] != 'QSO') & mgii_cat['IS_QSO_MGII']

    # find which objects from qn_cat are in the mgii_cat
    is_in_mgii_cat = np.isin(qn_cat['TARGETID'], mgii_cat['TARGETID'])
    sel_in_qn_cat = (qn_cat['SPECTYPE'] != 'QSO')
    sel_in_qn_cat[is_in_mgii_cat] &= mgii_cat['IS_QSO_MGII']

    QSO_cat = pd.DataFrame()
    #to avoid error with np.where in extract_info_from_fibermap
    if sel.sum() != 0:
        QSO_cat = extract_info_from_fibermap(fibermap, exp_fibermap, QSO_cat, sel)
        QSO_cat.insert(3, 'Z', mgii_cat['Z_RR'][sel_in_mgii_cat])
        QSO_cat.insert(4, 'ZERR', mgii_cat['ZERR'][sel_in_mgii_cat])
        QSO_cat.insert(5, 'SELECTION_METHOD', "sel:MgII - z:RR")
        QSO_cat.insert(6, 'Z_RR_INI', np.NaN)
        QSO_cat.insert(7, 'ZWARN', zbest['ZWARN'][sel])

        # add qn info
        i = 8
        sel_in_qn_tmp = np.isin(zbest['TARGETID'][sel], qn_cat['TARGETID'][sel_in_qn_cat])
        for name in ['Z_QN', 'C_LYA', 'C_CIV', 'C_CIII', 'C_MgII', 'C_Hbeta', 'C_Halpha',
                             'Z_LYA', 'Z_CIV', 'Z_CIII', 'Z_MgII', 'Z_Hbeta', 'Z_Halpha']:
            qn_tmp = np.NaN * np.zeros(sel.sum())
            qn_tmp[sel_in_qn_tmp] = qn_cat[name][sel_in_qn_cat]
            QSO_cat.insert(i, name, qn_tmp)
            i += 1

        for name in ['DELTA_CHI2', 'A', 'SIGMA', 'B', 'VAR_A', 'VAR_SIGMA', 'VAR_B']:
            QSO_cat.insert(i, name+'_MGII', mgii_cat[name][sel_in_mgii_cat])
            i += 1

    return QSO_cat


def select_qso_with_QN_new_run_RR(zbest, fibermap, exp_fibermap, qn_cat, targetid_already_selected_with_mgII):
    """
    Apply the selection based on QN with a new run of Redrock with prior
    Args:
        zbest: fitsio hdu 'REDSHIFTS' from redrock file
        fibermap:  fitsio hdu 'COADD_FIBERMAP' from redrock file
        exp_fibermap: fitsio hdu 'EXP_FIBERMAP' from redrock file
        qn_cat: fitsio hdu 'QN+RR' from qn file
        targetid_already_selected_with_mgII (int array): targetid already selected by the selection based on the MgII
    Returns:
        QSO_cat (pandas dataframe): Dataframe containing all the information
    """
    # find which objects from zbest are in the qn_cat
    is_in_qn_cat = np.isin(zbest['TARGETID'], qn_cat['TARGETID'])

    # find which targetid is already selected by the MgII fitter
    selected_by_mgii = np.isin(zbest['TARGETID'], targetid_already_selected_with_mgII)
    selected_by_mgii_in_qn_cat = np.isin(qn_cat['TARGETID'], targetid_already_selected_with_mgII)

    # sel objects with SPECTYPE != QSO and unselected by the MgII and selected by QN
    sel = (zbest['SPECTYPE'] != 'QSO') & is_in_qn_cat & ~selected_by_mgii
    sel[is_in_qn_cat] &= qn_cat['IS_QSO_QN_NEW_RR']

    sel_in_qn_cat = (qn_cat['SPECTYPE'] != 'QSO') & qn_cat['IS_QSO_QN_NEW_RR'] & ~selected_by_mgii_in_qn_cat

    QSO_cat = pd.DataFrame()
    #to avoid error with np.where in extract_info_from_fibermap
    if sel.sum() != 0:
        QSO_cat = extract_info_from_fibermap(fibermap, exp_fibermap, QSO_cat, sel)
        QSO_cat.insert(3, 'Z', qn_cat['Z_NEW'][sel_in_qn_cat])
        QSO_cat.insert(4, 'ZERR', qn_cat['ZERR_NEW'][sel_in_qn_cat])
        QSO_cat.insert(5, 'SELECTION_METHOD', "sel:QN - z:QN/RR")
        QSO_cat.insert(6, 'Z_RR_INI', zbest['Z'][sel])
        QSO_cat.insert(7, 'ZWARN', zbest['ZWARN'][sel])

        i = 8
        for name in ['Z_QN', 'C_LYA', 'C_CIV', 'C_CIII', 'C_MgII', 'C_Hbeta', 'C_Halpha',
                             'Z_LYA', 'Z_CIV', 'Z_CIII', 'Z_MgII', 'Z_Hbeta', 'Z_Halpha']:
            QSO_cat.insert(i, name, qn_cat[name][sel_in_qn_cat])
            i += 1

    return QSO_cat


def save_dataframe_to_fits(dataframe, filename, clobber=True):
    """
    Save info from pandas dataframe in a fits file.
    Args:
        dataframe (pandas dataframe): dataframe containg the all the necessary QSO info
        filename (str):  name of the fits file
        clobber (bool):  overwrite the fits file defined by filename ?
    Returns:
        None
    """
    # No complex structure, to_records() is sufficient.
    fits = fitsio.FITS(filename, 'rw', clobber=clobber)
    if clobber:
        log.warning(f'OVERWRITE the file : {filename}')
    else:
        log.warning(f'EXPAND the file : {filename}')
    fits.write(dataframe.to_records(index=False), extname="QSO_CAT")
    fits.close()


def qso_catalog_maker(redrock, mgii, qn, use_old_extname_for_redrock=False, use_old_extname_for_fitsio=False):
    """
    Compile the different QSO identifications to build the QSO catalog from a RR, mgII, Qn file.
    Args:
        redrock (str): redrock file with redshifts (formerly zbest)
        mgii (str): mgii file containing the mgii afterburner output
        qn (str): qn file containing the qn afterburner (with new run of RR) output
        use_old_extname_for_redrock (bool); default=False, If true use ZBEST instead REDSHIFTS for extname in redrock file?
        use_old_extname_for_fitsio (bool): default=False, For FUJI extname QN+RR is remplaced by QN_RR to avoid error with newer version of fitsio (>= 1.1.3).
                                           To use desi_qso_qn_afterburner for everest and older files please activate this flag and use ONLY fitsio = 1.1.2.
                                           For daily production, this modification was done in: 18/01/2022.
    Returns:
        QSO_cat (pandas dataframe): Dataframe containing all the information
    """
    # load best fit info generated by redrock
    with fitsio.FITS(redrock) as redrock_file:
        if use_old_extname_for_redrock:
            zbest = redrock_file['ZBEST'].read()
        else:
            zbest = redrock_file['REDSHIFTS'].read()
        fibermap = redrock_file['FIBERMAP'].read()
        # from Everest NIGHT was put in the EXP_FIBERMAP hdu ...
        exp_fibermap = redrock_file['EXP_FIBERMAP'].read()

    # from everest REDROCK hdu and FIBERMAP hdu have the same order (the indices match)
    if np.sum(zbest['TARGETID'] == fibermap['TARGETID']) == zbest['TARGETID'].size:
        log.info("SANITY CHECK: The indices of REDSHIFTS HDU and FIBERMAP HDU match.")
    else:
        log.error("**** The indices of REDSHIFTS HDU AND FIBERMAP DHU do not match. This is not expected ! ****")
        sys.exit()

    # load info from mgii
    mgii_cat = fitsio.read(mgii, 'MGII')
    # load info from qn with new run of RR
    print(fitsio.FITS(qn))
    if use_old_extname_for_fitsio:
        # before FUJI
        qn_cat = fitsio.read(qn, 'QN+RR')
    else:
        # from FUJI
        qn_cat = fitsio.read(qn, 'QN_RR')
        print(qn_cat)

    # Apply the selection following the flowchart
    QSO_from_RR = select_qso_with_RR(zbest, fibermap, exp_fibermap, qn_cat)
    QSO_from_RR_with_QN = select_qso_with_RR_new_run_RR(zbest, fibermap, exp_fibermap, qn_cat)
    QSO_from_MgII = select_qso_with_MgII(zbest, fibermap, exp_fibermap, mgii_cat, qn_cat)
    if QSO_from_MgII.size != 0:
        targetid_already_selected_with_mgII = QSO_from_MgII['TARGETID']
    else:
        targetid_already_selected_with_mgII = np.array([])
    QSO_from_QN = select_qso_with_QN_new_run_RR(zbest, fibermap, exp_fibermap, qn_cat, targetid_already_selected_with_mgII)

    # Concatene the different selection
    QSO_cat = pd.concat([QSO_from_RR, QSO_from_RR_with_QN, QSO_from_MgII, QSO_from_QN], ignore_index=True)
    log.info(f"Final selection gives: {QSO_cat.shape[0]} QSO with {(QSO_cat[main_cmx_or_sv(fibermap)[0][0]] & desi_mask.QSO != 0).sum()} DESI QSO targets")
    return QSO_cat


def afterburner_is_missing():
    dir = 'pernight'
    suff_dir = ''

    #dir = 'perexp'
    #suff_dir = 'exp'

    fuji = os.path.join('/global/cfs/cdirs/desi/spectro/redux/fuji/tiles/', dir)

    def check_afterburner(fuji, tile, night):
        pb_qn, pb_mgII = [], []
        for petal in range(10):
            if os.path.isfile(os.path.join(fuji, tile, night, f"redrock-{petal}-{tile}-{suff_dir}{night}.fits")):
                if not os.path.isfile(os.path.join(fuji, tile, night, f"qso_qn-{petal}-{tile}-{suff_dir}{night}.fits")):
                    pb_qn += [[int(tile), int(night), int(petal)]]
                if not os.path.isfile(os.path.join(fuji, tile, night, f"qso_mgii-{petal}-{tile}-{suff_dir}{night}.fits")):
                    pb_mgII += [[int(tile), int(night), int(petal)]]
        return pb_qn, pb_mgII

    tiles = np.sort([os.path.basename(path) for path in glob.glob(os.path.join(fuji, '*'))])

    pb_qn, pb_mgII = [], []

    for tile in tqdm.tqdm(tiles):
        nights = np.sort([os.path.basename(path) for path in glob.glob(os.path.join(fuji, tile, '*'))])
        for night in nights:
            qn_tmp, mgII_tmp = check_afterburner(fuji, tile, night)
            pb_qn += qn_tmp
            pb_mgII += mgII_tmp

    print(f'Under the directory {dir} it lacks:')
    print(f'    * {len(pb_qn)} QN files')
    print(f'    * {len(pb_mgII)} MgII files')

    np.savetxt(f'pb_qn_{dir}.txt', pb_qn)
    np.savetxt(f'pb_mgII_{dir}.txt', pb_mgII)

## Je dois faire deux fonctions --> l'une pour construire et concaténer tous les catalogus a partir du cumulatif + l'autre a partir du healpix direcotry.
# faire tout ce que je dois faire encore.
