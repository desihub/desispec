"""
desispec.validredshifts
=======================

"""
# Example usage:
# redrock_path = '/global/cfs/cdirs/desi/spectro/redux/guadalupe/tiles/cumulative/1392/20210517/redrock-5-1392-thru20210517.fits'
# cat = validate(redrock_path, return_target_columns=True)

import os, warnings
import numpy as np
from astropy.table import Table, hstack, join
import fitsio


def get_good_fiberstatus(cat, isqso=False):
    '''
    Validate the fiber status for a redrock catalog.

    Args:
        cat: redrock catalog (e.g., in astropy table format)

    Options:
        isqso: bool (default False), if True, use the specific QSO criteria

    Returns:
        good_fiberstatus: boolean array
    '''

    if not isqso:
        good_fiberstatus = cat['COADD_FIBERSTATUS']==0
    else:
        # good_fiberstatus = (cat['COADD_FIBERSTATUS']==0) | (cat['COADD_FIBERSTATUS']==8388608) | (cat['COADD_FIBERSTATUS']==16777216)
        from desispec.maskbits import fibermask
        good_fiberstatus = (cat['COADD_FIBERSTATUS']==0) | (cat['COADD_FIBERSTATUS']==fibermask['BADAMPR']) | (cat['COADD_FIBERSTATUS']==fibermask['BADAMPZ'])
    good_fiberstatus &= cat['ZWARN'] & 2**9 == 0  # NO DATA flag
    return good_fiberstatus


def validate(redrock_path, fiberstatus_cut=True, return_target_columns=False, extra_columns=None, emline_path=None, ignore_emline=False):
    '''
    Validate the redshift quality with tracer-dependent criteria for redrock+afterburner results.

    Args:
        redrock_path: str, path of redrock FITS file

    Options:
        fiberstatus_cut: bool (default True), if True, impose requirements on COADD_FIBERSTATUS and ZWARN
        return_target_columns: bool (default False), if True, include columns that indicate if the object belongs to each class of DESI targets
        extra_columns: list of str (default None), additional columns to include in the output
        emline_path: str (default None), specify the location of the emline file; by default the emline file is in the same directory as the redrock file
        ignore_emline: bool (default False), if True, ignore the emline file and do not validate the ELG redshift

    Returns:
        cat: astropy table with basic columns such as TARGETID and boolean columns (e.g., GOOD_BGS)
             that indicate if each object meets the redshift quality criteria of specific tracers
    '''

    output_columns = ['GOOD_BGS', 'GOOD_LRG', 'GOOD_ELG', 'GOOD_QSO']
    if return_target_columns:
        output_columns += ['LRG', 'ELG', 'QSO', 'ELG_LOP', 'ELG_HIP', 'ELG_VLO', 'BGS_ANY', 'BGS_FAINT', 'BGS_BRIGHT']

    if ignore_emline:
        output_columns.remove('GOOD_ELG')

    if extra_columns is None:
        extra_columns = ['TARGETID', 'Z', 'ZWARN', 'COADD_FIBERSTATUS']
    output_columns = output_columns + list(np.array(extra_columns)[~np.in1d(extra_columns, output_columns)])

    ############################ Load data ############################

    columns_redshifts = ['TARGETID', 'CHI2', 'Z', 'ZERR', 'ZWARN', 'SPECTYPE', 'DELTACHI2']
    columns_fibermap = ['TARGETID', 'COADD_FIBERSTATUS', 'TARGET_RA', 'TARGET_DEC', 'DESI_TARGET', 'BGS_TARGET']
    columns_emline = ['TARGETID', 'OII_FLUX', 'OII_FLUX_IVAR']
    columns_qso_mgii = ['TARGETID', 'IS_QSO_MGII']
    columns_qso_qn = ['TARGETID', 'Z_NEW', 'ZERR_NEW', 'IS_QSO_QN_NEW_RR', 'C_LYA', 'C_CIV', 'C_CIII', 'C_MgII', 'C_Hbeta', 'C_Halpha']

    dir_path = os.path.dirname(redrock_path)
    qso_mgii_path = os.path.join(dir_path, os.path.basename(redrock_path).replace('redrock-', 'qso_mgii-'))
    qso_qn_path = os.path.join(dir_path, os.path.basename(redrock_path).replace('redrock-', 'qso_qn-'))
    if emline_path is None:
        emline_path = os.path.join(dir_path, os.path.basename(redrock_path).replace('redrock-', 'emline-'))

    tmp_redshifts = Table(fitsio.read(redrock_path, ext='REDSHIFTS', columns=columns_redshifts))
    tmp_fibermap = Table(fitsio.read(redrock_path, ext='FIBERMAP', columns=columns_fibermap))

    tmp_qso_mgii = Table(fitsio.read(qso_mgii_path, columns=(columns_qso_mgii)))
    tmp_qso_qn = Table(fitsio.read(qso_qn_path, columns=(columns_qso_qn)))
    if not ignore_emline:
        tmp_emline = Table(fitsio.read(emline_path, columns=(columns_emline)))

    # Sanity check
    tid = tmp_redshifts['TARGETID'].copy()
    if not (np.all(tid==tmp_fibermap['TARGETID']) \
            # and np.all(tid==tmp_emline['TARGETID']) \
            and np.all(tid==tmp_qso_mgii['TARGETID']) \
            and np.all(tid==tmp_qso_qn['TARGETID'])):
        raise ValueError
    if (not ignore_emline) and (not np.all(tid==tmp_emline['TARGETID'])):
        raise ValueError

    tmp_fibermap.remove_column('TARGETID')
    tmp_qso_mgii.remove_column('TARGETID')
    tmp_qso_qn.remove_column('TARGETID')
    if not ignore_emline:
        tmp_emline.remove_column('TARGETID')

    if not ignore_emline:
        cat = hstack([tmp_redshifts, tmp_fibermap, tmp_qso_mgii, tmp_qso_qn, tmp_emline], join_type='inner')
    else:
        cat = hstack([tmp_redshifts, tmp_fibermap, tmp_qso_mgii, tmp_qso_qn], join_type='inner')

    if return_target_columns:
        from desitarget.targetmask import desi_mask, bgs_mask
        for name in ['LRG', 'ELG', 'QSO', 'ELG_LOP', 'ELG_HIP', 'ELG_VLO', 'BGS_ANY', 'BGS_FAINT', 'BGS_BRIGHT']:
            if name in ['BGS_FAINT', 'BGS_BRIGHT']:
                cat[name] = cat['BGS_TARGET'] & bgs_mask[name] > 0
            else:
                cat[name] = cat['DESI_TARGET'] & desi_mask[name] > 0
        # # Bitmask definitions: https://github.com/desihub/desitarget/blob/master/py/desitarget/data/targetmask.yaml
        # cat['LRG'] = cat['DESI_TARGET'] & 2**0 > 0
        # cat['ELG'] = cat['DESI_TARGET'] & 2**1 > 0
        # cat['QSO'] = cat['DESI_TARGET'] & 2**2 > 0
        # cat['ELG_LOP'] = cat['DESI_TARGET'] & 2**5 > 0
        # cat['ELG_HIP'] = cat['DESI_TARGET'] & 2**6 > 0
        # cat['ELG_VLO'] = cat['DESI_TARGET'] & 2**7 > 0
        # cat['BGS_ANY'] = cat['DESI_TARGET'] & 2**60 > 0
        # cat['BGS_FAINT'] = cat['BGS_TARGET'] & 2**0 > 0
        # cat['BGS_BRIGHT'] = cat['BGS_TARGET'] & 2**1 > 0

    cat = actually_validate(cat, fiberstatus_cut=fiberstatus_cut, ignore_emline=ignore_emline)
    cat = cat[output_columns]

    return cat


def actually_validate(cat, fiberstatus_cut=True, ignore_emline=False):
    '''
    Apply redshift quality criteria

    Args:
        cat: astropy table with the necessary columns for redshift quality determination

    Options:
        fiberstatus_cut: bool (default True), if True, impose requirements on COADD_FIBERSTATUS and ZWARN
        return_target_columns: bool (default False), if True, include columns that indicate if the object belongs to each class of DESI targets
        extra_columns: list of str (default None), additional columns to include in the output
        emline_path: str (default None), specify the location of the emline file; by default the emline file is in the same directory as the redrock file
        ignore_emline: bool (default False), if True, ignore the emline file and do not validate the ELG redshift

    Returns:
        cat: astropy table with boolean columns (e.g., GOOD_BGS) added
    '''

    # BGS
    cat['GOOD_BGS'] = cat['ZWARN']==0
    cat['GOOD_BGS'] &= cat['DELTACHI2']>40
    if fiberstatus_cut:
        cat['GOOD_BGS'] &= get_good_fiberstatus(cat)

    # LRG
    cat['GOOD_LRG'] = cat['ZWARN']==0
    cat['GOOD_LRG'] &= cat['Z']<1.5
    cat['GOOD_LRG'] &= cat['DELTACHI2']>15
    if fiberstatus_cut:
        cat['GOOD_LRG'] &= get_good_fiberstatus(cat)

    # ELG
    if not ignore_emline:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cat['GOOD_ELG'] = (cat['OII_FLUX']>0) & (cat['OII_FLUX_IVAR']>0)
            cat['GOOD_ELG'] &= np.log10(cat['OII_FLUX'] * np.sqrt(cat['OII_FLUX_IVAR'])) > 0.9 - 0.2 * np.log10(cat['DELTACHI2'])
        if fiberstatus_cut:
            cat['GOOD_ELG'] &= get_good_fiberstatus(cat)

    # QSO - adopted from the code from Edmond
    # https://github.com/echaussidon/LSS/blob/8ca53f4c38cfa29722ee6958687e188cc894ed2b/py/LSS/qso_cat_utils.py#L282
    cat['IS_QSO_QN'] = np.max(np.array([cat[name] for name in ['C_LYA', 'C_CIV', 'C_CIII', 'C_MgII', 'C_Hbeta', 'C_Halpha']]), axis=0) > 0.95
    cat['IS_QSO_QN_NEW_RR'] &= cat['IS_QSO_QN']
    cat['QSO_MASKBITS'] = np.zeros(len(cat), dtype=int)
    cat['QSO_MASKBITS'][cat['SPECTYPE']=='QSO'] += 2**1
    cat['QSO_MASKBITS'][cat['IS_QSO_MGII']] += 2**2
    cat['QSO_MASKBITS'][cat['IS_QSO_QN']] += 2**3
    cat['QSO_MASKBITS'][cat['IS_QSO_QN_NEW_RR']] += 2**4
    cat['Z'][cat['IS_QSO_QN_NEW_RR']] = cat['Z_NEW'][cat['IS_QSO_QN_NEW_RR']].copy()
    cat['ZERR'][cat['IS_QSO_QN_NEW_RR']] = cat['ZERR_NEW'][cat['IS_QSO_QN_NEW_RR']].copy()
    # Correct bump at z~3.7
    sel_pb_redshift = (((cat['Z'] > 3.65) & (cat['Z'] < 3.9)) | ((cat['Z'] > 5.15) & (cat['Z'] < 5.35))) & ((cat['C_LYA'] < 0.95) | (cat['C_CIV'] < 0.95))
    cat['QSO_MASKBITS'][sel_pb_redshift] = 0
    cat['GOOD_QSO'] = cat['QSO_MASKBITS']>0
    if fiberstatus_cut:
        cat['GOOD_QSO'] &= get_good_fiberstatus(cat, isqso=True)

    return cat
