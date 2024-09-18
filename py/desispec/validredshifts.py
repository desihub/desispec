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


def get_good_fiberstatus(cat):
    '''
    Validate the fiber status for a redrock catalog.

    Args:
        cat: redrock catalog (e.g., in astropy table format)

    Returns:
        good_fiberstatus: boolean array
    '''

    good_fiberstatus = (cat['COADD_FIBERSTATUS']==0) | (cat['COADD_FIBERSTATUS']==2**3)  # allow bit 3 (restricted fiber reach)
    return good_fiberstatus


def validate(redrock_path, fiberstatus_cut=True, return_target_columns=False, extra_columns=None):
    '''
    Validate the redshift quality with tracer-dependent criteria for redrock+afterburner results.

    Args:
        redrock_path: str, path of redrock FITS file

    Options:
        fiberstatus_cut: bool (default True), if True, impose requirements on COADD_FIBERSTATUS and ZWARN
        return_target_columns: bool (default False), if True, include columns that indicate if the object belongs to each class of DESI targets
        extra_columns: list of str (default None), additional columns to include in the output

    Returns:
        cat: astropy table with basic columns such as TARGETID and boolean columns (e.g., GOOD_BGS)
             that indicate if each object meets the redshift quality criteria of specific tracers
    '''

    output_columns = ['GOOD_Z_BGS', 'GOOD_Z_LRG', 'GOOD_Z_ELG', 'GOOD_Z_QSO']
    if return_target_columns:
        output_columns = ['LRG', 'ELG', 'QSO', 'ELG_LOP', 'ELG_HIP', 'ELG_VLO', 'BGS_ANY', 'BGS_FAINT', 'BGS_BRIGHT'] + output_columns

    if extra_columns is None:
        extra_columns = ['TARGETID', 'Z', 'ZWARN', 'COADD_FIBERSTATUS']
    output_columns = list(np.array(extra_columns)[~np.in1d(extra_columns, output_columns)]) + output_columns

    ############################ Load data ############################

    columns_redshifts = ['TARGETID', 'CHI2', 'Z', 'ZERR', 'ZWARN', 'SPECTYPE', 'DELTACHI2']
    columns_fibermap = ['TARGETID', 'COADD_FIBERSTATUS', 'TARGET_RA', 'TARGET_DEC', 'OBJTYPE']
    columns_emline = ['TARGETID', 'OII_FLUX', 'OII_FLUX_IVAR']
    columns_qso_mgii = ['TARGETID', 'IS_QSO_MGII']
    columns_qso_qn = ['TARGETID', 'Z_NEW', 'ZERR_NEW', 'IS_QSO_QN_NEW_RR', 'C_LYA', 'C_CIV', 'C_CIII', 'C_MgII', 'C_Hbeta', 'C_Halpha']

    dir_path = os.path.dirname(redrock_path)
    qso_mgii_path = os.path.join(dir_path, os.path.basename(redrock_path).replace('redrock-', 'qso_mgii-'))
    qso_qn_path = os.path.join(dir_path, os.path.basename(redrock_path).replace('redrock-', 'qso_qn-'))
    emline_path = os.path.join(dir_path, os.path.basename(redrock_path).replace('redrock-', 'emline-'))

    tmp_redshifts = Table(fitsio.read(redrock_path, ext='REDSHIFTS', columns=columns_redshifts))
    tid = tmp_redshifts['TARGETID'].copy()

    # read the full fibermap until we determine the targeting columns
    from desitarget.targets import main_cmx_or_sv
    tmp_fibermap = fitsio.read(redrock_path, ext='FIBERMAP')
    surv_target, surv_mask, surv = main_cmx_or_sv(tmp_fibermap)
    if surv.lower() == 'cmx':
        raise NotImplementedError('Determining valid redshifts for commissioning targets is not supported.')

    desi_target_col, bgs_target_col, _ = surv_target
    desi_mask, bgs_mask, _ = surv_mask
    tmp_fibermap = Table(tmp_fibermap[columns_fibermap + surv_target])
    assert np.all(tid==tmp_fibermap['TARGETID'])
    tmp_fibermap.remove_column('TARGETID')

    ignore_emline = False
    ignore_qso = False

    if os.path.isfile(emline_path):
        tmp_emline = Table(fitsio.read(emline_path, columns=(columns_emline)))
        assert np.all(tid==tmp_emline['TARGETID'])
        tmp_emline.remove_column('TARGETID')
    else:
        print('emline file not found:', emline_path)
        ignore_emline = True

    if os.path.isfile(qso_mgii_path):
        tmp_qso_mgii = Table(fitsio.read(qso_mgii_path, columns=(columns_qso_mgii)))
        assert np.all(tid==tmp_qso_mgii['TARGETID'])
        tmp_qso_mgii.remove_column('TARGETID')
    else:
        print('qso_mgii file not found:', qso_mgii_path)
        ignore_qso = True

    if os.path.isfile(qso_qn_path):
        tmp_qso_qn = Table(fitsio.read(qso_qn_path, columns=(columns_qso_qn)))
        assert np.all(tid==tmp_qso_qn['TARGETID'])
        tmp_qso_qn.remove_column('TARGETID')
    else:
        print('qso_qn file not found:', qso_qn_path)
        ignore_qso = True

    cat = hstack([tmp_redshifts, tmp_fibermap], join_type='exact')
    if not ignore_emline:
        cat = hstack([cat, tmp_emline], join_type='exact')
    if not ignore_qso:
        cat = hstack([cat, tmp_qso_mgii, tmp_qso_qn], join_type='exact')

    if return_target_columns:
        for name in ['LRG', 'ELG', 'QSO', 'ELG_LOP', 'ELG_HIP', 'ELG_VLO', 'BGS_ANY', 'BGS_FAINT', 'BGS_BRIGHT']:
            if name in ['BGS_FAINT', 'BGS_BRIGHT']:
                cat[name] = cat[bgs_target_col] & bgs_mask[name] > 0
            else:
                if name in desi_mask.names(): # not all bits were used in SV (e.g., ELG_LOP)
                    cat[name] = cat[desi_target_col] & desi_mask[name] > 0
                else:
                    cat[name] = np.zeros(len(cat), bool)
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

    res = actually_validate(cat, fiberstatus_cut=fiberstatus_cut, ignore_emline=ignore_emline, ignore_qso=ignore_qso)
    cat = hstack([cat, res])

    output_columns = [col for col in output_columns if col in cat.colnames]
    cat = cat[output_columns]

    return cat


def actually_validate(cat, fiberstatus_cut=True, ignore_emline=False, ignore_qso=False):
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
        ignore_qso: bool (default False), if True, do not validate the QSO redshift

    Returns:
        res: astropy table with boolean columns (e.g., GOOD_BGS)
    '''

    res = Table()

    # BGS
    res['GOOD_Z_BGS'] = cat['ZWARN']==0
    res['GOOD_Z_BGS'] &= cat['DELTACHI2']>40
    if fiberstatus_cut:
        res['GOOD_Z_BGS'] &= get_good_fiberstatus(cat)

    # LRG
    res['GOOD_Z_LRG'] = cat['ZWARN']==0
    res['GOOD_Z_LRG'] &= cat['Z']<1.5
    res['GOOD_Z_LRG'] &= cat['DELTACHI2']>15
    if fiberstatus_cut:
        res['GOOD_Z_LRG'] &= get_good_fiberstatus(cat)

    # ELG
    if not ignore_emline:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res['GOOD_Z_ELG'] = (cat['OII_FLUX']>0) & (cat['OII_FLUX_IVAR']>0)
            res['GOOD_Z_ELG'] &= np.log10(cat['OII_FLUX'] * np.sqrt(cat['OII_FLUX_IVAR'])) > 0.9 - 0.2 * np.log10(cat['DELTACHI2'])
        if fiberstatus_cut:
            res['GOOD_Z_ELG'] &= get_good_fiberstatus(cat)

    if not ignore_qso:
        # QSO - adopted from the code from Edmond
        # https://github.com/echaussidon/LSS/blob/8ca53f4c38cfa29722ee6958687e188cc894ed2b/py/LSS/qso_cat_utils.py#L282
        res['IS_QSO_QN'] = np.max(np.array([cat[name] for name in ['C_LYA', 'C_CIV', 'C_CIII', 'C_MgII', 'C_Hbeta', 'C_Halpha']]), axis=0) > 0.99  # new threshold for Y3
        res['IS_QSO_QN_NEW_RR'] = cat['IS_QSO_QN_NEW_RR'] & res['IS_QSO_QN']
        res['QSO_MASKBITS'] = np.zeros(len(cat), dtype=int)
        res['QSO_MASKBITS'][cat['SPECTYPE']=='QSO'] += 2**1
        res['QSO_MASKBITS'][cat['IS_QSO_MGII']] += 2**2
        res['QSO_MASKBITS'][res['IS_QSO_QN']] += 2**3
        res['QSO_MASKBITS'][res['IS_QSO_QN_NEW_RR']] += 2**4
        res['GOOD_Z_QSO'] = res['QSO_MASKBITS']>0
        res['GOOD_Z_QSO'] &= cat['OBJTYPE']=='TGT'
        if fiberstatus_cut:
            res['GOOD_Z_QSO'] &= get_good_fiberstatus(cat)
        mask_new_z = res['GOOD_Z_QSO'] & res['IS_QSO_QN_NEW_RR']
        res['Z_QSO'] = cat['Z'].copy()
        res['Z_QSO'][mask_new_z] = cat['Z_NEW'][mask_new_z].copy()
        res['ZERR_QSO'] = cat['ZERR'].copy()
        res['ZERR_QSO'][mask_new_z] = cat['ZERR_NEW'][mask_new_z].copy()

    # Remove unnecessary columns
    columns_to_keep = ['GOOD_Z_BGS', 'GOOD_Z_LRG', 'GOOD_Z_ELG', 'GOOD_Z_QSO', 'Z_QSO', 'ZERR_QSO']
    columns_to_keep = [col for col in columns_to_keep if col in res.colnames]
    res = res[columns_to_keep]

    return res
