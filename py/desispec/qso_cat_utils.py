#!/usr/bin/env python
# coding: utf-8

"""
author:  edmond chaussidon (CEA saclay)
contact: edmond.chaussidon@cea.fr

Remarks:
    * 1) logger:

         If you want to desactivate the logger (ie) information display in your terminal.
         Add these two lines in your script once the module is loaded.
         # import logging
         # logging.getLogger("QSO_CAT_UTILS").setLevel(logging.ERROR)

    * 2) Data:

        The QSO catalog will be (for the moment) available here:
                `/global/cfs/cdirs/desi/users/edmondc/QSO_catalog/`.
        For additional information, please read the README.md file.
        Any requests or comments are welcome.

    * 3) Quality cut:

         Here we do not apply any quality cuts. Please consider to add the following cuts:
            * NO cut on ZWARN !!
            * for release <= everest: fiber_ok = (cat['COADD_FIBERSTATUS']==0)
            * for release >= fuji: fiber_ok = (cat['COADD_FIBERSTATUS']==0) | (cat['COADD_FIBERSTATUS']==8388608) | (cat['COADD_FIBERSTATUS']==16777216)
            * definition of maskbits: https://github.com/desihub/desispec/blob/master/py/desispec/maskbits.py
"""

import sys
import os
import glob
import logging

import fitsio
import numpy as np
import pandas as pd


log = logging.getLogger("QSO_CAT_UTILS")


def desi_target_from_survey(survey):
    """ Return the survey of DESI_TARGET as a function of survey used (cmx, sv1, sv2, sv3, main)."""
    if survey == 'cmx':
        return 'CMX_TARGET'
    elif survey == 'sv1':
        return 'SV1_DESI_TARGET'
    elif survey == 'sv2':
        return 'SV2_DESI_TARGET'
    elif survey == 'sv3':
        return 'SV3_DESI_TARGET'
    elif survey == 'main':
        return 'DESI_TARGET'


def read_fits_to_pandas(filename, ext=1, columns=None):
    """
    Read a .fits file and convert it into a :class:`pandas.DataFrame`.
    Warning: it does not work if a column contains a list or an array.
    Parameters
    ----------
    filename : str
        Path where the .fits file is saved.
    ext : int or str
        Extension to read.
    columns : list of str
        List of columns to read. Useful to avoid to use too much memory.
    Returns :
    ---------
    data_frame : pandas.DataFrame
        Data frame containing data in the fits file.
    """
    log.info(f'Read ext: {ext} from {filename}')
    file = fitsio.FITS(filename)[ext]
    if columns is not None: file = file[columns]
    return pd.DataFrame(file.read().byteswap().newbyteorder())


def save_dataframe_to_fits(dataframe, filename, extname="QSO_CAT", clobber=True):
    """
    Save info from pandas dataframe in a fits file.

    Remark: Here we do not expect complex structure into dataframe (ie) only int/float/bool are expected in columns.
            We can use df.to_records().
    Args:
        dataframe (pandas dataframe): dataframe containg the all the necessary QSO info
        filename (str):  name of the fits file
        extname (str): name of the hdu in which the dataframe will be written
        clobber (bool):  overwrite the fits file defined by filename ? default=True
    Returns:
        None
    """
    # No complex structure, to_records() is sufficient.
    fits = fitsio.FITS(filename, 'rw', clobber=clobber)
    if clobber:
        log.warning(f'OVERWRITE the file : {filename}')
    else:
        log.warning(f'EXPAND the file : {filename}')
    fits.write(dataframe.to_records(index=False), extname=extname)
    fits.close()


def qso_catalog_maker(redrock, mgii, qn, use_old_extname_for_redrock=False, use_old_extname_for_fitsio=False, keep_all=False):
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
        keep_all (bool): if True return all the targets. if False return only targets which are selected as QSO.
    Returns:
        QSO_cat (pandas dataframe): Dataframe containing all the information
    """
    from functools import reduce

    # selection of which column will be in the final QSO_cat:
    columns_zbest = ['TARGETID', 'Z', 'ZERR', 'ZWARN', 'SPECTYPE'] #, 'SUBTYPE', 'DELTACHI2', 'CHI2']
    # remark: check if the name exist before selecting them
    # for instance, in healpix directory 'LOCATION' / 'FIBER' will be not added as columns
    columns_fibermap = ['TARGETID', 'TARGET_RA', 'TARGET_DEC', 'LOCATION', 'FIBER', 'COADD_FIBERSTATUS', 'COADD_NUMEXP', 'COADD_EXPTIME',
                        'EBV', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2',
                        'FLUX_IVAR_G', 'FLUX_IVAR_R', 'FLUX_IVAR_Z', 'FLUX_IVAR_W1', 'FLUX_IVAR_W2', 'MASKBITS',
                        'CMX_TARGET', 'SV1_DESI_TARGET', 'SV2_DESI_TARGET', 'SV3_DESI_TARGET', 'DESI_TARGET']

    columns_tsnr2 = ['TARGETID', 'TSNR2_QSO', 'TSNR2_LYA']

    columns_mgii = ['TARGETID', 'IS_QSO_MGII', 'DELTA_CHI2', 'A', 'SIGMA', 'B', 'VAR_A', 'VAR_SIGMA', 'VAR_B']
    columns_mgii_rename = {"DELTA_CHI2": "DELTA_CHI2_MGII", "A": "A_MGII", "SIGMA":"SIGMA_MGII", "B":"B_MGII", "VAR_A":"VAR_A_MGII", "VAR_SIGMA":"VAR_SIGMA_MGII", "VAR_B":"VAR_B_MGII"}

    columns_qn = ['TARGETID', 'Z_NEW', 'ZERR_NEW', 'Z_RR', 'Z_QN', 'IS_QSO_QN_NEW_RR',
                  'C_LYA', 'C_CIV', 'C_CIII', 'C_MgII', 'C_Hbeta', 'C_Halpha',
                  'Z_LYA', 'Z_CIV', 'Z_CIII', 'Z_MgII', 'Z_Hbeta', 'Z_Halpha']

    #load data:
    zbest = read_fits_to_pandas(redrock, ext='ZBEST' if use_old_extname_for_redrock else 'REDSHIFTS', columns=columns_zbest)
    fibermap = read_fits_to_pandas(redrock, ext='FIBERMAP', columns=[name for name in columns_fibermap if name in fitsio.read(redrock, ext='FIBERMAP', rows=[0]).dtype.names])
    tsnr2 = read_fits_to_pandas(redrock, ext='TSNR2', columns=columns_tsnr2)
    mgii = read_fits_to_pandas(mgii, ext='MGII', columns=columns_mgii).rename(columns=columns_mgii_rename)
    qn = read_fits_to_pandas(qn, ext='QN+RR' if use_old_extname_for_fitsio else 'QN_RR', columns=columns_qn)

    # add DESI_TARGET column to avoid error of conversion when concatenate the different files with pd.concat() which fills with NaN columns that do not exit in a DataFrame.
    # convert int64 to float 64 --> destructive tranformation !!
    for DESI_TARGET in ['CMX_TARGET', 'SV1_DESI_TARGET', 'SV2_DESI_TARGET', 'SV3_DESI_TARGET', 'DESI_TARGET']:
        if not(DESI_TARGET in fibermap.columns):
            fibermap[DESI_TARGET] = np.zeros(fibermap['TARGETID'].size, dtype=np.int64)

    # QN afterburner is run with a threshold 0.5. With VI, we choose 0.95 as final threshold.
    # &= since IS_QSO_QN_NEW_RR contains only QSO for QN which are not QSO for RR.
    log.info('Increase the QN threshold selection from 0.5 to 0.95.')
    qn['IS_QSO_QN'] = np.max(np.array([qn[name] for name in ['C_LYA', 'C_CIV', 'C_CIII', 'C_MgII', 'C_Hbeta', 'C_Halpha']]), axis=0) > 0.95
    qn['IS_QSO_QN_NEW_RR'] &= qn['IS_QSO_QN']

    log.info('Merge on TARGETID all the info into a singe dataframe.')
    QSO_cat = reduce(lambda left, right: pd.merge(left, right, on=['TARGETID'], how='outer'), [zbest, fibermap, tsnr2, mgii, qn])

    # ADD BITMASK:
    QSO_cat['QSO_MASKBITS'] = np.zeros(QSO_cat.shape[0], dtype='i')
    log.info('Selection with SPECTYPE.')
    QSO_cat.loc[QSO_cat['SPECTYPE'] == 'QSO', 'QSO_MASKBITS'] += 2**1
    log.info('Selection with MgII.')
    QSO_cat.loc[QSO_cat['IS_QSO_MGII'], 'QSO_MASKBITS'] += 2**2
    log.info('Selection with QN (add new z from Redrock with QN prior where it is relevant).')
    QSO_cat.loc[QSO_cat['IS_QSO_QN'], 'QSO_MASKBITS'] += 2**3
    QSO_cat.loc[QSO_cat['IS_QSO_QN_NEW_RR'], 'QSO_MASKBITS'] += 2**4
    QSO_cat.loc[QSO_cat['IS_QSO_QN_NEW_RR'], 'Z'] = QSO_cat['Z_NEW'][QSO_cat['IS_QSO_QN_NEW_RR']].values
    QSO_cat.loc[QSO_cat['IS_QSO_QN_NEW_RR'], 'ZERR'] = QSO_cat['ZERR_NEW'][QSO_cat['IS_QSO_QN_NEW_RR']].values

    # remove useless columns:
    QSO_cat.drop(columns=['IS_QSO_MGII', 'IS_QSO_QN', 'IS_QSO_QN_NEW_RR', 'Z_NEW', 'ZERR_NEW'], inplace=True)

    # Correct bump at z~3.7
    sel_pb_redshift = (QSO_cat['Z'] > 3.65) & ((QSO_cat['C_LYA']<0.95) | (QSO_cat['C_CIV']<0.95))
    log.info(f'Remove bump at z~3.7: exclude {sel_pb_redshift.sum()} QSOs.')
    QSO_cat.loc[sel_pb_redshift, 'QSO_MASKBITS'] = 0

    if keep_all:
        log.info('Return all the targets without any cut on QSO selection.')
        return QSO_cat
    else:
        QSO_cat = QSO_cat[QSO_cat['QSO_MASKBITS'] > 0]
        if QSO_cat.shape[0] == 0:
            log.info('No QSO found...')
        else:
            log.info(f"Final selection gives: {QSO_cat.shape[0]} QSO !")
        return QSO_cat


def qso_catalog_for_a_tile(path_to_tile, tile, last_night, survey, program):
    """
    Build the QSO catalog for the tile using the last_night. It is relevant for cumulative directory.
    This function is usefull to be called in pool.starmap under multiprocessing.

    Args:
        path_to_tile (str): Where the tiles are.
        tile (str): which tile do you want to treat.
        last_night (str): corresponding last night to tile
        survey (str): sv3/main ... only to add information to the catalog
        program (str): dark/bright/backup only to add information to the catalog

    Return:
        QSO_cat (DataFrame): pandas DataFrame containing the concatenation of run_catalog_maker in each available petal
    """

    def run_catalog_maker(path_to_tile, tile, night, petal, survey, program):
        """Run qso_catalog_maker in the considered tile-last_night-petal. If one file does not exist it return a void DataFrame."""
        redrock          = os.path.join(path_to_tile, tile, night, f"redrock-{petal}-{tile}-thru{night}.fits")
        mgii_afterburner = os.path.join(path_to_tile, tile, night, f"qso_mgii-{petal}-{tile}-thru{night}.fits")
        qn_afterburner   = os.path.join(path_to_tile, tile, night, f"qso_qn-{petal}-{tile}-thru{night}.fits")

        if os.path.isfile(redrock) & os.path.isfile(mgii_afterburner) & os.path.isfile(qn_afterburner):
            qso_cat = qso_catalog_maker(redrock, mgii_afterburner, qn_afterburner)
            qso_cat['TILEID'] = tile
            qso_cat['LASTNIGHT'] = night
            qso_cat['PETAL_LOC'] = petal
            qso_cat['SURVEY'] = survey
            qso_cat['PROGRAM'] = program
        else:
            log.warning(f'There is a problem with: {redrock} | {mgii_afterburner} | {qn_afterburner}')
            qso_cat = pd.DataFrame()
        return qso_cat

    return pd.concat([run_catalog_maker(path_to_tile, tile, last_night, petal, survey, program) for petal in range(10)], ignore_index=True)


def build_qso_catalog_from_tiles(redux='/global/cfs/cdirs/desi/spectro/redux/', release='fuji', dir_output='', npool=20):
    """
    Build the QSO catalog from the healpix directory.

    Warning: no retro compatibility for release <= everest (extname has changed --> the option can be added since it exists in qso_catalog_maker)

    Args:
        * redux (str): path where is saved the spectroscopic data.
        * release (str): which release do you want to use (everest, fuji, guadalupe, ect...).
        * dir_output (str): directory where the QSO catalog will be saved.
        * npool (int): nbr of workers used for the parallelisation.
    """
    import multiprocessing
    from itertools import repeat
    import tqdm

    # remove desimodule log
    os.environ["DESI_LOGLEVEL"] = "ERROR"

    # Data directory
    DIR = os.path.join(redux, release, 'tiles', 'cumulative')

    # load tiles info:
    tile_info = fitsio.FITS(f'/global/cfs/cdirs/desi/spectro/redux/{release}/tiles-{release}.fits')[1][['TILEID', 'LASTNIGHT', 'SURVEY', 'PROGRAM']]

    tiles = np.array(tile_info['TILEID'][:], dtype='str')
    last_night = np.array(tile_info['LASTNIGHT'][:], dtype='str')
    survey = np.array(tile_info['SURVEY'][:], dtype='str')
    program = np.array(tile_info['PROGRAM'][:], dtype='str')

    log.info(f'There are {tiles.size} tiles to treat with npool={npool}')
    logging.getLogger("QSO_CAT_UTILS").setLevel(logging.ERROR)
    with multiprocessing.Pool(npool) as pool:
        arguments = zip(repeat(DIR), tiles, last_night, survey, program)
        QSO_cat = pd.concat(pool.starmap(qso_catalog_for_a_tile, arguments), ignore_index=True)
    logging.getLogger("QSO_CAT_UTILS").setLevel(logging.INFO)

    save_dataframe_to_fits(QSO_cat, os.path.join(dir_output, f'QSO_cat_{release}_cumulative.fits'))


def qso_catalog_for_a_pixel(path_to_pix, pre_pix, pixel, survey, program, keep_all=False):
    """
    Build the QSO catalog for the tile using the last_night. It is relevant for cumulative directory.
    This function is usefull to be called in pool.starmap under multiprocessing.

    Args:
        * path_to_pix (str): Where the pixels are.
        * pre_pix (str): which pre_pix in healpix directory do you want to use.
        * pixel (str): which pixel do you want to use.
        * survey (str): which TS do you want to use (sv1/sv3/main)
        * program (str): either dark / bright / backup
        * keep_all (bool): if True return all the targets. if False return only targets which are selected as QSO.

    Return:
        QSO_cat (DataFrame): pandas DataFrame containing the QSO_catalog for the considered pixel.
    """
    redrock          = os.path.join(path_to_pix, str(pre_pix), str(pixel), f"redrock-{survey}-{program}-{pixel}.fits")
    mgii_afterburner = os.path.join(path_to_pix, str(pre_pix), str(pixel), f"qso_mgii-{survey}-{program}-{pixel}.fits")
    qn_afterburner   = os.path.join(path_to_pix, str(pre_pix), str(pixel), f"qso_qn-{survey}-{program}-{pixel}.fits")

    if os.path.isfile(redrock) & os.path.isfile(mgii_afterburner) & os.path.isfile(qn_afterburner):
        qso_cat = qso_catalog_maker(redrock, mgii_afterburner, qn_afterburner, keep_all=keep_all)
        qso_cat['HPXPIXEL'] = pixel
        qso_cat['SURVEY'] = survey
        qso_cat['PROGRAM'] = program
    else:
        log.warning(f'There is a problem with: {redrock} | {mgii_afterburner} | {qn_afterburner}')
        qso_cat = pd.DataFrame()
    return qso_cat


def build_qso_catalog_from_healpix(redux='/global/cfs/cdirs/desi/spectro/redux/', release='fuji', survey='sv3', program='dark', dir_output='', npool=20, keep_qso_targets=True, keep_all=False):
    """
    Build the QSO catalog from the healpix directory.

    Warning: no retro compatibility for release <= everest (extname has changed --> the option can be added since it exists in qso_catalog_maker)

    Args:
        * redux (str): path where is saved the spectroscopic data.
        * release (str): which release do you want to use (everest, fuji, guadalupe, ect...).
        * survey (str): which survey of the target selection (sv1, sv3, main).
        * program (str) : either dark / bright or backup program.
        * dir_output (str): directory where the QSO catalog will be saved.
        * npool (int): nbr of workers used for the parallelisation.
        * keep_qso_targets (bool): if True save only QSO targets. default=True
        * keep_all (bool): if True return all the targets. if False return only targets which are selected as QSO. default=False
    """
    import multiprocessing
    from itertools import repeat
    import tqdm

    # remove desimodule log
    os.environ["DESI_LOGLEVEL"] = "ERROR"

    # Data directory
    DIR = os.path.join(redux, release, 'healpix', survey, program)

    # Collect the pre-pixel and pixel number
    pre_pix_list = np.sort([os.path.basename(path) for path in glob.glob(os.path.join(DIR, "*"))])
    pre_pix_list_long, pixel_list = [], []
    for pre_pix in pre_pix_list:
        pixel_list_tmp = [os.path.basename(path) for path in glob.glob(os.path.join(DIR, pre_pix, "*"))]
        pre_pix_list_long += [pre_pix]*len(pixel_list_tmp)
        pixel_list += pixel_list_tmp

    log.info(f'There are {len(pixel_list)} pixels to treat with npool={npool}')
    logging.getLogger("QSO_CAT_UTILS").setLevel(logging.ERROR)
    with multiprocessing.Pool(npool) as pool:
        arguments = zip(repeat(DIR), pre_pix_list_long, pixel_list, repeat(survey), repeat(program), repeat(keep_all))
        QSO_cat = pd.concat(pool.starmap(qso_catalog_for_a_pixel, arguments), ignore_index=True)
    logging.getLogger("QSO_CAT_UTILS").setLevel(logging.INFO)

    if keep_qso_targets:
        log.info('Keep only qso targets...')
        suffix = '_only_qso_targets'
        save_dataframe_to_fits(QSO_cat.iloc[QSO_cat[desi_target_from_survey(survey)].values & 2**2 != 0], os.path.join(dir_output, f'QSO_cat_{release}_{survey}_{program}_healpix_only_qso_targets.fits'))

    suffix = ''
    if keep_all:
        suffix = '_all'
    save_dataframe_to_fits(QSO_cat, os.path.join(dir_output, f'QSO_cat_{release}_{survey}_{program}_healpix{suffix}.fits'))


def afterburner_is_missing_in_tiles(redux='/global/cfs/cdirs/desi/spectro/redux/', release='fuji', outdir=''):
    """
    Goes throught all the directory of tiles and check if afterburner files exist when the associated redrock file exist also.
    If files are missing, they are saved in .txt file.
    Args:
        * redux (str): path where is saved the spectroscopic data.
        * release (str): which release do you want to check.
        * outdir (str): path where the .txt output will be saved in case if it lacks some afterburner files.
    """
    import tqdm

    dir_list = ['pernight', 'perexp', 'cumulative']
    suff_dir_list = ['', 'exp', 'thru']

    for dir, suff_dir in zip(dir_list, suff_dir_list):
        DIR = os.path.join(redux, release, 'tiles', dir)
        tiles = np.sort([os.path.basename(path) for path in glob.glob(os.path.join(DIR, '*'))])
        log.info(f'Inspection of {tiles.size} tiles in {DIR}...')
        pb_qn, pb_mgII = [], []

        for tile in tqdm.tqdm(tiles):
            nights = np.sort([os.path.basename(path) for path in glob.glob(os.path.join(DIR, tile, '*'))])
            for night in nights:
                for petal in range(10):
                    if os.path.isfile(os.path.join(DIR, tile, night, f"redrock-{petal}-{tile}-{suff_dir}{night}.fits")):
                        if not (os.path.isfile(os.path.join(DIR, tile, night, f"qso_qn-{petal}-{tile}-{suff_dir}{night}.fits"))
                             or os.path.isfile(os.path.join(DIR, tile, night, f"qso_qn-{petal}-{tile}-{suff_dir}{night}.notargets"))
                             or os.path.isfile(os.path.join(DIR, tile, night, f"qso_qn-{petal}-{tile}-{suff_dir}{night}.misscamera"))):
                            pb_qn += [[int(tile), int(night), int(petal)]]
                        if not (os.path.isfile(os.path.join(DIR, tile, night, f"qso_mgii-{petal}-{tile}-{suff_dir}{night}.fits"))
                             or os.path.isfile(os.path.join(DIR, tile, night, f"qso_mgii-{petal}-{tile}-{suff_dir}{night}.notargets"))
                             or os.path.isfile(os.path.join(DIR, tile, night, f"qso_mgii-{petal}-{tile}-{suff_dir}{night}.misscamera"))):
                            pb_mgII += [[int(tile), int(night), int(petal)]]

        log.info(f'Under the directory {DIR} it lacks:')
        log.info(f'    * {len(pb_qn)} QN files')
        log.info(f'    * {len(pb_mgII)} MgII files')
        if len(pb_qn) > 0:
            np.savetxt(os.path.join(outdir, f'pb_qn_{release}_{dir}.txt'), pb_qn, fmt='%d')
        if len(pb_mgII) > 0:
            np.savetxt(os.path.join(outdir, f'pb_mgII_{release}_{dir}.txt'), pb_mgII, fmt='%d')


def afterburner_is_missing_in_healpix(redux='/global/cfs/cdirs/desi/spectro/redux/', release='fuji', outdir=''):
    """
    Goes throught all the directory of healpix and check if afterburner files exist when the associated redrock file exist also.
    If files are missing, they are saved in .txt file.
    Args:
        * redux (str): path where is saved the spectroscopic data.
        * release (str): which release do you want to check.
        * outdir (str): path where the .txt output will be saved in case if it lacks some afterburner files.
    """
    import tqdm

    DIR = os.path.join(redux, release, 'healpix')

    #sv1 / sv3 / main
    survey_list = [os.path.basename(path) for path in glob.glob(os.path.join(DIR, '*'))]
    for survey in survey_list:

        # dark / bright / backup
        program_list = [os.path.basename(path) for path in glob.glob(os.path.join(DIR, survey, '*'))]
        for program in program_list:
            log.info(f'Inspection of {os.path.join(DIR, survey, program)}...')
            pb_qn, pb_mgII = [], []

            # collect the huge pixels directory
            healpix_huge_pixels = np.sort([os.path.basename(path) for path in glob.glob(os.path.join(DIR, survey, program, '*'))])
            for num in tqdm.tqdm(healpix_huge_pixels):
                pix_numbers = np.sort([os.path.basename(path) for path in glob.glob(os.path.join(DIR, survey, program, num, '*'))])
                for pix in pix_numbers:
                    if os.path.isfile(os.path.join(DIR, survey, program, num, pix, f"redrock-{survey}-{program}-{pix}.fits")):
                        if not (os.path.isfile(os.path.join(DIR, survey, program, num, pix, f"qso_qn-{survey}-{program}-{pix}.fits"))
                             or os.path.isfile(os.path.join(DIR, survey, program, num, pix, f"qso_qn-{survey}-{program}-{pix}.notargets"))
                             or os.path.isfile(os.path.join(DIR, survey, program, num, pix, f"qso_qn-{survey}-{program}-{pix}.misscamera"))):
                            pb_qn += [[int(num), int(pix)]]
                        if not (os.path.isfile(os.path.join(DIR, survey, program, num, pix, f"qso_mgii-{survey}-{program}-{pix}.fits"))
                            or  os.path.isfile(os.path.join(DIR, survey, program, num, pix, f"qso_mgii-{survey}-{program}-{pix}.notargets"))
                            or  os.path.isfile(os.path.join(DIR, survey, program, num, pix, f"qso_mgii-{survey}-{program}-{pix}.misscamera"))):
                            pb_mgII += [[int(num), int(pix)]]

            log.info(f'Under the directory {os.path.join(DIR, survey, program)} it lacks:')
            log.info(f'    * {len(pb_qn)} QN files')
            log.info(f'    * {len(pb_mgII)} MgII files')
            if len(pb_qn) > 0:
                np.savetxt(os.path.join(outdir, f'pb_qn_healpix_{survey}_{program}.txt'), pb_qn, fmt='%d')
            if len(pb_mgII) > 0:
                np.savetxt(os.path.join(outdir, f'pb_mgII_healpix_{survey}_{program}.txt'), pb_mgII, fmt='%d')


if __name__ == '__main__':
    from desiutil.log import get_logger
    log = get_logger()

    if sys.argv[1] == 'inspect_afterburners':
        """ Simple inspection of qso afterburner in fuji and guadalupe release. """
        log.info('Test the existence of QSO afterburners in Fuji and in Guadalupe.\nWe check is mgII and qn files were produced if the corresponding redrock file exits.')

        log.info('Inspect Fuji...')
        afterburner_is_missing_in_tiles(release='fuji')
        afterburner_is_missing_in_healpix(release='fuji')

        log.info('Inspect Guadalupe...')
        afterburner_is_missing_in_tiles(release='guadalupe')
        afterburner_is_missing_in_healpix(release='guadalupe')

    if sys.argv[1] == 'build_qso_catalog':
        """ Simple example of how to build the QSO catalog from healpix or cumulative directory"""

        redux = '/global/cfs/cdirs/desi/spectro/redux/'

        log.info(f'Build QSO catalog from cumulative directory for guadalupe release:')
        build_qso_catalog_from_tiles(redux=redux, release='guadalupe', dir_output='')

        log.info(f'Build QSO catalog from healpix directory for guadalupe release:')
        build_qso_catalog_from_healpix(redux=redux, release='guadalupe', survey='main', program='dark', dir_output='')
