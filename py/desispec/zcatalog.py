"""
desispec.zcatalog
=================

The zcatalog.py script contains the following utility functions for redshift catalogs:

(1) find_primary_spectra:

Given an input table with possibly multiple spectra per TARGETID, this function returns arrays for
a primary flag (whether a spectrum is the primary ["best"] spectrum based on the ZWARN value and
on a sort column [default='TSNR2_LRG']) and for the number of spectra per TARGETID.

Usage::

    nspec, spec_primary = find_primary_spectra(table, sort_column = 'TSNR2_LRG')

(2) create_summary_catalog:

This function combines individual redshift catalogs from a given release into a single compilation
catalog. The output catalog is saved in a FITS file in a user-specified location and filename.

This function can be used for 'fuji' (EDR) or 'guadalupe' (Main) by setting the keyword `specprod`.
By default, this function aggregates all the columns for the redshift catalogs, and adds columns to
quantify the number of coadded spectra listed in the catalogs for each TARGETID and to identify the
primary ("best") spectrum out of them using the `find_primary_spectra` function. Optionally, the
script can return a list of pre-selected "summary" column, or a list of user-specified
columns via the `columns_list` keyword.

Usage::

   create_summary_catalog(specprod, specgroup = 'zpix', all_columns = True, \
                          columns_list = None, output_filename = './zcat-all.fits')

Ragadeepika Pucha, Stephanie Juneau, and DESI data team
Version: 2022, March 31st
"""

####################################################################################################
####################################################################################################

import numpy as np
import os
from glob import glob
from astropy.io import fits
from astropy.table import Table, Column, vstack, join

## DESI related functions
from desispec.io import specprod_root
from desispec.io.util import get_tempfilename, write_bintable
from desiutil.log import get_logger
import desiutil.depend

####################################################################################################
####################################################################################################

log = get_logger()

def find_primary_spectra(table, sort_column = 'TSNR2_LRG'):
    """
    Function to select the best "primary" spectrum for objects with multiple (coadded) spectra.

    The best spectrum is selected based on the following steps:

    1. Spectra with ZWARN=0 are given top preference.
    2. Among multiple entries with ZWARN=0, spectra with the highest value of 'sort_column'
       are given the preference.
    3. If there are no spectra with ZWARN=0, spectra with the highest value of 'sort_column'
       are given the preference.

    Parameters
    ----------
    table : Numpy array or Astropy Table
        The input table should be a redshift catalog, with at least the following columns:
        'TARGETID', 'ZWARN' and the sort_column (Default:'TSNR2_LRG')
        Additional columns will be ignored.
    sort_column : str
        Column name that will be used to select the best spectrum. Default = 'TSNR2_LRG'
        The higher values of the column will be considered as better (descending order).

    Returns
    -------
    nspec : Numpy int array
        Array of number of entries available per target
    spec_primary : Numpy bool array
        Array of spec_primary (= TRUE for the best spectrum)
    """

    ## Convert into an astropy table
    table = Table(table)

    ## Main columns that are required:
    ## TARGETID, ZWARN, SORT_COLUMN that is given by the user (default=TSNR2_LRG)

    tsel = table['TARGETID', 'ZWARN', sort_column]

    ## Adding a row number column to sort the final table back to the same order as the input
    row = Column(np.arange(1, len(table)+1), name = 'ROW_NUMBER')
    tsel.add_column(row)

    ## Add the SPECPRIMARY and NSPEC columns - initialized to 0
    nspec = Column(np.array([0]*len(table)), name = 'NSPEC', dtype = '>i4')
    spec_prim = Column(np.array([0]*len(table)), name = 'SPECPRIMARY', dtype = 'bool')
    tsel.add_column(nspec)
    tsel.add_column(spec_prim)

    ## Create a ZWARN_NOT_ZERO column
    ## This helps to sort in the order such that ZWARN=0 is on top.
    ## The rest are then arranged based on the 'SORT_COLUMN'
    tsel['ZWARN_NOT_ZERO'] = (tsel['ZWARN'] != 0).astype(int)

    ## Create an inverse sort_column -- this is for sorting in the decreasing order of sort_column
    ## Higher values of sort_column are considered as better
    tsel['INV_SORT_COLUMN'] = 1/(tsel[sort_column] + 1e-99*(tsel[sort_column] == 0.0))
    ## The extra term in the denominator is added to avoid cases where the sort_column is 0, leading
    ## to unreal values when taking its inverse.

    ## Sort by TARGETID, ZWARN_NOT_ZERO, and inverse sort_column -- in this order
    tsel.sort(['TARGETID', 'ZWARN_NOT_ZERO', 'INV_SORT_COLUMN'])

    ## Selecting the unique targets, along with their indices and number of occurrences
    targets, indices, return_indices, num = np.unique(tsel['TARGETID'].data, return_index = True, \
                                                      return_inverse = True, return_counts = True)

    ## Since we sorted the table by TARGETID, ZWARN_NOT_ZERO and inverse sort_column
    ## The first occurence of each target is the PRIMARY (with ZWARN = 0 or with higher sort_column)
    ## This logic sets SPECPRIMARY = 1 for every first occurence of each target in this sorted table
    tsel['SPECPRIMARY'][indices] = 1

    # Set the NSPEC for every target
    tsel['NSPEC'] = num[return_indices].astype('>i2')

    # Note: SPECPRIMARY for negative TARGETIDs (stuck positioners on sky locations) is a bit
    # meaningless, but tile-based perexp and pernight catalogs can have repeats of those
    # and they are treated like other targets so that there is strictly one SPECPRIMARY
    # entry per TARGETID

    ## Sort by ROW_NUMBER to get the original order -
    tsel.sort('ROW_NUMBER')

    ## Final nspec and specprimary arrays -
    nspec = tsel['NSPEC'].data
    spec_primary = tsel['SPECPRIMARY'].data

    return (nspec, spec_primary)

####################################################################################################
####################################################################################################

def _get_survey_program_from_filename(filename):
    """
    Return SURVEY,PROGRAM parsed from zpix/ztile filename; fragile!
    """
    # zpix-SURVEY-PROGRAM.fits or ztile-SURVEY-PROGRAM-SPECGROUP.fits
    base = os.path.splitext(os.path.basename(filename))[0]
    arr = base.split('-')
    survey = arr[1]
    program = arr[2]
    return survey, program

def create_summary_catalog(specgroup, indir=None, specprod=None,
                           all_columns = True, columns_list = None,
                           output_filename=None):
    """
    This function combines all the individual redshift catalogs for either 'zpix' or 'ztile'
    with the desired columns (all columns, or a pre-selected list, or a user-given list).
    It further adds 'NSPEC' and 'PRIMARY' columns, two for SV(or MAIN),
    and two for the entire combined redshift catalog.

    Parameters
    ----------
    specgroup : str
        The option to run the code on ztile* files or zpix* files.
        It can either be 'zpix' or 'ztile'
    indir : str
        Input directory to look for zpix/ztile files.
    specprod : str
        Internal Release Name for the DESI Spectral Release.
        Used to derive input directory if indir is not provided.
    all_columns : bool
        Whether or not to include all the columns into the final table. Default is True.
    columns_list : list
        If all_columns = False, list of columns to include in the final table.
        If None, a list of pre-decided summary columns will be used. Default is None.
        The 'SV/MAIN' primary flag columns as well as the primary flag columns for the entire
        catalog witll be included.
    output_filename : str
        Path+Filename for the output summary redshift catalog.
        The output FITS file will be saved at this path.
        If not specified, the output filename will be derived from specgroup and $SPECPROD

    Returns
    -------
    None
        The function saves a FITS file at the location and file name specified by `output_filename`
        with the summary redshift catalog in HDU1.
    """

    ############################### Checking the inputs ##################################

    ## Initial check 1
    ## If specgroup = something else by mistake
    valid_specgroups = ('zpix', 'ztile')
    if specgroup not in valid_specgroups:
        errmsg = f'{specgroup=} not recognized, should be one of {valid_specgroups}'
        log.error(errmsg)
        raise ValueError(errmsg)

    ## set indir if needed
    if indir is None:
        indir = specprod_root(specprod) + '/zcatalog'
        log.info(f'Using input directory {indir}')

    ## Initial check 2
    ## Test whether the input directory exists
    if not os.path.isdir(indir):
        msg = f'"{indir}" directory does not exist'
        log.error(msg)
        raise ValueError(msg)

    ## Set output_filename if needed
    if output_filename is None:
        specprod = os.environ['SPECPROD']
        if specgroup == 'zpix':
            output_filename = f'zall-pix-{specprod}.fits'
        elif specgroup == 'ztile':
            output_filename = f'zall-tilecumulative-{specprod}.fits'
        else:
            # not yet used; future-proofing
            output_filename = f'zall-{specgroup}-{specprod}.fits'

        log.info(f'Will write output to {output_filename}')

    ######################################################################################

    ## Find all the filenames for a given specgroup
    if (specgroup == 'zpix'):
        ## List of all zpix* catalogs: zpix-survey-program.fits
        zcat = glob(f'{indir}/zpix-*.fits')
    elif (specgroup == 'ztile'):
        ## List of all ztile* catalogs, considering only cumulative catalogs
        zcat = glob(f'{indir}/ztile-*cumulative.fits')

    ## Sorting the list of zcatalogs by name
    ## This is to keep it neat, clean, and in order
    zcat.sort()

    ## Get all the zcatalogs for a given spectral release and specgroup
    ## Add the required columns or select a few of them
    ## Adding all the tables into a single list
    tables = []

    ## Handle DEPNAMnn DEPVERnn keyword merging separately
    dependencies = dict()

    ## Looping through the different zcatalogs and adding the survey and program columns
    for filename in zcat:
        basefile = os.path.basename(filename)

        ## Load the ZCATALOG table, along with the meta data
        log.info(f'Reading {filename}')
        t = Table.read(filename, hdu = 'ZCATALOG')

        ## Merge DEPNAMnn and DEPVERnn, then remove from header
        desiutil.depend.mergedep(t.meta, dependencies)
        desiutil.depend.remove_dependencies(t.meta)

        ## Remove other keys that we don't want to propagate
        for key in ('CHECKSUM', 'DATASUM'):
            if key in t.meta:
                del t.meta[key]

        ## Get SURVEY and PROGRAM from header, then remove from header
        ## because we are stacking catalogs from multiple surveys and programs
        if 'SURVEY' in t.meta:
            survey = t.meta['SURVEY']
            del t.meta['SURVEY']
        else:
            # parse filename if needed, but complain about it
            survey = _get_survey_program_from_filename(filename)[0]
            log.warning(f'{filename} header missing SURVEY; guessing {survey} from filename')

        if 'PROGRAM' in t.meta:
            program = t.meta['PROGRAM']
            del t.meta['PROGRAM']
        else:
            program = _get_survey_program_from_filename(filename)[1]
            log.warning(f'{filename} header missing PROGRAM; guessing {program} from filename')

        log.debug(f'{basefile} SURVEY={survey} PROGRAM={program}')
        ## We keep the rest of the meta data

        ## Add the SURVEY and PROGRAM columns
        ## SURVEY is added as a str7 and PROGRAM is added as str6 (to match other catalogs)
        ## 'special' consists of seven characters and is the maximum character for a SURVEY string
        ## 'backup' and 'bright' consists of six characters and is the maximum for a PROGRAM string
        col1 = Column(np.array([survey]*len(t)), name = 'SURVEY', dtype = '<U7')
        col2 = Column(np.array([program]*len(t)), name = 'PROGRAM', dtype = '<U6')
        t.add_column(col1, 1)
        t.add_column(col2, 2)
        ## The SURVEY and PROGRAM columns are added as second and third columns,
        ## immediately after TARGETID

        ## Appending the tables to the list
        tables.append(t)

    ## Stacking all the tables into a final table
    tab = vstack(tables)
    ## The output of this will have Masked Columns
    ## We will fix this at the end

    ## Selecting primary spectra for the whole combined ZCATALOG
    ## For SV, it selects the best spectrum including cmx+special+sv1+sv2+sv3
    ## For Main, it selects the best spectrum for main+special
    nspec, specprim = find_primary_spectra(tab)

    ## Replacing the existing 'ZCAT_NSPEC' and 'ZCAT_PRIMARY'
    ## If all_columns = False, and user-list does not contain this column -
    ## these columns will be added
    log.debug('Updating ZCAT_PRIMARY and ZCAT_NSPEC')
    tab['ZCAT_NSPEC'] = nspec
    tab['ZCAT_PRIMARY'] = specprim

    ############################### Adding SV/Main Primary Flags ##################################

    survey_col = tab['SURVEY'].astype(str)

    ## Check if SV1|SV2|SV3 targets are available and add SV Primary Flag columns
    if ('sv1' in survey_col)|('sv2' in survey_col)|('sv3' in survey_col):
        log.debug('Found SV inputs; adding SV_PRIMARY and SV_NSPEC columns')
        ## Add empty columns for SV NSPEC and PRIMARY
        col1 = Column(np.array([0]*len(tab)), name = 'SV_NSPEC', dtype = '>i2')
        col2 = Column(np.array([0]*len(tab)), name = 'SV_PRIMARY', dtype = 'bool')
        tab.add_columns([col1, col2])

        ## Selecting primary spectra for targets within just SV
        ## For SV, it selects the primary spectra out of SV1+SV2+SV3 for every individual TARGETID
        ## Ignores cmx+special in SV
        is_sv = np.char.startswith(survey_col.data, 'sv')
        nspec, specprim = find_primary_spectra(tab[is_sv])
        tab['SV_NSPEC'][is_sv] = nspec
        tab['SV_PRIMARY'][is_sv] = specprim

    ## Check if 'main' targets are available and add Main Primary Flag Columns
    if ('main' in survey_col):
        log.debug('Found main survey inputs; adding MAIN_PRIMARY and MAIN_NSPEC columns')
        ## Add empty columns for Main NSPEC and PRIMARY
        col1 = Column(np.array([0]*len(tab)), name = 'MAIN_NSPEC', dtype = '>i2')
        col2 = Column(np.array([0]*len(tab)), name = 'MAIN_PRIMARY', dtype = 'bool')
        tab.add_columns([col1, col2])

        ## Selecting primary spectra for targets within just MAIN
        ## It selects the primary spectra just for 'main' and ignores 'special'
        is_main = (survey_col.data == 'main')
        nspec, specprim = find_primary_spectra(tab[is_main])
        tab['MAIN_NSPEC'][is_main] = nspec
        tab['MAIN_PRIMARY'][is_main] = specprim

    ###############################################################################################

    ## For convenience, sort by SURVEY, PROGRAM, and (HEALPIX or TILEID)
    if (specgroup == 'zpix'):
        tab.sort(['SURVEY', 'PROGRAM', 'HEALPIX', 'TARGETID'])
    elif (specgroup == 'ztile'):
        tab.sort(['SURVEY', 'PROGRAM', 'TILEID', 'LASTNIGHT', 'FIBER'])

    ## Convert the masked column table to normal astropy table and select required columns
    final_table = update_table_columns(table = tab, specgroup = specgroup, \
                                       all_columns = all_columns, columns_list = columns_list)

    ## Add merged DEPNAMnn / DEPVERnn dependencies back into final table
    desiutil.depend.mergedep(dependencies, final_table.meta)

    ## Write final output via a temporary filename
    tmpfile = get_tempfilename(output_filename)
    write_bintable(tmpfile, final_table, extname='ZCATALOG', clobber=True)
    os.rename(tmpfile, output_filename)
    log.info(f'Wrote {output_filename}')

####################################################################################################
####################################################################################################

def update_table_columns(table, specgroup = 'zpix', all_columns = True, columns_list = None):
    """
    This function fills the ``*TARGET`` masked columns and returns the final table
    with the required columns.

    Parameters
    ----------
    table : Astropy Table
        A table.
    specgroup : str
        The option to run the code on ztile* files or zpix* files.
        It can either be 'zpix' or 'ztile'. Default is 'zpix'
    all_columns : bool
        Whether or not to include all the columns into the final table. Default is True.
    columns_list : list
        If all_columns = False, list of columns to include in the final table.
        If None, a list of pre-decided summary columns will be used. Default is None.
        The 'SV/MAIN' primary flag columns as well as the primary flag columns for the entire
        catalog witll be included.

    Returns
    -------
    t_final : Astropy Table
        Final table with non-masked columns with required columns.
    """

    ## Due to stacking tables with different columns,
    ## We have *TARGET columns that are Masked. We need to fill the empty columns with zero.
    ## This is important - otherwise Astropy fills the table with different values.

    ## Array of all columns:
    tab_cols = np.array(table.colnames)

    ## Pick out columns ending with '_TARGET'
    sel = np.char.endswith(tab_cols, '_TARGET')

    ## This include FA_TARGET -- which needs to be removed
    ## Make a list of all the '*_TARGET' columns, which contain DESI targetting information
    ## This is both for correcting the masked columns, as well as
    ## for rearranging the columns into a proper order
    target_cols = list(tab_cols[sel])
    target_cols.remove('FA_TARGET')

    for col in target_cols:
        ## Fill the *TARGET columns that are masked with 0
        table[col].fill_value = 0

    ## Table with filled values
    tab = table.filled()

    ## Selecting the required columns for the final table
    ## If all_columns is True, then rearraning the columns into a proper order
    ## If all_columns is False, then only a subset of columns is selected
    ## If no input user-list of columns is given, a pre-selected list of columns is used
    ## to create a summary redshift catalog

    ## Find all the existing NSPEC and PRIMARY flag columns and order them
    nspec_cols = list(tab_cols[np.char.endswith(tab_cols, '_NSPEC')])
    prim_cols = list(tab_cols[np.char.endswith(tab_cols, '_PRIMARY')])
    nspec_cols.sort()
    prim_cols.sort()

    ## Ordering the primary columns
    ## If SV/MAIN flag columns also exist, the order is -
    ## MAIN_NSPEC, MAIN_PRIMARY, SV_NSPEC, SV_PRIMARY, ZCAT_NSPEC, ZCAT_PRIMARY
    ## This is to add these columns separately in the end
    primary_cols = []
    for xx in range(len(nspec_cols)):
        primary_cols.append(nspec_cols[xx])
        primary_cols.append(prim_cols[xx])

    if all_columns:
        ## Rearranging the columns to order all the *TARGET columns together
        ## TARGET columns sit between NUMOBS_INIT and PLATE_RA columns
        ## Last column in TSNR2_LRG in all the redshift catalogs
        ## We will add the PRIMARY columns in the end

        ## The indices of NUMOBS_INIT, PLATE_RA, and last TSNR2_* columns
        nobs = np.where(np.array(tab.colnames) == 'NUMOBS_INIT')[0][0]
        pra = np.where(np.array(tab.colnames) == 'PLATE_RA')[0][0]
        tsnr = np.where(np.char.startswith(np.array(tab.colnames), 'TSNR2_'))[0][-1]

        ## List of all columns
        all_cols = tab.colnames

        ## Reorder the columns
        ## This reorder is important for stacking the different redshift catalogs
        ## Also to keep it neat and clean
        req_columns = all_cols[0:nobs+1] + target_cols + all_cols[pra:tsnr+1]+primary_cols
    else:
        if (columns_list == None):
            ## These are the pre-selected list of columns
            pre_selected_cols = ['TARGETID', 'SURVEY', 'PROGRAM',
                               'TARGET_RA', 'TARGET_DEC', 'Z', 'ZERR', 'ZWARN',
                               'COADD_FIBERSTATUS',  'CHI2', 'DELTACHI2',
                               'MASKBITS', 'SPECTYPE', 'FLUX_G', 'FLUX_R',
                               'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 'FLUX_IVAR_G',
                               'FLUX_IVAR_R', 'FLUX_IVAR_Z','FLUX_IVAR_W1',
                               'FLUX_IVAR_W2', 'TSNR2_LRG', 'TSNR2_BGS', 'TSNR2_ELG',
                               'TSNR2_QSO', 'TSNR2_LYA'] + target_cols + primary_cols


            ## Add HEALPIX for zpix* files, and TILEID, LASTNIGHT for ztile* files
            if (specgroup == 'zpix'):
                req_columns = pre_selected_cols[0:3]+['HEALPIX']+pre_selected_cols[3:]
            else:
                req_columns = pre_selected_cols[0:3]+['TILEID', 'LASTNIGHT']+pre_selected_cols[3:]

        else:
            ## Adding the primary flag columns to the user-requested list
            req_columns = columns_list + primary_cols

    ## Final table with the required columns
    t_final = tab[req_columns]

    return (t_final)

####################################################################################################
####################################################################################################

