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

   from desispec.zcatalog import create_summary_catalog

   create_summary_catalog(specgroup = 'zpix', indir='/global/cfs/cdirs/desi/public/dr2/spectro/redux/loa/zcatalog/v2')
   create_summary_catalog(specgroup = 'ztile', indir='/global/cfs/cdirs/desi/public/dr2/spectro/redux/loa/zcatalog/v2')

Ragadeepika Pucha, Stephanie Juneau, and DESI data team
Version: 2022, March 31st
Updated: Summer 2025
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

    if 'ZWARN' in table.colnames:
        zwarn_col = 'ZWARN'
    else:
        zwarn_col = 'ZWARN_BEST'
    tsel = table['TARGETID', zwarn_col, sort_column]

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
    tsel['ZWARN_NOT_ZERO'] = (tsel[zwarn_col] != 0).astype(int)

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

def create_summary_catalog(specgroup, indir=None, specprod=None):
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

    Returns
    -------
    None
        The function saves FITS files in the input directory.
    """

    ############################### Checking the inputs ##################################

    ## Initial check 1
    ## If specgroup = something else by mistake
    valid_specgroups = ('zpix', 'ztile')
    if specgroup not in valid_specgroups:
        errmsg = f'{specgroup=} not recognized, should be one of {valid_specgroups}'
        log.error(errmsg)
        raise ValueError(errmsg)

    if specprod is None:
        specprod = os.environ['SPECPROD']

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

    if not os.path.isdir(f'{indir}/zall'):
        os.makedirs(f'{indir}/zall')

    ######################################################################################

    ## Find all the filenames for a given specgroup
    if (specgroup == 'zpix'):
        ## List of all zpix* catalogs: zpix-survey-program.fits
        zcat = glob(f'{indir}/*/zpix-*.fits')
    elif (specgroup == 'ztile'):
        ## List of all ztile* catalogs, considering only cumulative catalogs
        zcat = glob(f'{indir}/*/ztile-*cumulative.fits')

    # only keep the primary filenames
    for fn in zcat.copy():
        if ('-imaging.fits' in fn) or ('-extra.fits' in fn):
            print(fn)
            zcat.remove(fn)

    ## Sorting the list of zcatalogs by name
    ## This is to keep it neat, clean, and in order
    zcat.sort()

    for file_extension in ['ZCATALOG', 'ZCATALOG_IMAGING', 'ZCATALOG_EXTRA']:
        fn_suffix = file_extension.replace('ZCATALOG', '').replace('_', '-').lower()
        ## Set output_filename
        if specgroup == 'zpix':
            output_filename = f'{indir}/zall/zall-pix-{specprod}{fn_suffix}.fits'
        elif specgroup == 'ztile':
            output_filename = f'{indir}/zall/zall-tilecumulative-{specprod}{fn_suffix}.fits'
        else:
            # not yet used; future-proofing
            output_filename = f'{indir}/zall/zall-{specgroup}-{specprod}{fn_suffix}.fits'
            log.info(f'Will write output to {output_filename}')

        ## Get all the zcatalogs for a given spectral release and specgroup
        ## Add the required columns or select a few of them
        ## Adding all the tables into a single list
        tables = []

        ## Handle DEPNAMnn DEPVERnn keyword merging separately
        dependencies = dict()

        if file_extension=='ZCATALOG':
            zcat1 = zcat.copy()
        elif file_extension=='ZCATALOG_IMAGING':
            zcat1 = [fn.replace('.fits', '-imaging.fits') for fn in zcat]
        elif file_extension=='ZCATALOG_EXTRA':
            zcat1 = [fn.replace('.fits', '-extra.fits') for fn in zcat]

        ## Looping through the different zcatalogs and adding the survey and program columns
        for filename in zcat1:
            basefile = os.path.basename(filename)

            ## Load the ZCATALOG table, along with the meta data
            log.info(f'Reading {filename}')
            t = Table.read(filename, hdu = file_extension)

            ## Merge DEPNAMnn and DEPVERnn, then remove from header
            desiutil.depend.mergedep(t.meta, dependencies)
            desiutil.depend.remove_dependencies(t.meta)

            ## Remove other keys that we don't want to propagate
            for key in ('CHECKSUM', 'DATASUM'):
                if key in t.meta:
                    del t.meta[key]

            if file_extension=='ZCATALOG':

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

        if file_extension=='ZCATALOG':

            ## Selecting primary spectra for the whole combined ZCATALOG
            ## For SV, it selects the best spectrum including cmx+special+sv1+sv2+sv3
            ## For Main, it selects the best spectrum for main+special
            nspec, specprim = find_primary_spectra(tab, sort_column='EFFTIME_SPEC')

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
                nspec, specprim = find_primary_spectra(tab[is_sv], sort_column='EFFTIME_SPEC')
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
                nspec, specprim = find_primary_spectra(tab[is_main], sort_column='EFFTIME_SPEC')
                tab['MAIN_NSPEC'][is_main] = nspec
                tab['MAIN_PRIMARY'][is_main] = specprim

            ###############################################################################################

            # Sanity check that the TARGETIDs match in the row-matched catalogs
            if file_extension=='ZCATALOG':
                targetid_arr = np.array(tab['TARGETID']).copy()
            else:
                assert np.all(tab['TARGETID']==targetid_arr)

        ## Convert the masked column table to normal astropy table and select required columns
        final_table = update_table_columns(tab, specgroup)

        ## Add merged DEPNAMnn / DEPVERnn dependencies back into final table
        desiutil.depend.mergedep(dependencies, final_table.meta)

        ## Write final output via a temporary filename
        tmpfile = get_tempfilename(output_filename)
        write_bintable(tmpfile, final_table, extname=file_extension, clobber=True)
        os.rename(tmpfile, output_filename)
        log.info(f'Wrote {output_filename}')

####################################################################################################
####################################################################################################

def update_table_columns(table, specgroup):
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

    ## Make a list of all the targeting bit columns, which contain DESI targetting information
    ## This is both for correcting the masked columns, as well as
    ## for rearranging the columns into a proper order
    sel = np.char.endswith(tab_cols, '_TARGET') & (tab_cols!='FA_TARGET')
    target_cols = list(tab_cols[sel])

    for col in target_cols:
        ## Fill the *TARGET columns that are masked with 0
        table[col].fill_value = 0

    ## Table with filled values
    table = table.filled()

    # Move the target columns to the end
    reordered_cols = list(np.array(table.colnames)[~np.in1d(table.colnames, target_cols)]) + target_cols
    table = table[reordered_cols]

    return table

####################################################################################################
####################################################################################################

