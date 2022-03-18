"""
The zcatalog.py script contains the following utility functions for redshift catalogs:

(1) find_primary_spectra: 

Given an input table with possibly multiple spectra per TARGETID, this function returns arrays for
a primary flag (whether a spectrum is the primary ["best"] spectrum based on the ZWARN value and 
on a sort column [default='TSNR2_LRG']) and for the number of spectra per TARGETID.

Usage:
------
   nspec, spec_primary = find_primary_spectra(table, sort_column = 'TSNR2_LRG')

(2) create_summary_catalog:

This function combines individual redshift catalogs from a given release into a single compilation 
catalog. The output catalog is saved in a FITS file in a user-specified location and filename. 

This function can be used for 'fuji' (SV) or 'guadalupe' (Main) by setting the keyword `specprod`.
By default, this function aggregates all the columns for the redshift catalogs, and adds columns to 
quantify the number of coadded spectra listed in the catalogs for each TARGETID and to identify the 
primary ("best") spectrum out of them using the `find_primary_spectra` function. Optionally, the 
script can return a list of pre-selected "summary" column, or a list of user-specified 
columns via the `columns_list` keyword. 

Usage:
------
   create_summary_catalog(specprod, survey_name, specgroup = 'zpix', all_columns = True, \
                          columns_list = None, output_filename = './zcat-all.fits')

Ragadeepika Pucha, Stephanie Juneau, and DESI data team 
Version: 2022, March 17th
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
from desiutil.log import get_logger

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
    tsel['NSPEC'] = num[return_indices]

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

def create_summary_catalog(specprod, survey_name, specgroup = 'zpix', \
                           all_columns = True, columns_list = None, output_filename = './zcat-all.fits'):
    """
    This function combines all the individual redshift catalogs for either 'zpix' or 'ztile' 
    with the desired columns (all columns, or a pre-selected list, or a user-given list).
    It further adds 'NSPEC' and 'PRIMARY' columns, two for SV(or MAIN),
    and two for the entire combined redshift catalog.
    
    Parameters
    ----------
    
    specprod : str
        Internal Release Name for the DESI Spectral Release
        This is required to create the directory path for the redshift catalogs.
        (fuji|guadalupe|other names in the future)
        
    survey_name : str
        'sv' or 'main'. This is required for adding survey-based primary flags 
        and also for defining the list of '*_TARGET' columns.
    
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
        
    output_filename : str
        Path+Filename for the output summary redshift catalog.
        The output *fits file will be saved at this path. 
        If not specified, the output will be saved locally as ./zcat-all.fits

    Returns
    -------
    
    None. The function saves a FITS file at the location and file name specified by `output_filename` 
          with the summary redshift catalog in HDU1.

    """
    
    ############################### Checking the inputs ##################################
    
    ## Initial check 1
    ## Test whether the specprod exists or not
    ## Spectral Directory Path for a given internal release name
    specred_dir = specprod_root(specprod)    
    if (os.path.isdir(specred_dir) == False):
        log.error(f'"{specprod}" directory does not exist')
        raise OSError(f'"{specprod}" directory does not exist')
        
    ## Initial check 2
    ## If specgroup = something else by mistake  
    if (specgroup != 'zpix')&(specgroup != 'ztile'):
        log.error(f'"{specgroup}" not recognized')
        raise NameError(f'"{specgroup}" not recognized')
        
    ## Initial check 3
    ## Check that survey_name is either sv or main
    if (survey_name != 'sv')&(survey_name != 'main'):
        log.error(f'"{survey_name}" not recognized')
        raise NameError(f'"{survey_name}" not recognized')
        
    ######################################################################################
    
    ## Directory path to all the redshift catalogs
    zcat_dir = f'{specred_dir}/zcatalog'
    
    ## Find all the filenames for a given specgroup
    if (specgroup == 'zpix'):
        ## List of all zpix* catalogs: zpix-survey-program.fits
        zcat = glob(f'{zcat_dir}/zpix*')
    elif (specgroup == 'ztile'):
        ## List of all ztile* catalogs, considering only cumulative catalogs
        zcat = glob(f'{zcat_dir}/ztile*cumulative.fits')
                    
    ## Sorting the list of zcatalogs by name
    ## This is to keep it neat, clean, and in order
    zcat.sort()
    
    ## Get all the zcatalogs for a given spectral release and specgroup
    ## Add the required columns or select a few of them
    ## Adding all the tables into a single list
    tables = []
    
    ## Looping through the different zcatalogs and adding the survey and program columns
    for filename in zcat:
        arr = filename.split('-')
        ## Filename can be used to get the survey and program of the given redshift catalog
        survey = arr[1]
        program = arr[2].split('.')[0]
        
        ## Load the ZCATALOG table, along with the meta data
        t = Table.read(filename, hdu = 'ZCATALOG')

        ## Removing the SURVEY and PROGRAM metadata
        ## This is because we are stacking catalogs from multiple surveys and programs
        del t.meta['SURVEY']
        del t.meta['PROGRAM']
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
    tab['ZCAT_NSPEC'] = nspec
    tab['ZCAT_PRIMARY'] = specprim
    
    ## Adding empty columns for SV|MAIN NSPEC and PRIMARY
    col1 = Column(np.array([0]*len(tab)), name = f'{survey_name.upper()}_NSPEC', dtype = '>i8')
    col2 = Column(np.array([0]*len(tab)), name = f'{survey_name.upper()}_PRIMARY', dtype = 'bool')
    tab.add_columns([col1, col2])
    
    ## Selecting primary spectra for targets within just SV or MAIN depending on the survey_name
    ## For SV, it selects the primary spectra out of SV1+SV2+SV3 for every individual TARGETID
    ## Ignores cmx+special in SV
    ## For Main, it selects the primary spectra just for 'main' and ignores 'special'
    is_survey = np.char.startswith(tab['SURVEY'].astype(str).data, survey_name)
    nspec, specprim = find_primary_spectra(tab[is_survey])
    tab[f'{survey_name.upper()}_NSPEC'][is_survey] = nspec
    tab[f'{survey_name.upper()}_PRIMARY'][is_survey] = specprim
    
    ## For convenience, sort by SURVEY, PROGRAM, and (HEALPIX or TILEID) 
    if (specgroup == 'zpix'):
        tab.sort(['SURVEY', 'PROGRAM', 'HEALPIX'])
    elif (specgroup == 'ztile'):
        tab.sort(['SURVEY', 'PROGRAM', 'TILEID', 'LASTNIGHT'])
            
        
    ## Convert the masked column table to normal astropy table and select required columns
    final_table = update_table_columns(table = tab, survey_name = survey_name, specgroup = specgroup, \
                                       all_columns = all_columns, columns_list = columns_list)
           
    final_table.meta['EXTNAME'] = 'ZCATALOG'
    final_table.write(output_filename, overwrite = True)
    
####################################################################################################
####################################################################################################

def update_table_columns(table, survey_name, specgroup = 'zpix', all_columns = True, columns_list = None):
    """
    This function fills the '*TARGET' masked columns and returns the final table 
    with the required columns. 
    
    Parameters
    ----------
    
    table : Astropy Table
    
    survey_name : str
        'sv' or 'main'. This is for defining the list of '*_TARGET' columns.
        
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
    if (survey_name == 'sv'):
        ## Making a list of all the '*_TARGET* columns 
        ## This is for rearranging the columns into proper order
        target_cols = ['CMX_TARGET', 'DESI_TARGET','BGS_TARGET','MWS_TARGET', 'SCND_TARGET',
                       'SV1_DESI_TARGET', 'SV1_BGS_TARGET', 'SV1_MWS_TARGET', 'SV1_SCND_TARGET', 
                       'SV2_DESI_TARGET', 'SV2_BGS_TARGET', 'SV2_MWS_TARGET', 'SV2_SCND_TARGET', 
                       'SV3_DESI_TARGET', 'SV3_BGS_TARGET', 'SV3_MWS_TARGET', 'SV3_SCND_TARGET']
    else:
        ## Making a list of all the '*_TARGET* columns 
        ## This is for rearranging the columns into proper order
        target_cols = ['DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET', 'SCND_TARGET']
        
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
    
    ## Keeping the primary flag columns separate to add in the end
    primary_cols = [f'{survey_name.upper()}_NSPEC', f'{survey_name.upper()}_PRIMARY', 
                                 'ZCAT_NSPEC', 'ZCAT_PRIMARY']
    
    if all_columns:
        ## Rearranging the columns to order all the *TARGET columns together
        ## TARGET columns sit between NUMOBS_INIT and PLATE_RA columns
        ## Last column in TSNR2_LRG in all the redshift catalogs
        ## We will add the PRIMARY columns in the end

        ## The indices of NUMOBS_INIT, PLATE_RA, and ZCAT_PRIMARY columns
        nobs = np.where(np.array(tab.colnames) == 'NUMOBS_INIT')[0][0]
        pra = np.where(np.array(tab.colnames) == 'PLATE_RA')[0][0]
        tsnr = np.where(np.array(tab.colnames) == 'TSNR2_LRG')[0][0]

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

