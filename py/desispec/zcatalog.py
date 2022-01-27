## This script is based on this discussion - https://github.com/desihub/desispec/issues/1355
## Version: January 21, 2022

####################################################################################################
####################################################################################################

import numpy as np
from astropy.table import Table, Column, join

####################################################################################################
####################################################################################################

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

