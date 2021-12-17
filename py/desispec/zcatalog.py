## This script is based on this discussion - https://github.com/desihub/desispec/issues/1355

###########################################################################################################################################################
###########################################################################################################################################################

import numpy as np
from astropy.table import Table, Column, join

###########################################################################################################################################################
###########################################################################################################################################################

def find_primary_spectra(table):
    """
    Function to select the best spectrum for sources with multiple coadded-spectra.
    
    Parameters
    ----------
    
    table : 
    
        Input table
        
    Returns
    -------
    nspec : Numpy int array
        Array of number of spectra available per source
        
    spec_primary : Numpy bool array
        Array of spec_primary (= TRUE for the best spectrum)
    
    """
    
    
    ## Convert into an astropy table 
    table = Table(table)
    
    ## Main columns that are required 
    ## TARGETID, Z, ZWARN, TSNR2_LRG
    ## Optional -- SURVEY, 'FAPRGRM', HPXPIXEL
    
    tsel = table['TARGETID', 'Z', 'ZWARN', 'TSNR2_LRG']
    
    ## Adding a row number column to sort the final table back to the same order as the input
    row = Column(np.arange(1, len(table)+1), name = 'ROW_NUMBER')
    tsel.add_column(row)
    
    ## Add the SPECPRIMARY and NSPEC columns - initialized to 0
    nspec = Column(np.array([0]*len(table)), name = 'NSPEC', dtype = '>i4')
    spec_prim = Column(np.array([0]*len(table)), name = 'SPECPRIMARY', dtype = 'bool')
    tsel.add_column(nspec)
    tsel.add_column(spec_prim)
    
    ## Create an inverse TSNR column -- this is for sorting in the decreasing order of TSNR
    tsel['INV_TSNR2_LRG'] = 1/tsel['TSNR2_LRG']
    
    ## Sort by TARGETID, ZWARN, and inverse TSNR -- in this order
    tsel.sort(['TARGETID', 'ZWARN', 'INV_TSNR2_LRG'])
    
    ## Selecting the unique targets, along with their indices and number of occurrences
    targets, indices, return_indices, num = np.unique(tsel['TARGETID'].data, return_index = True, return_inverse = True, return_counts = True)
    
    ## Since we sorted the table by TARGETID, ZWARN and inverse TSNR
    ## The first occurence of each target is the PRIMARY (either with ZWARN = 0 or with higher TSNR2_LRG)
    ## Using this logic to set the SPECPRIMARY = 1 for every first occurence of each target in this sorted table
    tsel['SPECPRIMARY'][indices] = 1

    # Set the NSPEC for every target 
    tsel['NSPEC'] = num[return_indices]
    
    # Some of the TARGETIDs are negative, these are either faulty fibers or sky fibers
    # However, even with matching TARGETIDs, these have different positions
    # By checking the difference between RA and DEC, we found that these differences are greater than the fiber size.
    neg_targets = (tsel['TARGETID'].data < 0)

    # Setting SPECPRIMARY as 1 and NSPEC as 1 for all the negative TARGETIDs
    tsel['SPECPRIMARY'][neg_targets] = 1
    tsel['NSPEC'][neg_targets] = 1
    
    ## Sort by ROW_NUMBER to get the original order -
    tsel.sort('ROW_NUMBER')
    
    ## Final nspec and specprimary arrays - 
    nspec = tsel['NSPEC']
    spec_primary = tsel['SPECPRIMARY']
    
    return (nspec, spec_primary)

###########################################################################################################################################################
###########################################################################################################################################################
    