"""
Test desispec.zcatalog
"""

import os
import unittest

import numpy as np
from astropy.table import Table

from desispec.zcatalog import find_primary_spectra

class TestZCatalog(unittest.TestCase):
    
    def test_find_primary_spectra(self):
        #- TARGETID ZWARN TSNR2_LRG TEST
        rows = [
           (10, 0, 100.0, 0),
           (10, 0, 200.0, 1),  # larger TSNR2_LRG = better
           (20, 4,   0.0, 1),  # only entry for this target
           (30, 4, 100.0, 0),
           (30, 0,  10.0, 1),  # zwarn=0 trumps larger TSNR2
           (40, 4, 100.0, 1),  # zwarn value doesn't matter except 0 or non-0
           (40, 8,  10.0, 0),
           (50, 8, 100.0, 1),  # zwarn value doesn't matter except 0 or non-0
           (50, 4,  10.0, 0),
           (60, 0,  10.0, 1),  # TSNR2=0 doesn't break things
           (60, 0,   0.0, 0),
           (-1, 0,  10.0, 1),  # negative TARGETIDs are ok
           (-1, 0,   0.0, 0),
        ]

        zcat = Table(rows=rows, names=('TARGETID','ZWARN','TSNR2_LRG','TEST'))
        n, best = find_primary_spectra(zcat)
        self.assertTrue( np.all(zcat['TEST'] == best) )
        self.assertTrue(isinstance(n, np.ndarray))
        self.assertTrue(isinstance(best, np.ndarray))

        # also works for numpy array input
        n, best = find_primary_spectra(np.array(zcat))
        self.assertTrue( np.all(zcat['TEST'] == best) )

        # custom column name
        zcat.rename_column('TSNR2_LRG', 'BLAT')
        n, best = find_primary_spectra(zcat, sort_column='BLAT')
        self.assertTrue( np.all(zcat['TEST'] == best) )

        # custom column name, even if TSNR2_LRG is present don't use it
        zcat['TSNR2_LRG'] = np.zeros(len(zcat))
        n, best = find_primary_spectra(zcat, sort_column='BLAT')
        self.assertTrue( np.all(zcat['TEST'] == best) )
