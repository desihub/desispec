"""
tests desispec.sky
"""

import unittest

import numpy as np
from desispec.frame import Frame
from desispec.specscore import compute_and_append_frame_scores
import desispec.io


class TestScores(unittest.TestCase):
    
    def _get_frame(self):        
        nspec=4
        wave=np.linspace(3600,6000,(6000-3600))
        Rdata = np.ones( (nspec, 1, wave.size) )
        flux = np.ones((nspec, wave.size))
        ivar = np.ones((nspec, wave.size))
        mask = np.zeros((nspec,wave.size), dtype=int)
        fibermap = desispec.io.empty_fibermap(nspec)
        return Frame(wave, flux, ivar, mask, Rdata, spectrograph=0, fibermap=fibermap)
    
                    
    def test_scores(self):        
        #- 
        frame = self._get_frame()
        scores,comments=compute_and_append_frame_scores(frame,suffix="RAW",flux_per_angstrom=False)        
        scores,comments=compute_and_append_frame_scores(frame,suffix="RAW",flux_per_angstrom=False)
        scores,comments=compute_and_append_frame_scores(frame,suffix="CALIB",flux_per_angstrom=True)
        print(scores)
        print(comments)

    def test_main(self):
        pass
        
        
    def runTest(self):
        pass

    
if __name__ == '__main__':
    unittest.main()
