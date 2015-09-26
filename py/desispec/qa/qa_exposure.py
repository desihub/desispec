"""
Class to organize and execute QA for a DESI exposure
"""

from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np

class QA_Exposure(object):
    def __init__(self, exptype='none', expid=-1, in_data=None):
        """
        Class to organize and execute QA for a DESI exposure

        x.expid, x.exptype
        
        Args:
            exptype: str, optional exposure type (e.g. flat, arc, science)
            expid: int, optional exposure number
            in_data: dict, optional -- Input data 
              Mainly for reading from disk

        Notes:

        Attributes:
            All input args become object attributes.
        """
        assert exptype in ['none', 'flat', 'arc', 'science']

        self.expid = expid
        self.exptype = exptype
        
        if in_data is None:
            self._data = dict(expid=self.expid, exptype=self.exptype)
            self.init_data()
        else:
            assert isinstance(in_data,dict)
            self._data = in_data

    def init_data(self):
        """Initialize QA data"""
        # Generate one QA dict per camera (aka Frame)
        for band in ['b','r','z']:
            for spectro in range(10):
                self._data[band+str(spectro)] = {}
         
    def init_skysub(self, camera, re_init=False):
        """Initialize for SkySub QA for a given camera

        Parameters:
        ------------
        re_init: bool, (optional)
          Re-initialize SKYSUB dict
        """
        # 
        assert self.exptype in ['science']

        # Check for previous initialization
        if ('SKYSUB' in self._data[camera].keys()) & (not re_init):
            print('SKYSUB already initialized in QA_Exposure.')
            return
        # Standard SKYSUB parameters
        sky_dict = dict(
            PCHI_RESID=0.05, # P(Chi^2) limit for bad skyfiber model residuals
            NBAD_PCHI=0,     # Number of bad sky fibers for chi^2 test on residuals
            NSKY_FIB=0,      # Number of sky fibers 
            MED_RESID=0.,    # Residual flux in sky fiber after subtraction
            )
        # Generate dict
        self._data[camera]['SKYSUB'] = sky_dict

    def __repr__(self):
        """
        Print formatting
        """
        return ('{:s}: Exposure={:08d}, ExpType={:s}'.format(
                self.__class__.__name__,self.expid, self.exptype))
