"""
Classes to organize and execute QA for a DESI exposure
"""

from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np

from desispec.sky import qa_skysub

class QA_Frame(object):
    def __init__(self, flavor='none', camera='none', in_data=None):
        """
        Class to organize and execute QA for a DESI frame

        x.flavor, x.data, x.camera
        
        Args:
            flavor: str, optional exposure type (e.g. flat, arc, science)
            camera: str, optional camera (e.g. 'b0')
            in_data: dict, optional -- Input data 
              Mainly for reading from disk

        Notes:

        Attributes:
            All input args become object attributes.
        """
        assert flavor in ['none', 'flat', 'arc', 'science']

        self.flavor = flavor
        self.camera = camera
        
        # Initialize data
        if in_data is None:
            self.data = dict(flavor=self.flavor, camera=self.camera)
        else:
            assert isinstance(in_data,dict)
            self.data = in_data

    def init_qatype(self, qatype, param, re_init=False):
        """Initialize parameters for a given qatype
        qatype: str  
          Type of QA to be performed (e.g. SKYSUB)
        param: dict
          Dict of parameters to guide QA
        re_init: bool, (optional)
          Re-initialize parameter dict
          Code will always add new parameters if any exist
        """
        # Fill and return if not set previously or if re_init=True
        if (qatype not in self.data.keys()) or re_init: 
            self.data[qatype] = {}
            self.data[qatype]['PARAM'] = param
            return

        # Update the new parameters only
        for key in param.keys():
            if key not in self.data[qatype]['PARAM'].keys():
                self.data[qatype]['PARAM'][key] = param[key]

    def init_skysub(self, re_init=False):
        """Initialize parameters for SkySub QA 
        QA method is desispec.sky.qa_skysub

        Parameters:
        ------------
        re_init: bool, (optional)
          Re-initialize SKYSUB parameter dict
        """
        # 
        assert self.flavor in ['science']

        # Standard SKYSUB input parameters
        sky_dict = dict(
            PCHI_RESID=0.05, # P(Chi^2) limit for bad skyfiber model residuals
            )
        # Init
        self.init_qatype('SKYSUB', sky_dict, re_init=re_init)

    def run_qa(self, qatype, inputs, clobber=True):
        """Run QA tests of a given type
        Over-writes previous QA of this type, unless otherwise specified

        qatype: str  
          Type of QA to be performed (e.g. SKYSUB)
        inputs: tuple
          Set of inputs for the tests
        clobber: bool, optional [True]
          Over-write previous QA 
        """
        # Check for previous QA if clobber==False
        if not clobber:
            # QA previously performed?
            if 'QA' in self.data[qatype].keys():
                return
        # Run
        if qatype == 'SKYSUB':
            # Expecting: frame, fibermap, skymodel
            assert len(inputs) == 3
            # Init parameters (as necessary)
            self.init_skysub()
            # Run
            qadict = qa_skysub(self.data[qatype]['PARAM'],
                inputs[0], inputs[1], inputs[2])
        else:
            raise ValueError('Not ready to perform {:s} QA'.format(qatype))
        # Update
        self.data[qatype]['QA'] = qadict


    def __repr__(self):
        """
        Print formatting
        """
        return ('{:s}: flavor={:s}'.format(
                self.__class__.__name__, self.flavor))

class QA_Exposure(object):
    def __init__(self, flavor='none', in_data=None):
        """
        Class to organize and execute QA for a DESI Exposure

        x.flavor, x.data
        
        Args:
            flavor: str, optional exposure type (e.g. flat, arc, science)
            in_data: dict, optional -- Input data 
              Mainly for reading from disk

        Notes:

        Attributes:
            All input args become object attributes.
        """
        assert flavor in ['none', 'flat', 'arc', 'science']

        self.flavor = flavor
        
        if in_data is None:
            self.data = dict(flavor=self.flavor)
            self.init_data()
        else:
            assert isinstance(in_data,dict)
            self.data = in_data
