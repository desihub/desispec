"""
Classes to organize and execute QA for a DESI exposure
"""

from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np

from desispec.sky import qa_skysub
from desispec.fiberflat import qa_fiberflat
from desispec.fluxcalibration import qa_fluxcalib

class QA_Frame(object):
    def __init__(self, frame=None, flavor='none', camera='none', in_data=None):
        """
        Class to organize and execute QA for a DESI frame

        x.flavor, x.data, x.camera
        
        Args:
            frame: Frame object, optional (should contain meta data)
            flavor: str, optional exposure type (e.g. flat, arc, science)
              Will use value in frame.meta, if present
            camera: str, optional camera (e.g. 'b0')
              Will use value in frame.meta, if present
            in_data: dict, optional -- Input data 
              Mainly for reading from disk

        Notes:

        Attributes:
            All input args become object attributes.
        """
        # Parse from frame.meta
        if frame is not None:
            # Parse from meta, if possible
            try:
                flavor = frame.meta['FLAVOR']
            except:
                pass
            else:
                camera = frame.meta['CAMERA']

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

    def init_fiberflat(self, re_init=False):
        """Initialize parameters for FIBERFLAT QA
        QA method is desispec.fiberflat.qa_fiberflat

        Parameters:
        ------------
        re_init: bool, (optional)
          Re-initialize FIBERFLAT parameter dict
        """
        #
        assert self.flavor in ['science']

        # Standard FIBERFLAT input parameters
        fflat_dict = dict(MAX_N_MASK=20000,  # Maximum number of pixels to mask
                          MAX_SCALE_OFF=0.05,  # Maximum offset in counts (fraction)
                          MAX_OFF=0.15,       # Maximum offset from unity
                          MAX_MEAN_OFF=0.05,  # Maximum offset in mean of fiberflat
                          MAX_RMS=0.02,      # Maximum RMS in fiberflat
                          )
        # Init
        self.init_qatype('FIBERFLAT', fflat_dict, re_init=re_init)

    def init_fluxcalib(self, re_init=False):
        """ Initialize parameters for FLUXCALIB QA
        Args:
            re_init: bool, (optional)
              Re-initialize  parameter dict

        Returns:

        """
        assert self.flavor in ['science']

        # Standard FLUXCALIB input parameters
        flux_dict = dict(ZP_WAVE=0.,          # Wavelength for ZP evaluation (camera dependent)
                         )

        if self.camera[0] == 'b':
            flux_dict['ZP_WAVE'] = 4800.  # Ang
        else:
            raise ValueError("Not ready for this camera!")

        # Init
        self.init_qatype('FLUXCALIB', flux_dict, re_init=re_init)

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
        elif qatype == 'FIBERFLAT':
            # Expecting: frame, fiberflat
            assert len(inputs) == 2
            # Init parameters (as necessary)
            self.init_fiberflat()
            # Run
            qadict = qa_fiberflat(self.data[qatype]['PARAM'], inputs[0], inputs[1])
        elif qatype == 'FLUXCALIB':
            # Expecting: frame, fluxcalib, individual_outputs (star by star)
            assert len(inputs) == 3
            # Init parameters (as necessary)
            self.init_fluxcalib()
            # Run
            qadict = qa_fluxcalib(self.data[qatype]['PARAM'], inputs[0], inputs[1], inputs[2])
        else:
            raise ValueError('Not ready to perform {:s} QA'.format(qatype))
        # Update
        self.data[qatype]['QA'] = qadict

    def __repr__(self):
        """
        Print formatting
        """
        return ('{:s}: camera={:s}, flavor={:s}'.format(
                self.__class__.__name__, self.camera, self.flavor))

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
