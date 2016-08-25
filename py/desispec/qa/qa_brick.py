"""
Classes to organize and execute QA for a DESI exposure
"""

from __future__ import print_function, absolute_import, division

import numpy as np


class QA_Brick(object):
    def __init__(self, name='None', in_data=None):
        """
        Class to organize and execute QA for a DESI brick
        x.flavor, x.data, x.camera

        Args:
            name: str, optional
            in_data: dict, optional
              Allows for previous data to be ingested
        Notes:
        Attributes:
            All input args become object attributes.
        """
        # Parse
        self.brick_name = name

        # Initialize data
        if in_data is None:
            self.data = dict(name='')
        else:
            assert isinstance(in_data,dict)
            self.data = in_data

    def init_qatype(self, qatype, param, re_init=False):
        """Initialize parameters for a given qatype
        qatype: str
          Type of QA to be performed (e.g. ZBEST)
        param: dict
          Dict of parameters to guide QA
        re_init: bool, (optional)
          Re-initialize parameter dict
          Code will always add new parameters if any exist
        """
        # Fill and return if not set previously or if re_init=True
        if (qatype not in self.data) or re_init:
            self.data[qatype] = {}
            self.data[qatype]['PARAM'] = param
            return

        # Update the new parameters only
        for key in param:
            if key not in self.data[qatype]['PARAM']:
                self.data[qatype]['PARAM'][key] = param[key]

    def init_zbest(self, re_init=False):
        """Initialize parameters for ZBEST output
        QA method is desispec.zfind.zfind
        Parameters:
        ------------
        re_init: bool, (optional)
          Re-initialize ZBEST parameter dict
        """
        #

        # Standard FIBERFLAT input parameters
        zbest_dict = dict(MAX_NFAIL=10,  # Maximum number of failed redshifts
                          ELG_TYPES=['ssp_em_galaxy', 'ELG'],
                          LRG_TYPES=['LRG'],
                          QSO_TYPES=['QSO'],
                          STAR_TYPES=['spEigenStar'],
                          )
        # Init
        self.init_qatype('ZBEST', zbest_dict, re_init=re_init)

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
        #from desispec.zfind.zfind import qa_zbest
        from desispec.zfind import zfind

        # Check for previous QA if clobber==False
        if not clobber:
            # QA previously performed?
            if 'QA' in self.data[qatype]:
                return
        # Run
        if qatype == 'ZBEST':
            # Expecting: zf, brick
            assert len(inputs) == 2
            # Init parameters (as necessary)
            self.init_zbest()
            # Run
            reload(zfind)
            qadict = zfind.qa_zbest(self.data[qatype]['PARAM'], inputs[0], inputs[1])
        else:
            raise ValueError('Not ready to perform {:s} QA'.format(qatype))
        # Update
        self.data[qatype]['QA'] = qadict

    def __repr__(self):
        """
        Print formatting
        """
        return ('{:s}: name={:s}'.format(
                self.__class__.__name__, self.brick_name))