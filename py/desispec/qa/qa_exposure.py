"""
Classes to organize and execute QA for a DESI exposure
"""

from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np

from desispec.log import get_logger

log=get_logger()

class QA_Frame(object):
    def __init__(self, inp):
        """
        Class to organize and execute QA for a DESI frame

        x.flavor, x.qa_data, x.camera
        
        Args:
            inp : Frame object or dict
              * Frame -- Must contain meta data
              * dict -- Usually read from hard-drive

        Notes:

        """
        if isinstance(inp,dict):
            assert len(inp.keys()) == 1
            self.night = inp.keys()[0]  # Requires night in first key
            assert len(inp[self.night].keys()) == 1
            self.expid = inp[self.night].keys()[0]
            assert len(inp[self.night][self.expid].keys()) == 2
            self.flavor = inp[self.night][self.expid].pop('flavor')
            self.camera = inp[self.night][self.expid].keys()[0]
            assert self.camera[0] in ['b','r','z']
            self.qa_data = inp[self.night][self.expid][self.camera]
        else:
            # Generate from Frame and init QA data
            qkeys = ['flavor', 'camera', 'expid', 'night']
            for key in qkeys:
                setattr(self, key, inp.meta[key.upper()])
            self.qa_data = dict(flavor=self.flavor)

        # Final test
        assert self.flavor in ['none', 'flat', 'arc', 'dark', 'bright', 'bgs', 'mws', 'lrg', 'elg', 'qso']

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
        if (qatype not in self.qa_data.keys()) or re_init:
            self.qa_data[qatype] = {}
            self.qa_data[qatype]['PARAM'] = param
            return

        # Update the new parameters only
        for key in param.keys():
            if key not in self.qa_data[qatype]['PARAM'].keys():
                self.qa_data[qatype]['PARAM'][key] = param[key]

    def init_fiberflat(self, re_init=False):
        """Initialize parameters for FIBERFLAT QA
        QA method is desispec.fiberflat.qa_fiberflat

        Parameters:
        ------------
        re_init: bool, (optional)
          Re-initialize FIBERFLAT parameter dict
        """
        #
        assert self.flavor in ['flat']

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
        assert self.flavor in ['dark','bright','bgs','mws','lrg','elg','qso']

        # Standard FLUXCALIB input parameters
        flux_dict = dict(ZP_WAVE=0.,        # Wavelength for ZP evaluation (camera dependent)
                         MAX_ZP_OFF=0.2,    # Max offset in ZP for individual star
                         )

        if self.camera[0] == 'b':
            flux_dict['ZP_WAVE'] = 4800.  # Ang
        elif self.camera[0] == 'r':
            flux_dict['ZP_WAVE'] = 6500.  # Ang
        elif self.camera[0] == 'z':
            flux_dict['ZP_WAVE'] = 8250.  # Ang
        else:
            log.error("Not ready for camera {}!".format(self.camera))

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
        assert self.flavor in ['dark','bright','bgs','mws','lrg','elg','qso']

        # Standard SKYSUB input parameters
        sky_dict = dict(
            PCHI_RESID=0.05, # P(Chi^2) limit for bad skyfiber model residuals
            PER_RESID=95.,   # Percentile for residual distribution
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
        from desispec.sky import qa_skysub
        from desispec.fiberflat import qa_fiberflat
        from desispec.fluxcalibration import qa_fluxcalib

        # Check for previous QA if clobber==False
        if not clobber:
            # QA previously performed?
            if 'QA' in self.qa_data[qatype].keys():
                return
        # Run
        if qatype == 'SKYSUB':
            # Expecting: frame, skymodel
            assert len(inputs) == 2
            # Init parameters (as necessary)
            self.init_skysub()
            # Run
            qadict = qa_skysub(self.qa_data[qatype]['PARAM'],
                inputs[0], inputs[1])
        elif qatype == 'FIBERFLAT':
            # Expecting: frame, fiberflat
            assert len(inputs) == 2
            # Init parameters (as necessary)
            self.init_fiberflat()
            # Run
            qadict = qa_fiberflat(self.qa_data[qatype]['PARAM'], inputs[0], inputs[1])
        elif qatype == 'FLUXCALIB':
            # Expecting: frame, fluxcalib, individual_outputs (star by star)
            assert len(inputs) == 3
            # Init parameters (as necessary)
            self.init_fluxcalib()
            # Run
            qadict = qa_fluxcalib(self.qa_data[qatype]['PARAM'],
                                  inputs[0], inputs[1], inputs[2])
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
    def __init__(self, expid, night, specprod_dir=None, flavor='none', in_data=None):
        """
        Class to organize and execute QA for a DESI Exposure

        x.flavor, x.data
        
        Args:
            expid: int -- Exposure ID
            night: str -- YYYYMMDD
            specprod_dir(str): Path containing the exposures/ directory to use. If the value
                is None, then the value of :func:`specprod_root` is used instead.
            flavor: str, optional exposure type (e.g. flat, arc, science)
            in_data: dict, optional -- Input data 
              Mainly for reading from disk

        Notes:

        Attributes:
            All input args become object attributes.
        """
        assert flavor in ['none', 'flat', 'arc', 'dark', 'bright', 'bgs', 'mws', 'lrg', 'elg', 'qso']

        self.expid = expid
        self.night = night
        self.specprod_dir = specprod_dir
        self.flavor = flavor
        
        if in_data is None:
            self.data = dict(flavor=self.flavor, expid=self.expid,
                             night=self.night, frames={})
            self.load_qa_data()
        else:
            assert isinstance(in_data,dict)
            self.data = in_data

    def fluxcalib(self, outfil):
        """ Perform QA on fluxcalib results for an Exposure

        Args:
            outfil: str -- Filename for PDF  (may automate)

        Independent results for each channel
        """
        from . import qa_plots
        # Init
        if 'FLUXCALIB' not in self.data.keys():
            self.data['FLUXCALIB'] = {}
        # Loop on channel
        cameras = self.data['frames'].keys()
        for channel in ['b','r','z']:
            # Init
            if channel not in self.data['FLUXCALIB'].keys():
                self.data['FLUXCALIB'][channel] = {}
            # Load
            ZPval = []
            for camera in cameras:
                if camera[0] == channel:
                    ZPval.append(self.data['frames'][camera]['FLUXCALIB']['QA']['ZP'])
            # Measure RMS
            if len(ZPval) > 0:
                self.data['FLUXCALIB'][channel]['ZP_RMS'] = np.std(ZPval)

        # Figure
        qa_plots.exposure_fluxcalib(outfil, self.data)

    def load_qa_data(self):
        """ Load the QA data files for a given exposure (currently yaml)
        """
        from desispec import io as desiio
        qadata = desiio.get_files(filetype='qa_data', night=self.night,
                                  expid=self.expid,
                                  specprod_dir=self.specprod_dir)
        # Load into frames
        for camera,qadata_path in qadata.iteritems():
            qa_frame = desiio.load_qa_frame(qadata_path)
            #qa_data = desiio.read_qa_data(qadata_path)
            # Test
            for key in ['expid','night','flavor']:
                assert getattr(qa_frame,key) == getattr(self, key)
            #assert qa_data['flavor'] == self.flavor
            # Save
            self.data['frames'][camera] = qa_frame.qa_data
            self.data['frames'][camera]['file'] = qadata_path
