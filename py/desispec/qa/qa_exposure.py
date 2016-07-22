"""
Classes to organize and execute QA for a DESI exposure
"""

from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import os

from desispec.log import get_logger

log=get_logger()


class QA_Exposure(object):
    def __init__(self, expid, night, flavor, specprod_dir=None, in_data=None, **kwargs):
        """
        Class to organize and execute QA for a DESI Exposure

        x.flavor, x.data
        
        Args:
            expid: int -- Exposure ID
            night: str -- YYYYMMDD
            flavor: str
              exposure type (e.g. flat, arc, science)
            specprod_dir(str): Path containing the exposures/ directory to use. If the value
                is None, then the value of :func:`specprod_root` is used instead.
            in_data: dict, optional -- Input data
              Mainly for reading from disk

        Notes:

        Attributes:
            All input args become object attributes.
        """
        flavors = ['none', 'flat', 'arc', 'gray', 'dark', 'bright', 'bgs', 'mws', 'lrg', 'elg', 'qso']
        assert flavor in flavors
        if flavor in ['dark', 'bright', 'bgs', 'mws', 'lrg', 'elg', 'qso']:
            self.type = 'data'
        else:
            self.type = 'calib'

        self.expid = expid
        self.night = night
        self.specprod_dir = specprod_dir
        self.flavor = flavor

        if in_data is None:
            self.data = dict(flavor=self.flavor, expid=self.expid,
                             night=self.night, frames={})
            self.load_qa_data(**kwargs)
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

    def load_qa_data(self, remove=True):
        """ Load the QA data files for a given exposure (currently yaml)
        Args:
            remove: bool, optional
              Remove QA frame files
        """

        from desispec import io as desiio
        qafiles = desiio.get_files(filetype='qa_'+self.type, night=self.night,
                                  expid=self.expid,
                                  specprod_dir=self.specprod_dir)
        # Load into frames
        for camera,qadata_path in qafiles.iteritems():
            qa_frame = desiio.load_qa_frame(qadata_path)
            # Remove?
            if remove:
                #import pdb; pdb.set_trace()
                os.remove(qadata_path)
            # Test
            for key in ['expid','night']:
                assert getattr(qa_frame,key) == getattr(self, key)
            # Save
            self.data['frames'][camera] = qa_frame.qa_data

    def __repr__(self):
        """ Print formatting
        """
        return ('{:s}: night={:s}, expid={:d}, type={:s}, flavor={:s}'.format(
                self.__class__.__name__, self.night, self.expid, self.type, self.flavor))
