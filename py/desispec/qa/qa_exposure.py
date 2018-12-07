"""
Classes to organize and execute QA for a DESI exposure
"""

from __future__ import print_function, absolute_import, division

import numpy as np
import os

from desiutil.log import get_logger
from desispec.io import read_params
from desispec import io as desiio
from desispec.qa.qa_frame import qaframe_from_frame
from desispec.io.qa import qafile_from_framefile

# log=get_logger()


class QA_Exposure(object):
    def __init__(self, expid, night, flavor, specprod_dir=None, in_data=None, no_load=False, **kwargs):
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
            no_load: bool, optional -- Do not load QA data (rare)

        Notes:

        Attributes:
            All input args become object attributes.
        """
        desi_params = read_params()
        assert flavor in desi_params['frame_types'], "Unknown flavor {} for night {} expid {}".format(flavor, night, expid)
        if flavor in ['science']:
            self.type = 'data'
        else:
            self.type = 'calib'

        self.expid = expid
        self.night = night
        self.specprod_dir = specprod_dir
        self.flavor = flavor
        self.meta = {}
        self.data = dict(flavor=self.flavor, expid=self.expid,
                         night=self.night, frames={})

        # Load?
        if no_load:
            return

        if in_data is None:
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
        if 'FLUXCALIB' not in self.data:
            self.data['FLUXCALIB'] = {}
        # Loop on channel
        cameras = list(self.data['frames'].keys())
        for channel in ['b','r','z']:
            # Init
            if channel not in self.data['FLUXCALIB']:
                self.data['FLUXCALIB'][channel] = {}
            # Load
            ZPval = []
            for camera in cameras:
                if camera[0] == channel:
                    ZPval.append(self.data['frames'][camera]['FLUXCALIB']['METRICS']['ZP'])
            # Measure RMS
            if len(ZPval) > 0:
                self.data['FLUXCALIB'][channel]['ZP_RMS'] = np.std(ZPval)

        # Figure
        qa_plots.exposure_fluxcalib(outfil, self.data)

    def load_meta(self, frame_meta):
        """ Load meta info from input Frame meta
        Args:
            frame_meta:
        """
        desi_params = read_params()
        for key in desi_params['frame_meta']:
            if key in ['CAMERA']:  # Frame specific
                continue
            try:
                self.meta[key] = frame_meta[key]
            except KeyError:
                print("Keyword {:s} not present!  Could be a problem".format(key))

    def load_qa_data(self, remove=False):
        """ Load the QA data files for a given exposure (currently yaml)
        Args:
            remove: bool, optional
              Remove QA frame files
        """
        qafiles = desiio.get_files(filetype='qa_'+self.type, night=self.night,
                                  expid=self.expid,
                                  specprod_dir=self.specprod_dir)
        # Load into frames
        for camera,qadata_path in qafiles.items():
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

    def build_qa_data(self, rebuild=False):
        """
        Build or re-build QA data

        Args:
            rebuild: bool, optional

        :return:
        """
        frame_files = desiio.get_files(filetype='frame', night=self.night,
                                   expid=self.expid,
                                   specprod_dir=self.specprod_dir)
        # Load into frames
        for camera, frame_file in frame_files.items():
            if rebuild:
                qafile, qatype = qafile_from_framefile(frame_file)
                if os.path.isfile(qafile):
                    os.remove(qafile)
            # Generate qaframe (and figures?)
            _ = qaframe_from_frame(frame_file, specprod_dir=self.specprod_dir, make_plots=False)
        # Reload
        self.load_qa_data()

    def __repr__(self):
        """ Print formatting
        """
        return ('{:s}: night={:s}, expid={:d}, type={:s}, flavor={:s}'.format(
                self.__class__.__name__, self.night, self.expid, self.type, self.flavor))
