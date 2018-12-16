"""
Classes to organize and execute QA for a DESI exposure
"""

from __future__ import print_function, absolute_import, division

import numpy as np
import os

from astropy.table import Table, vstack

from desiutil.log import get_logger
from desispec.io import read_params
from desispec import io as desiio
from desispec.qa.qa_frame import qaframe_from_frame
from desispec.io.qa import qafile_from_framefile
from desispec.io import load_qa_multiexp
from desispec.io import qaprod_root
from desispec.io import read_meta_frame
from desispec.io import get_files
from desispec.io import write_qa_exposure
from desispec.io import write_qa_multiexp

# log=get_logger()
desi_params = read_params()


class QA_Exposure(object):
    def __init__(self, expid, night, flavor=None, specprod_dir=None, in_data=None,
                 qaprod_dir=None, no_load=False, multi_root=None, **kwargs):
        """
        Class to organize and execute QA for a DESI Exposure

        x.flavor, x.data

        Args:
            expid: int -- Exposure ID
            night: str -- YYYYMMDD
            specprod_dir(str): Path containing the exposures/ directory to use. If the value
                is None, then the value of :func:`specprod_root` is used instead.
            in_data: dict, optional -- Input data
              Mainly for reading from disk
            no_load: bool, optional -- Do not load QA data (rare)
            multi_root: str, optional
              Load QA from a slurped file.
              This is the root and the path is qaprod_dir

        Notes:

        Attributes:
            All input args become object attributes.
        """
        # Init
        self.expid = expid
        self.night = night
        self.meta = {}
        # Paths
        self.specprod_dir = specprod_dir
        if qaprod_dir is None:
            qaprod_dir = qaprod_root(self.specprod_dir)
        self.qaprod_dir  = qaprod_dir

        # Load meta
        frames_dict = get_files(filetype = str('frame'), night = night,
                                expid=expid, specprod_dir = self.specprod_dir)
        frame_file = list(frames_dict.items())[0][1]  # Any one will do
        frame_meta = read_meta_frame(frame_file)
        self.load_meta(frame_meta)
        flavor = self.meta['FLAVOR']  # Over-rides any input value

        assert flavor in desi_params['frame_types'], "Unknown flavor {} for night {} expid {}".format(flavor, night, expid)
        if flavor in ['science']:
            self.type = 'data'
        else:
            self.type = 'calib'
        self.flavor = flavor

        # Internal dicts
        self.data = dict(flavor=self.flavor, expid=self.expid,
                         night=self.night, frames={})

        # Load?
        if no_load:
            return

        if in_data is None:
            self.load_qa_data(multi_root=multi_root, **kwargs)
        else:
            assert isinstance(in_data,dict)
            self.data = in_data

        # Others
        self.qa_s2n = None

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
            frame_meta: dict of meta data from a frame file
        """
        desi_params = read_params()
        for key in desi_params['frame_meta']:
            if key in ['CAMERA']:  # Frame specific
                continue
            try:
                self.meta[key] = frame_meta[key]
            except KeyError:
                print("Keyword {:s} not present!  Could be a problem".format(key))

    def load_qa_data(self, remove=False, multi_root=None):
        """ Load the QA data files for a given exposure (currently yaml)
        Args:
            remove: bool, optional
              Remove QA frame files
        """
        if multi_root is None:
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
        else:
            # Load
            mdict = load_qa_multiexp(os.path.join(self.qaprod_dir, multi_root))
            self.parse_multi_qa_dict(mdict)

    def parse_multi_qa_dict(self, mdict):
        """ Deal with different packing of QA data in slurp file

        Args:
            mdict: dict

        Returns:

        """
        # Parse
        for key in mdict[self.night][str(self.expid)].keys():
            # A bit kludgy
            if len(key) > 2:
                if key == 'meta':
                    self.data[key] = mdict[self.night][str(self.expid)][key].copy()
                continue
            # Load em
            self.data['frames'][key] = mdict[self.night][str(self.expid)][key].copy()

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

    def s2n_table(self):
        """
        Generate a flat Table of QA S/N measurements for the Exposure
          Includes all fibers of the exposure

        Args:

        Returns:

        """
        from desispec.qa.qalib import s2n_funcs

        sub_tbls = []
        # Load up
        for camera in self.data['frames'].keys():
            # Sub_tbl
            sub_tbl = Table()
            sub_tbl['MEDIAN_SNR'] = self.data['frames'][camera]['S2N']['METRICS']['MEDIAN_SNR']
            sub_tbl['FIBER'] = np.arange(len(sub_tbl), dtype=int)
            sub_tbl['CAMERA'] = camera
            sub_tbl['NIGHT'] = self.night
            sub_tbl['EXPID'] = self.expid
            sub_tbl['CHANNEL'] = camera[0]
            # Ugly S/N (Object/fiber based)
            s2n_dict = self.data['frames'][camera]['S2N']
            max_o = np.max([len(otype) for otype in s2n_dict['METRICS']['OBJLIST']])
            objtype = np.array([' '*max_o]*len(sub_tbl))
            # Coeffs
            coeffs = np.zeros((len(sub_tbl), len(s2n_dict['METRICS']['FITCOEFF_TGT'][0])))
            # Others
            mags = np.zeros_like(sub_tbl['MEDIAN_SNR'].data)
            resid = -999. * np.ones_like(sub_tbl['MEDIAN_SNR'].data)
            # Fitting
            funcMap = s2n_funcs(exptime=s2n_dict['METRICS']['EXPTIME']) #r2=s2n_dict['METRICS']['r2'])
            fitfunc = funcMap['astro']
            for oid, otype in enumerate(s2n_dict['METRICS']['OBJLIST']):
                fibers = np.array(s2n_dict['METRICS']['{:s}_FIBERID'.format(otype)])
                if len(fibers) == 0:
                    continue
                coeff = s2n_dict['METRICS']['FITCOEFF_TGT'][oid]
                coeffs[fibers,:] = np.outer(np.ones_like(fibers), coeff)
                # Set me
                objtype[fibers] = otype
                mags[fibers] = np.array(s2n_dict["METRICS"]["SNR_MAG_TGT"][oid][1])

                # Residuals
                flux = 10 ** (-0.4 * (mags[fibers] - 22.5))
                fit_snr = fitfunc(flux, *coeff)
                resid[fibers] = (sub_tbl['MEDIAN_SNR'][fibers] - fit_snr) / fit_snr
            # Sub_tbl
            sub_tbl['MAGS'] = mags
            sub_tbl['RESID'] = resid
            sub_tbl['OBJTYPE'] = objtype
            sub_tbl['COEFFS'] = coeffs
            # Save
            sub_tbls.append(sub_tbl)
        # Stack me
        qa_tbl = vstack(sub_tbls)
        # Hold
        self.qa_s2n = qa_tbl
        # Add meta
        self.qa_s2n.meta = self.data['meta']

    def slurp_into_file(self, multi_root):
        # Load
        mdict_root = os.path.join(self.qaprod_dir, multi_root)
        mdict = load_qa_multiexp(mdict_root)
        # Check on night
        if self.night not in mdict.keys():
            mdict[self.night] = {}
        # Insert
        idict = write_qa_exposure('foo', self, ret_dict=True)
        mdict[self.night][str(self.expid)] = idict[self.night][self.expid]
        # Write
        write_qa_multiexp(mdict_root, mdict)

    def __repr__(self):
        """ Print formatting
        """
        return ('{:s}: night={:s}, expid={:d}, type={:s}, flavor={:s}'.format(
                self.__class__.__name__, self.night, self.expid, self.type, self.flavor))
