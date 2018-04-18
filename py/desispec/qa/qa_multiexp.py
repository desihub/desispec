""" Class to organize QA for multiple exposures
Likely to only be used as parent of QA_Night or QA_Prod
"""

from __future__ import print_function, absolute_import, division

import numpy as np
import glob, os
import warnings

from desispec.io import get_exposures
from desispec.io import get_files
from desispec.io import read_meta_frame
from desispec.io import specprod_root
from desispec.io import write_qa_exposure
from desispec.io import write_qa_multiexp
from desispec.io import qaprod_root

from desiutil.log import get_logger

# log = get_logger()


class QA_MultiExp(object):
    def __init__(self, specprod_dir=None, qaprod_dir=None):
        """ Class to organize and execute QA for a DESI production

        Args:
            specprod_dir(str): Path containing the exposures/ directory to use. If the value
                is None, then the value of :func:`specprod_root` is used instead.
            qaprod_dir(str): Path containing the root path for QA output
        Notes:

        Attributes:
            qa_exps : list
              List of QA_Exposure classes, one per exposure in production
            data : dict
        """
        # Init
        if specprod_dir is None:
            specprod_dir = specprod_root()
        if qaprod_dir is None:
            qaprod_dir = qaprod_root()
        #
        self.specprod_dir = specprod_dir
        self.qaprod_dir = qaprod_dir
        tmp = specprod_dir.split('/')
        self.prod_name = tmp[-1] if (len(tmp[-1]) > 0) else tmp[-2]
        # Exposure dict
        self.mexp_dict = {}
        # QA Exposure objects
        self.qa_exps = []
        # dict to hold QA data
        #  Data Model :  key1 = Night(s);  key2 = Expids
        self.data = {}
        #
        self.qaexp_outroot = None

    def build_data(self):
        """  Build QA data dict
        """
        from desiutil.io import combine_dicts
        # Loop on exposures
        odict = {}
        for qaexp in self.qa_exps:
            # Get the exposure dict
            idict = write_qa_exposure('foo', qaexp, ret_dict=True)
            odict = combine_dicts(odict, idict)
        # Finish
        self.data = odict

    def get_qa_table(self, qatype, metric, nights='all', channels='all'):
        """ Generate a table of QA values from .data
        Args:
            qatype: str
              FIBERFLAT, SKYSUB
            metric: str
            nights: str or list of str, optional
            channels: str or list of str, optional
              'b', 'r', 'z'

        Returns:
            qa_tbl: Table
        """
        from astropy.table import Table
        out_list = []
        out_expid = []
        out_expmeta = []
        out_cameras = []
        # Nights
        for night in self.data:
            if (night not in nights) and (nights != 'all'):
                continue
            # Exposures
            for expid in self.data[night]:
                # Cameras
                exp_meta = self.data[night][expid]['meta']
                for camera in self.data[night][expid]:
                    if camera in ['flavor', 'meta']:
                        continue
                    if (camera[0] not in channels) and (channels != 'all'):
                        continue
                    # Grab
                    try:
                        val = self.data[night][expid][camera][qatype]['METRICS'][metric]
                    except KeyError:  # Each exposure has limited qatype
                        pass
                    except TypeError:
                        import pdb; pdb.set_trace()
                    else:
                        if isinstance(val, (list,tuple)):
                            out_list.append(val[0])
                        else:
                            out_list.append(val)
                        # Meta data
                        out_expid.append(expid)
                        out_cameras.append(camera)
                        out_expmeta.append(exp_meta)
        # Return Table
        qa_tbl = Table()
        qa_tbl[metric] = out_list
        qa_tbl['EXPID'] = out_expid
        qa_tbl['CAMERA'] = out_cameras
        # Add expmeta
        for key in out_expmeta[0].keys():
            tmp_list = []
            for exp_meta in out_expmeta:
                tmp_list.append(exp_meta[key])
            qa_tbl[key] = tmp_list
        return qa_tbl

    def load_data(self, inroot=None):
        """ Load QA data from disk
        """
        from desispec.io import load_qa_multiexp
        # Init
        if inroot is None:
            inroot = self.qaexp_outroot
        # Load
        self.data = load_qa_multiexp(inroot)

    def make_frameqa(self, make_plots=False, clobber=False):
        """ Work through the exposures and make QA for all frames

        Parameters:
            make_plots: bool, optional
              Remake the plots too?
            clobber: bool, optional
        Returns:

        """
        # imports
        from desispec.qa.qa_frame import qaframe_from_frame
        from desispec.io.qa import qafile_from_framefile

        # Loop on nights
        for night in self.mexp_dict.keys():
            for exposure in self.mexp_dict[night]:
                # Object only??
                for camera,frame_fil in self.mexp_dict[night][exposure].items():
                    # Load frame
                    qafile, _ = qafile_from_framefile(frame_fil, qaprod_dir=self.qaprod_dir)
                    if os.path.isfile(qafile) and (not clobber):
                        continue
                    qaframe_from_frame(frame_fil, make_plots=make_plots, qaprod_dir=self.qaprod_dir)

    def slurp(self, make_frameqa=False, remove=True, **kwargs):
        """ Slurp all the individual QA files to generate
        a list of QA_Exposure objects

        Args:
            make_frameqa: bool, optional
              Regenerate the individual QA files (at the frame level first)
            remove: bool, optional
              Remove the individual QA files?

        Returns:

        """
        from desispec.qa import QA_Exposure
        log = get_logger()
        # Remake?
        if make_frameqa:
            self.make_frameqa(**kwargs)
        # Loop on nights
        # Reset
        log.info("Resetting QA_Exposure objects")
        self.qa_exps = []
        # Loop
        for night in self.mexp_dict.keys():
            # Loop on exposures
            for exposure in self.mexp_dict[night].keys():
                frames_dict = self.mexp_dict[night][exposure]
                if len(frames_dict) == 0:
                    continue
                # Load any frame (for the type and meta info)
                key = list(frames_dict.keys())[0]
                frame_fil = frames_dict[key]
                frame_meta = read_meta_frame(frame_fil)
                qa_exp = QA_Exposure(exposure, night, frame_meta['FLAVOR'],
                                     specprod_dir=self.specprod_dir, remove=remove)
                qa_exp.load_meta(frame_meta)
                # Append
                self.qa_exps.append(qa_exp)

    def write_qa_exposures(self, outroot=None, **kwargs):
        """  Write the slurp of QA Exposures to the hard drive
        Args:
            outroot: str
            **kwargs:

        Returns:
            output_file : str

        """
        if outroot is None:
            outroot = self.qaexp_outroot
        return write_qa_multiexp(outroot, self, **kwargs)

    def __repr__(self):
        """ Print formatting
        """
        return ('{:s}: specprod_dir={:s}'.format(self.__class__.__name__, self.specprod_dir))
