"""
desispec.qa.qa_prod
===================

Class to organize QA for a full DESI production run.
"""

from __future__ import print_function, absolute_import, division

import numpy as np
import glob, os
import warnings

from desispec.io import get_exposures
from desispec.io import get_files
from desispec.io import specprod_root
from desispec.io import get_nights
from .qa_multiexp import QA_MultiExp
from .qa_night import QA_Night
from desispec.io import write_qa_exposure

from desiutil.log import get_logger

# log = get_logger()

from . import qa_multiexp

class QA_Prod(qa_multiexp.QA_MultiExp):
    def __init__(self, specprod_dir=None, **kwargs):
        """ Class to organize and execute QA for a DESI production

        Args:
            specprod_dir(str): Path containing the exposures/ directory to use. If the value
                is None, then the value of :func:`specprod_root` is used instead.
        Notes:

        Attributes:
            qa_exps : list
              List of QA_Exposure classes, one per exposure in production
            data : dict
        """
        if specprod_dir is None:
            specprod_dir = specprod_root()
        self.specprod_dir = specprod_dir
        # Init
        QA_MultiExp.__init__(self, specprod_dir=specprod_dir, **kwargs)
        # Load up exposures for the full production
        nights = get_nights(specprod_dir=self.specprod_dir)
        for night in nights:
            self.mexp_dict[night] = {}
            for exposure in get_exposures(night, specprod_dir = self.specprod_dir):
                # Object only??
                frames_dict = get_files(filetype = str('frame'), night = night,
                                        expid = exposure, specprod_dir = self.specprod_dir)
                self.mexp_dict[night][exposure] = frames_dict
        # Output file names
        self.qaexp_outroot = self.qaprod_dir+'/'+self.prod_name+'_qa'
        # Nights list
        self.qa_nights = []

    def load_data(self, inroot=None):
        """ Load QA data from night objects on disk
        """
        self.data = {}
        # Load
        for night in self.mexp_dict.keys():
            qaNight = QA_Night(night, specprod_dir=self.specprod_dir, qaprod_dir=self.qaprod_dir)
            qaNight.load_data()
            #
            self.data[night] = qaNight.data[night]

    def build_data(self):
        """  Build QA data dict from the nights
        """
        from desiutil.io import combine_dicts
        # Loop on exposures
        odict = {}
        for qanight in self.qa_nights:
            for qaexp in qanight.qa_exps:
                # Get the exposure dict
                idict = write_qa_exposure('foo', qaexp, ret_dict=True)
                odict = combine_dicts(odict, idict)
        # Finish
        self.data = odict

    def slurp_nights(self, make_frameqa=False, remove=True, restrict_nights=None,
                     write_nights=False, **kwargs):
        """ Slurp all the individual QA files, night by night
        Loops on nights, generating QANight objects along the way

        Args:
            make_frameqa: bool, optional
              Regenerate the individual QA files (at the frame level first)
            remove: bool, optional
              Remove the individual QA files?
            restrict_nights: list, optional
            **kwargs:
              Passed to make_frameqa()

        Returns:

        """
        log = get_logger()
        # Remake?
        if make_frameqa:
            self.make_frameqa(**kwargs)
        # Reset
        log.info("Resetting QA_Night objects")
        self.qa_nights = []
        # Loop on nights
        for night in self.mexp_dict.keys():
            if restrict_nights is not None:
                if night not in restrict_nights:
                    continue
            qaNight = QA_Night(night, specprod_dir=self.specprod_dir, qaprod_dir=self.qaprod_dir)
            qaNight.slurp(remove=remove)
            # Save nights
            self.qa_nights.append(qaNight)
            # Write?
            if write_nights:
                qaNight.write_qa_exposures()
