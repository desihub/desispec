"""
desispec.qa.qa_night
====================

Class to organize QA for one night of DESI exposures.
"""

from __future__ import print_function, absolute_import, division

import numpy as np
import glob, os
import warnings

from desispec.io import get_exposures
from desispec.io import get_files
from desispec.io import read_meta_frame
from desispec.io import get_nights
from .qa_multiexp import QA_MultiExp

from desiutil.log import get_logger

# log = get_logger()


class QA_Night(QA_MultiExp):
    def __init__(self, night, **kwargs):
        """ Class to organize and execute QA for a DESI production

        Args:
            specprod_dir(str): Path containing the exposures/ directory to use. If the value
                is None, then the value of :func:`specprod_root` is used instead.
        Notes:
            **kwargs are passed to QA_MultiExp

        Attributes:
            qa_exps : list
              List of QA_Exposure classes, one per exposure in production
            data : dict
        """
        # Init
        self.night = night
        # Instantiate
        QA_MultiExp.__init__(self, **kwargs)
        # Load up exposures for the full production
        nights = get_nights(specprod_dir=self.specprod_dir)
        # Check the night exists
        if self.night not in nights:
            raise IOError("Night {} not in known nights in {}".format(
                self.night, self.specprod_dir))
        # Load up
        self.mexp_dict[self.night] = {}
        for exposure in get_exposures(self.night, specprod_dir = self.specprod_dir):
            # Object only??
            frames_dict = get_files(filetype = str('frame'), night = self.night,
                                    expid = exposure, specprod_dir = self.specprod_dir)
            self.mexp_dict[self.night][exposure] = frames_dict
        # Output file names
        self.qaexp_outroot = self.qaprod_dir+'/'+self.night+'_qa'


