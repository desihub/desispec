# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.log.
"""
from __future__ import absolute_import, print_function
import unittest, os
import desispec.log as log

class TestLog(unittest.TestCase):
    """Test desispec.log
    """

    def setUp(self):
        """Reset the cached logging object for each test.
        """
        log.desi_logger = None

    def test_log(self):
        """Test basic logging functionality.
        """
        desi_level=os.getenv("DESI_LOGLEVEL")
        for level in [None,log.DEBUG,log.INFO,log.WARNING,log.ERROR] :
            logger=log.get_logger(level)
            print("with the requested debugging level={0}".format(level))
            if desi_level is not None and (desi_level != "" ) :
                print("(but overuled by env. DESI_LOGLEVEL='{0}')".format(desi_level))
            print("--------------------------------------------------")
            logger.debug("This is a debugging message")
            logger.info("This is an information")
            logger.warning("This is an warning")
            logger.error("This is an error")
            logger.critical("This is a critical error")

    def test_log_with_timestamp(self):
        """Test logging with timestamps.
        """
        desi_level=os.getenv("DESI_LOGLEVEL")
        for level in [None,log.DEBUG,log.INFO,log.WARNING,log.ERROR] :
            logger=log.get_logger(level, timestamp=True)
            print("with the requested debugging level={0}".format(level))
            if desi_level is not None and (desi_level != "" ) :
                print("(but overuled by env. DESI_LOGLEVEL='{0}')".format(desi_level))
            print("--------------------------------------------------")
            logger.debug("This is a debugging message")
            logger.info("This is an information")
            logger.warning("This is an warning")
            logger.error("This is an error")
            logger.critical("This is a critical error")

def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
