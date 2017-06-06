# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.io.params
"""
from __future__ import absolute_import, division
# The line above will help with 2to3 support.
import unittest, os
from ..io import params as io_params

class TestParams(unittest.TestCase):
    """Test desiutil.io.params.
    """

    @classmethod
    def setUpClass(cls):
        """Create unique test filename in a subdirectory.
        """
        pass

    @classmethod
    def tearDownClass(cls):
        """Cleanup test files if they exist.
        """
        pass

    def test_read(self):
        """Test desispec.io.params.read_params
        """
        params = io_params.read_params()
        # Test dict
        self.assertTrue(isinstance(params, dict))
        # Test global
        self.assertTrue(params == io_params.params)

def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
