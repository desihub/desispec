# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.io.params
"""
from __future__ import absolute_import, division
# The line above will help with 2to3 support.
import unittest, os
from ..io import params as io_params
import tempfile

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
        # Test global; changing params changes global
        params['blat'] = 1
        p2 = io_params.read_params()
        self.assertIn('blat', p2)
        self.assertIs(params, p2)
        
        # Test that reload replaces the global
        p3 = io_params.read_params(reload=True)
        self.assertNotIn('blat', p3)
        self.assertIsNot(params, p3)
        
        #Test reading a different file
        with tempfile.NamedTemporaryFile() as fx:
            fx.write(b'{bar: 1, biz: 2, bat: 3}')
            fx.flush()
            p4 = io_params.read_params(filename=fx.name)
            self.assertIn('bar', p4)
            self.assertIn('biz', p4)
            self.assertIn('bat', p4)
            self.assertIsNot(p3, p4)
            self.assertIsNot(p2, p4)

