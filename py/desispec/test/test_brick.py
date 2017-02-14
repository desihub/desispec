# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.test.test_brick
========================

Test desispec.brick.
"""
from __future__ import absolute_import, unicode_literals
# The line above will help with 2to3 support.
import unittest
import numpy as np
from ..brick import brickname


class TestBrick(unittest.TestCase):
    """Test desispec.brick.
    """

    def setUp(self):
        n = 10
        self.ra = np.linspace(0, 3, n) - 1.5
        self.dec = np.linspace(0, 3, n) - 1.5
        self.names = np.array(
            ['3587m015', '3592m010', '3597m010', '3597m005', '0002p000',
            '0002p000', '0007p005', '0007p010', '0012p010', '0017p015'])

    def test_brickname_scalar(self):
        """Test scalar to brick name conversion.
        """
        b = brickname(0, 0)
        self.assertEqual(b, '0002p000')

    def test_brickname_array(self):
        """Test array to brick name conversion.
        """
        bricknames = brickname(self.ra, self.dec)
        self.assertEqual(len(bricknames), len(self.ra))
        self.assertTrue((bricknames == self.names).all())

    def test_brickname_list(self):
        """Test list to brick name conversion.
        """
        bricknames = brickname(self.ra.tolist(), self.dec.tolist())
        self.assertEqual(len(bricknames), len(self.ra))
        self.assertTrue((bricknames == self.names).all())


def test_suite():
    """Allows testing of only this module with the command::
        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
