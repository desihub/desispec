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
from .. import brick


class TestBrick(unittest.TestCase):
    """Test desispec.brick.
    """

    def setUp(self):
        n = 10
        self.ra = np.linspace(0, 3, n) - 1.5
        self.dec = np.linspace(0, 3, n) - 1.5
        #ADM note that these are the correct brickIDs for bricksize=0.25
        self.brickids = np.array(
            [323162, 324603, 327484, 328926, 330367, 331808, 333250, 334691,
             337572, 339014])
        #ADM note that these are the correct brick names for bricksize=0.5
        self.names = np.array(
            ['3587m015', '3587m010', '3592m010', '3597m005', '3597p000',
            '0002p000', '0007p005', '0007p010', '0012p010', '0017p015'])
        #ADM some brick areas
        self.areas = np.array(
            [0.062478535,  0.062485076,  0.062494595,  0.062497571,  0.062499356,
             0.062499356,  0.062497571,  0.062494595,  0.062485076,  0.062478535], dtype='<f4')

    def test_brickvertices_scalar(self):
        """Test scalar to brick area conversion.
        """
        b = brick.Bricks()
        ra, dec = 0, 0
        bverts = b.brickvertices(ra,dec)
        self.assertTrue( (np.min(bverts[:,0]) <= ra) & (np.max(bverts[:,0]) >= ra) )
        self.assertTrue( (np.max(bverts[:,1]) <= dec) & (np.max(bverts[:,1]) >= dec) )

    def test_brickvertices_array(self):
        """Test array to brick area conversion.
        """
        b = brick.Bricks()
        bverts = b.brickvertices(self.ra, self.dec)
        self.assertEqual(len(bareas), len(self.ra))
        self.assertTrue((bareas == self.areas).all())

    def test_brickarea_scalar(self):
        """Test scalar to brick area conversion.
        """
        b = brick.Bricks()
        barea = b.brickarea(0, 0)
        self.assertEqual(barea, np.array([0.0624999515],dtype='<f4')[0])

    def test_brickarea_array(self):
        """Test array to brick area conversion.
        """
        b = brick.Bricks()
        bareas = b.brickarea(self.ra, self.dec)
        self.assertEqual(len(bareas), len(self.ra))
        self.assertTrue((bareas == self.areas).all())

    def test_brickarea_wrap(self):
        """Test RA wrap and poles for brick areas"""
        b = brick.Bricks()
        b1 = b.brickarea(1, 0)
        b2 = b.brickarea(361, 0)
        self.assertEqual(b1, b2)

        b1 = b.brickarea(-0.5, 0)
        b2 = b.brickarea(359.5, 0)
        self.assertEqual(b1, b2)

        b1 = b.brickarea(0, 90)
        b2 = b.brickarea(90, 90)
        self.assertEqual(b1, b2)
        self.assertEqual(b1, np.array([0.049087364],dtype='<f4')[0])

        b1 = b.brickarea(0, -90)
        b2 = b.brickarea(90, -90)
        self.assertEqual(b1, b2)
        self.assertEqual(b1, np.array([0.049087364],dtype='<f4')[0])

    def test_brickid_scalar(self):
        """Test scalar to BRICKID conversion.
        """
        b = brick.Bricks()
        bid = b.brickid(0, 0)
        self.assertEqual(bid, 330368)

    def test_brickid_array(self):
        """Test array to BRICKID conversion.
        """
        b = brick.Bricks()
        bids = b.brickid(self.ra, self.dec)
        self.assertEqual(len(bids), len(self.ra))
        self.assertTrue((bids == self.brickids).all())

    def test_brickid_wrap(self):
        """Test RA wrap and poles for BRICKIDs"""
        b = brick.Bricks()
        b1 = b.brickid(1, 0)
        b2 = b.brickid(361, 0)
        self.assertEqual(b1, b2)

        b1 = b.brickid(-0.5, 0)
        b2 = b.brickid(359.5, 0)
        self.assertEqual(b1, b2)

        b1 = b.brickid(0, 90)
        b2 = b.brickid(90, 90)
        self.assertEqual(b1, b2)
        self.assertEqual(b1, 662174)

        b1 = b.brickid(0, -90)
        b2 = b.brickid(90, -90)
        self.assertEqual(b1, b2)
        self.assertEqual(b1, 1)

    def test_brickid_wrap(self):
        """Test RA wrap and poles for BRICKIDs"""
        b = brick.Bricks()
        b1 = b.brickid(1, 0)
        b2 = b.brickid(361, 0)
        self.assertEqual(b1, b2)

        b1 = b.brickid(-0.5, 0)
        b2 = b.brickid(359.5, 0)
        self.assertEqual(b1, b2)

        b1 = b.brickid(0, 90)
        b2 = b.brickid(90, 90)
        self.assertEqual(b1, b2)
        self.assertEqual(b1, 662174)

        b1 = b.brickid(0, -90)
        b2 = b.brickid(90, -90)
        self.assertEqual(b1, b2)
        self.assertEqual(b1, 1)

    def test_brickname_scalar(self):
        """Test scalar to brick name conversion.
        """
        b = brickname(0, 0, bricksize=0.5)
        self.assertEqual(b, '0002p000')

    def test_brickname_array(self):
        """Test array to brick name conversion.
        """
        bricknames = brickname(self.ra, self.dec, bricksize=0.5)
        self.assertEqual(len(bricknames), len(self.ra))
        self.assertTrue((bricknames == self.names).all())

    def test_brickname_wrap(self):
        """Test RA wrap and poles for bricknames"""
        b1 = brickname(1, 0)
        b2 = brickname(361, 0)
        self.assertEqual(b1, b2)

        b1 = brickname(-0.5, 0)
        b2 = brickname(359.5, 0)
        self.assertEqual(b1, b2)

        b1 = brickname(0, 90)
        b2 = brickname(90, 90)
        self.assertEqual(b1, b2)
        self.assertEqual(b1, '1800p900')

        b1 = brickname(0, -90)
        b2 = brickname(90, -90)
        self.assertEqual(b1, b2)
        self.assertEqual(b1, '1800m900')

    def test_brickname_list(self):
        """Test list to brick name conversion.
        """
        bricknames = brickname(self.ra.tolist(), self.dec.tolist(),bricksize=0.5)
        self.assertEqual(len(bricknames), len(self.ra))
        self.assertTrue((bricknames == self.names).all())

    def test_bricksize(self):
        brick._bricks = None
        blat = brickname(0, 0, bricksize=0.5)
        self.assertEqual(brick._bricks.bricksize, 0.5)
        blat = brickname(0, 0, bricksize=0.25)
        self.assertEqual(brick._bricks.bricksize, 0.25)
        brick._bricks = None

def test_suite():
    """Allows testing of only this module with the command::
        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
