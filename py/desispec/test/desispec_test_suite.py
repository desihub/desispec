"""
desispec.test.desispec_test_suite
=================================

Used to initialize the unit test framework via ``python setup.py test``.
"""
#
from __future__ import absolute_import, division, print_function, unicode_literals
#
import unittest
from os.path import dirname
#
def desispec_test_suite():
    desispec_dir = dirname(dirname(__file__))
    return unittest.defaultTestLoader.discover(desispec_dir)
