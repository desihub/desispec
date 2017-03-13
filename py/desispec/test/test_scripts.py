# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.scripts.
"""
from __future__ import absolute_import, division
# The line above will help with 2to3 support.
import unittest


class TestIO(unittest.TestCase):
    """Test desispec.scripts.
    """

    @classmethod
    def setUpClass(cls):
        cls.environ_cache = dict()

    @classmethod
    def tearDownClass(cls):
        cls.environ_cache.clear()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def dummy_environment(self, env):
        """Set dummy environment variables for testing.

        Parameters
        ----------
        env : :class:`dict`
            Mapping of variables to values
        """
        from os import environ
        for k in env:
            if k in environ:
                self.environ_cache[k] = environ[k]
            else:
                self.environ_cache[k] = None
            environ[k] = env[k]
        return

    def clear_environment(self):
        """Reset environment variables after a test.
        """
        from os import environ
        for k in self.environ_cache:
            if self.environ_cache[k] is None:
                del environ[k]
            else:
                environ[k] = self.environ_cache[k]
        self.environ_cache.clear()
        return

    def test_night(self):
        """Test desispec.scripts.night.
        """
        from ..scripts.night import validate_inputs
        class DummyOptions(object): pass
        options = DummyOptions()
        self.dummy_environment({'DESI_SPECTRO_REDUX': 'foo', 'SPECPROD': 'bar'})
        status = validate_inputs(options)
        self.assertEqual(status, 1)
        statuses = {'foobar': 2, '20170317': 0, '18580101': 3}
        for n in statuses:
            options.night = n
            status = validate_inputs(options)
            self.assertEqual(status, statuses[n])
        self.clear_environment()
        options.night = '20170317'
        status = validate_inputs(options)
        self.assertEqual(status, 4)


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
