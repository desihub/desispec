# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.scripts.
"""
from __future__ import absolute_import, division
# The line above will help with 2to3 support.

import os
import unittest


class TestScripts(unittest.TestCase):
    """Test desispec.scripts.
    """

    @classmethod
    def setUpClass(cls):
        # from os import environ
        # for k in ('DESI_SPECTRO_REDUX', 'SPECPROD'):
        #     if k in environ:
        #         raise AssertionError("{0}={1} was pre-defined in the environment!".format(k, environ[k]))
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

    def test_delivery(self):
        """Test desispec.scripts.delivery.
        """
        from ..scripts.delivery import parse_delivery
        with self.assertRaises(SystemExit):
            options = parse_delivery([])
        with self.assertRaises(SystemExit):
            options = parse_delivery('filename', '2', '20170317', 'foo')
        with self.assertRaises(SystemExit):
            options = parse_delivery('filename', 'foo', '20170317', 'start')
        options = parse_delivery('filename', '2', '20170317', 'start')
        self.assertEqual(options.filename, 'filename')
        self.assertEqual(options.exposure, 2)
        self.assertEqual(options.night, '20170317')
        self.assertEqual(options.nightStatus, 'start')

    def test_night(self):
        """Test desispec.scripts.night.
        """
        from ..scripts.night import validate_inputs, parse_night
        #
        # Test argument parser.
        #
        with self.assertRaises(KeyError):
            options = parse_night('foo')
        options = parse_night('start', '20170317')
        self.assertEqual(options.stage, 'start')
        self.assertEqual(options.night, '20170317')
        #
        # Test validator.
        #
        class DummyOptions(object): pass
        options = DummyOptions()
        self.dummy_environment({'DESI_SPECTRO_DATA': 'data',
                                'DESI_SPECTRO_REDUX': 'foo',
                                'SPECPROD': 'bar'})
        status = validate_inputs(options)
        self.assertEqual(status, 1)
        statuses = {'foobar': 2, '20170317': 0, '18580101': 3}
        for n in statuses:
            options.night = n
            status = validate_inputs(options)
            self.assertEqual(status, statuses[n])
        self.clear_environment()
        options.night = '20170317'

        #- Test that a missing environment variable causes a validation failure
        orig_DSR = os.getenv('DESI_SPECTRO_REDUX')
        if 'DESI_SPECTRO_REDUX' in os.environ:
            del os.environ['DESI_SPECTRO_REDUX']

        status = validate_inputs(options)
        self.assertEqual(status, 4)
        if orig_DSR is not None:
            os.environ['DESI_SPECTRO_REDUX'] = orig_DSR


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
