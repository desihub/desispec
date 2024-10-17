"""
Example test that can be used as a starting point for other tests

run these with pytest py/desispec/test/test_example.py
"""

import unittest

class TestExample(unittest.TestCase):

    # setUpClass runs once at the start before any tests
    @classmethod
    def setUpClass(cls):
        cls.blat = 1

    # setUpClass runs once at the end after every test
    # e.g. to remove files created by setUpClass
    @classmethod
    def setUpClass(cls):
        cls.blat = 1

    # setUp runs before every test, e.g. to reset state
    def setUp(self):
        self.foo = 2

    # setUp runs after every test, e.g. to reset state
    def tearDown(self):
        pass

    def test_blat(self):
        self.assertEqual(self.blat, 1)

    def test_foo(self):
        self.assertEqual(self.foo, 2)
        self.foo *= 2
        self.assertEqual(self.foo, 4)

    def test_foo_again(self):
        #- even though test_foo changed self.foo, self.setUp() should reset it
        self.assertEqual(self.foo, 2)
        self.foo *= 2
        self.assertEqual(self.foo, 4)

