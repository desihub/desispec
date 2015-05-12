"""
tests desispec.pipeline.core
"""

import os
import unittest
from uuid import uuid4

from desispec.pipeline.core import runcmd

#- TODO: override log level to quiet down error messages that are supposed
#- to be there from these tests
class TestRunCmd(unittest.TestCase):
    
    def test_runcmd(self):
        self.assertEqual(0, runcmd('echo hello > /dev/null'))

    def test_missing_inputs(self):
        cmd = 'echo hello > /dev/null'
        self.assertNotEqual(0, runcmd(cmd, inputs=[uuid4().hex]))

    def test_existing_inputs(self):
        cmd = 'echo hello > /dev/null'
        self.assertEqual(0, runcmd(cmd, inputs=[self.infile]))

    def test_missing_outputs(self):
        cmd = 'echo hello > /dev/null'
        self.assertNotEqual(0, runcmd(cmd, outputs=[uuid4().hex]))

    def test_existing_outputs(self):
        token = uuid4().hex
        cmd = 'echo {} > {}'.format(token, self.testfile)
        self.assertEqual(0, runcmd(cmd, outputs=[self.outfile]))
        fx = open(self.testfile)
        line = fx.readline().strip()
        #- command should not have run, so tokens should not be equal
        self.assertNotEqual(token, line)

    def test_clobber(self):
        token = uuid4().hex
        cmd = 'echo {} > {}'.format(token, self.testfile)
        self.assertEqual(0, runcmd(cmd, outputs=[self.outfile], clobber=True))
        fx = open(self.testfile)
        line = fx.readline().strip()
        #- command should have run, so tokens should be equal
        self.assertEqual(token, line)

    def test_zz(self):
        """
        Even if clobber=False and outputs exist, run cmd if inputs are
        newer than outputs.  Run this test last since it alters timestamps.
        """
        #- update input timestamp to be newer than output
        fx = open(self.infile, 'w')
        fx.write('This file is leftover from a test; you can remove it\n')
        fx.close()
        
        #- run a command
        token = uuid4().hex
        cmd = 'echo {} > {}'.format(token, self.testfile)
        self.assertEqual(0, runcmd(cmd, outputs=[self.outfile], clobber=False))

        #- command should have run even though outputs exist,
        #- so tokens should be equal
        fx = open(self.testfile)
        line = fx.readline().strip()        
        self.assertNotEqual(token, line)

    @classmethod
    def setUpClass(cls):
        cls.infile = 'test-'+uuid4().hex
        cls.outfile = 'test-'+uuid4().hex
        cls.testfile = 'test-'+uuid4().hex
        for filename in [cls.infile, cls.outfile]:
            fx = open(filename, 'w')
            fx.write('This file is leftover from a test; you can remove it\n')
            fx.close()

    @classmethod
    def tearDownClass(cls):
        for filename in [cls.infile, cls.outfile, cls.testfile]:
            if os.path.exists(filename):
                os.remove(filename)

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           
