"""
tests desispec.pipeline.core
"""

import os
import unittest
import shutil
import time
import numpy as np

import subprocess as sp

from desispec.pipeline.common import *
from desispec.pipeline.graph import *
from desispec.pipeline.plan import *
from desispec.pipeline.run import *
from desispec.util import option_list

import desispec.scripts.pipe_prod as pipe_prod

import desispec.io as io

from desispec.log import get_logger

from . import pipehelpers as ph




class TestPipelineRun(unittest.TestCase):

    def setUp(self):
        self.prod = "test"
        self.shifter = "docker:tskisner/desipipe:latest"
        self.raw = ph.fake_raw()
        self.redux = ph.fake_redux(self.prod)
        # (dummy value for DESIMODEL)
        self.model = ph.fake_redux(self.prod)
        ph.fake_env(self.raw, self.redux, self.prod, self.model)

    def tearDown(self):
        if os.path.exists(self.raw):
            shutil.rmtree(self.raw)
        if os.path.exists(self.redux):
            shutil.rmtree(self.redux)


    def test_run(self):
        opts = {}
        opts["spectrographs"] = "0"
        opts["data"] = self.raw
        opts["redux"] = self.redux
        opts["prod"] = self.prod
        opts["model"] = self.model
        opts["shifter"] = self.shifter
        sopts = option_list(opts)
        sargs = pipe_prod.parse(sopts)
        pipe_prod.main(sargs)
        
        # modify the options to use our No-op worker
        rundir = io.get_pipe_rundir()
        optfile = os.path.join(rundir, "options.yaml")

        opts = {}
        for step in step_types:
            opts["{}_worker".format(step)] = "Noop"
            opts["{}_worker_opts".format(step)] = {}
            opts[step] = {}
        yaml_write(optfile, opts)

        envfile = os.path.join(rundir, "env.sh")
        with open(envfile, "w") as f:
            f.write("export DESIMODEL={}\n".format(rundir))
            if "PATH" in os.environ:
                f.write("export PATH={}\n".format(os.environ["PATH"]))
            if "PYTHONPATH" in os.environ:
                f.write("export PYTHONPATH={}\n".format(os.environ["PYTHONPATH"]))
            if "LD_LIBRARY_PATH" in os.environ:
                f.write("export LD_LIBRARY_PATH={}\n".format(os.environ["LD_LIBRARY_PATH"]))

        com = ". {}; eval {}".format(envfile, os.path.join(rundir, "scripts", "run_shell_all.sh"))
        print(com)
        sp.call(com, shell=True, env=os.environ.copy())


#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           
