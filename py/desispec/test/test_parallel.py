"""
tests desispec.parallel.py
"""

import os
import unittest
import shutil
import time
import copy

import numpy as np

# Import all functions from the module we are testing.
from desispec.parallel import *


class TestParallel(unittest.TestCase):


    def setUp(self):

        self.ntask = 100
        self.nworker = 9

        self.unicheck = self.ntask // self.nworker

        self.blocks = []
        for i in range(self.nworker):
            self.blocks.append(7)
            self.blocks.append(3)

        self.blocktot = np.sum(self.blocks)

        self.pipeworkers = 100
        self.npipetasks = 201
        self.pipecheck = 67
        self.taskcheck = 3


    def test_dist(self):

        for id in range(self.nworker):
            uni = dist_uniform(self.ntask, self.nworker, id)
            if id == 0:
                assert(uni[0] == 0)
                assert(uni[1] == self.unicheck + 1)
            else:
                assert(uni[0] == id * self.unicheck + 1)
                assert(uni[1] == self.unicheck)

            disc = dist_discrete(self.blocks, self.nworker, id)
            assert(disc[0] == 2 * id)
            assert(len(disc) == 2)

        # In this case, we have more tasks per worker than workers,
        # so the result should be the same.
        bal = dist_balanced(self.ntask, self.nworker)
        #print(bal)
        for id in range(self.nworker):
            if id == 0:
                assert(bal[id][0] == 0)
                assert(bal[id][1] == self.unicheck + 1)
            else:
                assert(bal[id][0] == id * self.unicheck + 1)
                assert(bal[id][1] == self.unicheck)

        # Now try it with many workers and fewer tasks
        bal = dist_balanced(self.npipetasks, self.pipeworkers)
        #print(bal)
        assert(len(bal) == self.pipecheck)
        off = 0
        for w in bal:
            assert(w[0] == off)
            assert(w[1] == self.taskcheck)
            off += self.taskcheck


    def test_turns(self):

        def fake_func(prefix, rank):
            return "{}_{}".format(prefix, rank)

        comm = None
        nproc = 1
        rank = 0
        if use_mpi:
            import mpi4py.MPI as MPI
            comm = MPI.COMM_WORLD
            nproc = comm.size
            rank = comm.rank

        ret = take_turns(comm, 2, fake_func, "turns", rank)

        assert(ret == "turns_{}".format(rank))


#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
