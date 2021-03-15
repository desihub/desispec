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
        if use_mpi():
            import mpi4py.MPI as MPI
            comm = MPI.COMM_WORLD
            nproc = comm.size
            rank = comm.rank

        ret = take_turns(comm, 2, fake_func, "turns", rank)

        assert(ret == "turns_{}".format(rank))

    def test_weighted_partion(self):
        """test desispec.parallel.weighted_partition"""
        weights = np.arange(1,7)

        #- [1,2,3,4,5,6] split x3 should result in perfectly balanced weights
        groups = weighted_partition(weights, 3)
        self.assertEqual(len(groups), 3)
        sumweights = np.array([np.sum(weights[ii]) for ii in groups])
        self.assertTrue(np.all(sumweights == 7))
        
        #- also if shuffled
        np.random.shuffle(weights)
        groups = weighted_partition(weights, 3)
        self.assertEqual(len(groups), 3)
        sumweights = np.array([np.sum(weights[ii]) for ii in groups])
        self.assertTrue(np.all(sumweights == 7))

        #- split in 4 gives close balance
        groups = weighted_partition(weights, 4)
        self.assertEqual(len(groups), 4)
        sumweights = sorted([np.sum(weights[ii]) for ii in groups])
        self.assertEqual(sumweights, [5,5,5,6])

        #- split in 2 gives close balance
        groups = weighted_partition(weights, 2)
        self.assertEqual(len(groups), 2)
        sumweights = sorted([np.sum(weights[ii]) for ii in groups])
        self.assertEqual(sumweights, [10,11])

    def test_weighted_partition_nodes(self):
        """test weighted_partition spreading across nodes"""
        
        #- 28 tasks, 3 of which are bigger than the others
        weights = np.ones(28)
        weights[0:3] = 3

        #- spread these across 12 workers on 3 nodes (4 workers per node)
        num_groups = 12
        num_nodes = 3
        groups_per_node = num_groups // num_nodes

        def check_weight_distribution(weights, groups, num_nodes, groups_per_node):
            bigweight = np.max(weights)
            for i in range(num_nodes):
                nbig = 0
                for j in range(i*groups_per_node, (i+1)*groups_per_node):
                    if j < len(groups):
                        largest_weight = np.max(weights[groups[j]])
                        if largest_weight == bigweight:
                            nbig += 1
            
            #- Node i should only have one weight==3
            self.assertEqual(nbig, 1, 'Node {} had {} groups with big tasks'.format(i, nbig))

        groups = weighted_partition(weights, num_groups, groups_per_node=groups_per_node)
        check_weight_distribution(weights, groups, num_nodes, groups_per_node)

        #- Should also work if the groups don't fill the nodes
        groups = weighted_partition(weights, num_groups-1, groups_per_node=groups_per_node)
        check_weight_distribution(weights, groups, num_nodes, groups_per_node)

        groups = weighted_partition(weights, num_groups-2, groups_per_node=groups_per_node)
        check_weight_distribution(weights, groups, num_nodes, groups_per_node)

        groups = weighted_partition(weights, num_groups-3, groups_per_node=groups_per_node)
        check_weight_distribution(weights, groups, num_nodes, groups_per_node)

def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
