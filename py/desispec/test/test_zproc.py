"""
Test desispec.scripts.zproc, sort of

This doesn't actually test zproc.main itself (too data/CPU intensive),
but it does test a helper function.
"""

import os
import time
import unittest

import numpy as np

from desispec import util
import desispec.parallel as dpl

class TestZProc(unittest.TestCase):
    
    def test_distribute_ranks_to_blocks(self):
        from desispec.scripts.zproc import distribute_ranks_to_blocks as dist

        #- test sizes evenly divisible by nblocks, and one more and one less
        #- and sizes <= nblocks
        nblocks_requested = 3
        for size in (1,2,3,8,9,10,64):
            nblocks_possible = min(nblocks_requested, size)
            print(f'{size=} {nblocks_requested=} {nblocks_possible=}')
            block_ranks = dict()
            block_sizes = dict()
            for rank in range(size):
                nblocks_actual, block_size, block_rank, block_num = dist(nblocks_requested, rank=rank, size=size)
                self.assertEqual(nblocks_possible, nblocks_actual)
                print(f'    {rank=} -> {block_num=} {block_rank=} {block_size=}')
                self.assertLess(block_num, nblocks_possible, f'{rank=} assigned {block_num=} >= {nblocks_possible=}')

                if block_num not in block_ranks:
                    block_ranks[block_num] = [block_rank,]
                else:
                    block_ranks[block_num].append(block_rank)

                if block_num not in block_sizes:
                    block_sizes[block_num] = block_size
                else:
                    self.assertEqual(block_size, block_sizes[block_num], 'Inconsistent block_size for different ranks')

            #- are the actually number of blocks represented across ranks?
            self.assertEqual(len(block_ranks), nblocks_possible)

            #- are the ranks assigend within each block actually unique and contiguous?
            for block_num in block_ranks:
                self.assertEqual(set(block_ranks[block_num]), set(range(block_sizes[block_num])))
