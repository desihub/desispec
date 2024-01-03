"""
test desispec.io.*
"""

import unittest

import numpy as np
import numpy.random
from desispec.linalg import cholesky_solve
from desispec.linalg import cholesky_solve_and_invert
from desispec.linalg import cholesky_invert

class TestLinalg(unittest.TestCase):
    
    #- Create unique test filename in a subdirectory
    def setUp(self):
        pass
                    
    def test_cholesky_solve(self): 
        # create a random positive definite matrix A
        n = 12
        A = np.zeros((n,n))                
        for i in range(n) :
            H = numpy.random.random(n)
            A += np.outer(H,H.T)
        # random X
        X = numpy.random.random(n)
        # compute B
        B = A.dot(X)
        # solve for X given A and B
        Xs=cholesky_solve(A,B)
        # compute diff
        delta=Xs-X
        d=np.inner(delta,delta)
        self.assertAlmostEqual(d,0.)
        
    def test_cholesky_solve_and_invert(self): 
        # create a random positive definite matrix A
        n = 12
        A = np.zeros((n,n))                
        for i in range(n) :
            H = numpy.random.random(n)
            A += np.outer(H,H.T)
        # random X
        X = numpy.random.random(n)
        # compute B
        B = A.dot(X)
        # solve for X given A and B
        Xs,Ai=cholesky_solve_and_invert(A,B)
        # checck inverse
        Id=A.dot(Ai)
        # test some non-diagonal elements
        self.assertAlmostEqual(Id[0,1],0.)
        self.assertAlmostEqual(Id[1,0],0.)
        self.assertAlmostEqual(Id[0,-1],0.)
        self.assertAlmostEqual(Id[-1,0],0.)
        # test diagonal
        Iddiag=np.diag(Id)
        delta=np.diag(Id)-np.ones((n))
        d=np.inner(delta,delta)
        self.assertAlmostEqual(d,0.)
        
    def test_cholesky_invert(self): 
        # create a random positive definite matrix A
        n = 12
        A = np.zeros((n,n))                
        for i in range(n) :
            H = numpy.random.random(n)
            A += np.outer(H,H.T)
        Ai=cholesky_invert(A)
        # checck inverse
        Id=A.dot(Ai)
        # test some non-diagonal elements
        self.assertAlmostEqual(Id[0,1],0.)
        self.assertAlmostEqual(Id[1,0],0.)
        self.assertAlmostEqual(Id[0,-1],0.)
        self.assertAlmostEqual(Id[-1,0],0.)
        # test diagonal
        Iddiag=np.diag(Id)
        delta=np.diag(Id)-np.ones((n))
        d=np.inner(delta,delta)
        self.assertAlmostEqual(d,0.)
        
        
                
    def runTest(self):
        pass
