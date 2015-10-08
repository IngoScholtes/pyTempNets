import unittest
import pyTempNet as tn

class TrivialThreePath( unittest.TestCase ):
    def setUp(self):
        self.order = 3
        self.delta = 1
        self.t2 = tn.TemporalNetwork()
        self.t2.addEdge('a', 'b', 1)
        self.t2.addEdge('b', 'd', 2)
        self.t2.addEdge('d', 'e', 3)

        self.h2 = tn.HigherOrderNetwork(self.t2, self.order, self.delta)
        
    def test_threePath(self):
        kpaths = self.h2.extractKPaths()
        solution = [{'nodes': ['a', 'b', 'd', 'e'], 'weight': 1.0}]
        self.assertEqual(kpaths, solution)

class ThreePathWithLargeDelta( unittest.TestCase ):
    def setUp( self ):
        self.order = 3
        self.delta = 25
        self.t3 = tn.TemporalNetwork()
        self.t3.addEdge('a', 'b', 1)
        self.t3.addEdge('b', 'c', 2)
        self.t3.addEdge('b', 'd', 3)
        self.t3.addEdge('d', 'e', 8)
        self.t3.addEdge('b', 'f', 24)
        self.t3.addEdge('f', 'g', 25)
        self.h3 = tn.HigherOrderNetwork(self.t3, self.order, maxTimeDiff=self.delta)
        
    def test_threePath(self):
        kpaths = self.h3.extractKPaths()
        solution = [{'nodes': ['a', 'b', 'd', 'e'], 'weight': 1.0}, {'nodes': ['a', 'b', 'f', 'g'], 'weight': 1.0}]
        self.assertEqual( kpaths, solution )

class CrossingThreePaths( unittest.TestCase ):
    def setUp( self ):
        self.order = 3
        self.delta = 25
        self.t4 = tn.TemporalNetwork()
        self.t4.addEdge('a', 'b', 1)
        self.t4.addEdge('x', 'b', 2)
        self.t4.addEdge('b', 'c', 3)
        self.t4.addEdge('b', 'd', 4)
        self.t4.addEdge('d', 'e', 5)
        self.t4.addEdge('d', 'f', 6)

        self.h4 = tn.HigherOrderNetwork(self.t4, self.order, maxTimeDiff=self.delta)
        
    def test_threePath(self):
        kpaths = self.h4.extractKPaths()
        solution = [{'nodes': ['a', 'b', 'd', 'e'], 'weight': 0.25}, {'nodes': ['x', 'b', 'd', 'e'], 'weight': 0.25}, {'nodes': ['a', 'b', 'd', 'f'], 'weight': 0.25}, {'nodes': ['x', 'b', 'd', 'f'], 'weight': 0.25}]
        self.assertEqual( kpaths, solution )


#class WeightCummulationTest( unittest.TestCase ):
    #def setUp( self ):
        #self.order = 3
        #self.delta = 25
        #self.t4 = tn.TemporalNetwork()
        #self.t4.addEdge('a', 'b', 1)
        #self.t4.addEdge('x', 'b', 2)
        #self.t4.addEdge('b', 'c', 3)
        #self.t4.addEdge('b', 'd', 4)
        #self.t4.addEdge('d', 'e', 5)
        #self.t4.addEdge('d', 'f', 6)

        #self.t4.addEdge('x', 'z', 22)
        #self.t4.addEdge('z', 'd', 24)
        #self.t4.addEdge('d', 'e', 25)
        #self.t4.addEdge('d', 'f', 26)

        #self.h4 = tn.HigherOrderNetwork(self.t4, self.order, maxTimeDiff=self.delta)
        
    #def test_threePath(self):
        #kpaths = self.h4.extractKPaths()
        ## TODO find the solution for this test case by hand
        #solution = 
        #self.assertEqual( kpaths, solution )
