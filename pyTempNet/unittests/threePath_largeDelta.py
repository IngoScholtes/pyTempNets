import unittest
import pyTempNet as tn

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
        self.h3 = tn.AggregateNetwork(self.t3, self.order, maxTimeDiff=self.delta)
        
    def test_threePath(self):
        kpaths = self.h3.kPaths()
        solution = [{'nodes': ('a', 'b', 'd', 'e'), 'weight': 1.0},
                    {'nodes': ('a', 'b', 'f', 'g'), 'weight': 1.0}]
        self.assertEqual( kpaths, solution )