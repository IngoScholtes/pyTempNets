import unittest
import pyTempNet as tn

class CrossingThreePaths( unittest.TestCase ):
    def setUp( self ):
        self.order = 3
        self.delta = 25
        self.t4 = tn.TemporalNetwork( maxTimeDiff = self.delta )
        self.t4.addEdge('a', 'b', 1)
        self.t4.addEdge('x', 'b', 2)
        self.t4.addEdge('b', 'c', 3)
        self.t4.addEdge('b', 'd', 4)
        self.t4.addEdge('d', 'e', 5)
        self.t4.addEdge('d', 'f', 6)

        self.h4 = tn.AggregateNetwork(self.t4, self.order)
        
    def test_threePath(self):
        kpaths = self.h4.kPaths()
        solution = [{'nodes': ('a', 'b', 'd', 'e'), 'weight': 0.25}, 
                    {'nodes': ('a', 'b', 'd', 'f'), 'weight': 0.25}, 
                    {'nodes': ('x', 'b', 'd', 'f'), 'weight': 0.25},
                    {'nodes': ('x', 'b', 'd', 'e'), 'weight': 0.25}]
        self.assertEqual( kpaths, solution )