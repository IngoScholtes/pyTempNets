import unittest
import pyTempNet as tn

class WeightCummulationTest( unittest.TestCase ):
    def setUp( self ):
        self.order = 3
        self.delta = 25
        self.t5 = tn.TemporalNetwork()
        self.t5.addEdge('a', 'b', 1)
        self.t5.addEdge('x', 'b', 2)
        self.t5.addEdge('b', 'c', 3)
        self.t5.addEdge('b', 'd', 4)
        self.t5.addEdge('d', 'e', 5)
        self.t5.addEdge('d', 'f', 6)

        self.t5.addEdge('x', 'z', 22)
        self.t5.addEdge('z', 'd', 24)
        self.t5.addEdge('d', 'e', 25)
        self.t5.addEdge('d', 'f', 26)

        self.h5 = tn.AggregateNetwork(self.t5, self.order, maxTimeDiff=self.delta)
        
    def test_threePath(self):
        kpaths = self.h5.kPaths()
        solution = [{'nodes': ('a', 'b', 'd', 'f'), 'weight': 0.16666666666666666}, 
                    {'nodes': ('x', 'z', 'd', 'e'), 'weight': 0.16666666666666666}, 
                    {'nodes': ('x', 'b', 'd', 'f'), 'weight': 0.16666666666666666},
                    {'nodes': ('a', 'b', 'd', 'e'), 'weight': 0.16666666666666666}, 
                    {'nodes': ('x', 'z', 'd', 'f'), 'weight': 0.16666666666666666}, 
                    {'nodes': ('x', 'b', 'd', 'e'), 'weight': 0.16666666666666666}]
        self.assertEqual( kpaths, solution )
