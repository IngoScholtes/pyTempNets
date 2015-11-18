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

        self.h2 = tn.AggregateNetwork(self.t2, self.order, self.delta)
        
    def test_threePath(self):
        kpaths = self.h2.kPaths()
        print(kpaths)
        solution = [{'nodes': ('a', 'b', 'd', 'e'), 'weight': 1.0}]
        self.assertEqual(kpaths, solution)