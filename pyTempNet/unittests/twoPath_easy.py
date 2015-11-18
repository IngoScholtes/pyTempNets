import unittest
import pyTempNet as tn

class EasyTwoPathExample( unittest.TestCase ):
    def setUp(self):
        self.t1 = tn.TemporalNetwork()
        self.t1.addEdge('a', 'c', 1)
        self.t1.addEdge('c', 'e', 2)

        self.t1.addEdge('b', 'c', 3)
        self.t1.addEdge('c', 'd', 4)

        self.t1.addEdge('x', 'y', 24)

        self.t1.addEdge('b', 'c', 5)
        self.t1.addEdge('c', 'e', 6)

        self.t1.addEdge('a', 'c', 7)
        self.t1.addEdge('c', 'd', 8)
        
        self.h1 = tn.AggregateNetwork(self.t1, 2)
        
    def test_kpaths(self):
        kpaths   = self.h1.kPaths()
        x = [{'nodes': ('a', 'c', 'e'), 'weight': 1.0},
             {'nodes': ('b', 'c', 'd'), 'weight': 1.0},
             {'nodes': ('b', 'c', 'e'), 'weight': 1.0},
             {'nodes': ('a', 'c', 'd'), 'weight': 1.0}]
        self.assertEqual(x, kpaths)