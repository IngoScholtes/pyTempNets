import unittest
import pyTempNet as tn

class TrivialNetwork( unittest.TestCase ):
    def setUp(self):
        self.order = 3
        self.delta = 2
        self.t2 = tn.TemporalNetwork()
        self.t2.addEdge('a', 'b', 1)
        self.t2.addEdge('b', 'd', 2)
        self.t2.addEdge('d', 'e', 3)

        self.h2 = tn.AggregateNetwork(self.t2, self.order, self.delta)

    def test_order(self):
        self.assertEqual( self.h2.order(), self.order )

    def test_maxTimeDiff(self):
        self.assertEqual( self.h2.maxTimeDiff(), self.delta )

    def test_kPath(self):
        kpaths = self.h2.kPaths()
        solution = [{'nodes': ('a', 'b', 'd', 'e'), 'weight': 1.0}]
        self.assertEqual(kpaths, solution)

    def test_kPathCount(self):
        kpaths = self.h2.kPaths()
        kPathCount = self.h2.kPathCount()
        self.assertEqual( kPathCount, len(kpaths) )
