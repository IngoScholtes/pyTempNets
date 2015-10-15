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

        self.h2 = tn.HigherOrderNetwork(self.t2, self.order, self.delta)

    def test_clearCacheCount( self ):
        self.h2.clearCache()
        self.assertEqual(self.h2.KPathCount(), -1)
        
    def test_setMaxTimeDiff(self):
        value = 5
        self.h2.setMaxTimeDiff(value)
        self.assertEqual(self.h2.maxTimeDiff(), value)
        
    def test_setMaxTimeDiffZero(self):
        value = 0
        self.assertRaises(ValueError, self.h2.setMaxTimeDiff, value)
    
    def test_setMaxTimeDiffNegative(self):
        value = -5
        self.assertRaises(ValueError, self.h2.setMaxTimeDiff, value)
        
    def test_setOrder(self):
        value = 5
        self.h2.setOrder(value)
        self.assertEqual(self.h2.order(), value)
        
    def test_setOrderZero(self):
        value = 0
        self.assertRaises(ValueError, self.h2.setOrder, value)
    
    def test_setMaxTimeDiffNegative(self):
        value = -5
        self.assertRaises(ValueError, self.h2.setOrder, value)
        
    def test_resetOrder(self):
        self.h2.setOrder(5)
        self.h2.resetOrder()
        self.assertEqual(self.h2.order(), 1)
        
    def test_resetMaxTimeDiff(self):
        self.h2.setMaxTimeDiff(42)
        self.h2.resetMaxTimeDiff()
        self.assertEqual( self.h2.maxTimeDiff(), 1 )
        
    def test_order(self):
        self.assertEqual( self.h2.order(), self.order )
        
    def test_maxTimeDiff(self):
        self.assertEqual( self.h2.maxTimeDiff(), self.delta )
        
    #def test_threePath(self):
        #kpaths = self.h2.extractKPaths()
        #solution = [{'nodes': ['a', 'b', 'd', 'e'], 'weight': 1.0}]
        #self.assertEqual(kpaths, solution)