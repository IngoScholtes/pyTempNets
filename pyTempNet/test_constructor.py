import unittest
import pyTempNet as tn

class TrivialNetwork( unittest.TestCase ):
    def setUp(self):
        self.tempNet = tn.TemporalNetwork()
        self.tempNet.addEdge('a', 'b', 1)
        self.tempNet.addEdge('x', 'b', 2)
        self.tempNet.addEdge('b', 'c', 3)
        self.tempNet.addEdge('b', 'd', 4)
        self.tempNet.addEdge('d', 'e', 5)
        self.tempNet.addEdge('d', 'f', 6)

        self.tempNet.addEdge('x', 'z', 22)
        self.tempNet.addEdge('z', 'd', 24)
        self.tempNet.addEdge('d', 'e', 25)
        self.tempNet.addEdge('d', 'f', 26)

    def test_OrderValueErrorNull(self):
        self.assertRaises(ValueError, tn.AggregateNetwork, self.tempNet, order=0)
        
    def test_DeltaValueErrorNull(self):
        self.assertRaises(ValueError, tn.AggregateNetwork, self.tempNet, 1, maxTimeDiff=0)

    def test_OrderValueErrorMinus(self):
        self.assertRaises(ValueError, tn.AggregateNetwork, self.tempNet, order=-3)
        
    def test_DeltaValueErrorMinus(self):
        self.assertRaises(ValueError, tn.AggregateNetwork, self.tempNet, 1, maxTimeDiff=-3)


