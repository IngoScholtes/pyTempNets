import unittest
import pyTempNet as tn

class TrivialNetwork( unittest.TestCase ):
    def setUp(self):
        self.tempNet = tn.TemporalNetwork()

    def test_OrderValueErrorNull(self):
        self.assertRaises(ValueError, tn.AggregateNetwork, self.tempNet, order=0)

    def test_DeltaValueErrorNull(self):
        self.assertRaises(ValueError, tn.AggregateNetwork, self.tempNet, 1, maxTimeDiff=0)

    def test_OrderValueErrorMinus(self):
        self.assertRaises(ValueError, tn.AggregateNetwork, self.tempNet, order=-3)

    def test_DeltaValueErrorMinus(self):
        self.assertRaises(ValueError, tn.AggregateNetwork, self.tempNet, 1, maxTimeDiff=-3)

    def test_order(self):
        self.order = 5
        self.h2 = tn.AggregateNetwork( self.tempNet, order=self.order )
        self.assertEqual( self.h2.order(), self.order )

    def test_maxTimeDiff(self):
        self.delta = 2
        self.h2 = tn.AggregateNetwork( self.tempNet, order=2, maxTimeDiff=self.delta )
        self.assertEqual( self.h2.maxTimeDiff(), self.delta )

    def test_firstOrder(self):
        self.order = 1
        self.h2 = tn.AggregateNetwork( self.tempNet, order=1 )
        self.assertEqual(self.h2.kPathCount(), 0)