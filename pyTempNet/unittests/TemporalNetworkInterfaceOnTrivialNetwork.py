import unittest
import pyTempNet as tn

class TrivialNetwork( unittest.TestCase ):

    def test_DeltaValueErrorNull(self):
        self.assertRaises(ValueError, tn.TemporalNetwork, maxTimeDiff=0)

    def test_DeltaValueErrorMinus(self):
        self.assertRaises(ValueError, tn.TemporalNetwork, maxTimeDiff=-3)
