# -*- coding: utf-8 -*-
"""
Created on Don Nov  5 17:01:02 CET 2015
@author: Ingo Scholtes

(c) Copyright ETH Zürich, Chair of Systems Design, 2015
"""

import pyTempNet as tn
import unittest

class IngosTest( unittest.TestCase ):
    def setUp( self ):
        self.order = 2
        self.delta = 1
        
        # Set up a canonical example network in order to make sure that everything 
        # is calculated correctly
        self.t = tn.TemporalNetwork()
        self.t.addEdge("c", "e", 1);
        self.t.addEdge("e", "f", 2);

        self.t.addEdge("a", "e", 3);
        self.t.addEdge("e", "g", 4);

        self.t.addEdge("c", "e", 5);
        self.t.addEdge("e", "f", 6);

        self.t.addEdge("a", "e", 7);
        self.t.addEdge("e", "g", 8);

        self.t.addEdge("c", "e", 9);
        self.t.addEdge("e", "f", 10);

        # Note that the next added edge additionally continues a two-path e -> f -> e
        self.t.addEdge("f", "e", 11);
        self.t.addEdge("e", "b", 12);

        # An additional edge that should be filtered during preprocessing ...
        self.t.addEdge("e", "b", 13);

        # And one case where we have multiple edges in a single time step
        self.t.addEdge("g", "e", 14);
        self.t.addEdge("c", "e", 14);
        self.t.addEdge("e", "f", 15);

        self.t.addEdge("b", "e", 16);
        self.t.addEdge("e", "g", 17);

        self.t.addEdge("c", "e", 18);
        self.t.addEdge("e", "f", 19);

        self.t.addEdge("c", "e", 20);
        self.t.addEdge("e", "f", 21);
        
        # TODO first order networks
        # aggregate network of order 1
        #self.a1 = tn.AggregateNetwork( self.t, 1, self.delta )
        # aggregate network of order 2
        self.a2 = tn.AggregateNetwork( self.t, self.order, self.delta )
        
    def test_TwoPathCount( self ):
        count = self.a2.kPathCount()
        print("This network has", count, "two-paths]")
        self.assertEqual( count, 12 )
