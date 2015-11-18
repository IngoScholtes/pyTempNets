import unittest
import pyTempNet as tn

class IngosTestCase(unittest.TestCase):
    def setUp(self):
        self.order = 2
        self.delta = 1
        
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

        ## And one case where we have multiple edges in a single time step
        self.t.addEdge("g", "e", 14);
        self.t.addEdge("c", "e", 14);
        self.t.addEdge("e", "f", 15);

        self.t.addEdge("b", "e", 16);
        self.t.addEdge("e", "g", 17);

        self.t.addEdge("c", "e", 18);
        self.t.addEdge("e", "f", 19);

        self.t.addEdge("c", "e", 20);
        self.t.addEdge("e", "f", 21);
        
        self.h = tn.AggregateNetwork(self.t, self.order, self.delta)

    def test_kpaths(self):
        kpaths = self.h.kPaths()
        x = [{'nodes': ('c', 'e', 'f'), 'weight': 1.0},
             {'nodes': ('a', 'e', 'g'), 'weight': 1.0},
             {'nodes': ('c', 'e', 'f'), 'weight': 1.0},
             {'nodes': ('a', 'e', 'g'), 'weight': 1.0}, 
             {'nodes': ('c', 'e', 'f'), 'weight': 1.0},
             {'nodes': ('e', 'f', 'e'), 'weight': 1.0}, 
             {'nodes': ('f', 'e', 'b'), 'weight': 1.0},
             {'nodes': ('g', 'e', 'f'), 'weight': 0.5}, 
             {'nodes': ('c', 'e', 'f'), 'weight': 0.5},
             {'nodes': ('b', 'e', 'g'), 'weight': 1.0},
             {'nodes': ('c', 'e', 'f'), 'weight': 1.0}, 
             {'nodes': ('c', 'e', 'f'), 'weight': 1.0}]
        self.assertListEqual( x, kpaths )


# NOTE: use this to run it standalone as
# NOTE:   python test_kpath.py
# 
# suite = unittest.TestLoader().loadTestsFromTestCase(IngosTestCase)
# unittest.TextTestRunner(verbosity=2).run(suite)
#
# or use python auto-discovery:
#   python -m unittest discover
# which will look for (and execute) every file that matches test*.py in the 
# current folder
# use --start-discovery to change to directory
# or  --pattern         to change to search pattern