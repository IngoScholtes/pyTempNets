import unittest
import pyTempNet as tn
import igraph

class TrivialThirdOrderNet( unittest.TestCase ):
    def setUp( self ):
        self.order = 3
        self.delta = 25
        self.tempNet = tn.TemporalNetwork()
        self.tempNet.addEdge('a', 'b', 1)
        self.tempNet.addEdge('x', 'b', 2)
        self.tempNet.addEdge('b', 'c', 3)
        self.tempNet.addEdge('b', 'd', 4)
        self.tempNet.addEdge('d', 'e', 5)
        self.tempNet.addEdge('d', 'f', 6)

        self.h3 = tn.HigherOrderNetwork(self.tempNet, self.order, maxTimeDiff=self.delta)
        
    def test_threePathAndPrintThirdOrderNetwork(self):
        kpaths = self.h3.extractKPaths()
	network = self.h3.igraphThirdOrder()

	visual_style = {}
	visual_style["bbox"] = (600, 400)
	visual_style["margin"] = 60
	visual_style["vertex_size"] = 80
	visual_style["vertex_label_size"] = 24
	visual_style["vertex_color"] = "lightblue"
	visual_style["edge_curved"] = 0.2
	visual_style["edge_width"] = 1
	visual_style["edge_arrow_size"] = 2
	
	visual_style["layout"] = network.layout_auto()
	visual_style["vertex_label"] = network.vs["name"]
	visual_style["edge_label"] = network.es["weight"]
	igraph.plot(network, '../output/G3_test.pdf', **visual_style)

        solution = [{'nodes': ['a', 'b', 'd', 'e'], 'weight': 1.0}]
        self.assertEqual(kpaths, solution)
