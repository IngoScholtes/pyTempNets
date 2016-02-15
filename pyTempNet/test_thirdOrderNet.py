import pyTempNet as tn
import igraph


order = 3
delta = 25
tempNet = tn.TemporalNetwork()
tempNet.addEdge('a', 'b', 1)
tempNet.addEdge('x', 'b', 2)
tempNet.addEdge('b', 'c', 3)
tempNet.addEdge('b', 'd', 4)
tempNet.addEdge('d', 'e', 5)
tempNet.addEdge('d', 'f', 6)

h3 = tn.HigherOrderNetwork(tempNet, order, maxTimeDiff=delta)
        

kpaths = h3.extractKPaths()
network = h3.igraphThirdOrder()

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
igraph.plot(network, 'G3_test.pdf', **visual_style)

solution = [{'nodes': ['a', 'b', 'd', 'e'], 'weight': 1.0}]
