# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 11:59:22 2015

@author: Ingo
"""

import pyTempNet as tn
import igraph
import pkg_resources

# Set up a canonical example network in order to make sure that everything 
# is calculated correctly
t = tn.TemporalNetwork()
t.addEdge("c", "e", 1);
t.addEdge("e", "f", 2);

t.addEdge("a", "e", 3);
t.addEdge("e", "g", 4);

t.addEdge("c", "e", 5);
t.addEdge("e", "f", 6);

t.addEdge("a", "e", 7);
t.addEdge("e", "g", 8);

t.addEdge("c", "e", 9);
t.addEdge("e", "f", 10);

# Note that the next added edge additionally continues a two-path e -> f -> e
t.addEdge("f", "e", 11);
t.addEdge("e", "b", 12);

# An additional edge that should be filtered during preprocessing ...
t.addEdge("e", "b", 13);

# And one case where we have multiple edges in a single time step
t.addEdge("g", "e", 14);
t.addEdge("c", "e", 14);
t.addEdge("e", "f", 15);

t.addEdge("b", "e", 16);
t.addEdge("e", "g", 17);

t.addEdge("c", "e", 18);
t.addEdge("e", "f", 19);

t.addEdge("c", "e", 20);
t.addEdge("e", "f", 21);

print("Test network has", t.TwoPathCount(), "two-paths")

# Plot the three aggregate networks
g1 = t.igraphFirstOrder()

# Compute betweenness preference of nodes
bw = tn.Measures.BetweennessPreference(t, v='e')

print("Betweenness pref. of node e =", bw)

assert bw == 1.2954618442383219

shuffled = t.ShuffleEdges(l=10000)
bw = tn.Measures.BetweennessPreference(shuffled, v='e')
print("Betweenness pref. of node e in edge-shuffled network =", bw)

shuffled = t.ShuffleTwoPaths(l=10000)
bw = tn.Measures.BetweennessPreference(shuffled, v='e')
print("Betweenness pref. of node e in two-path-shuffled network =", bw)


visual_style = {}
visual_style["layout"] = g1.layout_auto()
visual_style["vertex_size"] = 30
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_label"] = g1.vs["name"]
visual_style["edge_curved"] = 0.2
visual_style["edge_width"] = 0.2
visual_style["edge_arrow_size"] = 0.5
visual_style["edge_label"] = g1.es["weight"]
igraph.plot(g1, **visual_style)

g2 = t.iGraphSecondOrder()
visual_style["edge_label"] = str(g2.es["weight"])
visual_style["layout"] = g2.layout_auto()
visual_style["vertex_label"] = g2.vs["name"]
visual_style["edge_label"] = g2.es["weight"]
igraph.plot(g2, **visual_style)

g2n = t.iGraphSecondOrderNull()
visual_style["edge_label"] = str(g2n.es["weight"])
visual_style["layout"] = g2n.layout_auto()
visual_style["vertex_label"] = g2n.vs["name"]
visual_style["edge_label"] = g2n.es["weight"]
igraph.plot(g2n, **visual_style)

# Read temporal network from sample data file
filename = pkg_resources.resource_filename('pyTempNet', 'example.tedges')

t = tn.TemporalNetwork.readFile(filename, sep=' ')
print("Temporal network has", t.vcount(), "nodes")
print("Temporal network has", t.ecount(), "time-stamped edges")

# If the following explicit call is omitted, the two-path extraction will be 
# executed whenever the time-respecting paths are needed the first time
print("Extracting two-paths ...")
t.extractTwoPaths()
print("Temporal network has", t.TwoPathCount(), "two-paths")

g1 = t.igraphFirstOrder()
print("First-order aggregate network has", len(g1.vs), "nodes and", len(g1.es), "edges")

g2 = t.igraphSecondOrder().components(mode="STRONG").giant()
print("Second-order aggregate network has", len(g2.vs), "nodes and", len(g2.es), "edges")

g2n = t.igraphSecondOrderNull().components(mode="STRONG").giant()
print("Second-order null aggregate network has", len(g2n.vs), "nodes and", len(g2n.es), "edges")

print("Slow-down factor for diffusion is", tn.Measures.SlowDownFactor(t))

print("Entropy growth rate ratio is", tn.Measures.EntropyGrowthRateRatio(t))


# Test with actual data set 
t = tn.TemporalNetwork.readFile('manufacturing_30d_agg_3600_scc.tedges', sep=' ')
print("Temporal network has", t.vcount(), "nodes")
print("Temporal network has", t.ecount(), "time-stamped edges")

print("Slow-down factor for diffusion is", tn.Measures.SlowDownFactor(t))
print("Entropy growth rate ratio is", tn.Measures.EntropyGrowthRateRatio(t))

t = tn.TemporalNetwork.readFile('RealityMining_agg_300s_scc.tedges', sep=' ')
print("Temporal network has", t.vcount(), "nodes")
print("Temporal network has", t.ecount(), "time-stamped edges")

print("Slow-down factor for diffusion is", tn.Measures.SlowDownFactor(t))
print("Entropy growth rate ratio is", tn.Measures.EntropyGrowthRateRatio(t))