# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:49:39 2015

@author: Ingo Scholtes
"""

import igraph    
import collections
import datetime as dt
import time
import numpy as np

class TemporalNetwork:
    """A class representing a temporal network"""
    
    def __init__(self, tedges = None):
        """Constructor generating an empty temporal network"""
        self.tedges = []
        self.nodes = []
        if tedges is not None:
            for e in tedges:
                self.tedges.append(e)            
        for e in self.tedges:
            source = e[0]
            target = e[1]
            if source not in self.nodes:
                self.nodes.append(source)
            if target not in self.nodes:
                self.nodes.append(target)        
        self.twopaths = []
        self.twopathsByNode = {}
        self.tpcount = -1
        
        
    def readFile(filename, sep=',', fformat="TEDGE", timestampformat="%s"):
        """ Reads time-stamped edges from TEDGE or TRIGRAM file. If fformat is TEDGES,
            the file is expected to contain lines in the format 'v,w,t' each line 
            representing a directed time-stamped link from v to w at time t.
            Semantics of columns should be given in a header file indicating either 
            'node1,node2,time' or 'source,target,time' (in arbitrary order).
            If fformat is TRIGRAM the file is expected to contain lines in the format
            'u,v,w' each line representing a time-respecting path (u,v) -> (v,w) consisting 
            of two consecutive links (u,v) and (v,w). Timestamps can be integer numbers or
            string timestamps (in which case the timestampformat string is used for parsing)
        """
        
        assert filename is not ""
        
        f = open(filename, 'r')
        tedges = []
        
        header = f.readline()
        header = header.split(sep)
        
        # Support for arbitrary column ordering
        time_ix = -1
        source_ix = -1
        target_ix = -1        
        for i in range(len(header)):
            if header[i] == 'node1' or header[i] == 'source':
                source_ix = i
            elif header[i] == 'node2' or header[i] == 'target':
                target_ix = i
            elif header[i] == 'time':
                time_ix = i
        
        # Read time-stamped edges
        line = f.readline()
        while not line is '':
            fields = line.rstrip().split(sep)
            if fformat =="TEDGE":
                timestamp = fields[time_ix]
                if (timestamp.isdigit()):
                    t = int(timestamp)
                else:
                    x = dt.datetime.strptime(timestamp, "%Y-%m-%d %H:%M")
                    t = int(time.mktime(x.timetuple()))

                tedge = (fields[source_ix], fields[target_ix], t)
                tedges.append(tedge)
            elif fformat =="TRIGRAM":
                # TODO: Add support for trigram files
                pass
            line = f.readline()
        return TemporalNetwork(tedges)



    def addEdge(self, source, target, time):
        """Adds a (directed) time-stamped edge to the temporal network"""
        
        self.tedges.append((source, target, time))
        if source not in self.nodes:
            self.nodes.append(source)
        if target not in self.nodes:
            self.nodes.append(target)
            
        self.tpcount = -1


        
    def vcount(self):
        """Returns the number of vertices in the static network"""
        return len(self.nodes)

        
    def ecount(self):
        """Returns the number of time-stamped edges"""
        return len(self.tedges)


    def extractTwoPaths(self, delta=1):
        """Extracts all time-respecting paths of length two. The delta parameter indicates the maximum
        temporal distance below which two consecutive links will be considered as a time-respecting path.
        For (u,v,3) and (v,w,7) a time-respecting path (u,v)->(v,w) will be inferred for all delta < 4, 
        while no time-respecting path will be inferred for all delta >= 4.
        """
        
        self.twopaths = []
        
        # An index structure to quickly access tedges by time
        time = collections.OrderedDict()
        for e in self.tedges:
            if not e[2] in time:
                time[e[2]] = []
            time[e[2]].append(e)

        # An index structure to quickly access tedges by time and target/source
        targets = collections.OrderedDict()
        sources = collections.OrderedDict()
        for e in self.tedges:
            if not e[2] in targets:
                targets[e[2]] = dict()
            if not e[2] in sources:
                sources[e[2]] = dict()
            if not e[1] in targets[e[2]]:
                targets[e[2]][e[1]] = []
            if not e[0] in sources[e[2]]:
                sources[e[2]][e[0]] = []
            targets[e[2]][e[1]].append(e)
            sources[e[2]][e[0]].append(e)

        # Extract time-respecting paths of length two             
        prev_t = -1
        for t in time:
            if prev_t ==-1: 
                pass
            elif prev_t < t-delta:
                pass
            else:
                for v in targets[prev_t]:
                    if v in sources[t]:
                        for e_out in sources[t][v]:
                            for e_in in targets[prev_t][v]:
                                s = e_in[0]
                                d = e_out[1]
                                
                                # TODO: Add support for weighted time-
                                # stamped links
                                pass
                                indeg_v = len(targets[prev_t][v])
                                outdeg_v = len(sources[t][v])                                
                                
                                # Create a weighted two-path tuple
                                # (s, v, d, weight)
                                # representing (s,v) -> (v,d)
                                two_path = (s,v,d, float(1)/(indeg_v*outdeg_v))
                                self.twopaths.append(two_path)
                                if not v in self.twopathsByNode:
                                    # Generate dictionary indexed by time stamps
                                    self.twopathsByNode[v] = {}
                                if not t in self.twopathsByNode[v]:
                                    # Generate list taking all two paths through node v at time t
                                    self.twopathsByNode[v][t] = []
                                self.twopathsByNode[v][t].append(two_path)
            prev_t = t
        
        # Update the count of two-paths
        self.tpcount = len(self.twopaths)


        
    def TwoPathCount(self):
        """Returns the total number of time-respecting paths of length two which have
            been extracted from the time-stamped edge sequence."""
        
        # If two-paths have not been extracted yet, do it now
        if self.tpcount == -1:
            self.extractTwoPaths()

        return self.tpcount
    
    
    
    def iGraphFirstOrder(self):
        """Returns the first-order time-aggregated network
           corresponding to this temporal network. This network corresponds to 
           a first-order Markov model reproducing the link statistics in the 
           weighted, time-aggregated network."""
           
        # If two-paths have not been extracted yet, do it now
        if self.tpcount == -1:
            self.extractTwoPaths()

        g1 = igraph.Graph(directed=True)
        
        for v in self.nodes:
            g1.add_vertex(str(v))

        # We first keep multiple (weighted) edges
        for tp in self.twopaths:
            g1.add_edge(str(tp[0]), str(tp[1]), weight=tp[3])
            g1.add_edge(str(tp[1]), str(tp[2]), weight=tp[3])
            
        # We then collapse them, while summing their weights
        g1 = g1.simplify(combine_edges="sum")
        return g1


        
    def iGraphSecondOrder(self):
        """Returns the second-order time-aggregated network
           corresponding to this temporal network. This network corresponds to 
           a second-order Markov model reproducing both the link statistics and 
           (first-order) order correlations in the underlying temporal network."""

        if self.tpcount == -1:
            self.extractTwoPaths()        
        
        g2 = igraph.Graph(directed=True)
        g2.vs["name"] = []
        for tp in self.twopaths:            
            n1 = str(tp[0])+";"+str(tp[1])
            n2 = str(tp[1])+";"+str(tp[2])
            if not n1 in g2.vs["name"]:
                g2.add_vertex(name=n1)
            if not n2 in g2.vs["name"]:
                g2.add_vertex(name=n2)
            if not g2.are_connected(n1, n2):
                g2.add_edge(n1, n2, weight=tp[3])
        return g2


        
    def iGraphSecondOrderNull(self):
        """Returns a second-order null Markov model 
           corresponding to the first-order aggregate network. This network
           is a second-order representation of the weighted time-aggregated network."""

        g1 = self.iGraphFirstOrder()
        
        D = g1.strength(mode="out", weights=g1.es["weight"])
        
        g2n = igraph.Graph(directed=True)
        g2n.vs["name"] = []
        
        for e1 in g1.es:
            for e2 in g1.es:
                if e1.target == e2.source:
                    a = e1.source
                    b = e1.target
                    c = e2.target
                    n1 = g1.vs[a]["name"]+";"+g1.vs[b]["name"]
                    n2 = g1.vs[b]["name"]+";"+g1.vs[c]["name"]
                    if n1 not in g2n.vs["name"]:
                        g2n.add_vertex(name=n1)
                    if n2 not in g2n.vs["name"]:
                        g2n.add_vertex(name=n2)
                    # Compute expected weight                        
                    w = 0.5 * g1[a,b] * g1[b,c] / D[b]
                    if w>0 and not D[b]==0: # and not g2n.are_connected(n1, n2)
                        g2n.add_edge(n1, n2, weight = w)
        return g2n

    def ShuffleEdges(self, l=0):        
        tedges = []
        
        if l==0:
            l = len(self.tedges)
        for i in range(l):
            edge = self.tedges[np.random.randint(len(self.tedges))]
            tedges.append((edge[0], edge[1], i))
        tn = TemporalNetwork(tedges)
        return tn
        
        
    def ShuffleTwoPaths(self, l=0):
        ""
        tedges = []
        
        t = 0
        
        if l==0:
            l = len(self.twopaths)
        for i in range(int(l/2)):
            tp = self.twopaths[np.random.randint(len(self.twopaths))]
            tedges.append((tp[0], tp[1], t))
            tedges.append((tp[1], tp[2], t+1))
            t+=1
            
        tn = TemporalNetwork(tedges)
        return tn

        
    def exportMovie(self, filename, fps=5, dpi=100):
        """Exports an animated movie showing the temporal
           evolution of the network"""
        
        import matplotlib.animation as anim
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        
        def next_img(n):                        
            g = igraph.Graph.Erdos_Renyi(n=5, m=7)
            igraph.plot(g, "frame.png")
            return plt.imshow(mpimg.imread("frame.png"))
        
        ani = anim.FuncAnimation(fig, next_img, 300, interval=30)
        writer = anim.FFMpegWriter(fps=fps)
        ani.save(filename, writer=writer, dpi=dpi)