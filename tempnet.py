# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:49:39 2015

@author: Ingo Scholtes
"""

import igraph    
import collections
import numpy as np
import scipy.linalg as spl

class TemporalNetwork:
    """A python class for the analysis of 
    temporal networks"""
        
    def __init__(self, tedges = None):
        """Constructor for temporal network instance"""
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
        self.tpcount = -1
        
        
    def readFile(filename, sep=',', fformat="TEDGE"):
        """Reads a tedge file """
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
                tedge = (fields[source_ix], fields[target_ix], int(fields[time_ix]))
                tedges.append(tedge)
            elif fformat =="TRIGRAM":
                # TODO: Add support for trigram files
                pass
            line = f.readline()
        return TemporalNetwork(tedges)
        

    def addEdge(self, source, target, time):
        """Adds a time-stamped edge to the temporal network"""
        
        self.tedges.append((source, target, time))
        if source not in self.nodes:
            self.nodes.append(source)
        if target not in self.nodes:
            self.nodes.append(target)
            
        self.tpcount = -1

        
    def vcount(self):
        """Returns the number of vertices"""
        return len(self.nodes)

        
    def ecount(self):
        """Returns the number of time-stamped edges"""
        return len(self.tedges)


    def extractTwoPaths(self, delta=1):
        """Extracts all time-respecting paths of length two"""
        
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
            prev_t = t
        
        # Update the count of two-paths 
        self.tpcount = len(self.twopaths)

        
    def TwoPathCount(self):
        """Returns the number of two-paths edges"""
        
        # If two-paths have not been extracted yet, do it now
        if self.tpcount == -1:
            self.extractTwoPaths()

        return self.tpcount
    
    
    def iGraphFirstOrder(self):
        """Returns the first-order Markov model
           corresponding to this temporal network"""
           
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
        """Returns the second-order Markov model
           corresponding to this temporal network"""

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
        """Returns the second-order null Markov model 
           corresponding to the first-order aggregate network"""

        g1 = self.iGraphFirstOrder()
        
        D = g1.strength(mode="OUT", weights=g1.es["weight"])
        
        g2n = igraph.Graph(directed=True)
        g2n.vs["name"] = []
        
        for e_in in g1.es:
            for e_out in g1.es:
                if e_in.target == e_out.source:
                    n1 = g1.vs[e_in.source]["name"]+";"+g1.vs[e_in.target]["name"]
                    n2 = g1.vs[e_out.source]["name"]+";"+g1.vs[e_out.target]["name"]
                    if n1 not in g2n.vs["name"]:
                        g2n.add_vertex(name=n1)
                    if n2 not in g2n.vs["name"]:
                        g2n.add_vertex(name=n2)
                    
                    w = 0.5 * e_in["weight"] * e_out["weight"] / D[e_out.source]
                    g2n.add_edge(n1, n2, weight = w)
        return g2n

        
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
    

class Measures:
    """Computes a set of temporal network measures"""
    
    def EntropyGrowthRateRatio(t):
        """Computes the ratio between the entropy growth rate of the 
           second-order model and the first-order model"""
        pass

    def BetweennessPreference(t, v, normalized = False):
        """Computes a list of betweenness preferences of nodes"""
        
        g = t.iGraphFirstOrder()        
        
        # If the network is empty, just return zero
        if len(g.vs) == 0:
            return 0.0
        
        I = 0.0
        
        indeg = g[v].degree(mode="IN")
        outdeg = g[v].degree(mode="OUT")
                
        P = np.zeros(shape=(indeg, outdeg))
        # TODO: compute betweenness preference matrix                
        
        marginal_s = []
        marginal_d = []
        
        # Marginal probabilities P_d = \sum_s'{P_{s'd}}
        for d in range(d):
            P_d = 0.0
            for s_prime in range(indeg):
                P_d += P[s_prime, d]
            marginal_d.append(P_d)
        
        # Marginal probabilities P_s = \sum_d'{P_{sd'}}
        for s in range(indeg):
            P_s = 0.0
            for d_prime in range(outdeg):
                P_s += P[s, d_prime]
            marginal_s.append(P_s)
        
        H_s = Measures.Entropy(marginal_s)
        H_d = Measures.Entropy(marginal_d)
        
        # Here we just compute equation (4) of the paper ... 
        for s in range(indeg):
            for d in range(outdeg):
                if P[s, d] != 0: # 0 * Log(0)  = 0
                    # Compute Mutual information
                    I += P[s, d] * np.log2(P[s, d] / (marginal_s[s] * marginal_d[d]))
        
        if normalized:
            return I/(H_s+H_d)
        else:
            return I
    
    
    def Entropy(prob):        
        H = 0
        for p in prob:
            H = H+np.log2(p)*p
        return -H

    
    def SlowDown(t):
        """Computes the slow-down ratio of the corresponding
           second-order model, compared to a first-order model"""
           
        g2 = t.iGraphSecondOrder().components(mode="STRONG").giant()
        g2n = t.iGraphSecondOrderNull().components(mode="STRONG").giant()
        
        A2 = np.matrix(list(g2.get_adjacency()))
        T2 = np.zeros(shape=(len(g2.vs), len(g2.vs)))
        D2 = np.diag(g2.strength(mode='out', weights=g2.es["weight"]))
        
        for i in range(len(g2.vs)):
            for j in range(len(g2.vs)):
                T2[i,j] = A2[i,j]/D2[i,i]
        
        A2n = np.matrix(list(g2n.get_adjacency()))
        T2n = np.zeros(shape=(len(g2n.vs), len(g2n.vs)))
        D2n = np.diag(g2n.strength(mode='out', weights=g2n.es["weight"]))
        
        for i in range(len(g2n.vs)):
                    for j in range(len(g2n.vs)):
                        T2n[i,j] = A2n[i,j]/D2n[i,i]
        
        w2, v2 = spl.eig(T2, left=True, right=False)
        w2n, v2n = spl.eig(T2n, left=True, right=False)
        
        return np.log(np.abs(w2n[1]))/np.log(np.abs(w2[1]))