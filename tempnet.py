# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:49:39 2015

@author: Ingo Scholtes
"""

import igraph    
import collections

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
                                

                                
                                # Create weighted two-path tuple
                                two_path = (s,v,d, float(1)/(indeg_v*outdeg_v))
                                self.twopaths.append(two_path)
            prev_t = t
        
        # Update count of two-paths 
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
        
        g2n = igraph.Graph(directed=True)
        g2n.vs["name"] = []
        
        #TODO: Built actual second order null model
        pass
        
        return g2n

        
    def exportMovie(self, filename):
        """Exports an animated movie showing the temporal
           evolution of the network"""
        #TODO: Write code to generate movie frames using igraph
        pass