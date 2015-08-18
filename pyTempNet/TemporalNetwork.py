# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:49:39 2015
@author: Ingo Scholtes, Roman Cattaneo

(c) Copyright ETH Zürich, Chair of Systems Design, 2015
"""

import igraph
import numpy as np
from collections import defaultdict

from bisect import bisect_right

from pyTempNet.Utilities import RWTransitionMatrix
from pyTempNet.Utilities import StationaryDistribution

class TemporalNetwork:
    """A class representing a temporal network consisting of a sequence of time-stamped edges"""
    
    def __init__(self,  sep=',', tedges = None, twopaths = None):
        """Constructor generating a temporal network instance
        
        @param sep: a separator character to be used for the naming of higher-order nodes v-w
        @param tedges: an optional list of (possibly unordered time-stamped) links from which to 
            construct a temporal network instance
        @param twopaths: an optional list of two-paths from which to 
            construct a temporal network instance
        """
        
        self.tedges = []
        nodes_seen = defaultdict( lambda:False )
        self.nodes = []

        # Generate index structures which help to efficiently extract time-respecting paths

        # A dictionary storing all time-stamped links, indexed by time-stamps
        self.time = defaultdict( lambda: list() )

        # A dictionary storing all time-stamped links, indexed by target and source nodes
        self.targets = defaultdict( lambda: dict() )

        # A dictionary storing all time-stamped links, indexed by source and target nodes
        self.sources = defaultdict( lambda: dict() )

        # A dictionary storing time stamps at which links (v,*;t) originate from node v
        self.activities = defaultdict( lambda: list() )

        # An ordered list of time-stamps
        self.ordered_times = []

        self.tedges = []

        if tedges is not None:
            print('Building index data structures ...', end='')
            for e in tedges:
                # TODO: This could probably be done more efficiently ...
                if e[2] not in self.activities[e[0]]:
                    self.activities[e[0]].append(e[2])
                self.time[e[2]].append(e)
                self.targets[e[2]].setdefault(e[1], []).append(e)
                self.sources[e[2]].setdefault(e[0], []).append(e)
                if not nodes_seen[e[0]]:
                    nodes_seen[e[0]] = True
                if not nodes_seen[e[1]]:
                    nodes_seen[e[1]] = True
            self.tedges = tedges
            self.nodes = list(nodes_seen.keys())
            print('finished.')

            print('Sorting time stamps ...', end = '')
            self.ordered_times = np.sort(list(self.time.keys()))
            for v in self.nodes:
                self.activities[v] = np.sort(self.activities[v])
            print('finished.')

        # Index structures for two-path structures
        self.twopaths = []
        self.twopathsByNode = defaultdict( lambda: dict() )
        self.twopathsByTime = defaultdict( lambda: dict() )
        self.tpcount = -1

        """The separator character to be used to generate higher-order nodes"""
        self.separator = sep

        """The maximum time difference between consecutive links to be used 
        for extraction of time-respecting paths of length two"""
        self.delta = 1                                    

        # Generate index structures if temporal network is constructed from two-paths
        if twopaths is not None:
            t = 0
            for tp in twopaths:
                self.twopaths.append(tp)
                s = tp[0]
                v = tp[1]
                d = tp[2]

                if s not in self.nodes:
                    self.nodes.append(s)
                if v not in self.nodes:
                    self.nodes.append(v)
                if d not in self.nodes:
                    self.nodes.append(d)
  
                self.twopathsByNode[v].setdefault(t, []).append(tp)
                t +=1
            self.tpcount = len(twopaths)        

        # Cached instances of first- and second-order aggregate networks
        self.g1 = 0
        self.g2 = 0
        self.g2n = 0
        

    def addEdge(self, source, target, ts):
        """Adds a directed time-stamped edge (source,target;time) to the temporal network. To add an undirected 
            time-stamped link (u,v;t) at time t, please call addEdge(u,v;t) and addEdge(v,u;t).
        
        @param source: naem of the source node of a directed, time-stamped link
        @param target: name of the target node of a directed, time-stamped link
        @param ts: (integer) time-stamp of the time-stamped link
        """
        e = (source, target, ts)
        self.tedges.append(e)
        if source not in self.nodes:
            self.nodes.append(source)
        if target not in self.nodes:
            self.nodes.append(target)

        # Add edge to index structures
        self.time[ts].append(e)
        self.targets[ts].setdefault(target, []).append(e)
        self.sources[ts].setdefault(source, []).append(e)

        # TODO: This could probably be done more efficiently ...
        if ts not in self.activities[source]:
            self.activities[source].append(ts)

        # Reorder time stamps
        self.ordered_times = np.sort(list(self.time.keys()))
        
        self.InvalidateTwoPaths()


    def InvalidateTwoPaths(self):
        """Invalidates all cached two-paths, as well as any (higher-order) aggregate networks"""
        
        # Invalidate indexed data 
        self.tpcount = -1
        self.twopaths = []
        self.twopathsByNode = defaultdict( lambda: dict() )
        self.twopathsByTime = defaultdict( lambda: dict() )
        self.g1 = 0
        self.g2 = 0
        self.g2n = 0
        

    def vcount(self):
        """Returns the total number of different vertices active across the whole evolution of the temporal network. 
        This number corresponds to the number of nodes in the (first-order) time-aggregated network."""
        return len(self.nodes)

        
    def ecount(self):
        """Returns the number of time-stamped edges (u,v;t) in this temporal network"""
        return len(self.tedges)


    def setMaxTimeDiff(self, delta):
        """Sets the maximum time difference delta between consecutive links to be used for 
        the extraction of time-respecting paths of length two (two-paths). If two-path structures
        and/or second-order networks have previously been computed, this method will invalidate all
        cached data if the new delta is different from the old one (for which two-path statistics have been computed)

        @param delta: Indicates the maximum temporal distance up to which time-stamped links will be 
        considered to contribute to time-respecting paths. For (u,v;3) and (v,w;7) a time-respecting path (u,v)->(v,w) 
        will be inferred for all 0 < delta <= 4, while no time-respecting path will be inferred for all delta > 4. 
        If the max time diff is not set specifically, the default value of delta=1 will be used, meaning that a
        time-respecting path u -> v will only be inferred if there are *directly consecutive* time-stamped 
        links (u,v;t) (v,w;t+1).
        """
        
        if delta != self.delta:
            # Set new value and invalidate two-path structures
            self.delta = delta
            self.InvalidateTwoPaths()
    

    def getInterEventTimes(self):
        """Returns a numpy array containing all time differences between any 
            two consecutive time-stamped links (involving any node)"""

        timediffs = []
        for i in range(1, len(self.ordered_times)):
            timediffs += [self.ordered_times[i] - self.ordered_times[i-1]]
        return np.array(timediffs)


    def getInterPathTimes(self):
        """Returns a dictionary which, for each node v, contains all time differences 
            between any time-stamped link (*,v;t) and the next link (v,*;t') (t'>t)
            in the temporal network"""

        interPathTimes = defaultdict( lambda: list() )
        for e in self.tedges:
            # Get target v of current edge e=(u,v,t)
            v = e[1]
            t = e[2]

            # Get time stamp of link (v,*,t_next) with smallest t_next such that t_next > t
            i = bisect_right(self.activities[v], t)
            if i != len(self.activities[v]):
                interPathTimes[v].append(self.activities[v][i]-t)
        return interPathTimes


    def Summary(self):
        """Returns a string containing basic summary statistics of this temporal network"""

        summary = ''

        summary += 'Nodes:\t\t\t' +  str(self.vcount()) + '\n'
        summary += 'Time-stamped links:\t' + str(self.ecount()) + '\n'
        summary += 'Links/Nodes:\t\t' + str(self.ecount()/self.vcount()) + '\n'
        summary += 'Observation period:\t[' + str(min(self.ordered_times)) + ', ' + str(max(self.ordered_times)) + ']\n'
        summary += 'Observation length:\t' + str(max(self.ordered_times) - min(self.ordered_times)) + '\n'
        summary += 'Time stamps:\t\t' + str(len(self.ordered_times)) + '\n'

        d = self.getInterEventTimes()
    
        summary += 'Avg. inter-event dt:\t' + str(np.mean(d)) + '\n'
        summary += 'Min/Max inter-event dt:\t' + str(min(d)) + '/' + str(max(d)) + '\n'

        summary += 'Max Time Diff (delta):\t' +str(self.delta) + '\n'
        summary += 'Two-paths:\t\t'
        if self.tpcount>=0:
            summary += str(self.tpcount) + '\n'
        else:
            summary += 'not calculated\n'
        
        if self.g1!=0:
            summary += 'First-order nodes:\t' + str(self.g1.vcount()) + '\n'
            summary += 'First-order links:\t' + str(self.g1.ecount()) + '\n'
        else:
            summary += 'First-order network:\tnot constructed\n'
        
        if self.g2!=0:
            summary += 'Second-order nodes:\t' + str(self.g2.vcount())+ '\n'
            summary += 'Second-order links:\t' + str(self.g2.ecount())+ '\n'
        else:
            summary += 'Second-order network:\tnot constructed\n'
        
        return summary


    def extractTwoPaths(self):
        """Extracts all time-respecting paths of length two in this temporal network for the currently set 
        maximum time difference delta. The two-paths extracted by this method will be used in the 
        construction of second-order time-aggregated networks, as well as in the analysis of 
        causal structures of this temporal network. If an explicit call to this method is omitted, 
        it will be run with the current parameter delta set in the 
        TemporalNetwork instance (default: delta=1) whenever two-paths are needed for the first time.
        Once two-paths have been computed, they will be cached and reused until the maximum time difference 
        delta is changed.
        """

        print('Extracting two-paths for delta =', self.delta, '...', end ='')

        self.tpcount = -1
        self.twopaths = []
        self.twopathsByNode = defaultdict( lambda: dict() )
        self.twopathsByTime = defaultdict( lambda: dict() )        

        # Avoid reevaluations in loop
        tpappend = self.twopaths.append
        srcs = self.sources
        tgts = self.targets
        odts = self.ordered_times
        dt = self.delta

        # For each time stamp in the ordered list of time stamps
        for i in range(len(odts)):
            t = odts[i]

            # For each possible middle node v (i.e. all target nodes at time t) ... 
            for v in tgts[t]:
                # Get the minimum and maximum indices of time stamps in the ordered list of "activities" of node v
                # which continue a time-respecting path, i.e. we are interested in 
                # the time stamps t' of all links (v,*;t') such that t'  \in (t, t+delta]

                # The minimum index is the index of the smallest time stamp that is larger than t
                min_ix = bisect_right(self.activities[v], t)

                # The maximum index is the index of the largest time stamp that is smaller or equal than t + delta
                max_ix = bisect_right(self.activities[v], t+dt)-1

                # For all time-stamped links (v,*;t') with t' \in (t, t+delta] ...
                for j in range(min_ix, max_ix+1):
                    future_t = self.activities[v][j]
                    # For all possible IN-edges at time t that link *to* node v
                    for e_in in tgts[t][v]:
                        # Combine with all OUT-edges at time future_t that link *from* v
                        for e_out in srcs[future_t][v]:
                            s = e_in[0]
                            d = e_out[1]
                            indeg_v = len(tgts[t][v])
                            outdeg_v = len(srcs[future_t][v])                                    

                            # Create a weighted two-path tuple
                            # (s, v, d, weight)
                            two_path = (s,v,d, float(1)/(indeg_v*outdeg_v))

                            # TODO: Add support for time-stamped links which have link weights w by themselves, i.e. (u,v;t;w)

                            tpappend(two_path)
                            self.twopathsByNode[v].setdefault(t, []).append(two_path)
                            self.twopathsByTime[t].setdefault(v, []).append(two_path)
        
        self.tpcount = len(self.twopaths)

        # Invalidate cached aggregate networks
        g1 = 0
        g2 = 0
        g2n = 0                
        print('finished.')

        
    def TwoPathCount(self):
        """Returns the total number of time-respecting paths of length two (two-paths) 
            which have been extracted from the time-stamped edge sequence."""
        
        # If two-paths have not been extracted yet, do it now
        if self.tpcount == -1:
            self.extractTwoPaths()

        return self.tpcount
    

    def igraphFirstOrder(self, all_links=False, force=False):
        """Returns the first-order time-aggregated network
           corresponding to this temporal network. This network corresponds to 
           a first-order Markov model reproducing the link statistics in the 
           weighted, time-aggregated network."""
        
        if self.g1 != 0 and not force:
            return self.g1
           
        # If two-paths have not been extracted yet, do it now
        if self.tpcount == -1:
            self.extractTwoPaths()

        print('Constructing first-order aggregate network ...', end='')

        self.g1 = igraph.Graph(n=len(self.nodes), directed=True)
        self.g1.vs["name"] = self.nodes

        edge_list = {}

        # Gather all edges and their (accumulated) weights in a directory        
        if all_links:
            for e in self.tedges:
                edge_list[(e[0], e[1])] = edge_list.get((e[0], e[1]), 0) + 1
        else:                    
            for tp in self.twopaths:
                key1 = (tp[0], tp[1])
                key2 = (tp[1], tp[2])
                # get key{1,2} with default value 0 from edge_list directory
                edge_list[key1] = edge_list.get(key1, 0) + tp[3]
                edge_list[key2] = edge_list.get(key2, 0) + tp[3]
            
        # adding all edges at once is much faster as igraph updates internal
        # data structures after each vertex/edge added
        self.g1.add_edges( edge_list.keys() )
        self.g1.es["weight"] = list(edge_list.values())
        
        print('finished.')

        return self.g1


    def igraphSecondOrder(self):
        """Returns the second-order time-aggregated network
           corresponding to this temporal network. This network corresponds to 
           a second-order Markov model reproducing both the link statistics and 
           (first-order) order correlations in the underlying temporal network.
           """

        if self.g2 != 0:
            return self.g2

        if self.tpcount == -1:
            self.extractTwoPaths()

        print('Constructing second-order aggregate network ...', end='')

        # create vertex list and edge directory first
        vertex_list = []
        edge_dict = {}
        sep = self.separator
        for tp in self.twopaths:
            n1 = str(tp[0])+sep+str(tp[1])
            n2 = str(tp[1])+sep+str(tp[2])
            vertex_list.append(n1)
            vertex_list.append(n2)
            key = (n1, n2)
            edge_dict[key] = edge_dict.get(key, 0) + tp[3]
            
        # remove duplicate vertices by building a set
        vertex_list = list(set(vertex_list))
        
        # build 2nd order graph
        self.g2 = igraph.Graph( n=len(vertex_list), directed=True )
        self.g2.vs["name"] = vertex_list
        
        # add all edges in one go
        self.g2.add_edges( edge_dict.keys() )
        self.g2.es["weight"] = list(edge_dict.values())

        print('finished.')

        return self.g2


    def igraphSecondOrderNull(self):
        """Returns a second-order null Markov model 
           corresponding to the first-order aggregate network. This network
           is a second-order representation of the weighted time-aggregated network. In order to 
           compute the null model, the strongly connected component of the second-order network 
           needs to have at least two nodes.          
           """
        if self.g2n != 0:
            return self.g2n

        g2 = self.igraphSecondOrder().components(mode='STRONG').giant()
        n_vertices = len(g2.vs)

        assert n_vertices>1, print('Error: Strongly connected component is empty.')
        
        T = RWTransitionMatrix( g2 )
        pi = StationaryDistribution(T)
        
        # Construct null model second-order network
        self.g2n = igraph.Graph(directed=True)

        # This ensures that vertices are ordered in the same way as in the empirical second-order network
        for v in self.g2.vs():
            self.g2n.add_vertex(name=v["name"])
        
        ## TODO: This operation is the bottleneck for large data sets. We should only iterate over those edge pairs, that actually are two-paths
        edge_dict = {}
        vertices = g2.vs()
        sep = self.separator
        for i in range(n_vertices):
            e1 = vertices[i]
            e1name = e1["name"]
            a,b = e1name.split(sep)
            for j in range(i+1, n_vertices):
                e2 = vertices[j]
                e2name = e2["name"]
                a_,b_ = e2name.split(sep)
                
                # Check whether this pair of nodes in the second-order 
                # network is a *possible* forward two-path
                if b == a_:
                    w = np.abs(pi[e2.index])
                    if w>0:
                        edge_dict[(e1name, e2name)] = w
                        
                if b_ == a:
                    w = np.abs(pi[e1.index])
                    if w>0:
                        edge_dict[(e2name, e1name)] = w
        
        # add all edges to the graph in one go
        self.g2n.add_edges( edge_dict.keys() )
        self.g2n.es["weight"] = list(edge_dict.values())
        
        return self.g2n


    def ShuffleEdges(self, l=0):        
        """Generates a shuffled version of the temporal network in which edge statistics (i.e.
        the frequencies of time-stamped edges) are preserved, while all order correlations are 
        destroyed. The shuffling procedure randomly reshuffles the time-stamps of links.
        
        @param l: the length of the sequence to be generated (in terms of the number of time-stamped links.
            For the default value l=0, the length of the generated shuffled temporal network will be equal to that of 
            the original temporal network. 
        """
        tedges = []
        
        if self.tpcount == -1:
            self.extractTwoPaths()
        
        if l==0:
            l = 2*int(len(self.tedges)/2)
        for i in range(l):
            # Pick random link
            edge = self.tedges[np.random.randint(0, len(self.tedges))]
            # Pick random time stamp
            time = self.tedges[np.random.randint(0, len(self.ordered_times))]
            # Generate new time-stamped link
            tedges.append( (edge[0], edge[1], time) )

        # Generate temporal network
        t = TemporalNetwork(sep=self.separator, tedges=tedges)

        # Fix node order to correspond to original network
        t.nodes = self.nodes
            
        return t
        
        
    def ShuffleTwoPaths(self, l=0):
        """Generates a shuffled version of the temporal network in which two-path statistics (i.e.
        first-order correlations in the order of time-stamped edges) are preserved
        
        @param l: the length of the sequence to be generated (in terms of the number of time-stamped links.
            For the default value l=0, the length of the generated shuffled temporal network will be equal to that of 
            the original temporal network. 
        """
        
        tedges = []
        
        if self.tpcount == -1:
            self.extractTwoPaths()
        
        t = 0
        
        times = list(self.twopathsByTime.keys())
        
        if l==0:
            l = len(self.tedges)
        for i in range(int(l/2)):
            # Chose a time uniformly at random
            rt = times[np.random.randint(0, len(self.twopathsByTime))]
            
            # Chose a node active at that time uniformly at random
            rn = list(self.twopathsByTime[rt].keys())[np.random.randint(0, len(self.twopathsByTime[rt]))]
            
            # Chose a two path uniformly at random
            paths = self.twopathsByTime[rt][rn]
            tp = paths[np.random.randint(0, len(paths))]
            
            tedges.append((tp[0], tp[1], t))
            t += 1
            tedges.append((tp[1], tp[2], t))
            t += 1
            
        tempnet = TemporalNetwork(sep=',', tedges=tedges)
        return tempnet