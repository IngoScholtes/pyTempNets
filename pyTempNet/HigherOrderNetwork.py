# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:47:31 CEST 2015
@author: Ingo Scholtes, Roman Cattaneo

(c) Copyright ETH Zürich, Chair of Systems Design, 2015
"""

from collections import defaultdict
import time as tm                      # timings
import igraph                          # construction & output of networks
from pyTempNet.Log import *            # logging


class HigherOrderNetwork:
    """A class representing higher order networks of a given temporal network 
    instance"""
    
    def __init__(self, tempNet, order, maxTimeDiff=1):
        """Constructs an aggregated temporal network of order k
        
        @param tn:          a temporal network instance
        @param order:       order of the aggregated network, length of time 
                            respecting paths.
        @param maxTimeDiff: maximal temporal distance up to which time-stamped 
                            links will be considered to contribute to a time-
                            respecting path. Default: 1
        
        Note: If the max time diff is not set specifically, the default value of 
        delta=1 will be used, meaning that a time-respecting path u -> v will 
        only be inferred if there are *directly consecutive* time-stamped links
        (u,v;t) (v,w;t+1).
        """
        
        assert order >= 1
        assert maxTimeDiff >= 1
        
        self.tn    = tempNet
        self.k     = order
        self.delta = maxTimeDiff
        
        # time-respecting k-paths and their count
        self.kpaths = []
        self.kpcount = -1
        
        # do not extract k-paths as long as they are not needed
        ## TODO: extract k-path with respect to delta
        #self.extractKPaths()
        
        # order k aggregated network
        self.gk  = 0
        self.gkn = 0
        
        # 3rd order network
        self.g3  = 0
        
        
    def clearCache(self):
        """Clears the cached information: K-paths, aggregated k-th order 
        network and null model"""
        self.kpaths = []
        self.kpcount = -1
        
        self.gk  = 0
        self.gkn = 0
        self.g3  = 0
        Log.add("Cache cleared.", Severity.INFO)
        
        
    def setMaxTimeDiff(self, delta):
        """Sets the maximum time difference delta between consecutive links to
        be used for the extraction of time-respecting paths of length k, so 
        called k-paths.
        Note: If k-path structures and/or kth-order networks have previously 
        been computed, this method will invalidate all cached data if the new
        delta is different from the old one (for which k-path statistics have 
        been computed)

        @param delta: Indicates the maximum temporal distance up to which 
        time-stamped links will be considered to contribute to time-
        respecting paths. For (u,v;3) and (v,w;7) a time-respecting path 
        (u,v)->(v,w) will be inferred for all delta >= 4, while no 
        time-respecting path will be inferred for all delta < 4.
        """
        assert delta >= 1
        if delta != self.delta:
            # Set new value and invalidate two-path structures
            Log.add("Changing maximal time difference from " + str(self.data) 
                    + " to " + str(delta), Severity.INFO)
            self.delta = delta
            self.clearCache()
    
    def resetMaxTimeDiff(self):
        """Resets the maximal time difference delta between consecutive links 
           to be used for the extraction of time-respecting paths to the
           default value of delta=1"""
        setMaxTimeDiff()

    def setOrder(self, k):
        """Changes the order of the aggregated temporal network and therefore 
        the length of time-respecting paths (k-paths).
        
        Note: If k-path structures and/or kth-order networks have previously 
        been computed, this method will invalidate all cached data if the new 
        order is different from the old one (for which k-path statistics have 
        been computed)
        """
        assert k >= 1
        if k != self.k:
            # Set new value and invalidate any cached data
            Log.add("Changeing order of aggregated network from " + str(self.k) 
                    + " to " + k, Severity.k)
            self.k = k
            self.clearCache()


    def KPathCount(self):
        """Returns the total number of time-respecting paths of length k 
        (so called k-paths) which have been extracted from the temporal 
        network.
        A count of -1 indicates that they have not yet been extracted."""

        return self.kpcount


    def order(self):
        """Returns the order, k, of the aggregated network"""
        return self.k


    def maxTimeDiff(self):
        """Returns the maximal time difference, delta, between consecutive 
        links in the temporal network"""
        return self.delta
    
    def Summary(self):
        """ some meaning full docstring here... give a summary of the net"""
        summary = ''
        summary += "Higher order network with the following params:"
        summary += "order: " + str(self.order())
        summary += "delta: " + str(self.maxTimeDiff())
        
        summary += "kpaths"
        if self.KPathCount == -1:
            summary += "  count: " + self.KPathCount
            summary += "  list of paths: " + self.kpaths
        else:
            summary += "  not yet extracted."
            
        return summary
    
    def extractKPaths(self):
        """Extracts all time-respecting paths of length k in this temporal 
        network for the currently set maximum time difference delta. The 
        k-paths extracted by this method will be used in the construction of 
        higher-order time-aggregated networks, as well as in the analysis of 
        causal structures of this temporal network. If an explicit call to this
        method is omitted, it will be run whenever k-paths are needed for the 
        first time.
        
        Once k-paths have been computed, they will be cached and reused until 
        the maximum time difference, delta, and/or the order k is changed.        
        """
        
        # TODO this is possibly not the best/fastest solution to the problem
        # TODO since foreach time-step all possible k-paths are generated
        # TODO again
        
        # TODO make sure the weights get added, if the same path is found twice
        # TODO @see test_threePaths.py WeightCummulationTest
        
        start = tm.clock()
        
        tmpNet = self.tn
        
        #loop over all time-steps (at which something is happening)
        #print("ordered times:", tmpNet.ordered_times)
        next_valid_t = 0
        for t in tmpNet.ordered_times:
            if t < next_valid_t:
                continue
            
            next_valid_t = t + self.delta
            possible_path = defaultdict( lambda: list() )
            candidate_nodes = set()
            #print("current t", t)
            
            # case k == 0
            current_edges = list()
            for i in range(self.delta):
                current_edges.extend(tmpNet.time[t+i])
                
            for e in current_edges:
                possible_path[e[1]].append( [e[0], e[1]] )
                candidate_nodes.add(e[1])
            
            #print("possible paths after k = 0", possible_path)
            
            # 1 <= current_k < k
            for current_k in range(1, self.k):
                new_candidate_nodes = set()
                #print("  current_k", current_k)
                
                #print("this are the candidate_nodes:", candidate_nodes)
                for node in candidate_nodes:
                   # print("    processing node", node)
                    # all edges orginating from node at times t in [t+1, t+delta]
                    new_edges = list()
                    for i in range(self.delta):
                        new_edges.extend( tmpNet.sources[t+current_k+i].get(node, list()) )
                    #print("    new_edges", new_edges)
                    for e in new_edges:
                        src = e[0]
                        dst = e[1]
                        #print("      possible_path[src]", possible_path[src])
                        for path in possible_path[src]:
                           # print("        processing path:", path)
                            # NOTE: you have to do this in two steps. you can
                            # NOTE: not directly append 'dst'
                            new_path = list(path)
                            new_path.append( dst )
                            #print("      intended new path: ", new_path )
                            possible_path[dst].append( new_path )
                            #print("        new possible paths:", possible_path)
                            new_candidate_nodes.add( dst )
                            if( (current_k+1 == self.k) and (len(new_path) == self.k+1) ):
                                # readd weights w again
                                # TODO: make the next line more readable
                                w = 1. / (len(new_edges) * len([i for i in possible_path[src] if len(i) == self.k]))
                                self.kpaths.append( {"nodes": new_path,
                                                     "weight": w} )
                               # print("        found new kpath! these are now all kpaths:", self.kpaths)
                                #print("        # new edges:", len(new_edges))
                               # print("        # possible_paths[src]", len(possible_path[src]))
                
                candidate_nodes = new_candidate_nodes
            
            # NOTE: possible_path will hold all k-paths for 1 <= k <= self.k and
            # this time-step at point in the program
            
        self.kpcount = len(self.kpaths)
        end = tm.clock()
        
        print( 'time elapsed:', (end-start))
        return self.kpaths
    
    def igraphThirdOrder( self ):
        """Returns the 3rd-order time-aggregated network corresponding to this
        temporal network.
        In a 3rd-order network is a directed network where nodes consist of 
        two-paths which are only connected, if there exists a time-respecting
        three-path connecting the source node of the first two-path with 
        the target node of the second two-path.
        """
        # TODO add an example to make this docstring understandable
        
        Log.add('Constructing third-order aggregate network ...')
        
        # NOTE this only makes sense for 3rd order 
        assert( self.k == 3 )
        
        # extract two-paths for nodes
        if( self.tn.tpcount == -1 ):
            self.tn.extractTwoPaths()

        # extract three-paths to connect the nodes
        if( self.kpcount == -1 ):
            self.extractKPaths()
        
        # create vertex list and edge directory first
        vertices = []
        edge_dict = {}
        sep = self.tn.separator
        for path in self.kpaths:
            n1 = str(path['nodes'][0])+sep+str(path['nodes'][1])+sep+str(path['nodes'][2])
            n2 = str(path['nodes'][1])+sep+str(path['nodes'][2])+sep+str(path['nodes'][3])
            vertices.append(n1)
            vertices.append(n2)
            key = (n1, n2)
            edge_dict[key] = edge_dict.get(key, 0.) + path['weight']

        # remove duplicate vertices by building a set
        vertices = list(set(vertices))
        
        # build 2nd order graph
        self.g3 = igraph.Graph( n=len(vertices), directed=True )
        self.g3.vs["name"] = vertices
        
        # add all edges in one go
        self.g3.add_edges( edge_dict.keys() )
        self.g3.es["weight"] = list(edge_dict.values())

        Log.add('finished.')

        return self.g3
    
    def igraphKthOrder(self):
        """Returns the kth-order time-aggregated network
           corresponding to this temporal network. This network corresponds to 
           a kth-order Markov model reproducing both the link statistics and 
           (first-order) order correlations in the underlying temporal network.
           """

           # TODO
           # self.gk = ...
        return igraph.Graph()


    def igraphKthOrderNull(self):
        """Returns a kth-order null Markov model 
           corresponding to the first-order aggregate network. This network
           is a kth-order representation of the weighted time-aggregated network.
           
           Note: In order to compute the null model, the strongly connected 
           component of the kth-order network needs to have at least two nodes.          
           """
           
           # TODO
           # self.gkn = ...
           
           # NOTE: pay attention to the fact, that a null model only makes sense for k > 1.
        return igraph.Graph()