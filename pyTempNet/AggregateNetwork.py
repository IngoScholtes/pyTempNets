# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:47:31 CEST 2015
@author: Ingo Scholtes, Roman Cattaneo

(c) Copyright ETH Zürich, Chair of Systems Design, 2015
"""

from collections import defaultdict    # default dictionaries
import time as tm                      # timings
import igraph as ig                    # graph construction & output
from pyTempNet.Log import *            # logging infrastructure
import copy


class AggregateNetwork:
    """A class representing (higher order) aggregated networks of a given 
    temporal network instance"""

#############################################################################
# private methods of the aggregated network class
#############################################################################

    def __extract_k_paths(self, tmpNet, order, dt):
        assert( order > 1 )
        # TODO this is possibly not the best/fastest solution to the problem
        # TODO since foreach time-step all possible k-paths are generated
        # TODO again

        kpaths = list()
        #loop over all time-steps (at which something is happening)
        next_valid_t = 0
        for t in tmpNet.ordered_times:
            if t < next_valid_t:
                continue
            
            next_valid_t = t + dt
            possible_path = defaultdict( lambda: list() )
            candidate_nodes = set()
            
            # case k == 0
            current_edges = list()
            for i in range(dt):
                current_edges.extend(tmpNet.time[t+i])
                
            for e in current_edges:
                # NOTE that we do not want to consider self loops
                if e[0] != e[1]:
                    possible_path[e[1]].append( [e[0], e[1]] )
                    candidate_nodes.add(e[1])
            
            # 1 <= current_k < k
            for current_k in range(1, order):
                new_candidate_nodes = set()

                for node in candidate_nodes:
                    update = dict()
                    
                    # all edges orginating from node at times t in [t+1, t+delta]
                    new_edges = list()
                    for i in range(dt):
                        new_edges.extend( tmpNet.sources[t+current_k+i].get(node, list()) )

                    len_new_edges = len(new_edges)
                    for e in new_edges:
                        src = e[0]
                        dst = e[1]
                        for path in possible_path[src]:
                            # NOTE: avoid self loops
                            if len(path) > 0 and path[-1] == dst:
                                continue;
                            
                            # NOTE: you have to do this in two steps. you can
                            # NOTE: not directly append 'dst'
                            new_path = list(path)
                            new_path.append( dst )
                            possible_path[dst].append( new_path )
                            new_candidate_nodes.add( dst )
                            if( (current_k+1 == order) and (len(new_path) == order+1) ):
                                # readd weights w again
                                # TODO: make the next line more readable
                                w = 1. / (len_new_edges * len([i for i in possible_path[src] if len(i) == order]))
                                key = tuple(new_path)
                                update[key] = update.get(key, 0) + w
                    
                    for key, val in update.items():
                        kpaths.append( { "nodes": key, "weight": val } )
                
                candidate_nodes = new_candidate_nodes
            
            # NOTE: possible_path will hold all k-paths for 1 <= k <= self.k and
            # this time-step at point in the program
        return kpaths
    
#############################################################################
# public API
#############################################################################
    
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
        
        if( order < 1 ):
            raise ValueError("order must be >= 1")
        if( maxTimeDiff < 1 ):
            raise ValueError("maxTimeDiff must be >= 1")
        
        self.k     = order
        self.delta = maxTimeDiff
        self.sep   = tempNet.separator
        
        # time-respecting k-paths and their count
        if( order == 1 ):
            # NOTE make a deep copy such that changed edges in the temporal 
            # NOTE network do not propagate into independant aggregated negworks
            self.kp = copy.deepcopy(tempNet.tedges)
        else:
            self.kp = self.__extract_k_paths( tempNet, self.k, self.delta )
        self.kpcount = len(self.kp)
        
        # order k aggregated network
        self.gk = 0

    def order(self):
        """Returns the order, k, of the aggregated network"""
        return self.k

    def maxTimeDiff(self):
        """Returns the maximal time difference, delta, between consecutive 
        links in the temporal network"""
        return self.delta

    def kPathCount(self):
        """Returns the total number of time-respecting paths of length k 
        (so called k-paths) which have been extracted from the temporal 
        network.
        """
        return self.kpcount
    
    def kPaths(self):
        """Returns all time-respecting paths of length k (k-paths) which
        have been extracted from teh temporal network.
        """
        return self.kp

    def Summary(self):
        """returns a rather brief summary of the higher order network"""
        summary = ''
        summary += "Higher order network with the following params:"
        summary += "order: " + str(self.order())
        summary += "delta: " + str(self.maxTimeDiff())
        
        summary += "kpaths"
        summary += "  count: " + self.KPathCount
        summary += "  list of paths: " + self.kp
            
        return summary


    def igraphKthOrder(self):
        """Returns the kth-order time-aggregated network
           corresponding to this temporal network. This network corresponds to
           a kth-order Markov model reproducing both the link statistics and
           (first-order) order correlations in the underlying temporal network.
           """
        if self.gk != 0:
            Log.add('Delivering cached version of k-th-order aggregate network')
            return self.gk
        
        Log.add('Constructing k-th-order aggregate network ...')
        assert( self.kp > 0 )

        # create vertex list and edge directory
        vertices = list()
        edges    = dict()

        if( self.k == 1 ):
            for edge in self.kp:
                vertices.append(edge[0])
                vertices.append(edge[1])
                key = (edge[0], edge[1])
                edges[key] = edges.get(key, 0) + 1
        else:
            for path in self.kp:
                n1 = self.sep.join([str(n) for (i,n) in enumerate(path['nodes']) if i < self.k])
                n2 = self.sep.join([str(n) for (i,n) in enumerate(path['nodes']) if i > 0])
                vertices.append(n1)
                vertices.append(n2)
                key = (n1, n2)
                edges[key] = edges.get(key, 0) + path['weight']

        # remove duplicate vertices by building a set
        vertices = list(set(vertices))
        n = len(vertices)

        # Sanity check
        if n == 0:
            Log.add('K-th order aggregate network has no nodes. Consider using a smaller value for k.', Severity.WARNING)

        # build graph and return
        self.gk = ig.Graph( n, directed=True )
        self.gk.vs['name'] = vertices
        self.gk.add_edges( edges.keys() )
        self.gk.es['weight'] = list( edges.values() )

        Log.add('finished.')
        return self.gk

#############################################################################
## Measures
#############################################################################
