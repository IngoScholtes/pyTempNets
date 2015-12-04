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
        k1_paths = list()
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
                    k1_update = dict()
                    
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
                            # these are (k-1)-paths
                            if( (current_k == order) and (len(new_path) == order) ):
                                weight = 1. / (len_new_edges * len([i for i in possible_path[src] if len(i) == (order-1)]))
                                k1_key = tuple(new_path)
                                k1_update[key] = update.get(key, 0) + weight
                            # these are k-paths
                            if( (current_k+1 == order) and (len(new_path) == order+1) ):
                                # readd weights w again
                                # TODO: make the next line more readable
                                weight = 1. / (len_new_edges * len([i for i in possible_path[src] if len(i) == order]))
                                key = tuple(new_path)
                                update[key] = update.get(key, 0) + weight
                    
                    for key, val in update.items():
                        kpaths.append( { "nodes": key, "weight": val } )
                    for key, val in k1_update.items():
                        k1_paths.append( {"nodes": key, "weight": val} )
                
                candidate_nodes = new_candidate_nodes
            
            # NOTE: possible_path will hold all k-paths for 1 <= k <= self.k and
            # this time-step at point in the program
        return (kpaths, k1_paths)
    
#############################################################################
# public API
#############################################################################
    
    def __init__(self, tempNet, order):
        """Constructs an aggregated temporal network of order k
        
        @param tempNet:     temporal network instance
        @param order:       order of the aggregated network, length of time 
                            respecting paths.
        """
        
        if( order < 1 ):
            raise ValueError("order must be >= 1")
        
        self._k     = order
        self._delta = tempNet.delta
        self._sep   = tempNet.separator
        
        # time-respecting k-paths and their count
        if( order == 1 ):
            # NOTE make a deep copy such that changed edges in the temporal 
            # NOTE network do not propagate into independant aggregated negworks
            self._kp = copy.deepcopy(tempNet.tedges)
        else:
            self._kp, self._k1_p = self.__extract_k_paths( tempNet, self._k, self._delta )
        self._kpcount = len(self._kp)
        
        # igraph representation of order k aggregated network
        self._gk = 0
        # igraph representation of order k null model
        self._gk_null = 0

    def order(self):
        """Returns the order, k, of the aggregated network"""
        return self._k

    def maxTimeDiff(self):
        """Returns the maximal time difference, delta, between consecutive 
        links in the temporal network"""
        return self._delta

    def kPathCount(self):
        """Returns the total number of time-respecting paths of length k 
        (so called k-paths) which have been extracted from the temporal 
        network.
        """
        return self._kpcount
    
    def kPaths(self):
        """Returns all time-respecting paths of length k (k-paths) which
        have been extracted from teh temporal network.
        """
        return self._kp

    def Summary(self):
        """returns a rather brief summary of the higher order network"""
        summary = ''
        summary += "Higher order network with the following params:"
        summary += "order: " + str(self._order())
        summary += "delta: " + str(self._maxTimeDiff())
        
        summary += "kpaths"
        summary += "  count: " + self._kpcount
        summary += "  list of paths: " + self._kp
            
        return summary


    def igraphKthOrder(self):
        """Returns the kth-order time-aggregated network
           corresponding to this temporal network. This network corresponds to
           a kth-order Markov model reproducing both the link statistics and
           (first-order) order correlations in the underlying temporal network.
           """
        if self._gk != 0:
            Log.add('Delivering cached version of k-th-order aggregate network')
            return self._gk
        
        Log.add('Constructing k-th-order aggregate network ...')
        assert( self._kp > 0 )

        # create vertex list and edge directory
        vertices = list()
        edges    = dict()

        if( self._k == 1 ):
            for edge in self._kp:
                vertices.append(edge[0])
                vertices.append(edge[1])
                key = (edge[0], edge[1])
                edges[key] = edges.get(key, 0) + 1
        else:
            for path in self._kp:
                n1 = self._sep.join([str(n) for (i,n) in enumerate(path['nodes']) if i < self._k])
                n2 = self._sep.join([str(n) for (i,n) in enumerate(path['nodes']) if i > 0])
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
        self._gk = ig.Graph( n, directed=True )
        self._gk.vs['name'] = vertices
        self._gk.add_edges( edges.keys() )
        self._gk.es['weight'] = list( edges.values() )

        Log.add('finished.')
        return self._gk
    
    def igraph_null_model(self):
        """Returns null model of order k"""

        assert( k > 1 )
        # get all k-1 paths -> these are saved in k1_p
        
        # for each k-1 path (p1)
            # for each other k-1 path (p2)
                # if( p1 == p2 ) continue
                
                # check, if last part of p1 is identical to first part of p2
                # if so:
                    # calculate weight:
                    # ( weight(p1) * weight(p2) ) / ( weighted intput degree (middle node) * weighted output degree(middle node) )
        
        

#############################################################################
## Measures
#############################################################################
