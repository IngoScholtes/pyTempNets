# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 17:28:39 2015
@author: Ingo Scholtes

(c) Copyright ETH Zürich, Chair of Systems Design, 2015
"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sla

from collections import defaultdict

from bisect import bisect_left
from bisect import bisect_right

from pyTempNet import Utilities
from pyTempNet.Log import *


def GetFirstOrderDistanceMatrix(t):        
    """Calculates a matrix D containing the shortest path lengths between all
    pairs of nodes calculated based on the topology of the *first-order* aggregate network. 
    The ordering of rows/columns corresponds to the ordering of nodes in the vertex sequence of 
    the igraph first order time-aggregated network. A mapping between nodes and indices can be 
    found in Utilities.firstOrderNameMap().    
    
    @param t: the temporal network to calculate shortest path lengths for based on a first-order
        aggregate representation    
    """   

    # This way of generating the first-order time-aggregated network makes sure that 
    # links are not omitted even if they do not contribute to any time-respecting path
    g1 = t.igraphFirstOrder(all_links=False, force=True)

    name_map = Utilities.firstOrderNameMap( t )

    D = np.zeros(shape=(len(t.nodes),len(t.nodes)))
    D.fill(np.inf)
    np.fill_diagonal(D, 0)

    for v in g1.vs()["name"]:
        for w in g1.vs()["name"]:
            # Compute all shortest paths using igraph
            X = g1.get_shortest_paths(v,w)
            for p in X:
                if len(p)>0:
                    D[name_map[v], name_map[w]] = len(p)-1
    return D


def GetSecondOrderDistanceMatrix(t, model='SECOND'):
    """Calculates a matrix D containing the shortest path lengths between all
    pairs of nodes calculated based on the topology of the *second-order* aggregate network. 
    The ordering of rows/columns corresponds to the ordering of nodes in the vertex sequence of 
    the igraph first order time-aggregated network. A mapping between nodes and indices can be 
    found in Utilities.firstOrderNameMap().    
    
    @param t: the temporal network to calculate shortest path lengths for based on a second-order
        aggregate representation 
    @param model: either C{"SECOND"} or C{"NULL"}, where C{"SECOND"} is the 
      the default value.   
    """   

    if (model is "SECOND" or "NULL") == False:
        raise ValueError("model must be one of \"SECOND\" or \"NULL\"")

    name_map = Utilities.firstOrderNameMap( t )

    if model == 'SECOND':
        g2 = t.igraphSecondOrder()
    else:
        g2 = t.igraphSecondOrderNull()    

    D = np.zeros(shape=(len(t.nodes),len(t.nodes)))
    D.fill(np.inf)
    np.fill_diagonal(D, 0)

    sep = t.separator

    for v in g2.vs()["name"]:
        source = v.split(sep)[0]
        for w in g2.vs()["name"]:
            target = w.split(sep)[1]
            X = g2.get_shortest_paths(v,w)            
            for p in X:
                if len(p)>0:
                    D[name_map[source], name_map[target]] = min(len(p), D[name_map[source], name_map[target]])
    return D


def GetMinTemporalDistance(t, delta=1, collect_paths=True):
    """ Computes the minimum temporal distance between all pairs of nodes in 
        terms of time-respecting paths (using a given maximum time difference delta), 
        across all possible starting times in the temporal network

        @param t: the temporal network to calculate the distance for
        @param delta: the maximum waiting time to be used for the definition of time-respecting paths
        @param collect_paths: whether or not to return all shortest time-respecting paths
    """

    Log.add('Computing minimum temporal distances for delta = ' + str(int(delta)) + ' ...')

    name_map = Utilities.firstOrderNameMap( t )

    minD = np.zeros(shape=(len(t.nodes),len(t.nodes)))
    minD.fill(np.inf)

    # Each node is connected to itself via a path of length zero
    np.fill_diagonal(minD, 0)

    minPaths = defaultdict( lambda: defaultdict( lambda: [] ) )

    for start_t in t.ordered_times:
        D, paths = GetTemporalDistanceMatrix(t, start_t, delta, collect_paths)
        for v in t.nodes:
            for w in t.nodes:
                if D[name_map[v], name_map[w]] < minD[name_map[v], name_map[w]]:
                    minD[name_map[v], name_map[w]] = D[name_map[v], name_map[w]]
                    minPaths[v][w] = paths[v][w]
                elif D[name_map[v], name_map[w]] == minD[name_map[v], name_map[w]] and minD[name_map[v], name_map[w]] < np.inf:
                    for p in paths[v][w]:
                        if p not in minPaths[v][w]:
                            minPaths[v][w] = minPaths[v][w] + [p]
    Log.add('finished.')
    return minD, minPaths


def GetTemporalDistanceMatrix(t, start_t=-1, delta=1, collect_paths=True):
    """A new and faster method to compute the (topologically) shortest time-respecting paths between 
    all pairs of nodes starting at time start_t in an empirical temporal network t.
    This function returns a tuple consisting of 
        1) a matrix D containing the shortest time-respecting path lengths between all
            pairs of nodes. The ordering of rows/columns corresponds to the ordering of nodes 
            in the vertex sequence of the igraph first order time-aggregated network. A
            mapping between nodes and indices can be found in Utilities.firstOrderNameMap().
        2) a list of shortest time-respecting paths, each entry being an ordered sequence 
            of nodes on the corresponding path.
    
    @param t: the temporal network to calculate shortest time-respecting paths for
    @param start_t: the start time for which to consider time-respecting paths (default is t.ordered_times[0])
    @param delta: the maximum time difference to be used in the time-respecting path definition (default 1).
        Note that this parameter is independent from the internal parameter delta used for two-path extraction
        in the class TemporalNetwork
    @param collect_paths: whether or not to collect all shortest time-respecting paths (default = True). If this is 
        set to False, the method will only compute the lengths of shortest time-respecting paths, but not return the actual 
        paths.
        """

    if start_t == -1:
        start_t = t.ordered_times[0]

    # Initialize dictionary taking shortest paths 
    Paths = defaultdict( lambda: defaultdict( lambda: [] ) )

    # Get a mapping between node names and matrix indices
    name_map = Utilities.firstOrderNameMap( t )

    # Initialize topological distance matrix
    # TODO: This may yield a memory problem for large graphs 
    D = np.zeros(shape=(len(t.nodes),len(t.nodes)))
    D.fill(np.inf)

    # For each node v, calculate shortest/fastest paths to all other nodes ... 
    for v in t.nodes:
        
        # Mark node v as visited at the start time start_t... 
        D[name_map[v], name_map[v]] = 0        
        Paths[v][v] = [ [(v,start_t)] ]

        stack = [ (v, start_t) ]
        
        # While there are nodes, which could possibly continue a time-respecting path
        while len(stack)>0:

            (x,ts) = stack.pop()

            # Get indices of time range which can possibly continue a time-respecting path
            min_ix = bisect_left(t.activities[x], ts)
            max_ix = bisect_left(t.activities[x], ts+delta)-1

            # For all time-stamps at which x is a source node ... 
            for j in range(min_ix, max_ix+1):
                time = t.activities[x][j]

                # For all edges starting at node x at this time
                for e in t.sources[time][x]:

                    # We found a new node that can continue time-respecting paths
                    new_node = (e[1], time+1)
                    
                    if new_node not in stack:
                        stack.append( new_node )

                    # Check whether we found a time-respecting path shorter than the current shortest one ... 
                    if D[name_map[v], name_map[e[1]]] > D[name_map[v], name_map[e[0]]] + 1:
                        
                        # In this case we update the distance matrix
                        D[name_map[v], name_map[e[1]]] = D[name_map[v], name_map[e[0]]] + 1

                        if collect_paths == True:
                            # Delete any previous shortest paths 
                            Paths[v][e[1]] = []

                            # Collect all paths to e[0] and concatenate with the current node e[1]
                            for p in Paths[v][e[0]]:
                                Paths[v][e[1]] = Paths[v][e[1]] + [p + [(e[1],time+1)]]

                    # We may also have found a path that has the same length as other shortest paths ...
                    elif D[name_map[v], name_map[e[1]]] == D[name_map[v], name_map[e[0]]] + 1 and collect_paths == True:

                        # Collect all paths to e[0] and concatenate with the current node e[1]
                        for p in Paths[v][e[0]]:
                            Paths[v][e[1]] = Paths[v][e[1]] + [p + [(e[1],time+1)]]
        
    # The algorithm terminates as soon as it is impossible to continue any of the time-respecting paths
    return D, Paths