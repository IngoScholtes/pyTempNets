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
import sys

from bisect import bisect_left
from bisect import bisect_right

from pyTempNet import Utilities


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
                    #print(source, '->', target, ':', p)
    return D



def GetMinTemporalDistance(t, delta=1, collect_paths=True):

    name_map = Utilities.firstOrderNameMap( t )
    minD = np.zeros(shape=(len(t.nodes),len(t.nodes)))
    minD.fill(np.inf)

    minPaths = defaultdict( lambda: defaultdict( lambda: [] ) )

    # Each node is connected to itself via a path of length zero
    np.fill_diagonal(minD, 0)

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
    return minD, minPaths




def newGetTemporalDistanceMatrix(t, start_t=-1, delta=1, collect_paths=True):
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

    # For each node v, calculate shortest/fastest paths to all other nodes ... 
    for v in t.nodes:

        # Initialize distances and paths
        # TODO: We could remove this and directly use the matrices above 
        distances = [np.inf]*t.vcount()
        paths = defaultdict( lambda:list())
        
        # Mark node v as visited at the start time start_t... 
        distances[name_map[v]] = 0
        stack = [ (v, start_t) ]
        paths[v] = [ [(v,start_t)] ]
        
        # While there are nodes, which could possibly continue a time-respecting path
        while len(stack)>0:

            (x,ts) = stack.pop();

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
                    if distances[name_map[e[1]]] > distances[name_map[e[0]]] + 1:
                        
                        # In this case we update the distance matrix
                        distances[name_map[e[1]]] = distances[name_map[e[0]]] + 1

                        if collect_paths == True:
                            # Delete any previous shortest paths 
                            paths[e[1]] = []

                            # Collect all paths to e[0] and concatenate with the current node e[1]
                            for p in paths[e[0]]:
                                paths[e[1]] = paths[e[1]] + [p + [(e[1],time+1)]]

                    # We may also have found a path that has the same length as other shortest paths ...
                    elif distances[name_map[e[1]]] == distances[name_map[e[0]]] + 1 and collect_paths == True:

                        # Collect all paths to e[0] and concatenate with the current node e[1]
                        for p in paths[e[0]]:
                            paths[e[1]] = paths[e[1]] + [p + [(e[1],time+1)]]
        
        # This algorithm terminates as soon as it is impossible to continue time-respecting paths
        Paths[v] = paths
        D[name_map[v],:] = distances
    return D, Paths



def GetTemporalDistanceMatrix(t, start_t=0, delta=1, collect_paths=True):
    """Calculates the (topologically) shortest time-respecting paths between 
    all pairs of nodes starting at time start_t in an empirical temporal network t.
    This function returns a tuple consisting of 
        1) a matrix D containing the shortest time-respecting path lengths between all
            pairs of nodes. The ordering of rows/columns corresponds to the ordering of nodes 
            in the vertex sequence of the igraph first order time-aggregated network. A
            mapping between nodes and indices can be found in Utilities.firstOrderNameMap().
        2) a list of shortest time-respecting paths, each entry being an ordered sequence 
            of nodes on the corresponding path.
    
    @param t: the temporal network to calculate shortest time-respecting paths for
    @param start_t: the start time for which to consider time-respecting paths (default 0)
    @param delta: the maximum time difference to be used in the time-respecting path definition (default 1).
        Note that this parameter is independent from the internal parameter delta used for two-path extraction
        in the class TemporalNetwork"""

    # TODO: Fix calculation for arbitrary delta. Right now we may not find all shortest time-respecting paths for delta > 1

    # Get a mapping between node names and matrix indices
    name_map = Utilities.firstOrderNameMap( t )

    # This distance matrix will contain the (topological) lengths of shortest 
    # time-respecting paths between all pairs of nodes
    # the default value (indicating a missing path) is infinity
    D = np.zeros(shape=(len(t.nodes),len(t.nodes)))
    D.fill(np.inf)

    # Each node is connected to itself via a path of length zero
    np.fill_diagonal(D, 0)

    # In this matrix, we keep a record of the time stamp of the last 
    # time-stamped link for each of the current shortest time-respecting paths
    T = np.zeros(shape=(len(t.nodes),len(t.nodes)))
    T.fill(-np.inf)

    # This will take the time stamp of the time-stamped edge considered 
    # in the previous step of the algorithm
    last_ts = -np.inf
    
    # Find first index i such that ordered_times[i] is greater or equal than the given start time
    start_ix = bisect_left(t.ordered_times, start_t)

    # Initialize matrix T
    for j in range(start_ix, len(t.ordered_times)):
        ts = t.ordered_times[j]
        if last_ts < 0:
                last_ts = ts
        # We initialize the last time stamp for all source nodes that are 
            # active in the beginning of the link sequence, so that they 
            # can act as seeds for the time-respecting path construction.  
        if ts - start_t < delta:
            for e in t.time[ts]:                
                T[name_map[e[0]], name_map[e[0]]] = start_t-1
        else:
                break

    # Initialize shortest path tree for path reconstruction
    Paths = defaultdict( lambda: defaultdict( lambda: [] ) )    
    if collect_paths == True:
        for v in t.nodes:
            Paths[v][v] = [ [v] ]

    # Consider all time-respecting paths starting in any node v at the start time
    for v in t.nodes:
        # Consider the ordered sequence of time-stamps, starting from the first index greater or equal to start_t
        for j in range(start_ix, len(t.ordered_times)):
            ts = t.ordered_times[j]

            # Since time stamps are ordered, we can stop as soon the current time stamp 
            # is more than delta time steps away from the last time step. In this case, by definition 
            # none of the future time-stamped links can contribute to a time-respecting path
            if ts-last_ts > delta:
                break

            last_ts = ts

            # Consider all time-stamped links (e[0], e[1], ts) occuring at time ts
            for e in t.time[ts]:
                
                # If there is a time-respecting path v -> e[0] and if the previous time step on 
                # this time-respecting path is not older than delta ...
                if D[name_map[v], name_map[e[0]]] < np.inf and ts - T[name_map[v], name_map[e[0]]] > 0 and ts - T[name_map[v], name_map[e[0]]] <= delta:

                    # ... then the time-stamped link (e[0], e[1], ts) leads to a 
                    # new shortest path v -> e[0] -> e[1] iff the current distance D[v,e[1]] > D[v,e[0]] + 1

                    if D[name_map[v], name_map[e[1]]] > D[name_map[v], name_map[e[0]]] + 1:
                        # Update the distance between v and e[1]
                        D[name_map[v], name_map[e[1]]] = D[name_map[v], name_map[e[0]]] + 1
                        # Remember the last time stamp on this path
                        T[name_map[v], name_map[e[1]]] = ts
                        # Update the shortest path tree
                        if collect_paths == True:
                            Paths[v][e[1]] = []
                            for p in Paths[v][e[0]]:
                                Paths[v][e[1]] = Paths[v][e[1]] + [p + [e[1]]]
                    elif D[name_map[v], name_map[e[1]]] == D[name_map[v], name_map[e[0]]] + 1 and collect_paths==True:
                        for p in Paths[v][e[0]]:
                            Paths[v][e[1]] = Paths[v][e[1]] + [p + [e[1]]]


    return (D, Paths)