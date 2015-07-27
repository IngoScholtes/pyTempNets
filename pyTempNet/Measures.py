# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:49:39 2015
@author: Ingo Scholtes

(c) Copyright ETH Zürich, Chair of Systems Design, 2015
"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sla

from collections import defaultdict
import sys

from pyTempNet import Utilities

def Laplacian(temporalnet, model="SECOND"):
    """Returns the Laplacian matrix corresponding to the the second-order (model=SECOND) or 
    the second-order null (model=NULL) model for a temporal network.
    
    @param temporalnet: The temporalnetwork instance to work on
    @param model: either C{"SECOND"} or C{"NULL"}, where C{"SECOND"} is the 
      the default value.
    """
    if (model is "SECOND" or "NULL") == False:
        raise ValueError("model must be one of \"SECOND\" or \"NULL\"")
    
    if model == "SECOND":
        network = temporalnet.igraphSecondOrder().components(mode="STRONG").giant()
    elif model == "NULL": 
        network = temporalnet.igraphSecondOrderNull().components(mode="STRONG").giant()  
    
    T2 = Utilities.RWTransitionMatrix( network )
    I  = sparse.identity( len(network.vs()) )

    return I-T2


def FiedlerVector(temporalnet, model="SECOND", lanczosVecs=15, maxiter=10):
    """Returns the Fiedler vector of the second-order (model=SECOND) or the
    second-order null (model=NULL) model for a temporal network. The Fiedler 
     vector can be used for a spectral bisectioning of the network.
     
    @param temporalnet: The temporalnetwork instance to work on
    @param model: either C{"SECOND"} or C{"NULL"}, where C{"SECOND"} is the 
      the default value.
    @param lanczosVecs: number of Lanczos vectors to be used in the approximate
        calculation of eigenvectors and eigenvalues. This maps to the ncv parameter 
        of scipy's underlying function eigs. 
    @param maxiter: scaling factor for the number of iterations to be used in the 
        approximate calculation of eigenvectors and eigenvalues. The number of iterations 
        passed to scipy's underlying eigs function will be n*maxiter where n is the 
        number of rows/columns of the Laplacian matrix.
    """
    if (model is "SECOND" or "NULL") == False:
        raise ValueError("model must be one of \"SECOND\" or \"NULL\"")
    
    # NOTE: The transposed matrix is needed to get the "left" eigen vectors
    L = Laplacian(temporalnet, model)
    # NOTE: ncv=13 sets additional auxiliary eigenvectors that are computed
    # NOTE: in order to be more confident to find the one with the largest
    # NOTE: magnitude, see
    # NOTE: https://github.com/scipy/scipy/issues/4987
    maxiter = maxiter*L.get_shape()[0]

    import scipy.linalg as la

    w, v = la.eig(L.todense())
    # TODO: ask, if this vector should be normalized. Sparse Linalg sometimes
    # TODO: finds the EV scaled factor (-1)
    return v[:,np.argsort(np.absolute(w))][:,1]


def AlgebraicConn(temporalnet, model="SECOND"):
    """Returns the algebraic connectivity of the second-order (model=SECOND) or the
    second-order null (model=NULL) model for a temporal network.
    
     @param temporalnet: The temporalnetwork to work on
     @param model: either C{"SECOND"} or C{"NULL"}, where C{"SECOND"} is the 
      the default value.
    """
    
    if (model is "SECOND" or "NULL") == False:
        raise ValueError("model must be one of \"SECOND\" or \"NULL\"")
    
    L = Laplacian(temporalnet, model)
    # NOTE: ncv=13 sets additional auxiliary eigenvectors that are computed
    # NOTE: in order to be more confident to find the one with the largest
    # NOTE: magnitude, see
    # NOTE: https://github.com/scipy/scipy/issues/4987
    w = sla.eigs( L, which="SM", k=2, ncv=13, return_eigenvectors=False )
    evals_sorted = np.sort(np.absolute(w))
    return np.abs(evals_sorted[1])
    
    
def EntropyGrowthRateRatio(t, mode='FIRSTORDER'):
    """Computes the ratio between the entropy growth rate ratio between
    the second-order and first-order model of a temporal network t. Ratios smaller
    than one indicate that the temporal network exhibits non-Markovian characteristics"""
    
    # NOTE to myself: most of the time here goes into computation of the
    # NOTE            EV of the transition matrix for the bigger of the
    # NOTE            two graphs below (either 2nd-order or 2nd-order null)
    
    # Generate strongly connected component of second-order networks
    g2 = t.igraphSecondOrder().components(mode="STRONG").giant()
    
    if mode == 'FIRSTORDER':
        g2n = t.igraphFirstOrder().components(mode="STRONG").giant()
    else:
        g2n = t.igraphSecondOrderNull().components(mode="STRONG").giant()
    
    # Calculate transition matrices
    T2 = Utilities.RWTransitionMatrix(g2)
    T2n = Utilities.RWTransitionMatrix(g2n)

    # Compute entropy growth rates of transition matrices        
    H2 = np.absolute(Utilities.EntropyGrowthRate(T2))
    H2n = np.absolute(Utilities.EntropyGrowthRate(T2n))

    # Return ratio
    return H2/H2n


def BetweennessPreference(t, v, normalized = False):
    """Computes the betweenness preference of a node v in a temporal network t
    
    @param t: The temporalnetwork instance to work on
    @param v: Name of the node to compute its BetweennessPreference
    @param normalized: whether or not (default) to normalize
    """
    g = t.igraphFirstOrder()
    
    # If the network is empty, just return zero
    if len(g.vs) == 0:
        return 0.0

    # First create the betweenness preference matrix (equation (2) of the paper)
    B_v = Utilities.BWPrefMatrix(t, v)
    
    # Normalize matrix (equation (3) of the paper)
    # NOTE: P_v has the same shape as B_v
    P_v = np.zeros(shape=B_v.shape)
    S = np.sum(B_v)
    
    if S>0:
        P_v = B_v / S

    ## Compute marginal probabilities
    ## Marginal probabilities P^v_d = \sum_s'{P_{s'd}}
    marginal_d = np.sum(P_v, axis=0)

    ## Marginal probabilities P^v_s = \sum_d'{P_{sd'}}
    marginal_s = np.sum(P_v, axis=1)
    
    H_s = Utilities.Entropy(marginal_s)
    H_d = Utilities.Entropy(marginal_d)
    
    # build mask for non-zero elements
    row, col = np.nonzero(P_v)
    pv = P_v[(row,col)]
    marginal = np.outer(marginal_s, marginal_d)
    log_argument = np.divide( pv, marginal[(row,col)] )
    
    I = np.dot( pv, np.log2(log_argument) )
    
    if normalized:
        I =  I/np.min([H_s,H_d])

    return I

def SlowDownFactor(t):    
    """Returns a factor S that indicates how much slower (S>1) or faster (S<1)
    a diffusion process in the temporal network evolves on a second-order model 
    compared to a first-order model. This value captures the effect of order
    correlations on a diffusion process in the temporal network.
    
    @param t: The temporalnetwork instance to work on
    """
    
    #NOTE to myself: most of the time goes for construction of the 2nd order
    #NOTE            null graph, then for the 2nd order null transition matrix
    
    g2 = t.igraphSecondOrder().components(mode="STRONG").giant()
    g2n = t.igraphSecondOrderNull().components(mode="STRONG").giant()
    
    # Build transition matrices
    T2 = Utilities.RWTransitionMatrix(g2)
    T2n = Utilities.RWTransitionMatrix(g2n)
    
    # Compute eigenvector sequences
    # NOTE: ncv=13 sets additional auxiliary eigenvectors that are computed
    # NOTE: in order to be more confident to find the one with the largest
    # NOTE: magnitude, see
    # NOTE: https://github.com/scipy/scipy/issues/4987
    w2 = sla.eigs(T2, which="LM", k=2, ncv=13, return_eigenvectors=False)
    evals2_sorted = np.sort(-np.absolute(w2))

    w2n = sla.eigs(T2n, which="LM", k=2, ncv=13, return_eigenvectors=False)
    evals2n_sorted = np.sort(-np.absolute(w2n))
    
    return np.log(np.abs(evals2n_sorted[1]))/np.log(np.abs(evals2_sorted[1]))


def EigenvectorCentrality(t, model='SECOND'):
    """Computes eigenvector centralities of nodes in the second-order networks, 
    and aggregates them to obtain the eigenvector centrality of nodes in the 
    first-order network.
    
    @param t: The temporalnetwork instance to work on
    @param model: either C{"SECOND"} or C{"NULL"}, where C{"SECOND"} is the 
      the default value."""

    if (model is "SECOND" or "NULL") == False:
        raise ValueError("model must be one of \"SECOND\" or \"NULL\"")

    name_map = Utilities.firstOrderNameMap(t)
    
    if model == 'SECOND':
        g2 = t.igraphSecondOrder()
    else:
        g2 = t.igraphSecondOrderNull()
    
    # Compute eigenvector centrality in second-order network
    A = Utilities.getSparseAdjacencyMatrix( g2, attribute="weight", transposed=True )
    evcent_2 = Utilities.StationaryDistribution( A, False )
    
    # Aggregate to obtain first-order eigenvector centrality
    evcent_1 = np.zeros(len(name_map))
    sep = t.separator
    for i in range(len(evcent_2)):
        # Get name of target node
        target = g2.vs()[i]["name"].split(sep)[1]
        evcent_1[name_map[target]] += np.real(evcent_2[i])
    
    return np.real(evcent_1/sum(evcent_1))


def BetweennessCentrality(t, model='SECOND'):
    """Computes betweenness centralities of nodes in the second-order networks, 
    and aggregates them to obtain the betweenness centrality of nodes in the 
    first-order network.
    
    @param t: The temporalnetwork instance to work on
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

    # Compute betweenness centrality in second-order network
    bwcent_2 = np.array(g2.betweenness(weights=g2.es()['weight'], directed=True))
    
    # Aggregate to obtain first-order eigenvector centrality
    bwcent_1 = np.zeros(len(name_map))
    sep = t.separator
    for i in range(len(bwcent_2)):
        # Get name of target node
        target = g2.vs()[i]["name"].split(sep)[1]
        bwcent_1[name_map[target]] += bwcent_2[i]
    
    return bwcent_1/sum(bwcent_1)


def PageRank(t, model='SECOND'):
    """Computes PageRank of nodes in the second-order networks, 
    and aggregates them to obtain the PageRank of nodes in the 
    first-order network.
    
    @param t: The temporalnetwork instance to work on
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

    # Compute betweenness centrality in second-order network
    pagerank_2 = np.array(g2.pagerank(weights=g2.es()['weight'], directed=True))
    
    # Aggregate to obtain first-order eigenvector centrality
    pagerank_1 = np.zeros(len(name_map))
    sep = t.separator
    for i in range(len(pagerank_2)):
        # Get name of target node
        target = g2.vs()[i]["name"].split(sep)[1]
        pagerank_1[name_map[target]] += pagerank_2[i]
    
    return pagerank_1/sum(pagerank_1)


def GetStaticDistanceMatrix(t):        

    g1 = t.igraphFirstOrder(all_links=True, force=True)

    name_map = Utilities.firstOrderNameMap( t )

    D = np.zeros(shape=(len(t.nodes),len(t.nodes)))
    D.fill(np.inf)
    np.fill_diagonal(D, 0)

    for v in g1.vs()["name"]:
        for w in g1.vs()["name"]:
            X = g1.get_shortest_paths(v,w)
            for p in X:
                D[name_map[v], name_map[w]] = len(p)-1
    return D


def GetDistanceMatrix(t, start_t=0, delta=1):
    """Calculates the (topologically) shortest time-respecting paths between 
    all pairs of nodes starting at time start_t in an empirical temporal network t.
    This function returns a tuple consisting of 
        1) a matrix D containing the shortest time-respecting path lengths between all
            pairs of nodes. The ordering of these values corresponds to the ordering of nodes 
            in the vertex sequence of the igraph first order time-aggregated network
        2) a list of shortest time-respecting paths, each entry being an ordered sequence 
            of nodes on the corresponding path. This list can be used to compute the 
            time-respecting path betweenness of nodes in a temporal network
    
    @param t: the temporal network to calculate shortest path for
    @param start_t: the start time for which to consider time-respecting paths (default 0)
    @param delta: the maximum waiting time used in the time-respecting path definition (default 1)
    """

    # We first build some index structures to quickly access tedges by time, target and source
    time = defaultdict( lambda: list() )
    targets = defaultdict( lambda: dict() )
    sources = defaultdict( lambda: dict() )
    for e in t.tedges:
        source = e[0]
        target = e[1]
        ts = e[2]

        # Only keep those time-stamped edges that occur after the start_time
        if ts >= start_t:
            time[ts].append(e)
            targets[ts].setdefault(target, []).append(e)
            sources[ts].setdefault(source, []).append(e)

    ordered_times = np.sort(list(time.keys()))

    # Mapping between node names and matrix indices
    name_map = Utilities.firstOrderNameMap( t )

    # This distance matrix will contain the lengths of shortest 
    # time-respecting paths between all pairs of nodes
    D = np.zeros(shape=(len(t.nodes),len(t.nodes)))
    D.fill(np.inf)
    np.fill_diagonal(D, 0)

    # In this matrix, we keep a record of the time stamps of the last 
    # time-stamped links on alle current shortest time-respecting paths
    T = np.zeros(shape=(len(t.nodes),len(t.nodes)))
    T.fill(-np.inf)
    
    for ts in ordered_times:
        for e in time[ts]:
            # initialize the last time stamp for all source nodes that are 
            # active in the beginning of the link sequence, so that they 
            # can act as seeds for the time-respecting path construction
            if ts - start_t < delta:
                T[name_map[e[0]], name_map[e[0]]] = start_t-1

    Paths = defaultdict( lambda: defaultdict( lambda: [] ) )

    # initialize shortest path tree for path reconstruction 
    for v in t.nodes:
        Paths[v][v] = [v]

    for e in t.igraphFirstOrder().es():
        u = t.igraphFirstOrder().vs["name"][e.source]
        v = t.igraphFirstOrder().vs["name"][e.target]
        Paths[u][v] = [u,v]    
    
    # We consider all time-respecting paths starting in any node v at the start time
    for v in t.nodes:        
        # Consider the ordered sequence of time-stamps
        for ts in ordered_times:
           
            # Consider all time-stamped links (e[0], e[1], ts) occuring at time ts
            for e in time[ts]:
                
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
                        Paths[v][e[1]] = Paths[v][e[0]] + [e[1]]
    
    return (D, Paths)


def GetTimeRespectingBetweenness(t, start_t=0, delta=1, normalized=False):
    """Calculates the time-respecting path betweennness values of 
    all nodes starting at time start_t in an empirical temporal network t.
    This function returns a numpy array of betweenness centrality values. The ordering
    of these values corresponds to the ordering of nodes in the vertex sequence of the 
    igraph first order time-aggregated network
    
    @param t: the temporal network to calculate shortest path for
    @param start_t: the start time for which to consider time-respecting paths (default 0)
    @param delta: the maximum waiting time used in the time-respecting path definition (default 1)
    @param normalized: whether or not to normalize the betweenness values by the number of all
        shortest time-respecting paths in the temporal network.
    """

    # First calculate all shortest time-respecting paths
    D, paths = GetDistanceMatrix(t, start_t, delta)

    bw = np.array([0]*len(t.nodes))

    # Mapping between node names and matrix indices
    name_map = Utilities.firstOrderNameMap( t )
    k=0
    for u in t.nodes:
        for v in t.nodes:
            if u != v:
                for i in range(1, len(paths[u][v])-1):
                    bw[name_map[paths[u][v][i]]] += 1
                    k+=1
    if normalized:
        bw = bw/k
    return bw

