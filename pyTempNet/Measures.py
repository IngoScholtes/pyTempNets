# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:49:39 2015
@author: Ingo Scholtes

(c) Copyright ETH Zürich, Chair of Systems Design, 2015
"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
import time as tm

from pyTempNet import Utilities

def Laplacian(temporalnet, model="SECOND"):
    """Returns the Laplacian matrix corresponding to the the second-order (model=SECOND) or 
    the second-order null (model=NULL) model for a temporal network.
    
    @param temporalnet: The temporalnetwork instance to work on
    @param model: either C{"SECOND"} or C{"NULL"}, where C{"SECOND"} is the 
      the default value.
    @param sparseLA: whether or not (default) to use sparse linear algebra for 
      calculations.
    @param transposed: wheter or not (default) to transpose the matrix
    """
    if (model is "SECOND" or "NULL") == False:
        raise ValueError("model must be one of \"SECOND\" or \"NULL\"")
    start = tm.clock()
    
    if model == "SECOND":
        network = temporalnet.igraphSecondOrder().components(mode="STRONG").giant()
    elif model == "NULL": 
        network = temporalnet.igraphSecondOrderNull().components(mode="STRONG").giant()  
    
    T2 = Utilities.RWTransitionMatrix( network )
    I  = sparse.identity( len(network.vs()) )

    end = tm.clock()
    print("Calculating Laplacian took: ", (end - start))
    return I-T2


def FiedlerVector(temporalnet, model="SECOND"):
    """Returns the Fiedler vector of the second-order (model=SECOND) or the
    second-order null (model=NULL) model for a temporal network. The Fiedler 
     vector can be used for a spectral bisectioning of the network.
     
    @param temporalnet: The temporalnetwork instance to work on
    @param model: either C{"SECOND"} or C{"NULL"}, where C{"SECOND"} is the 
      the default value.
    """
    if (model is "SECOND" or "NULL") == False:
        raise ValueError("model must be one of \"SECOND\" or \"NULL\"")
    
    # NOTE: The transposed matrix is needed to get the "left" eigen vectors
    L = Laplacian(temporalnet, model)
    # NOTE: ncv=13 sets additional auxiliary eigenvectors that are computed
    # NOTE: in order to be more confident to find the one with the largest
    # NOTE: magnitude, see
    # NOTE: https://github.com/scipy/scipy/issues/4987
    w, v = sla.eigs( L, k=2, which="SM", ncv=13 )
    
    # TODO: ask, if this vector should be normalized. Sparse Linalg sometimes
    # TODO: finds the EV scaled factor (-1)
    return v[:,np.argsort(np.absolute(w))][:,1]


def AlgebraicConn(temporalnet, model="SECOND"):
    """Returns the Fiedler vector of the second-order (model=SECOND) or the
    second-order null (model=NULL) model for a temporal network. The Fiedler 
     vector can be used for a spectral bisectioning of the network.
     
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
    start = tm.clock()
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
    
    end = tm.clock()
    print('Time for EntropyGrowthRateRatio:', (end - start))
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
    
    # NOTE: this might raise a ValueError if vertex v is not found
    v_vertex = g.vs.find(name=v)    
    indeg = v_vertex.degree(mode="IN")        
    outdeg = v_vertex.degree(mode="OUT")

    # First create the betweenness preference matrix (equation (2) of the paper)
    B_v = Utilities.BWPrefMatrix(t, v)
    
    # Normalize matrix (equation (3) of the paper)
    P_v = np.matrix(np.zeros(shape=(indeg, outdeg)))
    S = 0.0
    for s in range(indeg):
        for d in range(outdeg):
            S += B_v[s,d]
    
    if S>0:
        for s in range(indeg):
            for d in range(outdeg):
                P_v[s,d] = B_v[s,d]/S                    
    
    # Compute marginal probabilities
    marginal_s = []
    marginal_d = []
    
    # Marginal probabilities P^v_d = \sum_s'{P_{s'd}}
    for d in range(outdeg):
        P_d = 0.0
        for s_prime in range(indeg):
            P_d += P_v[s_prime, d]
        marginal_d.append(P_d)
    
    # Marginal probabilities P^v_s = \sum_d'{P_{sd'}}
    for s in range(indeg):
        P_s = 0.0
        for d_prime in range(outdeg):
            P_s += P_v[s, d_prime]
        marginal_s.append(P_s)
    
    H_s = Utilities.Entropy(marginal_s)
    H_d = Utilities.Entropy(marginal_d)
    
    I = 0.0
    # Here we just compute equation (4) of the paper ... 
    for s in range(indeg):
        for d in range(outdeg):
            if B_v[s, d] != 0: # 0 * Log(0)  = 0
                # Compute Mutual information
                I += P_v[s, d] * np.log2(P_v[s, d] / (marginal_s[s] * marginal_d[d]))
    
    if normalized:
        return I/np.min([H_s,H_d])
    else:
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
    start = tm.clock()
    
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
    
    end = tm.clock()
    print('Time for SlowDownFactor: ', (end - start))
    
    return np.log(np.abs(evals2n_sorted[1]))/np.log(np.abs(evals2_sorted[1]))

def EigenvectorCentrality(t, model='SECOND'):
    """Computes eigenvector centralities of nodes in the second-order networks, 
    and aggregates them to obtain the eigenvector centrality of nodes in the 
    first-order network.
    
    @param t: The temporalnetwork instance to work on
    @param model: either C{"SECOND"} or C{"NULL"}, where C{"SECOND"} is the 
      the default value."""
    
    start = tm.clock()

    if (model is "SECOND" or "NULL") == False:
        raise ValueError("model must be one of \"SECOND\" or \"NULL\"")

    name_map = {}

    g1 = t.igraphFirstOrder()
    i = 0 
    for v in g1.vs()["name"]:
        name_map[v] = i
        i += 1
    evcent_1 = np.zeros(len(g1.vs()))

    if model == 'SECOND':
        g2 = t.igraphSecondOrder()
    else:
        g2 = t.igraphSecondOrderNull()
    
    # Compute eigenvector centrality in second-order network
    A = getSparseAdjacencyMatrix( g2, attribute="weight", transposed=True )
    evcent_2 = Utilities.StationaryDistribution( A, False )
    
    # Aggregate to obtain first-order eigenvector centrality
    for i in range(len(evcent_2)):
        # Get name of target node
        target = g2.vs()[i]["name"].split(';')[1]
        evcent_1[name_map[target]] += np.real(evcent_2[i])
    
    end = tm.clock()
    print("Time for EigenvectorCentrality: ", (end - start))
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

    name_map = {}

    g1 = t.igraphFirstOrder()
    i = 0 
    for v in g1.vs()["name"]:
        name_map[v] = i
        i += 1
    bwcent_1 = np.zeros(len(g1.vs()))

    if model == 'SECOND':
        g2 = t.igraphSecondOrder()
    else:
        g2 = t.igraphSecondOrderNull()    

    # Compute betweenness centrality in second-order network
    bwcent_2 = np.array(g2.betweenness(weights=g2.es()['weight'], directed=True))
    
    # Aggregate to obtain first-order eigenvector centrality
    for i in range(len(bwcent_2)):
        # Get name of target node
        target = g2.vs()[i]["name"].split(';')[1]
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

    name_map = {}

    g1 = t.igraphFirstOrder()
    i = 0 
    for v in g1.vs()["name"]:
        name_map[v] = i
        i += 1
    pagerank_1 = np.zeros(len(g1.vs()))

    if model == 'SECOND':
        g2 = t.igraphSecondOrder()
    else:
        g2 = t.igraphSecondOrderNull()    

    # Compute betweenness centrality in second-order network
    pagerank_2 = np.array(g2.pagerank(weights=g2.es()['weight'], directed=True))
    
    # Aggregate to obtain first-order eigenvector centrality
    for i in range(len(pagerank_2)):
        # Get name of target node
        target = g2.vs()[i]["name"].split(';')[1]
        pagerank_1[name_map[target]] += pagerank_2[i]
    
    return pagerank_1/sum(pagerank_1)

