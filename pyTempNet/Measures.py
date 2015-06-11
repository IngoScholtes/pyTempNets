# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:49:39 2015

@author: Ingo Scholtes
"""
import numpy as np
import scipy.linalg as spl
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
from pyTempNet import *
from pyTempNet.Processes import *

def Laplacian(temporalnet, model="SECOND"):
    """Returns the Laplacian matrix corresponding to the the second-order (model=SECOND) or 
    the second-order null (model=NULL) model for a temporal network.
    """
    assert model is "SECOND" or "NULL"
    
    if model == "SECOND":
        network = temporalnet.igraphSecondOrder().components(mode="STRONG").giant()
    elif model == "NULL": 
        network = temporalnet.igraphSecondOrderNull().components(mode="STRONG").giant()  
    
    T2 = Processes.RWTransitionMatrix(network)
    I = np.diag([1]*len(network.vs()))

    return I-T2


def FiedlerVector(temporalnet, model="SECOND"):
    """Returns the Fiedler vector of the second-order (model=SECOND) or the
    second-order null (model=NULL) model for a temporal network. The Fiedler 
     vector can be used for a spectral bisectioning of the network.
    """
    assert model is "SECOND" or "NULL"
    
    L = Laplacian(temporalnet, model)
            
    w, v = spl.eig(L, left=True, right=False)
    return v[:,np.argsort(np.absolute(w))][:,1]


def AlgebraicConn(temporalnet, model="SECOND"):
    """Returns the Fiedler vector of the second-order (model=SECOND) or the
    second-order null (model=NULL) model for a temporal network. The Fiedler 
     vector can be used for a spectral bisectioning of the network.
    """

    L = Laplacian(temporalnet, model)

    w, v = spl.eig(L, left=True, right=False)
    evals_sorted = np.sort(np.absolute(w))
    return np.abs(evals_sorted[1])


def __log(p):
    """Logarithm (base two) which defines log2(0)=0"""
    if p == 0: 
        return 0.0
    else:
        return np.log2(p)
      
# makes the above function also usable for vectors as input
__log = np.vectorize(__log, otypes=[np.float])

def EntropyGrowthRate(T):
    """Computes the entropy growth rate of a transition matrix"""
    
    # Compute normalized leading eigenvector of T (stationary distribution)
    w, v = spl.eig(T, left=True, right=False)
    pi = v[:,np.argsort(-np.absolute(w))][:,0]
    pi = pi/sum(pi)
    
    H = 0.0
    for i in range(T.shape[0]):
        H += pi[i] * np.dot( T[i,range(T.shape[1])], __log(T[i, range(T.shape[1])]) )
    return -H
    
    
def EntropyGrowthRateRatio(t, mode='FIRSTORDER'):
        """Computes the ratio between the entropy growth rate ratio between
        the second-order and first-order model of a temporal network t. Ratios smaller
        than one indicate that the temporal network exhibits non-Markovian characteristics"""
        
        # Generate strongly connected component of second-order networks
        g2 = t.igraphSecondOrder().components(mode="STRONG").giant()
        
        if mode == 'FIRSTORDER':
            g2n = t.igraphFirstOrder().components(mode="STRONG").giant()
        else:
            g2n = t.igraphSecondOrderNull().components(mode="STRONG").giant()
        
        # Calculate transition matrices
        T2 = Processes.RWTransitionMatrix(g2)
        T2n = Processes.RWTransitionMatrix(g2n)

        # Compute entropy growth rates of transition matrices        
        H2 = np.absolute(EntropyGrowthRate(T2))
        H2n = np.absolute(EntropyGrowthRate(T2n))
        
        # Return ratio
        return H2/H2n        
        

def __Entropy(prob):
        H = 0
        for p in prob:
            H = H+np.log2(p)*p
        return -H

def __BWPrefMatrix(t, v):
    """Computes a betweenness preference matrix for a node v in a temporal network t"""
    g = t.igraphFirstOrder()
    v_vertex = g.vs.find(name=v)
    indeg = v_vertex.degree(mode="IN")        
    outdeg = v_vertex.degree(mode="OUT")
    index_succ = {}
    index_pred = {}
    
    B_v = np.matrix(np.zeros(shape=(indeg, outdeg)))
        
    # Create an index-to-node mapping for predecessors and successors
    i = 0
    for u in v_vertex.predecessors():
        index_pred[u["name"]] = i
        i = i+1
    
    
    i = 0
    for w in v_vertex.successors():
        index_succ[w["name"]] = i
        i = i+1

    # Calculate entries of betweenness preference matrix
    for time in t.twopathsByNode[v]:
        for tp in t.twopathsByNode[v][time]:
            B_v[index_pred[tp[0]], index_succ[tp[2]]] += (1. / float(len(t.twopathsByNode[v][time])))
    
    return B_v

def BetweennessPreference(t, v, normalized = False):
        """Computes the betweenness preference of a node v in a temporal network t"""
        
        g = t.igraphFirstOrder()
        
        # If the network is empty, just return zero
        if len(g.vs) == 0:
            return 0.0        
        
        v_vertex = g.vs.find(name=v)    
        indeg = v_vertex.degree(mode="IN")        
        outdeg = v_vertex.degree(mode="OUT")
    
        # First create the betweenness preference matrix (equation (2) of the paper)
        B_v = __BWPrefMatrix(t, v)
        
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
        
        H_s = __Entropy(marginal_s)
        H_d = __Entropy(marginal_d)
        
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
    """
    g2 = t.igraphSecondOrder().components(mode="STRONG").giant()
    g2n = t.igraphSecondOrderNull().components(mode="STRONG").giant()
    
    # Build transition matrices
    T2 = Processes.RWTransitionMatrix(g2)
    T2n = Processes.RWTransitionMatrix(g2n)    
    
    # Compute eigenvector sequences
    w2, v2 = spl.eig(T2, left=True, right=False)
    evals2_sorted = np.sort(-np.absolute(w2))
    w2n, v2n = spl.eig(T2n, left=True, right=False)
    evals2n_sorted = np.sort(-np.absolute(w2n))
        
    return np.log(np.abs(evals2n_sorted[1]))/np.log(np.abs(evals2_sorted[1]))

def EigenvectorCentrality(t, model='SECOND'):
    """Computes eigenvector centralities of nodes in the second-order networks, 
    and aggregates them to obtain the eigenvector centrality of nodes in the 
    first-order network."""
    
    start = tm.clock()

    assert model == 'SECOND' or model == 'NULL'

    name_map = {}

    g1 = t.igraphFirstOrder()
    i = 0 
    for v in g1.vs()["name"]:
        name_map[v] = i
        i += 1
    evcent_1 = [0]*len(g1.vs())

    if model == 'SECOND':
        g2 = t.igraphSecondOrder()
    else:
        g2 = t.igraphSecondOrderNull()    

    beforeMatrix = tm.clock()
    print("\tbefore matrix took: ", (beforeMatrix - start))
    
    # Compute eigenvector centrality in second-order network
    A = getTransposedSparseWeightedAdjacencyMatrix( g2 )
    matrix = tm.clock()
    print("\tmatrix took: ", (matrix - beforeMatrix))
    
    w, evcent_2 = sla.eigs( A, k=1, which="LM" )
    eig = tm.clock()
    print("\teig took: ", (eig - matrix))
    print("sparse la eigs EW: ", w[0])
    print("sparse la eigs EV[0]:", evcent_2[range(10)])
    #print v
    
    #evcent_2 = v[:,np.argsort(-w)][:,0]
    #print evcent_2
    
    # Aggregate to obtain first-order eigenvector centrality
    for i in range(len(evcent_2)):
        # Get name of target node
        target = g2.vs()[i]["name"].split(';')[1]        
        evcent_1[name_map[target]] += evcent_2[i]
    rest = tm.clock()
    print("\trest took: ", (rest - eig))
    
    return np.real(evcent_1/sum(evcent_1))

def EigenvectorCentralityLegacy(t, model='SECOND'):
    """Computes eigenvector centralities of nodes in the second-order networks, 
    and aggregates them to obtain the eigenvector centrality of nodes in the 
    first-order network."""
    
    start = tm.clock()

    assert model == 'SECOND' or model == 'NULL'

    name_map = {}

    g1 = t.igraphFirstOrder()
    i = 0 
    for v in g1.vs()["name"]:
        name_map[v] = i
        i += 1
    evcent_1 = [0]*len(g1.vs())

    if model == 'SECOND':
        g2 = t.igraphSecondOrder()
    else:
        g2 = t.igraphSecondOrderNull()    

    beforeMatrix = tm.clock()
    print("\tbefore matrix took: ", (beforeMatrix - start))
    
    # Compute eigenvector centrality in second-order network
    A = getWeightedAdjacencyMatrix( g2 )
    matrix = tm.clock()
    print("\tmatrix took: ", (matrix - beforeMatrix))
    
    w, v = spl.eig(A, left=True, right=False)
    eig = tm.clock()
    print("\teig took: ", (eig - matrix))
    print("legacy implementation EW: ", w[np.argsort(-w)[0]])
    #print w[np.argsort(-w)]
    #print v
    
    evcent_2 = v[:,np.argsort(-w)][:,0]
    print("legacy implementation EV[0]:", evcent_2[range(10)])
    #print evcent_2
    
    # Aggregate to obtain first-order eigenvector centrality
    for i in range(len(evcent_2)):
        # Get name of target node
        target = g2.vs()[i]["name"].split(';')[1]        
        evcent_1[name_map[target]] += evcent_2[i]
    rest = tm.clock()
    print("\trest took: ", (rest - eig))
    
    return np.real(evcent_1/sum(evcent_1))

def BetweennessCentrality(t, model='SECOND'):
    """Computes betweenness centralities of nodes in the second-order networks, 
    and aggregates them to obtain the betweenness centrality of nodes in the 
    first-order network."""

    assert model == 'SECOND' or model == 'NULL'

    name_map = {}

    g1 = t.igraphFirstOrder()
    i = 0 
    for v in g1.vs()["name"]:
        name_map[v] = i
        i += 1
    bwcent_1 = [0]*len(g1.vs())

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
    first-order network."""

    assert model == 'SECOND' or model == 'NULL'

    name_map = {}

    g1 = t.igraphFirstOrder()
    i = 0 
    for v in g1.vs()["name"]:
        name_map[v] = i
        i += 1
    pagerank_1 = [0]*len(g1.vs())

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

