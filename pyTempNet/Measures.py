# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:49:39 2015

@author: Ingo Scholtes
"""

import numpy as np
import scipy.linalg as spl
from pyTempNet import *
from pyTempNet.Processes import *


def FiedlerVector(temporalnet, model="SECOND"):
    """Returns the Fiedler vector of the second-order (model=SECOND) or the
    second-order null (model=NULL) model for a temporal network. The Fiedler 
     vector can be used for a spectral bisectioning of the network.
    """
    
    assert model is "SECOND" or "NULL"
    
    if model == "SECOND":
        network = temporalnet.iGraphSecondOrder().components(mode="STRONG").giant()
    elif model == "NULL": 
        network = temporalnet.iGraphSecondOrderNull().components(mode="STRONG").giant()  
    
    T = Processes.RWTransitionMatrix(network)
            
    w, v = spl.eig(T, left=True, right=False)
    return v[:,np.argsort(-np.absolute(w))][:,1] 


def __log(p):
    """Logarithm (base two) which defines log2(0)=0"""
    if p == 0: 
        return 0.0
    else:
        return np.log2(p)

def EntropyGrowthRate(T):
    """Computes the entropy growth rate of a transition matrix"""
    
    # Compute normalized leading eigenvector of T (stationary distribution)
    w, v = spl.eig(T, left=True, right=False)
    pi = v[:,np.argsort(-np.absolute(w))][:,0]
    pi = pi/sum(pi)
    
    H = 0.0
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):                 
            H += pi[i] * T[i,j] * __log(T[i,j])
    return -H
    
    
def EntropyGrowthRateRatio(t):
        """Computes the ratio between the entropy growth rate ratio between
        the second-order and first-order model of a temporal network t. Ratios smaller
        than one indicate that the temporal network exhibits non-Markovian characteristics"""
        
        # Generate strongly connected component of second-order networks
        g2 = t.iGraphSecondOrder().components(mode="STRONG").giant()
        g2n = t.iGraphSecondOrderNull().components(mode="STRONG").giant()
        
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
    g = t.iGraphFirstOrder()
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
        
        g = t.iGraphFirstOrder()
        
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
    g2 = t.iGraphSecondOrder().components(mode="STRONG").giant()
    g2n = t.iGraphSecondOrderNull().components(mode="STRONG").giant()
    
    # Build transition matrices
    T2 = Processes.RWTransitionMatrix(g2)
    T2n = Processes.RWTransitionMatrix(g2n)    
    
    # Compute eigenvector sequences
    w2, v2 = spl.eig(T2, left=True, right=False)
    evals2_sorted = np.sort(-np.absolute(w2))
    w2n, v2n = spl.eig(T2n, left=True, right=False)
    evals2n_sorted = np.sort(-np.absolute(w2n))
        
    return np.log(np.abs(evals2n_sorted[1]))/np.log(np.abs(evals2_sorted[1]))
