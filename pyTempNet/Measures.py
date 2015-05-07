# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:49:39 2015

@author: Ingo Scholtes
"""

import numpy as np
import scipy.linalg as spl


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

    n = len(network.vs)    
    
    A2 = np.matrix(list(network.get_adjacency()))
    T2 = np.zeros(shape=(n, n))
    D2 = np.diag(network.strength(mode='out', weights=network.es["weight"]))
    
    for i in range(n):
        for j in range(n):
            T2[i,j] = A2[i,j]/D2[i,i]
            
    w2, v2 = spl.eig(T2, left=True, right=False)
    return v2[1]
    
def EntropyGrowthRateRatio(t):
        """Computes the ratio between the entropy growth rate of the 
           second-order model and the first-order model"""
        pass

def __Entropy(prob):
        H = 0
        for p in prob:
            H = H+np.log2(p)*p
        return -H

def BetweennessPreference(t, v, normalized = False):
        """Computes a list of betweenness preferences of nodes"""
        
        g = t.iGraphFirstOrder()        
        
        # If the network is empty, just return zero
        if len(g.vs) == 0:
            return 0.0
        
        I = 0.0
        
        indeg = g[v].degree(mode="IN")
        outdeg = g[v].degree(mode="OUT")
                
        P = np.zeros(shape=(indeg, outdeg))
        # TODO: compute betweenness preference matrix                
        
        marginal_s = []
        marginal_d = []
        
        # Marginal probabilities P_d = \sum_s'{P_{s'd}}
        for d in range(d):
            P_d = 0.0
            for s_prime in range(indeg):
                P_d += P[s_prime, d]
            marginal_d.append(P_d)
        
        # Marginal probabilities P_s = \sum_d'{P_{sd'}}
        for s in range(indeg):
            P_s = 0.0
            for d_prime in range(outdeg):
                P_s += P[s, d_prime]
            marginal_s.append(P_s)
        
        H_s = __Entropy(marginal_s)
        H_d = __Entropy(marginal_d)
        
        # Here we just compute equation (4) of the paper ... 
        for s in range(indeg):
            for d in range(outdeg):
                if P[s, d] != 0: # 0 * Log(0)  = 0
                    # Compute Mutual information
                    I += P[s, d] * np.log2(P[s, d] / (marginal_s[s] * marginal_d[d]))
        
        if normalized:
            return I/(H_s+H_d)
        else:
            return I    


def SlowDownFactor(temporalnet):    
    """Returns a factor S that indicates how much slower (S>1) or faster (S<1)
    a diffusion process in the temporal network evolves on a second-order model 
    compared to a first-order model. This value captures the effect of order
    correlations on dynamical processes.
    """
    g2 = temporalnet.iGraphSecondOrder().components(mode="STRONG").giant()
    g2n = temporalnet.iGraphSecondOrderNull().components(mode="STRONG").giant()
    
    A2 = np.matrix(list(g2.get_adjacency()))
    T2 = np.zeros(shape=(len(g2.vs), len(g2.vs)))
    D2 = np.diag(g2.strength(mode='out', weights=g2.es["weight"]))
    
    for i in range(len(g2.vs)):
        for j in range(len(g2.vs)):
            T2[i,j] = A2[i,j]/D2[i,i]
    
    A2n = np.matrix(list(g2n.get_adjacency()))
    T2n = np.zeros(shape=(len(g2n.vs), len(g2n.vs)))
    D2n = np.diag(g2n.strength(mode='out', weights=g2n.es["weight"]))
    
    for i in range(len(g2n.vs)):
        for j in range(len(g2n.vs)):
            T2n[i,j] = A2n[i,j]/D2n[i,i]
    
    w2, v2 = spl.eig(T2, left=True, right=False)
    w2n, v2n = spl.eig(T2n, left=True, right=False)
    
    return np.log(np.abs(w2n[1]))/np.log(np.abs(w2[1]))
