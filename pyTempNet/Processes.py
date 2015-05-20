# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:35:22 2015

@author: Ingo
"""
import numpy as np
import scipy.linalg as spl


def RWTransitionMatrix(g):
    """Generates a random walk transition matrix corresponding to a (possibly) weighted
    and directed network""" 
    if g.is_weighted() == False:
        A = np.matrix(list(g.get_adjacency()))
        D = np.diag(g.degree(mode='out'))        
    else:
        A = np.matrix(list(g.get_adjacency(attribute='weight', default=0)))
        D = np.diag(g.strength(mode='out', weights=g.es["weight"]))

    T = np.zeros(shape=(len(g.vs), len(g.vs)))    
    
    for i in range(len(g.vs)):
        for j in range(len(g.vs)):
            # facilitate debugging of assertion errors ... 
            a = A[i,j]
            d = D[i,i]           
            T[i,j] = a/d
            assert T[i,j]>=0 and T[i,j] <= 1
    return T


def TVD(p1, p2):
    """Compute total variation distance between two stochastic column vectors"""
    tvd = 0
    for i in range(len(p1)):
        tvd+=abs(p1[i] - p2[i])
    return tvd/2
    
    
def RWDiffusion(g, samples = 5, epsilon=0.01):
    """Computes the average number of steps requires by a random walk process
    to fall below a total variation distance below epsilon (TVD computed between the momentary 
    visitation probabilities \pi^t and the stationary distribution \pi = \pi^{\infty}. This time can be 
    used to measure diffusion speed in a given (weighted and directed) network."""
    avg_speed = 0
    T = RWTransitionMatrix(g)
    for s in range(samples):
        w, v = spl.eig(T, left=True, right=False)
        pi = v[:,np.argsort(-w)][:,0]
        pi = pi/sum(pi)
        x = [0] * len(g.vs)
        x[np.random.randint(len(g.vs()))] = 1
        t = 0
        while TVD(x,pi)>epsilon:
            t += 1
            x = np.dot(x,T)
        avg_speed += t
    return avg_speed/samples
    
    
    