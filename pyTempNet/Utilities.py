# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:48:48 2015
@author: Ingo Scholtes, Roman Cattaneo

(c) Copyright ETH Zürich, Chair of Systems Design, 2015
"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sla

from pyTempNet.Log import *

def getSparseAdjacencyMatrix( graph, attribute=None, transposed=False ):
    """Returns a sparse adjacency matrix of the given graph.
    
    @param attribute: if C{None}, returns the ordinary adjacency matrix.
      When the name of a valid edge attribute is given here, the matrix
      returned will contain the value of the given attribute where there
      is an edge. Default value is assumed to be zero for places where 
      there is no edge. Multiple edges are not supported.
      
    @param transposed: whether to transpose the matrix or not.
    """
    if (attribute is not None) and (attribute not in graph.es.attribute_names()):
      raise ValueError( "Attribute does not exists." )
    
    row = []
    col = []
    data = []
    
    if attribute is None:
      if transposed:
        for edge in graph.es():
          s,t = edge.tuple
          row.append(t)
          col.append(s)
      else:
        for edge in graph.es():
          s,t = edge.tuple
          row.append(s)
          col.append(t)
      data = np.ones(len(graph.es()))
    else:
      if transposed:
        for edge in graph.es():
            s,t = edge.tuple
            row.append(t)
            col.append(s)
      else:
        for edge in graph.es():
            s,t = edge.tuple
            row.append(s)
            col.append(t)
      data = np.array(graph.es()[attribute])

    return sparse.coo_matrix((data, (row, col)) , shape=(len(graph.vs), len(graph.vs))).tocsr()


def RWTransitionMatrix(g):
    """Generates a transposed random walk transition matrix corresponding to a (possibly) weighted
    and directed network
    
    @param g: the graph"""
    row = []
    col = []
    data = []
    if g.is_weighted():
      D = g.strength(mode='out', weights=g.es["weight"])
      for edge in g.es():
          s,t = edge.tuple
          row.append(t)
          col.append(s)
          tmp = edge["weight"] / D[s]
          if tmp <0 or tmp > 1:
              tn.Log.add('Encountered transition probability outside [0,1] range.', Severity.ERROR)
              raise ValueError()
          data.append( tmp )
    else:
      D = g.degree(mode='out')
      for edge in g.es():
          s,t = edge.tuple
          row.append(t)
          col.append(s)
          tmp = 1. / D[s]
          if tmp <0 or tmp > 1:
              tn.Log.add('Encountered transition probability outside [0,1] range.', Severity.ERROR)
              raise ValueError()
          data.append( tmp )
    
    # TODO: find out why data is some times of type (N, 1)
    # TODO: and sometimes of type (N,). The latter is desired
    # TODO: otherwise scipy.coo will raise a ValueError
    data = np.array(data)
    data = data.reshape(data.size,)

    return sparse.coo_matrix((data, (row, col)), shape=(len(g.vs), len(g.vs))).tocsr()


def EntropyGrowthRate(T):
    """Computes the entropy growth rate of a transition matrix
    
    @param T: Transition matrix in sparse format."""
    pi = StationaryDistribution(T)
    
    # directly work on the data object of the sparse matrix
    # NOTE: np.log2(T.data) has no problem with elements being zeros
    # NOTE: as we work with a sparse matrix here, where the zero elements
    # NOTE: are not stored
    T.data *=  np.log2(T.data)
    
    # NOTE: the matrix vector product only works because T is assumed to be
    # NOTE: transposed. This is needed for sla.eigs(T) to return the correct
    # NOTE: eigenvector anyway
    return -np.sum( T * pi )

def Entropy( prob ):
    idx = np.nonzero(prob)
    return -np.inner( np.log2(prob[idx]), prob[idx] )


def BWPrefMatrix(t, v):
    """Computes a betweenness preference matrix for a node v in a temporal network t
    
    @param t: The temporalnetwork instance to work on
    @param v: Name of the node to compute its BetweennessPreference
    """
    g = t.igraphFirstOrder()
    # NOTE: this might raise a ValueError if vertex v is not found
    v_vertex = g.vs.find(name=v)
    indeg = v_vertex.degree(mode="IN")        
    outdeg = v_vertex.degree(mode="OUT")
    index_succ = {}
    index_pred = {}
    
    B_v = np.zeros(shape=(indeg, outdeg))
        
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
  
  
def TVD(p1, p2):
    """Compute total variation distance between two stochastic column vectors"""
    assert p1.shape == p2.shape
    return 0.5 * np.sum(np.absolute(np.subtract(p1, p2)))


def StationaryDistribution( T, normalize=True ):
    """Compute normalized leading eigenvector of T (stationary distribution)

    @param T: (Transition) matrix in any sparse format
    @param normalize: wheter or not to normalize. Default is C{True}
    """
    if sparse.issparse(T) == False:
        raise TypeError("T must be a sparse matrix")
    # NOTE: ncv=13 sets additional auxiliary eigenvectors that are computed
    # NOTE: in order to be more confident to find the one with the largest
    # NOTE: magnitude, see
    # NOTE: https://github.com/scipy/scipy/issues/4987
    w, pi = sla.eigs( T, k=1, which="LM", ncv=13 )
    pi = pi.reshape(pi.size,)
    if normalize:
        pi /= sum(pi)
    return pi


def firstOrderNameMap( t ):
    """returns a name map of the first order network of a given temporal network t"""

    g1 = t.igraphFirstOrder()
    name_map = {}
    for idx,v in enumerate(g1.vs()["name"]):
        name_map[v] = idx
    return name_map
