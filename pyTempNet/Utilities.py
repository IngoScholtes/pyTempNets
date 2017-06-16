# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:48:48 2015
@author: Ingo Scholtes, Roman Cattaneo

(c) Copyright ETH Zürich, Chair of Systems Design, 2015
"""

import sys
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sla

from collections import defaultdict

import itertools

import pyTempNet as tn
import datetime as dt

from pyTempNet.Log import *

import sys

def readFile(filename, sep=',', fformat="TEDGE", timestampformat="%s", maxlines=sys.maxsize):
    """ Reads time-stamped edges from TEDGE or TRIGRAM file. If fformat is TEDGES,
        the file is expected to contain lines in the format 'v,w,t' each line 
        representing a directed time-stamped link from v to w at time t.
        Semantics of columns should be given in a header file indicating either 
        'node1,node2,time' or 'source,target,time' (in arbitrary order).
        If fformat is TRIGRAM the file is expected to contain lines in the format
        'u,v,w' each line representing a time-respecting path (u,v) -> (v,w) consisting 
        of two consecutive links (u,v) and (v,w). Timestamps can be integer numbers or
        string timestamps (in which case the timestampformat string is used for parsing)
    """
    
    assert filename is not ""
    assert (fformat is "TEDGE") or (fformat is "TRIGRAM")
    
    with open(filename, 'r') as f:
        tedges = []
        twopaths = []
        
        header = f.readline()
        header = header.split(sep)

        # We support arbitrary column ordering, if header columns are included
        time_ix = -1
        source_ix = -1
        mid_ix = -1
        weight_ix = -1
        target_ix = -1
        if fformat =="TEDGE":
            for i in range(len(header)):
                header[i] = header[i].strip()
                if header[i] == 'node1' or header[i] == 'source':
                    source_ix = i
                elif header[i] == 'node2' or header[i] == 'target':
                    target_ix = i
                elif header[i] == 'time' or header[i] == 'timestamp':
                    time_ix = i
        elif fformat =="TRIGRAM":
            # For trigram files, we assume a default of (unweighted) trigrams in the form source;mid;target
            # Any other ordering, as well as the additional inclusion of weights requires the definition of 
            # column headers in the data file!
            source_ix = 0
            mid_ix = 1
            target_ix = 2
            for i in range(len(header)):
                header[i] = header[i].strip()
                if header[i] == 'node1' or header[i] == 'source':
                    source_ix = i                
                elif header[i] == 'node2' or header[i] == 'mid':
                    mid_ix = i
                elif header[i] == 'node3' or header[i] == 'target':
                    target_ix = i
                elif header[i] == 'weight':
                    weight_ix = i    

        assert( (source_ix >= 0 and target_ix >= 0) or
                (source_ix >= 0 and mid_ix >= 0 and target_ix >= 0 and weight_ix >= 0)), "Detected invalid header columns: %s" % header

        if time_ix<0:
            Log.add('No time stamps found in data, assuming consecutive links', Severity.WARNING)
        
        # Read time-stamped links
        if fformat == "TEDGE":
            Log.add('Reading time-stamped links ...')
        else:
            Log.add('Reading trigram data ...')

        line = f.readline()
        n = 1 
        while line and n <= maxlines:
            fields = line.rstrip().split(sep)
            if fformat =="TEDGE":
                try:
                    if time_ix >=0:
                        timestamp = fields[time_ix]            
                        if timestamp.isdigit():
                            t = int(timestamp)
                        else:
                            x = dt.datetime.strptime(timestamp, "%Y-%m-%d %H:%M")
                            t = int(time.mktime(x.timetuple()))
                    else:
                        t = n                
                    if t>=0:
                        tedge = (fields[source_ix], fields[target_ix], t)
                        tedges.append(tedge)
                    else:
                        Log.add('Ignoring negative timestamp in line ' + str(n+1) + ': "' + line.strip() + '"', Severity.WARNING)
                except (IndexError, ValueError):
                    Log.add('Ignoring malformed data in line ' + str(n+1) + ': "' +  line.strip() + '"', Severity.WARNING)

            elif fformat =="TRIGRAM":
                source = fields[source_ix].strip('"')
                mid = fields[mid_ix].strip('"')
                target = fields[target_ix].strip('"')
                if weight_ix >=0: 
                    weight = float(fields[weight_ix].strip('"'))
                else:
                    weight = 1
                tp = (source, mid, target, weight)
                twopaths.append(tp)

            line = f.readline()
            n += 1
    # end of with open()
    
    Log.add('finished.')
    if fformat == "TEDGE":        
        return tn.TemporalNetwork(tedges = tedges, sep=sep)
    elif fformat =="TRIGRAM":
        # If trigram data did not contain a weight column, we aggregate
        # multiple occurrences to weighted trigrams
        if weight_ix < 0:            
            Log.add('Calculating trigram weights ...')
            tp_dict = defaultdict( lambda: 0)
            for trigram in twopaths:
                tp = (trigram[0], trigram[1], trigram[2])
                tp_dict[tp] = tp_dict[tp] + trigram[3]
            twopaths = []
            for tp in tp_dict.keys():
                twopaths.append((tp[0], tp[1], tp[2], tp_dict[tp]))
            Log.add('finished.')
        return tn.TemporalNetwork(twopaths = twopaths, sep=sep)


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


def Entropy_Miller( prob, K, N ): 
    """ Computes a Miller-corrected MLE estimation of the entropy
    @param prob: the observed probabilities (i.e. relative frequencies) 
    @param K: the number of possible outcomes, i.e. outcomes with non-zero probability
    @param N: size of the sample based on which relative frequencies have been computed
    """
    
    if N == 0:
        return 0
    else:
        idx = np.nonzero(prob)
        return -np.inner( np.log2(prob[idx]), prob[idx] ) + (K-1)/(2*N)


def Entropy( prob ):
    """ Computes a naive MLE estimation of the entropy
    @param prob: the observed probabilities (i.e. relative frequencies)    
    """

    idx = np.nonzero(prob)
    return -np.inner( np.log2(prob[idx]), prob[idx] )
  
  
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

def getPossibleTwoPaths(edges):
    """Returns the list of different two-paths that can be constructed from edges""" 
    twopaths = [tp for tp in itertools.combinations(edges, 2) if tp[0][1] == tp[1][0]]
    return twopaths


def firstOrderNameMap( t ):
    """returns a name map of the first order network of a given temporal network t"""

    g1 = t.igraphFirstOrder()
    name_map = {}
    for idx,v in enumerate(g1.vs()["name"]):
        name_map[v] = idx
    return name_map
