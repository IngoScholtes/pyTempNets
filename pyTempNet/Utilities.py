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

import pyTempNet as tn
import datetime as dt

from pyTempNet.Log import *

import sys

def readTimeStampedEdges(filename, sep=',', timestampformat="%Y-%m-%d %H:%M",
                         delta=1, maxlines=sys.maxsize, skip=0):
    """ Reads time-stamped edges from a file.

        The file is expected to have one time-stamped link per line. Elements
        of the link (source, target, timestamp) have to be given in a header
        line indicating either 'source,target,timestamp' or 'node1,node2,time'
        in arbitrary order.
        @param filename: path to file. Only read permissions are necessary
        @param sep: optional separator. Default: comma separated values
        @param teimestapformat: Timestamps can be integer numbers or string
        timestamps, in which case this string is used for parsing.
        @param skip: number of lines to be skipped at the
        beginning of the file, after the header line. optional
        @param maxlines: maximal number of lines to be read
        @param delta: maximal temporal distance up to which time-stamped
                      links will be considered to contribute to a time-
                      respecting path. Default: 1

        Note: If the max time diff is not set specifically, the default value of
        delta=1 will be used, meaning that a time-respecting path u -> v will
        only be inferred if there are *directly consecutive* time-stamped links
        (u,v;t) (v,w;t+1).
    """

    assert filename is not ""
    tedges = list()

    # NOTE with open(...) opens files by default read only mode
    # NOTE plus files are automatically closed
    with open(filename) as f:
        header = f.readline()
        header = header.split(sep)
        # Support for arbitrary column ordering
        time_ix = -1
        source_ix = -1
        target_ix = -1

        for i in range(len(header)):
            name = header[i].strip()
            if name == 'node1' or name == 'source':
                source_ix = i
            elif name == 'node2' or name == 'target':
                target_ix = i
            elif name == 'time' or name == 'timestamp':
                time_ix = i

        assert( source_ix >= 0 and target_ix >= 0 and time_ix >=0 ), "Detected invalid header columns: %s" % header

        # Read time-stamped links
        Log.add('Reading time-stamped links ...')

        line = f.readline()
        n = 0
        while not line.strip() == '' and n < maxlines+skip:
            if n >= skip:
                fields = line.rstrip().split(sep)
                try:
                    timestamp = fields[time_ix]
                    if (timestamp.isdigit()):
                        t = int(timestamp)
                    else:
                        x = dt.datetime.strptime(timestamp, timestampformat)
                        t = int(time.mktime(x.timetuple()))

                    if t>=0:
                        tedge = (fields[source_ix], fields[target_ix], t)
                        tedges.append(tedge)
                    else:
                        Log.add('Ignoring link with negative timestamp in line ' + str(n+1) + ': "' + line.strip() + '"', Severity.WARNING)
                except (IndexError, ValueError):
                    Log.add('Ignoring malformed data in line ' + str(n+1) + ': "' +  line.strip() + '"', Severity.WARNING)
            line = f.readline()
            n += 1

    Log.add('finished.')
    return tn.TemporalNetwork(tedges, delta, sep)

def readNGramData(filename, sep=',', maxlines=sys.maxsize, skip=0):
    """ Reads weighted time-respecting paths from a TRIGRAM file, where each
        line describes a weighted twopath (source,mid,taret,weight).

        A header line is assumed marking the ordering of the columns using the
        keywords: {source,node1}, {mid,node2}, {target,node3}, weight

        @param filename: path to file. Only read permissions are necessary
        @param sep: optional separator. Default: comma separated values
        @param maxlines: maximal number of lines to be read. optional.
        @param skip: number of lines to be skipped at the
        beginning of the file, after the header line. optional.
    """

    # more general docstring for nGRAM files (to be done)
    #""" Reads weighted paths from a general nGRAM file. nGRAM files are gene-
        #ralized TRIGRAM files, where each line describes an arbitrary long
        #weighted time-respecting path.

        #As a consequence of the possibly changing path length, no header line
        #is assumed as the format is fixed to
            #n1,n2,n3,...,weight
        #denoting a time-respecting path n1->n2->n3->... with corresponding
        #weight in the last row.

        #@param filename: path to file. Only read permissions are necessary
        #@param sep: optional separator. Default: comma separated values
        #@param maxlines: maximal number of lines to be read. optional.
        #@param skip: number of lines to be skipped at the
        #beginning of the file, after the header line. optional.
    #"""
    
    assert filename is not ""
    tedges = list()
    delta = 1
    
    with open(filename) as f:
        header = f.readline()
        header = header.split(sep)
        # Support for arbitrary column ordering
        source_ix = -1
        mid_ix = -1
        weight_ix = -1
        target_ix = -1
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

        assert( source_ix >= 0 and mid_ix >= 0 and target_ix >= 0 and weight_ix >= 0 ), "Detected invalid header columns: %s" % header

        # Read time-stamped links
        Log.add('Reading time-stamped links ...')

        line = f.readline()
        n = 0
        counter = 0
        while not line.strip() == '' and n < maxlines+skip:
            if n >= skip:
                fields = line.rstrip().split(sep)

                source = fields[source_ix].strip('"')
                mid = fields[mid_ix].strip('"')
                target = fields[target_ix].strip('"')
                weight = float(fields[weight_ix].strip('"'))
                e1 = (source, mid, counter)
                counter += delta
                e2 = (mid, target, counter)
                counter += (delta+1)
                tedges.append(e1)
                tedges.append(e2)

            line = f.readline()
            n += 1

    Log.add('finished.')
    return tn.TemporalNetwork(tedges, delta, sep)

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
