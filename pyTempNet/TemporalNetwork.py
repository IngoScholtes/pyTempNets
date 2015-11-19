# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:49:39 2015
@author: Ingo Scholtes, Roman Cattaneo

(c) Copyright ETH Zürich, Chair of Systems Design, 2015
"""

import igraph
import numpy as np
from collections import defaultdict

from bisect import bisect_right

from pyTempNet.Utilities import RWTransitionMatrix
from pyTempNet.Utilities import StationaryDistribution
from pyTempNet.Log import *

import time as tm

class EmptySCCError(Exception):
    """An exception that will be thrown whenever we require a non-empty strongly 
    connected component, but encounter an empty one"""
    pass

class TemporalNetwork:
    """A class representing a temporal network consisting of a sequence of 
    time-stamped edges"""
    
    def __init__(self, tedges, delta, sep):
        """Constructor generating a temporal network instance
        
        @param tedges: a (possibly empty) list of (possibly unordered time-stamped) links from 
            which to construct a temporal network instance
        @param sep: a separator character to be used for the naming of higher-
            order nodes v-w
        @param delta: maximal temporal distance up to which time-stamped
                      links will be considered to contribute to a time-
                      respecting path. Default: 1
        
        Note: If the max time diff is not set specifically, the default value of
        delta=1 will be used, meaning that a time-respecting path u -> v will
        only be inferred if there are *directly consecutive* time-stamped links
        (u,v;t) (v,w;t+1).
        """
        self.delta = delta
        self.tedges = list()
        nodes_seen = defaultdict( lambda:False )
        self.nodes = list()

        # Generate index structures which help to efficiently extract time-respecting paths

        # A dictionary storing all time-stamped links, indexed by time-stamps
        self.time = defaultdict( lambda: list() )

        # A dictionary storing all time-stamped links, indexed by time and target node
        self.targets = defaultdict( lambda: defaultdict(list) )

        # A dictionary storing all time-stamped links, indexed by time and source node 
        self.sources = defaultdict( lambda: defaultdict(list) )

        # A dictionary storing time stamps at which links (v,*;t) originate from node v
        self.activities = defaultdict( lambda: list() )

        # A dictionary storing sets of time stamps at which links (v,*;t) originate from node v
        # Note that the insertion into a set is much faster than repeatedly checking whether 
        # an element already exists in a list!
        self.activities_sets = defaultdict( lambda: set() )

        # An ordered list of time-stamps
        self.ordered_times = []

        # NOTE building index data structures can take some time for large data
        # NOTE sets. Consider merging this loop with the one from eiter
        # NOTE  - Utilities.readTimeStampedData()
        # NOTE  - Utilities.readNGramData()
        Log.add('Building index data structures ...')
        for e in tedges:
            self.activities_sets[e[0]].add(e[2])
            self.time[e[2]].append(e)
            self.targets[e[2]][e[1]].append(e)
            self.sources[e[2]][e[0]].append(e)
            if not nodes_seen[e[0]]:
                nodes_seen[e[0]] = True
            if not nodes_seen[e[1]]:
                nodes_seen[e[1]] = True
        self.tedges = tedges
        self.nodes = list(nodes_seen.keys())
        Log.add('finished.')

        Log.add('Sorting time stamps ...')

        self.ordered_times = sorted(self.time.keys())
        for v in self.nodes:
            self.activities[v] = sorted(self.activities_sets[v])
        Log.add('finished.')

        """The separator character to be used to generate higher-order nodes"""
        self.separator = sep
      
    def filterEdges(self, edge_filter):
        """Allows to filter time-stamped edges according to a given filter 
        expression. 

        @param edge_filter: an arbitrary (lambda) expression of the form 
            filter_func(v, w, time) that returns True for time-stamped edges
            that shall pass the filter, and False for all edges that shall be
            filtered out.
            Note that for the purpose of filtering, data structures such as the
            activities dictionary, the first- or the second-order aggregate 
            networks of the TemporalNetwork instance can be used. 
        """

        Log.add('Starting filtering ...', Severity.INFO)
        new_t_edges = []

        for (v,w,t) in self.tedges:
            if edge_filter(v,w,t):
                new_t_edges.append((v,w,t))

        Log.add('finished. Filtered out ' + str(self.ecount() - len(new_t_edges)) + ' time-stamped edges.', Severity.INFO)

        return TemporalNetwork(self.separator, new_t_edges, None)


    def addEdge(self, source, target, ts):
        """Adds a directed time-stamped edge (source,target;time) to the 
        temporal network. To add an undirected time-stamped link (u,v;t) at
        time t, please call addEdge(u,v;t) and addEdge(v,u;t).
        
        @param source: name of the source node of a directed, time-stamped link
        @param target: name of the target node of a directed, time-stamped link
        @param ts: (integer) time-stamp of the time-stamped link
        """
        e = (source, target, ts)
        self.tedges.append(e)
        if source not in self.nodes:
            self.nodes.append(source)
        if target not in self.nodes:
            self.nodes.append(target)

        # Add edge to index structures
        self.time[ts].append(e)
        self.targets[ts].setdefault(target, []).append(e)
        self.sources[ts].setdefault(source, []).append(e)

        if ts not in self.activities[source]:
            self.activities[source].append(ts)
            self.activities[source].sort()

        # Reorder time stamps
        self.ordered_times = sorted(self.time.keys())


    def vcount(self):
        """Returns the total number of different vertices active across the 
        whole evolution of the temporal network. 
        This number corresponds to the number of nodes in the first-order 
        time-aggregated network.
        """
        return len(self.nodes)

        
    def ecount(self):
        """Returns the number of time-stamped edges (u,v;t) in this temporal 
        network"""
        return len(self.tedges)


    def getObservationLength(self):
        """Returns the length of the observation time, i.e. the difference 
        between the maximum and minimum time stamp of any time-stamped link.
        """
        return max(self.ordered_times)-min(self.ordered_times)
    

    def getInterEventTimes(self):
        """Returns a numpy array containing all time differences between any 
            two consecutive time-stamped links (involving any node)"""

        timediffs = []
        for i in range(1, len(self.ordered_times)):
            timediffs += [self.ordered_times[i] - self.ordered_times[i-1]]
        return np.array(timediffs)


    def getInterPathTimes(self):
        """Returns a dictionary which, for each node v, contains all time differences 
            between any time-stamped link (*,v;t) and the next link (v,*;t') (t'>t)
            in the temporal network"""

        interPathTimes = defaultdict( lambda: list() )
        for e in self.tedges:
            # Get target v of current edge e=(u,v,t)
            v = e[1]
            t = e[2]

            # Get time stamp of link (v,*,t_next) with smallest t_next such that t_next > t
            i = bisect_right(self.activities[v], t)
            if i != len(self.activities[v]):
                interPathTimes[v].append(self.activities[v][i]-t)
        return interPathTimes


    def Summary(self):
        """Returns a string containing basic summary statistics of this temporal network"""

        summary = ''

        summary += 'Nodes:\t\t\t' +  str(self.vcount()) + '\n'
        summary += 'Time-stamped links:\t' + str(self.ecount()) + '\n'
        summary += 'Links/Nodes:\t\t' + str(self.ecount()/self.vcount()) + '\n'
        summary += 'Delta:\t\t\t' + str(self.delta) + '\n'
        summary += 'Observation period:\t[' + str(min(self.ordered_times)) + ', ' + str(max(self.ordered_times)) + ']\n'
        summary += 'Observation length:\t' + str(max(self.ordered_times) - min(self.ordered_times)) + '\n'
        summary += 'Time stamps:\t\t' + str(len(self.ordered_times)) + '\n'

        d = self.getInterEventTimes()
    
        summary += 'Avg. inter-event dt:\t' + str(np.mean(d)) + '\n'
        summary += 'Min/Max inter-event dt:\t' + str(min(d)) + '/' + str(max(d)) + '\n'
        
        return summary


    def ShuffleEdges(self, l=0):        
        """Generates a shuffled version of the temporal network in which edge 
        statistics (i.e.the frequencies of time-stamped edges) are preserved, 
        while all order correlations are destroyed. The shuffling procedure 
        randomly reshuffles the time-stamps of links.
        
        @param l: the length of the sequence to be generated (in terms of the 
        number of time-stamped links.
        
        For the default value l=0, the length of the generated shuffled temporal
        network will be equal to that of the original temporal network. 
        """
        tedges = []
        
        if l==0:
            l = 2*int(len(self.tedges)/2)
        for i in range(l):
            # Pick random link
            edge = self.tedges[np.random.randint(0, len(self.tedges))]
            # Pick random time stamp
            time = self.ordered_times[np.random.randint(0, len(self.ordered_times))]
            # Generate new time-stamped link
            tedges.append( (edge[0], edge[1], time) )

        # Generate temporal network
        t = TemporalNetwork(tedges, sep=self.separator)

        # Fix node order to correspond to original network
        t.nodes = self.nodes
            
        return t
        
        
    #def ShuffleTwoPaths(self, l=0):
        #"""Generates a shuffled version of the temporal network in which two-path statistics (i.e.
        #first-order correlations in the order of time-stamped edges) are preserved
        
        #@param l: the length of the sequence to be generated (in terms of the number of time-stamped links.
            #For the default value l=0, the length of the generated shuffled temporal network will be equal to that of 
            #the original temporal network. 
        #"""
        
        #tedges = []
        
        #if self.tpcount == -1:
            #self.extractTwoPaths()
        
        #t = 0
        
        #times = list(self.twopathsByTime.keys())
        
        #if l==0:
            #l = len(self.tedges)
        #for i in range(int(l/2)):
            ## Chose a time uniformly at random
            #rt = times[np.random.randint(0, len(self.twopathsByTime))]
            
            ## Chose a node active at that time uniformly at random
            #rn = list(self.twopathsByTime[rt].keys())[np.random.randint(0, len(self.twopathsByTime[rt]))]
            
            ## Chose a two path uniformly at random
            #paths = self.twopathsByTime[rt][rn]
            #tp = paths[np.random.randint(0, len(paths))]
            
            #tedges.append((tp[0], tp[1], t))
            #t += 1
            #tedges.append((tp[1], tp[2], t))
            #t += 1
            
        #tempnet = TemporalNetwork(sep=',', tedges=tedges)
        #return tempnet