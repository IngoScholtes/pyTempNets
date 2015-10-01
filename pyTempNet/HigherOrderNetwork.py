# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:47:31 CEST 2015
@author: Ingo Scholtes, Roman Cattaneo

(c) Copyright ETH Zürich, Chair of Systems Design, 2015
"""

class HigherOrderNetwork:
    """A class representing higher order networks of a given temporal network 
    instance"""
    
    def __init__(self, tempNet, order, maxTimeDiff=1):
        """Constructs an aggregated temporal network of order k
        
        @param tn:          a temporal network instance
        @param order:       order of the aggregated network, length of time 
                            respecting paths.
        @param maxTimeDiff: maximal temporal distance up to which time-stamped 
                            links will be considered to contribute to a time-
                            respecting path.
        
        Note: If the max time diff is not set specifically, the default value of 
        delta=1 will be used, meaning that a time-respecting path u -> v will 
        only be inferred if there are *directly consecutive* time-stamped links
        (u,v;t) (v,w;t+1).
        """
        
        self.tn    = tempNet
        self.k     = order
        self.delta = maxTimeDiff
        
        # time-respecting k-paths and their count
        self.kpaths = []
        self.kpcount = 0
        
        # do not extract k-paths as long as they are not needed
        ## TODO: extract k-path with respect to delta
        #self.extractKPaths()
        
        # order k aggregated network
        self.gk  = 0
        self.gkn = 0
        
        
    def clearCache(self):
        """Clears the cached information: K-paths, aggregated k-th order 
        network and null model"""
        self.kpaths = []
        self.kpcount = 0
        
        self.gk  = 0
        self.gkn = 0
        Log.add("Cache cleared.", Severity.INFO)
        
        
    def setMaxTimeDiff(self, delta):
        """Sets the maximum time difference delta between consecutive links to
        be used for the extraction of time-respecting paths of length k, so 
        called k-paths.
        Note: If k-path structures and/or kth-order networks have previously 
        been computed, this method will invalidate all cached data if the new
        delta is different from the old one (for which k-path statistics have 
        been computed)

        @param delta: Indicates the maximum temporal distance up to which 
        time-stamped links will be considered to contribute to time-
        respecting paths. For (u,v;3) and (v,w;7) a time-respecting path 
        (u,v)->(v,w) will be inferred for all 0 < delta <= 4, while no 
        time-respecting path will be inferred for all delta > 4. 
        If the max time diff is not set specifically, the default value of 
        delta=1 will be used, meaning that a time-respecting path u -> v will 
        only be inferred if there are *directly consecutive* time-stamped 
        links (u,v;t) (v,w;t+1).
        """
        
        if delta != self.delta:
            # Set new value and invalidate two-path structures
            Log.add("Changing maximal time difference from " + str(self.data) 
                    + " to " + str(delta), Severity.INFO)
            self.delta = delta
            self.clearCache()


    def setOrder(self, k):
        """Changes the order of the aggregated temporal network and therefore 
        the length of time-respecting paths (k-paths).
        
        Note: If k-path structures and/or kth-order networks have previously 
        been computed, this method will invalidate all cached data if the new 
        order is different from the old one (for which k-path statistics have 
        been computed)
        """
        
        if k != self.k:
            # Set new value and invalidate any cached data
            Log.add("Changeing order of aggregated network from " + str(self.k) 
                    + " to " + k, Severity.k)
            self.k = k
            self.clearCache()


    def KPathCount(self):
        """Returns the total number of time-respecting paths of length k 
        (so called k-paths) which have been extracted from the temporal 
        network."""
        
        # If two-paths have not been extracted yet, do it now
        if self.kpcount == -1:
            self.extractKPaths()

        return self.kpcount


    def order(self):
        """Returns the order, k, of the aggregated network"""
        return k


    def maxTimeDiff(self):
        """Returns the maximal time difference, delta, between consecutive 
        links in the temporal network"""
        return delta
    
    
    def extractKPaths():
        """Extracts all time-respecting paths of length k in this temporal 
        network for the currently set maximum time difference delta. The 
        k-paths extracted by this method will be used in the construction of 
        higher-order time-aggregated networks, as well as in the analysis of 
        causal structures of this temporal network. If an explicit call to this
        method is omitted, it will be run whenever k-paths are needed for the 
        first time.
        
        Once k-paths have been computed, they will be cached and reused until 
        the maximum time difference, delta, and/or the order k is changed.        
        """
        
        tmpNet = self.tn
        
        #loop over all time-steps (at which something is happening)
        for t in range(tmpNet.ordered_times):
            possible_path = defaultdict( lambda: list() )
            candidates = set()
            
            # case k == 0
            current_edges = tmpNet.time[t]
            for e in current_edges:
                possible_path[e[1]].append( [e[0], e[1]] )
                candidates.add(e[1])
                
            # 1 <= current_k <= k
            for current_k in range(1, self.k+1):
                new_candidates = set()
                
                for node in candidates:
                    # edges at time t+1 originating from node
                    new_edges = tmpNet.sources[t+current_k][node]
                    for e in new_edges:
                        src = e[0]
                        dst = e[1]
                        for path in possible_path[src]:
                            possible_path[dst].append( path.append(dst) )
                            new_candidates.add( dst )
                            if current_k == self.k:
                                self.kpaths.append( path.append(dst) )
                
                candidates = new_candidates
            
            # NOTE: possible_path will hold all k-paths for 1 <= k <= self.k at
            # NOTE: this time-step
        
        self.KPathCount = len(self.kpaths)
        
        return self.kpaths
    
    
    def igraphKthOrder(self):
        """Returns the kth-order time-aggregated network
           corresponding to this temporal network. This network corresponds to 
           a kth-order Markov model reproducing both the link statistics and 
           (first-order) order correlations in the underlying temporal network.
           """
           
           # TODO
           # self.gk = ...
           
           return 0
       
       
    def igraphKthOrderNull(self):
        """Returns a kth-order null Markov model 
           corresponding to the first-order aggregate network. This network
           is a kth-order representation of the weighted time-aggregated network.
           
           Note: In order to compute the null model, the strongly connected 
           component of the kth-order network needs to have at least two nodes.          
           """
           
           # TODO
           # self.gkn = ...
           
           # NOTE: pay attention to the fact, that a null model only makes sense for k > 1.
           
           return 0