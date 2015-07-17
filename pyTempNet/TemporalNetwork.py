# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:49:39 2015
@author: Ingo Scholtes

(c) Copyright ETH Zürich, Chair of Systems Design, 2015
"""

import igraph
import numpy as np
import os
from collections import defaultdict

from pyTempNet.Utilities import RWTransitionMatrix
from pyTempNet.Utilities import StationaryDistribution

class TemporalNetwork:
    """A class representing a temporal network consisting of a sequence of time-stamped edges"""
    
    def __init__(self,  sep=',', tedges = None, twopaths = None):
        """Constructor generating an empty temporal network"""
        
        self.separator = sep
        self.tedges = []
        self.nodes = []
        if tedges is not None:
            for e in tedges:
                self.tedges.append(e)            
        for e in self.tedges:
            source = e[0]
            target = e[1]
            if source not in self.nodes:
                self.nodes.append(source)
            if target not in self.nodes:
                self.nodes.append(target)
        self.twopaths = []
        self.twopathsByNode = defaultdict( lambda: dict() )
        self.twopathsByTime = defaultdict( lambda: dict() )
        self.tpcount = -1

        if twopaths is not None:
            t = 0
            for tp in twopaths:
                self.twopaths.append(tp)
                s = tp[0]
                v = tp[1]
                d = tp[2]

                if s not in self.nodes:
                    self.nodes.append(s)
                if v not in self.nodes:
                    self.nodes.append(v)
                if d not in self.nodes:
                    self.nodes.append(d)
  
                self.twopathsByNode[v].setdefault(t, []).append(tp)
                t +=1
            self.tpcount = len(twopaths)

        self.g1 = 0
        self.g2 = 0
        self.g2n = 0
        

    def addEdge(self, source, target, time):
        """Adds a (directed) time-stamped edge to the temporal network"""
        
        self.tedges.append((source, target, time))
        if source not in self.nodes:
            self.nodes.append(source)
        if target not in self.nodes:
            self.nodes.append(target)
            
        self.tpcount = -1

        
    def vcount(self):
        """Returns the number of vertices in the static network"""
        return len(self.nodes)

        
    def ecount(self):
        """Returns the number of time-stamped edges"""
        return len(self.tedges)


    def extractTwoPaths(self, delta=1):
        """Extracts all time-respecting paths of length two. The delta parameter indicates the maximum
        temporal distance below which two consecutive links will be considered as a time-respecting path.
        For (u,v,3) and (v,w,7) a time-respecting path (u,v)->(v,w) will be inferred for all delta 0 < 4, 
        while no time-respecting path will be inferred for all delta >= 4.
        """
        self.twopaths = []
        
        # An index structure to quickly access tedges by time, target and source
        time = defaultdict( lambda: list() )
        targets = defaultdict( lambda: dict() )
        sources = defaultdict( lambda: dict() )
        for e in self.tedges:
            source = e[0]
            target = e[1]
            ts = e[2]
            
            time[ts].append(e)
            targets[ts].setdefault(target, []).append(e)
            sources[ts].setdefault(source, []).append(e)

        # Extract time-respecting paths of length two             
        prev_t = -1
        for t in np.sort(list(time.keys())):
            if prev_t ==-1: 
                pass
            elif prev_t < t-delta:
                pass
            else:
                for v in targets[prev_t]:
                    if v in sources[t]:
                        for e_out in sources[t][v]:
                            for e_in in targets[prev_t][v]:
                                s = e_in[0]
                                d = e_out[1]
                                
                                # TODO: Add support for weighted time-
                                # TODO: stamped links
                                pass
                                indeg_v = len(targets[prev_t][v])
                                outdeg_v = len(sources[t][v])                                
                                
                                # Create a weighted two-path tuple
                                # (s, v, d, weight)
                                # representing (s,v) -> (v,d)
                                two_path = (s,v,d, float(1)/(indeg_v*outdeg_v))
                                self.twopaths.append(two_path)
                                
                                self.twopathsByNode[v].setdefault(t, []).append(two_path) 
                                self.twopathsByTime[t].setdefault(v, []).append(two_path)
            prev_t = t
        
        # Update the count of two-paths
        self.tpcount = len(self.twopaths)

        
    def TwoPathCount(self):
        """Returns the total number of time-respecting paths of length two which have
            been extracted from the time-stamped edge sequence."""
        
        # If two-paths have not been extracted yet, do it now
        if self.tpcount == -1:
            self.extractTwoPaths()

        return self.tpcount
    
    def igraphFirstOrder(self):
        """Returns the first-order time-aggregated network
           corresponding to this temporal network. This network corresponds to 
           a first-order Markov model reproducing the link statistics in the 
           weighted, time-aggregated network."""
        
        if self.g1 != 0:
            return self.g1
           
           
        # If two-paths have not been extracted yet, do it now
        if self.tpcount == -1:
            self.extractTwoPaths()

        self.g1 = igraph.Graph(n=len(self.nodes), directed=True)
        self.g1.vs["name"] = self.nodes
        
        # Gather all edges and their (accumulated) weights in a directory
        edge_list = {}
        for tp in self.twopaths:
            key1 = (tp[0], tp[1])
            key2 = (tp[1], tp[2])
            # get key{1,2} with default value 0 from edge_list directory
            edge_list[key1] = edge_list.get(key1, 0) + tp[3]
            edge_list[key2] = edge_list.get(key2, 0) + tp[3]
            
        # adding all edges at once is much faster as igraph updates internal
        # data structures after each vertex/edge added
        self.g1.add_edges( edge_list.keys() )
        self.g1.es["weight"] = list(edge_list.values())
        
        return self.g1


    def igraphSecondOrder(self):
        """Returns the second-order time-aggregated network
           corresponding to this temporal network. This network corresponds to 
           a second-order Markov model reproducing both the link statistics and 
           (first-order) order correlations in the underlying temporal network."""

        if self.g2 != 0:
            return self.g2

        if self.tpcount == -1:
            self.extractTwoPaths() 

        # create vertex list and edge directory first
        vertex_list = []
        edge_dict = {}
        for tp in self.twopaths:
            n1 = str(tp[0])+self.separator+str(tp[1])
            n2 = str(tp[1])+self.separator+str(tp[2])
            vertex_list.append(n1)
            vertex_list.append(n2)
            key = (n1, n2)
            edge_dict[key] = edge_dict.get(key, 0) + tp[3]
            
        # remove duplicate vertices by building a set
        vertex_list = list(set(vertex_list))
        
        # build 2nd order graph
        self.g2 = igraph.Graph( n=len(vertex_list), directed=True )
        self.g2.vs["name"] = vertex_list
        
        # add all edges in one go
        self.g2.add_edges( edge_dict.keys() )
        self.g2.es["weight"] = list(edge_dict.values())

        return self.g2


    def igraphSecondOrderNull(self):
        """Returns a second-order null Markov model 
           corresponding to the first-order aggregate network. This network
           is a second-order representation of the weighted time-aggregated network.           
           """
        if self.g2n != 0:
            return self.g2n

        g2 = self.igraphSecondOrder().components(mode='STRONG').giant()
        n_vertices = len(g2.vs)
        
        T = RWTransitionMatrix( g2 )
        pi = StationaryDistribution(T)
        
        # Construct null model second-order network
        self.g2n = igraph.Graph(directed=True)
        # NOTE: This ensures that vertices are ordered in the same way as in 
        # NOTE: the empirical second-order network
        for v in self.g2.vs():
            self.g2n.add_vertex(name=v["name"])
        
        ## TODO: This operation is the bottleneck for large data sets!
        ## TODO: Only iterate over those edge pairs, that actually are two paths!
        edge_dict = {}
        vertices = g2.vs()
        for i in range(n_vertices):
            e1 = vertices[i]
            e1name = e1["name"]
            a,b = e1name.split(self.separator)
            for j in range(i+1, n_vertices):
                e2 = vertices[j]
                e2name = e2["name"]
                a_,b_ = e2name.split(self.separator)
                
                # Check whether this pair of nodes in the second-order 
                # network is a *possible* forward two-path
                if b == a_:
                    w = pi[e2.index]
                    if w>0:
                        edge_dict[(e1name, e2name)] = w
                        
                if b_ == a:
                    w = pi[e1.index]
                    if w>0:
                        edge_dict[(e2name, e1name)] = w
        
        # add all edges to the graph in one go
        self.g2n.add_edges( edge_dict.keys() )
        self.g2n.es["weight"] = list(edge_dict.values())
        
        return self.g2n


    def ShuffleEdges(self, l=0):        
        """Generates a shuffled version of the temporal network in which edge statistics (i.e.
        the frequencies of time-stamped edges) are preserved, while all order correlations are 
        destroyed"""
        tedges = []
        
        if self.tpcount == -1:
            self.extractTwoPaths()
        
        if l==0:
            l = 2*int(len(self.tedges)/2)
        for i in range(l):
            # We simply shuffle the order of all edges
            edge = self.tedges[np.random.randint(0, len(self.tedges))]
            tedges.append((edge[0], edge[1], i))
        tn = TemporalNetwork(tedges)
        tn.nodes = self.nodes
            
        return tn
        
        
    def ShuffleTwoPaths(self, l=0):
        """Generates a shuffled version of the temporal network in which two-path statistics (i.e.
        first-order correlations in the order of time-stamped edges) are preserved"""
        
        tedges = []
        
        if self.tpcount == -1:
            self.extractTwoPaths()
        
        t = 0
        
        times = list(self.twopathsByTime.keys())        
        
        if l==0:
            l = len(self.tedges)
        for i in range(int(l/2)):
            # Chose a time uniformly at random
            rt = times[np.random.randint(0, len(self.twopathsByTime))]
            
            # Chose a node active at that time uniformly at random
            rn = list(self.twopathsByTime[rt].keys())[np.random.randint(0, len(self.twopathsByTime[rt]))]
            
            # Chose a two path uniformly at random
            paths = self.twopathsByTime[rt][rn]
            tp = paths[np.random.randint(0, len(paths))]
            
            tedges.append((tp[0], tp[1], t))
            tedges.append((tp[1], tp[2], t+1))
            
        tn = TemporalNetwork(tedges)
        return tn


    def exportTikzUnfolded(self, filename):
        """Generates a tikz file that can be compiled to obtain a time-unfolded 
            representation of the temporal network"""    
    
        # An index structure to quickly access tedges by time
        time = defaultdict( lambda: list() )
        for e in self.tedges:
            time[e[2]].append(e)
        
        output = []
            
        output.append('\\documentclass{article}\n')
        output.append('\\usepackage{tikz}\n')
        output.append('\\usepackage{verbatim}\n')
        output.append('\\usepackage[active,tightpage]{preview}\n')
        output.append('\\PreviewEnvironment{tikzpicture}\n')
        output.append('\\setlength\PreviewBorder{5pt}%\n')
        output.append('\\usetikzlibrary{arrows}\n')
        output.append('\\usetikzlibrary{positioning}\n')
        output.append('\\begin{document}\n')
        output.append('\\begin{center}\n')
        output.append('\\newcounter{a}\n')
        output.append("\\begin{tikzpicture}[->,>=stealth',auto,scale=0.5, every node/.style={scale=0.9}]\n")
        output.append("\\tikzstyle{node} = [fill=lightgray,text=black,circle]\n")
        output.append("\\tikzstyle{v} = [fill=black,text=white,circle]\n")
        output.append("\\tikzstyle{dst} = [fill=lightgray,text=black,circle]\n")
        output.append("\\tikzstyle{lbl} = [fill=white,text=black,circle]\n")

        last = ''
            
        for n in self.nodes:
            if last == '':
                output.append("\\node[lbl]                     (" + n + "-0)   {$" + n + "$};\n")
            else:
                output.append("\\node[lbl,right=0.5cm of "+last+"-0] (" + n + "-0)   {$" + n + "$};\n")
            last = n
            
        output.append("\\setcounter{a}{0}\n")
        output.append("\\foreach \\number in {1,...," + str(len(self.tedges)+2) + "}{\n")
        output.append("\\setcounter{a}{\\number}\n")
        output.append("\\addtocounter{a}{-1}\n")
        output.append("\\pgfmathparse{\\thea}\n")
        
        for n in self.nodes:
            output.append("\\node[v,below=0.3cm of " + n + "-\\pgfmathresult]     (" + n + "-\\number) {};\n")
        output.append("\\node[lbl,left=0.5cm of " + self.nodes[0] + "-\\number]    (col-\\pgfmathresult) {$t=$\\number};\n")
        output.append("}\n")
        output.append("\\path[->,thick]\n")
        i = 1
        
        for t in time:
            for edge in time[t]:
                output.append("(" + edge[0] + "-" + str(t+1) + ") edge (" + edge[1] + "-" + str(t + 2) + ")\n")
                i += 1                                
        output.append(";\n")
        output.append("""\end{tikzpicture}
\end{center}
\end{document}""")
        
        # create directory if necessary to avoid IO errors
        directory = os.path.dirname( filename )
        if not os.path.exists( directory ):
          os.makedirs( directory )
        
        text_file = open(filename, "w")
        text_file.write(''.join(output))
        text_file.close()
                    


    def exportMovie(self, output_file, visual_style = None, realtime = True, maxSteps=-1, delay=10):
        """Exports a video of the temporal network"""
        prefix = str(np.random.randint(0,10000))
        
        self.exportMovieFrames('frames\\' + prefix, visual_style = visual_style, realtime = realtime, maxSteps=maxSteps)
        
        from subprocess import call

        x = call("convert -delay " + str(delay) +" frames\\"+prefix+"_frame_* "+output_file, shell=True)

    def exportMovieFrames(self, fileprefix, visual_style = None, realtime = True, maxSteps=-1):
        """Exports an animation showing the temporal
           evolution of the network"""

        g = self.igraphFirstOrder()

        # An index structure to quickly access tedges by time
        time = defaultdict( lambda: list() )
        for e in self.tedges:
            time[e[2]].append(e)

        if visual_style == None:
            print('No visual style specified, setting to defaults')
            visual_style = {}
            visual_style["vertex_color"] = "lightblue"
            visual_style["vertex_label"] = g.vs["name"]
            visual_style["edge_curved"] = .5
            visual_style["vertex_size"] = 30
            
            # Use layout from first-order aggregate network
            visual_style["layout"] = g.layout_auto() 
        
        # make sure there is a directory for the frames to avoid IO errors
        directory = os.path.dirname(fileprefix)
        if not os.path.exists( directory ):
          os.makedirs( directory )
         
        i = 0
        # Generate movie frames
        if realtime == True:
            t_range = range(min(time.keys()), max(time.keys())+1)
        else:
            t_range = list(time.keys())

        if maxSteps>0:
            t_range = t_range[:maxSteps]

        for t in t_range:
            i += 1
            slice = igraph.Graph(n=len(g.vs()), directed=True)
            slice.vs["name"] = g.vs["name"]
            # this should work as time is a defaultdict
            for e in time[t]:
                slice.add_edge(e[0], e[1])
            igraph.plot(slice, fileprefix + '_frame_' + str(t).zfill(5) + '.png', **visual_style)
            if i % 100 == 0:
                print('Wrote movie frame', i, ' of', len(t_range))
