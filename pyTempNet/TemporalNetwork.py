# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:49:39 2015

@author: Ingo Scholtes
"""

import igraph    
import datetime as dt
import time as tm
import numpy as np
import scipy.linalg as spl
import os

class TemporalNetwork:
    """A class representing a temporal network consisting of a sequence of time-stamped edges"""
    
    def __init__(self, tedges = None, twopaths = None):
        """Constructor generating an empty temporal network"""
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
        self.twopathsByNode = {}
        self.twopathsByTime = {}
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

                try:
                    x = self.twopathsByNode[v]
                except KeyError:
                    # Generate dictionary indexed by (artificial) time stamps
                    self.twopathsByNode[v] = {}
                try:
                    self.twopathsByNode[v][t].append(tp)
                except KeyError:
                    # Generate list taking all two paths through node v at (artificial) time t
                    self.twopathsByNode[v][t] = [tp]
                t +=1
            self.tpcount = len(twopaths)

        self.g1 = 0
        self.g2 = 0
        self.g2n = 0
        
        
    def readFile(self, filename, sep=',', fformat="TEDGE", timestampformat="%s"):
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
        assert fformat is "TEDGE" or "TRIGRAM"
        
        f = open(filename, 'r')
        tedges = []
        twopaths = []
        
        header = f.readline()
        header = header.split(sep)
        
        # Support for arbitrary column ordering
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
                elif header[i] == 'time':
                    time_ix = i
        elif fformat =="TRIGRAM":
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
        
        # Read time-stamped edges
        line = f.readline()
        while not line.strip() == '':
            fields = line.rstrip().split(sep)
            if fformat =="TEDGE":
                timestamp = fields[time_ix]
                if (timestamp.isdigit()):
                    t = int(timestamp)
                else:
                    x = dt.datetime.strptime(timestamp, "%Y-%m-%d %H:%M")
                    t = int(time.mktime(x.timetuple()))

                tedge = (fields[source_ix], fields[target_ix], t)
                tedges.append(tedge)
            elif fformat =="TRIGRAM":             
                source = fields[source_ix].strip('"')
                mid = fields[mid_ix].strip('"')
                target = fields[target_ix].strip('"')
                weight = float(fields[weight_ix].strip('"'))
                tp = (source, mid, target, weight)
                twopaths.append(tp)

            line = f.readline()
        if fformat == "TEDGE":
            return TemporalNetwork(tedges = tedges)
        elif fformat =="TRIGRAM":           
            return TemporalNetwork(twopaths = twopaths)



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
        start = tm.clock()
        
        self.twopaths = []
        
        # An index structure to quickly access tedges by time
        time = {}
        for e in self.tedges:
            try:
                time[e[2]].append(e)
            except KeyError:
                time[e[2]] = [e]

        # Index structures to quickly access tedges by target/source
        targets = {}
        sources = {}
        for e in self.tedges:            
            source = e[0]
            target = e[1]
            ts = e[2]
            
            try:
                t = targets[ts]
            except KeyError:
                targets[ts] = dict()
            try:
                targets[ts][target].append(e)
            except KeyError:
                targets[ts][target] = [e]
                
            try:
                s = sources[ts]
            except KeyError:
                sources[ts] = dict()
            try:
                sources[ts][source].append(e)
            except KeyError:
                sources[ts][source] = [e]
                

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
                                # stamped links
                                pass
                                indeg_v = len(targets[prev_t][v])
                                outdeg_v = len(sources[t][v])                                
                                
                                # Create a weighted two-path tuple
                                # (s, v, d, weight)
                                # representing (s,v) -> (v,d)
                                two_path = (s,v,d, float(1)/(indeg_v*outdeg_v))
                                self.twopaths.append(two_path)                                
                                
                                # fast solution with try/except
                                try:
                                    x = self.twopathsByNode[v]
                                except KeyError:
                                    # Generate dictionary indexed by time stamps
                                   self.twopathsByNode[v] = {}

                                try:
                                    self.twopathsByNode[v][t].append(two_path)
                                except KeyError:
                                    # Generate list taking all two paths through node v at time t
                                    self.twopathsByNode[v][t] = [two_path]
                                    
                                try:
                                    x = self.twopathsByTime[t]
                                except KeyError:
                                    # Generate dictionary indexed by time stamps
                                   self.twopathsByTime[t] = {}

                                try:
                                    self.twopathsByTime[t][v].append(two_path)
                                except KeyError:
                                    # Generate list taking all two paths through node v at time t
                                    self.twopathsByTime[t][v] = [two_path]
            prev_t = t
        
        # Update the count of two-paths
        self.tpcount = len(self.twopaths)
        
        end = tm.clock()
        end = end - start
        print("Time elapsed in extractTwoPaths(): %1.2f" % end)

        
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
        
        start = tm.clock()
        
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
        self.g1.es["weight"] = edge_list.values()
        
        end = tm.clock() - start
        print("Time spent in igraphFirstOrder(): %1.2f" % end)
        return self.g1


        
    def igraphSecondOrderLegacy(self):
        """Returns the second-order time-aggregated network
           corresponding to this temporal network. This network corresponds to 
           a second-order Markov model reproducing both the link statistics and 
           (first-order) order correlations in the underlying temporal network.
           
           NOTE: This code is substantially slower than the version below.
           Edgelist with corresponding weights are the same, but the measures
           * Entropy growth rate ratio
           * Analytical slow-down factor for diffusion
           * Algebraic Connectivity (G2)
           * Algebraic Connectivity (G2 null)
           give slightly varied results. They are however equivalent up to e-11"""
        
        start = tm.clock()
        
        if self.g2 != 0:
            return self.g2

        if self.tpcount == -1:
            self.extractTwoPaths() 

        # Index dictionaries to speed up network construction
        # (circumventing inefficient igraph operations to check 
        # whether nodes or edges exist)
        vertices = {}
        edges = {}       
        
        self.g2 = igraph.Graph(directed=True)
        self.g2.vs["name"] = []
        for tp in self.twopaths:            
            n1 = str(tp[0])+";"+str(tp[1])
            n2 = str(tp[1])+";"+str(tp[2])
            try:
                x = vertices[n1]
            except KeyError:                
                self.g2.add_vertex(name=n1)
                vertices[n1] = True
            try:
                x = vertices[n2]
            except KeyError:                
                self.g2.add_vertex(name=n2)
                vertices[n2] = True
            try:
                x = edges[n1+n2]
                self.g2.es()[edges[n1+n2]]["weight"] += tp[3]
            except KeyError:
                edges[n1+n2] = len(self.g2.es())
                self.g2.add_edge(n1, n2, weight=tp[3])
            
        end = tm.clock() - start
        print("Time elapsed in igraphSecondOrderLegacy(): %1.2f" % end)
        return self.g2
      
      
    def igraphSecondOrder(self):
        """Returns the second-order time-aggregated network
           corresponding to this temporal network. This network corresponds to 
           a second-order Markov model reproducing both the link statistics and 
           (first-order) order correlations in the underlying temporal network."""
        
        start = tm.clock()
        
        if self.g2 != 0:
            return self.g2

        if self.tpcount == -1:
            self.extractTwoPaths() 

        # create vertex list and edge directory first
        vertex_list = []
        edge_dict = {}
        for tp in self.twopaths:
            n1 = str(tp[0])+";"+str(tp[1])
            n2 = str(tp[1])+";"+str(tp[2])
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
        self.g2.es["weight"] = edge_dict.values()

        end = tm.clock() - start
        print("Time elapsed in igraphSecondOrder(): %1.2f" % end)
        return self.g2


        
    def igraphSecondOrderNull(self):
        """Returns a second-order null Markov model 
           corresponding to the first-order aggregate network. This network
           is a second-order representation of the weighted time-aggregated network.           
           """

        start = tm.clock()
        if self.g2n != 0:
            return self.g2n

        g2 = self.igraphSecondOrder().components(mode='STRONG').giant()

        # Compute stationary distribution to obtain expected edge weights in pi
        A = np.matrix(list(g2.get_adjacency(attribute='weight', default=0)))
        D = np.diag(g2.strength(mode='out', weights=g2.es["weight"]))

        T = np.zeros(shape=(len(g2.vs), len(g2.vs)))
    
        for i in range(len(g2.vs)):
            for j in range(len(g2.vs)):       
                T[i,j] = A[i,j]/D[i,i]
                assert T[i,j]>=0 and T[i,j] <= 1

        w, v = spl.eig(T, left=True, right=False)
        pi = v[:,np.argsort(-w)][:,0]
        pi = np.real(pi/sum(pi))

        
        # Construct null model second-order network
        self.g2n = igraph.Graph(directed=True)

        # This ensures that vertices are ordered in the same way as in the empirical second-order network
        for v in self.g2.vs():
            self.g2n.add_vertex(name=v["name"])
        
        # TODO: This operation is the bottleneck for large data sets!
        # TODO: Only iterate over those edge pairs, that actually are two paths!
        for e1 in g2.vs():
            for e2 in g2.vs():
                b = e1["name"].split(';')[1]
                b_ = e2["name"].split(';')[0]

                # Check whether this pair of nodes in the second-order 
                # network is a *possible* two-path
                if b == b_:
                    w = pi[e2.index]
                    if w>0:
                        self.g2n.add_edge(e1["name"], e2["name"], weight = w)
        end = tm.clock()
        end = end - start
        print("time elapsed in igraphSecondOrderNull(): %1.2f" % end )
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
    
        time = {}
        for e in self.tedges:
            try:
                time[e[2]].append(e)
            except KeyError:
                time[e[2]] = [e]
        
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
        time = {}
        for e in self.tedges:
            try:
                time[e[2]].append(e)
            except KeyError:
                time[e[2]] = [e]

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

        # Generate movie frames
        if realtime == True:
            t_range = range(min(time.keys()), max(time.keys())+1)
        else:
            t_range = list(time.keys())

        if maxSteps>0:
            t_range = t_range[:maxSteps]

        i = 0
        for t in t_range:
            i += 1
            slice = igraph.Graph(n=len(g.vs()), directed=True)
            slice.vs["name"] = g.vs["name"]            
            try:
                edges = time[t]
            except KeyError:
                edges = []
            for e in edges:
                slice.add_edge(e[0], e[1])
            igraph.plot(slice, fileprefix + '_frame_' + str(t).zfill(5) + '.png', **visual_style)
            if i % 100 == 0:
                print('Wrote movie frame', i, ' of', len(t_range))
