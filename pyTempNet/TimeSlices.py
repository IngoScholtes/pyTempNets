# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 10:27:00 2015

@author: Ingo Scholtes
"""

import igraph
import pyTempNet as tn
import numpy as np

class TimeSlices:
    def __init__(self, tempnet, start=0, end=0, window=1, delta=1):
        """ Generates an iterator that generates a sequence of time-slice graphs. 
        Parameters start and end determine 
        based on 
        the given start time, window size and step size delta
        """
        self.time = defaultdict( lambda: list() )
        for e in tempnet.tedges:
            self.time[e[2]].append(e)

        self.t = max(start, min(self.time.keys()))
        if end == 0:
            end = max(self.time.keys())
        self.end = min(end, max(self.time.keys()))
        self.delta = delta
        self.window = window
        self.tempnet = tempnet

    def __iter__(self):
        return self


    def AggregateNet(self, t_from, t_to):
        """Generates a (first-order) weighted time-aggregated network
        capturing all time-stamped links (v,w,t) where 
        t \in [t_from, t_to)"""
        edges = {}

        g = igraph.Graph(directed=True)
        for v in self.tempnet.nodes:
            g.add_vertex(str(v))
        
        # TODO: first, add edges all in one go at the end of the loop
        # TODO: then, use a suitable data structure like defaultdict() to 
        #       avoid try-except block, see igraphFirstOrder
        # TODO: Then, improve efficiency by only iterating over those time 
        #       stamps that are within the actual time window [t_from, t_to)
        for t in self.time.keys():
            if t >= t_from and t<t_to:
                for edge in self.time[t]:                    
                    name = edge[0]+edge[1]
                    try:
                        x = edges[name]
                        g.es()[edges[name]]["weight"] += 1.
                    except KeyError:
                        edges[name] = len(g.es())
                        g.add_edge(edge[0], edge[1], weight=1.)
        return g

    def __next__(self):
        """ Iterator that generates a sequence of time-slice graphs based on 
        the given start time, window size and step size delta
        """
        if self.t <= self.end:
            g = self.AggregateNet(self.t, self.t+self.window)
            self.t += self.delta
            return g
        else:
            raise StopIteration()

    def ExportVideo(slices, output_file, visual_style={}, delay=10):
        """ Exports a video showing the evolution of time-slices"""

        prefix = str(np.random.randint(0,10000))
        
        if visual_style == None:
            print('No visual style specified, setting to defaults')
            visual_style = {}
            visual_style["vertex_color"] = "lightblue"
            visual_style["edge_curved"] = .5
            visual_style["vertex_size"] = 30
        i=0

        from subprocess import call

        for slice in slices:
            fname = 'frames\\'+ prefix + '_frame_' + str(i).zfill(5) + '.png'
            igraph.plot(slice, fname, **visual_style)
            x = call("convert "+ str(fname) + " -background LightBlue label:"+str(i)+" -gravity Center -append "+ str(fname), shell=True)
            i+=1

        x = call("convert -delay " + str(delay) +" frames\\"+prefix+"_frame_*.png "+output_file, shell=True)