# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 10:27:00 2015

@author: Ingo Scholtes
"""

import igraph
import pyTempNet as tn

class TimeSlices:
    def __init__(self, tempnet, start=0, end=0, window=1, delta=1):

        self.time = {}
        for e in tempnet.tedges:
            try:
                self.time[e[2]].append(e)
            except KeyError:
                self.time[e[2]] = [e]

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
        if self.t <= self.end:
            g = self.AggregateNet(self.t, self.t+self.window)
            self.t += self.delta
            return g
        else:
            raise StopIteration()