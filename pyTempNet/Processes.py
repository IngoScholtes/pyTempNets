# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:35:22 2015
@author: Ingo Scholtes, Roman Cattaneo

(c) Copyright ETH Zürich, Chair of Systems Design, 2015
"""

import numpy as np
import igraph
from collections import defaultdict
from collections import Counter
import os

from subprocess import call

import pyTempNet as tn
from pyTempNet import Utilities
from pyTempNet.Log import *
    
def RWDiffusion(g, samples = 5, epsilon=0.01, max_iterations=100000):
    """Computes the average number of steps requires by a random walk process
    to fall below a total variation distance below epsilon (TVD computed between the momentary 
    visitation probabilities \pi^t and the stationary distribution \pi = \pi^{\infty}. This time can be 
    used to measure diffusion speed in a given (weighted and directed) network."""
    avg_speed = 0
    
    T = Utilities.RWTransitionMatrix(g)
    pi = Utilities.StationaryDistribution(T)
    
    n = len(g.vs())
    for s in range(samples):
        x = np.zeros(n)
        seed = np.random.randint(n)
        x[seed] = 1
        while Utilities.TVD(x,pi)>epsilon:
            avg_speed += 1
            # NOTE x * T = (T^T * x^T)^T
            # NOTE T is already transposed to get the left EV
            x = (T.dot(x.transpose())).transpose()
            if avg_speed > max_iterations:
              Log.add("x[0:10] = " + str(x[0:10]))
              Log.add("pi[0:10] = " + str(pi[0:10]))
              raise RuntimeError("Failed to converge within maximal number of iterations. Start of current x and pi are printed above")

    return avg_speed/samples
    

def exportDiffusionMovieFrames(g, file_prefix='diffusion', visual_style = None, steps=100, initial_index=-1):
    """Exports an animation showing the evolution of a diffusion
           process on the network"""

    T = Utilities.RWTransitionMatrix(g)

    if visual_style == None:
            visual_style = {}
            visual_style["vertex_color"] = "lightblue"
            visual_style["vertex_label"] = g.vs["name"]
            visual_style["edge_curved"] = .5
            visual_style["vertex_size"] = 30

    # lambda expression for the coloring of nodes according to some quantity p \in [0,1]
    # p = 1 ==> color red 
    # p = 0 ==> color white
    color_p = lambda p: "rgb(255,"+str(int((1-p)*255))+","+str(int((1-p)*255))+")"

    # Initial state of random walker
    if initial_index<0:
        initial_index = np.random.randint(0, len(g.vs()))

    x = np.zeros(len(g.vs()))
    x[initial_index] = 1

    # compute stationary state
    pi = Utilities.StationaryDistribution(T)

    scale = np.mean(np.abs(x-pi))

    # Plot network (useful as poster frame of video)
    igraph.plot(g, file_prefix + "_network.pdf", **visual_style)

    # Create frames
    for i in range(0,steps):
        visual_style["vertex_color"] = [color_p(p**0.1) for p in x]
        igraph.plot(g, file_prefix + "_frame_" + str(i).zfill(5) +".png", **visual_style)
        if i % 10 == 0:
            Log.add('Step' + str(i) + '\tTVD = ' + str(Utilities.TVD(x,pi)), Severity.INFO)
        # NOTE x * T = (T^T * x^T)^T
        # NOTE T is already transposed to get the left EV
        x = (T.dot(x.transpose())).transpose()


def exportDiffusionComparisonVideo(t, output_file, visual_style = None, steps = 100, initial_index=-1, fps=10, dynamic=False, NWframesPerRWStep=5):
    """Exports an mp4 file containing a side-by-side comparison of a diffusion process in a Markovian (left) and a non-Markovian temporal network"""
    prefix_1 = str(np.random.randint(0, 10000))
    prefix_2 = str(np.random.randint(0, 10000))
    prefix_3 = str(np.random.randint(0, 10000))

    Log.add('Calculating diffusion dynamics in non-Markovian temporal network')
    exportDiffusionMovieFramesFirstOrder(t, file_prefix='frames' + os.sep + prefix_1, visual_style=visual_style, steps=steps, initial_index=initial_index, model='SECOND', dynamic=dynamic, NWframesPerRWStep=NWframesPerRWStep)
    Log.add('finished.')

    Log.add('Calculating diffusion dynamics in Markovian temporal network ...')
    exportDiffusionMovieFramesFirstOrder(t, file_prefix='frames' + os.sep + prefix_2, visual_style=visual_style, steps=steps, initial_index=initial_index, model='NULL', dynamic=dynamic, NWframesPerRWStep=NWframesPerRWStep)
    Log.add('finished.')

    Log.add('Stitching video frames ...')
    for i in range(steps):
        x = call("convert frames" + os.sep + prefix_1 + "_frame_" + str(i).zfill(5)+ ".png frames" + os.sep + prefix_2+"_frame_" + str(i).zfill(5) + ".png +append " + "frames" + os.sep + prefix_3+"_stitched_" + str(i).zfill(5) + ".png", shell=True) 
    Log.add('finished.')
    if os.path.exists(output_file):
        os.remove(output_file)
    
    Log.add('Encoding video ...')
    x = call("ffmpeg -nostdin -framerate " + str(fps) + " -i frames" + os.sep + prefix_3 + "_stitched_%05d.png -c:v libx264 -pix_fmt yuv420p " + output_file, shell=True)    
    Log.add('finished.')


def exportRandomWalkVideo(t, output_file, visual_style = None, steps = 100, initial_index=-1, fps=10, model='SECOND', dynamic=False, NWframesPerRWStep=5):
    prefix = str(np.random.randint(0, 10000))

    if model == 'SECOND':
        Log.add('Calculating random walk in non-Markovian temporal network ...')
    else:
        Log.add('Calculating random walk in Markovian temporal network ...')
    exportRandomWalkMovieFramesFirstOrder(t, file_prefix='frames' + os.sep + prefix, visual_style=visual_style, steps=steps, initial_index=initial_index, model=model, dynamic=dynamic, NWframesPerRWStep=NWframesPerRWStep)
    Log.add('finished.')

    if os.path.exists(output_file):
        os.remove(output_file)

    Log.add('Encoding video ...')
    x = call("ffmpeg -nostdin -framerate " + str(fps) + " -i frames" + os.sep + prefix + "_frame_%05d.png -c:v libx264 -pix_fmt yuv420p " + output_file, shell=True)    
    Log.add('finished.')


def exportRandomWalkMovieFramesFirstOrder(t, file_prefix='random_walk', visual_style = None, steps=100, initial_index=-1, model='SECOND', dynamic=False, NWframesPerRWStep=5):
    """Exports an animation showing a random walk
           process on the first-order aggregate network, where random walk dynamics 
           either follows a first-order (mode='NULL') or second-order (model='SECOND') Markov 
           model"""
    assert model == 'SECOND' or model =='NULL'

    g1 = t.igraphFirstOrder()

    if model == 'SECOND':
        g2 = t.igraphSecondOrder().components(mode='STRONG').giant()
        temporal = tn.TemporalNetwork.ShuffleTwoPaths(t)
    elif model == 'NULL':
        g2 = t.igraphSecondOrderNull().components(mode='STRONG').giant()
        temporal = tn.TemporalNetwork.ShuffleEdges(t) 

    #T = Utilities.RWTransitionMatrix(g2)

    # visual style is for *first-order* aggregate network
    if visual_style == None:
            visual_style = {}
            visual_style["vertex_color"] = ["lightblue"]*g1.vcount()
            visual_style["vertex_label"] = g1.vs["name"]
            visual_style["edge_curved"] = .5
            visual_style["vertex_size"] = 30

    if type(visual_style["vertex_color"]) == str:
        visual_style["vertex_color"] = [visual_style["vertex_color"]]*g1.vcount()

    vertex_colors = list(visual_style["vertex_color"])

    # Initial state of random walker in SECOND_ORDER network
    if initial_index<0:
        initial_index = np.random.randint(0, len(g2.vs()))    

    rw_position = initial_index

    # This index allows to quickly map node names to indices in the first-order network
    map_name_to_id = {}
    for i in range(len(g1.vs())):
        map_name_to_id[g1.vs()['name'][i]] = i
    
    # Index to quickly map second-order node indices to first-order node indices
    map_2_to_1 = {}
    for j in range(len(g2.vs())):
        # j is index of node in *second-order* network
        # we first get the name of the *target* of the underlying edge
        node = g2.vs()["name"][j].split(t.separator)[1]

        # we map the target of second-order node j to the index of the *first-order* node
        map_2_to_1[j] = map_name_to_id[node]   


    # Create video frames
    for i in range(0,steps):            
        
        # highlight current position of random walker 
        visual_style["vertex_color"][map_2_to_1[rw_position]] = "green"

        # Visualize illustrative network dynamics
        if dynamic == True:
            #slice = igraph.Graph(n=len(g1.vs()))
            #slice.vs["name"] = g1.vs["name"]
            L = len(temporal.ordered_times)
            visual_style["edge_color"] = ["darkgrey"]*g1.ecount()
            visual_style["edge_width"] = [.5]*g1.ecount()
            for e in temporal.time[temporal.ordered_times[i%L]]:
                e_id = g1.get_eid(e[0], e[1])
                visual_style["edge_width"][e_id] = 5
                visual_style["edge_color"][e_id] = "black"
            igraph.plot(g1, file_prefix + "_frame_" + str(i).zfill(5) +".png", **visual_style)
        else:
            igraph.plot(g1, file_prefix + "_frame_" + str(i).zfill(5) +".png", **visual_style)

        # restore colors 
        visual_style["vertex_color"] = list(vertex_colors)

        if i % 50 == 0:
            Log.add('Frame ' + str(i))
        
        # every NWframesPerRWStep frames, perform one random walk step
        if i % NWframesPerRWStep == 0:
            successors = g2.successors(rw_position)
            probs = [g2.es()[g2.get_eid(rw_position, s)]["weight"] for s in successors]
            probs = probs/np.sum(probs)
            rw_position = np.random.choice(a=successors, p=probs)


def exportDiffusionVideo(t, output_file, visual_style = None, steps = 100, initial_index=-1, fps=10, model='SECOND'):
    prefix = str(np.random.randint(0, 10000))

    if model == 'SECOND':
        Log.add('Calculating diffusion dynamics in non-Markovian temporal network ...')
    else:
        Log.add('Calculating diffusion dynamics in Markovian temporal network ...')
    exportDiffusionMovieFramesFirstOrder(t, file_prefix='frames' + os.sep + prefix, visual_style=visual_style, steps=steps, initial_index=initial_index, model=model)
    Log.add('finished.')

    if os.path.exists(output_file):
        os.remove(output_file)

    Log.add('Encoding video ...')
    x = call("ffmpeg -nostdin -framerate " + str(fps) + " -i frames" + os.sep + prefix + "_frame_%05d.png -c:v libx264 -pix_fmt yuv420p " + output_file, shell=True)    
    Log.add('finished.')


def exportDiffusionMovieFramesFirstOrder(t, file_prefix='diffusion', visual_style = None, steps=100, initial_index=-1, model='SECOND', dynamic=False, NWframesPerRWStep=5):
    """Exports an animation showing the evolution of a diffusion
           process on the first-order aggregate network, where random walk dynamics 
           either follows a first-order (mode='NULL') or second-order (model='SECOND') Markov 
           model"""
    assert model == 'SECOND' or model =='NULL'

    g1 = t.igraphFirstOrder()

    if model == 'SECOND':
        g2 = t.igraphSecondOrder()
        temporal = tn.TemporalNetwork.ShuffleTwoPaths(t)
    elif model == 'NULL':
        g2 = t.igraphSecondOrderNull()
        temporal = tn.TemporalNetwork.ShuffleEdges(t) 

    T = Utilities.RWTransitionMatrix(g2)

    # visual style is for *first-order* aggregate network
    if visual_style == None:
            visual_style = {}
            visual_style["vertex_color"] = "lightblue"
            visual_style["vertex_label"] = g1.vs["name"]
            visual_style["edge_curved"] = .5
            visual_style["vertex_size"] = 30

    # Initial state of random walker
    if initial_index<0:
        initial_index = np.random.randint(0, len(g2.vs()))

    exp = 1.0/.75

    x = np.zeros(len(g2.vs()))
    x[initial_index] = 1

    # This index allows to quickly map node names to indices in the first-order network
    map_name_to_id = {}
    for i in range(len(g1.vs())):
        map_name_to_id[g1.vs()['name'][i]] = i
    
    # Index to quickly map second-order node indices to first-order node indices
    map_2_to_1 = {}
    for j in range(len(g2.vs())):
        # j is index of node in *second-order* network
        # we first get the name of the *target* of the underlying edge
        node = g2.vs()["name"][j].split(t.separator)[1]

        # we map the target of second-order node j to the index of the *first-order* node
        map_2_to_1[j] = map_name_to_id[node]

    # compute stationary state of random walk process
    pi = Utilities.StationaryDistribution(T)

    scale = np.mean(np.abs(x-pi))

    # lambda expression for the coloring of nodes according to some quantity p \in [0,1]
    # p = 1 ==> color red 
    # p = 0 ==> color white
    color_p = lambda p: "rgb(255,"+str(int((1-p)*255))+","+str(int((1-p)*255))+")"

    # Create video frames
    for i in range(0,steps):
        # based on visitation probabilities in *second-order* aggregate network, 
        # we need to compute visitation probabilities of nodes in the *first-order* 
        # aggregate network
        x_firstorder = np.zeros(len(g1.vs()))
        
        # j is the index of nodes in the *second-order* network, which we need to map 
        # to nodes in the *first-order* network
        for j in range(len(x)):
            if x[j] > 0:
                x_firstorder[map_2_to_1[j]] = x_firstorder[map_2_to_1[j]] + x[j]            
        
        # Perform some reasonable color scaling
        visual_style["vertex_color"] = [color_p(np.power((p-min(x))/(max(x)-min(x)),exp)) for p in x_firstorder]

        # Visualize illustrative network dynamics
        if dynamic == True:
            #slice = igraph.Graph(n=len(g1.vs()))
            #slice.vs["name"] = g1.vs["name"]
            L = len(temporal.ordered_times)
            visual_style["edge_color"] = ["darkgrey"]*g1.ecount()
            visual_style["edge_width"] = [.5]*g1.ecount()
            for e in temporal.time[temporal.ordered_times[i%L]]:
                e_id = g1.get_eid(e[0], e[1])
                visual_style["edge_width"][e_id] = 5
                visual_style["edge_color"][e_id] = "black"
                #slice.add_edge(e[0], e[1])
            igraph.plot(g1, file_prefix + "_frame_" + str(i).zfill(5) +".png", **visual_style)
        else:
            igraph.plot(g1, file_prefix + "_frame_" + str(i).zfill(5) +".png", **visual_style)

        # Plot first-order aggregate network after 30 RW steps (particularly useful as poster frame of video)
        if i == 30:
            visual_style["edge_color"] = "black"
            visual_style["edge_width"] = 1
            igraph.plot(g1, file_prefix + "_network.pdf", **visual_style)

        if i % 50 == 0:
            Log.add('Frame ' + str(i) + '\tTVD = ' + str(Utilities.TVD(x,pi)))
        
        # every NWframesPerRWStep frames, perform one random walk step
        if i % NWframesPerRWStep == 0:
            # NOTE x * T = (T^T * x^T)^T
            # NOTE T is already transposed to get the left EV
            x = (T.dot(x.transpose())).transpose()


def exportSIVideoStatic(g, output_file, visual_style = None, steps = 100, initial_index=0, fps=1):
    """Exports an mp4 file containing a side-by-side comparison of an SI process in a Markovian (left) and a non-Markovian temporal network"""
    
    r = np.random.randint(0, 10000)
    prefix = str(r)

    Log.add('Simulating SI process on network ...')
    exportSIMovieFramesStatic(g, file_prefix='frames' + os.sep + prefix, visual_style=visual_style, steps=steps, initial_index=initial_index)
    Log.add('finished.')    

    if os.path.exists(output_file):
        os.remove(output_file)
    
    Log.add('Encoding video ...')
    x = call("ffmpeg -nostdin -framerate " + str(fps) + " -i " + " frames" + os.sep + prefix + "_frame_%05d.png -c:v libx264 -r 30 -pix_fmt yuv420p " + output_file, shell=True) 
    Log.add('finished.')

    # Alternatively, we could have used the convert frontend of imagemagick, but this is known to generate a "delegate" error on Windows machines when the number of frames is too large
    # x = call("convert -delay " + str(delay) +" frames\\"+prefix_3+"_frame_* "+output_file, shell=True) 



def exportSIComparisonVideo(t, output_file, visual_style = None, steps = 700, initial_index=0, delay=0):
    """Exports an mp4 file containing a side-by-side comparison of an SI process in a Markovian (left) and a non-Markovian temporal network"""
    
    r = np.random.randint(0, 10000)
    prefix_1 = str(r)
    prefix_2 = str(r + 1)
    prefix_3 = str(r + 2)

    Log.add('Simulating SI dynamics in Markovian temporal network ...')
    exportSIMovieFrames(t, file_prefix='frames' + os.sep + prefix_2, visual_style=visual_style, steps=steps, initial_index=initial_index, model='NULL')
    Log.add('finished.')

    Log.add('Simulating SI dynamics in non-Markovian temporal network ...')
    exportSIMovieFrames(t, file_prefix='frames' + os.sep + prefix_1, visual_style=visual_style, steps=steps, initial_index=initial_index, model='SECOND')
    Log.add('finished.')

    Log.add('Stitching video frames ...')
    for i in range(steps):
        x = call("convert frames" + os.sep + prefix_1 + "_frame_" + str(i).zfill(5)+ ".png frames"+os.sep + prefix_2+"_frame_" + str(i).zfill(5) + ".png +append " + "frames" + os.sep + prefix_3+"_frame_" + str(i).zfill(5) + ".png", shell=True) 
    Log.add('finished.')

    if os.path.exists(output_file):
        os.remove(output_file)
    
    Log.add('Encoding video ...')
    x = call("ffmpeg -nostdin -framerate 30 -i " + " frames" + os.sep + prefix_3 + "_frame_%05d.png -c:v libx264 -r 30 -pix_fmt yuv420p " + output_file, shell=True) 
    Log.add('finished.')

    # Alternatively, we could have used the convert frontend of imagemagick, but this is known to generate a "delegate" error on Windows machines when the number of frames is too large
    # x = call("convert -delay " + str(delay) +" frames\\"+prefix_3+"_frame_* "+output_file, shell=True) 


def exportSIMovieFramesStatic(g, file_prefix='SI', visual_style = None, steps=100, initial_index=-1):
    """Exports an animation showing the evolution of an SI epidemic
           process on a static network"""    


    # default visual style
    if visual_style == None:
            visual_style = {}
            visual_style["vertex_color"] = "lightblue"
            # visual_style["vertex_label"] = g1.vs["name"]
            visual_style["layout"] = g.layout_auto()
            visual_style["edge_curved"] = .5
            visual_style["vertex_size"] = 30
    

    # Plot first-order aggregate network (particularly useful as poster frame of video)
    igraph.plot(g, file_prefix + "_network.pdf", **visual_style)

    # Initially infected node
    if initial_index<0:
        initial_index = np.random.randint(0, len(g.vs()))

    # zero entries: not infected, one entries: infected
    infected = np.zeros(len(g.vs()))
    infected_list = [] 
    infected[initial_index] = 1
    infected_list.append(initial_index)

    # lambda expression for the coloring of nodes according to infection status
    # x = 1 ==> color red 
    # x = 0 ==> color white
    color_infected = lambda x: "rgb(255,"+str(int((1-x)*255))+","+str(int((1-x)*255))+")"

    # Create video frames
    for i in range(0,steps):

        visual_style["vertex_color"] = [color_infected(x) for x in infected]
        igraph.plot(g, file_prefix + '_frame_' + str(i).zfill(5) + '.png', **visual_style)

        for v in infected_list:
            for w in g.neighbors(v, mode='out'):
                infected[w] = 1
        infected_list = []
        for j in range(len(infected)):
            if infected[j] == 1:
                infected_list.append(j)        
        
        c = Counter(infected)

        if i % 10 == 0:
            Log.add('Step ' +str(i) + ' infected = ' + str(c[1]))



def exportSIMovieFrames(t, file_prefix='SI', visual_style = None, steps=100, initial_index=-1, model='SECOND'):
    """Exports an animation showing the evolution of an SI epidemic
           process on the first-order aggregate network, where the underlying link dynamics 
           either follows a first-order (mode='NULL') or second-order (model='SECOND') Markov 
           model"""
    assert model == 'SECOND' or model =='NULL'

    g1 = t.igraphFirstOrder()

    Log.add('Shuffling edges in temporal network ...')

    if model == 'SECOND':
        t_shuffled = t.ShuffleTwoPaths(l=steps)
    elif model == 'NULL':
        t_shuffled = t.ShuffleEdges(l=steps)

    Log.add('Generated shuffled temporal network with ' + str(len(t_shuffled.tedges)) + ' time-stamped links')

    time = defaultdict( lambda: list() )
    for e in t_shuffled.tedges:
        time[e[2]].append(e)

    map_name_to_id = {}
    for i in range(len(g1.vs())):
        map_name_to_id[g1.vs()['name'][i]] = i

    # default visual style for *first-order* aggregate network
    if visual_style == None:
            visual_style = {}
            visual_style["vertex_color"] = "lightblue"
            # visual_style["vertex_label"] = g1.vs["name"]
            visual_style["layout"] = g1.layout_auto()
            visual_style["edge_curved"] = .5
            visual_style["vertex_size"] = 30
    

    # Plot first-order aggregate network (particularly useful as poster frame of video)
    igraph.plot(g1, file_prefix + "_network.pdf", **visual_style)

    # Initially infected node
    if initial_index<0:
        initial_index = np.random.randint(0, len(g1.vs()))

    # zero entries: not infected, one entries: infected
    infected = np.zeros(len(g1.vs()))
    infected[initial_index] = 1

    # lambda expression for the coloring of nodes according to infection status
    # x = 1 ==> color red 
    # x = 0 ==> color white
    color_infected = lambda x: "rgb(255,"+str(int((1-x)*255))+","+str(int((1-x)*255))+")"

    t_range = range(min(time.keys()), max(time.keys())+1)   

    # Create video frames
    i = 0
    for t in t_range:        
        i += 1
        slice = igraph.Graph(n=len(g1.vs()), directed=False)
        slice.vs["name"] = g1.vs["name"]
        # this should work as time is a defaultdict
        for e in time[t]:
            slice.add_edge(e[0], e[1])
            if infected[map_name_to_id[e[0]]] == 1:
                infected[map_name_to_id[e[1]]] = 1
            if infected[map_name_to_id[e[1]]] == 1:
                infected[map_name_to_id[e[0]]] = 1
        visual_style["vertex_color"] = [color_infected(x) for x in infected]
        igraph.plot(slice, file_prefix + '_frame_' + str(t).zfill(5) + '.png', **visual_style)
        
        c = Counter(infected)

        if i % 100 == 0:
            Log.add('Step ' +str(i) + ' infected = ' + str(c[1]))
