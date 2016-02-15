# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 17:01:45 2015
@author: Ingo Scholtes

(c) Copyright ETH Zürich, Chair of Systems Design, 2015
"""

import os
import igraph
import numpy as np
from subprocess import call

import pyTempNet as tn

from pyTempNet.Log import *

def exportTikzUnfolded(t, filename):
    """Generates a tikz file that can be compiled to obtain a time-unfolded 
        representation of the temporal network.
            
    @param filename: the name of the tex file to be generated."""    
        
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
            
    for n in t.nodes:
        if last == '':
            output.append("\\node[lbl]                     (" + n + "-0)   {$" + n + "$};\n")
        else:
            output.append("\\node[lbl,right=0.5cm of "+last+"-0] (" + n + "-0)   {$" + n + "$};\n")
        last = n
            
    output.append("\\setcounter{a}{0}\n")
    output.append("\\foreach \\number in {"+ str(min(t.ordered_times))+ ",...," + str(max(t.ordered_times)+1) + "}{\n")
    output.append("\\setcounter{a}{\\number}\n")
    output.append("\\addtocounter{a}{-1}\n")
    output.append("\\pgfmathparse{\\thea}\n")
        
    for n in t.nodes:
        output.append("\\node[v,below=0.3cm of " + n + "-\\pgfmathresult]     (" + n + "-\\number) {};\n")
    output.append("\\node[lbl,left=0.5cm of " + t.nodes[0] + "-\\number]    (col-\\pgfmathresult) {$t=$\\number};\n")
    output.append("}\n")
    output.append("\\path[->,thick]\n")
    i = 1
        
    for ts in t.ordered_times:
        for edge in t.time[ts]:
            output.append("(" + edge[0] + "-" + str(ts) + ") edge (" + edge[1] + "-" + str(ts + 1) + ")\n")
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
                    


def exportMovie(t, output_file, visual_style = None, realtime = True, directed = True, maxSteps=-1, fps=10):
    """Exports a video showing the evolution of the temporal network.
        
    @param output_file: the filename of the mp4 video to be generated
    @param visual_style: the igraph visual style to be used for the individual frames of the video.
        If the parameter value is None, a standard visual style will be used.
    @param realtime: Whether to generate a frame for every time step between the minimum and maximum timestamps, or only for those 
        where at least one node is active. For realtime=true, phases with no activity are retained in the video and there is a direct relation
        between the real time and the frame number. 
    @param maxSteps: The maximum number of time steps to export. For the default value -1 all steps in the evolution of the temporal network
        will be exported.
    @param delay: The delay in ms after each frame in the video. For the default value of 10, the framerate of the generated video will be 100 fps. 
    """
    prefix = str(np.random.randint(0,10000))
        
    # TODO: Make sure that frames directory exists
    exportMovieFrames(t, 'frames' + os.sep + prefix, visual_style = visual_style, realtime = realtime, directed = directed, maxSteps=maxSteps)
    os.remove(output_file)        

    Log.add('Encoding video ...')
    x = call("ffmpeg -nostdin -framerate " + str(fps) + " -i frames" + os.sep + prefix + "_frame_%05d.png -c:v libx264 -r 30 -pix_fmt yuv420p " + output_file, shell=True)
    Log.add('finished.')




def exportMovieFrames(t, fileprefix, visual_style = None, realtime = True, directed = True, maxSteps=-1):
    """Exports a sequence of numbered images showing the evolution of the temporal network. The resulting frames can be encoded into 
    custm video formats, for instance using ffmpeg. 
        
    @param output_file: the prefix of the file names to be used for the individual images
    @param visual_style: the igraph visual style to be used for the individual frames of the video.
        If the parameter value is None, a standard visual style will be used.
    @param realtime: Whether to generate a frame for every time step between the minimum and maximum timestamps, or only for those 
        where at least one node is active. For realtime=true, phases with no activity are retained in the video and there is a direct relation
        between the real time and the frame number. 
    @param maxSteps: The maximum number of time steps to export. For the default value -1 all steps in the evolution of the temporal network
        will be exported.
    """

    g = t.igraphFirstOrder()        

    if visual_style == None:
        Log.add('No visual style specified, setting to defaults', Severity.WARNING)
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
        t_range = range(min(t.time.keys()), max(t.time.keys())+1)
    else:
        t_range = t.ordered_times

    if maxSteps>0:
        t_range = t_range[:maxSteps]

    for ts in t_range:
        i += 1
        slice = igraph.Graph(n=len(g.vs()), directed=directed)
        slice.vs["name"] = g.vs["name"]

        for e in t.time[ts]:
            slice.add_edge(e[0], e[1])
        igraph.plot(slice, fileprefix + '_frame_' + str(ts).zfill(5) + '.png', **visual_style)
        if i % 100 == 0:
            Log.add('Wrote movie frame ' + str(i) + ' of ' + str(len(t_range)))
