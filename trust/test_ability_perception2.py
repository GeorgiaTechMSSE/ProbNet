####################################################################
### ProbNet: A probabilistic graph modeling toolkit that enables the design of cyber-physical-social systems 
###	     with the considerations of trust, resilience, and information dynamics',
###   author: Yan Wang,
###   author_email: yan-wang@gatech.edu
####################################################################
####################################################################
### Test the perception model of ability 
###     --- from the 11-node network
####################################################################
from probGraph import *
import numpy as np
from numpy import random
import math
##import pdb
##import sys
##sys.path.append("C:\\Users\\Wang\\VMShared\\py\\cps\\ProbNet")
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
import networkx as nx

"""
edgelist = [(0,1, .9,.5, 0.1, 0.1),
 (1,2, .5,.5,  0.3, 0.3),
 (2,0, .5,.5,  0.3, 0.3),
 (0,3, .5,.5,  0.3, 0.3),
 (3,0, .5,.5,  0.3, 0.3),
 (4,2, .5,.5,  0.3, 0.3),
 (4,5, .5,.5,  0.3, 0.3),
 (4,6, .5,.5,  0.3, 0.3),
 (4,7, .5,.5,  0.3, 0.3),
 (6,7, .5,.5,  0.3, 0.3),
 (3,8, .5,.5,  0.3, 0.3),
 (8,9, .5,.5,  0.3, 0.3),
 (9,10, .5,.5,  0.3, 0.3),
 (10,8, .5,.5,  0.3, 0.3),
 (7,10, .5,.5,  0.3, 0.3)]"""
edgelist = [(0,1, .9,.5, 0.1, 0.1),
 (1,2, .9,.1,  0.1, 0.1),
 (2,0, .9,.1,  0.1, 0.1),
 (0,3, .9,.1,  0.1, 0.1),
 (3,0, .9,.1,  0.1, 0.1),
 (4,2, .9,.1,  0.1, 0.1),
 (4,5, .9,.1,  0.1, 0.1),
 (4,6, .9,.1,  0.1, 0.1),
 (4,7, .9,.1,  0.1, 0.1),
 (6,7, .9,.1,  0.1, 0.1),
 (3,8, .9,.1,  0.1, 0.1),
 (8,9, .9,.1,  0.1, 0.1),
 (9,10, .9,.1,  0.1, 0.1),
 (10,8, .9,.1,  0.1, 0.1),
 (7,10, .9,.1,  0.1, 0.1)]
"""
nodelist = [(0,.9, 0.1),
            (1,.9, 0.1),
            (2,.9, 0.1),
            (3,.9, 0.1),
            (4,.9, 0.1),
            (5,.9, 0.1),
            (6,.9, 0.1),
            (7,.9, 0.1),
            (8,.9, 0.1),
            (9,.9, 0.1),
            (10,.9, 0.1)]"""
nodelist = [(0,.5, 0.3),
            (1,.5, 0.3),
            (2,.5, 0.3),
            (3,.5, 0.3),
            (4,.5, 0.3),
            (5,.5, 0.3),
            (6,.5, 0.3),
            (7,.5, 0.3),
            (8,.5, 0.3),
            (9,.5, 0.3),
            (10,.5, 0.3)]

g = ProbGraph(nodelist,edgelist)
TotalNumNode = g.number_of_nodes()

#nodeset = set([n for n,d in g.nodes_iter(data=True)])  #@obsolete networkx 1.1

ab=g.ability_perception(TotalNumNode)
ab2=g.ability_perception_second_order(TotalNumNode)

cb=g.capability_perception(TotalNumNode)
ib=g.influence_perception(TotalNumNode)

ri=g.reciprocity(TotalNumNode,'P_Prob')
mo=g.motive(TotalNumNode)
be=g.benevolence(TotalNumNode)

### plot the original graph
pos=nx.spring_layout(g)
color_values={x: 'yellow' for x in g.nodes()}
#color_values[initNode]='red'
nx.draw(g,pos,node_color=color_values.values(),with_labels=True)
plt.show()

fig=plt.figure()
plt.subplot(4,1,1)
plt.errorbar(np.arange(0,TotalNumNode),cb[:,0],np.sqrt(cb[:,1]),fmt='-o',linestyle='None')
plt.ylabel('capability')
plt.xlim([-0.5,10.5])
plt.ylim([0,1])
plt.subplot(4,1,2)
plt.errorbar(np.arange(0,TotalNumNode),ib[:,0],np.sqrt(ib[:,1]),fmt='-o',linestyle='None')
plt.ylabel('influence')
plt.xlim([-0.5,10.5])
plt.ylim([0,1])
plt.subplot(4,1,3)
plt.errorbar(np.arange(0,TotalNumNode),ab[:,0],np.sqrt(ab[:,1]),fmt='-o',linestyle='None')
plt.ylabel('ability')
plt.xlim([-0.5,10.5])
plt.ylim([0,1])
##
##fig=plt.figure()
##plt.subplot(2,1,1)
##plt.errorbar(np.arange(0,TotalNumNode),ab[:,0],np.sqrt(ab[:,1]),fmt='-o',linestyle='None')
##plt.ylabel('ability')
##plt.ylim([0,1])
plt.subplot(4,1,4)
plt.errorbar(np.arange(0,TotalNumNode),ab2[:,0],np.sqrt(ab2[:,1]),fmt='-o',linestyle='None')
plt.ylabel('second-order ability')
plt.xlim([-0.5,10.5])
plt.ylim([0,1])
plt.show()

fig=plt.figure()
plt.subplot(3,1,1)
plt.errorbar(np.arange(0,TotalNumNode),ri[0,0,:],np.sqrt(ri[1,0,:]),fmt='-o',linestyle='None')
plt.ylabel('reciprocity')
plt.xlim([-0.5,10.5])
plt.ylim([0,1])
plt.subplot(3,1,2)
plt.errorbar(np.arange(0,TotalNumNode),mo[:,0],np.sqrt(mo[:,1]),fmt='-o',linestyle='None')
plt.ylabel('motive')
plt.xlim([-0.5,10.5])
plt.ylim([0,1])
plt.subplot(3,1,3)
plt.errorbar(np.arange(0,TotalNumNode),be[0,0,:],np.sqrt(be[1,0,:]),fmt='-o',linestyle='None')
plt.ylabel('benevolence')
plt.xlim([-0.5,10.5])
plt.ylim([0,1])
plt.show()
