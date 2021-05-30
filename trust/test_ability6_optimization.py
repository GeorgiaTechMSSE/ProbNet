####################################################################
### ProbNet: A probabilistic graph modeling toolkit that enables the design of cyber-physical-social systems 
###	     with the considerations of trust, resilience, and information dynamics',
###   author: Yan Wang,
###   author_email: yan-wang@gatech.edu
####################################################################
####################################################################
### generate an optimum network based on maxmization of 
### individual's ability from a randomly generated graph
####################################################################
import numpy as np
from numpy import random
import math
from probGraph import *
##import pdb
import networkx as nx

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt



######## Randomly generated graph ######
TotalNumNode = 10
ConnectProb = 0.08
##TotalNumNode = 40
##ConnectProb = 0.08
Seed = 21211

g=RandomProbGraph(TotalNumNode, ConnectProb, Seed)


##nodeset = set([n for n,d in g.nodes_iter(data=True)])  ##deprecated
nodeset = set(g.nodes)

TotalNumTrial = 1000



Abili = np.zeros((TotalNumTrial,TotalNumNode))
Utility = np.zeros(TotalNumTrial)

################################################################################
### Greedy algorithm to search for strategic networks based on ability ###

### before adding any other node
initNode = 0
subnodeset = set([initNode])

### Queue for breadth-first search
import queue
que = queue.Queue()
que.put(initNode)

### create a undirected copy of graph for search
ug = g.to_undirected()

### keep track of which nodes have been visited
visited = {}
for n in g.nodes():
    visited[n]=False

#pdb.set_trace()

NumTrial=1
Abili[:,initNode]=0

### adding additional nodes in graph
###for NumTrial in range(1,TotalNumTrial):
while que.qsize()>0 and NumTrial<TotalNumTrial:    
    if que.qsize()>0:
        s = que.get()
        trialnodeset = subnodeset.union(set([s]))
        sg=g.subgraph(trialnodeset)
        Abili[NumTrial,:] = np.maximum(Abili[NumTrial,:],sg.ability_perception(TotalNumNode)[:,0])
#@@#        Abili[NumTrial,:] = np.maximum(Abili[NumTrial,:],sg.ability_perception_second_order(TotalNumNode)[:,0])
        deltaTrial1 = Abili[NumTrial,initNode] - Abili[NumTrial-1,initNode]
        deltaTrial2 = 0
        for t in trialnodeset-set([initNode]):
            deltaTrial2 += Abili[NumTrial,t] - Abili[NumTrial-1,t]
        deltaTrial2 /= len(subnodeset)
        delta=deltaTrial1 - deltaTrial2
##        if deltaTrial1 > 0 or delta > 0:           ####accept the new node in subnodeset
        if deltaTrial1 > 0:
            visited[s]=True
            subnodeset=subnodeset.union(set([s]))
##            for n in ug.neighbors_iter(s):  @obsolete networkx1.1
            for n in ug.neighbors(s):
                if not visited[n]:
                    que.put(n)
        else:                   ####roll back to the previous subnodeset
            visited[s]=False
            Abili[NumTrial,:]=Abili[NumTrial-1,:]
###    print que.qsize(), deltaTrial1, deltaTrial2, Abili[NumTrial,:]
#    Utility[NumTrial]=Utility[NumTrial-1]+np.maximum(0,delta)
#    Utility[NumTrial]=Utility[NumTrial-1]+np.maximum(0,deltaTrial1)
    Utility[NumTrial]=Abili[NumTrial,initNode]
    NumTrial+=1

print('\noriginal graph # of nodes: ', g.number_of_nodes())
print('\noriginal graph # of edges: ', g.number_of_edges())
print('\ninitial/reference node ', initNode)
print('\noptimized network # of nodes: ', sg.number_of_nodes())
print('optimized network # of edges: ', sg.number_of_edges())
print('optimized network: ', sg.nodes())
print('final queue size: ', que.qsize())
print('# of trials: ', NumTrial, ' final ability: ', Abili[NumTrial-1,:])

### plot the original graph
pos=nx.spring_layout(g)
nx.draw(g,pos,node_color='white',with_labels=True)
plt.show()

### plot the optimized network
pos=nx.spring_layout(sg)
color_values={x: 'yellow' for x in sg.nodes()}
color_values[initNode]='red'
nx.draw(sg,pos,node_color=color_values.values(),with_labels=True)
plt.show()

### plot the Utility[] for each iteration
plt.plot(np.linspace(0,NumTrial-1,NumTrial), Utility[0:NumTrial], color='blue', alpha=1.00)
plt.ylabel('Utility(Ability)',fontsize=18)
plt.xlabel('Iteration',fontsize=18)
plt.show()
