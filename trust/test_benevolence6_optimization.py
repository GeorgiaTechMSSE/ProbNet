####################################################################
### ProbNet: A probabilistic graph modeling toolkit that enables the design of cyber-physical-social systems 
###	     with the considerations of trust, resilience, and information dynamics',
###   author: Yan Wang,
###   author_email: yan-wang@gatech.edu
####################################################################################################################################################
### Greedy algorithm to search for strategic networks based on benevolence   ###
### as utility function                                                      ### 
###       --- from a random network                                          ### 
################################################################################
from probGraph import *
import numpy as np
from numpy import random
import math
##import pdb

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
######## Randomly generated graph ######
#TotalNumNode = 100
#ConnectProb = 0.08
TotalNumNode = 50
ConnectProb = 0.08
Seed = 21211

g=RandomProbGraph(TotalNumNode, ConnectProb, Seed)

#TotalNumNode = g.number_of_nodes()
nodeset = set([n for n,d in g.nodes_iter(data=True)])

TotalNumTrial = 1000
Utility = np.zeros(TotalNumTrial)

Benev = np.zeros((TotalNumTrial,TotalNumNode,TotalNumNode))
Benev_ave = np.zeros((TotalNumTrial,TotalNumNode))


################################################################################
### Greedy algorithm to search for strategic networks based on benevolence ###

###the altruism weight that considers others' reciprocation
### w=1: selfish, w=0: altruistic
w = 0

### before adding any other node
initNode = 0
subnodeset = set([initNode])
Benev[0,initNode,initNode] = 0.5

### Queue for breadth-first search
import Queue
que = Queue.Queue()
que.put(initNode)

### create a undirected copy of graph for search
ug = g.to_undirected()

### keep track of which nodes have been visited
visited = {}
for n in g.nodes():
    visited[n]=False

NumTrial=1
### adding additional nodes in graph
while que.qsize()>0 and NumTrial<TotalNumTrial:    
#$#    if len(subnodeset)<len(nodeset):
    if que.qsize()>0:
#        pdb.set_trace()
#$#        s=random.choice(list(nodeset-subnodeset), 1)
        s = que.get()
        visited[s]=True
        trialnodeset = subnodeset.union(set([s]))
        sg=g.subgraph(trialnodeset)
        Benev[NumTrial,:,:] = sg.benevolence(TotalNumNode)[0,:,:]
        Benev_ave[NumTrial,:] = (np.sum(Benev[NumTrial,:,:],axis=1)-Benev[NumTrial,:,:].diagonal())/(len(subnodeset))
        deltaTrial1 = Benev_ave[NumTrial,initNode] - Benev_ave[NumTrial-1,initNode]
        deltaTrial2 = 0
        for t in trialnodeset-set([initNode]):
            deltaTrial2 += Benev_ave[NumTrial,t] - Benev_ave[NumTrial-1,t]
        deltaTrial2 /= len(subnodeset) 
#@@#        if deltaTrial1*w + deltaTrial2*(1-w) > 0:
        if deltaTrial1*w + deltaTrial2*(1-w) >= 0:
            subnodeset=subnodeset.union(set([s]))
            for n in ug.neighbors_iter(s):
                if not visited[n]:
                    que.put(n)
        else:                 ####roll back to the previous subnodeset
            visited[s]=False
            Benev[NumTrial,:,:]=Benev[NumTrial-1,:,:]   
            Benev_ave[NumTrial,:]=Benev_ave[NumTrial-1,:]
#    print que.qsize(), Benev_ave[NumTrial,:]
    Utility[NumTrial]=Benev_ave[NumTrial,initNode]
    NumTrial+=1
##    if que.qsize()==0:
##        break

print 'original graph: # of nodes'
print g.number_of_nodes()
print 'original graph: # of edges'
print g.number_of_edges()
print 'initial/reference node'
print initNode
print 'self-interest weight (w=1: selfish, w=0: altruistic):'
print w
print 'optimized network: # of nodes:'
print sg.number_of_nodes()
print 'optimized network: # of edges'
print sg.number_of_edges()
print 'optimized network:'
print sg.nodes()
print 'final queue size:'
print que.qsize()
print '# of trials  and   final average benevolence:'
print NumTrial, Benev_ave[NumTrial-1,:]

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
plt.ylabel('Utility (Ave. Benevolence)',fontsize=18)
plt.xlabel('Iteration',fontsize=18)
plt.show()


'''
########################################
### Print the abilities of all nodes ###
alpha = 0.333   # weight[0]
beta = 0.333    # weight[1]
gamma = 0.333   # weight[2]
fig = plt.figure()
index = np.arange(TotalNumNode)
bar_width = 0.55
opacity = 1.0
def cc(arg):
    return colorConverter.to_rgba(arg)
facecolors=[cc('red'), cc('green'), cc('blue'),cc('yellow'),
            cc('pink'), cc('purple'), cc('navy'),cc('grey'),
            cc('lime'), cc('gold'), cc('cyan')]
rects1 = plt.bar(index, g.ability(TotalNumNode,[alpha,beta,gamma]), bar_width,
                 alpha=opacity,
                 color=facecolors,
                 label='ability')
plt.title('Ability of nodes ($p_{i}='+str(g.get_node_data(0)['Prob'])+'$, $p_{ij}='+
          str(g.get_edge_data(0,1)['P_Prob'])+'$, $q_{ij}='+str(g.get_edge_data(0,1)['Q_Prob'])+'$)' )
#plt.text(9.7, 0.4, '$p_{11}=0.1$')
plt.text(7, 2.8, r'$\alpha='+str(alpha)+r'$, $\beta='+str(beta)+r'$, $\gamma='+str(gamma)+'$')
plt.xlabel('Node')
plt.ylabel('Ability')
plt.xlim(-bar_width/2.,11)
plt.ylim(0,1.2)
plt.xticks(index + bar_width/2., ('1','2','3','4','5','6','7','8','9','10','11'))
plt.yticks(np.arange(0, 3.2, 0.2))
plt.show()

### Print the abilities of all nodes ###
########################################
'''
            


"""
#####################################################
### Print the average reciprocations of all nodes ###

fig = plt.figure()
ax = fig.gca(projection='3d')

def cc(arg):
    return colorConverter.to_rgba(arg)

xs = np.arange(0,TotalNumNode)
verts = []
zs = np.arange(TotalNumTrial-1,-1,-1)
facecolors=[cc('red'), cc('green'), cc('blue'),cc('yellow'),
            cc('pink'), cc('purple'), cc('navy'),cc('grey'),
            cc('lime'), cc('gold'), cc('cyan')]

for z in zs:
    ax.bar(xs,Recip_ave[z,:],z, zdir='y',color=facecolors[z%11],alpha=0.7)

plt.show()

### Print the average reciprocations of all nodes ###
#####################################################

"""
