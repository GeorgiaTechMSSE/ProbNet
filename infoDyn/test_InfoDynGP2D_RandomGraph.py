####################################################################
### ProbNet: A probabilistic graph modeling toolkit that enables the design of cyber-physical-social systems 
###	     with the considerations of trust, resilience, and information dynamics',
###   author: Yan Wang,
###   author_email: yan-wang@gatech.edu
####################################################################
####################################################################
### Simulate the information dynamics with input-output relation 
### from a randomly generated graph based on sampling and different fusion rules
####################################################################
from probGraph import *
import numpy as np
from numpy import random
import math
#import pdb
import networkx as nx

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF, ConstantKernel, RationalQuadratic, Sum, Product
from kernelfunc import SquaredExponential, SquaredExponentialSine


from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
#Timesteps = 10   ##total number of steps in simulation
Timesteps = 80   ##total number of steps in simulation
#Timesteps = 200   ##total number of steps in simulation
#Timesteps = 1000   ##total number of steps in simulation
#trainsteps = 5  ## the size of training data
trainsteps = 50  ## the size of training data
#trainsteps = 70  ## the size of training data
#trainsteps = 600  ## the size of training data

#Pjoint = np.zeros((8,Timesteps))

######## Randomly generated graph ######
TotalNumNode = 8
#ConnectProb = 0.03
#ConnectProb = 0.08
#ConnectProb = 0.28      # 3-node-3-edge  # 15-node-66-edge
#ConnectProb = 0.1       # 8 nodes 10 edges   # 10 nodes 14 edges
ConnectProb = 0.85       # full connections with all edges
##TotalNumNode = 50
##ConnectProb = 0.08
Seed = 21211

g=RandomProbGraph(TotalNumNode, ConnectProb, Seed)

P=np.zeros((Timesteps, TotalNumNode))
pred=np.zeros((Timesteps, TotalNumNode))
stdev=np.zeros((Timesteps, TotalNumNode))
err=np.zeros((Timesteps, TotalNumNode))     # store forecast error
mean_sqr_err = np.zeros(TotalNumNode)

time=np.zeros(Timesteps)    ## record time step
### store training data
Data=np.zeros((trainsteps, TotalNumNode))   ## store prediction prob's as 2D array
DataTrainIn = np.zeros((trainsteps*TotalNumNode,2))  ##input of training
DataTrainOut = np.zeros(trainsteps*TotalNumNode)     ##output of training

### Generate the label for each node based on adjacency matrix
### so that the Hamming distance between nodes can be calculated
"""
nodelabel=[int("011",2),
           int("101",2),
           int("110",2)]
"""
Adj=nx.adjacency_matrix(g).todense()
nodelabel=np.zeros(TotalNumNode)
for i in range(0,TotalNumNode):
    label=''
    for j in range(0,TotalNumNode):
        if j == i:      ## set bit = 1 for the node itself
            label+=str(1)
        else:
            label+=str(Adj[i,j])
#        label += str(Adj[i, j])
    nodelabel[i]=int(label,2)

### when time = 0
for j in g.nodes():
    P[0,j]=g.get_node_data(j)['Prob']
    pred[0,j]=P[0,j]
    Data[0,j]=P[0,j]
    DataTrainIn[trainsteps*j,0]=0
    DataTrainIn[trainsteps*j,1]=nodelabel[j]
    DataTrainOut[trainsteps*j]=P[0,j]
#pdb.set_trace()
### prediction probabilty update by sampling with different fusion rules



### TRAINING STAGE
for t in range(1,trainsteps):
    time[t]=t
    g.updatePredictionWorstcaseSampling(300)
#    g.updatePredictionBestcaseSampling(300)
#    g.updatePredictionBayesianSampling(80)
##    Pjoint[:,t] = g.updatePredictionWorstcaseSampling(100, SelfInclusion=False)
#    Pjoint[:,t] = g.updatePredictionBayesianSampling(100)
    for j in g.nodes():
        P[t,j]=g.get_node_data(j)['Prob']
        pred[t,j]=P[t,j]
        Data[t,j]=P[t,j]
        DataTrainIn[trainsteps*j+t,0]=time[t]
        DataTrainIn[trainsteps*j+t,1]=nodelabel[j]     #full-connected nodes which have the same pairwise distance
        DataTrainOut[trainsteps*j+t]=P[t,j]
"""
DataTrainOut = []
for j in g.node():
    DataTrainOut = np.concatenate((DataTrainOut,Data[:,j]), axis=0)
"""

#### GP models ####
#sexp = expSine()     # kernel function

"""
### each 1D GP model corresponds one node  ###
gp = []
gp_kernel = []
for j in g.node():
    # Instantiate GaussianProcess class
#    gp_kernel.append( ExpSineSquared(1.0, 5.0) )
#    gp_kernel.append( ExpSineSquared(1.0, 5.0) + WhiteKernel(0.1) )
#    gp_kernel.append( ExpSineSquared(length_scale=1.0, periodicity=5, periodicity_bounds=(2, 100)) + WhiteKernel(noise_level=1.0) )
    gp_kernel.append( Sum(ExpSineSquared(length_scale=1.0, periodicity=5, periodicity_bounds=(2, 100)) , WhiteKernel(noise_level=2.0) ) )
    gp.append( GaussianProcessRegressor(kernel=gp_kernel[j]))
    # Fit the model to the data
    gp[j].fit(np.array([np.atleast_2d(u) for u in time])[0:trainsteps, 0], Data[0:trainsteps,j])
"""
### continuous dimension along time
kernel_c = SquaredExponentialSine(length_scale=1e-2, periodicity=20, length_scale_bounds=(1e-3, 1e-1), periodicity_bounds=(1, 100))
### discrete dimension for different nodes
kernel_d = SquaredExponential(length_scale=1e-2, length_scale_bounds=(1e-3, 1e-1), disttype='hamming',num_discrete_values=TotalNumNode)
### 2D output kernal
gp_kernel2D = Product(kernel_c,kernel_d) + WhiteKernel(1.0)
### GP model for 3 nodes
#gp = GaussianProcessRegressor(kernel=gp_kernel2D, alpha=5e0, normalize_y=True)  ## for 3 nodes
### GP model for 8 nodes
gp = GaussianProcessRegressor(kernel=gp_kernel2D, alpha=8e0, normalize_y=True)  ## for 8 nodes
### GP model for 15 nodes
#gp = GaussianProcessRegressor(kernel=gp_kernel2D, alpha=4.5e1, normalize_y=True)  ## for 15 nodes
# Fit the model to the data
gp.fit(DataTrainIn, DataTrainOut)

### PREDICITON by GP models
for t in range(trainsteps,Timesteps):
    time[t]=t
    g.updatePredictionWorstcaseSampling(300)
#    g.updatePredictionBestcaseSampling(300)
#    g.updatePredictionBayesianSampling(80)
    for j in g.nodes():
        ### actual probability
        P[t,j]=g.get_node_data(j)['Prob']
        ### predicted probability
#        pred[t,j],stdev[t,j]=gp[j].predict(np.array([np.atleast_2d(u) for u in time])[t:t+1, 0],return_std=True)
#        pred[t,j]+=pred[t-1,j]


for j in g.nodes():
    ### predicted probability
    #pred[trainsteps:Timesteps,j],stdev[trainsteps:Timesteps,j]=gp[j].predict(np.array([np.atleast_2d(u) for u in time])[trainsteps:Timesteps, 0],return_std=True)
    X=np.concatenate((np.array([np.atleast_2d(u) for u in time])[trainsteps:Timesteps, 0],np.ones((Timesteps-trainsteps,1))*nodelabel[j]), axis=1)
    pred[trainsteps:Timesteps,j],stdev[trainsteps:Timesteps,j]=gp.predict(X,return_std=True)
    err[trainsteps:Timesteps,j]=pred[trainsteps:Timesteps,j]-P[trainsteps:Timesteps,j]

# obtain the mean square error of forecasts
mean_sqr_err=np.sum(err**2, axis=0)
mean_sqr_err = mean_sqr_err / (Timesteps-trainsteps)

print('mean squared error (GP): '+str(mean_sqr_err)+' average for all nodes: '+str(mean_sqr_err.mean()))


#print('GP kernel:')
#print(gp_kernel.hyperparameters())

### plot the graph
#pos=nx.spring_layout(g)
pos=nx.kamada_kawai_layout(g)
plt.figure(figsize=(2,2))
nx.draw(g,pos,node_color='gold',with_labels=True)
plt.show()


### plot the dynamics of prediction probabilities
#fig, axs = plt.subplots(3,1)
fig, axs = plt.subplots(nrows=TotalNumNode,ncols=1,figsize=(6,5))
#fig, axs = plt.subplots(nrows=TotalNumNode,ncols=1,figsize=(6,7))
fig.suptitle("Info Dynamics: GP (Worst-Case fusion): # of nodes="+str(TotalNumNode))
#fig.suptitle("Info Dynamics: GP (Best-Case Fusion): # of nodes="+str(TotalNumNode))
#fig.suptitle("Info Dynamics: GP (Bayesian Fusion): # of nodes="+str(TotalNumNode))
j = 0
xticks = np.arange(0, Timesteps, 20)
for ax in axs.flat:
    ax.set_xlim([0, time[Timesteps-1]])
    ax.set_ylim([0.0, 1.05])
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(['','','',''])
#    ax.set_xticklabels(['', '', '', '', ''])
    ax.plot(time[:], P[:,j], 'b-', label="node "+str(j))
###    ax.plot(time[trainsteps:Timesteps], pred[trainsteps:Timesteps,j], 'r--', label="forecast")
    ax.plot(time[trainsteps:Timesteps], pred[trainsteps:Timesteps,j], 'r--')
###    ax.plot(time[trainsteps:Timesteps], pred[trainsteps:Timesteps,j]+2*stdev[j], 'r-.', label="$\pm 2\sigma$")
#    ax.plot(time[trainsteps:Timesteps], pred[trainsteps:Timesteps,j]+2*stdev[trainsteps:Timesteps,j], 'r-.')
#    ax.plot(time[trainsteps:Timesteps], pred[trainsteps:Timesteps,j]-2*stdev[trainsteps:Timesteps,j], 'r-.')
##    ax.fill_between(time[trainsteps:Timesteps], pred[trainsteps:Timesteps,j]-2*stdev[trainsteps:Timesteps,j], \
##                     pred[trainsteps:Timesteps,j]+2*stdev[trainsteps:Timesteps,j], color='darkorange', \
##                     alpha=0.2)
    ax.fill_between(time[trainsteps:Timesteps], pred[trainsteps:Timesteps,j]-stdev[trainsteps:Timesteps,j], \
                     pred[trainsteps:Timesteps,j]+stdev[trainsteps:Timesteps,j], color='gold')
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='upper left',ncol=3)
    j+=1
ax.set_xticklabels(['0','20','40','60'])
#ax.set_xticklabels(['0','20','40','60','80'])
##plt.xlabel('time')
plt.subplots_adjust(left=0.06, right=0.98, bottom=0.04, top=0.94, wspace=0, hspace=0)
#plt.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.98, wspace=0, hspace=0)
#plt.subplots_adjust(left=0.04, right=0.98, bottom=0.04, top=0.98, wspace=0, hspace=0)
plt.show()




