####################################################################
### ProbNet: A probabilistic graph modeling toolkit that enables the design of cyber-physical-social systems 
###	     with the considerations of trust, resilience, and information dynamics',
###   author: Yan Wang,
###   author_email: yan-wang@gatech.edu
####################################################################
####################################################################
### Simulate the information dynamics with input-output relation 
### from a randomly generated graph based on sampling and different fusion rules
### topology of network changes dynamically
####################################################################
from probGraph import *
import numpy as np
from numpy import random
import math
#import pdb
import networkx as nx


from statsmodels.tsa.api import AR, VAR
from gaussian.cVAR import cVAR
##from gaussian.lvVAR import lvVAR

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
Timesteps = 90   ##total number of steps in simulation
#Timesteps = 60   ##total number of steps in simulation
#Timesteps = 50   ##total number of steps in simulation
#trainsteps = 150  ## the size of training data
trainsteps = 20  ## the size of training data
#trainsteps = 30  ## the size of training data
predictsteps = 10  ## the size of prediction steps
Epoch = 30      ## the window length for dynamic topological changes


######## Randomly generated graph ######
#TotalNumNode = 15
TotalNumNode = 8
#ConnectProb = 0.03
#ConnectProb = 0.28      # 3-node-3-edge  # 15-node-66-edge
#ConnectProb = 0.2       # 3-node-2-edge   #8-node-15-edge
ConnectProb = 0.1       # 8 nodes 10 edges   # 15 nodes 23 edges
#ConnectProb = 0.85       # full connections with all edges
#ConnectProb = 0.08
##TotalNumNode = 50
#ConnectProb = 0.0
Seed = 21211

#lag_max = 1
#lag_max = 2             ## the lag in AR model. E.g.  x[t]=a1*x[t-1]+a2*x[t-2]+sigma where lag = 2
lag_max = 3             ## the lag in AR model. E.g.  x[t]=a1*x[t-1]+a2*x[t-2]+sigma where lag = 2
#lag_max = 4             ## the lag in AR model. E.g.  x[t]=a1*x[t-1]+a2*x[t-2]+sigma where lag = 2


P=np.zeros((Timesteps, TotalNumNode))
P_var=np.zeros((Timesteps, TotalNumNode))
pred=np.zeros((Timesteps, TotalNumNode))
pred2=np.zeros((Timesteps, TotalNumNode))
pred3=np.zeros((Timesteps, TotalNumNode))
err=np.zeros((Timesteps, TotalNumNode))     # store forecast error
err2=np.zeros((Timesteps, TotalNumNode))     # store forecast error
err3=np.zeros((Timesteps, TotalNumNode))     # store forecast error
mean_sqr_err = np.zeros(TotalNumNode)
mean_sqr_err2 = np.zeros(TotalNumNode)
mean_sqr_err3 = np.zeros(TotalNumNode)

Data=np.zeros((trainsteps, TotalNumNode))

num_forecast = 0    ## keep track of # of forecast

g=RandomProbGraph(TotalNumNode, ConnectProb, Seed)
TotalNumEdge = g.number_of_edges()

"""
for j in g.nodes():
    P[0,j]=g.get_node_data(j)['Prob']
    pred[0,j]=P[0,j]
    pred2[0,j]=P[0,j]
    pred3[0,j]=P[0,j]
    Data[0,j]=P[0,j]
"""
"""
#pdb.set_trace()
A=g.adj
### worst-cast approach for prediction probability upate
time=np.zeros(Timesteps)
for t in range(1,Timesteps):
    time[t]=t
    for j in g.node():
        P[t,j]=P[t-1,j]
        for i in g.predecessors(j):
            if random.random() < 1.5:
                P[t,j]*=A[i][j]['P_Prob']*P[t-1,i]
            else:
                P[t,j]*=A[i][j]['Q_Prob']*P[t-1,i]
"""
#pdb.set_trace()
### prediction probabilty update by sampling with different fusion rules


time=np.zeros(Timesteps)    ## record time step

for currstep in range(0,Timesteps,Epoch):
    if currstep != 0:
        ### generate a new random graph
        ConnectProb = np.random.random()
        g = RandomProbGraph(TotalNumNode, ConnectProb, Seed)
        TotalNumEdge = g.number_of_edges()
        ### INITIATE Prediction probabilities of newly generated random graph
        for j in g.nodes():
            g.set_prediction_prob(j, P[currstep-1,j], P_var[currstep-1,j])

    ####################################################################
    ### plot the graph
    print('TotalNumNode = '+str(TotalNumNode))
    print('TotalNumEdge = '+str(TotalNumEdge))
    pos = nx.kamada_kawai_layout(g)
    plt.figure(figsize=(2, 2))
    nx.draw(g, pos, node_color='gold', with_labels=True)
    plt.show()

    ### TRAINING STAGE
    for t in range(currstep,currstep+trainsteps):
        time[t]=t
        g.updatePredictionWorstcaseSampling(300)
    #    g.updatePredictionBestcaseSampling(300)
    #    g.updatePredictionBayesianSampling(80)
    ##    Pjoint[:,t] = g.updatePredictionWorstcaseSampling(100, SelfInclusion=False)
    #    Pjoint[:,t] = g.updatePredictionBayesianSampling(100)
        for j in g.nodes():
            P[t,j]=g.get_node_data(j)['Prob']
            P_var[t,j]=g.get_node_data(j)['Prob_Var']
            pred[t,j]=P[t,j]
            pred2[t,j]=P[t,j]
            pred3[t,j]=P[t,j]
            Data[t-currstep,j]=P[t,j]
    #Rel = g.get_reliance_prob('P_Prob')
    #V0=(Rel[0]+Rel[0].T)/2+np.identity(TotalNumNode)
    ##print(V0)
    #X = np.random.multivariate_normal(P[0,:], V0/100, traindatasize)


    ### the adjacency matrix of graph
    adj = np.array(nx.adjacency_matrix(g).todense())



    #### VAR model ####
    var_model = cVAR(Data, adj=adj, bnds=[0,1])   # constrained by topology
    var_model2 = VAR(Data)                      # regular VAR
    var_model3 = cVAR(Data, bnds=[0,1])         # constrained by value range only
    results=var_model.fit(lag_max)
    results2=var_model2.fit(lag_max)
    results3=var_model3.fit(lag_max)
    lag_order = results.k_ar

    ##results.forecast(P[trainsteps:Timesteps,:], 50)
    #results.plot_forecast(Timesteps-trainsteps)
    #plt.show()

    ### PREDICITON by VAR model
    for t in range(currstep+trainsteps,currstep+trainsteps+predictsteps):
        time[t]=t
        g.updatePredictionWorstcaseSampling(300)
    #    g.updatePredictionBestcaseSampling(300)
    #    g.updatePredictionBayesianSampling(80)
        for j in g.nodes():
            P[t,j]=g.get_node_data(j)['Prob']
            P_var[t, j] = g.get_node_data(j)['Prob_Var']
        ### use coefficients as vectors / matrices
        pred[t,:]=results.params[0,:]
        pred2[t,:]=results2.params[0,:]
        pred3[t,:]=results3.params[0,:]
        for q in range(0,lag_order):
            pred[t,:]+=np.dot(pred[t-1-q,:], results.params[q*TotalNumNode+1:(q+1)*TotalNumNode+1,:])
            pred2[t,:]+=np.dot(pred2[t-1-q,:], results2.params[q*TotalNumNode+1:(q+1)*TotalNumNode+1,:])
            pred3[t,:]+=np.dot(pred3[t-1-q,:], results3.params[q*TotalNumNode+1:(q+1)*TotalNumNode+1,:])
        ### make sure it is within the probability range [0,1]
        pred[t,:]=np.maximum(np.zeros(TotalNumNode), pred[t,:])
        pred[t,:]=np.minimum(np.ones(TotalNumNode), pred[t,:])
        pred2[t,:]=np.maximum(np.zeros(TotalNumNode), pred2[t,:])
        pred2[t,:]=np.minimum(np.ones(TotalNumNode), pred2[t,:])
        pred3[t,:]=np.maximum(np.zeros(TotalNumNode), pred3[t,:])
        pred3[t,:]=np.minimum(np.ones(TotalNumNode), pred3[t,:])
        ### the forecast error in comparison with simulated real value
        err[t,:]=pred[t,:]-P[t,:]
        err2[t,:]=pred2[t,:]-P[t,:]
        err3[t,:]=pred3[t,:]-P[t,:]
        mean_sqr_err += err[t,:]**2
        mean_sqr_err2 += err2[t,:]**2
        mean_sqr_err3 += err3[t,:]**2
        num_forecast += 1

    ## obtain the standard deviation for predictions
    stdev = np.zeros(TotalNumNode)
    stdev2 = np.zeros(TotalNumNode)
    stdev3 = np.zeros(TotalNumNode)
    for j in g.nodes():
    #    stdev[j] = results.sigma_u[j,j]**0.5
    #    stdev2[j] = results2.sigma_u[j,j]**0.5
    #    stdev3[j] = results3.sigma_u[j,j]**0.5
        stdev[j] = results.sigma_u_mle[j,j]**0.5
        stdev2[j] = results2.sigma_u_mle[j,j]**0.5
        stdev3[j] = results3.sigma_u_mle[j,j]**0.5

# obtain the mean square error of forecasts
mean_sqr_err = mean_sqr_err / num_forecast
mean_sqr_err2 = mean_sqr_err2 / num_forecast
mean_sqr_err3 = mean_sqr_err3 / num_forecast

##print('Coefficents of topology constrained VAR model:')
##print(results.params)
##print('Coefficents of VAR model:')
##print(results2.params)
##print('Coefficents of value-constrained VAR model:')
##print(results3.params)
#print('covariance matrix of topology constrained VAR model:')
#print(results.sigma_u)
#print('covariance matrix of VAR model:')
#print(results2.sigma_u)
#print('covariance matrix of value-constrained VAR model:')
#print(results3.sigma_u)

print('mean squared error (topology constrained VAR): '+str(mean_sqr_err)+' average for all nodes: '+str(mean_sqr_err.mean()))
print('mean squared error (regular VAR): '+str(mean_sqr_err2)+' average for all nodes: '+str(mean_sqr_err2.mean()))
print('mean squared error (value constrained VAR): '+str(mean_sqr_err3)+' average for all nodes: '+str(mean_sqr_err3.mean()))







### plot the dynamics of prediction probabilities
### Constrained VAR model
#fig, axs = plt.subplots(nrows=TotalNumNode,ncols=1,figsize=(3,4))
fig, axs = plt.subplots(nrows=TotalNumNode,ncols=1,figsize=(6,8))
fig.suptitle("Topology constrained VAR (Worst-Case fusion): # of nodes="+str(TotalNumNode))
#fig.suptitle("Topology constrained VAR (Best-Case Fusion): # of nodes="+str(TotalNumNode))
#fig.suptitle("Topology constrained VAR (Bayesian Fusion): # of nodes="+str(TotalNumNode))
j = 0
xticks = np.arange(0, Timesteps, 20)
for ax in axs.flat:
    ax.set_xlim([0, time[Timesteps-1]])
    ax.set_ylim([0, 1.05])
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.set_xticks(xticks)
#    ax.set_xticklabels(['','',''])
    ax.set_xticklabels(['','','','',''])
    ax.plot(time[:], P[:,j], 'b-', label="node "+str(j))
#    ax.plot(time[trainsteps:Timesteps], pred[trainsteps:Timesteps,j], 'r--', label="forecast")
#    ax.plot(time[trainsteps:Timesteps], pred[trainsteps:Timesteps,j], 'c--', label="cVAR forecast")
    for currstep in range(0,Timesteps,Epoch):
        ax.plot(time[currstep+trainsteps:currstep+trainsteps+predictsteps], pred[currstep+trainsteps:currstep+trainsteps+predictsteps,j], 'c--')
        ### standard deviation of constrained VAR
        ax.fill_between(time[currstep+trainsteps:currstep+trainsteps+predictsteps], pred[currstep+trainsteps:currstep+trainsteps+predictsteps,j]-stdev[j], \
                     pred[currstep+trainsteps:currstep+trainsteps+predictsteps,j]+stdev[j], color='gold')

    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='upper left',ncol=3)
    j+=1
#ax.set_xticklabels(['0','20','40'])
ax.set_xticklabels(['0','20','40','60','80'])
plt.xlabel('time')
plt.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.95, wspace=0, hspace=0)
#plt.subplots_adjust(left=0.04, right=0.98, bottom=0.06, top=0.98, wspace=0, hspace=0)
plt.show()


### plot the dynamics of prediction probabilities
### value-constrained VAR model
#fig, axs = plt.subplots(nrows=TotalNumNode,ncols=1,figsize=(3,4))
fig, axs = plt.subplots(nrows=TotalNumNode,ncols=1,figsize=(6,8))
fig.suptitle("Value Constrained VAR (Worst-Case fusion): # of nodes="+str(TotalNumNode))
#fig.suptitle("Value Constrained VAR (Best-Case Fusion): # of nodes="+str(TotalNumNode))
#fig.suptitle("Value Constrained VAR (Bayesian Fusion): # of nodes="+str(TotalNumNode))
j = 0
for ax in axs.flat:
    ax.set_xlim([0, time[Timesteps-1]])
    ax.set_ylim([0, 1.05])
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.set_xticks(xticks)
#    ax.set_xticklabels(['','',''])
    ax.set_xticklabels(['','','','',''])
    ax.plot(time[:], P[:,j], 'b-', label="node "+str(j))
    for currstep in range(0,Timesteps,Epoch):
        ax.plot(time[currstep+trainsteps:currstep+trainsteps+predictsteps], pred3[currstep+trainsteps:currstep+trainsteps+predictsteps,j], 'c--')
        ### standard deviation of constrained VAR
        ax.fill_between(time[currstep+trainsteps:currstep+trainsteps+predictsteps], pred3[currstep+trainsteps:currstep+trainsteps+predictsteps,j]-stdev3[j], \
                     pred3[currstep+trainsteps:currstep+trainsteps+predictsteps,j]+stdev3[j], color='gold')

    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='upper left',ncol=3)
    j+=1
#ax.set_xticklabels(['0','20','40'])
ax.set_xticklabels(['0','20','40','60','80'])
plt.xlabel('time')
plt.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.95, wspace=0, hspace=0)
#plt.subplots_adjust(left=0.04, right=0.98, bottom=0.06, top=0.98, wspace=0, hspace=0)
plt.show()


### plot the dynamics of prediction probabilities
### VAR model
#fig, axs = plt.subplots(nrows=TotalNumNode,ncols=1,figsize=(3,4))
fig, axs = plt.subplots(nrows=TotalNumNode,ncols=1,figsize=(6,8))
fig.suptitle("VAR (Worst-Case fusion): # of nodes="+str(TotalNumNode))
#fig.suptitle("VAR (Best-Case Fusion): # of nodes="+str(TotalNumNode))
#fig.suptitle("VAR (Bayesian Fusion): # of nodes="+str(TotalNumNode))
j = 0
for ax in axs.flat:
    ax.set_xlim([0, time[Timesteps-1]])
    ax.set_ylim([0, 1.05])
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.set_xticks(xticks)
#    ax.set_xticklabels(['','',''])
    ax.set_xticklabels(['','','','',''])
    ax.plot(time[:], P[:,j], 'b-', label="node "+str(j))
#    ax.plot(time[trainsteps:Timesteps], pred[trainsteps:Timesteps,j], 'r--', label="forecast")
##    ax.plot(time[trainsteps:Timesteps], pred[trainsteps:Timesteps,j], 'c--', label="cVAR forecast")
#    ax.plot(time[trainsteps:Timesteps], pred2[trainsteps:Timesteps,j], 'r--', label="VAR forecast")
    for currstep in range(0,Timesteps,Epoch):
        ax.plot(time[currstep+trainsteps:currstep+trainsteps+predictsteps], pred2[currstep+trainsteps:currstep+trainsteps+predictsteps,j], 'c--')
        ### standard deviation of constrained VAR
        ax.fill_between(time[currstep+trainsteps:currstep+trainsteps+predictsteps], pred2[currstep+trainsteps:currstep+trainsteps+predictsteps,j]-stdev2[j], \
                     pred2[currstep+trainsteps:currstep+trainsteps+predictsteps,j]+stdev2[j], color='gold')

    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='upper left',ncol=3)
    j+=1
#ax.set_xticklabels(['0','20','40'])
ax.set_xticklabels(['0','20','40','60','80'])
plt.xlabel('time')
plt.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.95, wspace=0, hspace=0)
#plt.subplots_adjust(left=0.04, right=0.98, bottom=0.06, top=0.98, wspace=0, hspace=0)
plt.show()

