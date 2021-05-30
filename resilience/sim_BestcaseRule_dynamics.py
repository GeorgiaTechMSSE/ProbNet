####################################################################
### ProbNet: A probabilistic graph modeling toolkit that enables the design of cyber-physical-social systems 
###	     with the considerations of trust, resilience, and information dynamics',
###   author: Yan Wang,
###   author_email: yan-wang@gatech.edu
####################################################################
import numpy as np
import matplotlib.pyplot as plt
import pdb
##import sys
##sys.path.append("C:\\Users\\Wang\\VMShared\\py\\cps\\ProbNet")
from Dijkstra_shortest_path import *

TimeOfFailure = 50 ##time step when failure starts
TimeOfStable = 100 ##time step when failure stops
TimeOfRecover = 150 ##time step when recovery starts
TimeOfFullRecover = 250  ##time step when recovery stops
Timesteps = 300   ##total number of steps in simulation
#NumNode = 10
NumNode = 30
SampleSize = 500  ##number of MC samples used to update P() in each time step

#FailRate = 2   ##how many nodes are disconnected in each time step
FailRate = 50   ##how many nodes are disconnected in each time step

#the smalleast probability to avoid log(0) in calculating entropy
EPSILON=np.ones(NumNode)*1e-10

time = np.zeros((1,Timesteps))

#node probabilities
P = np.zeros((NumNode,Timesteps))
P[:,0]=np.ones(NumNode)*0.5

#node entropy
H = np.zeros((NumNode,Timesteps))
H2 = np.zeros((NumNode,Timesteps))

#connection probabilities
"""PP = np.zeros((NumNode,NumNode))
PP[0,1]=0.6
PP[0,2]=0.5
PP[0,3]=0.6
PP[1,0]=0.5
PP[1,2]=0.5
PP[2,0]=0.5
PP[3,0]=0.1
PP[:,0]=np.ones(NumNode)*0.5
PP[2,:]=np.ones(NumNode)*0.5
"""
PP = np.random.random_sample((NumNode,NumNode))

#pdb.set_trace()

#connection probabilities
"""QQ = np.zeros((NumNode,NumNode))
QQ[0,1]=0.1
QQ[0,2]=0.1
QQ[0,3]=0.1
QQ[1,0]=0.1
QQ[1,2]=0.1
QQ[2,0]=0.1
QQ[3,0]=0.1
QQ[:,0]=np.ones(NumNode)*0.1
QQ[2,:]=np.ones(NumNode)*0.1
"""
QQ = np.random.random_sample((NumNode,NumNode))

###################
##set Pr(x|x) to consider its own prediction or not
for k in range(0,NumNode):
    PP[k,k]=P[k,0]  ## its own prediction is considered
#    PP[k,k]=0       ## its own prediction is not considered
    QQ[k,k]=0
###################
 
Z_PP=np.array(PP)
Z_QQ=np.array(QQ)

#conditional entropy
HH = np.zeros((NumNode,NumNode,Timesteps))
'''HH[0,1]=-PP[0,1]*P[0,0]*np.log2(PP[0,1])-(1-PP[0,1])*P[0,0]*np.log2(1-PP[0,1])-QQ[0,1]*(1-P[0,0])*np.log2(QQ[0,1])-(1-QQ[0,1])*(1-P[0,0])*np.log2(1-QQ[0,1])
HH[0,3]=-PP[0,3]*P[0,0]*np.log2(PP[0,3])-(1-PP[0,3])*P[0,0]*np.log2(1-PP[0,3])-QQ[0,3]*(1-P[0,0])*np.log2(QQ[0,3])-(1-QQ[0,3])*(1-P[0,0])*np.log2(1-QQ[0,3])
HH[1,2]=-PP[1,2]*P[1,0]*np.log2(PP[1,2])-(1-PP[1,2])*P[1,0]*np.log2(1-PP[1,2])-QQ[1,2]*(1-P[1,0])*np.log2(QQ[1,2])-(1-QQ[1,2])*(1-P[1,0])*np.log2(1-QQ[1,2])
HH[2,0]=-PP[2,0]*P[2,0]*np.log2(PP[2,0])-(1-PP[2,0])*P[2,0]*np.log2(1-PP[2,0])-QQ[2,0]*(1-P[2,0])*np.log2(QQ[2,0])-(1-QQ[2,0])*(1-P[2,0])*np.log2(1-QQ[2,0])
HH[3,0]=-PP[3,0]*P[3,0]*np.log2(PP[3,0])-(1-PP[3,0])*P[3,0]*np.log2(1-PP[3,0])-QQ[3,0]*(1-P[3,0])*np.log2(QQ[3,0])-(1-QQ[3,0])*(1-P[3,0])*np.log2(1-QQ[3,0])
'''
#mutual information
MU = np.zeros((NumNode,NumNode,Timesteps))
MU2 = np.zeros((NumNode,NumNode,Timesteps))

#conditional probabilities P(i,j)
P01 = np.zeros((NumNode,NumNode,Timesteps))
P00 = np.zeros((NumNode,NumNode,Timesteps))
P11 = np.zeros((NumNode,NumNode,Timesteps))
P10 = np.zeros((NumNode,NumNode,Timesteps))

### initial values
t=0
H[:,0]=-P[:,0]*np.log2(np.maximum(P[:,0],EPSILON))-(1-P[:,0])*np.log2(np.maximum(1-P[:,0],EPSILON))
#H[:,t]=-P[:,t]*np.log2(np.maximum(P[:,t],EPSILON))-(1-P[:,t])*np.log2(np.maximum(1-P[:,t],EPSILON))
for k in range(0,NumNode):  #for each StartNode, sample EndNode, i.e. row of PP or QQ 
    HH[k,:,t]=-PP[k,:]*P[k,t]*np.log2(np.maximum(PP[k,:],EPSILON))      \
                -(1-PP[k,:])*P[k,t]*np.log2(np.maximum(1-PP[k,:],EPSILON))  \
                -QQ[k,:]*(1-P[k,t])*np.log2(np.maximum(QQ[k,:],EPSILON))    \
                -(1-QQ[k,:])*(1-P[k,t])*np.log2(np.maximum(1-QQ[k,:],EPSILON))
    MU[k,:,t]=PP[k,:]*P[k,t]*(np.log2(np.maximum(PP[k,:],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))        \
                +(1-PP[k,:])*P[k,t]*(np.log2(np.maximum(1-PP[k,:],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
                +QQ[k,:]*(1-P[k,t])*(np.log2(np.maximum(QQ[k,:],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))      \
                +(1-QQ[k,:])*(1-P[k,t])*(np.log2(np.maximum(1-QQ[k,:],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))
    MU2[k,:,t]=P11[k,:,t]*(np.log2(np.maximum(P11[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))  \
                +P10[k,:,t]*(np.log2(np.maximum(P10[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
                +P01[k,:,t]*(np.log2(np.maximum(P01[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
                +P00[k,:,t]*(np.log2(np.maximum(P00[k,:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))

### Check the relation between mutual information and entropy
for k in range(0,NumNode):
    H2[k,t]=np.average(MU[:,k,t])
#    H2[k,t]=np.sum(MU2[:,k,t])
    
#pdb.set_trace()
### simulation main loop - before failure
for t in range(1,TimeOfFailure):
    time[0,t]=t
    EndNode_sample = np.zeros(NumNode)
    for s in range(0,SampleSize):
        StartNodeTally=np.random.random_sample(NumNode)<P[:,t-1]
#######best-case/optimistic
        EndNodeTally = np.zeros(NumNode)
#######worst-case/pessimistic
#        EndNodeTally = np.ones(NumNode)
        for k in range(0,NumNode):  #for each StartNode, sample EndNode, i.e. row of PP or QQ 
            if StartNodeTally[k]:
                P11Tally=1-(np.random.random_sample(NumNode)>PP[k,:])
                P10Tally=1-P11Tally
                P01Tally=0
                P00Tally=0
                P11[k,:,t]+=P11Tally
                P10[k,:,t]+=P10Tally
#######best-case/optimistic
                EndNodeTally = np.maximum(P11Tally,EndNodeTally)
#######worst-case/pessimistic
#                EndNodeTally = np.minimum(P11Tally,EndNodeTally)
            else:
                P01Tally=1-(np.random.random_sample(NumNode)>QQ[k,:])
                P00Tally=1-P01Tally
                P11Tally=0
                P10Tally=0
                P01[k,:,t]+=P01Tally
                P00[k,:,t]+=P00Tally
##                EndNodeTally = np.maximum(P01Tally,EndNodeTally)
#######worst-case/pessimistic
#                EndNodeTally = np.minimum(P01Tally,EndNodeTally)
##                EndNodeTally += P01Tally
###### Bayesian update ####
#            pdb.set_trace()
#            if t>0:
#                np.copyto(P[:,t],P[:,t-1])
#            P[:,t]=np.power(P[:,t],P11Tally+P01Tally)*np.power((1-P[:,t]),P10Tally+P00Tally)
#            P[:,t]=P[:,t]/np.sum(P[:,t])
###########################
        EndNode_sample+=EndNodeTally
    P[:,t]=EndNode_sample/SampleSize
    P01[:,:,t]=P01[:,:,t]/SampleSize
    P11[:,:,t]=P11[:,:,t]/SampleSize
    P00[:,:,t]=P00[:,:,t]/SampleSize
    P10[:,:,t]=P10[:,:,t]/SampleSize

    H[:,t]=-P[:,t]*np.log2(np.maximum(P[:,t],EPSILON))-(1-P[:,t])*np.log2(np.maximum(1-P[:,t],EPSILON))
    for k in range(0,NumNode):  #for each StartNode, sample EndNode, i.e. row of PP or QQ 
        HH[k,:,t]=-PP[k,:]*P[k,t]*np.log2(np.maximum(PP[k,:],EPSILON))      \
                -(1-PP[k,:])*P[k,t]*np.log2(np.maximum(1-PP[k,:],EPSILON))  \
                -QQ[k,:]*(1-P[k,t])*np.log2(np.maximum(QQ[k,:],EPSILON))    \
                -(1-QQ[k,:])*(1-P[k,t])*np.log2(np.maximum(1-QQ[k,:],EPSILON))
        MU[k,:,t]=PP[k,:]*P[k,t]*(np.log2(np.maximum(PP[k,:],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))        \
                +(1-PP[k,:])*P[k,t]*(np.log2(np.maximum(1-PP[k,:],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
                +QQ[k,:]*(1-P[k,t])*(np.log2(np.maximum(QQ[k,:],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))      \
                +(1-QQ[k,:])*(1-P[k,t])*(np.log2(np.maximum(1-QQ[k,:],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))
        MU2[k,:,t]=P11[k,:,t]*(np.log2(np.maximum(P11[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))  \
                +P10[k,:,t]*(np.log2(np.maximum(P10[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
                +P01[k,:,t]*(np.log2(np.maximum(P01[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
                +P00[k,:,t]*(np.log2(np.maximum(P00[k,:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))
    ### Check the relation between mutual information and entropy
    for k in range(0,NumNode):
        H2[k,t]=np.average(MU[:,k,t])
#        H2[k,t]=np.sum(MU2[:,k,t])

#node indices that connections are broken
S_discon=np.random.random_integers(0,NumNode-1,(2,FailRate,TimeOfStable-TimeOfFailure))
#store the original reliance probabilies of the disconnections
#S_pq=np.zeros(S_discon.shape)

#pdb.set_trace()
### simulation main loop - During failure
for t in range(TimeOfFailure,TimeOfStable):
    time[0,t]=t
    EndNode_sample = np.zeros(NumNode)
    for s in range(0,SampleSize):
        StartNodeTally=np.random.random_sample(NumNode)<P[:,t-1]
#######best-case/optimistic
        EndNodeTally = np.zeros(NumNode)
#######worst-case/pessimistic
#        EndNodeTally = np.ones(NumNode)
        for k in range(0,NumNode):  #for each StartNode, sample EndNode, i.e. row of PP or QQ 
            if StartNodeTally[k]:
                P11Tally=1-(np.random.random_sample(NumNode)>PP[k,:])
                P10Tally=1-P11Tally
                P01Tally=0
                P00Tally=0
                P11[k,:,t]+=P11Tally
                P10[k,:,t]+=P10Tally
#######best-case/optimistic
                EndNodeTally = np.maximum(P11Tally,EndNodeTally)  #best-case/optimistic
#######worst-case/pessimistic
#                EndNodeTally = np.minimum(P11Tally,EndNodeTally)
            else:
                P01Tally=1-(np.random.random_sample(NumNode)>QQ[k,:])
                P00Tally=1-P01Tally
                P11Tally=0
                P10Tally=0
                P01[k,:,t]+=P01Tally
                P00[k,:,t]+=P00Tally
##                EndNodeTally = np.maximum(P01Tally,EndNodeTally)
#######worst-case/pessimistic
#                EndNodeTally = np.minimum(P01Tally,EndNodeTally)
#                EndNodeTally += P01Tally
###### Bayesian update ####
#            np.copyto(P[:,t],P[:,t-1])
#            P[:,t]=np.power(P[:,t],P11Tally+P01Tally)*np.power((1-P[:,t]),P10Tally+P00Tally)
#            P[:,t]=P[:,t]/np.sum(P[:,t])
########################### 
        EndNode_sample+=EndNodeTally
    P[:,t]=EndNode_sample/SampleSize
    P01[:,:,t]=P01[:,:,t]/SampleSize
    P11[:,:,t]=P11[:,:,t]/SampleSize
    P00[:,:,t]=P00[:,:,t]/SampleSize
    P10[:,:,t]=P10[:,:,t]/SampleSize

    H[:,t]=-P[:,t]*np.log2(np.maximum(P[:,t],EPSILON))-(1-P[:,t])*np.log2(np.maximum(1-P[:,t],EPSILON))
    for k in range(0,NumNode):  #for each StartNode, sample EndNode, i.e. row of PP or QQ 
        HH[k,:,t]=-PP[k,:]*P[k,t]*np.log2(np.maximum(PP[k,:],EPSILON))      \
                -(1-PP[k,:])*P[k,t]*np.log2(np.maximum(1-PP[k,:],EPSILON))  \
                -QQ[k,:]*(1-P[k,t])*np.log2(np.maximum(QQ[k,:],EPSILON))    \
                -(1-QQ[k,:])*(1-P[k,t])*np.log2(np.maximum(1-QQ[k,:],EPSILON))
        MU[k,:,t]=PP[k,:]*P[k,t]*(np.log2(np.maximum(PP[k,:],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))        \
                +(1-PP[k,:])*P[k,t]*(np.log2(np.maximum(1-PP[k,:],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
                +QQ[k,:]*(1-P[k,t])*(np.log2(np.maximum(QQ[k,:],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))      \
                +(1-QQ[k,:])*(1-P[k,t])*(np.log2(np.maximum(1-QQ[k,:],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))
        MU2[k,:,t]=P11[k,:,t]*(np.log2(np.maximum(P11[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))  \
                +P10[k,:,t]*(np.log2(np.maximum(P10[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
                +P01[k,:,t]*(np.log2(np.maximum(P01[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
                +P00[k,:,t]*(np.log2(np.maximum(P00[k,:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))
    ### Check the relation between mutual information and entropy
    for k in range(0,NumNode):
        H2[k,t]=np.average(MU[:,k,t])
#        H2[k,t]=np.sum(MU2[:,k,t])
######disconnect the connection between a random pair of nodes
    for j in range(0,FailRate):
##### store the original reliance probabilities
#        S_pq[0,j,t-TimeOfFailure]=PP[S_discon[0,j,t-TimeOfFailure],S_discon[1,j,t-TimeOfFailure]]
#        S_pq[1,j,t-TimeOfFailure]=QQ[S_discon[0,j,t-TimeOfFailure],S_discon[1,j,t-TimeOfFailure]]
##### set the reliance probabilities as disruption
        PP[S_discon[0,j,t-TimeOfFailure],S_discon[1,j,t-TimeOfFailure]]=0.  ##
        QQ[S_discon[0,j,t-TimeOfFailure],S_discon[1,j,t-TimeOfFailure]]=0.  ##
#    if t == 60:
#        pdb.set_trace()

#pdb.set_trace()
### simulation main loop - before recovery
for t in range(TimeOfStable, TimeOfRecover):
    time[0,t]=t
    EndNode_sample = np.zeros(NumNode)
    for s in range(0,SampleSize):
        StartNodeTally=np.random.random_sample(NumNode)<P[:,t-1]
#######best-case/optimistic
        EndNodeTally = np.zeros(NumNode)
#######worst-case/pessimistic
#        EndNodeTally = np.ones(NumNode)
        for k in range(0,NumNode):  #for each StartNode, sample EndNode, i.e. row of PP or QQ 
            if StartNodeTally[k]:
                P11Tally=1-(np.random.random_sample(NumNode)>PP[k,:])
                P10Tally=1-P11Tally
                P01Tally=0
                P00Tally=0
                P11[k,:,t]+=P11Tally
                P10[k,:,t]+=P10Tally
#######best-case/optimistic
                EndNodeTally = np.maximum(P11Tally,EndNodeTally)
#######worst-case/pessimistic
#                EndNodeTally = np.minimum(P11Tally,EndNodeTally)
            else:
                P01Tally=1-(np.random.random_sample(NumNode)>QQ[k,:])
                P00Tally=1-P01Tally
                P11Tally=0
                P10Tally=0
                P01[k,:,t]+=P01Tally
                P00[k,:,t]+=P00Tally
##                EndNodeTally = np.maximum(P01Tally,EndNodeTally)
#                EndNodeTally += P01Tally
###### Bayesian update ####
#            np.copyto(P[:,t],P[:,t-1])
#            P[:,t]=np.power(P[:,t],P11Tally+P01Tally)*np.power((1-P[:,t]),P10Tally+P00Tally)
#            P[:,t]=P[:,t]/np.sum(P[:,t])
###########################    
        EndNode_sample+=EndNodeTally
    P[:,t]=EndNode_sample/SampleSize
    P01[:,:,t]=P01[:,:,t]/SampleSize
    P11[:,:,t]=P11[:,:,t]/SampleSize
    P00[:,:,t]=P00[:,:,t]/SampleSize
    P10[:,:,t]=P10[:,:,t]/SampleSize
    H[:,t]=-P[:,t]*np.log2(np.maximum(P[:,t],EPSILON))-(1-P[:,t])*np.log2(np.maximum(1-P[:,t],EPSILON))
    for k in range(0,NumNode):  #for each StartNode, sample EndNode, i.e. row of PP or QQ 
        HH[k,:,t]=-PP[k,:]*P[k,t]*np.log2(np.maximum(PP[k,:],EPSILON))      \
                -(1-PP[k,:])*P[k,t]*np.log2(np.maximum(1-PP[k,:],EPSILON))  \
                -QQ[k,:]*(1-P[k,t])*np.log2(np.maximum(QQ[k,:],EPSILON))    \
                -(1-QQ[k,:])*(1-P[k,t])*np.log2(np.maximum(1-QQ[k,:],EPSILON))
        MU[k,:,t]=PP[k,:]*P[k,t]*(np.log2(np.maximum(PP[k,:],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))        \
                +(1-PP[k,:])*P[k,t]*(np.log2(np.maximum(1-PP[k,:],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
                +QQ[k,:]*(1-P[k,t])*(np.log2(np.maximum(QQ[k,:],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))      \
                +(1-QQ[k,:])*(1-P[k,t])*(np.log2(np.maximum(1-QQ[k,:],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))
        MU2[k,:,t]=P11[k,:,t]*(np.log2(np.maximum(P11[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))  \
                +P10[k,:,t]*(np.log2(np.maximum(P10[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
                +P01[k,:,t]*(np.log2(np.maximum(P01[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
                +P00[k,:,t]*(np.log2(np.maximum(P00[k,:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))
    ### Check the relation between mutual information and entropy
    for k in range(0,NumNode):
        H2[k,t]=np.average(MU[:,k,t])
#        H2[k,t]=np.sum(MU2[:,k,t])

#pdb.set_trace()
MaxNumDisconnect=np.sum(1-(1-(PP==0.)))

####
g=Graph()
for k in range(0,NumNode):
    g.add_vertex('v'+str(k))
for i in range(0,NumNode):
    for j in range(0,NumNode):
        if PP[i,j]>0:
            g.add_edge('v'+str(i),'v'+str(j),PP[i,j])

f = open('dotgraph.txt','w')
f.writelines('digraph G {\nnode [width=.3,height=.3,shape=octagon,style=filled,color=skyblue];\noverlap="false";\nrankdir="LR";\n')
f.writelines
for v in g:
    for w in v.get_connections():
        s= '      '+ v.get_id()
        s +=  ' -> ' +  w.get_id() + ' [label="' + str(v.get_weight(w)) + '"]'
        s+=';\n'
        f.writelines(s)
        
f.writelines('}')
f.close()
####

### simulation main loop - For recovery
for t in range(TimeOfRecover, TimeOfFullRecover):
    time[0,t]=t
    EndNode_sample = np.zeros(NumNode)
    for s in range(0,SampleSize):
        StartNodeTally=np.random.random_sample(NumNode)<P[:,t-1]
#######best-case/optimistic
        EndNodeTally = np.zeros(NumNode)
#######worst-case/pessimistic
#        EndNodeTally = np.ones(NumNode)
        for k in range(0,NumNode):  #for each StartNode, sample EndNode, i.e. row of PP or QQ 
            if StartNodeTally[k]:
                P11Tally=1-(np.random.random_sample(NumNode)>PP[k,:])
                P10Tally=1-P11Tally
                P01Tally=0
                P00Tally=0
                P11[k,:,t]+=P11Tally
                P10[k,:,t]+=P10Tally
#######best-case/optimistic
                EndNodeTally = np.maximum(P11Tally,EndNodeTally)
#######worst-case/pessimistic
#                EndNodeTally = np.minimum(P11Tally,EndNodeTally)
            else:
                P01Tally=1-(np.random.random_sample(NumNode)>QQ[k,:])
                P00Tally=1-P01Tally
                P11Tally=0
                P10Tally=0
                P01[k,:,t]+=P01Tally
                P00[k,:,t]+=P00Tally
##                EndNodeTally = np.maximum(P01Tally,EndNodeTally)
#                EndNodeTally += P01Tally
###### Bayesian update ####
#            np.copyto(P[:,t],P[:,t-1])
#            P[:,t]=np.power(P[:,t],P11Tally+P01Tally)*np.power((1-P[:,t]),P10Tally+P00Tally)
#            P[:,t]=P[:,t]/np.sum(P[:,t])
###########################    
        EndNode_sample+=EndNodeTally
    P[:,t]=EndNode_sample/SampleSize
    #####simulated conditional probabilities
    P01[:,:,t]=P01[:,:,t]/SampleSize
    P11[:,:,t]=P11[:,:,t]/SampleSize
    P00[:,:,t]=P00[:,:,t]/SampleSize
    P10[:,:,t]=P10[:,:,t]/SampleSize
#    pdb.set_trace()
    H[:,t]=-P[:,t]*np.log2(np.maximum(P[:,t],EPSILON))-(1-P[:,t])*np.log2(np.maximum(1-P[:,t],EPSILON))
    for k in range(0,NumNode):  #for each StartNode, sample EndNode, i.e. row of PP or QQ 
        HH[k,:,t]=-PP[k,:]*P[k,t]*np.log2(np.maximum(PP[k,:],EPSILON))      \
                -(1-PP[k,:])*P[k,t]*np.log2(np.maximum(1-PP[k,:],EPSILON))  \
                -QQ[k,:]*(1-P[k,t])*np.log2(np.maximum(QQ[k,:],EPSILON))    \
                -(1-QQ[k,:])*(1-P[k,t])*np.log2(np.maximum(1-QQ[k,:],EPSILON))
        MU[k,:,t]=PP[k,:]*P[k,t]*(np.log2(np.maximum(PP[k,:],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))        \
                +(1-PP[k,:])*P[k,t]*(np.log2(np.maximum(1-PP[k,:],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
                +QQ[k,:]*(1-P[k,t])*(np.log2(np.maximum(QQ[k,:],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))      \
                +(1-QQ[k,:])*(1-P[k,t])*(np.log2(np.maximum(1-QQ[k,:],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))
        ##### mutual information calculated from simulated conditional probabilities
        MU2[k,:,t]=P11[k,:,t]*(np.log2(np.maximum(P11[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))  \
                +P10[k,:,t]*(np.log2(np.maximum(P10[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
                +P01[k,:,t]*(np.log2(np.maximum(P01[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
                +P00[k,:,t]*(np.log2(np.maximum(P00[k,:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))
    ### Check the relation between mutual information and entropy
    for k in range(0,NumNode):
        H2[k,t]=np.average(MU[:,k,t])
#        H2[k,t]=np.sum(MU2[:,k,t])
######recover the connection of nodes
    if t-TimeOfRecover<TimeOfStable-TimeOfFailure:
        for j in range(0,FailRate):
            PP[S_discon[0,j,t-TimeOfRecover],S_discon[1,j,t-TimeOfRecover]]=Z_PP[S_discon[0,j,t-TimeOfRecover],S_discon[1,j,t-TimeOfRecover]]
            QQ[S_discon[0,j,t-TimeOfRecover],S_discon[1,j,t-TimeOfRecover]]=Z_QQ[S_discon[0,j,t-TimeOfRecover],S_discon[1,j,t-TimeOfRecover]]
##            PP[S_discon[0,j,t-TimeOfRecover],S_discon[1,j,t-TimeOfRecover]]=S_pq[0,j,t-TimeOfRecover]
##            QQ[S_discon[0,j,t-TimeOfRecover],S_discon[1,j,t-TimeOfRecover]]=S_pq[1,j,t-TimeOfRecover]
"""            if PP[S_discon[0,j,t-TimeOfRecover],S_discon[1,j,t-TimeOfRecover]]==0.5:
                PP[S_discon[0,j,t-TimeOfRecover],S_discon[1,j,t-TimeOfRecover]]=np.random.random_sample()  
            if QQ[S_discon[0,j,t-TimeOfRecover],S_discon[1,j,t-TimeOfRecover]]==0.5:
                QQ[S_discon[0,j,t-TimeOfRecover],S_discon[1,j,t-TimeOfRecover]]=np.random.random_sample()  
"""
#pdb.set_trace()

### simulation main loop - after full recovery
for t in range(TimeOfFullRecover,Timesteps):
    time[0,t]=t
    EndNode_sample = np.zeros(NumNode)
    for s in range(0,SampleSize):
        StartNodeTally=np.random.random_sample(NumNode)<P[:,t-1]
#######best-case/optimistic
        EndNodeTally = np.zeros(NumNode)
#######worst-case/pessimistic
#        EndNodeTally = np.ones(NumNode)
        for k in range(0,NumNode):  #for each StartNode, sample EndNode, i.e. row of PP or QQ 
            if StartNodeTally[k]:
                P11Tally=1-(np.random.random_sample(NumNode)>PP[k,:])
                P10Tally=1-P11Tally
                P01Tally=0
                P00Tally=0
                P11[k,:,t]+=P11Tally
                P10[k,:,t]+=P10Tally
#######best-case/optimistic
                EndNodeTally = np.maximum(P11Tally,EndNodeTally)
#######worst-case/pessimistic
#                EndNodeTally = np.minimum(P11Tally,EndNodeTally)
            else:
                P01Tally=1-(np.random.random_sample(NumNode)>QQ[k,:])
                P00Tally=1-P01Tally
                P11Tally=0
                P10Tally=0
                P01[k,:,t]+=P01Tally
                P00[k,:,t]+=P00Tally
##                EndNodeTally = np.maximum(P01Tally,EndNodeTally)
#                EndNodeTally += P01Tally
###### Bayesian update ####
#            np.copyto(P[:,t],P[:,t-1])
#            P[:,t]=np.power(P[:,t],P11Tally+P01Tally)*np.power((1-P[:,t]),P10Tally+P00Tally)
#            P[:,t]=P[:,t]/np.sum(P[:,t])
###########################    
        EndNode_sample+=EndNodeTally
    P[:,t]=EndNode_sample/SampleSize
    P01[:,:,t]=P01[:,:,t]/SampleSize
    P11[:,:,t]=P11[:,:,t]/SampleSize
    P00[:,:,t]=P00[:,:,t]/SampleSize
    P10[:,:,t]=P10[:,:,t]/SampleSize
#    pdb.set_trace()
    H[:,t]=-P[:,t]*np.log2(np.maximum(P[:,t],EPSILON))-(1-P[:,t])*np.log2(np.maximum(1-P[:,t],EPSILON))
    for k in range(0,NumNode):  #for each StartNode, sample EndNode, i.e. row of PP or QQ 
        HH[k,:,t]=-PP[k,:]*P[k,t]*np.log2(np.maximum(PP[k,:],EPSILON))      \
                -(1-PP[k,:])*P[k,t]*np.log2(np.maximum(1-PP[k,:],EPSILON))  \
                -QQ[k,:]*(1-P[k,t])*np.log2(np.maximum(QQ[k,:],EPSILON))    \
                -(1-QQ[k,:])*(1-P[k,t])*np.log2(np.maximum(1-QQ[k,:],EPSILON))
        MU[k,:,t]=PP[k,:]*P[k,t]*(np.log2(np.maximum(PP[k,:],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))        \
                +(1-PP[k,:])*P[k,t]*(np.log2(np.maximum(1-PP[k,:],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
                +QQ[k,:]*(1-P[k,t])*(np.log2(np.maximum(QQ[k,:],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))      \
                +(1-QQ[k,:])*(1-P[k,t])*(np.log2(np.maximum(1-QQ[k,:],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))
        MU2[k,:,t]=P11[k,:,t]*(np.log2(np.maximum(P11[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))  \
                +P10[k,:,t]*(np.log2(np.maximum(P10[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
                +P01[k,:,t]*(np.log2(np.maximum(P01[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
                +P00[k,:,t]*(np.log2(np.maximum(P00[k,:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))
    ### Check the relation between mutual information and entropy
    for k in range(0,NumNode):
        H2[k,t]=np.average(MU[:,k,t])
#        H2[k,t]=np.sum(MU2[:,k,t])

"""
plt.subplot(4, 1, 1)
plt.plot(time[0,:], P[0,:], 'y-')
plt.plot(time[0,:], HH[0,1,:], 'r--')
plt.plot(time[0,:], MU[0,1,:], 'g--')
plt.title('evolution of probabilities')
plt.ylabel('$P_1$')
plt.subplot(4, 1, 2)
plt.plot(time[0,:], P[1,:], 'y-')
plt.plot(time[0,:], HH[1,2,:], 'r--')
plt.plot(time[0,:], MU[1,0,:], 'g--')
plt.ylabel('$P_2$')
"""
"""
plt.subplot(2, 1, 1)
###plt.plot(time[0,:], P[2,:], 'y-')
###plt.plot(time[0,:], HH[2,3,:], 'r--')
plt.plot(time[0,:], np.average(P,axis=0), 'g-')
plt.ylabel('$average(P)$')
plt.ylim([0, 1.1])
"""
fig, ax = plt.subplots()
plt.plot(time[0,:], np.amin(P,axis=0), 'c.-', label="min pred. prob.")
plt.plot(time[0,:], np.amax(P,axis=0), 'm-', label="max pred. prob.")
ax.set_title('Total number of nodes='+str(NumNode)+', a maximum of '+str(MaxNumDisconnect)+' disrupted edges')
plt.ylabel('Probability')
plt.ylim([0.0, 1.05])
plt.xlabel('time')
ax.xaxis.grid(False)
ax.yaxis.grid(True)
plt.grid()
plt.legend(loc='lower right')

fig, ax = plt.subplots()
plt.plot(time[0,:], np.average(np.average(HH,axis=0),axis=0), 'g--', label="ave. cond. entropy")
plt.plot(time[0,:], np.average(H,axis=0), 'b-', label="ave. entropy")
plt.plot(time[0,:], np.average(H2,axis=0), 'r*-', label="ave. entropy2")
ax.set_title('Total number of nodes='+str(NumNode)+', a maximum of '+str(MaxNumDisconnect)+' disrupted edges')
#ax.set_ylim([0, 1])
ax.set_ylabel('entropy and conditional entropy')
ax.set_xlabel('time')
ax.xaxis.grid(False)
ax.yaxis.grid(True)
plt.grid()
plt.legend(loc='upper right')


fig, ax = plt.subplots()
plt.plot(time[0,:], np.average(np.average(MU,axis=0),axis=0)/2, 'r-', label="MU")
#plt.plot(time[0,:], np.average(np.average(MU2,axis=0),axis=0)/2, 'c-', label="MU2")
ax.set_title('Total number of nodes='+str(NumNode)+', a maximum of '+str(MaxNumDisconnect)+' disrupted edges')
#ax.set_ylim([0, 15])
ax.set_ylabel('$F=\sum(mutual information)/(2N^2)$')
ax.set_xlabel('time')
ax.xaxis.grid(False)
ax.yaxis.grid(True)
plt.grid()

plt.show()
