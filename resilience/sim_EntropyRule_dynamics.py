####################################################################
### ProbNet: A probabilistic graph modeling toolkit that enables the design of cyber-physical-social systems 
###	     with the considerations of trust, resilience, and information dynamics',
###   author: Yan Wang,
###   author_email: yan-wang@gatech.edu
####################################################################
import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy import optimize

TimeOfFailure = 50 ##time step when failure starts
TimeOfStable = 100 ##time step when failure stops
TimeOfRecover = 150 ##time step when recovery starts
TimeOfFullRecover = 250  ##time step when recovery stops
Timesteps = 300   ##total number of steps in simulation
NumNode = 30
SampleSize = 500  ##number of MC samples used to update P() in each time step

#FailRate = 40   ##how many nodes are disconnected in each time step
FailRate = 50   ##how many nodes are disconnected in each time step

#the smalleast probability to avoid log(0) in calculating entropy
EPSILON=np.ones(NumNode)*1e-10
EPSILON2=np.ones([NumNode,NumNode])*1e-10

time = np.zeros((1,Timesteps))

#node probabilities
P = np.zeros((NumNode,Timesteps))
P[:,0]=np.ones(NumNode)*0.95

#node entropy
H = np.zeros((NumNode,Timesteps))

Hjt2= np.zeros((NumNode,NumNode,Timesteps))
HH2= np.zeros((NumNode,NumNode,Timesteps))
H_ave = np.zeros((NumNode,Timesteps))

# a function to inversely find probability from entropy as root finding
def funcInvEntropy(x,H):
        return np.power(1-x,1-x)*np.power(x,x)-np.power(2,-H)
    
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
Hjt2[:,:,t]=-P01[:,:,t]*np.log2(np.maximum(P01[:,:,t],EPSILON2))-P00[:,:,t]*np.log2(np.maximum(P00[:,:,t],EPSILON2))  \
          -P11[:,:,t]*np.log2(np.maximum(P11[:,:,t],EPSILON2))-P10[:,:,t]*np.log2(np.maximum(P10[:,:,t],EPSILON2))
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
    HH2[k,:,t]=P11[k,:,t]*(np.log2(np.maximum(P11[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))  \
                +P10[k,:,t]*(np.log2(np.maximum(P10[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))  \
                +P01[k,:,t]*(np.log2(np.maximum(P01[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))  \
                +P00[k,:,t]*(np.log2(np.maximum(P00[k,:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))

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
###    P[:,t]=EndNode_sample/SampleSize
    P01[:,:,t]=P01[:,:,t]/SampleSize
    P11[:,:,t]=P11[:,:,t]/SampleSize
    P00[:,:,t]=P00[:,:,t]/SampleSize
    P10[:,:,t]=P10[:,:,t]/SampleSize
##### Entropy update ###    
    Hjt2[:,:,t]=-P01[:,:,t]*np.log2(np.maximum(P01[:,:,t],EPSILON2))-P00[:,:,t]*np.log2(np.maximum(P00[:,:,t],EPSILON2))  \
                -P11[:,:,t]*np.log2(np.maximum(P11[:,:,t],EPSILON2))-P10[:,:,t]*np.log2(np.maximum(P10[:,:,t],EPSILON2))
#    for k in range(0, NumNode): #for each StartNode
#        MU2[k,:,t]=P11[k,:,t]*(np.log2(np.maximum(P11[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON)))  \
#                +P10[k,:,t]*(np.log2(np.maximum(P10[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))  \
#                +P01[k,:,t]*(np.log2(np.maximum(P01[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))  \
#                +P00[k,:,t]*(np.log2(np.maximum(P00[k,:,t],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))
    for k in range(0,NumNode):  #for each EndNode
        MU2[:,k,t]=P11[:,k,t]*(np.log2(np.maximum(P11[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON)))  \
                +P10[:,k,t]*(np.log2(np.maximum(P10[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))  \
                +P01[:,k,t]*(np.log2(np.maximum(P01[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))  \
                +P00[:,k,t]*(np.log2(np.maximum(P00[:,k,t],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))
        HH2[:,k,t]=P11[:,k,t]*(np.log2(np.maximum(P11[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON)))  \
                +P10[:,k,t]*(np.log2(np.maximum(P10[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON)))  \
                +P01[:,k,t]*(np.log2(np.maximum(P01[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON)))  \
                +P00[:,k,t]*(np.log2(np.maximum(P00[:,k,t],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))
#        H_ave[k,t]=np.average(Hc[:,k,t])
#        H_ave[k,t]=np.amin(Hc[:,k,t])
#        H_ave[k,t]=np.amax(Hc[:,k,t])
        H_ave[k,t]=np.sum(MU2[:,k,t])
#        H_ave[k,t]=np.average(Hjt2[:,k,t]-HH2[:,k,t])
        P[k,t]=optimize.fsolve(funcInvEntropy,P[k,t-1],H_ave[k,t])
#########################        
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
#        MU2[k,:,t]=P11[k,:,t]*(np.log2(np.maximum(P11[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))  \
#                +P10[k,:,t]*(np.log2(np.maximum(P10[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
#                +P01[k,:,t]*(np.log2(np.maximum(P01[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
#                +P00[k,:,t]*(np.log2(np.maximum(P00[k,:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))


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
#    P[:,t]=EndNode_sample/SampleSize
    P01[:,:,t]=P01[:,:,t]/SampleSize
    P11[:,:,t]=P11[:,:,t]/SampleSize
    P00[:,:,t]=P00[:,:,t]/SampleSize
    P10[:,:,t]=P10[:,:,t]/SampleSize
##### Entropy update ###    
    Hjt2[:,:,t]=-P01[:,:,t]*np.log2(np.maximum(P01[:,:,t],EPSILON2))-P00[:,:,t]*np.log2(np.maximum(P00[:,:,t],EPSILON2))  \
                -P11[:,:,t]*np.log2(np.maximum(P11[:,:,t],EPSILON2))-P10[:,:,t]*np.log2(np.maximum(P10[:,:,t],EPSILON2))
    for k in range(0,NumNode):  #for each EndNode
        MU2[:,k,t]=P11[:,k,t]*(np.log2(np.maximum(P11[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON)))  \
                +P10[:,k,t]*(np.log2(np.maximum(P10[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))  \
                +P01[:,k,t]*(np.log2(np.maximum(P01[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))  \
                +P00[:,k,t]*(np.log2(np.maximum(P00[:,k,t],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))
        HH2[:,k,t]=P11[:,k,t]*(np.log2(np.maximum(P11[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON)))  \
                +P10[:,k,t]*(np.log2(np.maximum(P10[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON)))  \
                +P01[:,k,t]*(np.log2(np.maximum(P01[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON)))  \
                +P00[:,k,t]*(np.log2(np.maximum(P00[:,k,t],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))
#        H_ave[k,t]=np.average(Hc[:,k,t])
#        H_ave[k,t]=np.amin(Hc[:,k,t])
#        H_ave[k,t]=np.amax(Hc[:,k,t])
        H_ave[k,t]=np.sum(MU2[:,k,t])
#        H_ave[k,t]=np.average(Hjt2[:,k,t]-HH2[:,k,t])
        P[k,t]=optimize.fsolve(funcInvEntropy,P[k,t-1],H_ave[k,t])
#########################        
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
#        MU2[k,:,t]=P11[k,:,t]*(np.log2(np.maximum(P11[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))  \
#                +P10[k,:,t]*(np.log2(np.maximum(P10[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
#                +P01[k,:,t]*(np.log2(np.maximum(P01[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
#                +P00[k,:,t]*(np.log2(np.maximum(P00[k,:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))
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
#    P[:,t]=EndNode_sample/SampleSize
    P01[:,:,t]=P01[:,:,t]/SampleSize
    P11[:,:,t]=P11[:,:,t]/SampleSize
    P00[:,:,t]=P00[:,:,t]/SampleSize
    P10[:,:,t]=P10[:,:,t]/SampleSize
##### Entropy update ###    
    Hjt2[:,:,t]=-P01[:,:,t]*np.log2(np.maximum(P01[:,:,t],EPSILON2))-P00[:,:,t]*np.log2(np.maximum(P00[:,:,t],EPSILON2))  \
                -P11[:,:,t]*np.log2(np.maximum(P11[:,:,t],EPSILON2))-P10[:,:,t]*np.log2(np.maximum(P10[:,:,t],EPSILON2))
    for k in range(0,NumNode):  #for each EndNode
        MU2[:,k,t]=P11[:,k,t]*(np.log2(np.maximum(P11[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON)))  \
                +P10[:,k,t]*(np.log2(np.maximum(P10[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))  \
                +P01[:,k,t]*(np.log2(np.maximum(P01[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))  \
                +P00[:,k,t]*(np.log2(np.maximum(P00[:,k,t],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))
        HH2[:,k,t]=P11[:,k,t]*(np.log2(np.maximum(P11[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON)))  \
                +P10[:,k,t]*(np.log2(np.maximum(P10[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON)))  \
                +P01[:,k,t]*(np.log2(np.maximum(P01[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON)))  \
                +P00[:,k,t]*(np.log2(np.maximum(P00[:,k,t],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))
#        H_ave[k,t]=np.average(Hc[:,k,t])
#        H_ave[k,t]=np.amin(Hc[:,k,t])
#        H_ave[k,t]=np.amax(Hc[:,k,t])
        H_ave[k,t]=np.sum(MU2[:,k,t])
#        H_ave[k,t]=np.average(Hjt2[:,k,t]-HH2[:,k,t])
        P[k,t]=optimize.fsolve(funcInvEntropy,P[k,t-1],H_ave[k,t])
#########################        
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
#        MU2[k,:,t]=P11[k,:,t]*(np.log2(np.maximum(P11[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))  \
#                +P10[k,:,t]*(np.log2(np.maximum(P10[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
#                +P01[k,:,t]*(np.log2(np.maximum(P01[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
#                +P00[k,:,t]*(np.log2(np.maximum(P00[k,:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))

#pdb.set_trace()
MaxNumDisconnect=np.sum(1-(1-(PP==0.)))

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
#    P[:,t]=EndNode_sample/SampleSize
    #####simulated conditional probabilities
    P01[:,:,t]=P01[:,:,t]/SampleSize
    P11[:,:,t]=P11[:,:,t]/SampleSize
    P00[:,:,t]=P00[:,:,t]/SampleSize
    P10[:,:,t]=P10[:,:,t]/SampleSize
##### Entropy update ###    
    Hjt2[:,:,t]=-P01[:,:,t]*np.log2(np.maximum(P01[:,:,t],EPSILON2))-P00[:,:,t]*np.log2(np.maximum(P00[:,:,t],EPSILON2))  \
                -P11[:,:,t]*np.log2(np.maximum(P11[:,:,t],EPSILON2))-P10[:,:,t]*np.log2(np.maximum(P10[:,:,t],EPSILON2))
    for k in range(0,NumNode):  #for each EndNode
        MU2[:,k,t]=P11[:,k,t]*(np.log2(np.maximum(P11[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON)))  \
                +P10[:,k,t]*(np.log2(np.maximum(P10[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))  \
                +P01[:,k,t]*(np.log2(np.maximum(P01[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))  \
                +P00[:,k,t]*(np.log2(np.maximum(P00[:,k,t],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))
        HH2[:,k,t]=P11[:,k,t]*(np.log2(np.maximum(P11[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON)))  \
                +P10[:,k,t]*(np.log2(np.maximum(P10[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON)))  \
                +P01[:,k,t]*(np.log2(np.maximum(P01[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON)))  \
                +P00[:,k,t]*(np.log2(np.maximum(P00[:,k,t],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))
#        H_ave[k,t]=np.average(Hc[:,k,t])
#        H_ave[k,t]=np.amin(Hc[:,k,t])
#        H_ave[k,t]=np.amax(Hc[:,k,t])
        H_ave[k,t]=np.sum(MU2[:,k,t])
#        H_ave[k,t]=np.average(Hjt2[:,k,t]-HH2[:,k,t])
        P[k,t]=optimize.fsolve(funcInvEntropy,P[k,t-1],H_ave[k,t])
#########################        
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
#        MU2[k,:,t]=P11[k,:,t]*(np.log2(np.maximum(P11[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))  \
#                +P10[k,:,t]*(np.log2(np.maximum(P10[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
#                +P01[k,:,t]*(np.log2(np.maximum(P01[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
#                +P00[k,:,t]*(np.log2(np.maximum(P00[k,:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))
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
#    P[:,t]=EndNode_sample/SampleSize
    P01[:,:,t]=P01[:,:,t]/SampleSize
    P11[:,:,t]=P11[:,:,t]/SampleSize
    P00[:,:,t]=P00[:,:,t]/SampleSize
    P10[:,:,t]=P10[:,:,t]/SampleSize
##### Entropy update ###    
    Hjt2[:,:,t]=-P01[:,:,t]*np.log2(np.maximum(P01[:,:,t],EPSILON2))-P00[:,:,t]*np.log2(np.maximum(P00[:,:,t],EPSILON2))  \
                -P11[:,:,t]*np.log2(np.maximum(P11[:,:,t],EPSILON2))-P10[:,:,t]*np.log2(np.maximum(P10[:,:,t],EPSILON2))
    for k in range(0,NumNode):  #for each EndNode
        MU2[:,k,t]=P11[:,k,t]*(np.log2(np.maximum(P11[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON)))  \
                +P10[:,k,t]*(np.log2(np.maximum(P10[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))  \
                +P01[:,k,t]*(np.log2(np.maximum(P01[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))  \
                +P00[:,k,t]*(np.log2(np.maximum(P00[:,k,t],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))
        HH2[:,k,t]=P11[:,k,t]*(np.log2(np.maximum(P11[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON)))  \
                +P10[:,k,t]*(np.log2(np.maximum(P10[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON)))  \
                +P01[:,k,t]*(np.log2(np.maximum(P01[:,k,t],EPSILON))-np.log2(np.maximum(P[:,t-1],EPSILON)))  \
                +P00[:,k,t]*(np.log2(np.maximum(P00[:,k,t],EPSILON))-np.log2(np.maximum(1-P[:,t-1],EPSILON)))
#        H_ave[k,t]=np.average(Hc[:,k,t])
#        H_ave[k,t]=np.amin(Hc[:,k,t])
#        H_ave[k,t]=np.amax(Hc[:,k,t])
        H_ave[k,t]=np.sum(MU2[:,k,t])
#        H_ave[k,t]=np.average(Hjt2[:,k,t]-HH2[:,k,t])
        P[k,t]=optimize.fsolve(funcInvEntropy,P[k,t-1],H_ave[k,t])
#########################        
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
#        MU2[k,:,t]=P11[k,:,t]*(np.log2(np.maximum(P11[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON)))  \
#                +P10[k,:,t]*(np.log2(np.maximum(P10[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
#                +P01[k,:,t]*(np.log2(np.maximum(P01[k,:,t],EPSILON))-np.log2(np.maximum(P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))  \
#                +P00[k,:,t]*(np.log2(np.maximum(P00[k,:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON))-np.log2(np.maximum(1-P[:,t],EPSILON)))

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
plt.plot(time[0,:], np.average(np.average(Hjt2,axis=0),axis=0)/2, 'c-', label="ave. joint H")
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
plt.plot(time[0,:], np.average(np.average(MU2,axis=0),axis=0)/2, 'c-', label="MU2")
ax.set_title('Total number of nodes='+str(NumNode)+', a maximum of '+str(MaxNumDisconnect)+' disrupted edges')
#ax.set_ylim([0, 15])
ax.set_ylabel('$F=\sum(mutual information)/(2N^2)$')
ax.set_xlabel('time')
ax.xaxis.grid(False)
ax.yaxis.grid(True)
plt.grid()

plt.show()
