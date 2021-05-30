####################################################################
### ProbNet: A probabilistic graph modeling toolkit that enables the design of cyber-physical-social systems 
###	     with the considerations of trust, resilience, and information dynamics',
###   author: Yan Wang,
###   author_email: yan-wang@gatech.edu
####################################################################
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
The Input-Output Analysis with copula -- three nodes
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import matplotlib.pyplot as plt
#import pdb
from sklearn.linear_model import LinearRegression
from probGraph import *

######## Randomly generated graph ######
TotalNumNode = 3
#ConnectProb = 0.05
#ConnectProb = 0.1       # 3-node-1-edge
#ConnectProb = 0.2      # 3-node-2-edge
#ConnectProb = 0.28      # 3-node-3-edge
ConnectProb = 0.85     # 3-node-6-edge
##TotalNumNode = 50
##ConnectProb = 0.08
Seed = 21211

#Seed = 21211
Timesteps = 80          ##total number of steps in simulation
#Timesteps = 100          ##total number of steps in simulation
#Trainingsteps = 60      ##the number of straining steps
Trainingsteps = 50      ##the number of straining steps

lag_max = 2             ## the lag in AR model. E.g.  x[t]=a1*x[t-1]+a2*x[t-2]+sigma where lag = 2
#lag_max = 1             ## the lag in AR model. E.g.  x[t]=a1*x[t-1]+a2*x[t-2]+sigma where lag = 2


#SelfIncluded = False #to include self prediction or not in updating prediction probability

#the smalleast probability to avoid log(0) in calculating entropy
EPSILON=np.ones((TotalNumNode,TotalNumNode))*1e-10

time = np.zeros((1,Timesteps))

### initialization
P = np.zeros((TotalNumNode,Timesteps))   # marginal P-probabilities from simulation
P_cop = np.zeros((TotalNumNode,Timesteps)) # marginal P-probabilities predicted from copulas

#P_L = np.zeros((TotalNumNode,Timesteps))
#P_U = np.zeros((TotalNumNode,Timesteps))
NumJoint = np.power(2,TotalNumNode)     # number of T/F value combinations for joint probabilities
NumExtrem = 5                           # number of subset combinations for correlation cases as extremal distributions
Pjt = np.zeros((NumJoint,NumExtrem,Timesteps))    # extremal probabilities for all combintations of binary T's and F's
#Pjt_L = np.ones((NumJoint,Timesteps))  # lower bounds of joint probabilities estimated from copulas
#Pjt_U = np.zeros((NumJoint,Timesteps))   # upper bounds of joint probabilities estimated from copulas
Pjt_predicted = np.zeros((NumJoint,NumExtrem,Timesteps))    # predicted extremal probabilities from AR models

param = np.zeros((NumJoint,NumExtrem))    # joint probabilities for all combintations of binary T's and F's

Pjoint = np.zeros((NumJoint,Timesteps))   # joint probabilities from simulation
Pjoint_cop = np.zeros((NumJoint,Timesteps))  # copulas predicted from extremal probabilities

#node entropy
###H = np.zeros((NumNode,Timesteps))

#reliancce probabilities
#PP = np.zeros((TotalNumNode,TotalNumNode))
#PP = np.random.random_sample((NumNode,NumNode))

#pdb.set_trace()

#reliance probabilities
#QQ = np.zeros((TotalNumNode,TotalNumNode))
#QQ = np.random.random_sample((NumNode,NumNode))



"""
### initial values when t=0
#P[:,0]=np.ones(NumNode)*0.5
P[:,0]=np.random.random_sample(NumNode)
P_o[:,0]=P[:,0]
Pjt[:,0]=np.zeros(NumJoint)
Pjt_o[:,0]=np.zeros(NumJoint)
#Q[:,0]=1-P[:,0]
"""


g=RandomProbGraph(TotalNumNode, ConnectProb, Seed)
##for i in g.node():
for i in g.nodes:
        P[i,0]=g.get_node_data(i)['Prob']
#    Psim[i,0]=g.get_node_data(i)['Prob']
##    for j in g.successors(i):
##        PP[i,j] = g.get_edge_data(i,j)['P_Prob']
##        QQ[i,j] = g.get_edge_data(i,j)['Q_Prob']

print('Init Prediction Prob.='+str(P[:,0]))
print('P-reliance prob.=')
print(g.get_reliance_prob('P_Prob')[0,:,:])
print('Q-reliance prob.=')
print(g.get_reliance_prob('Q_Prob')[0,:,:])


#w_p123=1
#w_p12n3=0
#w_p1n23=0
#w_p13n2=0
#w_indep = 1-w_p123-w_p12n3-w_p1n23-w_p13n2
#pdb.set_trace()
### simulation main loop
for t in range(1,Timesteps):
    time[0,t]=t
    ### Monte Carlo simulation to simulate information sharing and update prediction probabilities
    Pjoint[:,t]=g.updatePredictionWorstcaseSampling_joint(300, SelfInclusion=False)
###    for i in g.node():
    for i in g.nodes:
        P[i,t]=g.get_node_data(i)['Prob']       ### retrieve the updated prediction prbabilities
    """w_p123=Pjoint[0,t]+Pjoint[7,t]
    w_p12n3=Pjoint[1,t]+Pjoint[6,t]
    w_p1n23=Pjoint[3,t]+Pjoint[4,t]
    w_p13n2=Pjoint[2,t]+Pjoint[5,t]
    w_indep=1-w_p123-w_p12n3-w_p1n23-w_p13n2"""

for t in range(1,Timesteps):
    #### x1=T, x2=T, x3=T ####
    Pjt[0,0,t] = min(P[0,t],P[1,t],P[2,t])
#    Pjt[0,t]+=w_p123*tmp
    Pjt[0,1,t] = np.maximum(0,np.minimum(P[0,t],P[1,t])+P[2,t]-1)
#    Pjt[0,t]+=w_p12n3*tmp
    Pjt[0,2,t] = np.maximum(0,np.minimum(P[1,t],P[2,t])+P[0,t]-1)
#    Pjt[0,t]+=w_p1n23*tmp
    Pjt[0,3,t] = np.maximum(0,np.minimum(P[0,t],P[2,t])+P[1,t]-1)
#    Pjt[0,t]+=w_p13n2*tmp
    Pjt[0,4,t] = P[0,t]*P[1,t]*P[2,t]
#    Pjt[0,t]+=w_indep*tmp
    #### x1=T, x2=T, x3=F  ####
    Pjt[1,0,t] = min(P[0,t],P[1,t],1-P[2,t])
#    Pjt[1,t]+=w_p123*tmp
    Pjt[1,1,t] = np.maximum(0,np.minimum(P[0,t],P[1,t])+1-P[2,t]-1)
#    Pjt[1,t]+=w_p12n3*tmp
    Pjt[1,2,t] = np.maximum(0,np.minimum(P[1,t],1-P[2,t])+P[0,t]-1)
#    Pjt[1,t]+=w_p1n23*tmp
    Pjt[1,3,t] = np.maximum(0,np.minimum(P[0,t],1-P[2,t])+P[1,t]-1)
#    Pjt[1,t]+=w_p13n2*tmp
    Pjt[1,4,t] = P[0,t]*P[1,t]*(1-P[2,t])
#    Pjt[1,t]+=w_indep*tmp
    #### x1=T, x2=F, x3=T  ####
    Pjt[2,0,t] = min(P[0,t],1-P[1,t],P[2,t])
#    Pjt[2,t]+=w_p123*tmp
    Pjt[2,1,t] = np.maximum(0,np.minimum(P[0,t],1-P[1,t])+P[2,t]-1)
#    Pjt[2,t]+=w_p12n3*tmp
    Pjt[2,2,t] = np.maximum(0,np.minimum(1-P[1,t],P[2,t])+P[0,t]-1)
#    Pjt[2,t]+=w_p1n23*tmp
    Pjt[2,3,t] = np.maximum(0,np.minimum(P[0,t],P[2,t])+1-P[1,t]-1)
#    Pjt[2,t]+=w_p13n2*tmp
    Pjt[2,4,t] = P[0,t]*(1-P[1,t])*P[2,t]
#    Pjt[2,t]+=w_indep*tmp
    #### x1=T, x2=F, x3=F  ####
    Pjt[3,0,t] = min(P[0,t],1-P[1,t],1-P[2,t])
#    Pjt[3,t]+=w_p123*tmp
    Pjt[3,1,t] = np.maximum(0,np.minimum(P[0,t],1-P[1,t])+1-P[2,t]-1)
#    Pjt[3,t]+=w_p12n3*tmp
    Pjt[3,2,t] = np.maximum(0,np.minimum(1-P[1,t],1-P[2,t])+P[0,t]-1)
#    Pjt[3,t]+=w_p1n23*tmp
    Pjt[3,3,t] = np.maximum(0,np.minimum(P[0,t],1-P[2,t])+1-P[1,t]-1)
#    Pjt[3,t]+=w_p13n2*tmp
    Pjt[3,4,t] = P[0,t]*(1-P[1,t])*(1-P[2,t])
#    Pjt[3,t]+=w_indep*tmp
    #### x1=F, x2=T, x3=T  ####
    Pjt[4,0,t] = min(1-P[0,t],P[1,t],P[2,t])
#    Pjt[4,t]+=w_p123*tmp
    Pjt[4,1,t] = np.maximum(0,np.minimum(1-P[0,t],P[1,t])+P[2,t]-1)
#    Pjt[4,t]+=w_p12n3*tmp
    Pjt[4,2,t] = np.maximum(0,np.minimum(P[1,t],P[2,t])+1-P[0,t]-1)
#    Pjt[4,t]+=w_p1n23*tmp
    Pjt[4,3,t] = np.maximum(0,np.minimum(1-P[0,t],P[2,t])+P[1,t]-1)
#    Pjt[4,t]+=w_p13n2*tmp
    Pjt[4,4,t] = (1-P[0,t])*P[1,t]*P[2,t]
#    Pjt[4,t]+=w_indep*tmp
    #### x1=F, x2=T, x3=F  ####
    Pjt[5,0,t] = min(1-P[0,t],P[1,t],1-P[2,t])
#    Pjt[5,t]+=w_p123*tmp
    Pjt[5,1,t] = np.maximum(0,np.minimum(1-P[0,t],P[1,t])+1-P[2,t]-1)
#    Pjt[5,t]+=w_p12n3*tmp
    Pjt[5,2,t] = np.maximum(0,np.minimum(P[1,t],1-P[2,t])+1-P[0,t]-1)
#    Pjt[5,t]+=w_p1n23*tmp
    Pjt[5,3,t] = np.maximum(0,np.minimum(1-P[0,t],1-P[2,t])+P[1,t]-1)
#    Pjt[5,t]+=w_p13n2*tmp
    Pjt[5,4,t] = (1-P[0,t])*P[1,t]*(1-P[2,t])
#    Pjt[5,t]+=w_indep*tmp
    #### x1=F, x2=F, x3=T  ####
    Pjt[6,0,t] = min(1-P[0,t],1-P[1,t],P[2,t])
#    Pjt[6,t]+=w_p123*tmp
    Pjt[6,1,t] = np.maximum(0,np.minimum(1-P[0,t],1-P[1,t])+P[2,t]-1)
#    Pjt[6,t]+=w_p12n3*tmp
    Pjt[6,2,t] = np.maximum(0,np.minimum(1-P[1,t],P[2,t])+1-P[0,t]-1)
#    Pjt[6,t]+=w_p1n23*tmp
    Pjt[6,3,t] = np.maximum(0,np.minimum(1-P[0,t],P[2,t])+1-P[1,t]-1)
#    Pjt[6,t]+=w_p13n2*tmp
    Pjt[6,4,t] = (1-P[0,t])*(1-P[1,t])*P[2,t]
#    Pjt[6,t]+=w_indep*tmp
    #### x1=F, x2=F, x3=F  #####
    Pjt[7,0,t] = min(1-P[0,t],1-P[1,t],1-P[2,t])
#    Pjt[7,t]+=w_p123*tmp
    Pjt[7,1,t] = np.maximum(0,np.minimum(1-P[0,t],1-P[1,t])+1-P[2,t]-1)
#    Pjt[7,t]+=w_p12n3*tmp
    Pjt[7,2,t] = np.maximum(0,np.minimum(1-P[1,t],1-P[2,t])+1-P[0,t]-1)
#    Pjt[7,t]+=w_p1n23*tmp
    Pjt[7,3,t] = np.maximum(0,np.minimum(1-P[0,t],1-P[2,t])+1-P[1,t]-1)
#    Pjt[7,t]+=w_p13n2*tmp
    Pjt[7,4,t] = (1-P[0,t])*(1-P[1,t])*(1-P[2,t])
#    Pjt[7,t]+=w_indep*tmp
#    sum_joint = Pjt[0,t]+Pjt[1,t]+Pjt[2,t]+Pjt[3,t]+Pjt[4,t]+Pjt[5,t]+Pjt[6,t]+Pjt[7,t]
##    P[0,t]=(Pjt[0,t]+Pjt[1,t]+Pjt[2,t]+Pjt[3,t])/sum_joint  #marginal Pr(x0=T)
#    P[0,t]=(Pjt[0,t]+Pjt[1,t]+Pjt[2,t]+Pjt[3,t])  #marginal Pr(x0=T)
##    P[1,t]=(Pjt[0,t]+Pjt[1,t]+Pjt[4,t]+Pjt[5,t])/sum_joint	#marginal Pr(x1=T)
#    P[1,t]=(Pjt[0,t]+Pjt[1,t]+Pjt[4,t]+Pjt[5,t])	#marginal Pr(x1=T)
##    P[2,t]=(Pjt[0,t]+Pjt[2,t]+Pjt[4,t]+Pjt[6,t])/sum_joint	#marginal Pr(x2=T)
#    P[2,t]=(Pjt[0,t]+Pjt[2,t]+Pjt[4,t]+Pjt[6,t])	#marginal Pr(x2=T)
    #    pdb.set_trace()


###### Train the correlation parameters  ######
### The joint probability is the convex linear combination of extramal probabilities
### Pjoint[k,t] = param[k,0]*Pjt[k,0,t]+param[k,1]*Pjt[k,1,t]+param[k,2]*Pjt[k,2,t]+param[k,3]*Pjt[k,3,t]+param[k,4]*Pjt[k,4,t]
X=Pjt[0,:,0:Trainingsteps].T
Y=Pjoint[0,0:Trainingsteps]
for k in  range(1,NumJoint):
    X=np.vstack((X, Pjt[k,:,0:Trainingsteps].T))
    Y=np.concatenate((Y, Pjoint[k,0:Trainingsteps]))
##### without constraints
##reg=LinearRegression(fit_intercept=False).fit(X,Y)
##param[0,:]=reg.coef_
##### constrained least-square error fitting with constraint: c1+c2+...+cn=1
from scipy.optimize import minimize
def loss(c):
    return np.sum(np.square((np.dot(X, c) - Y)))
cons = ({'type': 'eq',
         'fun' : lambda c: np.sum(c) - 1.0})
c0 = np.zeros(X.shape[1])
res = minimize(loss, c0, method='SLSQP', constraints=cons,
               bounds=[(0.0, 1.0) for i in range(X.shape[1])], options={'disp': True})
param[0,:]=res.x
print("weight coefs to calculate joint probabilities from 5 extremal probabilities: ")
print(res.x)

###### Train the extremal probabilities with autoregression  ######
#### AR model ####
from statsmodels.tsa.api import AR
coef = np.empty((NumJoint,NumExtrem,lag_max+1))
sigma2 = np.empty((NumJoint,NumExtrem))
stdev = np.empty(TotalNumNode)
lag = np.empty((NumJoint,NumExtrem))
for i in range(0,NumJoint):
    for j in range(0,NumExtrem):
        ar_model = AR(Pjt[i,j,0:Trainingsteps])
        ar_results=ar_model.fit(maxlag=lag_max)
        coef[i,j,:]=ar_results.params
        sigma2[i,j]=ar_results.sigma2
        lag[i,j]=ar_results.k_ar

print('Coef. of AR model: ')
print(coef)
print('Sigma of AR model: ')
print(np.sqrt(sigma2))

### use AR model to predict extremal probabilities
for t in range(0, Trainingsteps):
    for i in range(0,NumJoint):
        for j in range(0,NumExtrem):
            Pjt_predicted[i,j,t]=Pjt[i,j,t]
for t in range(Trainingsteps, Timesteps):
    for i in range(0,NumJoint):
        for j in range(0,NumExtrem):
            Pjt_predicted[i,j,t]=coef[i,j,0]
            for q in range(0,int(lag[i,j])):
                Pjt_predicted[i,j,t]+=Pjt_predicted[i,j,t-1-q]*coef[i,j,q+1]

### use fitted parameters to predict the joint probabilities from extremal probabilities
for k in  range(0,NumJoint):
#    Pjoint_cop[k,Trainingsteps:Timesteps] = np.inner(Pjt[k,:,Trainingsteps:Timesteps].T, param[0,:])
    Pjoint_cop[k,Trainingsteps:Timesteps] = np.inner(Pjt_predicted[k,:,Trainingsteps:Timesteps].T, param[0,:])

#marginal Pr(x0=T)
P_cop[0,Trainingsteps:Timesteps]=(Pjoint_cop[0,Trainingsteps:Timesteps]+ \
                                  Pjoint_cop[1,Trainingsteps:Timesteps]+ \
                                  Pjoint_cop[2,Trainingsteps:Timesteps]+ \
                                  Pjoint_cop[3,Trainingsteps:Timesteps])
"""stdev[0]=(sigma2[0,0]+sigma2[0,1]+sigma2[0,2]+sigma2[0,3]+sigma2[0,4]+ \
          sigma2[1,0]+sigma2[1,1]+sigma2[1,2]+sigma2[1,3]+sigma2[1,4]+ \
          sigma2[2,0]+sigma2[2,1]+sigma2[2,2]+sigma2[2,3]+sigma2[2,4]+ \
          sigma2[3,0]+sigma2[3,1]+sigma2[3,2]+sigma2[3,3]+sigma2[3,4] )**0.5"""
stdev[0]=( (sigma2[0,0]+sigma2[0,1]+sigma2[0,2]+sigma2[0,3]+sigma2[0,4]+ \
          sigma2[1,0]+sigma2[1,1]+sigma2[1,2]+sigma2[1,3]+sigma2[1,4]+ \
          sigma2[2,0]+sigma2[2,1]+sigma2[2,2]+sigma2[2,3]+sigma2[2,4]+ \
          sigma2[3,0]+sigma2[3,1]+sigma2[3,2]+sigma2[3,3]+sigma2[3,4] )/5 )**0.5
#marginal Pr(x1=T)
P_cop[1,Trainingsteps:Timesteps]=(Pjoint_cop[0,Trainingsteps:Timesteps]+ \
                                  Pjoint_cop[1,Trainingsteps:Timesteps]+ \
                                  Pjoint_cop[4,Trainingsteps:Timesteps]+ \
                                  Pjoint_cop[5,Trainingsteps:Timesteps])
"""stdev[1]=(sigma2[0,0]+sigma2[0,1]+sigma2[0,2]+sigma2[0,3]+sigma2[0,4]+ \
          sigma2[1,0]+sigma2[1,1]+sigma2[1,2]+sigma2[1,3]+sigma2[1,4]+ \
          sigma2[4,0]+sigma2[4,1]+sigma2[4,2]+sigma2[4,3]+sigma2[4,4]+ \
          sigma2[5,0]+sigma2[5,1]+sigma2[5,2]+sigma2[5,3]+sigma2[5,4] )**0.5"""
stdev[1]=( (sigma2[0,0]+sigma2[0,1]+sigma2[0,2]+sigma2[0,3]+sigma2[0,4]+ \
          sigma2[1,0]+sigma2[1,1]+sigma2[1,2]+sigma2[1,3]+sigma2[1,4]+ \
          sigma2[4,0]+sigma2[4,1]+sigma2[4,2]+sigma2[4,3]+sigma2[4,4]+ \
          sigma2[5,0]+sigma2[5,1]+sigma2[5,2]+sigma2[5,3]+sigma2[5,4] )/5 )**0.5
#marginal Pr(x2=T)
P_cop[2,Trainingsteps:Timesteps]=(Pjoint_cop[0,Trainingsteps:Timesteps]+ \
                                  Pjoint_cop[2,Trainingsteps:Timesteps]+ \
                                  Pjoint_cop[4,Trainingsteps:Timesteps]+ \
                                  Pjoint_cop[6,Trainingsteps:Timesteps])
"""stdev[2]=(sigma2[0,0]+sigma2[0,1]+sigma2[0,2]+sigma2[0,3]+sigma2[0,4]+ \
          sigma2[2,0]+sigma2[2,1]+sigma2[2,2]+sigma2[2,3]+sigma2[2,4]+ \
          sigma2[4,0]+sigma2[4,1]+sigma2[4,2]+sigma2[4,3]+sigma2[4,4]+ \
          sigma2[6,0]+sigma2[6,1]+sigma2[6,2]+sigma2[6,3]+sigma2[6,4] )**0.5"""
stdev[2]=( (sigma2[0,0]+sigma2[0,1]+sigma2[0,2]+sigma2[0,3]+sigma2[0,4]+ \
          sigma2[2,0]+sigma2[2,1]+sigma2[2,2]+sigma2[2,3]+sigma2[2,4]+ \
          sigma2[4,0]+sigma2[4,1]+sigma2[4,2]+sigma2[4,3]+sigma2[4,4]+ \
          sigma2[6,0]+sigma2[6,1]+sigma2[6,2]+sigma2[6,3]+sigma2[6,4] )/5 )**0.5


### plot the graph
import networkx as nx
pos=nx.spring_layout(g)
plt.figure(figsize=(3,3))
nx.draw(g,pos,node_color='gold',with_labels=True)
plt.show()


### plot the prediction probabilities
fig, axes = plt.subplots(nrows=TotalNumNode, ncols=1, figsize=(6,4))
#fig.suptitle("Info Dynamics: Copula AR (Worst-Case Fusion) # of nodes=" + str(TotalNumNode))
j = 0
xticks = np.arange(0, Timesteps, 20)
for ax in axes.flat:
    ax.set_xlim([0, time[0,Timesteps-1]])
    ax.set_ylim([0, 1.05])
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(['','','',''])
    ax.plot(time[0,:], P[j,:], 'b-', label="node "+str(j))
#    ax.plot(time[0,Trainingsteps:Timesteps], P_cop[j,Trainingsteps:Timesteps], 'r--', label="forecast")
    ax.plot(time[0,Trainingsteps:Timesteps], P_cop[j,Trainingsteps:Timesteps], 'r--')
#    ax.plot(time[0,Trainingsteps:Timesteps], P_cop[j,Trainingsteps:Timesteps]+2*stdev[j], 'r-.', label="$\pm 2\sigma$")
#    ax.plot(time[0,Trainingsteps:Timesteps], P_cop[j,Trainingsteps:Timesteps]-2*stdev[j], 'r-.')
    ax.fill_between(time[0,Trainingsteps:Timesteps], P_cop[j,Trainingsteps:Timesteps]-stdev[j], \
                     P_cop[j,Trainingsteps:Timesteps]+stdev[j], color='gold')
    ax.legend(loc='upper left',ncol=3)
    j+=1
ax.set_xticklabels(['0','20','40','60'])
plt.xlabel('time')
plt.show()


#plt.plot(time[0,:], np.amin(P,axis=0), 'c.-', label="min pred. prob.")
#plt.plot(time[0,:], np.amax(P,axis=0), 'm-', label="max pred. prob.")
#plt.plot(time[0,:], np.mean(P,axis=0), 'r-', label="average pred. prob.")
plt.plot(time[0,:], P[0,:], 'b-', label="node 0 (simulated)")
#plt.plot(time[0,:], P_L[0,:], 'b-.', label="node 0 - L")
#plt.plot(time[0,:], P_U[0,:], 'b-*', label="node 0 - U")
plt.plot(time[0,:], P[1,:], 'g-', label="node 1 (simulated)")
#plt.plot(time[0,:], P_L[1,:], 'c-.', label="node 1 - L")
#plt.plot(time[0,:], P_U[1,:], 'c-*', label="node 1 - U")
plt.plot(time[0,:], P[2,:], 'r-', label="node 2 (simulated)")

plt.plot(time[0,Trainingsteps:Timesteps], P_cop[0,Trainingsteps:Timesteps], 'b--', label="x0=T - predicted")
plt.plot(time[0,Trainingsteps:Timesteps], P_cop[0,Trainingsteps:Timesteps]+stdev[0], 'c-.')
plt.plot(time[0,Trainingsteps:Timesteps], P_cop[0,Trainingsteps:Timesteps]-stdev[0], 'c-.')
plt.plot(time[0,Trainingsteps:Timesteps], P_cop[1,Trainingsteps:Timesteps], 'g--', label="x1=T - predicted")
plt.plot(time[0,Trainingsteps:Timesteps], P_cop[1,Trainingsteps:Timesteps]+stdev[1], 'c-.')
plt.plot(time[0,Trainingsteps:Timesteps], P_cop[1,Trainingsteps:Timesteps]-stdev[1], 'c-.')
plt.plot(time[0,Trainingsteps:Timesteps], P_cop[2,Trainingsteps:Timesteps], 'r--', label="x2=T - predicted")
plt.plot(time[0,Trainingsteps:Timesteps], P_cop[2,Trainingsteps:Timesteps]+stdev[2], 'c-.')
plt.plot(time[0,Trainingsteps:Timesteps], P_cop[2,Trainingsteps:Timesteps]-stdev[2], 'c-.')

"""plt.plot(time[0,:], Pjt[7,0,:], 'c*', label="x0=T,x1=T,x2=T - p123")
plt.plot(time[0,:], Pjt[7,1,:], 'c.', label="x0=T,x1=T,x2=T - p12n3")
plt.plot(time[0,:], Pjt[7,2,:], 'c<', label="x0=T,x1=T,x2=T - p13n2")
plt.plot(time[0,:], Pjt[7,3,:], 'c>', label="x0=T,x1=T,x2=T - p1n23")
plt.plot(time[0,:], Pjt[7,4,:], 'cd', label="x0=T,x1=T,x2=T - indep")"""
#plt.plot(time[0,:], Pjoint[7,:], 'r.', label="x0=T,x1=T,x2=T")
"""plt.plot(time[0,:], Pjt[0,0,:]+Pjt[2,0,:]+Pjt[4,0,:]+Pjt[6,0,:], 'c*', label="x2=T - p123")
plt.plot(time[0,:], Pjt[0,1,:]+Pjt[2,1,:]+Pjt[4,1,:]+Pjt[6,1,:], 'c.', label="x2=T - p12n3")
plt.plot(time[0,:], Pjt[0,2,:]+Pjt[2,2,:]+Pjt[4,2,:]+Pjt[6,2,:], 'c<', label="x2=T - p13n2")
plt.plot(time[0,:], Pjt[0,3,:]+Pjt[2,3,:]+Pjt[4,3,:]+Pjt[6,3,:], 'c>', label="x2=T - p1n23")
plt.plot(time[0,:], Pjt[0,4,:]+Pjt[2,4,:]+Pjt[4,4,:]+Pjt[6,4,:], 'cd', label="x2=T - indep")"""
"""plt.plot(time[0,:], Pjt[0,0,:]+Pjt[2,0,:]+Pjt[4,0,:]+Pjt[5,0,:], 'c*', label="x1=T - p123")
plt.plot(time[0,:], Pjt[0,1,:]+Pjt[2,1,:]+Pjt[4,1,:]+Pjt[5,1,:], 'c.', label="x1=T - p12n3")
plt.plot(time[0,:], Pjt[0,2,:]+Pjt[2,2,:]+Pjt[4,2,:]+Pjt[5,2,:], 'c<', label="x1=T - p13n2")
plt.plot(time[0,:], Pjt[0,3,:]+Pjt[2,3,:]+Pjt[4,3,:]+Pjt[5,3,:], 'c>', label="x1=T - p1n23")
plt.plot(time[0,:], Pjt[0,4,:]+Pjt[2,4,:]+Pjt[4,4,:]+Pjt[5,4,:], 'cd', label="x1=T - indep")"""
"""plt.plot(time[0,:], Pjt[0,0,:]+Pjt[1,0,:]+Pjt[2,0,:]+Pjt[3,0,:], 'c*', label="x0=T - p123")
plt.plot(time[0,:], Pjt[0,1,:]+Pjt[1,1,:]+Pjt[2,1,:]+Pjt[3,1,:], 'c.', label="x0=T - p12n3")
plt.plot(time[0,:], Pjt[0,2,:]+Pjt[1,2,:]+Pjt[2,2,:]+Pjt[3,2,:], 'c<', label="x0=T - p13n2")
plt.plot(time[0,:], Pjt[0,3,:]+Pjt[1,3,:]+Pjt[2,3,:]+Pjt[3,3,:], 'c>', label="x0=T - p1n23")
plt.plot(time[0,:], Pjt[0,4,:]+Pjt[1,4,:]+Pjt[2,4,:]+Pjt[3,4,:], 'cd', label="x0=T - indep")"""
"""plt.plot(time[0,:], Psim[0,:], 'b.', label="node 0 (simulated)")
plt.plot(time[0,:], Psim[1,:], 'c.', label="node 1 (simulated)")
plt.plot(time[0,:], Psim[2,:], 'r.', label="node 2 (simulated)")"""
##for k in range(0,TotalNumNode):
##    plt.plot(time[0,:], P[k,:], label="node "+str(k)+" (pessimistic)")
##    #plt.plot(time[0,:], P_o[k,:], '.', label="node "+str(k)+" (optimistic)")
#plt.plot(time[0,:], Pjt[0,:]+Pjt[1,:]+Pjt[2,:]+Pjt[3,:]+Pjt[4,:]+Pjt[5,:]+Pjt[6,:]+Pjt[7,:], label="sum of joint prob (pessimistic)")
##plt.plot(time[0,:], Pjt_o[0,:]+Pjt_o[1,:]+Pjt_o[2,:]+Pjt_o[3,:], '.-', label="sum of joint prob (optimistic)")
#ax.set_title('Input-Output Analysis (copula): # of nodes='+str(TotalNumNode))
plt.ylabel('Prediction Probability')
#plt.ylim([0.0, 1.05])
plt.xlabel('time')
ax.xaxis.grid(False)
ax.yaxis.grid(True)
plt.grid()
plt.legend(loc='upper right')

"""
fig, ax = plt.subplots()
##plt.plot(time[0,:], np.average(np.average(HH,axis=0),axis=0), 'g--', label="ave. cond. entropy")
plt.plot(time[0,:], np.average(H,axis=0), 'b-', label="ave. entropy")
##plt.plot(time[0,:], np.average(H2,axis=0), 'r*-', label="ave. entropy2")
ax.set_title('Total number of nodes='+str(NumNode))
#ax.set_ylim([0, 1])
ax.set_ylabel('entropy')
ax.set_xlabel('time')
ax.xaxis.grid(False)
ax.yaxis.grid(True)
plt.grid()
plt.legend(loc='upper right')
"""

plt.show()
