####################################################################
### ProbNet: A probabilistic graph modeling toolkit that enables the design of cyber-physical-social systems 
###	     with the considerations of trust, resilience, and information dynamics',
###   author: Yan Wang,
###   author_email: yan-wang@gatech.edu
####################################################################
import numpy as np
import math
import random
import networkx as nx
from networkx import DiGraph
import pdb

class ProbGraph(DiGraph):
################################################################################
### The constructor of the probabilistic graph ###
###   Input parameters:
###     node_list = [(NodeIndex, Mean_PredictionProbability, Variance_PredictionProbability),
###                   ... ]   
###     edge_list = [(StartNodeIndex, EndNodeIndex, Mean_P-RelianceProbability, Mean_Q-RelianceProbability,
###                                   Variance_P-RelianceProbability, Variance_Q-RelianceProbability),
###                   ... ]   
    def __init__(self,node_list=[],edge_list=[]):
        DiGraph.__init__(self)
        try:
#            pdb.set_trace()
#            if len(node_list)!=self.number_of_nodes():
#                raise Exception("The numbers of nodes and node weights do not match")
            for k in range(len(node_list)):
                self.add_node(node_list[k][0],Prob=node_list[k][1],Prob_Var=node_list[k][2])
            for i in range(len(edge_list)):
                self.add_edge(edge_list[i][0],edge_list[i][1],P_Prob=edge_list[i][2],Q_Prob=edge_list[i][3],P_Prob_Var=edge_list[i][4],Q_Prob_Var=edge_list[i][5])
            self._nodeName = {}
            self._nodeIndex = {}
            for i, n in enumerate(node_list):
                self._nodeIndex[n]=i
                self._nodeName[i]=n
        except ValueError as err:
            print(err.args)

    ############################################################
    ### Return the mean "p[i]"and variance "var[i]" of prediction
    ### probability of node i
    def get_node_data(self, i):
        return self.nodes(data=True)[i]

    ############################################################
    ### Return the node name from index
    def nodeIndex2Name(self, i):
        return self._nodeName[i]

    ############################################################
    ### Return the node index from name
    def nodeName2Index(self, n):
        return self._nodeIndex[n]

    ############################################################
    ### Return the expected value "p[i]"and variance "var[i]"
    ### of prediction probability for all nodes as an N x 2 array
    def get_prediction_prob(self):
        TotalNumNode=self.number_of_nodes()
        P=np.zeros((TotalNumNode,2))
        for i in range(0,TotalNumNode):
            P[i][0]=self.nodes(data=True)[i]['Prob']
            P[i][1]=self.nodes(data=True)[i]['Prob_Var']
        return P

    ############################################################
    ### Return the expected value and variance of P- or Q-reliance probabilities
    ### for all edges as a 2 x N x N  array
    def get_reliance_prob(self, prob_type='P_Prob'):
        TotalNumNode=self.number_of_nodes()
        P=np.zeros((2,TotalNumNode,TotalNumNode))
        for i in self.nodes():
            for j in self.successors(i):
                P[0][i][j]=self.get_edge_data(i,j)[prob_type]
                P[1][i][j]=self.get_edge_data(i,j)[prob_type+'_Var']
        return P


    ############################################################
    ### Simply set the values of the expected value "p[i]"and variance "var[i]"
    ### of prediction probability for node i 
    def set_prediction_prob(self, i, new_mean, new_var):
        self.nodes(data=True)[i]['Prob']=new_mean
        self.nodes(data=True)[i]['Prob_Var']=new_var

    def set_Preliance_prob(self, i, j, new_Pmean, new_Pvar):
        self.edges(data=True)[i][j]['P_Prob']=new_Pmean
        self.edges(data=True)[i][j]['P_Prob_Var']=new_Pvar

    def set_Qreliance_prob(self, i, j, new_Qmean, new_Qvar):
        self.edges(data=True)[i][j]['Q_Prob']=new_Qmean
        self.edges(data=True)[i][j]['Q_Prob_Var']=new_Qvar


    ############################################################
    ### Bayesian update of the expected value "p{i]"and variance "var[i]" of prediction probability
    ### for node i based on new observed mean "obs_mean" and variance "obs_var"
    ###     p0[i]=(p[i]/var[i] + obs_mean/obs_var)/(1/var[i] + obs_mean/obs_var)
    ###     var0[i]=abs(p0[i]-p[i])
    def update_perceived_prediction_prob(self, i, obs_mean, obs_var):
        VAR_MIN = 1e-10
        p=self.get_node_data(i)['Prob']
        var=np.maximum(self.get_node_data(i)['Prob_Var'],VAR_MIN)
        obs_var=np.maximum(obs_var,VAR_MIN)
#        diff_mean=(obs_mean-p)*(obs_mean-p)
#        var = np.maximum(diff_mean, var)
#        obs_var = np.maximum(diff_mean, obs_var)
        var0 = 1./(1./var+1./obs_var)
        p0=(p/var+obs_mean/obs_var)*var0
        self.set_prediction_prob(i, p0, var0)

    def update_perceived_Preliance_prob(self, i, j, obs_mean, obs_var):
        VAR_MIN = 1e-10
        p=self.get_edge_data(i,j)['P_Prob']
        var=np.maximum(self.get_edge_data(i,j)['P_Prob_Var'],VAR_MIN)
        obs_var=np.maximum(obs_var,VAR_MIN)
##        diff_mean=(obs_mean-p)*(obs_mean-p)
##        var = np.maximum(diff_mean, var)
##        obs_var = np.maximum(diff_mean, obs_var)
        var0 = 1./(1./var+1./obs_var)
        p0=(p/var+obs_mean/obs_var)*var0
        self.set_Preliance_prob(i, j, p0, var0)


    def update_perceived_Qreliance_prob(self, i, j, obs_mean, obs_var):
        VAR_MIN = 1e-10
        p=self.get_edge_data(i,j)['Q_Prob']
        var=np.maximum(self.get_edge_data(i,j)['Q_Prob_Var'],VAR_MIN)
        obs_var=np.maximum(obs_var,VAR_MIN)
##        diff_mean=(obs_mean-p)*(obs_mean-p)
##        var = np.maximum(diff_mean, var)
##        obs_var = np.maximum(diff_mean, obs_var)
        var0 = 1./(1./var+1./obs_var)
        p0=(p/var+obs_mean/obs_var)*var0
        self.set_Qreliance_prob(i, j, p0, var0)


                   
    ####@@@@@## OUTDATED: calculate 
    def ability(self,TotalNumNode,weights):
        if len(weights)!=3:
            raise Exception("The number of weights to calculate ability should be 3.")
        Abil=np.zeros(TotalNumNode)
###        for i in (n for n,d in self.nodes_iter(data=True)): @obsolete networkx 1.1
        for i in self.nodes():
            p=self.get_node_data(i)['Prob']
            Abil[i]=weights[0]*p
###            S=self.predecessors(i)
            S=[n for n in self.predecessors(i)]
            for k in S:
                z=len(S)
                Abil[i]+=weights[1]*self.get_edge_data(k,i)['Q_Prob']/z            
            for j in self.neighbors(i):
                Abil[i]+=weights[2]*p*self.get_edge_data(i,j)['P_Prob']*self.get_node_data(j)['Prob']
        return Abil

    ############################################################
    #### calculate perceived ability from both capability and influence
    ###  output:  means and variances of ability for all nodes
    ###             Abil[:,0] - means
    ###             Abil[:,1] - variances
    def ability_perception(self,TotalNumNode):
        Abil=np.zeros([TotalNumNode,2])
###        for i in (n for n,d in self.nodes_iter(data=True)):  @obsolete networkx 1.1
        for i in self.nodes():
            Abil[i,0]=self.get_node_data(i)['Prob']/self.get_node_data(i)['Prob_Var']
            Abil[i,1]=1/self.get_node_data(i)['Prob_Var']
            for k in self.predecessors(i):  ## Source nodes
                Abil[i,0]+=self.get_edge_data(k,i)['P_Prob']/self.get_edge_data(k,i)['P_Prob_Var']
                Abil[i,1]+=1/self.get_edge_data(k,i)['P_Prob_Var']
                Abil[i,0]+=self.get_edge_data(k,i)['Q_Prob']/self.get_edge_data(k,i)['Q_Prob_Var']
                Abil[i,1]+=1/self.get_edge_data(k,i)['Q_Prob_Var']
            for j in self.neighbors(i):     ## Destination nodes
                Abil[i,0]+=self.get_edge_data(i,j)['P_Prob']/self.get_edge_data(i,j)['P_Prob_Var']
                Abil[i,1]+=1/self.get_edge_data(i,j)['P_Prob_Var']
                Abil[i,0]+=(1.0-self.get_edge_data(i,j)['Q_Prob'])/self.get_edge_data(i,j)['Q_Prob_Var']
                Abil[i,1]+=1/self.get_edge_data(i,j)['Q_Prob_Var']
            Abil[i,1]=1/Abil[i,1]
            Abil[i,0]=Abil[i,0]*Abil[i,1]
        return Abil


    ############################################################
    #### calculate perceived the second-order ability from both capability and influence
    ###  output:  means and variances of second-order ability for all nodes
    ###             Abil2[:,0] - means
    ###             Abil2[:,1] - variances
    def ability_perception_second_order(self,TotalNumNode):
        Abil=self.ability_perception(TotalNumNode)
        Abil2=np.zeros([TotalNumNode,2])
###        for i in (n for n,d in self.nodes_iter(data=True)): @obsolete networkx 1.1
        for i in self.nodes():
            Abil2[i,0]=Abil[i,0]/Abil[i,1]
            Abil2[i,1]=1/Abil[i,1]
            for k in self.predecessors(i):     ## Source nodes
                Abil2[i,0]+=self.get_edge_data(k,i)['P_Prob']/self.get_edge_data(k,i)['P_Prob_Var']*Abil[k,0]/Abil[k,1]
                Abil2[i,1]+=self.get_edge_data(k,i)['P_Prob']/self.get_edge_data(k,i)['P_Prob_Var']/Abil[k,1]
                Abil2[i,0]+=self.get_edge_data(k,i)['Q_Prob']/self.get_edge_data(k,i)['Q_Prob_Var']*Abil[k,0]/Abil[k,1]
                Abil2[i,1]+=self.get_edge_data(k,i)['Q_Prob']/self.get_edge_data(k,i)['Q_Prob_Var']/Abil[k,1]
            for j in self.neighbors(i):     ## Destination nodes
                Abil2[i,0]+=self.get_edge_data(i,j)['P_Prob']/self.get_edge_data(i,j)['P_Prob_Var']*Abil[j,0]/Abil[j,1]
                Abil2[i,1]+=self.get_edge_data(i,j)['P_Prob']/self.get_edge_data(i,j)['P_Prob_Var']/Abil[j,1]
                Abil2[i,0]+=(1.0-self.get_edge_data(i,j)['Q_Prob'])/self.get_edge_data(i,j)['Q_Prob_Var']*Abil[j,0]/Abil[j,1]
                Abil2[i,1]+=(1.0-self.get_edge_data(i,j)['Q_Prob'])/self.get_edge_data(i,j)['Q_Prob_Var']/Abil[j,1]
            Abil2[i,1]=1/Abil2[i,1]
            Abil2[i,0]=Abil2[i,0]*Abil2[i,1]
        return Abil2


    ############################################################
    #### calculate perceived capability - from source nodes
    ###  output:  means and variances of capability for all nodes
    ###             Capa[:,0] - means
    ###             Capa[:,1] - variances
    def capability_perception(self,TotalNumNode):
        Capa=np.zeros([TotalNumNode,2])
###        for i in (n for n,d in self.nodes_iter(data=True)):  @obsolete networkx 1.1
        for i in self.nodes():
            Capa[i,0]=self.get_node_data(i)['Prob']/self.get_node_data(i)['Prob_Var']
            Capa[i,1]=1/self.get_node_data(i)['Prob_Var']
            for k in self.predecessors(i):  ## Source nodes
                Capa[i,0]+=self.get_edge_data(k,i)['P_Prob']/self.get_edge_data(k,i)['P_Prob_Var']
                Capa[i,1]+=1/self.get_edge_data(k,i)['P_Prob_Var']
                Capa[i,0]+=self.get_edge_data(k,i)['Q_Prob']/self.get_edge_data(k,i)['Q_Prob_Var']
                Capa[i,1]+=1/self.get_edge_data(k,i)['Q_Prob_Var']
            Capa[i,1]=1/Capa[i,1]
            Capa[i,0]=Capa[i,0]*Capa[i,1]
        return Capa

    #### calculate perceived influence - from destination nodes
    ###  output:  means and variances of influence for all nodes
    ###             Infl[:,0] - means
    ###             Infl[:,1] - variances
    def influence_perception(self,TotalNumNode):
        Infl=np.zeros([TotalNumNode,2])
###        for i in (n for n,d in self.nodes_iter(data=True)):  #@ obsolete for networkx 2
        for i in self.nodes():
            Infl[i,0]=self.get_node_data(i)['Prob']/self.get_node_data(i)['Prob_Var']
            Infl[i,1]=1/self.get_node_data(i)['Prob_Var']
            for j in self.neighbors(i):     ## Destination nodes
                Infl[i,0]+=self.get_edge_data(i,j)['P_Prob']/self.get_edge_data(i,j)['P_Prob_Var']
                Infl[i,1]+=1/self.get_edge_data(i,j)['P_Prob_Var']
                Infl[i,0]+=(1.0-self.get_edge_data(i,j)['Q_Prob'])/self.get_edge_data(i,j)['Q_Prob_Var']
                Infl[i,1]+=1/self.get_edge_data(i,j)['Q_Prob_Var']
            Infl[i,1]=1/Infl[i,1]
            Infl[i,0]=Infl[i,0]*Infl[i,1]
        return Infl


    #### calculate the variance of ability from both capability and influence
    def ability_variance(self,TotalNumNode):
        Vari=np.zeros(TotalNumNode)
###        for i in (n for n,d in self.nodes_iter(data=True)): @obsolete networkx 1.1
        for i in self.nodes():
            Vari[i]=1/self.get_node_data(i)['Prob_Var']
            for k in self.predecessors(i):  ## Source nodes
                Vari[i]+=1/self.get_edge_data(k,i)['P_Prob_Var']
                Vari[i]+=1/self.get_edge_data(k,i)['Q_Prob_Var']
            for j in self.neighbors(i):     ## Destination nodes
                Vari[i]+=1/self.get_edge_data(i,j)['P_Prob_Var']
                Vari[i]+=1/self.get_edge_data(i,j)['Q_Prob_Var']
            Vari[i]=1/Vari[i]
        return Vari


    ####@@@@@## OUTDATED: calculate reciprocation based on probability instead of KL divergence
    ###         R[i,j] = P(j->i)-P(i->j)+(P(j->i)+P(i->j))/2   where P(i->j)=P(i->i+1)*...*P(j-1->j)
    def reciprocation(self,TotalNumNode,weight='None'):
        pathLen = nx.shortest_path_length(self)
#        pdb.set_trace()
        Recip=np.zeros((TotalNumNode,TotalNumNode))
        if weight=='P_Prob' or weight=='Q_Prob':       #### consider probabilistic weight ####
###            for i in (n for n,d in self.nodes_iter(data=True)): @obsolete networkx 1.1
            for i in self.nodes():
                if pathLen.has_key(i):  #if start node exists
                    D1=pathLen.get(i)
###                    for j in (n for n,d in self.nodes_iter(data=True)): @obsolete networkx 1.1
                    for j in self.nodes():
                        if D1.has_key(j):  #if end node exists
                            path1=nx.dijkstra_path(self,i,j)
###                            L1=nx.dijkstra_path_length(self,i,j)
                            if nx.dijkstra_path_length(self,i,j)==0:  ##if i==j the same node
                                L1=1.0
                            else:  ##if i!=j
                                L1=1.0
                                for k in range(len(path1)-1):
                                    L1*=self.get_edge_data(path1[k],path1[k+1])[weight]
                            if pathLen.has_key(j):
                                D2=pathLen.get(j)
                                if D2.has_key(i):   #if both forward and backward edges exist
                                    path2=nx.dijkstra_path(self,j,i)
###                                    L2=nx.dijkstra_path_length(self,j,i)
                                    if nx.dijkstra_path_length(self,j,i)==0:  ##if i==j, the same node
                                        L2=1.0
                                    else:  ##if i!=j
                                        L2=1.0
                                        for k in range(len(path2)-1):
                                            L2*=self.get_edge_data(path2[k],path2[k+1])[weight]
#                                    Recip[i,j]=math.exp(-L2)-math.exp(-L1)+math.exp(-(L2+L1))
                                    Recip[i,j]=L2-L1+(L2+L1)/2
                                else:               #if only forward edge exist, and backward edge does not exist
#                                    Recip[i,j]=-math.exp(-L1)
                                    Recip[i,j]=-L1
                            else:                   #if reciprical edge does not exist
#                                Recip[i,j]=-math.exp(-L1)
                                Recip[i,j]=-L1
                        else:                       #if end node does not exist
                            if pathLen.has_key(j):
                                D2=pathLen.get(j)
                                if D2.has_key(i):   #if forward edge does not exist, while backward edge exists
                                    path2=nx.dijkstra_path(self,j,i)
###                                    L2=nx.dijkstra_path_length(self,j,i)
                                    if nx.dijkstra_path_length(self,j,i)==0:  ##if i==j, the same node
                                        L2=1.0
                                    else:  ##if i==j
                                        L2=1.0
                                        for k in range(len(path2)-1):
                                            L2*=self.get_edge_data(path2[k],path2[k+1])[weight]
#                                    Recip[i,j]=math.exp(-L2)
                                    Recip[i,j]=L2
                                else:               #if neither forward nor backward edge exist
                                    Recip[i,j]=0
                            else:                   #if reciprical edge does not exist
                                Recip[i,j]=0
                else:                   #if start node does not exist
                    for j in (n for n,d in self.nodes_iter(data=True)):
                        D2=pathLen.get(j)
                        if D2.has_key(i):   #if forward edge does not exist, while backward edge exists
                            path2=nx.dijkstra_path(self,j,i)
###                             L2=nx.dijkstra_path_length(self,j,i)
                            if nx.dijkstra_path_length(self,j,i)==0:  ##if i!=j, the same node
                                L2=1.0
                            else:  ##if i==j
                                L2=1.0
                                for k in range(len(path2)-1):
                                    L2*=self.get_edge_data(path2[k],path2[k+1])[weight]
                            Recip[i,j]=L2
                        else:               #if neither forward nor backward edge exist
                            Recip[i,j]=0
        else:    #### not considering probabilistic weight ####
###            for i in (n for n,d in self.nodes_iter(data=True)): @obsolete networkx 1.1
            for i in self.nodes():
                if pathLen.has_key(i):  #if start node exists
                    D1=pathLen.get(i)
###                    for j in (n for n,d in self.nodes_iter(data=True)): @obsolete networkx 1.1
                    for j in self.nodes():
                        if D1.has_key(j):  #if end node exists
                            if pathLen.has_key(j):  
                                D2=pathLen.get(j)
                                if D2.has_key(i):   #if both forward and backward edges exist
                                    Recip[i,j]=math.exp(-D2[i])-math.exp(-D1[j])+math.exp(-(D2[i]+D1[j]))
                                else:               #if only forward edge exist, and backward edge does not exist
                                    Recip[i,j]=-math.exp(-D1[j])
                            else:                   #if reciprical edge does not exist
                                Recip[i,j]=-math.exp(-D1[j])
                        else:                       #if end node does not exist
                            if pathLen.has_key(j):
                                D2=pathLen.get(j)
                                if D2.has_key(i):   #if forward edge does not exist, while backward edge exists
                                    Recip[i,j]=math.exp(-D2[i])
                                else:               #if neither forward nor backward edge exist
                                    Recip[i,j]=0
                            else:                   #if reciprical edge does not exist
                                Recip[i,j]=0
                else:                   #if start node does not exist
###                    for j in (n for n,d in self.nodes_iter(data=True)): @obsolete networkx 1.1
                    for j in self.nodes():
                        D2=pathLen.get(j)
                        if D2.has_key(i):   #if forward edge does not exist, while backward edge exists
                            Recip[i,j]=math.exp(-D2[i])
                        else:               #if neither forward nor backward edge exist
                            Recip[i,j]=0
        return Recip


    ########################################################################################
    #### calculate reciprocity based on Kullback-Leibler divergence
    ###  input:
    ###     weight='None' - default value
    ###         it is deterministic reciprocity as a function of topological distance h (# of hops)
    ###         r[i,j] = exp(-h(j->i))-exp(-h(i->j))+exp(-h(j->i)-h(i->j))
    ###     weight='P_Prob' or 'Q_Prob'
    ###         R[i,j] = P(i->j)log(P(i->j)/P(j->i))-P(j->i)log(P(j->i)/P(i->j))+b0
    ###             where P(i->j)=P(i->i+1)*...*P(j-1->j)
    ###  output:  reciprocities for all pair-wise nodes as a 3D matrix R[:,:,:]
    ###         R[0,:,:] - mean
    ###         R[1,:,:] - variance
    def reciprocity(self,TotalNumNode,weight='None'):
###        TotalNumNode = self.number_of_nodes()
        EPSILON=1e-10       ## machine epsilon limit for zero
        NEUTRAL=0.5         ## neutral threshold
        MAX_VAR=1.0         ## maximum limit of variance
        pathLen = dict(nx.shortest_path_length(self))
        Recip=np.zeros((TotalNumNode,TotalNumNode))
        Vari=np.zeros((TotalNumNode,TotalNumNode))
        if weight=='P_Prob' or weight=='Q_Prob':       #### consider probabilistic weight ####
###            for i in (n for n,d in self.nodes_iter(data=True)): @obsolete networkx 1.1
            for i in self.nodes():
###                if pathLen.has_key(i):  #if start node exists @obsolete python3
                if i in pathLen:  #if start node exists
                    D1=pathLen.get(i)
###                    for j in (n for n,d in self.nodes_iter(data=True)): @obsolete networkx 1.1
                    for j in self.nodes():
###                        if D1.has_key(j):  #if end node exists @obsolete python3
                        if j in D1:  #if end node exists
                            path1=nx.dijkstra_path(self,i,j)    ## i-->j
                            if nx.dijkstra_path_length(self,i,j)==0:  ##if i==j the same node
                                L1=1.0
                                V1=0.0
                            else:  ##if i!=j
                                L1=1.0
                                V1=0.0
                                for k in range(len(path1)-1):
                                    L1*=self.get_edge_data(path1[k],path1[k+1])[weight]
                                    V1+=self.get_edge_data(path1[k],path1[k+1])[weight+'_Var']
###                            if pathLen.has_key(j):  @obsolete python3
                            if j in pathLen:
                                D2=pathLen.get(j)
###                                if D2.has_key(i):   #if both forward and backward edges exist @obsolete python3
                                if i in D2:   #if both forward and backward edges exist
                                    path2=nx.dijkstra_path(self,j,i)    ## j-->i
                                    if nx.dijkstra_path_length(self,j,i)==0:  ##if i==j, the same node
                                        L2=1.0
                                        V2=0.0
                                    else:  ##if i!=j
                                        L2=1.0
                                        V2=0.0
                                        for k in range(len(path2)-1):
                                            L2*=self.get_edge_data(path2[k],path2[k+1])[weight]
                                            V2+=self.get_edge_data(path2[k],path2[k+1])[weight+'_Var']
                                    DKL1=L1*(np.log2(np.maximum(L1,EPSILON))-np.log2(np.maximum(L2,EPSILON)))+(1-L1)*(np.log2(np.maximum((1-L1),EPSILON))-np.log2(np.maximum((1-L2),EPSILON)))
                                    DKL2=L2*(np.log2(np.maximum(L2,EPSILON))-np.log2(np.maximum(L1,EPSILON)))+(1-L2)*(np.log2(np.maximum((1-L2),EPSILON))-np.log2(np.maximum((1-L1),EPSILON)))
###                                    Recip[i,j]=L2-L1+(L2+L1)/2
                                    Recip[i,j]=(DKL1-DKL2)+NEUTRAL
                                    Vari[i,j]=np.minimum(V1+V2,MAX_VAR)
                                else:               #if only forward edge exist, and backward edge does not exist
                                    L2=0.5
                                    DKL1=L1*(np.log2(np.maximum(L1,EPSILON))-np.log2(np.maximum(L2,EPSILON)))+(1-L1)*(np.log2(np.maximum((1-L1),EPSILON))-np.log2(np.maximum((1-L2),EPSILON)))
                                    DKL2=L2*(np.log2(np.maximum(L2,EPSILON))-np.log2(np.maximum(L1,EPSILON)))+(1-L2)*(np.log2(np.maximum((1-L2),EPSILON))-np.log2(np.maximum((1-L1),EPSILON)))
                                    Recip[i,j]=(DKL1-DKL2)+NEUTRAL
###                                    Recip[i,j]=-L1
                                    Vari[i,j]=np.minimum(V1,MAX_VAR)
                            else:                   #if reciprical edge does not exist
                                L2=0.5
                                DKL1=L1*(np.log2(np.maximum(L1,EPSILON))-np.log2(np.maximum(L2,EPSILON)))+(1-L1)*(np.log2(np.maximum((1-L1),EPSILON))-np.log2(np.maximum((1-L2),EPSILON)))
                                DKL2=L2*(np.log2(np.maximum(L2,EPSILON))-np.log2(np.maximum(L1,EPSILON)))+(1-L2)*(np.log2(np.maximum((1-L2),EPSILON))-np.log2(np.maximum((1-L1),EPSILON)))
                                Recip[i,j]=(DKL1-DKL2)+NEUTRAL
###                                Recip[i,j]=-L1
                                Vari[i,j]=np.minimum(V1,MAX_VAR)
                        else:                       #if end node does not exist
###                            if pathLen.has_key(j):  @obsolete python3
                            if j in pathLen:
                                D2=pathLen.get(j)
###                                if D2.has_key(i):   #if forward edge does not exist, while backward edge exists  @obsolete python3
                                if i in D2:   #if forward edge does not exist, while backward edge exists
                                    path2=nx.dijkstra_path(self,j,i)
                                    if nx.dijkstra_path_length(self,j,i)==0:  ##if i==j, the same node
                                        L2=1.0
                                        V2=0.0
                                    else:  ##if i==j
                                        L2=1.0
                                        V2=0.0
                                        for k in range(len(path2)-1):
                                            L2*=self.get_edge_data(path2[k],path2[k+1])[weight]
                                            V2+=self.get_edge_data(path2[k],path2[k+1])[weight+'_Var']
                                    L1=0.5
                                    DKL1=L1*(np.log2(np.maximum(L1,EPSILON))-np.log2(np.maximum(L2,EPSILON)))+(1-L1)*(np.log2(np.maximum((1-L1),EPSILON))-np.log2(np.maximum((1-L2),EPSILON)))
                                    DKL2=L2*(np.log2(np.maximum(L2,EPSILON))-np.log2(np.maximum(L1,EPSILON)))+(1-L2)*(np.log2(np.maximum((1-L2),EPSILON))-np.log2(np.maximum((1-L1),EPSILON)))
                                    Recip[i,j]=(DKL1-DKL2)+NEUTRAL
###                                    Recip[i,j]=L2
                                    Vari[i,j]=np.minimum(V2,MAX_VAR)
                                else:               #if neither forward nor backward edge exist
                                    Recip[i,j]=NEUTRAL
                                    Vari[i,j]=MAX_VAR
                            else:                   #if reciprical edge does not exist
                                Recip[i,j]=NEUTRAL
                                Vari[i,j]=MAX_VAR
                else:                   #if start node does not exist
###                    for j in (n for n,d in self.nodes_iter(data=True)): @obsolete networkx 1.1
                    for j in self.nodes():
                        D2=pathLen.get(j)
###                        if D2.has_key(i):   #if forward edge does not exist, while backward edge exists  @obsolete python3
                        if i in D2:   #if forward edge does not exist, while backward edge exists
                            path2=nx.dijkstra_path(self,j,i)
                            if nx.dijkstra_path_length(self,j,i)==0:  ##if i!=j, the same node
                                L2=1.0
                                V2=0.0
                            else:  ##if i==j
                                L2=1.0
                                V2=0.0
                                for k in range(len(path2)-1):
                                    L2*=self.get_edge_data(path2[k],path2[k+1])[weight]
                                    V2+=self.get_edge_data(path2[k],path2[k+1])[weight+'_Var']
                            L1=0.5
                            DKL1=L1*(np.log2(np.maximum(L1,EPSILON))-np.log2(np.maximum(L2,EPSILON)))+(1-L1)*(np.log2(np.maximum((1-L1),EPSILON))-np.log2(np.maximum((1-L2),EPSILON)))
                            DKL2=L2*(np.log2(np.maximum(L2,EPSILON))-np.log2(np.maximum(L1,EPSILON)))+(1-L2)*(np.log2(np.maximum((1-L2),EPSILON))-np.log2(np.maximum((1-L1),EPSILON)))
                            Recip[i,j]=(DKL1-DKL2)+NEUTRAL
###                            Recip[i,j]=L2
                            Vari[i,j]=np.minimum(V2,MAX_VAR)
                        else:               #if neither forward nor backward edge exist
                            Recip[i,j]=NEUTRAL
                            Vari[i,j]=MAX_VAR
        else:    #### NOT CONSIDER probabilistic weight ####
###            for i in (n for n,d in self.nodes_iter(data=True)): @obsolete networkx 1.1
            for i in self.nodes():
###                if pathLen.has_key(i):  #if start node exists   @obsolete python3
                if i in pathLen:  #if start node exists
                    D1=pathLen.get(i)
###                    for j in (n for n,d in self.nodes_iter(data=True)): @obsolete networkx 1.1
                    for j in self.nodes():
###                        if D1.has_key(j):  #if end node exists   @obsolete python3
                        if j in D1:  #if end node exists
###                            if pathLen.has_key(j):   @obsolete python3
                            if j in pathLen:
                                D2=pathLen.get(j)
###                                if D2.has_key(i):   #if both forward and backward edges exist
                                if i in D2:   #if both forward and backward edges exist  @obsolete python3
                                    Recip[i,j]=math.exp(-D2[i])-math.exp(-D1[j])+math.exp(-(D2[i]+D1[j]))
                                else:               #if only forward edge exist, and backward edge does not exist
                                    Recip[i,j]=-math.exp(-D1[j])
                            else:                   #if reciprical edge does not exist
                                Recip[i,j]=-math.exp(-D1[j])
                        else:                       #if end node does not exist
###                            if pathLen.has_key(j):  @obsolete python3
                            if j in pathLen:
                                D2=pathLen.get(j)
###                                if D2.has_key(i):   #if forward edge does not exist, while backward edge exists  @obsolete python3
                                if i in D2:   #if forward edge does not exist, while backward edge exists
                                    Recip[i,j]=math.exp(-D2[i])
                                else:               #if neither forward nor backward edge exist
                                    Recip[i,j]=0
                            else:                   #if reciprical edge does not exist
                                Recip[i,j]=0
                else:                   #if start node does not exist
###                    for j in (n for n,d in self.nodes_iter(data=True)): @obsolete networkx 1.1
                    for j in self.nodes():
                        D2=pathLen.get(j)
###                        if D2.has_key(i):   #if forward edge does not exist, while backward edge exists  @obsolete python3
                        if i in D2:   #if forward edge does not exist, while backward edge exists
                            Recip[i,j]=math.exp(-D2[i])
                        else:               #if neither forward nor backward edge exist
                            Recip[i,j]=0
        return np.array([Recip,Vari])

    ###########################################################################
    #### calculate motive
    ###  input:
    ###  output:  Motive for all nodes as a 2D matrix M[:,:]
    ###         M[:,0] - mean
    ###         M[:,1] - variance
    def motive(self,TotalNumNode):
###        TotalNumNode=self.number_of_nodes()
        moti=np.zeros((TotalNumNode, 2))
####        vari=np.zeros(TotalNumNode)
        deg=self.out_degree()
###        for i in (n for n,d in self.nodes_iter(data=True)): @obsolete networkx 1.1
        for i in self.nodes():
            moti[i,0]=math.pow(self.get_node_data(i)['Prob'],deg[i])
            moti[i,1]=self.get_node_data(i)['Prob_Var']
        return moti
    

    ###########################################################################
    #### calculate overall benevolence
    ###  input:
    ###  output:  Benevolence for all pair-wise nodes as a 3D matrix B[:,:,:]
    ###         B[0,:,:] - mean
    ###         B[1,:,:] - variance
    def benevolence(self,TotalNumNode):
###        TotalNumNode = self.number_of_nodes()
        bene=np.zeros([TotalNumNode,TotalNumNode])
        vari=np.zeros([TotalNumNode,TotalNumNode])
        Reci=self.reciprocity(TotalNumNode,'P_Prob')
        Moti=self.motive(TotalNumNode)
###        for i in (n for n,d in self.nodes_iter(data=True)): @obsolete networkx 1.1
        for i in self.nodes():
###            for j in (n for n,d in self.nodes_iter(data=True)): @obsolete networkx 1.1
            for j in self.nodes():
                if i != j:
                    vari[i,j]=1./(1./Reci[1,i,j]+1./Moti[j,1])
                    bene[i,j]=(Reci[0,i,j]/Reci[1,i,j]+Moti[j,0]/Moti[j,1])*vari[i,j]
                else:
                    vari[i,i]=Reci[1,i,i]
                    bene[i,i]=Reci[0,i,i]
        return np.array([bene,vari])
    

    ###########################################################################
    ### update the prediction probabilities based on Monte Carlo sampling
    ### with Bayesian fusion rule
    ###  input: number of samples to draw
    ###  output:  
    ###         prediction probabilities of all nodes
    def updatePredictionBayesianSampling(self, SampleSize, TruthProb=None):
        NumNode = self.number_of_nodes()
        NumPositive = np.zeros(NumNode)
        if TruthProb==None:         ### if the ground truth probability is not specified, randomly generate
            TruthProb = np.random.random()
        for s in range(0,SampleSize):
            StartNodeTally=np.random.random_sample(NumNode)<self.get_prediction_prob()[:,0]  #Sampling based on predicted value
            GroundTruth = np.random.random()<TruthProb
            if not GroundTruth:
                StartNodeTally = ~StartNodeTally
            EndNodeTally = np.zeros(NumNode)
            ###for each source node, sample the value of destination node based on reliance probabilities
            for i in self.nodes():
                for j in self.successors(i):
                    if StartNodeTally[i]:  #given that source node i predicts TRUE
                        P11Tally=1-(np.random.random_sample()>self.get_edge_data(i,j)['P_Prob']) #indices of destination nodes predicting TRUE
                        P10Tally=1-P11Tally                                   #indices of destination nodes predicting FALSE
                        P01Tally=0
                        P00Tally=0
                        EndNodeTally[j] = np.maximum(P11Tally,EndNodeTally[j])   #record final nodes predicting T
                    else:                  #given that source node k predicts FALSE
                        P01Tally=1-(np.random.random_sample()>self.get_edge_data(i,j)['Q_Prob']) #indices of destination nodes predicting T
                        P00Tally=1-P01Tally                                   #indices of destination nodes predicting F
                        P11Tally=0
                        P10Tally=0
                        EndNodeTally[j] = np.maximum(P01Tally,EndNodeTally[j])   #record final nodes predicting T
            for i in self.nodes():
                if GroundTruth==EndNodeTally[i]:    ## If prediction is correct for node i
                    NumPositive[i]+=EndNodeTally[i]
            ### Tally for joint probabilities    -- BEGIN
            ### only 2 nodes (4 combinations) are considered
            """if EndNodeTally[0] and  EndNodeTally[1]:
                NumJoint[0] += 1
            elif EndNodeTally[0] and  1-EndNodeTally[1]:
                NumJoint[1] += 1
            elif 1-EndNodeTally[0] and  EndNodeTally[1]:
                NumJoint[2] += 1
            elif 1-EndNodeTally[0] and  1-EndNodeTally[1]:
                NumJoint[3] += 1 
            ### Tally for joint probabilities    -- END"""
            """### Tally for joint probabilities    -- BEGIN
            ### only 3 nodes (8 combinations) are considered
            if EndNodeTally[0] and  EndNodeTally[1] and EndNodeTally[2]:
                NumJoint[0] += 1
            elif EndNodeTally[0] and  EndNodeTally[1] and 1-EndNodeTally[2]:
                NumJoint[1] += 1
            elif EndNodeTally[0] and  1-EndNodeTally[1] and EndNodeTally[2]:
                NumJoint[2] += 1
            elif EndNodeTally[0] and  1-EndNodeTally[1] and 1-EndNodeTally[2]:
                NumJoint[3] += 1
            elif 1-EndNodeTally[0] and  EndNodeTally[1] and EndNodeTally[2]:
                NumJoint[4] += 1
            elif 1-EndNodeTally[0] and  EndNodeTally[1] and 1-EndNodeTally[2]:
                NumJoint[5] += 1
            elif 1-EndNodeTally[0] and  1-EndNodeTally[1] and EndNodeTally[2]:
                NumJoint[6] += 1
            elif 1-EndNodeTally[0] and  1-EndNodeTally[1] and 1-EndNodeTally[2]:
                NumJoint[7] += 1
            ### Tally for joint probabilities    -- END """
        NumEval=31    #resolution or num of evaluation points for probability from 0 to 1
        for i in self.nodes():
            likelihood=np.zeros(NumEval) #all likelihood values
            max_likelihood=0.0            #keep track of the maximum posterior
            for r in range(0,NumEval):
                p=float(r)/(NumEval-1)  #probability of certain value
                likelihood[r]=np.power(p,NumPositive[i])*np.power((1.0-p),SampleSize-NumPositive[i]) #prior*likelihood
                max_likelihood=max(max_likelihood,likelihood[r])
            self.nodes(data=True)[i]['Prob']=self.nodes(data=True)[i]['Prob']*max_likelihood/max(sum(likelihood),1e-30) #update prediction prob. to the normalized max posterior

        """return NumJoint/SampleSize"""
    
    ###########################################################################
    ### update the prediction probabilities based on Monte Carlo sampling
    ### with Worst-case fusion rule
    ###  input: number of samples to draw
    ###  output:  
    ###         prediction probabilities of all nodes
    def updatePredictionWorstcaseSampling(self, SampleSize, SelfInclusion=False, TruthProb=None):
        NumNode = self.number_of_nodes()
        NumPositive = np.zeros(NumNode)
        NumJoint = np.zeros(8)      ### Tally of joint probabilities  """
        if TruthProb==None:         ### if the ground truth probability is not specified, randomly generate
            TruthProb = np.random.random()
        for s in range(0,SampleSize):
            StartNodeTally=np.random.random_sample(NumNode)<self.get_prediction_prob()[:,0]
            GroundTruth = np.random.random()<TruthProb
            if not GroundTruth:
                StartNodeTally = ~StartNodeTally
            if SelfInclusion:
                EndNodeTally = 1*StartNodeTally
            else:
                EndNodeTally = np.ones(NumNode)
            ###for each source node, sample the value of destination node based on reliance probabilities
            for i in self.nodes():
                for j in self.successors(i):
                    if StartNodeTally[i]:  #given that source node i predicts T
                        P11Tally=1-(np.random.random_sample()>self.get_edge_data(i,j)['P_Prob']) #indices of destination nodes predicting T
                        P10Tally=1-P11Tally                                   #indices of destination nodes predicting F
                        P01Tally=0
                        P00Tally=0
                        EndNodeTally[j] *= P11Tally   #record final nodes predicting T
                    else:                  #given that source node k predicts F
                        P01Tally=1-(np.random.random_sample()>self.get_edge_data(i,j)['Q_Prob']) #indices of destination nodes predicting T
                        P00Tally=1-P01Tally                                   #indices of destination nodes predicting F
                        P11Tally=0
                        P10Tally=0
                        EndNodeTally[j] *= P01Tally   #record final nodes predicting T
            for i in self.nodes():
                if GroundTruth==EndNodeTally[i]:    ## If prediction is correct for node i
                    NumPositive[i]+=1
            ### Tally for joint probabilities    -- BEGIN
            ### only 2 nodes (4 combinations) are considered
            """if EndNodeTally[0] and  EndNodeTally[1]:
                NumJoint[0] += 1
            elif EndNodeTally[0] and  1-EndNodeTally[1]:
                NumJoint[1] += 1
            elif 1-EndNodeTally[0] and  EndNodeTally[1]:
                NumJoint[2] += 1
            elif 1-EndNodeTally[0] and  1-EndNodeTally[1]:
                NumJoint[3] += 1 
            ### Tally for joint probabilities    -- END """
            """### Tally for joint probabilities    -- BEGIN
            ### only 3 nodes (8 combinations) are considered
            if GroundTruth==EndNodeTally[0] and  GroundTruth==EndNodeTally[1] and GroundTruth==EndNodeTally[2]:
                NumJoint[0] += 1
            elif GroundTruth==EndNodeTally[0] and  GroundTruth==EndNodeTally[1] and GroundTruth!=EndNodeTally[2]:
                NumJoint[1] += 1
            elif GroundTruth==EndNodeTally[0] and  GroundTruth!=EndNodeTally[1] and GroundTruth==EndNodeTally[2]:
                NumJoint[2] += 1
            elif GroundTruth==EndNodeTally[0] and  GroundTruth!=EndNodeTally[1] and GroundTruth!=EndNodeTally[2]:
                NumJoint[3] += 1
            elif GroundTruth!=EndNodeTally[0] and  GroundTruth==EndNodeTally[1] and GroundTruth==EndNodeTally[2]:
                NumJoint[4] += 1
            elif GroundTruth!=EndNodeTally[0] and  GroundTruth==EndNodeTally[1] and GroundTruth!=EndNodeTally[2]:
                NumJoint[5] += 1
            elif GroundTruth!=EndNodeTally[0] and  GroundTruth!=EndNodeTally[1] and GroundTruth==EndNodeTally[2]:
                NumJoint[6] += 1
            elif GroundTruth!=EndNodeTally[0] and  GroundTruth!=EndNodeTally[1] and GroundTruth!=EndNodeTally[2]:
                NumJoint[7] += 1
            ### Tally for joint probabilities    -- END  """
        for i in self.nodes():
            self.nodes(data=True)[i]['Prob']=float(NumPositive[i])/SampleSize

        #return NumJoint/SampleSize


    def updatePredictionWorstcaseSampling_joint(self, SampleSize, SelfInclusion=False, TruthProb=None):
        NumNode = self.number_of_nodes()
        NumPositive = np.zeros(NumNode)
        NumJoint = np.zeros(8)      ### Tally of joint probabilities  """
        if TruthProb==None:         ### if the ground truth probability is not specified, randomly generate
            TruthProb = np.random.random()
        for s in range(0,SampleSize):
            StartNodeTally=np.random.random_sample(NumNode)<self.get_prediction_prob()[:,0]
            GroundTruth = np.random.random()<TruthProb
            if not GroundTruth:
                StartNodeTally = ~StartNodeTally
            if SelfInclusion:
                EndNodeTally = 1*StartNodeTally
            else:
                EndNodeTally = np.ones(NumNode)
            ###for each source node, sample the value of destination node based on reliance probabilities
            for i in self.nodes():
                for j in self.successors(i):
                    if StartNodeTally[i]:  #given that source node i predicts T
                        P11Tally=1-(np.random.random_sample()>self.get_edge_data(i,j)['P_Prob']) #indices of destination nodes predicting T
                        P10Tally=1-P11Tally                                   #indices of destination nodes predicting F
                        P01Tally=0
                        P00Tally=0
                        EndNodeTally[j] *= P11Tally   #record final nodes predicting T
                    else:                  #given that source node k predicts F
                        P01Tally=1-(np.random.random_sample()>self.get_edge_data(i,j)['Q_Prob']) #indices of destination nodes predicting T
                        P00Tally=1-P01Tally                                   #indices of destination nodes predicting F
                        P11Tally=0
                        P10Tally=0
                        EndNodeTally[j] *= P01Tally   #record final nodes predicting T
            for i in self.nodes():
                if GroundTruth==EndNodeTally[i]:    ## If prediction is correct for node i
                    NumPositive[i]+=1
            ### Tally for joint probabilities    -- BEGIN
            ### only 2 nodes (4 combinations) are considered
            """if EndNodeTally[0] and  EndNodeTally[1]:
                NumJoint[0] += 1
            elif EndNodeTally[0] and  1-EndNodeTally[1]:
                NumJoint[1] += 1
            elif 1-EndNodeTally[0] and  EndNodeTally[1]:
                NumJoint[2] += 1
            elif 1-EndNodeTally[0] and  1-EndNodeTally[1]:
                NumJoint[3] += 1 
            ### Tally for joint probabilities    -- END """
            ### Tally for joint probabilities    -- BEGIN
            ### only 3 nodes (8 combinations) are considered
            if GroundTruth==EndNodeTally[0] and  GroundTruth==EndNodeTally[1] and GroundTruth==EndNodeTally[2]:
                NumJoint[0] += 1
            elif GroundTruth==EndNodeTally[0] and  GroundTruth==EndNodeTally[1] and GroundTruth!=EndNodeTally[2]:
                NumJoint[1] += 1
            elif GroundTruth==EndNodeTally[0] and  GroundTruth!=EndNodeTally[1] and GroundTruth==EndNodeTally[2]:
                NumJoint[2] += 1
            elif GroundTruth==EndNodeTally[0] and  GroundTruth!=EndNodeTally[1] and GroundTruth!=EndNodeTally[2]:
                NumJoint[3] += 1
            elif GroundTruth!=EndNodeTally[0] and  GroundTruth==EndNodeTally[1] and GroundTruth==EndNodeTally[2]:
                NumJoint[4] += 1
            elif GroundTruth!=EndNodeTally[0] and  GroundTruth==EndNodeTally[1] and GroundTruth!=EndNodeTally[2]:
                NumJoint[5] += 1
            elif GroundTruth!=EndNodeTally[0] and  GroundTruth!=EndNodeTally[1] and GroundTruth==EndNodeTally[2]:
                NumJoint[6] += 1
            elif GroundTruth!=EndNodeTally[0] and  GroundTruth!=EndNodeTally[1] and GroundTruth!=EndNodeTally[2]:
                NumJoint[7] += 1
            ### Tally for joint probabilities    -- END
        for i in self.nodes():
            self.nodes(data=True)[i]['Prob']=float(NumPositive[i])/SampleSize

        return NumJoint/SampleSize

    ###########################################################################
    ### update the prediction probabilities based on Monte Carlo sampling
    ### with Best-case fusion rule
    ###  input: number of samples to draw
    ###  output:  
    ###         prediction probabilities of all nodes
    def updatePredictionBestcaseSampling(self,SampleSize, SelfInclusion=False, TruthProb=None):
        NumNode = self.number_of_nodes()
        NumPositive = np.zeros(NumNode)
        """NumJoint = np.zeros(8)      ### Tally of joint probabilities """
        if TruthProb==None:         ### if the ground truth probability is not specified, randomly generate
            TruthProb = np.random.random()
        for s in range(0,SampleSize):
            StartNodeTally=np.random.random_sample(NumNode)<self.get_prediction_prob()[:,0]
            GroundTruth = np.random.random()<TruthProb
            if not GroundTruth:
                StartNodeTally = ~StartNodeTally
            if SelfInclusion:
                EndNodeTally = 1*StartNodeTally
            else:
                EndNodeTally = np.zeros(NumNode)
            ###for each source node, sample the value of destination node based on reliance probabilities
            for i in self.nodes():
                for j in self.successors(i):
                    if StartNodeTally[i]:  #given that source node i predicts T
                        P11Tally=1-(np.random.random_sample()>self.get_edge_data(i,j)['P_Prob']) #indices of destination nodes predicting T
                        P10Tally=1-P11Tally                                   #indices of destination nodes predicting F
                        P01Tally=0
                        P00Tally=0
                        EndNodeTally[j]=np.maximum(P11Tally,EndNodeTally[j])   #record final nodes predicting T
                    else:                  #given that source node k predicts F
                        P01Tally=1-(np.random.random_sample()>self.get_edge_data(i,j)['Q_Prob']) #indices of destination nodes predicting T
                        P00Tally=1-P01Tally                                   #indices of destination nodes predicting F
                        P11Tally=0
                        P10Tally=0
                        EndNodeTally[j]=np.maximum(P01Tally,EndNodeTally[j])   #record final nodes predicting T
            for i in self.nodes():
                if GroundTruth==EndNodeTally[i]:    ## If prediction is correct for node i
                    NumPositive[i]+=1

            """### Tally for joint probabilities    -- BEGIN
            if EndNodeTally[0] and  EndNodeTally[1]:
                NumJoint[0] += 1
            elif EndNodeTally[0] and  1-EndNodeTally[1]:
                NumJoint[1] += 1
            elif 1-EndNodeTally[0] and  EndNodeTally[1]:
                NumJoint[2] += 1
            elif 1-EndNodeTally[0] and  1-EndNodeTally[1]:
                NumJoint[3] += 1
            ### Tally for joint probabilities    -- END """
            """### Tally for joint probabilities    -- BEGIN
            if EndNodeTally[0] and  EndNodeTally[1] and EndNodeTally[2]:
                NumJoint[0] += 1
            elif EndNodeTally[0] and  EndNodeTally[1] and 1-EndNodeTally[2]:
                NumJoint[1] += 1
            elif EndNodeTally[0] and  1-EndNodeTally[1] and EndNodeTally[2]:
                NumJoint[2] += 1
            elif EndNodeTally[0] and  1-EndNodeTally[1] and 1-EndNodeTally[2]:
                NumJoint[3] += 1
            elif 1-EndNodeTally[0] and  EndNodeTally[1] and EndNodeTally[2]:
                NumJoint[4] += 1
            elif 1-EndNodeTally[0] and  EndNodeTally[1] and 1-EndNodeTally[2]:
                NumJoint[5] += 1
            elif 1-EndNodeTally[0] and  1-EndNodeTally[1] and EndNodeTally[2]:
                NumJoint[6] += 1
            elif 1-EndNodeTally[0] and  1-EndNodeTally[1] and 1-EndNodeTally[2]:
                NumJoint[7] += 1
            ### Tally for joint probabilities    -- END """
        for i in self.nodes():
            self.nodes(data=True)[i]['Prob']=float(NumPositive[i])/SampleSize

        """return NumJoint/SampleSize"""

        
#######################################################
### Randomly generated probabilistic graph
#######################################################
#
#class RandomProbGraph(ProbGraph):
#    def __init__(self,numNode=0,probConnect=0,seed=None):
#        ProbGraph.__init__(self)
#        try:
#            g=nx.fast_gnp_random_graph(numNode,probConnect,seed,True)
#            for k in g.nodes_iter():
#                self.add_node(k,Prob=random.random(),Prob_Var=random.uniform(0,0.5))
#            for i,j in g.edges_iter():
#                self.add_edge(i,j,P_Prob=random.random(),Q_Prob=random.random(),P_Prob_Var=random.uniform(0,0.5),Q_Prob_Var=random.uniform(0,0.5))
#        except ValueError as err:
#            print(err.args)

