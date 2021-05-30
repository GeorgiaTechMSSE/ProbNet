####################################################################
### ProbNet: A probabilistic graph modeling toolkit that enables the design of cyber-physical-social systems 
###	     with the considerations of trust, resilience, and information dynamics',
###   author: Yan Wang,
###   author_email: yan-wang@gatech.edu
####################################################################
####################################################################
### Evaluate the trustworthiness of the subgraph with respect to a 
### reference node within a probabilitic graph 
### --- To initiate:
###     TrustEvaluation(prGraph, referenceNodeName, criterion)
###       INPUT:
###       - prGraph: the original probabilistic graph
###       - refNodeName: the reference node
###       - criterion:
###             'ability', 'ability_2nd_order', (subcategory of ability: 'capability', 'influence'),
###             'benevolence', (subcategory of benevelence: 'reciprocity_deterministic', 'reciprocity_P_prob', 'reciprocity_Q_prob', 'motive')
### --- Methods:
###     evaluateSubgraph(statevec)
###     evaluateSubgraph_mean(statevec)
###     evaluateSubgraph_vari(statevec)
###
####################################################################
from probGraph.probGraph import ProbGraph
#from probGraph.randomProbGraph import RandomProbGraph
import numpy as np

class TrustEvaluation:
    def __init__(self, prGraph=None, refNodeName=None, criterion='ability', selfInterestLevel=1.0):
        if selfInterestLevel > 1 or selfInterestLevel < 0:
            raise ValueError('"selfInterestLevel" should have the value range between 0 and 1')
        self.prGraph = prGraph
        self.refNodeName = refNodeName
        self.criterion = criterion
        self.selfInterestLevel = selfInterestLevel
        self.nodeName = {}
        self.nodeIndex = {}
        i = 0
        for n in self.prGraph.nodes():
            self.nodeIndex[n]=i
            self.nodeName[i]=n
            i+=1

    ### evaluate the trustworthiness of the subgraph, return both mean and variance
    ###       INPUT:
    ###       - statevec: the state vector contains an array of binary (0 or 1)
    ###                   value corresponding to the node indices,
    ###                   "1" indicates that a node is included in the subgraph
    ###                   "0" indicates that the node is not included
    ###       OUTPUT:
    ###            The trustworthiness values w.r.t. the reference node as array [mean, variance]
    def evaluateSubgraph(self, statevec):
        subnodeset = set([self.refNodeName])
        for i in range(statevec.size):
            if statevec[i] == 1:
                subnodeset = subnodeset.union(set([self.nodeName[i]]))
        sg=self.prGraph.subgraph(subnodeset)
        if self.criterion == 'ability':
            eval=sg.ability_perception(self.prGraph.number_of_nodes())[self.nodeIndex[self.refNodeName],:]
        elif self.criterion == 'ability_2nd_order': 
            eval=sg.ability_perception_second_order(self.prGraph.number_of_nodes())[self.nodeIndex[self.refNodeName],:]
        elif self.criterion == 'capability': 
            eval=sg.capability_perception(self.prGraph.number_of_nodes())[self.nodeIndex[self.refNodeName],:]
        elif self.criterion == 'influence': 
            eval=sg.influence_perception(self.prGraph.number_of_nodes())[self.nodeIndex[self.refNodeName],:]
        elif self.criterion == 'benevolence':
            benev=sg.benevolence(self.prGraph.number_of_nodes())
            meanall=(np.sum(benev[0,:,:],axis=1)-benev[0,:,:].diagonal())/np.maximum(sg.number_of_nodes()-1,1)
            variall=(np.sum(benev[1,:,:],axis=1)-benev[1,:,:].diagonal())/np.maximum(sg.number_of_nodes()-1,1)
            weights=np.zeros(self.prGraph.number_of_nodes())
            w=(1-self.selfInterestLevel)/np.maximum(sg.number_of_nodes()-1,1)
            for n in sg.nodes():
                weights[self.nodeIndex[n]]=w
            weights[self.nodeIndex[self.refNodeName]]=self.selfInterestLevel
            eval=[np.dot(meanall,weights), np.dot(variall,weights)]
        elif self.criterion == 'reciprocity_deterministic':
            recip=sg.reciprocity(self.prGraph.number_of_nodes(),weight='None')
            meanall=(np.sum(recip[0,:,:],axis=1)-recip[0,:,:].diagonal())/np.maximum(sg.number_of_nodes()-1,1)
            variall=(np.sum(recip[1,:,:],axis=1)-recip[1,:,:].diagonal())/np.maximum(sg.number_of_nodes()-1,1)
            weights=np.zeros(self.prGraph.number_of_nodes())
            w=(1-self.selfInterestLevel)/np.maximum(sg.number_of_nodes()-1,1)
            for n in sg.nodes():
                weights[self.nodeIndex[n]]=w
            weights[self.nodeIndex[self.refNodeName]]=self.selfInterestLevel
            eval=[np.dot(meanall,weights), np.dot(variall,weights)]
        elif self.criterion == 'reciprocity_P_prob':
            recip=sg.reciprocity(self.prGraph.number_of_nodes(),weight='P_Prob')
            meanall=(np.sum(recip[0,:,:],axis=1)-recip[0,:,:].diagonal())/np.maximum(sg.number_of_nodes()-1,1)
            variall=(np.sum(recip[1,:,:],axis=1)-recip[1,:,:].diagonal())/np.maximum(sg.number_of_nodes()-1,1)
            weights=np.zeros(self.prGraph.number_of_nodes())
            w=(1-self.selfInterestLevel)/np.maximum(sg.number_of_nodes()-1,1)
            for n in sg.nodes():
                weights[self.nodeIndex[n]]=w
            weights[self.nodeIndex[self.refNodeName]]=self.selfInterestLevel
            eval=[np.dot(meanall,weights), np.dot(variall,weights)]
        elif self.criterion == 'reciprocity_Q_prob':
            recip=sg.reciprocity(self.prGraph.number_of_nodes(),weight='Q_Prob')
            meanall=(np.sum(recip[0,:,:],axis=1)-recip[0,:,:].diagonal())/np.maximum(sg.number_of_nodes()-1,1)
            variall=(np.sum(recip[1,:,:],axis=1)-recip[1,:,:].diagonal())/np.maximum(sg.number_of_nodes()-1,1)
            weights=np.zeros(self.prGraph.number_of_nodes())
            w=(1-self.selfInterestLevel)/np.maximum(sg.number_of_nodes()-1,1)
            for n in sg.nodes():
                weights[self.nodeIndex[n]]=w
            weights[self.nodeIndex[self.refNodeName]]=self.selfInterestLevel
            eval=[np.dot(meanall,weights), np.dot(variall,weights)]
        elif self.criterion == 'motive':
            eval=sg.motive(self.prGraph.number_of_nodes())[self.nodeIndex[self.refNodeName],:]
        else:
            eval=None
        return eval

    ### evaluate the mean value of trustworthiness of the subgraph
    ###     evaluateSubgraph_mean(statevec)
    ###       INPUT:
    ###       - statevec: the state vector contains an array of binary (0 or 1)
    ###                   value corresponding to the node indices,
    ###                   "1" indicates that a node is included in the subgraph
    ###                   "0" indicates that the node is not included
    ###       OUTPUT:
    ###            The mean value of trustworthiness  w.r.t. the reference node
    def evaluateSubgraph_mean(self, statevec):
        return self.evaluateSubgraph(statevec)[0]

    ### evaluate the variance of trustworthiness of the subgraph
    ###     evaluateSubgraph_vari(statevec)
    ###       INPUT:
    ###       - statevec: the state vector contains an array of binary (0 or 1)
    ###                   value corresponding to the node indices,
    ###                   "1" indicates that a node is included in the subgraph
    ###                   "0" indicates that the node is not included
    ###       OUTPUT:
    ###            The variance of trustworthiness  w.r.t. the reference node
    def evaluateSubgraph_vari(self, statevec):
        return self.evaluateSubgraph(statevec)[1]


