####################################################################
### ProbNet: A probabilistic graph modeling toolkit that enables the design of cyber-physical-social systems 
###	     with the considerations of trust, resilience, and information dynamics',
###   author: Yan Wang,
###   author_email: yan-wang@gatech.edu
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
from probGraph.trustEvaluation import TrustEvaluation
#from randomProbGraph import RandomProbGraph
import numpy as np

class GreedyGraphOpt:
    def __init__(self, evaluation, maxNumTrial=1000):
        self.evaluation = evaluation
        self.prGraph = evaluation.prGraph
#        self.refNodeName = evaluation.refNodeName
#        self.criterion = evaluation.criterion
        self.maxNumTrial = maxNumTrial
        self.NumTrial = 0
        self.optNodeSet = None
        self.bestSol = None
        self.bestObj = None
        self.nodeName = {}
        self.nodeIndex = {}
##        self.objValue = np.zeros((self.TotalNumTrial,self.prGraph.number_of_nodes()))
        self.Utility = np.zeros(self.maxNumTrial)
        i = 0
        for n in self.prGraph.nodes():
            self.nodeIndex[n]=i
            self.nodeName[i]=n
            i+=1

    ### Find the optimal subgraph that maximize the trustworthiness respect to the reference node
    ###       INPUT:
    ###       OUTPUT:
    ###            The optimal subgraph
    def run(self):
        self.optNodeSet = set([self.evaluation.refNodeName])

        ### Queue for breadth-first search
        import queue
        que = queue.Queue()
        que.put(self.evaluation.refNodeName)

        ### create a undirected copy of graph for search
        ug = self.prGraph.to_undirected()

        ### keep track of which nodes have been visited
        statevec = np.zeros(self.prGraph.number_of_nodes())
        visited = {}
        for n in self.prGraph.nodes():
            visited[n]=False

        self.NumTrial=1
        while que.qsize()>0 and self.NumTrial<self.maxNumTrial:
            if que.qsize()>0:
                s = que.get()
                statevec[self.nodeIndex[s]]=1
##                trialnodeset = subnodeset.union(set([s]))
##                sg=self.prGraph.subgraph(trialnodeset)
                self.Utility[self.NumTrial] = self.evaluation.evaluateSubgraph_mean(statevec)
                if self.Utility[self.NumTrial] >= self.Utility[self.NumTrial-1]:  ####accept the new node in subnodeset
                    self.optNodeSet=self.optNodeSet.union(set([s]))
                    visited[s]=True
                    for n in ug.neighbors(s):
                        if not visited[n]:
                            que.put(n)
                else:                   ####roll back to the previous subnodeset
                    if not self.optNodeSet.__contains__(s):
                        visited[s]=False
                        statevec[self.nodeIndex[s]]=0
                    self.Utility[self.NumTrial]=self.Utility[self.NumTrial-1]
            self.NumTrial+=1
            self.bestSol = statevec
            self.bestObj = self.evaluation.evaluateSubgraph_mean(self.bestSol)
        return


