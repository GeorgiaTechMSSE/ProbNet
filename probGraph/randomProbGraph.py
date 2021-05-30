####################################################################
### ProbNet: A probabilistic graph modeling toolkit that enables the design of cyber-physical-social systems 
###	     with the considerations of trust, resilience, and information dynamics',
###   author: Yan Wang,
###   author_email: yan-wang@gatech.edu
####################################################################
import networkx as nx
from probGraph.probGraph import ProbGraph
import random
#import pdb

class RandomProbGraph(ProbGraph):
    def __init__(self,numNode=0,probConnect=0,seed=None):
        ProbGraph.__init__(self)
        try:
            g=nx.fast_gnp_random_graph(numNode,probConnect,seed,True)
#            self.add_nodes_from(g.nodes)
#            self.add_edges_from(g.edges)
#            self=(ProbGraph)g.copy()
            for k in g.nodes():
                self.add_node(k,Prob=random.random(),Prob_Var=random.random())
#                self.node[k]['Prob']=random.random()
#                self.node[k]['Prob_Var']=random.random()
            for i,j in g.edges():
                self.add_edge(i,j,P_Prob=random.random(),Q_Prob=random.random(),P_Prob_Var=random.random(),Q_Prob_Var=random.random())
#                self.edge[i][j]['P_Prob']=random.random()
#                self.edge[i][j]['Q_Prob']=random.random()
#                self.edge[i][j]['P_Prob_Var']=random.random()
#                self.edge[i][j]['Q_Prob_Var']=random.random()
        except ValueError as err:
            print(err.args)
