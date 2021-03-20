# ProbNet

ProbNet is a python based cyber-physical-social network modeling and simulation toolkit. It is based on the concept of probabilistic graph model. Cyber-physical-social systems (CPSS) are physical devices that are embedded in human society and possess highly integrated functionalities of sensing, computing, communication, and control. 

In ProbNet, information collection, processing, and exchange among CPSS nodes are modeled based on the Prediction Probability associated with each node and Reliance Probaiblity associated with each edge. It has several modules:

1) resilience:

It simulates network disruption and demonstrates how the resilience of the network can be quantified based on mutual information.

Details are available in:
- Wang, Y. (2018). Resilience quantification for probabilistic design of cyber-physical system networks. ASCE-ASME J Risk and Uncert in Engrg Sys Part B, 4(3): 031006. https://doi.org/10.1115/1.4039148

2) trust:

The trustworthiness of CPSS is quantified by ability, benevolence, and integrity. ProbNet provides tools to calculate the metrics of ability and benevolence.

Details are available in:
- Wang, Y. (2018). Trust quantification for networked cyber-physical systems. IEEE Internet of Things Journal, 5(3), 2055-2070. https://doi.org/10.1109/JIOT.2018.2822677

- Wang, Y. (2021). Design of Trustworthy Cyber-Physical-Social Systems With Discrete Bayesian Optimization. Journal of Mechanical Design,143(7): 071702.  https://doi.org/10.1115/1.4049532

3) infoDyn:

The information dynamics of CPSS is modeled and predicted with copulas, vector autoregression, and Gaussian process regression.

