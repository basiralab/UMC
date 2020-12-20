
import numpy as np
from UnifiedClassificationModel import MultipleClassifications


# Reading simulated brain connectomes
A = np.load('Data/Class1_50.npy')
B = np.load('Data/Class1_40.npy')
C = np.load('Data/Class1_30.npy')
D  = np.load('Data/Class2_50.npy')
E  = np.load('Data/Class2_40.npy')
F  = np.load('Data/Class2_30.npy')


# Combining ASD and NC subjects to get modalities
Graphs_1 = np.concatenate([A, D], axis=0)
Graphs_2 = np.concatenate([B, E], axis=0)
Graphs_3 = np.concatenate([C, F], axis=0)


# Class label (i.e, health status):   ASD: 1 / NC: 0
Labels_1 = np.array([1]*len(A) + [0]*len(D))
Labels_2 = np.array([1]*len(B) + [0]*len(E))
Labels_3 = np.array([1]*len(C) + [0]*len(F))


# Putting all modalities together in a list
All_Graphs = [Graphs_1, Graphs_2, Graphs_3]
All_Labels = [Labels_1, Labels_2, Labels_3]


nt_list = [10,15,20,25]
K_list = [20,30,40]


# Calculate all possible scores and plot them collectively
MultipleClassifications(All_Graphs, All_Labels, nt_list, K_list, P=3, d=20, eta=0.9, stdScale_GW=True, Fold=5, seed=100, Names_UC=None)