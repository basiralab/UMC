
import numpy as np
from UnifiedClassificationModel import MultipleClassifications


# Reading simulated brain connectomes
ALL_ASD_50 = np.load('Data/ALL_ASD_50.npy')
ALL_NC_50  = np.load('Data/ALL_NC_50.npy')
ALL_ASD_40 = np.load('Data/ALL_ASD_40.npy')
ALL_NC_40  = np.load('Data/ALL_NC_40.npy')
ALL_ASD_30 = np.load('Data/ALL_ASD_30.npy')
ALL_NC_30  = np.load('Data/ALL_NC_30.npy')


# Combining ASD and NC subjects to get modalities
Graphs_1 = np.concatenate([ALL_ASD_50, ALL_NC_50], axis=0)
Graphs_2 = np.concatenate([ALL_ASD_40, ALL_NC_40], axis=0)
Graphs_3 = np.concatenate([ALL_ASD_30, ALL_NC_30], axis=0)


# Class label (i.e, health status):   ASD: 1 / NC: 0
Labels_1 = np.array([1]*len(ALL_ASD_50) + [0]*len(ALL_NC_50))
Labels_2 = np.array([1]*len(ALL_ASD_40) + [0]*len(ALL_NC_40))
Labels_3 = np.array([1]*len(ALL_ASD_30) + [0]*len(ALL_NC_30))


# Putting all modalities together in a list
All_Graphs = [Graphs_1, Graphs_2, Graphs_3]
All_Labels = [Labels_1, Labels_2, Labels_3]


nt_list = [10,15,20,25]
K_list = [20,30,40]


# Calculate all possible scores and plot them collectively
MultipleClassifications(All_Graphs, All_Labels, nt_list, K_list, P=3, d=20, eta=0.9, stdScale_GW=True, Fold=5, seed=100, Names_UC=None)