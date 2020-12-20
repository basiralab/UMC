
import numpy as np
from UnifiedClassificationModel import UMC
from Classification import Classify_Unimodal_Connectomes


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


# Unified classification on multi-modal dataset
# Scores = UMC(All_Graphs, All_Labels, nt=25, feat_method='DB', P=3, corr='hard')
# Scores = UMC(All_Graphs, All_Labels, nt=25, feat_method='DB', P=3, corr='soft')
# Scores = UMC(All_Graphs, All_Labels, nt=25, feat_method='GW', d=20, corr='hard')
Scores = UMC(All_Graphs, All_Labels, nt=25, feat_method='GW', d=20, corr='soft')


# Independent classifications on each unimodal dataset
# Scores = Classify_Unimodal_Connectomes(Graphs_1, Labels_1, FS_strategy='SNF', K=30)
# Scores = Classify_Unimodal_Connectomes(Graphs_2, Labels_2, FS_strategy='Averaging', K=30)
# Scores = Classify_Unimodal_Connectomes(Graphs_3, Labels_3, FS_strategy='Averaging', K=30)


Acc,Sens,Spec = Scores

print(f'\n\nPerformance Scores')
print('-----------------------')
print(f'Accuracy     :  {Acc} %')
print(f'Sensitivity  :  {Sens} %')
print(f'Specificity  :  {Spec} %')