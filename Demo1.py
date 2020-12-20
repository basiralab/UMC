
import numpy as np
from UnifiedClassificationModel import UCM
from Classification import Classify_Unimodal_Connectomes


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


# Unified classification on multi-modal dataset
# Scores = UCM(All_Graphs, All_Labels, nt=25, feat_method='DB', P=3, corr='hard')
# Scores = UCM(All_Graphs, All_Labels, nt=25, feat_method='DB', P=3, corr='soft')
# Scores = UCM(All_Graphs, All_Labels, nt=25, feat_method='GW', d=20, corr='hard')
Scores = UCM(All_Graphs, All_Labels, nt=25, feat_method='GW', d=20, corr='soft')


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