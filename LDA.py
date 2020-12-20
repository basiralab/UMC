

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from ModalityCombination import get_train_test_data




def dim_reduction_with_LDA(All_Aligned_Graphs, All_Labels, Tr_Ind, Tst_Ind):
	"""
	Given list of M sets of aligned graphs, apply LDA modality-wise as a dimensionality reduction and
	combine the reduced graphs across modalities by further splitting them to "training" and "testing" sets.

	Parameters:
	----------
	All_Aligned_Graphs : list of M sets of aligned graphs of M distinct modalities, each with shape (N_m, n_t, n_t)

	All_Labels : list of M label arrays, each with length N_m

	Tr_Ind : list of M 1-D index arrays,
		the m-th index array holds the indices of the training subjects in 
		All_Graphs[m] or All_Features[m] during particular fold

	Tst_Ind : list of M 1-D index arrays,
		the m-th index array holds the indices of the testing subjects in 
		All_Graphs[m] or All_Features[m] during particular fold


	Return:
	-------
	out : training and testing datasets with shapes (~(Fold-1)*N/Fold, 1) and (~N/Fold, 1),
		where "N" is the total number of graphs from all M modalities (i.e, N = N_1 + ... + N_M).
	"""

	# All_Data_Vec :  list of M sets of "vectorized" graphs
	All_Data_Vec = [ np.array([graph_i[np.triu_indices(len(graph_i),1)] for graph_i in Aligned_Graphs_m])  for Aligned_Graphs_m in All_Aligned_Graphs]

	All_Data_reduced = [ lda(Data_Vec_m, Labels_m, tr_i) for Data_Vec_m, Labels_m, tr_i in zip(All_Data_Vec, All_Labels, Tr_Ind)]
	tr_data, tst_data = get_train_test_data(All_Data_reduced, Tr_Ind, Tst_Ind)

	return tr_data, tst_data




def lda(data, Labels_m, tr_i):
	"""
	Apply (supervised) LDA to the data matrix of modality-m as a dimensionality reduction.

	Parameters:
	----------
	data : vectorized and stacked graphs of modality-m with shape (N_m, nt*(nt-1)/2). The i-th row in "data" is
			the vectorized Graphs_m[i]. Note that the vectorization involves only the upper off-diagonal parts of matrices.

	Labels_m : label array for "Graphs_m" with length N_m

	tr_i : 1-D training index array for modality-m, it holds the indices of the training subjects in "data".


	Return:
	-------
	out : reduced data matrix with shape (N_m, 1).
	"""

	# data (single modality vectorized) :  (N_m, nt*(nt-1)/2)
	data2 = data[:,np.sum(data, axis=0)>0]

	train_data = data2[tr_i]
	train_labels = Labels_m[tr_i]

	clf = LDA(solver='svd', n_components=1)
	clf.fit(train_data, train_labels)
	train_data_new = clf.transform(train_data).ravel()

	tr_asd_data = train_data_new[train_labels==1]
	tr_nc_data  = train_data_new[train_labels==0]

	mu_asd = tr_asd_data.mean()
	mu_nc  = tr_nc_data.mean()

	middle = (mu_asd+mu_nc)/2
	distance = mu_asd-mu_nc

	data_new = (clf.transform(data2) - middle) / distance

	# data_new :  (N_m, 1)
	return data_new