
import numpy as np



def get_train_test_data(All_Data, Tr_Ind, Tst_Ind):
	"""
	Get single training and testing data by combining multiple (M) data 
	with nearly the same shape in "All_Data". Data can be graphs or labels.

	Parameters:
	----------
	All_Data : list,
		list of M aligned graph or label arrays for M distinct modalities

	Tr_Ind : list of M 1-D index arrays,
		the m-th index array holds the indices of the training subjects in 
		All_Graphs[m] or All_Features[m] during particular fold

	Tst_Ind : list of M 1-D index arrays,
		the m-th index array holds the indices of the testing subjects in 
		All_Graphs[m] or All_Features[m] during particular fold

	Return:
	-------
	out : tuple of 2 arrays,
		each (training or testing) is 3-D combined aligned graph array
		or 1-D combined index array, depending on "All_Data"

	"""

	tr_data  = np.concatenate([data_m[ind] for data_m, ind in zip(All_Data, Tr_Ind)], axis=0)
	tst_data = np.concatenate([data_m[ind] for data_m, ind in zip(All_Data, Tst_Ind)], axis=0)

	return tr_data, tst_data