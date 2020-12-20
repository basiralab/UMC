
import numpy as np
from DepthCoefficients import max_SP
from DB_Calculations import extract_DB_vectors
from GW_Calculations import extract_GW_vectors
from Alignment import Align_Multimodal_Connectomes as Align_MC
from LDA import dim_reduction_with_LDA
from ModalityCombination import get_train_test_data
from Classification import predict_test_labels, calculate_scores, Classify_Unimodal_Connectomes
from Plot import plot_scores, get_Names_Labels_Colors




def Extract_Connectomic_Features(All_Graphs, feat_method='DB', P=3, d=20, eta=0.9, stdScale_GW=True):
	"""
	Extract DB or GW feature vectors from graphs of M distinct modalities.

	Parameters:
	----------
	All_Graphs : list of M sets of graphs from M distinct modalities, each with shape (N_m, n_m, n_m)

	feat_method : 'DB' (default) or 'GW',
		Type of extracted connectomic features.
		If 'DB', depth-based vector representations are calculated from graphs using "subgraph expansions".
		If 'GW', structural GraphWave node embeddings are extracted using "heat diffusion wavelets".

	P : int (default: 3),
		Maximum depth level at which graphs are almost completely covered during subgraph expansions in DB-based alignment.
		Also, length of the feature vectors extracted from graphs. It is used if feat_method=='DB', otherwise discarded.

	d : int (default: 20),
		Number of (equally spaced) time points at which diffusion wavelets are evaluated in GW-based alignment.
		Length of the feature vectors extracted from graphs is 2*d. It is used if feat_method=='GW', otherwise discarded.

	eta : float (default: 0.9),
		Scaling parameter used in extraction of "GW" feature vectors. It is used if feat_method=='GW', otherwise discarded.
		It adjusts the radius of the local network neighborhoods to be discovered. If it is large, larger local neigborhoods
		are taken into account during feature extraction and vice versa. It must be selected in the range (eta_min, eta_max)
		which are determined by the second and the largest eigenvalues of the graph laplacian plus pre-determined coefficients.
		(For more, refer to the original paper: "Learning Structural Node Embeddings via Diffusion Wavelets")

	stdScale_GW : Bool (default: True),
		Whether or not to independently apply standard scaling to extracted features of each modality before graph alignment.
		It is used if feat_method=='GW', otherwise discarded. If "True", for any modality-m, P-dimensional feature vectors
		are scaled so that the distribution mean and standard deviation are 0 and 1, respectively. If "False", no changes
		made to the feature vectors and they are input to the alignment phase in their original forms.


	Return:
	-------
	out : list of M sets of feature matrices, each with shape (N_m, n_m, P)
	"""

	if feat_method == 'DB':
		maxSP = max_SP(All_Graphs)
		coeffs = maxSP / P
		All_Features = extract_DB_vectors(All_Graphs, P=P, depth_coeffs=coeffs)
		print('\n\n--> All depth-based vector representations are extracted.\n')

	elif feat_method == 'GW':
		All_Features = extract_GW_vectors(All_Graphs, d=d, eta=eta, stdScale=stdScale_GW)
		print('\n\n--> All GraphWave embeddings are extracted.\n')

	else:
		raise ValueError('Invalid feature method\n')

	return All_Features





def Classify_Aligned_Connectomes(All_Aligned_Graphs, All_Labels, Tr_Ind, Tst_Ind):
	"""
	Given training aligned connectomes and their labels, predict the labels of the testing aligned connectomes
	using linear SVM and calculate the performance scores (i.e., accuracy, sensitivity, specificity).

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
	out : array of 3 performance metrics (i.e., accuracy, sensitivity, specificity)
	"""

	Train_Data, Test_Data = dim_reduction_with_LDA(All_Aligned_Graphs, All_Labels, Tr_Ind, Tst_Ind)
	Train_Labels, Test_Labels = get_train_test_data(All_Labels, Tr_Ind, Tst_Ind)

	Test_Labels_Pred = predict_test_labels(Train_Data, Test_Data, Train_Labels)
	Scores = calculate_scores(Test_Labels, Test_Labels_Pred)

	return Scores  # Acc, Sens, Spec





def UCM(All_Graphs, All_Labels, nt=None, feat_method='DB', P=3, d=20, eta=0.9, stdScale_GW=True, corr='hard', Fold=5, seed=100):
	"""
	<<< MAIN FUNCTION OF THE PROPOSED METHOD ("UNIFIED CLASSIFICATION MODEL") >>>

	Classify (heterogeneous) multi-modal and multi-sized brain connectomes
	derived from "M" distinct neuroimaging modalities using graph alignment.

	Parameters:
	----------
	All_Graphs : list of M sets of graphs from M distinct modalities, each with shape (N_m, n_m, n_m)

	All_Labels : list of M label arrays, each with length N_m

	nt : int or None (default)
		number of nodes in template graphs, number of nodes in resulting aligned graphs,
		or number of cluster centroids during K-Means clustering. It should be lower than
		the size of the smallest graphs across different modalities. If not provided (None),
		it is set to int(min_size * 0.9), where "min_size" is the size of the smallest graphs.

	feat_method : 'DB' (default) or 'GW',
		Type of extracted connectomic features.
		If 'DB', depth-based vector representations are calculated from graphs using "subgraph expansions".
		If 'GW', structural GraphWave node embeddings are extracted using "heat diffusion wavelets".

	P : int (default: 3),
		Maximum depth level at which graphs are almost completely covered during subgraph expansions in DB-based alignment.
		Also, length of the feature vectors extracted from graphs. It is used if feat_method=='DB', otherwise discarded.

	d : int (default: 20),
		Number of (equally spaced) time points at which diffusion wavelets are evaluated in GW-based alignment.
		Length of the feature vectors extracted from graphs is 2*d. It is used if feat_method=='GW', otherwise discarded.

	eta : float (default: 0.9),
		Scaling parameter used in extraction of "GW" feature vectors. It is used if feat_method=='GW', otherwise discarded.
		It adjusts the radius of the local network neighborhoods to be discovered. If it is large, larger local neigborhoods
		are taken into account during feature extraction and vice versa. It must be selected in the range (eta_min, eta_max)
		which are determined by the second and the largest eigenvalues of the graph laplacian plus pre-determined coefficients.
		(For more, refer to the original paper: "Learning Structural Node Embeddings via Diffusion Wavelets")

	stdScale_GW : Bool (default: True),
		Whether or not to independently apply standard scaling to extracted features of each modality before graph alignment.
		It is used if feat_method=='GW', otherwise discarded. If "True", for any modality-m, P-dimensional feature vectors
		are scaled so that the distribution mean and standard deviation are 0 and 1, respectively. If "False", no changes
		made to the feature vectors and they are input to the alignment phase in their original forms.

	corr : 'hard' (default) or 'soft',
		correspondence strategy used in feature extraction phase. If 'hard', correspondence of a node
		in a graph is entirely reserved to the closest cluster centroid that is generated by KMeans clustering
		(correspondence: 1) and the correspondences to the other centroids are set to 0. If 'soft',
		unlike hard correspondence, correspondence of a node is distributed between cluster centroids based on
		their euclidean distances to the node of interest such that all correspondences add up to 1.
		In this case, correspondences are inversely proportional to pairwise distances.

	Fold : int (default: 5),
			number of fold to use in Cross-Validation

	seed : int (default: 100),
			seed value to anchor randomization for multiple runs


	Return:
	-------
	out : array of 3 performance metrics (i.e., accuracy, sensitivity, specificity)
	"""

	print('\nCONNECTOMIC DATASET')
	print('--------------------------')
	for i in range(len(All_Graphs)):
		print(f'Modality-{i+1}:  {All_Graphs[i].shape}')

	if nt is None:
		min_size = min(map(lambda G:G.shape[-1], All_Graphs))
		nt = int(min_size * 0.9)

	All_Features = Extract_Connectomic_Features(All_Graphs, feat_method, P, d, eta, stdScale_GW)

	generator = Align_MC(All_Graphs, All_Features, All_Labels, nt, feat_method=feat_method, corr=corr, Fold=Fold, seed=seed)

	MEAN_LIST_fold = []

	for f, (All_Aligned_Graphs, Tr_Ind, Tst_Ind) in enumerate(generator):

		MEAN = Classify_Aligned_Connectomes(All_Aligned_Graphs, All_Labels, Tr_Ind, Tst_Ind)
		MEAN_LIST_fold.append(MEAN)

	Scores = np.average(MEAN_LIST_fold, axis=0)
	Scores = np.round(Scores*100, 2)

	return Scores






def MultipleClassifications(All_Graphs, All_Labels, nt_list, K_list, P=3, d=20, eta=0.9, stdScale_GW=True, Fold=5, seed=100, Names_UC=None):

	def Classify_Diff_nts(All_Graphs, All_Features, All_Labels, nt_list, feat_method, corr, Fold, seed):

		SCORES = []

		for nt in nt_list:
			generator = Align_MC(All_Graphs, All_Features, All_Labels, nt, feat_method, corr, Fold, seed)

			MEAN_LIST_fold = []

			for f, (All_Aligned_Graphs, Tr_Ind, Tst_Ind) in enumerate(generator):
				MEAN = Classify_Aligned_Connectomes(All_Aligned_Graphs, All_Labels, Tr_Ind, Tst_Ind)
				MEAN_LIST_fold.append(MEAN)

			Scores = np.average(MEAN_LIST_fold, axis=0)
			Scores = np.round(Scores*100, 2)
			SCORES.append(Scores)

		return np.array(SCORES)

	# ---------------------------------------------------

	def Classify_UC(All_Graphs, All_Labels, K_list, Fold, seed):

		SCORES, ERRORS = [], []
		FS_list = ['SNF','Averaging']

		for Graphs_m, Labels_m in zip(All_Graphs, All_Labels):

			for FS_strategy in FS_list:

				Score_list_K = []

				for K in K_list:
					s = Classify_Unimodal_Connectomes(Graphs_m, Labels_m, FS_strategy, K, Fold, seed)
					Score_list_K.append(s)

				Scores = np.mean(Score_list_K, axis=0)
				Errors = np.std(Score_list_K, axis=0)

				SCORES.append(Scores)
				ERRORS.append(Errors)

		Scores_UC = np.array(SCORES).reshape(len(All_Graphs),len(FS_list),3)
		Errors_UC = np.array(ERRORS).reshape(len(All_Graphs),len(FS_list),3)

		return Scores_UC, Errors_UC

	# ---------------------------------------------------

	print('\nCONNECTOMIC DATASET')
	print('--------------------------')
	for i in range(len(All_Graphs)):
		print(f'Modality-{i+1}:  {All_Graphs[i].shape}')


	All_GW_Features = Extract_Connectomic_Features(All_Graphs, 'GW', P, d, eta, stdScale_GW)
	All_DB_Features = Extract_Connectomic_Features(All_Graphs, 'DB', P, d, eta, stdScale_GW)


	GW_hard = Classify_Diff_nts(All_Graphs, All_GW_Features, All_Labels, nt_list, 'GW', 'hard', Fold, seed)[:,np.newaxis,:]
	DB_hard = Classify_Diff_nts(All_Graphs, All_DB_Features, All_Labels, nt_list, 'DB', 'hard', Fold, seed)[:,np.newaxis,:]
	GW_soft = Classify_Diff_nts(All_Graphs, All_GW_Features, All_Labels, nt_list, 'GW', 'soft', Fold, seed)[:,np.newaxis,:]
	DB_soft = Classify_Diff_nts(All_Graphs, All_DB_Features, All_Labels, nt_list, 'DB', 'soft', Fold, seed)[:,np.newaxis,:]

	Err_GW_hard = np.zeros_like(GW_hard)
	Err_DB_hard = np.zeros_like(DB_hard)
	Err_GW_soft = np.zeros_like(GW_soft)
	Err_DB_soft = np.zeros_like(DB_soft)

	Scores_MC = np.array([np.concatenate((GW_hard,DB_hard),axis=-2), np.concatenate((GW_soft,DB_soft),axis=-2)])
	Errors_MC = np.array([np.concatenate((Err_GW_hard,Err_DB_hard),axis=-2), np.concatenate((Err_GW_soft,Err_DB_soft),axis=-2)])

	Scores_UC, Errors_UC = Classify_UC(All_Graphs, All_Labels, K_list, Fold, seed)

	Names_UC = list(map(lambda n:'$m_'+str(n)+'$',range(1,len(All_Graphs)+1))) if not Names_UC else Names_UC

	Params1, Params2 = get_Names_Labels_Colors(Names_UC)

	Params_MC = [Scores_MC, Errors_MC] + Params1
	Params_UC = [Scores_UC, Errors_UC] + Params2

	plot_scores(Params_MC, Params_UC, nt_list)