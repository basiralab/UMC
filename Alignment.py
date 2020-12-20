
import numpy as np
from sklearn.cluster import KMeans




def get_centroids(X, nt):

	"""
	Get nt well-representative centroids from stacked feature matrix X

	Parameters:
	----------
	X : 2-D concatenated feature array (Δ) with shape (D, p),
		D = (Fold-1)/Fold  * ((N_1*n_1) + ... + (N_m*n_m) + ... + (N_M*n_M))

	nt : int, number of nodes in template graphs, number of nodes in resulting aligned graphs
		or number of cluster centroids during K-Means clustering. It should be lower than
		the size of the smallest graphs across different modalities.


	Return:
	-------
	out : 2-D centroid array with shape (nt, p), each row is a cluster center
	"""

	clustering = KMeans(n_clusters=nt, n_init=10).fit(X)
	L = clustering.labels_
	Centroids = np.array([np.average(X[L == node_v], axis=0) for node_v in range(nt)])

	return Centroids





def obtain_Corr_for_single_modality(partial_feats_m, Centroids, corr='hard'):
	"""
	Obtain correspondence matrices for graphs of a particular modality

	Parameters:
	----------
	partial_feats_m : For DB-based alignment, 3-D DB array with shape (N_m, n_m, p),
		only the first p columns of original "DB_m" are used. For GW-based alignment,
		3-D GW array with shape (N_m, n_m, P), no partial use of features (i.e, p=P).
		All columns of "GW_m" are used. Note that the term "partial" is related only to DB-based alignment.

	Centroids : 2-D array with shape (nt, P), nt cluster centroids calculated from feature vectors

	corr : 'hard' (default) or 'soft',
		correspondence strategy used in feature extraction phase. If 'hard', correspondence of a node
		in a graph is entirely reserved to the closest cluster centroid that is generated by KMeans clustering
		(correspondence: 1) and the correspondences to the other centroids are set to 0. If 'soft',
		unlike hard correspondence, correspondence of a node is distributed between cluster centroids based on
		their euclidean distances to the node of interest such that all correspondences add up to 1.
		In this case, correspondences are inversely proportional to pairwise distances.


	Return:
	-------
	out : 3-D correspondence matrix with shape (N_m, n_m, nt) for modality-m
	"""

	N_m, n_m, p = partial_feats_m.shape
	nt = len(Centroids)

	Corr_m_shape = (N_m, n_m, nt)

	Dist_m = np.zeros(Corr_m_shape, dtype=float)

	# Calculating distance matrix first
	# for each graph in modality-m
	for i in range(N_m):

		# for each node in "Graph_i"
		for v in range(n_m):

			# delta :  each p-dimensional feature in "feat_i"
			delta = partial_feats_m[i,v]

			# for each template node (i.e, centroid) in T_p
			for vt in range(nt):
				centr = Centroids[vt]

				dist = np.linalg.norm(delta-centr)
				Dist_m[i,v,vt] = dist


	# Hard Correspondence
	if corr=='hard':
		# print('Hard correspondence selected')
		Corr_m = np.zeros(Corr_m_shape, dtype=float)  # "int" casted to "float"
		I = np.concatenate((np.array(list(np.ndindex(Corr_m_shape[:-1]))), Dist_m.argmin(axis=-1).reshape(-1,1)), axis=-1)
		Corr_m[I[:,0], I[:,1], I[:,2]] = 1

	# Soft Correspondence
	elif corr=='soft':
		# print('Soft correspondence selected')
		Sim = np.exp(-Dist_m/p)	# similarity
		Corr_m = Sim / Sim.sum(axis=-1, keepdims=True)	# normalized similarity, elements in each row should add up to 1

	else:
		raise ValueError('Invalid correspondence type (corr), choose "hard" or "soft"\n')

	return Corr_m






def obtain_all_Corr(All_Features, Centroids, corr='hard'):
	"""
	Obtain all correspondence matrices for M modalities

	Parameters:
	----------
	All_Features : list of M sets of feature matrices, each with shape (N_m, n_m, P)

	Centroids : 2-D array with shape (nt, P), nt cluster centroids calculated from feature vectors

	corr : 'hard' (default) or 'soft',
		correspondence strategy used in feature extraction phase. If 'hard', correspondence of a node
		in a graph is entirely reserved to the closest cluster centroid that is generated by KMeans clustering
		(correspondence: 1) and the correspondences to the other centroids are set to 0. If 'soft',
		unlike hard correspondence, correspondence of a node is distributed between cluster centroids based on
		their euclidean distances to the node of interest such that all correspondences add up to 1.
		In this case, correspondences are inversely proportional to pairwise distances.


	Return:
	-------
	out : list of M 3-D correspondence matrix arrays for all modalities, each with shape (N_m, n_m, nt)
	"""

	M = len(All_Features)
	nt, p = Centroids.shape

	All_Corr = [obtain_Corr_for_single_modality(Features_m[:,:,:p], Centroids, corr) for Features_m in All_Features]

	return All_Corr





def get_aligned_graphs(All_Graphs, All_Corr):
	"""
	Align multimodal connectomes using correspondence matrices of original graphs

	Parameters:
	----------
	All_Graphs : list of M sets of graphs from M distinct modalities, each with shape (N_m, n_m, n_m)

	All_Corr : list of M sets of correspondence matrices for all M modalities, each with shape (N_m, n_m, nt)


	Return:
	-------
	out : list of M sets of aligned graphs, each with shape (N_m, nt, nt)
	"""

	All_Aligned_Graphs = []

	for Graphs_m, Corr_m in zip(All_Graphs, All_Corr):

		n_m, nt = Corr_m.shape[-2:]

		reduction_ratio = (nt/n_m)**2
		All_Aligned_Graphs.append( np.array([corr_i.T @ graph_i @ corr_i for graph_i, corr_i in zip(Graphs_m, Corr_m)]) * reduction_ratio )

	return All_Aligned_Graphs





def average_aligned_graphs_across_p(List_p, M, P):
	"""
	Average a set of aligned graphs that are created with respect to 
	different p values. p is in range [1-P]

	Parameters:
	----------
	List_p : list of list of all aligned modalities to be averaged

	M : int, total number of modalities

	P : int, default=3
		maximum depth level at which graphs are almost completely covered during subgraph expansions.

	
	Return:
	-------
	out : list of all averaged aligned modalities across p
	"""

	L = []

	for m in range(M):
		L.append(np.average([List_p[p][m] for p in range(P)], axis=0))

	return L






def alignment_to_template(All_Graphs, All_Features, Tr_Ind, nt, feat_method='DB', corr='hard', seed=100):
	"""
	Get aligned multimodal connectomes via shared template

	Parameters:
	----------
	All_Graphs : list of M sets of graphs from M distinct modalities, each with shape (N_m, n_m, n_m)

	All_Features : list of M sets of feature matrices, each with shape (N_m, n_m, P)

	Tr_Ind : list of M 1-D index arrays,
		the m-th index array holds the indices of the training subjects in 
		All_Graphs[m] or All_Features[m] during particular fold

	nt : int,
		number of nodes in template graphs, number of nodes in resulting aligned graphs,
		or number of cluster centroids during K-Means clustering. It should be lower than
		the size of the smallest graphs across different modalities.

	feat_method : 'DB' (default) or 'GW',
		Type of extracted connectomic features.
		If 'DB', depth-based vector representations are calculated from graphs using "subgraph expansions".
		If 'GW', structural GraphWave node embeddings are extracted using "heat diffusion wavelets".

	corr : 'hard' (default) or 'soft',
		correspondence strategy used in feature extraction phase. If 'hard', correspondence of a node
		in a graph is entirely reserved to the closest cluster centroid that is generated by KMeans clustering
		(correspondence: 1) and the correspondences to the other centroids are set to 0. If 'soft',
		unlike hard correspondence, correspondence of a node is distributed between cluster centroids based on
		their euclidean distances to the node of interest such that all correspondences add up to 1.
		In this case, correspondences are inversely proportional to pairwise distances.

	seed : int (default: 100),
			seed value to anchor randomization for multiple runs


	Return:
	-------
	out : list of M sets of aligned graphs, each with shape (N_m, nt, nt)
	"""

	if len(All_Graphs) == len(All_Features):
		M = len(All_Graphs)
		P = All_Features[0].shape[-1]
	else:
		raise ValueError('Number of Modalities is different in input parameters (All_Graphs, All_Features, All_Labels)')


	# Delta : tr_feats (all features concatenated vertically)
	Delta = np.concatenate([Features_m[ind].reshape(-1,P) for Features_m, ind in zip(All_Features, Tr_Ind)], axis=0)

	if feat_method == 'GW':

		GW_Centroids = get_centroids(X=Delta, nt=nt)
		All_Corr = obtain_all_Corr(All_Features, GW_Centroids, corr)
		All_Aligned_Graphs = get_aligned_graphs(All_Graphs, All_Corr)

	elif feat_method == 'DB':

		List_p = []

		for p in range(1, P+1):
			DB_Centroids = get_centroids(X=Delta[:,:p], nt=nt)
			All_Corr = obtain_all_Corr(All_Features, DB_Centroids, corr)
			Aligned_Graphs_p = get_aligned_graphs(All_Graphs, All_Corr)

			###  <---IF NODE REARRANGEMENT IS NEEDED AFTER ALIGNMENT, HERE--->  ###
			List_p.append(Aligned_Graphs_p)

		All_Aligned_Graphs = average_aligned_graphs_across_p(List_p, M, P)

	else:
		raise ValueError('Invalid "feat_method", it can be "GW" or "DB"\n')

	return All_Aligned_Graphs





def Align_Multimodal_Connectomes(All_Graphs, All_Features, All_Labels, nt, feat_method='DB', corr='hard', Fold=5, seed=100):
	"""
	Align multimodal connectomes using DB or GW-based graph alignment to get fixed-sized connectomes

	Parameters:
	----------
	All_Graphs : list of M sets of graphs from M distinct modalities, each with shape (N_m, n_m, n_m)

	All_Features : list of M sets of feature matrices, each with shape (N_m, n_m, P)

	All_Labels : list of M label arrays, each with length N_m

	nt : int,
		number of nodes in template graphs, number of nodes in resulting aligned graphs,
		or number of cluster centroids during K-Means clustering. It should be lower than
		the size of the smallest graphs across different modalities.

	feat_method : 'DB' (default) or 'GW',
		Type of extracted connectomic features.
		If 'DB', depth-based vector representations are calculated from graphs using "subgraph expansions".
		If 'GW', structural GraphWave node embeddings are extracted using "heat diffusion wavelets".

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
	out : list of M sets of aligned graphs, training indices and testing indices
	"""

	if len(All_Graphs) == len(All_Features) == len(All_Labels):
		M = len(All_Graphs)
		P = All_Features[0].shape[-1]
	else:
		raise ValueError('Number of Modalities is different in input parameters (All_Graphs, All_Features, All_Labels)')


	# N_list :			(M,)
	# Indices :			(M, ...)
	# Train_Indices :	(M, Fold, ...)
	# Test_Indices :	(M, Fold, ...)
	N_list, n_list, Indices, Train_Indices, Test_Indices = [], [], [], [], []

	# iterating over M modalities for index operations (train/test)
	for m in range(M):

		if not (len(All_Graphs[m]) == len(All_Features[m]) == len(All_Labels[m])):
			raise ValueError(f'Sample sizes in modality-{m+1} are different in input parameters.')

		N_m, n_m, _ = All_Features[m].shape

		N_list.append(N_m)
		n_list.append(n_m)

		indices = np.arange(N_m)

		np.random.seed(seed)
		np.random.shuffle(indices)
		Indices.append(indices)

		test_indices = np.array_split(indices, Fold)
		Test_Indices.append(test_indices)

		train_indices = [indices[np.in1d(indices, ind, invert=True)]  for ind in test_indices]
		Train_Indices.append(train_indices)

	if nt >= min(n_list):
		raise ValueError("nt value should be smaller than the minimum connectomic resolution (n_m) of all modalities")

	print("")

	for f in range(Fold):

		print(f"{feat_method} | nt={nt} | {corr} correspondence | Fold-{f+1}")

		# Tr_Ind  :	 (M, ...)
		# Tst_Ind :	 (M, ...)
		Tr_Ind  = [Train_Indices[m][f] for m in range(M)]
		Tst_Ind = [Test_Indices[m][f] for m in range(M)]

		# All_Aligned_Graphs :  [(N_1, nt, nt),  (N_2, nt, nt),  ...,  (N_M, nt, nt)]
		All_Aligned_Graphs = alignment_to_template(All_Graphs, All_Features, Tr_Ind, nt, feat_method, corr=corr, seed=seed)

		yield All_Aligned_Graphs, Tr_Ind, Tst_Ind