
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from snf.compute import snf




def predict_test_labels(Train_Data, Test_Data, Train_Labels):
	"""
	Apply learnt linear SVM model on connectomic testing data and predict their class labels

	Parameters:
	----------
	Train_Data : array,
		2-D training dataset. Its shape is (~(Fold-1)/Fold * (N_1 + ... + N_M), K) if a feature
		selection is used, (~(Fold-1)/Fold * (N_1 + ... + N_M), nt*(nt-1)/2) otherwise.

	Test_Data : array,
		2-D testing dataset. Its shape is (~1/Fold  * (N_1 + ... + N_M), K) if a feature
		selection is used, (~1/Fold  * (N_1 + ... + N_M), nt*(nt-1)/2) otherwise.

	Train_Labels : 1-D label array with length ~(Fold-1)/Fold * (N_1 + ... + N_M)


	Return:
	-------
	out : 1-D array of predicted test labels with length ~1/Fold  * (N_1 + ... + N_M)
	"""

	if len(Train_Data) != len(Train_Labels):
		raise ValueError('Number of training samples is not the same in Train_Data and Train_Labels!!')

	# clf : classifier trained
	clf = LinearSVC(tol=1e-6, max_iter=10**6)

	# clf.fit(X, y)
	clf.fit(Train_Data, Train_Labels)

	Predicted_Labels = clf.predict(Test_Data)

	return Predicted_Labels





def calculate_scores(GT, PRED):
	"""
	Calculate performance of the classifier as accuracy, sensitivity, specificity

	Parameters:
	----------
	GT : 1-D array, (actual) Ground Truth labels

	PRED : 1-D array, Predicted labels

	Return:
	-------
	out : array of 3 performance metrics (i.e., accuracy, sensitivity, specificity)
	"""

	tn, fp, fn, tp = confusion_matrix(GT,PRED).ravel()

	Acc = (tn+tp) / (tn+fp+fn+tp)
	Sens = tp / (tp+fn)
	Spec = tn / (tn+fp)

	return np.array([Acc, Sens, Spec])





def get_indices_of_Top_K_features(R, K):
	"""
	Get indices of K most discriminative connectomic features by analyzing
	the absolute difference (residual) between representative graphs of each classes

	Parameters:
	----------
	R : 2-D square Residual Matrix with shape (n_m, n_m)

	K : int,
		Number of most discriminative features set by user to be used in later classification.
		If "FS_strategy" is "SNF" or "Averaging", "K" should be a positive integer less than nt*(nt-1)/2.


	Return:
	-------
	out : indices of K most discriminative connectomic features in graphs (i.e, matrices)
	"""

	if (len(R.shape) != 2) or (R.shape[0] != R.shape[1]):
		raise ValueError('Residual Matrix is not square\n\n')

	n = len(R)

	T = np.sort(R[np.triu_indices(n,k=1)],axis=None)[::-1][K-1]

	Upper = np.zeros((n,n),int).astype(bool)
	Upper[np.triu_indices(n,k=1)] = True

	indices = np.argwhere((R >= T) & (Upper))

	values_of_indices = np.array([R[ind_1,ind_2] for ind_1,ind_2 in indices])

	indices_of_indices = np.argsort(values_of_indices)[::-1]

	return indices[indices_of_indices][:K]




def determine_features_to_select(Train_Graphs, Train_Labels, FS_strategy, K):
	"""
	Determine which connectomic features to select in graphs for feature selection

	Parameters:
	----------
	Train_Graphs : 3-D array with shape (~ (Fold-1)*N_m/Fold, n_m, n_m)

	Train_Labels : 1-D label array with length ~ (Fold-1)*N_m/Fold

	FS_strategy : "SNF" or "Averaging",
		Feature selection method to calculate the representative graphs from each class
		that are used to identify the most discriminative connectomic features.
		If 'SNF', the representative graphs are created with a graph fusion process
		(for more information, see "Similarity Netwok Fusion"). If "Averaging",
		the representative graphs are created by simply averaging graphs of each class.

	K : int,
		Number of most discriminative features set by user to be used in later classification.
		If "FS_strategy" is "SNF" or "Averaging", "K" should be a positive integer less than nt*(nt-1)/2.


	Return:
	-------
	out : indices of K most discriminative connection (i.e, graph edges) in graphs (i.e, matrices)
	"""

	TR_ASD = Train_Graphs[Train_Labels==1]
	TR_NC  = Train_Graphs[Train_Labels==0]

	if FS_strategy == 'SNF':

		for i in range(len(TR_ASD)):
			np.fill_diagonal(TR_ASD[i], 1e-10)

		for j in range(len(TR_NC)):
			np.fill_diagonal(TR_NC[j], 1e-10)

		ASD_Avg = snf(*TR_ASD, K = 1*TR_ASD.shape[1]//3, t=20, alpha=1.0)
		NC_Avg  = snf(*TR_NC, K = 1*TR_NC.shape[1]//3, t=20, alpha=1.0)

		for i in range(len(TR_ASD)):
			np.fill_diagonal(TR_ASD[i], 0.0)

		for j in range(len(TR_NC)):
			np.fill_diagonal(TR_NC[j], 0.0)


	elif FS_strategy == 'Averaging':
		ASD_Avg = np.average(TR_ASD, axis=0)
		NC_Avg = np.average(TR_NC, axis=0)


	else:
		raise ValueError(f'Invalid FS_strategy')


	Residual = np.abs(ASD_Avg-NC_Avg)
	indices = get_indices_of_Top_K_features(Residual, K)

	return indices





def classification(Train_Graphs, Test_Graphs, Train_Labels, Test_Labels, FS_strategy=None, K=None):
	"""
	Classify (regular) unimodal connectomes and get performance scores (i.e, accuracy, sensitivity, specificity)

	Parameters:
	----------
	Train_Graphs : 3-D array with shape (~ (Fold-1)*N_m/Fold, n_m, n_m)

	Test_Graphs : 3-D array with shape (~ N_m/Fold, n_m, n_m)

	Train_Labels : 1-D label array with length ~ (Fold-1)*N_m/Fold

	Test_Labels : 1-D label array with length ~ N_m/Fold

	FS_strategy : "SNF", "Averaging" or None (default),
		Feature selection method to calculate the representative graphs from each class
		that are used to identify the most discriminative connectomic features.
		If 'SNF', the representative graphs are created with a graph fusion process
		(for more information, see "Similarity Netwok Fusion"). If "Averaging",
		the representative graphs are created by simply averaging graphs of each class.
		If "None", no feature selection method to apply and all connectomic features of
		graphs are used in classification (all upper off-diagonal elements of graphs).

	K : int or None (default),
		Number of most discriminative features set by user to be used in later classification.
		If "FS_strategy" is "SNF" or "Averaging", "K" should be a positive integer
		less than nt*(nt-1)/2. If "FS_strategy" is "None", then K should be "None" too.


	Return:
	-------
	out : array of 3 performance metrics (i.e., accuracy, sensitivity, specificity)
	"""

	if Train_Graphs.shape[1:] == Test_Graphs.shape[1:]:
		n = Train_Graphs.shape[1]
	else:
		raise ValueError('Shapes of connectomes in "Train_Graphs" and "Test_Graphs" are different')

	if FS_strategy in ['SNF', 'Averaging']:

		if K == None:
			raise ValueError('Provide a proper K for feature selection (FS)')

		elif 2*K > n*(n-1):
			raise ValueError('K is too large for the current graph size (n)')

	elif (FS_strategy == None) and (K != None):
		raise ValueError('K can be used with a feature selection (FS)')

	else:
		raise ValueError('Invalid "FS_strategy", use "SNF" or "Averaging"')


	if FS_strategy != None:

		print(f'\n\nFS: {FS_strategy}')
		indices = determine_features_to_select(Train_Graphs, Train_Labels, FS_strategy=FS_strategy, K=K)

		TR_for_SVM = np.array([graph[indices[:,0],indices[:,1]] for graph in Train_Graphs])
		TST_for_SVM = np.array([graph[indices[:,0],indices[:,1]] for graph in Test_Graphs])

	else:
		TR_for_SVM = np.array([graph[np.triu_indices(n,1)] for graph in Train_Graphs])
		TST_for_SVM = np.array([graph[np.triu_indices(n,1)] for graph in Test_Graphs])


	Test_Labels_Pred = predict_test_labels(TR_for_SVM, TST_for_SVM, Train_Labels)
	Scores = calculate_scores(Test_Labels, Test_Labels_Pred)

	return Scores  # Acc, Sens, Spec





def Classify_Unimodal_Connectomes(Graphs_m, Labels_m, FS_strategy='SNF', K=30, Fold=5, seed=100):
	"""
	Classify a set of brain connectomes (i.e, graphs) derived from a single neuroimaging modality.
	Note that since the dataset is comprised of homogeneous and standard/same sized graphs (n_m, n_m),
	it does not require any "graph alignment" procedure before classification phase.

	Parameters:
	----------
	Graphs_m : set of N_m graphs of modality-m, each with shape (n_m, n_m)

	Labels_m : label array for "Graphs_m" with length N_m

	FS_strategy : "SNF" (default) or "Averaging",
		Feature selection method to calculate the representative graphs from each class
		that are used to identify the most discriminative connectomic features.
		If 'SNF', the representative graphs are created with a graph fusion process
		(for more information, see "Similarity Netwok Fusion"). If "Averaging",
		the representative graphs are created by simply averaging graphs of each class.

	K : int (default: 30),
		Number of most discriminative features set by user to be used in later classification.
		If "FS_strategy" is "SNF" or "Averaging", "K" should be a positive integer less than nt*(nt-1)/2.

	Fold : int (default: 5),
			number of fold to use in Cross-Validation

	seed : int (default: 100),
			seed value to anchor randomization for multiple runs


	Return:
	-------
	out : array of 3 performance metrics (i.e., accuracy, sensitivity, specificity)
	"""

	kf = KFold(n_splits=Fold, shuffle=True, random_state=seed)

	SCORES = []

	for f, (train_index, test_index) in enumerate(kf.split(Graphs_m)):
		# print(f'\nFOLD-{f+1}')

		Train_Graphs, Test_Graphs = Graphs_m[train_index], Graphs_m[test_index]
		Train_Labels, Test_Labels = Labels_m[train_index], Labels_m[test_index]

		Scores = classification(Train_Graphs, Test_Graphs, Train_Labels, Test_Labels, FS_strategy=FS_strategy, K=K)
		SCORES.append(Scores)

	Scores_Overall = np.average(SCORES, axis=0)
	Scores_Overall = np.round(Scores_Overall*100, 2)

	return Scores_Overall