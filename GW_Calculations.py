
import numpy as np
from scipy.sparse import csgraph
from sklearn.preprocessing import StandardScaler



def extract_GWs_for_single_modality(Graphs_m, d=20, eta=0.9, stdScale=True):
	"""
	Extract 3-D GW (GraphWave) feature matrix for modality-m.

	Parameters:
	----------
	Graphs_m : set of N_m graphs of modality-m, each with shape (n_m, n_m)

	d : int (default: 20),
		Number of (equally spaced) time points at which diffusion wavelets are evaluated in GW-based alignment.
		Length of the feature vectors extracted from graphs is 2*d. It is used if feat_method=='GW', otherwise discarded.

	eta : float (default: 0.9),
		Scaling parameter used in extraction of "GW" feature vectors. It is used if feat_method=='GW', otherwise discarded.
		It adjusts the radius of the local network neighborhoods to be discovered. If it is large, larger local neigborhoods
		are taken into account during feature extraction and vice versa. It must be selected in the range (eta_min, eta_max)
		which are determined by the second and the largest eigenvalues of the graph laplacian plus pre-determined coefficients.
		(For more, refer to the original paper: "Learning Structural Node Embeddings via Diffusion Wavelets")

	stdScale : Bool (default: True),
		Whether or not to independently apply standard scaling to extracted features of each modality before graph alignment.
		It is used if feat_method=='GW', otherwise discarded. If "True", for any modality-m, P-dimensional feature vectors
		are scaled so that the distribution mean and standard deviation are 0 and 1, respectively. If "False", no changes
		made to the feature vectors and they are input to the alignment phase in their original forms.


	Return:
	-------
	out : 3-D GW (GraphWave) matrix with shape (N_m, n_m, 2*d)
	"""

	CHI_LIST = []

	for graph_i in Graphs_m:

		Lapl = csgraph.laplacian(graph_i, normed=False)

		eigVals, U = np.linalg.eig(Lapl)
		eigVals = np.real(eigVals)

		sec_eig, largest_eig = np.sort(eigVals)[[1,-1]]
		s = -np.log(eta)/np.sqrt(sec_eig*largest_eig)

		Lambda = np.diag(eigVals)
		Lambda2 = np.diag(np.exp(-s*eigVals))
		Psi = np.real(U @ Lambda2 @ U.T)

		# Emb :  shape(n,d)
		Emb = np.array([np.mean(np.exp(1j*t*Psi), axis=0) for t in range(1,d+1)]).T

		# chi :  shape(n,2d)
		chi = np.array([np.concatenate([np.real(vec).reshape(-1,1), np.imag(vec).reshape(-1,1)], axis=1).ravel() for vec in Emb])

		CHI_LIST += [chi]

	CHI = np.array(CHI_LIST)

	if stdScale:

		orig_shape = CHI.shape
		CHI = CHI.reshape(-1, orig_shape[-1])
		scaler = StandardScaler()
		CHI = scaler.fit_transform(CHI)
		CHI = CHI.reshape(orig_shape)

	return CHI




def extract_GW_vectors(All_Graphs, d=20, eta=0.9, stdScale=True):
	"""
	Extract list of M 3-D GW (GraphWave) feature matrices for all M modalities

	Parameters:
	----------
	All_Graphs : list of M sets of graphs from M distinct modalities, each with shape (N_m, n_m, n_m)

	d : int (default: 20),
		Number of (equally spaced) time points at which diffusion wavelets are evaluated in GW-based alignment.
		Length of the feature vectors extracted from graphs is 2*d. It is used if feat_method=='GW', otherwise discarded.

	eta : float (default: 0.9),
		Scaling parameter used in extraction of "GW" feature vectors. It is used if feat_method=='GW', otherwise discarded.
		It adjusts the radius of the local network neighborhoods to be discovered. If it is large, larger local neigborhoods
		are taken into account during feature extraction and vice versa. It must be selected in the range (eta_min, eta_max)
		which are determined by the second and the largest eigenvalues of the graph laplacian plus pre-determined coefficients.
		(For more, refer to the original paper: "Learning Structural Node Embeddings via Diffusion Wavelets")

	stdScale : Bool (default: True),
		Whether or not to independently apply standard scaling to extracted features of each modality before graph alignment.
		It is used if feat_method=='GW', otherwise discarded. If "True", for any modality-m, P-dimensional feature vectors
		are scaled so that the distribution mean and standard deviation are 0 and 1, respectively. If "False", no changes
		made to the feature vectors and they are input to the alignment phase in their original forms.


	Return:
	-------
	out : list of M GW arrays, each with shape (N_m, n_m, 2*d)
	"""

	All_GWs = [extract_GWs_for_single_modality(Graphs_m, d=d, eta=eta, stdScale=stdScale) for m, Graphs_m in enumerate(All_Graphs)]
	return All_GWs