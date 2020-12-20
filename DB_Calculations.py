
import numpy as np
import networkx as nx



def evaluate_Shannon_Entropy(Deg):
	"""
	Evaluate the Shannon Entropy of set of probabilities that are calculated from 
	the normalized degrees.

	Parameters:
	----------
	Deg : array of floats, (possibly unnormalized) degrees of a subgraph's nodes


	Return:
	-------
	out : float, Shannon Entropy of the normalized degrees. It cannot be less 
		  than 1 by the formula.
	"""

	nDeg = Deg/np.sum(Deg)

	return np.sum(nDeg * np.log2(1/nDeg))




def extract_p_th_element_of_delta(G, Adj, v, p, coeff):
	"""
	Compute the value in p-th element of a DB vector (𝛿)

	Parameters:
	----------
	G : network object of "networkx" library constructed from "Adj",
		used to form expanding subgraphs rooted at node-v.

	Adj : 2-dim array of floats with shape (n_m, n_m), which encodes relationships 
		between ROIs of a brain connectome. It is used to get the matrix 
		representations of expanding subgraphs rooted at node-v

	v : int, index of the node in "Adj" from which the subgraphs are expanded

	p : int, depth value in range [1-P] that indicates at which depth level 
		the expanding subgraph rooted at node-v is to be restricted. It is used 
		with the factor "coeff"

	coeff : float, depth coefficient or factor used to adjust the amount of 
		subgraph expansion. Subgraphs are constructed such that most of the shortest 
		paths in graph-i cannot exceed coeff*p, where (1 <= p <= P)


	Return:
	-------
	out : float, Shannon Entropy of the subgraph of G associated with 
		  the steady-state of random walk
	"""

	d = nx.single_source_dijkstra_path(G, source=v, cutoff=coeff*p)

	if len(d) == 1:
		return 1.0

	subgraph_nodes = sorted(list(d.keys()))

	Adj_subgraph = Adj.copy()

	ls = sorted(list(set(range(len(Adj_subgraph))) - set(subgraph_nodes)))

	Adj_subgraph[ls] = 0.0
	Adj_subgraph.T[ls] = 0.0

	return evaluate_Shannon_Entropy(Deg=np.sum(Adj_subgraph, axis=1)[subgraph_nodes])





def extract_DB_for_single_graph(graph_i, i, P=3, coeff=1.0):
	"""
	Extract 2-D DB matrix for graph-i

	Parameters:
	----------
	graph_i :  connectivity matrix for the i-th graph for which the DB matrix is extracted

	i : int, index of the current graph in modality-m

	P : int, default=3
		maximum depth level at which graphs are almost completely covered during subgraph expansions.

	coeff : float, default=1.0
		depth coefficient or factor used to adjust the amount of 
		subgraph expansion. Subgraphs are constructed such that most of the shortest 
		paths in graph-i cannot exceed coeff*p, where (1 <= p <= P)


	Return:
	-------
	out : DB matrix array with shape (n_m, P)
	"""

	Adj = graph_i # renaming the graph to "adjacency" matrix
	G = nx.from_numpy_matrix(Adj)

	DB_i = []

	for v in range(len(Adj)):

		delta_v = np.array([extract_p_th_element_of_delta(G, Adj, v, p, coeff) for p in range(1, P+1)])

		DB_i.append(delta_v)

	DB_i = np.array(DB_i)

	return DB_i





def extract_DBs_for_single_modality(Graphs_m, m, P=3, coeff=1.0):
	"""
	Extract 3-D DB matrix for the modality-m ("Graphs_m")

	Parameters:
	----------
	Graphs_m :  3-dim array with shape (N_m, n_m, n_m),
		graphs of modality-m

	m : int, index of the provided modality, in range [1-M]

	P : int, default=3
		maximum depth level at which graphs are almost completely covered during subgraph expansions.

	coeff : float, default=1.0
		depth coefficient or factor used to adjust the amount of 
		subgraph expansion. Subgraphs are constructed such that most of the shortest 
		paths in graph-i cannot exceed coeff*p, where (1 <= p <= P)


	Return:
	-------
	out : list of M DB matrices, each with shape (N_m, n_m, P) 
	"""

	N_m, n_m, _ = Graphs_m.shape

	print(f'DBs of modality-{m+1} being extracted ({N_m} graphs in total)')

	if P < 1 or isinstance(P, float):
		raise ValueError('P must be positive integer')

	DB_m = np.array([extract_DB_for_single_graph(graph_i, i, P=P, coeff=coeff) for graph_i, i in zip(Graphs_m, range(N_m))])

	DB_m_normalized = ((DB_m-1)/(np.log2(n_m)-1)) + 1

	return DB_m_normalized





def extract_DB_vectors(All_Graphs, P=3, depth_coeffs=0):
	"""
	Extract list of M 3-D DB matrices for all M modalities

	Parameters:
	----------
	All_Graphs : list/tuple of M modalities, each with shape (N_m, n_m, n_m)

	P : int, default=3
		maximum depth level at which graphs are almost completely covered during subgraph expansions.

	depth_coeffs : list/array, default=1.0
		list of depth coefficients or factors for M modalities. These coefficients are derived from
		the maximum shortest paths and used to adjust the amount of subgraph expansion.


	Return:
	-------
	out : list of M DB arrays, each with shape (N_m, n_m, P)
	"""

	M = len(All_Graphs)

	if not isinstance(depth_coeffs, (np.ndarray, list, tuple)):
		print('DB coeffs are selected as 1s')
		depth_coeffs = np.ones(M)

	print('\n')

	All_DBs = [extract_DBs_for_single_modality(Graphs_m, m, P, depth_coeff_m) for Graphs_m, depth_coeff_m, m in zip(All_Graphs, depth_coeffs, range(M))]

	return All_DBs