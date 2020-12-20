

import numpy as np
import networkx as nx




def max_SP_of_single_modality(Graphs_m):
	"""
	Find the maximum of shortest path for the m-th modality ("Graphs_m").
	The result is the average of maximum shortest paths found in each graph-i of "Graphs_m"

	Parameters:
	----------
	Graphs_m : Graphs of single modality (m) with shape (N_m, n_m, n_m)


	Return:
	-------
	out : float,
		average of the maximums of shortest paths calculated for each graph in "Graphs_m".
	"""

	max_list = []

	for graph_i in Graphs_m:

		G = nx.from_numpy_matrix(graph_i)
		SP = list(dict(nx.all_pairs_dijkstra_path_length(G)).values())
		SP = np.array([np.array(list(SP[i].values())) for i in range(len(SP))])
		Max_of_SP = np.max(SP[:,-1])

		max_list.append(Max_of_SP)

	avg_max = np.mean(max_list)

	return avg_max




def max_SP(All_Graphs):
	"""
	Find the maximums of shortest paths for all M modalities in "All_Graphs".
	These values will be used in "depth coefficient" calculations in DB based alignment.

	Parameters:
	----------
	All_Graphs : list, tuple, array. If it is list/tuple, it must contain M distinct modalities, 
				each has shape (N_m, n_m, n_m). If it is array, it must be a single modality with shape (N_m, n_m, n_m).

	Return:
	-------
	out : list of maximum shortest paths (floats), each for a corresponding modality, if "All_Graphs" is list/tuple of modalities 
		or single maximum shortest path (float), if "All_Graphs" consists of single modality.
	"""

	if isinstance(All_Graphs, np.ndarray):

		if len(All_Graphs.shape)==3:
			maxSP = max_SP_of_single_modality(All_Graphs)
			return maxSP
		else:
			raise ValueError('Input parameter "All_Graphs" is not in the correct format!!')


	elif isinstance(All_Graphs, (tuple, list)):

		if len(All_Graphs[0].shape)==3:
			maxSP = [max_SP_of_single_modality(Graphs_m) for Graphs_m in All_Graphs]
			return np.array(maxSP)
		else:
			raise ValueError('Input parameter "All_Graphs" is not in the correct format!!')

	else:
		raise ValueError('Input parameter "All_Graphs" is ambiguous.')