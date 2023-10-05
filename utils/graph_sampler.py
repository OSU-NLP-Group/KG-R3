import os
import math
import struct
import logging
import random
import pickle as pkl
import pdb
from tqdm import tqdm
import lmdb
import multiprocessing as mp
import numpy as np
import scipy.io as sio
import scipy.sparse as ssp
import sys
import torch
from scipy.special import softmax
from utils.dgl_utils import _bfs_relational
from utils.graph_utils import incidence_matrix, remove_nodes, ssp_to_torch, serialize, deserialize, get_edge_count, diameter, radius
import networkx as nx
from utils.gen_utils import sample_subgraph_with_central_node_bfs, sample_subgraph_with_central_node_onehop

# @profile
def links2subgraphs(A, adj_list, graphs, db_path, triples_dataset, triples, mid2title, params):
	'''
	extract enclosing subgraphs, write map mode + named dbs
	'''
	print('inside links2subgraphs')

	subgraph_sizes = []
	enc_ratios = []
	num_pruned_nodes = []

	BYTES_PER_DATUM = get_average_subgraph_size(100, list(graphs.values())[0]['pos'], A, adj_list, triples_dataset, triples, mid2title, params) * 1.5
	links_length = 0
	for split_name, split in graphs.items():
		links_length += (len(split['pos'])) * 2
	map_size = links_length * BYTES_PER_DATUM

	# print('map_size = {}'.format(map_size))
	map_size = map_size*100

	neigh_dict = None
	if not params.prune_subgraph:
		neigh_dict = {}
		A_incidence = incidence_matrix(A)
		A_incidence += A_incidence.T
		num_entities = len(triples_dataset.entity2id)

		for idx in tqdm(range(num_entities)):
			nei_tuples = get_neighbor_nodes(set([idx]), A_incidence, adj_list, params.hop, params.max_nodes_per_hop, params.n_neigh_per_node, triples_dataset, mid2title)
			neigh_dict[idx] = set([x[1] for x in nei_tuples])

	# print('neigh_dict = {}'.format(neigh_dict))
	# sys.exit(0)
	env = lmdb.open(db_path, map_size=map_size, max_dbs=6)

	def extraction_helper(A, adj_list, links, split_env):

		with env.begin(write=True, db=split_env) as txn:
			txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)), byteorder='little'))
		
		with mp.Pool(processes=params.num_workers, initializer=intialize_worker, initargs=(A, adj_list, neigh_dict, triples_dataset, triples, mid2title, params)) as p:
			args_ = zip(range(len(links)), links)

			for (str_id, datum) in tqdm(p.imap(extract_save_subgraph, args_), total=len(links)):
				# print('str_id = {}, datum = {}'.format(str_id, datum))
				subgraph_sizes.append(datum['subgraph_size'])
				enc_ratios.append(datum['enc_ratio'])
				num_pruned_nodes.append(datum['num_pruned_nodes'])

				with env.begin(write=True, db=split_env) as txn:
					txn.put(str_id, serialize(datum))

	def extraction_helper_bfs(A, adj_list, links, split_env):

		with env.begin(write=True, db=split_env) as txn:
			txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)*2), byteorder='little'))
		
		with mp.Pool(processes=params.num_workers, initializer=intialize_worker, initargs=(A, adj_list, neigh_dict, triples_dataset, triples, mid2title, params)) as p:
			args_ = zip(range(len(links)*2), np.tile(links, (2,1)), [len(links)]*2*len(links))

			for (str_id, datum) in tqdm(p.imap(extract_save_subgraph_bfs, args_), total=len(links)*2):
				# print('str_id = {}, datum = {}'.format(str_id, datum))

				with env.begin(write=True, db=split_env) as txn:
					txn.put(str_id, serialize(datum))

	def extraction_helper_onehop(A, adj_list, links, split_env):

		with env.begin(write=True, db=split_env) as txn:
			txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)*2), byteorder='little'))
		
		with mp.Pool(processes=params.num_workers, initializer=intialize_worker, initargs=(A, adj_list, neigh_dict, triples_dataset, triples, mid2title, params)) as p:
			args_ = zip(range(len(links)*2), np.tile(links, (2,1)), [len(links)]*2*len(links))

			for (str_id, datum) in tqdm(p.imap(extract_save_subgraph_onehop, args_), total=len(links)*2):
				# print('str_id = {}, datum = {}'.format(str_id, datum))

				with env.begin(write=True, db=split_env) as txn:
					txn.put(str_id, serialize(datum))
		
	
	for split_name, split in graphs.items():
		print('split_name = {}'.format(split_name))
		logging.info(f"Extracting enclosing subgraphs for positive links in {split_name} set")
		print(f"Extracting enclosing subgraphs for positive links in {split_name} set")
		labels = np.ones(len(split['pos']))
		db_name_pos = split_name + '_pos'
		split_env = env.open_db(db_name_pos.encode())
		if params.sampling_type == 'khop':
			extraction_helper(A, adj_list, split['pos'], split_env)
		elif params.sampling_type == 'bfs':
			extraction_helper_bfs(A, adj_list, split['pos'], split_env)
		elif params.sampling_type == 'onehop':
			extraction_helper_onehop(A, adj_list, split['pos'], split_env)

	if params.sampling_type == 'khop':
		with env.begin(write=True) as txn:
			txn.put('avg_subgraph_size'.encode(), struct.pack('f', float(np.mean(subgraph_sizes))))
			txn.put('min_subgraph_size'.encode(), struct.pack('f', float(np.min(subgraph_sizes))))
			txn.put('max_subgraph_size'.encode(), struct.pack('f', float(np.max(subgraph_sizes))))
			txn.put('std_subgraph_size'.encode(), struct.pack('f', float(np.std(subgraph_sizes))))

			txn.put('avg_enc_ratio'.encode(), struct.pack('f', float(np.mean(enc_ratios))))
			txn.put('min_enc_ratio'.encode(), struct.pack('f', float(np.min(enc_ratios))))
			txn.put('max_enc_ratio'.encode(), struct.pack('f', float(np.max(enc_ratios))))
			txn.put('std_enc_ratio'.encode(), struct.pack('f', float(np.std(enc_ratios))))

			txn.put('avg_num_pruned_nodes'.encode(), struct.pack('f', float(np.mean(num_pruned_nodes))))
			txn.put('min_num_pruned_nodes'.encode(), struct.pack('f', float(np.min(num_pruned_nodes))))
			txn.put('max_num_pruned_nodes'.encode(), struct.pack('f', float(np.max(num_pruned_nodes))))
			txn.put('std_num_pruned_nodes'.encode(), struct.pack('f', float(np.std(num_pruned_nodes))))


def get_average_subgraph_size(sample_size, links, A, adj_list, triples_dataset, triples, mid2title, params):
	print('inside get_average_subgraph_size')
	total_size = 0
	# print(np.random.choice(len(links), sample_size))
	for (n1, r_label, n2) in tqdm(links[np.random.choice(len(links), sample_size)]):
		# print('flag 1')

		if params.sampling_type == 'khop':
			nodes, edges, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A, adj_list, params.hop, params.enclosing_sub_graph, params.max_nodes_per_hop, params.max_nodes_int, params.prune_subgraph, triples_dataset, triples, mid2title, n_neigh_per_node=params.n_neigh_per_node, n_max_tries=params.n_max_tries)
			datum = {'n1' : n1, 'n2' : n2, 'r_label': r_label, 'nodes': nodes, 'edges': edges, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
		elif params.sampling_type == 'bfs':
			num_nodes = len(triples_dataset.entity2id)
			train_data = triples_dataset.train # list of training triples
			if random.random()>0.5: # random sampling to get estimate of subgraph size
				central_node = n1
			else:
				central_node = n2

			edges = sample_subgraph_with_central_node_bfs(adj_list, central_node, num_nodes, len(train_data), params.sample_size, params.neigh_size).tolist()
			edges = [tuple(train_data[i]) for i in edges]

			heads = [x[0] for x in edges]
			tails = [x[2] for x in edges]
			nodes = set(heads + tails)
			subgraph_size = len(nodes)
			datum = {'n1' : n1, 'n2' : n2, 'r_label': r_label, 'nodes': nodes, 'edges': edges}
		elif params.sampling_type == 'onehop':
			num_nodes = len(triples_dataset.entity2id)
			train_data = triples_dataset.train # list of training triples
			if random.random()>0.5: # random sampling to get estimate of subgraph size
				central_node = n1
			else:
				central_node = n2

			edges = sample_subgraph_with_central_node_onehop(adj_list, central_node, num_nodes, len(train_data), params.sample_size).tolist()
			edges = [tuple(train_data[i]) for i in edges]

			heads = [x[0] for x in edges]
			tails = [x[2] for x in edges]
			nodes = set(heads + tails)
			subgraph_size = len(nodes)
			datum = {'n1' : n1, 'n2' : n2, 'r_label': r_label, 'nodes': nodes, 'edges': edges}

		total_size += len(serialize(datum))
	return total_size / sample_size


def intialize_worker(A, adj_list, neigh_dict, triples_dataset, triples, mid2title, params):
	# print('inside intialize_worker')
	# print('flag 2 adj_list[0] = {}'.format(adj_list[0]))
	global A_, adj_list_, neigh_dict_, params_, triples_dataset_, triples_, mid2title_
	A_, adj_list_, neigh_dict_, params_, triples_dataset_, triples_, mid2title_ = A, adj_list, neigh_dict, params, triples_dataset, triples, mid2title

def intialize_worker_neigh_dict(A_incidence, adj_list, hop, max_nodes_per_hop, n_neigh_per_node, triples_dataset, mid2title):
	# print('inside intialize_worker')
	# print('flag 2 adj_list[0] = {}'.format(adj_list[0]))
	global A_incidence_, adj_list_, hop_, max_nodes_per_hop_, n_neigh_per_node_, triples_dataset_, mid2title_
	A_incidence_, adj_list_, hop_, max_nodes_per_hop_, n_neigh_per_node_, triples_dataset_, mid2title_ = A_incidence, adj_list, hop, max_nodes_per_hop, n_neigh_per_node, triples_dataset, mid2title

# @profile
def extract_save_subgraph(args_):
	# print(args_)
	# print('inside process {}'.format(mp.current_process()))
	# print('flag 3 adj_list[0] = {}'.format(adj_list_[0]))
	idx, (n1, r_label, n2) = args_
	# print('idx = {}'.format(idx))

	if params_.prune_subgraph:
		nodes, edges, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A_, adj_list_, params_.hop, params_.enclosing_sub_graph, params_.max_nodes_per_hop, params_.max_nodes_int, params_.prune_subgraph, triples_dataset_, triples_, mid2title_, n_neigh_per_node=params_.n_neigh_per_node, n_max_tries=params_.n_max_tries)
	else:
		nodes, edges, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling_wo_pruning((n1, n2), r_label, neigh_dict_, adj_list_, params_.hop, params_.enclosing_sub_graph, params_.max_nodes_per_hop, params_.max_nodes_int, params_.prune_subgraph, triples_dataset_, triples_, mid2title_, n_neigh_per_node=params_.n_neigh_per_node, n_max_tries=params_.n_max_tries)

	datum = {'n1' : n1, 'n2' : n2, 'r_label': r_label, 'nodes': nodes, 'edges': edges, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
	str_id = '{:08}'.format(idx).encode('ascii')
	# print(datum)
	return (str_id, datum)

def extract_save_subgraph_bfs(args_):
	idx, (n1, r_label, n2), links_length = args_

	# print('idx = {}, links_length = {}'.format(idx, links_length))

	if idx < links_length: # head is corrupted
		# print('head is corrupted')
		central_node = n2
	else: # tail is corrupted
		# print('tail is corrupted')
		central_node = n1

	num_nodes = len(triples_dataset_.entity2id)
	train_data = triples_dataset_.train # list of training triples

	edges = sample_subgraph_with_central_node_bfs(adj_list_, central_node, num_nodes, len(train_data), params_.sample_size, params_.neigh_size).tolist()
	edges = [tuple(train_data[i]) for i in edges]

	# print('central_node = {}, edges = {}'.format(central_node, edges))

	heads = [x[0] for x in edges]
	tails = [x[2] for x in edges]
	nodes = set(heads + tails)
	subgraph_size = len(nodes)

	if central_node not in nodes:
		print('central_node = {}'.format(central_node))
		print('edges = {}'.format(edges))
		print('error!!!')

	datum = {'n1' : n1, 'n2' : n2, 'r_label': r_label, 'nodes': nodes, 'edges': edges}
	str_id = '{:08}'.format(idx).encode('ascii')

	return (str_id, datum)

def extract_save_subgraph_onehop(args_):
	idx, (n1, r_label, n2), links_length = args_

	# print('idx = {}, links_length = {}'.format(idx, links_length))

	if idx < links_length: # head is corrupted
		# print('head is corrupted')
		central_node = n2
	else: # tail is corrupted
		# print('tail is corrupted')
		central_node = n1

	num_nodes = len(triples_dataset_.entity2id)
	train_data = triples_dataset_.train # list of training triples

	edges = sample_subgraph_with_central_node_onehop(adj_list_, central_node, num_nodes, len(train_data), params_.sample_size).tolist()
	edges = [tuple(train_data[i]) for i in edges]

	# print('central_node = {}, edges = {}'.format(central_node, edges))

	heads = [x[0] for x in edges]
	tails = [x[2] for x in edges]
	nodes = set(heads + tails)
	subgraph_size = len(nodes)

	if central_node not in nodes:
		print('central_node = {}'.format(central_node))
		print('edges = {}'.format(edges))
		print('error!!!')

	datum = {'n1' : n1, 'n2' : n2, 'r_label': r_label, 'nodes': nodes, 'edges': edges}
	str_id = '{:08}'.format(idx).encode('ascii')

	return (str_id, datum)

def get_neighbor_nodes(roots, adj, adj_list, h=1, max_nodes_per_hop=None, n_neigh_per_node=None, triples_dataset=None, mid2title=None):
	bfs_generator = _bfs_relational(adj, adj_list, roots, max_nodes_per_hop, n_neigh_per_node, triples_dataset, mid2title)
	lvls = list()
	for _ in range(h):
		try:
			lvls.append(next(bfs_generator))
		except StopIteration:
			pass
	return set().union(*lvls)

def get_neighbor_nodes_multiproc(roots):
	bfs_generator = _bfs_relational(A_incidence_, adj_list_, roots, max_nodes_per_hop_, n_neigh_per_node_, triples_dataset_, mid2title_)
	lvls = list()
	for _ in range(hop_):
		try:
			lvls.append(next(bfs_generator))
		except StopIteration:
			pass

	start_idx = list(roots)[0]
	return start_idx, set().union(*lvls)

# @profile
def subgraph_extraction_labeling(ind, rel, A_list, adj_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, max_nodes_int=None, prune_subgraph=None, triples_dataset=None, triples=None, mid2title=None, max_node_label_value=None, n_neigh_per_node=None, n_max_tries=20):
	# TODO: remove use of adj_list

	# extract the h-hop enclosing subgraphs around link 'ind'
	# print('inside subgraph_extraction_labeling')
	A_incidence = incidence_matrix(A_list)
	# print('A_incidence = {}'.format(A_incidence))

	A_incidence += A_incidence.T
	# print('A_incidence = {}'.format(A_incidence.shape))

	# root1_nei contains (edge_id, node_id pairs)

	# print('ind[0] = {}'.format(mid2title.get(triples_dataset.id2entity[ind[0]])))
	# print('ind[1] = {}'.format(mid2title.get(triples_dataset.id2entity[ind[1]])))

	pruned_subgraph_edges = []
	n_tries = 0

	# print('adj_list = {}'.format(adj_list))
	
	while len(pruned_subgraph_edges)<3 and n_tries < n_max_tries: 
		root1_nei_tuples = get_neighbor_nodes(set([ind[0]]), A_incidence, adj_list, h, max_nodes_per_hop, n_neigh_per_node, triples_dataset, mid2title)
		root2_nei_tuples = get_neighbor_nodes(set([ind[1]]), A_incidence, adj_list, h, max_nodes_per_hop, n_neigh_per_node, triples_dataset, mid2title)

		# print('root1_nei_tuples = {}'.format(root1_nei_tuples))
		# print('root2_nei_tuples = {}'.format(root2_nei_tuples))

		root1_nei = set([x[1] for x in root1_nei_tuples])
		root2_nei = set([x[1] for x in root2_nei_tuples])

		subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
		subgraph_nei_nodes_un = root1_nei.union(root2_nei)
		# subgraph_nei_nodes_int = root2_nei

		# TODO: add parameter for this
		
		if max_nodes_int and len(subgraph_nei_nodes_int)>max_nodes_int:
			subgraph_nei_nodes_int = random.sample(subgraph_nei_nodes_int, max_nodes_int)
		
		# print('subgraph_nei_nodes_int = {}'.format(subgraph_nei_nodes_int))

		# Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
		# print('enclosing_sub_graph = {}'.format(enclosing_sub_graph))
		subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)

		subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]
		'''
		#****
		incidence_matrix_subgraph = incidence_matrix(subgraph)
		for i in range(incidence_matrix_subgraph.shape[0]):
			for j in range(incidence_matrix_subgraph.shape[1]):
				if incidence_matrix_subgraph[i, j]:
					print('edge ({},{})'.format(mid2title.get(triples_dataset.id2entity[subgraph_nodes[i]]), mid2title.get(triples_dataset.id2entity[subgraph_nodes[j]])))
		#****
		'''
		if prune_subgraph:
			labels, enclosing_subgraph_nodes = node_label(incidence_matrix(subgraph), max_distance=h)
			pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
		else:
			pruned_subgraph_nodes = subgraph_nodes

		pruned_subgraph_nodes_set = set(pruned_subgraph_nodes)

		pruned_subgraph_edges = []

		# print('len(pruned_subgraph_nodes_set) = {}'.format(len(pruned_subgraph_nodes_set)))
		# print('adj shape = ', [adj.shape for adj in subgraph])

		if prune_subgraph:
			'''
			for i,adj in enumerate(subgraph):
				for j in range(adj.shape[0]):
					for k in range(adj.shape[1]):
						if adj[j, k] and subgraph_nodes[j] in pruned_subgraph_nodes_set and subgraph_nodes[k] in pruned_subgraph_nodes_set:
							pruned_subgraph_edges.append((subgraph_nodes[j], i, subgraph_nodes[k]))
			'''
			for j in range(len(subgraph_nodes)):
				if subgraph_nodes[j] in pruned_subgraph_nodes_set:
					for (edge_id, node_id) in adj_list[subgraph_nodes[j]]:
						if node_id in pruned_subgraph_nodes_set:
							pruned_subgraph_edges.append(tuple(triples[edge_id]))
		else:
			'''
			for i,adj in enumerate(subgraph):
				for j in range(adj.shape[0]):
					for k in range(adj.shape[1]):
						if adj[j, k]:
							pruned_subgraph_edges.append((subgraph_nodes[j], i, subgraph_nodes[k]))
			'''
			for j in range(len(subgraph_nodes)):
				for (edge_id, node_id) in adj_list[subgraph_nodes[j]]:
					if node_id in pruned_subgraph_nodes_set:
						pruned_subgraph_edges.append(tuple(triples[edge_id]))
			
		pruned_subgraph_edges = list(set(pruned_subgraph_edges)) # remove duplicate edges

		n_tries += 1
		# print('n_tries = {}'.format(n_tries))

	# print('len(pruned_subgraph_nodes) = {}, len(pruned_subgraph_edges) = {}'.format(len(pruned_subgraph_nodes), len(pruned_subgraph_edges)))
	# print('pruned_subgraph_nodes = {}, pruned_subgraph_edges = {}'.format(pruned_subgraph_nodes, pruned_subgraph_edges))

	pruned_labels = None
	if prune_subgraph:
		pruned_labels = labels[enclosing_subgraph_nodes] # distance from each node
		# pruned_subgraph_nodes = subgraph_nodes
		# pruned_labels = labels

		if max_node_label_value is not None:
			pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

	subgraph_size = len(pruned_subgraph_nodes)
	enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)
	num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes) # no. of nodes which have been pruned

	# print('pruned_subgraph_nodes = {}'.format(pruned_subgraph_nodes))
	# print('pruned_labels = {}'.format(pruned_labels)) 
	# print('subgraph_size = {}'.format(subgraph_size))
	# print('enc_ratio = {}'.format(enc_ratio))
	# print('num_pruned_nodes = {}'.format(num_pruned_nodes))

	heads = [x[0] for x in pruned_subgraph_edges]
	tails = [x[2] for x in pruned_subgraph_edges]

	entities = set(heads + tails)
	return entities, pruned_subgraph_edges, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes

# @profile
def subgraph_extraction_labeling_wo_pruning(ind, rel, neigh_dict, adj_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, max_nodes_int=None, prune_subgraph=None, triples_dataset=None, triples=None, mid2title=None, max_node_label_value=None, n_neigh_per_node=None, n_max_tries=20):
	# extract the h-hop enclosing subgraphs around link 'ind'
	# print('inside subgraph_extraction_labeling')

	subgraph_edges = []
	n_tries = 0

	while len(subgraph_edges)<3 and n_tries < n_max_tries: 
		# root1_nei_tuples = get_neighbor_nodes(set([ind[0]]), A_incidence, adj_list, h, max_nodes_per_hop, n_neigh_per_node, triples_dataset, mid2title)
		# root2_nei_tuples = get_neighbor_nodes(set([ind[1]]), A_incidence, adj_list, h, max_nodes_per_hop, n_neigh_per_node, triples_dataset, mid2title)
		# print('ind[0] = {}'.format(mid2title.get(triples_dataset.id2entity[ind[0]])))
		# print('ind[1] = {}'.format(mid2title.get(triples_dataset.id2entity[ind[1]])))

		root1_nei = neigh_dict[ind[0]]
		root2_nei = neigh_dict[ind[1]]

		subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
		subgraph_nei_nodes_un = root1_nei.union(root2_nei)
		# print('root1_nei = {}'.format([mid2title.get(triples_dataset.id2entity[x]) for x in root1_nei]))
		# print('root2_nei = {}'.format([mid2title.get(triples_dataset.id2entity[x]) for x in root2_nei]))
		# print('subgraph_nei_nodes_int = {}\n'.format([mid2title.get(triples_dataset.id2entity[x]) for x in subgraph_nei_nodes_int]))

		# subgraph_nei_nodes_int = root2_nei

		# TODO: add parameter for this
		
		if max_nodes_int and len(subgraph_nei_nodes_int)>max_nodes_int:
			subgraph_nei_nodes_int = random.sample(subgraph_nei_nodes_int, max_nodes_int)
		
		# Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
		subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
		# print('subgraph_nodes = {}'.format([mid2title.get(triples_dataset.id2entity[x]) for x in subgraph_nodes]))

		subgraph_nodes_set = set(subgraph_nodes)

		subgraph_edges = []

		for j in range(len(subgraph_nodes)):
			for (edge_id, node_id) in adj_list[subgraph_nodes[j]]:
				if node_id in subgraph_nodes_set:
					subgraph_edges.append(tuple(triples[edge_id]))
			
		subgraph_edges = list(set(subgraph_edges)) # remove duplicate edges
		# print('subgraph_edges = {}'.format([(mid2title.get(triples_dataset.id2entity[h]), triples_dataset.id2relation[r], mid2title.get(triples_dataset.id2entity[t])) for h,r,t in subgraph_edges]))

		n_tries += 1
		# print('n_tries = {}'.format(n_tries))

	# print('len(subgraph_nodes) = {}, len(subgraph_edges) = {}'.format(len(subgraph_nodes), len(subgraph_edges)))
	# print('subgraph_nodes = {}, subgraph_edges = {}'.format(subgraph_nodes, subgraph_edges))

	pruned_labels = None

	subgraph_size = len(subgraph_nodes)
	enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)
	num_pruned_nodes = 0 # no. of nodes which have been pruned

	heads = [x[0] for x in subgraph_edges]
	tails = [x[2] for x in subgraph_edges]

	entities = set(heads + tails)
	return entities, subgraph_edges, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes

def node_label(subgraph, max_distance=1):
	# implementation of the node labeling scheme described in the paper
	roots = [0, 1]

	sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots] # len=2

	dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]

	dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

	target_node_labels = np.array([[0, 1], [1, 0]])
	labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

	enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
	return labels, enclosing_subgraph_nodes
