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
from utils.graph_utils import incidence_matrix, remove_nodes, ssp_to_torch, serialize, deserialize
import networkx as nx
from utils.gen_utils import sample_subgraph_with_central_node_bfs, sample_subgraph_with_central_node_onehop

def links2subgraphs(adj_list, graphs, db_path, triples_dataset, triples, params):
	'''
	extract enclosing subgraphs, write map mode + named dbs
	'''

	subgraph_sizes = []
	enc_ratios = []
	num_pruned_nodes = []

	BYTES_PER_DATUM = get_average_subgraph_size(100, list(graphs.values())[0]['pos'], adj_list, triples_dataset, triples, params) * 1.5
	links_length = 0
	for split_name, split in graphs.items():
		links_length += (len(split['pos'])) * 2
	map_size = links_length * BYTES_PER_DATUM

	map_size = map_size*100

	env = lmdb.open(db_path, map_size=map_size, max_dbs=6)

	def extraction_helper_bfs(adj_list, links, split_env):

		with env.begin(write=True, db=split_env) as txn:
			txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)*2), byteorder='little'))
		
		with mp.Pool(processes=params.num_workers, initializer=intialize_worker, initargs=(adj_list, triples_dataset, triples, params)) as p:
			args_ = zip(range(len(links)*2), np.tile(links, (2,1)), [len(links)]*2*len(links))

			for (str_id, datum) in tqdm(p.imap(extract_save_subgraph_bfs, args_), total=len(links)*2):
				# print('str_id = {}, datum = {}'.format(str_id, datum))

				with env.begin(write=True, db=split_env) as txn:
					txn.put(str_id, serialize(datum))

	def extraction_helper_onehop(adj_list, links, split_env):

		with env.begin(write=True, db=split_env) as txn:
			txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)*2), byteorder='little'))
		
		with mp.Pool(processes=params.num_workers, initializer=intialize_worker, initargs=(adj_list, triples_dataset, triples, params)) as p:
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
		if params.sampling_type == 'bfs':
			extraction_helper_bfs(adj_list, split['pos'], split_env)
		elif params.sampling_type == 'onehop':
			extraction_helper_onehop(adj_list, split['pos'], split_env)

def get_average_subgraph_size(sample_size, links, adj_list, triples_dataset, triples, params):
	total_size = 0
	for (n1, r_label, n2) in tqdm(links[np.random.choice(len(links), sample_size)]):

		if params.sampling_type == 'bfs':
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


def intialize_worker(adj_list, triples_dataset, triples, params):
	global adj_list_, params_, triples_dataset_, triples__
	adj_list_, params_, triples_dataset_, triples__ = adj_list, params, triples_dataset, triples

def extract_save_subgraph_bfs(args_):
	idx, (n1, r_label, n2), links_length = args_

	if idx < links_length: # head is corrupted
		central_node = n2
	else: # tail is corrupted
		central_node = n1

	num_nodes = len(triples_dataset_.entity2id)
	train_data = triples_dataset_.train # list of training triples

	edges = sample_subgraph_with_central_node_bfs(adj_list_, central_node, num_nodes, len(train_data), params_.sample_size, params_.neigh_size).tolist()
	edges = [tuple(train_data[i]) for i in edges]


	heads = [x[0] for x in edges]
	tails = [x[2] for x in edges]
	nodes = set(heads + tails)
	subgraph_size = len(nodes)

	datum = {'n1' : n1, 'n2' : n2, 'r_label': r_label, 'nodes': nodes, 'edges': edges}
	str_id = '{:08}'.format(idx).encode('ascii')

	return (str_id, datum)

def extract_save_subgraph_onehop(args_):
	idx, (n1, r_label, n2), links_length = args_

	if idx < links_length: # head is corrupted
		central_node = n2
	else: # tail is corrupted
		central_node = n1

	num_nodes = len(triples_dataset_.entity2id)
	train_data = triples_dataset_.train # list of training triples

	edges = sample_subgraph_with_central_node_onehop(adj_list_, central_node, num_nodes, len(train_data), params_.sample_size).tolist()
	edges = [tuple(train_data[i]) for i in edges]


	heads = [x[0] for x in edges]
	tails = [x[2] for x in edges]
	nodes = set(heads + tails)
	subgraph_size = len(nodes)

	datum = {'n1' : n1, 'n2' : n2, 'r_label': r_label, 'nodes': nodes, 'edges': edges}
	str_id = '{:08}'.format(idx).encode('ascii')

	return (str_id, datum)

def get_neighbor_nodes(roots, adj, adj_list, h=1, max_nodes_per_hop=None, n_neigh_per_node=None, triples_dataset=None):
	bfs_generator = _bfs_relational(adj, adj_list, roots, max_nodes_per_hop, n_neigh_per_node, triples_dataset)
	lvls = list()
	for _ in range(h):
		try:
			lvls.append(next(bfs_generator))
		except StopIteration:
			pass
	return set().union(*lvls)


