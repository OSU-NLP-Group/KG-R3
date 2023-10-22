import sys, os
import json
import numpy as np
import networkx as nx
from utils.gen_utils import *
from utils.graph_sampler import links2subgraphs
from utils.graph_utils import deserialize, deserialize_seg, deserialize_seg_dir
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
from scipy.sparse import csc_matrix
import lmdb
from copy import deepcopy

class LinkPredSubGraphDataset(Dataset):
	def __init__(self, dataset_path, split, mode, sample_size, neigh_size, sampling_type, graph_connection, device, args):
		super(LinkPredSubGraphDataset, self).__init__()
		self.dataset_path = dataset_path
		self.split = split
		self.mode = mode
		self.sample_size = sample_size
		self.neigh_size = neigh_size
		self.sampling_type = sampling_type
		self.graph_connection = graph_connection
		self.device = device
		self.add_segment_embed = args.add_segment_embed
		self.args = args

		# initialize dataset of KG triples
		self.triples_dataset = KGDataset(self.dataset_path)

		# add [PAD] token to entity vocab
		self.triples_dataset.entity2id['<pad>'] = len(self.triples_dataset.entity2id)

		# add [PAD] token to relation vocab
		self.triples_dataset.relation2id['<pad>'] = len(self.triples_dataset.relation2id)

		self.triples_dataset.id2entity = {v:k for k,v in self.triples_dataset.entity2id.items()}
		self.triples_dataset.id2relation = {v:k for k,v in self.triples_dataset.relation2id.items()}

		# build adj list and calculate degrees for sampling
		self.num_nodes = len(self.triples_dataset.entity2id)
		self.num_entities = len(self.triples_dataset.entity2id)
		self.num_relations = len(self.triples_dataset.relation2id)

		self.train_data = self.triples_dataset.train
		self.valid_data = self.triples_dataset.valid
		self.test_data = self.triples_dataset.test

		self.adj_list, self.degrees = get_adj_and_degrees(self.num_nodes, self.train_data)

		if self.split == 'train':
			self.triples = self.train_data
		elif self.split == 'valid':
			self.triples = self.valid_data
		else:
			self.triples = self.test_data
		
		if self.sampling_type == 'minerva':
			if not self.add_segment_embed:
				if self.args.beam_size == 100:
					if self.mode == 'head':
						self.db_path = os.path.join(self.dataset_path, f'subgraphs_minerva_{self.split}_rev')
					else:
						self.db_path = os.path.join(self.dataset_path, f'subgraphs_minerva_{self.split}')
				else:
					if self.mode == 'head':
						self.db_path = os.path.join(self.dataset_path, f'subgraphs_minerva_beam_{self.args.beam_size}_{self.split}_rev')
					else:
						self.db_path = os.path.join(self.dataset_path, f'subgraphs_minerva_beam_{self.args.beam_size}_{self.split}')
			else:
				if self.args.beam_size == 100:
					if self.mode == 'head':
						self.db_path = os.path.join(self.dataset_path, f'subgraphs_minerva_seg_{self.split}_rev')
					else:
						self.db_path = os.path.join(self.dataset_path, f'subgraphs_minerva_seg_{self.split}')
				else:
					if self.mode == 'head':
						self.db_path = os.path.join(self.dataset_path, f'subgraphs_minerva_beam_{self.args.beam_size}_seg_{self.split}_rev')
					else:
						self.db_path = os.path.join(self.dataset_path, f'subgraphs_minerva_beam_{self.args.beam_size}_seg_{self.split}')

			self.main_env = lmdb.open(self.db_path, readonly=True, max_dbs=3, lock=False)
			db_name_pos = self.split + '_pos'

			print('db_name_pos = {}'.format(db_name_pos))
			self.db_pos = self.main_env.open_db(db_name_pos.encode())
		else:
			self.db_path = None

		print(self.db_path)

	def __getitem__(self, idx):
		edge = self.triples[idx]
		# print('edge = {}'.format(edge))

		if self.mode == 'head': # head is corrupted
			central_node = edge[2]
			masked_node = edge[0]

			label = self.get_label(self.triples_dataset.or2s_all[(edge[2], edge[1])])
			is_head_corrupted = True
		else: # tail is corrupted
			central_node = edge[0]
			masked_node = edge[2]

			label = self.get_label(self.triples_dataset.sr2o_all[(edge[0], edge[1])])
			is_head_corrupted = False

		# sample connected subgraph
		if self.sampling_type == 'bfs':
			edges = sample_subgraph_with_central_node_bfs(self.adj_list, central_node, self.num_nodes, len(self.train_data), self.sample_size, self.neigh_size).tolist()
		elif self.sampling_type == 'onehop':
			edges = sample_subgraph_with_central_node_onehop(self.adj_list, central_node, self.num_nodes, len(self.train_data), self.sample_size).tolist()
		elif self.sampling_type == 'minerva':
			if not self.add_segment_embed:
				with self.main_env.begin(db=self.db_pos) as txn:
					str_id = '{:08}'.format(idx).encode('ascii')
					try:
						_, _, _, _, edges = deserialize(txn.get(str_id)).values()
					except:
						edges = []
			else:
				with self.main_env.begin(db=self.db_pos) as txn:
					str_id = '{:08}'.format(idx).encode('ascii')
					try:
						_, _, _, _, edges, ent_segment_scores = deserialize_seg(txn.get(str_id)).values()
					except:
						edges = []
						ent_segment_scores = {}
		else:
			raise NotImplementedError

		if self.sampling_type in ['bfs', 'onehop', 'rwr', 'bfs-complete']: # bfs
			edges = np.asarray([self.train_data[i] for i in edges])
		elif self.sampling_type in ['minerva', 'min_0.25_gold_0.75', 'min_0.5_gold_0.5', 'min_0.75_gold_0.25']:
			edges = list(edges)
			
					
		edges = np.asarray(edges)

		if edges.shape[0]>0:
			head, rel, tail = edges.transpose()
		else:
			head, rel, tail = edges, edges, edges

		entities = [central_node]
		relations = [self.triples_dataset.id2relation[x] for x in rel.tolist()]

		entities.extend(list(set(np.concatenate((head, tail)).tolist()) - set(entities)))

		if self.add_segment_embed:
			if len(ent_segment_scores)>0:
				segment_ids = [ent_segment_scores[e] for e in entities]
			else:
				segment_ids = [0]
			segment_ids = segment_ids + [0]*len(relations)

		entities = [self.triples_dataset.id2entity[x] for x in entities]

		ent_graph = nx.Graph()

		for ent in entities:
			if ent not in ent_graph.nodes:
				ent_graph.add_node(ent)

		# make the entity graph fully connected
		ent_graph = nx.complete_graph(ent_graph)

		# relations are represented by integers and multiple occurrences of same relation are counted as different tokens
		rel_graph = nx.Graph()

		for i,_ in enumerate(relations):
			rel_graph.add_node(str(i))

		# make the relation graph fully connected
		rel_graph = nx.complete_graph(rel_graph)

		G = nx.algorithms.operators.binary.union(ent_graph, rel_graph)

		for i, (h, r, t) in enumerate(zip(head, rel, tail)):
			G.add_edge(self.triples_dataset.id2entity[h], str(i))
			G.add_edge(self.triples_dataset.id2entity[t], str(i))
		

		# create graph for attention
		if self.graph_connection != 'type_5':
			adj = np.array(nx.adjacency_matrix(G).todense())
			adj = adj + np.eye(adj.shape[0], dtype=int)
		else:
			adj = np.ones((len(entities)+len(relations), len(entities)+len(relations)), dtype=int)

		adj = adj.tolist()

		# mask some entities and obtain masked_ent_labels
		entity_ids = [self.triples_dataset.entity2id[x] for x in entities]
		
		entity_mlm_labels = masked_node

		# relations are not masked
		relation_ids = [self.triples_dataset.relation2id[x] for x in relations]
		relation_mlm_labels = -1

		tokens = entity_ids + relation_ids
		token_type_ids = [0]*len(entity_ids) + [1]*len(relation_ids)

		data = {
					'entity_ids': entity_ids,
					'relation_ids': relation_ids,
					'attention_mask': torch.tensor(adj).to(self.device),
					'token_type_ids': torch.tensor(token_type_ids).to(self.device),
					'ent_masked_lm_labels': entity_mlm_labels,
					'rel_masked_lm_labels': relation_mlm_labels,
					'edge': edge,
					'ent': central_node,
					'rel': edge[1] + self.num_relations if self.args.add_inverse_rels and is_head_corrupted else edge[1],
					'label': label.to(self.device),
					'ent_len_tensor': len(entity_ids)
				}

		if self.add_segment_embed:
			data['segment_ids'] = torch.tensor(segment_ids).to(self.device)
		else:
			data['segment_ids'] = torch.zeros(len(entity_ids) + len(relation_ids), device=self.device)

		data['edges'] = edges
			
		return data

	def get_label(self, label):
		y = np.zeros([self.num_entities], dtype=np.float32)
		for e2 in label:
			y[e2] = 1.0

		return torch.FloatTensor(y)

	def __len__(self):
		return len(self.triples)

class KGDataset:
	'''Load a knowledge graph

	The folder with a knowledge graph has five files:
	* entities stores the mapping between entity Id and entity name.
	* relations stores the mapping between relation Id and relation name.
	* train stores the triples in the training set.
	* valid stores the triples in the validation set.
	* test stores the triples in the test set.

	The mapping between entity (relation) Id and entity (relation) name is stored as 'id\tname'.

	The triples are stored as 'head_name\trelation_name\ttail_name'.
	'''
	# path contains 'train.tsv', 'valid.tsv', 'test.tsv' files.
	def __init__(self, path, format=[0,1,2], skip_first_line=False):

		self.load_entity_relation(path, ['train.tsv', 'valid.tsv', 'test.tsv'], format)

		#####
		entity_path = os.path.join(path, 'entities.tsv')
		relation_path = os.path.join(path, 'relations.tsv')
		train_path = os.path.join(path, 'train.tsv')
		valid_path = os.path.join(path, 'valid.tsv')
		test_path = os.path.join(path, 'test.tsv')

		self.entity2id, self.n_entities = self.read_entity(entity_path)
		self.relation2id, self.n_relations = self.read_relation(relation_path)

		self.id2entity = {v:k for k,v in self.entity2id.items()}
		self.id2relation = {v:k for k,v in self.relation2id.items()}

		self.sr2o_all = defaultdict(set)
		self.or2s_all = defaultdict(set)
		
		self.e2re = defaultdict(set)

		self.train = self.read_triple(train_path, "train", self.sr2o_all, self.or2s_all, self.e2re, skip_first_line, format)

		self.sr2o = deepcopy(self.sr2o_all)
		self.or2s = deepcopy(self.or2s_all)

		self.valid = self.read_triple(valid_path, "valid", self.sr2o_all, self.or2s_all, self.e2re, skip_first_line, format)
		self.test = self.read_triple(test_path, "test", self.sr2o_all, self.or2s_all, self.e2re, skip_first_line, format)

		self.sr2o_all = {k: list(v) for k, v in self.sr2o_all.items()}
		self.or2s_all = {k: list(v) for k, v in self.or2s_all.items()}
		
	def load_entity_relation(self, path, files, format):
		if os.path.exists(os.path.join(path, "entities.tsv")) and os.path.exists(os.path.join(path, "relations.tsv")):
			return
			
		entity_map = {}
		rel_map = {}
		for fi in files:
			with open(os.path.join(path, fi)) as f:
				for line in f:
					triple = line.strip().split('\t')
					src, rel, dst = triple[format[0]], triple[format[1]], triple[format[2]]
					src_id = _get_id(entity_map, src)
					dst_id = _get_id(entity_map, dst)
					rel_id = _get_id(rel_map, rel)

		entities = ["{}\t{}\n".format(val, key) for key, val in entity_map.items()]
		with open(os.path.join(path, "entities.tsv"), "w+") as f:
			f.writelines(entities)
		self.entity2id = entity_map
		self.n_entities = len(entities)

		relations = ["{}\t{}\n".format(val, key) for key, val in rel_map.items()]
		with open(os.path.join(path, "relations.tsv"), "w+") as f:
			f.writelines(relations)
		self.relation2id = rel_map
		self.n_relations = len(relations)

	def read_entity(self, entity_path):
		with open(entity_path) as f:
			entity2id = {}
			for line in f:
				eid, entity = line.strip().split('\t')
				entity2id[entity] = int(eid)

		return entity2id, len(entity2id)

	def read_relation(self, relation_path):
		with open(relation_path) as f:
			relation2id = {}
			for line in f:
				rid, relation = line.strip().split('\t')
				relation2id[relation] = int(rid)

		return relation2id, len(relation2id)

	def read_triple(self, path, split, sr2o_all, or2s_all, e2re, skip_first_line=False, format=[0,1,2]):
		# split: train/valid/test
		if path is None:
			return None

		print('Reading {} triples....'.format(split))
		# heads = []
		# tails = []
		# rels = []
		triples = []
		with open(path) as f:
			if skip_first_line:
				_ = f.readline()
			for line in f:
				triple = line.strip().split('\t')
				h, r, t = triple[format[0]], triple[format[1]], triple[format[2]]
				# heads.append(self.entity2id[h])
				# rels.append(self.relation2id[r])
				# tails.append(self.entity2id[t])
				triples.append((self.entity2id[h], self.relation2id[r], self.entity2id[t]))

				sr2o_all[(self.entity2id[h], self.relation2id[r])].add(self.entity2id[t])
				or2s_all[(self.entity2id[t], self.relation2id[r])].add(self.entity2id[h])

				if split == 'train':
					e2re[self.entity2id[h]].add((self.relation2id[r], self.entity2id[t]))
					e2re[self.entity2id[t]].add((self.relation2id[r], self.entity2id[h]))

		print('Finished. Read {} {} triples.'.format(len(triples), split))

		return triples
