import sys, os
import json
import numpy as np
import networkx as nx
from utils.gen_utils import *
from utils.graph_sampler import links2subgraphs
from utils.graph_utils import deserialize
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
from scipy.sparse import csc_matrix
import lmdb
from copy import deepcopy


class SubGraphDataset(Dataset):
	def __init__(self, dataset_path, split, neigh_size, sampling_type, graph_connection, device, stat_mode, args, include_edges=False):
		super(SubGraphDataset, self).__init__()

		self.dataset_path = dataset_path
		self.split = split
		self.neigh_size = neigh_size
		self.sampling_type = sampling_type
		self.graph_connection = graph_connection
		self.device = device
		self.stat_mode = stat_mode

		self.args = args
		self.include_edges = include_edges
		self.add_segment_embed = args.add_segment_embed

		# initialize dataset of KG triples
		self.triples_dataset = KGDataset(self.dataset_path)

		self.triples_dataset.entity2id['<pad>'] = len(self.triples_dataset.entity2id)

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

		if self.split == 'train':
			self.triples = self.train_data
		elif self.split == 'valid':
			self.triples = self.valid_data
		else:
			self.triples = self.test_data

		self.graphs = {}
		if self.split == 'train':
			self.graphs['train'] = {'pos': np.asarray(self.triples_dataset.train)}
		elif self.split == 'valid':
			self.graphs['valid'] = {'pos': np.asarray(self.triples_dataset.valid)}
		else:
			self.graphs['test'] = {'pos': np.asarray(self.triples_dataset.test)}

		# val data can be based on train graph or eval graph for MLM; while for link prediction, it is always train graph
		self.adj_list, _ = get_adj_and_degrees(self.num_nodes, self.train_data)

		# Code credit: https://github.com/kkteru/grail
		# Construct the list of adjacency matrix each corresponding to each relation. Note that this is constructed only from the train data.

		self.triples_dataset.train = np.asarray(self.triples_dataset.train)
		self.triples_dataset.valid = np.asarray(self.triples_dataset.valid)
		self.triples_dataset.test = np.asarray(self.triples_dataset.test)

		if self.args.sampling_type == 'bfs':
			self.db_path = os.path.join(self.dataset_path, f'subgraphs_samplesize_{args.sample_size}_neigh_size_{args.neigh_size}_{self.split}')
		elif self.args.sampling_type == 'onehop':
			self.db_path = os.path.join(self.dataset_path, f'subgraphs_samplesize_{args.sample_size}_{self.split}')
		elif self.args.sampling_type == 'minerva':
			if not self.add_segment_embed:
				if self.args.beam_size == 100:
					self.db_path = os.path.join(self.dataset_path, f'subgraphs_minerva_{self.split}')
				else:
					self.db_path = os.path.join(self.dataset_path, f'subgraphs_minerva_beam_{self.args.beam_size}_{self.split}')
			else:
				if self.args.beam_size == 100:
					self.db_path = os.path.join(self.dataset_path, f'subgraphs_minerva_seg_{self.split}')
				else:
					self.db_path = os.path.join(self.dataset_path, f'subgraphs_minerva_beam_{self.args.beam_size}_seg_{self.split}')

			self.main_env_fw = lmdb.open(self.db_path, readonly=True, max_dbs=3, lock=False) # tail is currupted
			db_name_pos_fw = self.split + '_pos'

			self.db_pos_fw = self.main_env_fw.open_db(db_name_pos_fw.encode())

			self.main_env_bw = lmdb.open(self.db_path+'_rev', readonly=True, max_dbs=3, lock=False) # head is currupted
			db_name_pos_bw = self.split + '_pos'

			self.db_pos_bw = self.main_env_bw.open_db(db_name_pos_bw.encode())

		print('db_path = {}'.format(self.db_path))

		# dump subgraphs in db
		if not os.path.exists(self.db_path):
			# dump pickled graphs in db
			links2subgraphs(self.adj_list, self.graphs, self.db_path, self.triples_dataset, self.train_data, args)
		
		if self.args.sampling_type in ['bfs', 'onehop']:
			print('db dumping finished...')
			self.main_env = lmdb.open(self.db_path, readonly=True, max_dbs=3, lock=False)
			db_name_pos = self.split + '_pos'

			print('db_name_pos = {}'.format(db_name_pos))
			self.db_pos = self.main_env.open_db(db_name_pos.encode())


	def __getitem__(self, idx):
		# print(idx)

		if self.args.sampling_type in ['bfs', 'onehop']:
			with self.main_env.begin(db=self.db_pos) as txn:
				if self.args.sampling_type == 'khop':
					str_id = '{:08}'.format(idx % len(self.triples)).encode('ascii')
				else: #bfs 
					str_id = '{:08}'.format(idx).encode('ascii')
				n1, n2, r_label, nodes, edges = deserialize(txn.get(str_id)).values()
		
		else: #minerva
			if not self.args.add_segment_embed:
				if idx < len(self.triples): # head is corrupted
					with self.main_env_bw.begin(db=self.db_pos_bw) as txn:
						try:
							str_id = '{:08}'.format(idx).encode('ascii')
							n1, n2, r_label, nodes, edges = deserialize(txn.get(str_id)).values()
						except:
							edges = []
							n1, r_label, n2 = 0, 0, 0
				else: # tail is corrupted
					with self.main_env_fw.begin(db=self.db_pos_fw) as txn:
						try:
							str_id = '{:08}'.format(idx % len(self.triples)).encode('ascii')
							n1, n2, r_label, nodes, edges = deserialize(txn.get(str_id)).values()
						except:
							edges = []
							n1, r_label, n2 = 0, 0, 0
			else:
				if idx < len(self.triples): # head is corrupted
					with self.main_env_bw.begin(db=self.db_pos_bw) as txn:
						try:
							str_id = '{:08}'.format(idx).encode('ascii')
							n1, n2, r_label, nodes, edges, ent_segment_scores = deserialize_seg(txn.get(str_id)).values()
						except:
							edges = []

							n1, r_label, n2 = 0, 0, 0
							ent_segment_scores = {}
				else: # tail is corrupted
					with self.main_env_fw.begin(db=self.db_pos_fw) as txn:
						try:
							str_id = '{:08}'.format(idx % len(self.triples)).encode('ascii')
							n1, n2, r_label, nodes, edges, ent_segment_scores = deserialize_seg(txn.get(str_id)).values()
						except:
							edges = []

							n1, r_label, n2 = 0, 0, 0
							ent_segment_scores = {}

		edges = list(edges)

		edge = (n1, r_label, n2)

		if edge in edges:
			edges.remove(edge)

		if idx < len(self.triples):
			# head is corrupted
			masked_node = n1
			central_node = edge[2]
		else:
			# tail is corrupted
			masked_node = n2
			central_node = edge[0]

		edges = np.asarray(edges)

		# retrieve nodes and relations in the subgraph (these comprise the sequence)
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
		
		if self.stat_mode:
			return len(entities), len(relations)

		# get the adjacency matrix (e-e, e-r and r-r edges)

		if self.graph_connection == 'type_1':
			# entities are represented by strings and multiple entities are counted as single occurrence
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

		# add inverse relations
		if self.args.add_inverse_rels and idx < len(self.triples):
			r_label = r_label + self.num_relations

		data = {
					'entity_ids': entity_ids,
					'relation_ids': relation_ids,
					'attention_mask': torch.tensor(adj).to(self.device),
					'token_type_ids': torch.tensor(token_type_ids).to(self.device),
					'ent_masked_lm_labels': entity_mlm_labels,
					'rel_masked_lm_labels': relation_mlm_labels,
					'ent': central_node,
					'rel': r_label
				}
		# print('adj = {}'.format(data['attention_mask']))

		if self.include_edges:
			data['edges'] = edges

		if self.add_segment_embed:
			data['segment_ids'] = torch.tensor(segment_ids).to(self.device)
		else:
			data['segment_ids'] = None

		return data

	def __len__(self):
		return len(self.triples) * 2

class SimpleLinkPredSubGraphDataset(Dataset):
	def __init__(self, dataset_path, split, mode, neigh_size, sampling_type, graph_connection, device):
		super(SimpleLinkPredSubGraphDataset, self).__init__()
		self.dataset_path = dataset_path
		self.split = split
		self.mode = mode
		self.neigh_size = neigh_size
		self.sampling_type = sampling_type
		self.graph_connection = graph_connection
		self.device = device

		# initialize dataset of KG triples
		self.triples_dataset = KGDataset(self.dataset_path)

		# add [MASK] token to entity vocab
		self.triples_dataset.entity2id['<mask>'] = len(self.triples_dataset.entity2id)

		# add [MASK] token to relation vocab
		self.triples_dataset.relation2id['<mask>'] = len(self.triples_dataset.relation2id)

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

		if self.split == 'train':
			self.triples = self.train_data
		elif self.split == 'valid':
			self.triples = self.valid_data
		else:
			self.triples = self.test_data

		self.triples_dataset.train = np.asarray(self.triples_dataset.train)
		self.triples_dataset.valid = np.asarray(self.triples_dataset.valid)

	def __getitem__(self, idx):
		edge = self.triples[idx]
		# print('edge = {}'.format(edge))

		if self.mode == 'head': # head is corrupted
			central_node = edge[2]
			masked_node = edge[0]

			label = self.get_label(self.triples_dataset.or2s_all[(edge[2], edge[1])])
		else: # tail is corrupted
			central_node = edge[0]
			masked_node = edge[2]

			label = self.get_label(self.triples_dataset.sr2o_all[(edge[0], edge[1])])

		data = {
					'edge': edge,
					'label': label.to(self.device),
				}

		return data

	def get_label(self, label):
		y = np.zeros([self.triples_dataset.n_entities], dtype=np.float32)
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
				triples.append((self.entity2id[h], self.relation2id[r], self.entity2id[t]))

				sr2o_all[(self.entity2id[h], self.relation2id[r])].add(self.entity2id[t])
				or2s_all[(self.entity2id[t], self.relation2id[r])].add(self.entity2id[h])

				if split == 'train':
					e2re[self.entity2id[h]].add((self.relation2id[r], self.entity2id[t]))
					e2re[self.entity2id[t]].add((self.relation2id[r], self.entity2id[h]))

		print('Finished. Read {} {} triples.'.format(len(triples), split))

		return triples