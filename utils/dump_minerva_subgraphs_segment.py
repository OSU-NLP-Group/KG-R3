import sys, os
sys.path.append('../')

import lmdb
import networkx as nx
from dataset import SimpleLinkPredSubGraphDataset
from graph_utils import serialize, deserialize
from tqdm import tqdm
import argparse
import random
import codecs
import numpy as np

def parse_path_file(filename, dataset, dataset_path, triples, triple_id_to_subgraph_dict, reverse):
	f1 = codecs.open(filename, 'r', 'utf-8')
	if reverse:
		if dataset_path == 'data/FB15K-237/':
			query_rel = dataset.triples_dataset.relation2id[filename.split('/')[-1].replace('paths_','').replace('-','/')[1:]]
		else:
			query_rel = dataset.triples_dataset.relation2id[filename.split('/')[-1].replace('paths_','')[1:]]
	else:
		if dataset_path == 'data/FB15K-237/':
			query_rel = dataset.triples_dataset.relation2id[filename.split('/')[-1].replace('paths_','').replace('-','/')]
		else:
			query_rel = dataset.triples_dataset.relation2id[filename.split('/')[-1].replace('paths_','')]
		
	lines = f1.readlines()
	lines = [x.rstrip() for x in lines]

	line_id = 0

	while line_id < len(lines):
		start_e, end_e = lines[line_id].split('\t')
		start_e, end_e = dataset.triples_dataset.entity2id[start_e], dataset.triples_dataset.entity2id[end_e]

		test_triple_id = triples.index((start_e, query_rel, end_e))

		line_id += 1
		line_id += 1 # line corr. to Reward: x

		subgraph_edges = set()

		attr_dict = {}
		
		attr_dict['paths'] = []

		while not lines[line_id].startswith('#'):
			ent_trajectory = lines[line_id].split('\t')
			line_id += 1

			rel_trajectory = lines[line_id].split('\t')
			line_id += 1

			assert start_e == dataset.triples_dataset.entity2id[ent_trajectory[0]]
			
			prev_ent = start_e

			path = list()

			for ent,rel in zip(ent_trajectory[1:], rel_trajectory):
				if rel in ['NO_OP', 'PAD']:
					continue

				if dataset_path in ['data/FB15K-237/', 'data/YAGO3-10/', 'data/umls/', 'data/NELL-995/']:
					if rel.startswith('_'):
						rel = rel[1:]
				if dataset_path == 'data/WN18RR/':
					if rel.startswith('__'):
						rel = rel[1:]
						# print('flag 1')
				rel = dataset.triples_dataset.relation2id[rel]

				subgraph_edges.add((prev_ent, rel, dataset.triples_dataset.entity2id[ent]))
				
				if dataset.triples_dataset.entity2id[ent] != start_e:
					path.append(dataset.triples_dataset.entity2id[ent])

				prev_ent = dataset.triples_dataset.entity2id[ent]
			
			line_id += 1 # line corr. to reward (pos/neg)
			line_id += 1 # line corr. to log prob
			line_id += 1 # line corr. to __

			attr_dict['paths'].append(path)

		attr_dict['subgraph_edges'] = subgraph_edges

		triple_id_to_subgraph_dict[test_triple_id] = attr_dict

		line_id += 1 # line corr. to ##..#

def get_average_subgraph_size(triple_id_to_subgraph_dict, num_samples):
	total_edges = 0

	if num_samples > len(triple_id_to_subgraph_dict):
		num_samples = len(triple_id_to_subgraph_dict)

	random_sample_keys = random.sample(triple_id_to_subgraph_dict.keys(), num_samples)

	for test_triple_id in random_sample_keys:
		total_edges += len(triple_id_to_subgraph_dict[test_triple_id])

	avg_subgraph_size = total_edges*1.0/num_samples
	return avg_subgraph_size



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--path-dir', default='')
	parser.add_argument('--db-path', required=True)
	parser.add_argument('--split', choices=['train', 'valid', 'test'], default='test')
	parser.add_argument('--dataset-path', default='data/FB15K-237/')
	parser.add_argument('--dry-run', action='store_true')
	parser.add_argument('--reverse', action='store_true')
	args = parser.parse_args()

	dataset = SimpleLinkPredSubGraphDataset(args.dataset_path, 'train', 'head', None, None, None, None)

	if args.split == 'train':
		triples = dataset.train_data
	elif args.split == 'valid':
		triples = dataset.valid_data
	else:
		triples = dataset.test_data

	# reverse direction
	if args.reverse:
		triples = [(x[2], x[1], x[0]) for x in triples]

	# load entity vocab
	entity2id = dataset.triples_dataset.entity2id

	# load relation vocab
	relation2id = dataset.triples_dataset.relation2id

	# parse the path file into paths keyed by the triple
	triple_id_to_subgraph_dict = {}

	for filename in tqdm(os.listdir(args.path_dir)):
		if 'answers' in filename:
			continue
		parse_path_file(os.path.join(args.path_dir, filename), dataset, args.dataset_path, triples, triple_id_to_subgraph_dict, args.reverse)
		if args.dry_run:
			break

	# compute BYTES_PER_DATUM
	BYTES_PER_DATUM = get_average_subgraph_size(triple_id_to_subgraph_dict, 100) * 1.5

	map_size = len(triples) * BYTES_PER_DATUM * 1200

	# print('BYTES_PER_DATUM = {}'.format(BYTES_PER_DATUM))
	# print('map_size = {}'.format(map_size))

	env = lmdb.open(args.db_path, map_size=map_size, max_dbs=6)

	db_name = args.split + '_pos'
	split_env = env.open_db(db_name.encode())

	n_ex_ans_present = 0

	for test_triple_id in tqdm(triple_id_to_subgraph_dict):

		n1, r_label, n2 = triples[test_triple_id]
		# print(n1, r_label, n2)

		attr_dict = triple_id_to_subgraph_dict[test_triple_id]

		edges = list(attr_dict['subgraph_edges'])

		paths = attr_dict['paths']

		heads = [x[0] for x in edges]
		tails = [x[2] for x in edges]
		nodes = set(heads + tails)

		ent_segment_scores = {node:0 for node in nodes}

		for node in nodes:
			for path_id,path in enumerate(paths):
				# print(path)
				if len(path)>0 and node == path[-1]:
					ent_segment_scores[node] = 1

		if args.reverse:
			datum = {'n1' : n2, 'n2' : n1, 'r_label': r_label, 'nodes': nodes, 'edges': edges, 'ent_segment_scores': ent_segment_scores}
		else:
			datum = {'n1' : n1, 'n2' : n2, 'r_label': r_label, 'nodes': nodes, 'edges': edges, 'ent_segment_scores': ent_segment_scores}
		
		str_id = '{:08}'.format(test_triple_id).encode('ascii')

		with env.begin(write=True, db=split_env) as txn:
			txn.put(str_id, serialize(datum))

		if args.dry_run:
			break

if __name__=='__main__':
	main()