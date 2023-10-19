#!/home/pahuja.9/miniconda3/envs/colake/bin/python
import sys, os
import argparse
import numpy as np
from tqdm  import tqdm
import json
import random
import networkx as nx

import torch
from torch.utils.data import DataLoader

from utils.gen_utils import *
from dataset import SubGraphDataset
import pickle

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset-path', type=str, required=True)
	parser.add_argument('--split', type=str, choices=['train', 'valid', 'test'], default='valid')
	parser.add_argument('--out-dir', type=str, required=True)
	parser.add_argument('--sample-size', type=int, default=100, help='sample size in terms of no. of edges')
	parser.add_argument('--embed-dim', type=int, default=768, help='embedding dim.')
	parser.add_argument('--cuda', action='store_true')
	parser.add_argument('--neigh-size', type=int, default=10)
	parser.add_argument('--batch-size', type=int, default=1)
	parser.add_argument('--graph-connection', type=str, choices=['type_1', 'type_5'], default='type_1')
	parser.add_argument('--sampling-type', type=str, choices=['bfs', 'minerva', 'onehop'], default='rwr')
	parser.add_argument('--use-pretrained', action='store_true', help='if set, initialize from pre-trained weights', default=True)
	parser.add_argument('--seed', type=int, default=231327)

	parser.add_argument("--use-descending", action="store_true")
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--add-segment-embed', action='store_true', help='add segment embedding for triple vs context')

	parser.add_argument("--hop", type=int, default=3, help="Enclosing subgraph hop number")
	parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True, help='whether to only consider enclosing subgraph')
	parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None, help="if > 0, upper bound the # nodes per hop by subsampling")
	parser.add_argument("--n-neigh-per-node", type=int, default=None, help="if > 0, upper bound the # neigh nodes per node")
	parser.add_argument("--max_nodes_int", type=int, default=None, help="limit on max. no. of nodes in intersection")
	parser.add_argument("--prune-subgraph", action='store_true', default=True)
	parser.add_argument('--beam-size', type=int, default=100, help='beam size for Minerva model decoding')
	parser.add_argument('--add-inverse-rels', action='store_true', help='add inverse relations for query tower')

	args = parser.parse_args()

	device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

	# set seed
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	# dataset without stat mode
	dataset = SubGraphDataset(args.dataset_path, args.split, args.neigh_size, args.sampling_type, args.graph_connection, device, True, args)
	train_dataset_lens = torch.tensor([sum(dataset[i]) for i in range(len(dataset))])
	train_batch_sample_ids = torch.split(torch.argsort(train_dataset_lens, descending=args.use_descending), split_size_or_sections=args.batch_size)

	dataset = SubGraphDataset(args.dataset_path, args.split, args.neigh_size, args.sampling_type, args.graph_connection, device, False, args)
	my_collator = MyCollatorCustom(dataset.triples_dataset.entity2id['<pad>'], dataset.triples_dataset.relation2id['<pad>'], add_segment_embed=args.add_segment_embed)
	dataloader = DataLoader(dataset, collate_fn=my_collator, batch_sampler=train_batch_sample_ids)
	
	# print(os.path.join(args.out_dir, '{}_{}_{}_{}_{}_{}'.format(args.sampling_type, args.graph_connection, args.position_enc_type, args.split, args.batch_size, args.adj_type)))
	os.makedirs(os.path.join(args.out_dir, '{}_{}_{}_{}_{}'.format(args.sampling_type, args.graph_connection, args.split, args.batch_size, args.beam_size)))

	with torch.no_grad():
		for batch_id, batch in enumerate(tqdm(dataloader)):
			pickle.dump(batch, open(os.path.join(args.out_dir, '{}_{}_{}_{}_{}'.format(args.sampling_type, args.graph_connection, args.split, args.batch_size, args.beam_size), '{}.pkl'.format(batch_id)), 'wb'))

			if batch_id>100 and args.debug:
				break

if __name__=='__main__':
	main()
