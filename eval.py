#!/home/pahuja.9/miniconda3/envs/colake/bin/python
import sys, os
import argparse
import numpy as np
from tqdm  import tqdm
import json
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.gen_utils import *
from dataset import SubGraphDataset
from dataset_eval import LinkPredSubGraphDataset

from model_self_attn import GraphTransformer
from model_cross_attn import GraphTransformer as GraphTransformerCrossAttn

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset-path', type=str, required=True)
	parser.add_argument('--split', type=str, choices=['train', 'valid', 'test'], default='valid')
	parser.add_argument('--model-type', type=str, choices=['self-attn', 'cross-attn'], default='cross-attn')
	parser.add_argument('--ckpt-path', type=str, required=True)
	parser.add_argument('--sample-size', type=int, default=20, help='sample size in terms of no. of edges')
	parser.add_argument('--embed-dim', type=int, default=320, help='embedding dim.')
	parser.add_argument('--n-attn-heads', type=int, default=8)
	parser.add_argument('--n-bert-layers', type=int, default=3)
	parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
	parser.add_argument('--beta', type=float, default=0.999, help='beta_2 of adam')
	parser.add_argument('--cuda', action='store_true')
	parser.add_argument('--n-epochs', type=int, default=10)
	parser.add_argument('--neigh-size', type=int, default=10)
	parser.add_argument('--batch-size', type=int, default=1)
	parser.add_argument('--graph-connection', type=str, choices=['type_1', 'type_2'], default='type_3')
	parser.add_argument('--sampling-type', type=str, choices=['bfs', 'rwr', 'khop', 'bfs-complete', 'minerva'], default='rwr')
	parser.add_argument('--seed', type=int, default=231327)
	parser.add_argument('--beam-size', type=int, default=100, help='beam size for Minerva model decoding')
	parser.add_argument('--debug', action='store_true')
	parser.add_argument("--add-segment-embed", action="store_true", default=False)
	parser.add_argument('--add-inverse-rels', action='store_true', help='add inverse relations for query tower')
	parser.add_argument('--label-smoothing', type=float, default=0.0)
	parser.add_argument('--hidden-dropout-prob', type=float, default=0.1)
	parser.add_argument('--attention-probs-dropout-prob', type=float, default=0.1)
	
	# Bert model args
	

	args = parser.parse_args()

	device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

	# set seed
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	dataset = SubGraphDataset(args.dataset_path, 'train', args.neigh_size, args.sampling_type, args.graph_connection, device, False, args)

	my_collator_eval = MyCollatorCustom(dataset.triples_dataset.entity2id['<pad>'], dataset.triples_dataset.relation2id['<pad>'], include_edge=True, add_segment_embed=args.add_segment_embed)

	val_dataset_head = LinkPredSubGraphDataset(args.dataset_path, args.split, 'head', args.sample_size, args.neigh_size, args.sampling_type, args.graph_connection, device, args)
	val_dataloader_head = DataLoader(val_dataset_head, batch_size=16, collate_fn=my_collator_eval)

	val_dataset_tail = LinkPredSubGraphDataset(args.dataset_path, args.split, 'tail', args.sample_size, args.neigh_size, args.sampling_type, args.graph_connection, device, args)
	val_dataloader_tail = DataLoader(val_dataset_tail, batch_size=16, collate_fn=my_collator_eval)

	if args.model_type == 'self-attn':
		model = GraphTransformer(dataset.num_entities, dataset.num_relations, args).to(device)
	else:
		model = GraphTransformerCrossAttn(dataset.num_entities, dataset.num_relations, args).to(device)

	model.load_state_dict(torch.load(args.ckpt_path), strict=False)
	model.eval()
	
	with torch.no_grad():
		right_results = {} # head is corrupted

		for batch_id, batch in enumerate(tqdm(val_dataloader_head)):
			# read a batch of triples
			# sub, rel, obj = batch['edge']
			sub, rel, obj = batch['edge'][:, 0], batch['edge'][:, 1], batch['edge'][:, 2]
			label = batch['label'].to(device)
			# print('sub = {}'.format(sub))
			# print('rel = {}'.format(rel))
			# print('obj = {}'.format(obj))

			# do the forward pass

			if 'segment_ids' not in batch:
				outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), token_type_ids=batch['token_type_ids'].to(device), ent_masked_lm_labels=batch['ent_masked_lm_labels'].to(device), ent_len_tensor=batch['ent_len_tensor'].to(device), ent_ids=batch['ent_ids'].to(device), rel_ids=batch['rel_ids'].to(device))
			else:
				outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), token_type_ids=batch['token_type_ids'].to(device), ent_masked_lm_labels=batch['ent_masked_lm_labels'].to(device), ent_len_tensor=batch['ent_len_tensor'].to(device), ent_ids=batch['ent_ids'].to(device), rel_ids=batch['rel_ids'].to(device), segment_ids=batch['segment_ids'].to(device))

			pred = outputs['ent_logits'].squeeze(1)

			b_range			= torch.arange(pred.size()[0], device=device)
			target_pred		= pred[b_range, sub] # get score of tail ent in (sub, rel, obj) triple [batch]
			pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred) # [batch, num_ent] at indices which corr to 'sub' position, put a big negative no. AT other places it remains unchanged.
			pred[b_range, sub] 	= target_pred # assign it back for the ground truth object

			ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, sub]
			# print('ranks = {}'.format(ranks))
			# print('target_pred = {}'.format(target_pred))

			ranks 			= ranks.float()
			right_results['count']	= torch.numel(ranks) 		+ right_results.get('count', 0.0)
			right_results['mr']		= torch.sum(ranks).item() 	+ right_results.get('mr',    0.0)
			right_results['mrr']		= torch.sum(1.0/ranks).item()   + right_results.get('mrr',   0.0)
			# print(right_results)
			for k in range(10):
				right_results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + right_results.get('hits@{}'.format(k+1), 0.0)

			if args.debug and batch_id>100:
				break

	with torch.no_grad():
		left_results = {} # tail is corrupted

		for batch_id, batch in enumerate(tqdm(val_dataloader_tail)):

			# read a batch of triples
			# sub, rel, obj = batch['edge']
			sub, rel, obj = batch['edge'][:, 0], batch['edge'][:, 1], batch['edge'][:, 2]
			label = batch['label'].to(device)
			
			# do the forward pass
			if 'segment_ids' not in batch:
				outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), token_type_ids=batch['token_type_ids'].to(device), ent_masked_lm_labels=batch['ent_masked_lm_labels'].to(device), ent_len_tensor=batch['ent_len_tensor'].to(device), ent_ids=batch['ent_ids'].to(device), rel_ids=batch['rel_ids'].to(device))
			else:
				outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), token_type_ids=batch['token_type_ids'].to(device), ent_masked_lm_labels=batch['ent_masked_lm_labels'].to(device), ent_len_tensor=batch['ent_len_tensor'].to(device), ent_ids=batch['ent_ids'].to(device), rel_ids=batch['rel_ids'].to(device), segment_ids=batch['segment_ids'].to(device))
			
			pred = outputs['ent_logits'].squeeze(1)

			b_range			= torch.arange(pred.size()[0], device=device)
			target_pred		= pred[b_range, obj] # get score of tail ent in (sub, rel, obj) triple [batch]
			pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred) # [batch, num_ent] at indices which corr to 'obj' position, put a big negative no. AT other places it remains unchanged.
			pred[b_range, obj] 	= target_pred # assign it back for the ground truth object

			ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
			# print('ranks = {}'.format(ranks))

			ranks 			= ranks.float()
			left_results['count']	= torch.numel(ranks) 		+ left_results.get('count', 0.0)
			left_results['mr']		= torch.sum(ranks).item() 	+ left_results.get('mr',    0.0)
			left_results['mrr']		= torch.sum(1.0/ranks).item()   + left_results.get('mrr',   0.0)
			for k in range(10):
				left_results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + left_results.get('hits@{}'.format(k+1), 0.0)

			if args.debug and batch_id>100:
				break

	results = get_combined_results(left_results, right_results)
	print('[{}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(args.split, results['left_mrr'], results['right_mrr'], results['mrr']))
	print(results)

if __name__=='__main__':
	main()
