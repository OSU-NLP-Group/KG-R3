#!/usr/bin/env python
import sys, os
import argparse
import numpy as np
from tqdm  import tqdm
import json
import random
import networkx as nx

import pickle
import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import horovod.torch as hvd
from hvd_optimizer import DistributedOptimizer

from utils.gen_utils import *
from dataset import SubGraphDataset
from dataset_eval import LinkPredSubGraphDataset

from model_self_attn import GraphTransformer
from model_cross_attn import GraphTransformer as GraphTransformerCrossAttn

from pytorchtools import EarlyStopping
from transformers.models.bert.modeling_bert import BertConfig
from utils.bert_optim import Adamax


class filesDataset(torch.utils.data.Dataset):
	def __init__(self, path, length):
		self.path = path
		self.length = length

	def __getitem__(self, idx):
		return pickle.load(open(os.path.join(self.path, '{}.pkl'.format(idx)), 'rb'))

	def __len__(self):
		return self.length

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset-path', type=str, required=True)
	parser.add_argument('--save-dir', type=str, required=True)
	parser.add_argument('--ckpt-path', type=str, default=None)
	parser.add_argument('--sample-size', type=int, default=20, help='sample size in terms of no. of edges of subgraph')
	parser.add_argument('--embed-dim', type=int, default=320, help='embedding dim.')
	parser.add_argument('--n-attn-heads', type=int, default=8)
	parser.add_argument('--n-bert-layers', type=int, default=3)
	parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
	parser.add_argument('--warmup', type=float, default=0.1, help='percentage of steps for warmup')
	parser.add_argument('--beta', type=float, default=0.999, help='beta_2 of adam')
	parser.add_argument('--cuda', action='store_true')
	parser.add_argument('--n-epochs', type=int, default=10)
	parser.add_argument('--neigh-size', type=int, default=10)
	parser.add_argument('--batch-size', type=int, default=1, help='per-GPU batch size')
	parser.add_argument('--model-type', type=str, choices=['self-attn', 'cross-attn'], default='cross-attn')
	parser.add_argument('--graph-connection', type=str, choices=['type_1', 'type_5'], default='type_1')
	parser.add_argument('--sampling-type', type=str, choices=['bfs', 'onehop', 'minerva'], default='minerva')
	parser.add_argument('--seed', type=int, default=231327)
	parser.add_argument("--use-descending", action="store_true")
	parser.add_argument('--data-scratch', action='store_true', help='if set, do not use pre-cached data')
	parser.add_argument('--add-segment-embed', action='store_true', help='add segment embedding')
	parser.add_argument('--n-toy-ex', type=int, default=100)
	parser.add_argument('--label-smoothing', type=float, default=0.0)
	parser.add_argument('--eval-every', type=int, default=5)
	parser.add_argument('--num-gpus', type=int, default=1)
	parser.add_argument('--hidden-dropout-prob', type=float, default=0.1)
	parser.add_argument('--attention-probs-dropout-prob', type=float, default=0.1)
	parser.add_argument('--add-inverse-rels', action='store_true', help='add inverse relations for query tower')
	parser.add_argument('--beam-size', type=int, default=100, help='beam size for Minerva model decoding')

	# optimization args
	parser.add_argument('--patience', type=int, default=20)
	parser.add_argument('--debug', action='store_true', help='exit training loop after n_toy_ex batches')
	parser.add_argument('--warmup-steps', type=int, default=10000)
	parser.add_argument("--weight-decay", default=0.01, type=float, help="Weight decay for adamax optimizer")
	parser.add_argument('--gradient-accum-steps', type=int, default=1)
	parser.add_argument('--shuffle-batches', action='store_true')
	parser.add_argument('--num-workers', type=int, default=8, help='no. of workers for dataloader')

	args = parser.parse_args()

	print(args)
	
	device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
	print('device = {}'.format(device))

	if args.num_gpus > 1:
		hvd.init()

		if device == 'cuda':
		    # Horovod: pin GPU to local rank.
		    torch.cuda.set_device(hvd.local_rank())

	logging.basicConfig(level=logging.NOTSET)

	# set seed
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	# turn on anomaly detection
	torch.autograd.set_detect_anomaly(True)

	writer = SummaryWriter(log_dir=args.save_dir)

	dataset = SubGraphDataset(args.dataset_path, 'train', args.neigh_size, args.sampling_type, args.graph_connection, device, False, args)
	my_collator = MyCollatorCustom(dataset.triples_dataset.entity2id['<pad>'], dataset.triples_dataset.relation2id['<pad>'], add_segment_embed=args.add_segment_embed)
	my_collator_eval = MyCollatorCustom(dataset.triples_dataset.entity2id['<pad>'], dataset.triples_dataset.relation2id['<pad>'], include_edge=True, add_segment_embed=args.add_segment_embed)

	val_dataset_head = LinkPredSubGraphDataset(args.dataset_path, 'valid', 'head', args.sample_size, args.neigh_size, args.sampling_type, args.graph_connection, device, args)
	val_dataloader_head = DataLoader(val_dataset_head, batch_size=16, collate_fn=my_collator_eval)

	val_dataset_tail = LinkPredSubGraphDataset(args.dataset_path, 'valid', 'tail', args.sample_size, args.neigh_size, args.sampling_type, args.graph_connection, device, args)
	val_dataloader_tail = DataLoader(val_dataset_tail, batch_size=16, collate_fn=my_collator_eval)

	train_dataloader_file = os.path.join(args.dataset_path, 'train_preproc', '{}_{}_train_{}_{}'.format(args.sampling_type, args.graph_connection, args.batch_size, args.beam_size))

	if not args.add_inverse_rels:
		if not args.add_segment_embed:
			suffix = ''
		else:
			suffix = '_seg'
	else:
		if not args.add_segment_embed:
			suffix = '_inv'
		else:
			suffix = '_seg_inv'

	if len(suffix) > 0:
		train_dataloader_file = train_dataloader_file + suffix

	print(train_dataloader_file)

	if os.path.exists(train_dataloader_file) and not args.data_scratch:
		filedataset = filesDataset(train_dataloader_file, len(os.listdir(train_dataloader_file)))
		if args.num_gpus > 1:
			train_sampler = torch.utils.data.distributed.DistributedSampler(filedataset, num_replicas=hvd.size(), rank=hvd.rank())
			# batch size is per-node
			dataloader = DataLoader(filedataset, sampler=train_sampler, batch_size=1, num_workers=1)
		else:
			dataloader = DataLoader(filedataset, batch_size=1, num_workers=1, shuffle=args.shuffle_batches)
		logging.info('train dataloader loaded...')
	else:
		print('args.num_gpus = {}'.format(args.num_gpus))
		if args.num_gpus > 1:
			train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
			dataloader = DataLoader(dataset, sampler=train_sampler, collate_fn=my_collator) # sorting
		else:
			if not args.debug:
				dataset_stat = SubGraphDataset(args.dataset_path, 'train', args.neigh_size, args.sampling_type, args.graph_connection, device, True, args)
				train_dataset_lens = torch.tensor([sum(dataset_stat[i]) for i in range(len(dataset_stat))])
				train_batch_sample_ids = torch.split(torch.argsort(train_dataset_lens, descending=args.use_descending), split_size_or_sections=args.batch_size)

				dataloader = DataLoader(dataset, batch_sampler=train_batch_sample_ids, collate_fn=my_collator) # sorting
			else:
				dataloader = DataLoader(dataset, collate_fn=my_collator)

	if args.model_type == 'self-attn':
		model = GraphTransformer(dataset.num_entities, dataset.num_relations, args).to(device)
	else:
		model = GraphTransformerCrossAttn(dataset.num_entities, dataset.num_relations, args).to(device)

	# restore from ckpt
	if args.ckpt_path:
		model.load_state_dict(torch.load(args.ckpt_path), strict=False)

	num_params = sum([param.numel() for param_name, param in model.named_parameters()])
	print('num_params = {}'.format(num_params))

	# initialize optimizer
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": args.weight_decay,
		},
		{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
	]
	
	t_total = args.n_epochs * len(dataloader)
	optimizer = Adamax(model.parameters(), lr=args.lr, warmup=args.warmup, schedule='warmup_linear_xdl', t_total=t_total)

	# initialize the early_stopping object
	early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=os.path.join(args.save_dir, 'model.pt'))

	if args.num_gpus > 1:
		optimizer = DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), lr=args.lr, warmup=args.warmup, schedule='warmup_linear_xdl', t_total=t_total)
		hvd.broadcast_parameters(model.state_dict(), root_rank=0)

	for epoch_id in range(args.n_epochs):
		avg_train_loss = 0.0
		avg_train_ent_acc = 0.0

		model.train()

		optimizer.zero_grad()

		if args.shuffle_batches and not args.data_scratch:
			random.shuffle(dataloader)

		for batch_id, batch in enumerate(tqdm(dataloader)):
			
			if os.path.exists(train_dataloader_file) and not args.data_scratch:
				batch = {k:v.squeeze(0) for k,v in batch.items()}

			# outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), token_type_ids=batch['token_type_ids'].to(device), ent_masked_lm_labels=batch['ent_masked_lm_labels'].to(device), rel_masked_lm_labels=batch['rel_masked_lm_labels'].to(device))
			if 'segment_ids' not in batch:
				outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), token_type_ids=batch['token_type_ids'].to(device), ent_masked_lm_labels=batch['ent_masked_lm_labels'].to(device), ent_ids=batch['ent_ids'].to(device), rel_ids=batch['rel_ids'].to(device), ent_len_tensor=batch['ent_len_tensor'].to(device))
			else:
				outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), token_type_ids=batch['token_type_ids'].to(device), ent_masked_lm_labels=batch['ent_masked_lm_labels'].to(device), ent_ids=batch['ent_ids'].to(device), rel_ids=batch['rel_ids'].to(device), ent_len_tensor=batch['ent_len_tensor'].to(device), segment_ids=batch['segment_ids'].to(device))

			loss = outputs['loss']
			
			avg_train_loss += loss.item()

			loss.backward()
			
			if (batch_id + 1) % args.gradient_accum_steps == 0:
				optimizer.step(0)
				optimizer.zero_grad()

			target = batch['ent_masked_lm_labels'].to(device)

			masks = target != -1 # ignore_index = -1
			pred = outputs['entity_pred']
			pred = pred.squeeze(-1)
			acc = torch.sum(torch.eq(pred, target).masked_fill(masks.eq(0), 0)).item()*1.0/torch.sum(masks).item()

			avg_train_ent_acc += acc

			if args.debug and batch_id > args.n_toy_ex:
				break

		avg_train_loss = avg_train_loss/len(dataloader)
		avg_train_ent_acc = avg_train_ent_acc/len(dataloader)

		print('Epoch {}, avg_train_loss = {}'.format(epoch_id, avg_train_loss))
		print('Epoch {}, avg_train_ent_acc = {}'.format(epoch_id, avg_train_ent_acc))

		writer.add_scalar('loss/train', avg_train_loss, epoch_id)
		writer.add_scalar('ent_acc/train', avg_train_ent_acc, epoch_id)

		'''
		Credit for evaluation loop: https://github.com/malllabiisc/CompGCN/blob/master/run.py
		'''

		if (epoch_id+1) % args.eval_every == 0:
			# run validation
			model.eval()

			with torch.no_grad():
				right_results = {} # head is corrupted

				for batch_id, batch in enumerate(tqdm(val_dataloader_head)):

					# read a batch of triples
					sub, rel, obj = batch['edge'][:, 0], batch['edge'][:, 1], batch['edge'][:, 2]

					label = batch['label'].to(device)

					# print(batch)
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


			with torch.no_grad():
				left_results = {} # tail is corrupted

				for batch_id, batch in enumerate(tqdm(val_dataloader_tail)):

					# read a batch of triples
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

			val_results = get_combined_results(left_results, right_results)
			print(val_results)

			writer.add_scalar('mrr/val', val_results['mrr'], epoch_id)
			
			early_stopping(-1*val_results['mrr'], model)
			
			if early_stopping.early_stop:
				print("Early stopping")
				break
	
if __name__=='__main__':
	main()

