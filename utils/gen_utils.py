import sys, os
import numpy as np
import random
import collections
import torch
import dgl
from scipy import sparse as sp
import networkx as nx
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])



def get_adj_and_degrees(num_nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    adj_list = [[] for _ in range(num_nodes)]
    for i,triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees

def sample_subgraph_bfs(adj_list, n_nodes, n_triplets, sample_size, n_neigh_per_node):
    '''
    sample a central node first; then sample upto a fixed neighborhood; then repeat for adjacent nodes till we have req. no of edges
    '''
    edges = np.zeros((sample_size), dtype=np.int32)
    picked = np.array([False for _ in range(n_triplets)])
    visited = np.array([False for _ in range(n_nodes)])
    queue = []

    central_node = np.random.choice(range(n_nodes))
    queue.append(central_node)
    visited[central_node] = True

    i = 0 # count on edges

    while i < sample_size: # less than required no of edges
        # sample upto a fixed neighborhood of the central node
        if len(queue)==0:
            break

        s = queue.pop(0)

        n_edge_samples = min(n_neigh_per_node, adj_list[s].shape[0])

        for j in np.random.choice(range(adj_list[s].shape[0]), n_edge_samples, replace=False):
            chosen_edge = adj_list[s][j, :]
            if i >= sample_size:
                break

            if visited[chosen_edge[1]] == False and picked[chosen_edge[0]] == False and adj_list[s].shape[0]>0:
                queue.append(chosen_edge[1])
                visited[chosen_edge[1]] = True
                edge_number = chosen_edge[0]
                picked[edge_number] = True

                edges[i] = edge_number
                i += 1

    return edges

def sample_subgraph_with_central_node_bfs(adj_list, central_node, n_nodes, n_triplets, sample_size, n_neigh_per_node):
    '''
    sample a central node first; then sample upto a fixed neighborhood; then repeat for adjacent nodes till we have req. no of edges
    '''
    edges = np.zeros((sample_size), dtype=np.int32)
    picked = np.array([False for _ in range(n_triplets)])
    visited = np.array([False for _ in range(n_nodes)])
    queue = []
    i = 0 # count on edges

    queue.append(central_node)
    visited[central_node] = True

    while i < sample_size: # less than required no of edges
        # sample upto a fixed neighborhood of the central node
        if len(queue)==0:
            break

        s = queue.pop(0)

        n_edge_samples = min(n_neigh_per_node, adj_list[s].shape[0])

        for j in np.random.choice(range(adj_list[s].shape[0]), n_edge_samples):
            chosen_edge = adj_list[s][j, :]
            
            if i >= sample_size:
                break

            if visited[chosen_edge[1]] == False and picked[chosen_edge[0]] == False and adj_list[s].shape[0]>0:
                queue.append(chosen_edge[1])
                visited[chosen_edge[1]] = True
                edge_number = chosen_edge[0]
                picked[edge_number] = True

                edges[i] = edge_number
                i += 1

    # return the sampled edges; maybe zero in some corner cases
    return edges[:i]

def sample_subgraph_with_central_node_onehop(adj_list, central_node, n_nodes, n_triplets, sample_size):
    '''
    sample a central node first; then sample upto a fixed neighborhood; then repeat for adjacent nodes till we have req. no of edges
    '''
    edges = np.zeros((sample_size), dtype=np.int32)
    picked = np.array([False for _ in range(n_triplets)])
    visited = np.array([False for _ in range(n_nodes)])
    i = 0 # count on edges

    visited[central_node] = True

    # sample upto a fixed neighborhood of the central node

    s = central_node

    n_edge_samples = adj_list[s].shape[0]

    for j in np.random.choice(range(adj_list[s].shape[0]), n_edge_samples):
        chosen_edge = adj_list[s][j, :]
        
        if i >= sample_size:
            break

        if visited[chosen_edge[1]] == False and picked[chosen_edge[0]] == False and adj_list[s].shape[0]>0:
            visited[chosen_edge[1]] = True
            edge_number = chosen_edge[0]
            picked[edge_number] = True

            edges[i] = edge_number
            i += 1

    # return the sampled edges; maybe zero in some corner cases
    return edges[:i]

def sample_subgraph_with_start_node_bfs(adj_list, node_to_id_map, start_node, n_nodes, n_triplets):
    '''
    sample a start node first; then sample upto a fixed neighborhood; then repeat for adjacent nodes till we have req. no of edges
    '''
    picked = np.array([False for _ in range(n_triplets)])
    visited = np.array([False for _ in range(n_nodes)])
    queue = []

    if start_node not in node_to_id_map:
        return visited

    queue.append(node_to_id_map[start_node])
    visited[node_to_id_map[start_node]] = True

    while len(queue)>0: # less than required no of edges
        # sample upto a fixed neighborhood of the start node
        s = queue.pop(0)

        for j in range(len(adj_list[s])):
            chosen_edge = adj_list[s][j]

            if visited[chosen_edge[1]] == False and picked[chosen_edge[0]] == False and len(adj_list[s])>0:
                queue.append(chosen_edge[1])
                visited[chosen_edge[1]] = True
                edge_number = chosen_edge[0]
                picked[edge_number] = True

    return visited

def get_combined_results(left_results, right_results):
    results = {}
    count   = float(left_results['count'])

    results['left_mr']  = round(left_results ['mr'] /count, 5)
    results['left_mrr'] = round(left_results ['mrr']/count, 5)
    results['right_mr'] = round(right_results['mr'] /count, 5)
    results['right_mrr']    = round(right_results['mrr']/count, 5)
    results['mr']       = round((left_results['mr']  + right_results['mr']) /(2*count), 5)
    results['mrr']      = round((left_results['mrr'] + right_results['mrr'])/(2*count), 5)

    for k in range(10):
        results['left_hits@{}'.format(k+1)] = round(left_results ['hits@{}'.format(k+1)]/count, 5)
        results['right_hits@{}'.format(k+1)]    = round(right_results['hits@{}'.format(k+1)]/count, 5)
        results['hits@{}'.format(k+1)]      = round((left_results['hits@{}'.format(k+1)] + right_results['hits@{}'.format(k+1)])/(2*count), 5)
    return results

'''
Credit: huggingFace
'''
def get_polynomial_decay_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by `lr_end`, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        lr_end (:obj:`float`, `optional`, defaults to 1e-7):
            The end LR.
        power (:obj:`float`, `optional`, defaults to 1.0):
            Power factor.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Note: `power` defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """

    lr_init = optimizer.defaults["lr"]
    assert lr_init > lr_end, f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})"

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class MyCollatorCustom(object):
    '''
    applicable for model which uses query rep + single layer subgraph rep.
    '''
    def __init__(self, ent_pad_idx, rel_pad_idx, include_edges=False, include_edge=False, add_segment_embed=False):
        self.ent_pad_idx = ent_pad_idx
        self.rel_pad_idx = rel_pad_idx
        self.include_edges = include_edges
        self.include_edge = include_edge
        self.add_segment_embed = add_segment_embed

    def __call__(self, batch):
        '''
        batch contains 'entity_ids', 'relation_ids', 'attention_mask', 'token_type_ids', 'ent_masked_lm_labels', 'rel_masked_lm_labels'
        '''
        entity_ids_list = [x['entity_ids'] for x in batch]
        relation_ids_list = [x['relation_ids'] for x in batch]
        attention_mask_list = [x['attention_mask'] for x in batch]
        # token_type_ids_list = [x['token_type_ids'] for x in batch]
        ent_masked_lm_labels_list = [x['ent_masked_lm_labels'] for x in batch]
        rel_masked_lm_labels_list = [x['rel_masked_lm_labels'] for x in batch]

        if self.include_edges:
            edges_list = [x['edges'] for x in batch]

        if self.include_edge:
            edge_list = [torch.tensor(x['edge']) for x in batch]
            label_list = [x['label'] for x in batch]
        
        batch_size = len(batch)
        
        max_ent_len = max([len(x) for x in entity_ids_list])
        max_rel_len = max([len(x) for x in relation_ids_list])

        max_token_len = max_ent_len + max_rel_len
        
        input_ids_tensor = batch[0]['attention_mask'].new_ones(batch_size, max_token_len, dtype=torch.long)
        input_ids_tensor[:, :max_ent_len] = self.ent_pad_idx
        input_ids_tensor[:, max_ent_len : max_ent_len+max_rel_len] = self.rel_pad_idx

        attention_mask_tensor = batch[0]['attention_mask'].new_zeros(batch_size, max_token_len, max_token_len, dtype=torch.long)

        token_type_ids_tensor = batch[0]['token_type_ids'].new_zeros(batch_size, max_token_len, dtype=torch.long)
        if self.add_segment_embed:
            segment_ids_tensor = batch[0]['token_type_ids'].new_zeros(batch_size, max_token_len, dtype=torch.long)

        ent_masked_lm_labels_tensor = torch.tensor(ent_masked_lm_labels_list)

        token_type_ids_tensor[:, :max_ent_len] = 0
        token_type_ids_tensor[:, max_ent_len : max_ent_len+max_rel_len] = 1

        ent_len_tensor = batch[0]['attention_mask'].new_zeros(batch_size, dtype=torch.long)
        rel_len_tensor = batch[0]['attention_mask'].new_zeros(batch_size, dtype=torch.long)

        for i in range(batch_size):
            # print('i = {}'.format(i))
            ent_len = len(batch[i]['entity_ids'])
            rel_len = len(batch[i]['relation_ids'])

            ent_len_tensor[i] = ent_len
            rel_len_tensor[i] = rel_len

            input_ids_tensor[i, :ent_len] = torch.tensor(batch[i]['entity_ids'])
            input_ids_tensor[i, max_ent_len : max_ent_len+rel_len] = torch.tensor(batch[i]['relation_ids'])

            if self.add_segment_embed:
                segment_ids_tensor[i, :ent_len] = torch.tensor(batch[i]['segment_ids'][:ent_len])
            
            assert batch[i]['attention_mask'].size() == (ent_len+rel_len, ent_len+rel_len), (batch[i]['attention_mask'].size(), ent_len, rel_len)

            attention_mask_tensor[i, :ent_len, :ent_len] = batch[i]['attention_mask'][:ent_len, :ent_len]

            attention_mask_tensor[i, max_ent_len : max_ent_len+rel_len, max_ent_len : max_ent_len+rel_len] = batch[i]['attention_mask'][ent_len : ent_len+rel_len, ent_len : ent_len+rel_len]

            attention_mask_tensor[i, max_ent_len : max_ent_len+rel_len, :ent_len] = batch[i]['attention_mask'][ent_len : ent_len+rel_len, :ent_len]

            attention_mask_tensor[i, :ent_len, max_ent_len : max_ent_len+rel_len] = batch[i]['attention_mask'][:ent_len, ent_len : ent_len+rel_len]

        collate_batch = {'input_ids': input_ids_tensor, 'attention_mask': attention_mask_tensor, 'token_type_ids': token_type_ids_tensor, 'ent_masked_lm_labels': ent_masked_lm_labels_tensor, 'ent_len_tensor':ent_len_tensor, 'rel_len_tensor':rel_len_tensor}
        
        if self.include_edges:
            collate_batch['edges'] = edges_list
        if self.include_edge:
            collate_batch['edge'] = torch.stack(edge_list)
            collate_batch['label'] = torch.stack([torch.tensor(x) for x in label_list])

        if self.add_segment_embed:
            collate_batch['segment_ids'] = segment_ids_tensor

        collate_batch['ent_ids'] = torch.stack([torch.tensor(x['ent']) for x in batch])
        collate_batch['rel_ids'] = torch.stack([torch.tensor(x['rel']) for x in batch])

        return collate_batch

