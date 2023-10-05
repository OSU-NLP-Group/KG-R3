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

def get_adj(num_nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    adj_list = [[] for _ in range(num_nodes)]
    for i,triplet in enumerate(triplets):
        adj_list[triplet[0]].append((i, triplet[2]))
        adj_list[triplet[2]].append((i, triplet[0]))

    # adj_list = [np.array(a) for a in adj_list]
    return adj_list

# @profile
def get_adj_subset(nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    adj_list = [[] for _ in range(len(nodes))]
    node_to_id_map = {n:i for i,n in enumerate(nodes)}

    for i,triplet in enumerate(triplets):
        adj_list[node_to_id_map[triplet[0]]].append((i, node_to_id_map[triplet[2]]))
        adj_list[node_to_id_map[triplet[2]]].append((i, node_to_id_map[triplet[0]]))

    # adj_list = [np.array(a) for a in adj_list]
    return adj_list, node_to_id_map

def get_bernoulli_mask(shape, prob, device):
    return torch.bernoulli(torch.full(shape, prob, device=device)).bool()
    
def sample_edge_neighborhood(adj_list, degrees, n_triplets, sample_size):
    """Sample edges by neighborhood expansion.

    This guarantees that the sampled edges form a connected graph, which
    may help deeper GNNs that require information from more than one hop.
    """
    edges = np.zeros((sample_size), dtype=np.int32)

    #initialize
    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in range(n_triplets)])
    seen = np.array([False for _ in degrees])

    for i in range(0, sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        probabilities = (weights) / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                         p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    return edges

def sample_subgraph_bfs(adj_list, n_nodes, n_triplets, sample_size, n_neigh_per_node):
    '''
    sample a central node first; then sample upto a fixed neighborhood; then repeat for adjacent nodes till we have req. no of edges
    '''
    edges = np.zeros((sample_size), dtype=np.int32)
    picked = np.array([False for _ in range(n_triplets)])
    visited = np.array([False for _ in range(n_nodes)])
    queue = []

    # print('visited = {}'.format(visited.shape))
    # print('picked = {}'.format(picked.shape))

    central_node = np.random.choice(range(n_nodes))
    queue.append(central_node)
    visited[central_node] = True

    i = 0 # count on edges
    # print([x.shape for x in np.nonzero(edges)])

    while i < sample_size: # less than required no of edges
        # print('i = {}'.format(i))
        # sample upto a fixed neighborhood of the central node
        if len(queue)==0:
            break

        s = queue.pop(0)

        # print('adj_list[s].shape = {}'.format(adj_list[s].shape))
        n_edge_samples = min(n_neigh_per_node, adj_list[s].shape[0])
        # print('n_edge_samples = {}'.format(n_edge_samples))

        for j in np.random.choice(range(adj_list[s].shape[0]), n_edge_samples, replace=False):
        # for j in range(adj_list[s].shape[0]):
            # print('j = {}'.format(j))
            chosen_edge = adj_list[s][j, :]
            # print('len(np.flatnonzero(edges)) = {}'.format(len(np.flatnonzero(edges))))
            if i >= sample_size:
                break

            if visited[chosen_edge[1]] == False and picked[chosen_edge[0]] == False and adj_list[s].shape[0]>0:
                queue.append(chosen_edge[1])
                visited[chosen_edge[1]] = True
                edge_number = chosen_edge[0]
                picked[edge_number] = True
                # print('i = {}'.format(i))

                edges[i] = edge_number
                i += 1

        # print('len(queue) = {}'.format(len(queue)))

    return edges

def sample_subgraph_rwr(adj_list, n_triplets, sample_size, prob_restart):
    '''
    sampling strategy: random walks with restarts
    '''
    n_tries = 0
    path = random_walk(adj_list, n_triplets, path_length=sample_size, alpha=prob_restart, start=None)
    while len(path)==0 and n_tries < 10:
        n_tries += 1
        path = random_walk(adj_list, n_triplets, path_length=sample_size, alpha=prob_restart, start=None)

    edges = np.asarray(path)
    # print('edges = {}'.format(edges.shape))

    return edges

def random_walk(adj_list, n_triplets, path_length, alpha=0, start=None):
    """ Returns a truncated random walk.
        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
        path contains edge ids
    """
    picked = np.array([False for _ in range(n_triplets)])

    if start:
        curr = start
    else:
        # Sampling is uniform w.r.t V, and not w.r.t E
        curr = random.choice(list(range(len(adj_list))))

    # print('start at node {}'.format(curr))

    path = []
    # print('start node = {}'.format(curr))

    n_tries = 0

    while len(path) < path_length and n_tries < 1000:
        # print('curr = {}'.format(curr))
        n_tries += 1

        if len(adj_list[curr]) > 0:
            if random.random() >= alpha:
                # neigh_list = [x for x in adj_list[curr] if picked[x[0]] == False]
                # if len(neigh_list)==0:
                    # continue # eventually it will restart
                chosen_node = random.choice(adj_list[curr])
                
                if not picked[chosen_node[0]]:                
                    path.append(chosen_node) # add (edge id, node id) to path
                    # print('{} -> {}'.format(curr, chosen_node[1]))
                    # print('added edge {}'.format(dataset_triples[chosen_node[0]]))
                    picked[chosen_node[0]] = True

                curr = chosen_node[1] # assign new node to curr

                prev_path_nodes = [x[1] for x in path[:-1]]
                for node in adj_list[curr]:
                    if len(path) < path_length and node[1] in prev_path_nodes and picked[node[0]] == False:
                        path.append(node)
                        # print('add edge {} -> {}'.format(curr, node[1]))
                        # print('added edge {}'.format(dataset_triples[node[0]]))
                        picked[node[0]] = True
            else:
                # print('path = {}'.format(path))
                if len(path) > 0: # o/w curr is the initial node, no need to change
                    curr = path[0][1] # go back to initial node
                    # print('restart at {}'.format(curr))
        else:
            if len(path) > 0: # o/w curr is the initial node, no need to change
                curr = path[0][1] # go back to initial node
                # print('restart at {}'.format(curr))

    # print('*** END ***')
    # print(len(path))
    # print('n_tries = {}'.format(n_tries))

    path = [x[0] for x in path] # only select the edge ids (omit node ids)
    return path

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

    # print([x.shape for x in np.nonzero(edges)])

    while i < sample_size: # less than required no of edges
        # sample upto a fixed neighborhood of the central node
        if len(queue)==0:
            break

        s = queue.pop(0)

        # print('adj_list[s].shape = {}'.format(adj_list[s].shape))
        n_edge_samples = min(n_neigh_per_node, adj_list[s].shape[0])

        # print('n_edge_samples = {}'.format(n_edge_samples))

        for j in np.random.choice(range(adj_list[s].shape[0]), n_edge_samples):
        # for j in range(adj_list[s].shape[0]):
            # print('j = {}'.format(j))
            chosen_edge = adj_list[s][j, :]
            # print('chosen_edge = {}'.format(chosen_edge))
            
            # print('len(np.flatnonzero(edges)) = {}'.format(len(np.flatnonzero(edges))))
            if i >= sample_size:
                break

            if visited[chosen_edge[1]] == False and picked[chosen_edge[0]] == False and adj_list[s].shape[0]>0:
                queue.append(chosen_edge[1])
                visited[chosen_edge[1]] = True
                edge_number = chosen_edge[0]
                picked[edge_number] = True
                # print('i = {}'.format(i))

                edges[i] = edge_number
                i += 1

    # print('len(queue) = {}'.format(len(queue)))
    # print('i = {}'.format(i))
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

    # print([x.shape for x in np.nonzero(edges)])

    # sample upto a fixed neighborhood of the central node

    s = central_node

    # print('adj_list[s].shape = {}'.format(adj_list[s].shape))
    n_edge_samples = adj_list[s].shape[0]

    # print('n_edge_samples = {}'.format(n_edge_samples))

    for j in np.random.choice(range(adj_list[s].shape[0]), n_edge_samples):
    # for j in range(adj_list[s].shape[0]):
        # print('j = {}'.format(j))
        chosen_edge = adj_list[s][j, :]
        # print('chosen_edge = {}'.format(chosen_edge))
        
        # print('len(np.flatnonzero(edges)) = {}'.format(len(np.flatnonzero(edges))))
        if i >= sample_size:
            break

        if visited[chosen_edge[1]] == False and picked[chosen_edge[0]] == False and adj_list[s].shape[0]>0:
            visited[chosen_edge[1]] = True
            edge_number = chosen_edge[0]
            picked[edge_number] = True
            # print('i = {}'.format(i))

            edges[i] = edge_number
            i += 1

    # print('len(queue) = {}'.format(len(queue)))
    # print('i = {}'.format(i))
    # return the sampled edges; maybe zero in some corner cases
    return edges[:i]

def sample_subgraph_with_central_node_rwr(adj_list, central_node, n_triplets, sample_size, prob_restart):
    '''
    sampling strategy: random walks with restarts
    '''
    path = random_walk(adj_list, n_triplets, path_length=sample_size, alpha=prob_restart, start=central_node)
    while len(path)==0:
        path = random_walk(adj_list, n_triplets, path_length=sample_size, alpha=prob_restart, start=central_node)

    edges = np.asarray(path)
    # print('edges = {}'.format(edges.shape))

    return edges

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

        # print('adj_list[s].shape = {}'.format(adj_list[s].shape))

        for j in range(len(adj_list[s])):
        # for j in range(adj_list[s].shape[0]):
            # print('j = {}'.format(j))
            chosen_edge = adj_list[s][j]
            # print('chosen_edge = {}'.format(chosen_edge))
            # print('len(np.flatnonzero(edges)) = {}'.format(len(np.flatnonzero(edges))))

            if visited[chosen_edge[1]] == False and picked[chosen_edge[0]] == False and len(adj_list[s])>0:
                queue.append(chosen_edge[1])
                visited[chosen_edge[1]] = True
                edge_number = chosen_edge[0]
                picked[edge_number] = True

    return visited

def create_mlm_labels(tokens, mask_index, vocab_size, masked_lm_prob=0.15, max_predictions_per_seq=15):
    # cand_indexes indices of tokens which aren't special tokens
    cand_indexes = []

    '''
    if mask_index == 50264:  # indicates word nodes
        special_tokens = [0, 1, 2, 3]  # 0: <s>, 1: <pad>, 2: </s>, 3: <unk>
    else:
        special_tokens = [0, 1]  # 0: <unk> 1: <pad>
    '''
    special_tokens = []

    for (i, token) in enumerate(tokens):
        if token in special_tokens:
            continue
        cand_indexes.append(i)

    random.shuffle(cand_indexes)
    output_tokens = list(tokens)
    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    masked_labels = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_labels) >= num_to_predict:
            break
        else:
            if index in covered_indexes:
                continue
            covered_indexes.add(index)
            if random.random() < 0.8:
                masked_token = mask_index  # replace with [MASK] token 80% of time
            else:
                if random.random() < 0.5:
                    masked_token = tokens[index] # replace with same token 10% of time
                else:
                    masked_token = random.randint(0, vocab_size - 1) # replace with a random token 10% of time

        output_tokens[index] = masked_token
        masked_labels.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_labels = sorted(masked_labels, key=lambda x: x.index)
    masked_lm_positions = []
    masked_lm_labels = []

    for p in masked_labels:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    masked_labels = np.ones(len(tokens), dtype=int) * -1
    masked_labels[masked_lm_positions] = masked_lm_labels
    masked_labels = list(masked_labels)
    return output_tokens, masked_labels

def create_single_ent_mlm_labels(tokens, entity, mask_index, vocab_size):
    '''
    entity: id of entity to be masked
    '''
    # print('tokens = {}'.format(tokens))
    assert entity in tokens, (entity, tokens)
    cand_index = tokens.index(entity)

    # random.shuffle(cand_indexes)
    output_tokens = list(tokens)
    masked_labels = []
    covered_indexes = set()

    masked_token = mask_index 
    output_tokens[cand_index] = masked_token
    masked_labels.append(MaskedLmInstance(index=cand_index, label=tokens[cand_index]))

    masked_lm_positions = []
    masked_lm_labels = []

    for p in masked_labels:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    masked_labels = np.ones(len(tokens), dtype=int) * -1
    masked_labels[masked_lm_positions] = masked_lm_labels
    masked_labels = list(masked_labels)
    return output_tokens, masked_labels, cand_index

def create_single_ent_mlm_labels_given_index(tokens, cand_index, mask_index, vocab_size):
    '''
    entity: id of entity to be masked
    '''
    # print('tokens = {}'.format(tokens))

    # random.shuffle(cand_indexes)
    output_tokens = list(tokens)
    masked_labels = []
    covered_indexes = set()

    masked_token = mask_index 
    output_tokens[cand_index] = masked_token
    masked_labels.append(MaskedLmInstance(index=cand_index, label=tokens[cand_index]))

    masked_lm_positions = []
    masked_lm_labels = []

    for p in masked_labels:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    masked_labels = np.ones(len(tokens), dtype=int) * -1
    masked_labels[masked_lm_positions] = masked_lm_labels
    masked_labels = list(masked_labels)
    return output_tokens, masked_labels

def negative_sampling_weight(entity2id, ent_freq, pwr=0.75):
    ef = []
    for i, ent in enumerate(entity2id.keys()):
        assert entity2id[ent] == i
        ef.append(ent_freq[ent])
    # freq = np.array([self.ent_freq[ent] for ent in self.ent_vocab.keys()])
    ef = np.array(ef)
    ef = ef / ef.sum()
    ef = np.power(ef, pwr)
    ef = ef / ef.sum()
    return torch.FloatTensor(ef)


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

def get_combined_results_micro(left_results, right_results):
    results = {}
    count_left   = float(left_results['count'])
    count_right   = float(right_results['count'])

    if count_left>0:
        results['left_mr']  = round(left_results ['mr'] /count_left, 5)
        results['left_mrr'] = round(left_results ['mrr']/count_left, 5)
    else:
        results['left_mr']  = 0.0
        results['left_mrr'] = 0.0

    if count_left>0:
        results['right_mr'] = round(right_results['mr'] /count_right, 5)
        results['right_mrr']    = round(right_results['mrr']/count_right, 5)
    else:
        results['right_mr'] = 0.0
        results['right_mrr'] = 0.0

    results['mr']       = round((left_results['mr']  + right_results['mr']) /(count_left + count_right), 5)
    results['mrr']      = round((left_results['mrr'] + right_results['mrr'])/(count_left + count_right), 5)

    for k in range(10):
        results['left_hits@{}'.format(k+1)] = round(left_results ['hits@{}'.format(k+1)]/count_left, 5)
        results['right_hits@{}'.format(k+1)]    = round(right_results['hits@{}'.format(k+1)]/count_right, 5)
        results['hits@{}'.format(k+1)]      = round((left_results['hits@{}'.format(k+1)] + right_results['hits@{}'.format(k+1)])/(count_left + count_right), 5)
    return results

def get_combined_results_micro_2(left_results, right_results):
    results = {}
    count_left   = float(left_results['count'])
    count_right   = float(right_results['count'])

    if count_left>0:
        results['left_mr']  = round(left_results ['mr'] /count_left, 5)
        results['left_mrr'] = round(left_results ['mrr']/count_left, 5)
    else:
        results['left_mr']  = 0.0
        results['left_mrr'] = 0.0

    if count_left>0:
        results['right_mr'] = round(right_results['mr'] /count_right, 5)
        results['right_mrr']    = round(right_results['mrr']/count_right, 5)
    else:
        results['right_mr'] = 0.0
        results['right_mrr'] = 0.0

    results['mr']       = round((left_results['mr']  + right_results['mr']) /(count_left + count_right), 5)
    results['mrr']      = round((left_results['mrr'] + right_results['mrr'])/(count_left + count_right), 5)

    for k in [0, 2, 9]:
        results['left_hits@{}'.format(k+1)] = round(left_results ['hits@{}'.format(k+1)]/count_left, 5)
        results['right_hits@{}'.format(k+1)]    = round(right_results['hits@{}'.format(k+1)]/count_right, 5)
        results['hits@{}'.format(k+1)]      = round((left_results['hits@{}'.format(k+1)] + right_results['hits@{}'.format(k+1)])/(count_left + count_right), 5)
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

        # print('max_ent_len = {}'.format(max_ent_len))
        # print('max_rel_len = {}'.format(max_rel_len))

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
            # print(segment_ids_tensor)

        # print('input_ids_tensor = {}'.format(input_ids_tensor[0, :]))
        # print('token_type_ids_tensor = {}'.format(token_type_ids_tensor[0, :]))
        # print('attention_mask_tensor = {}'.format(attention_mask_tensor[0, :]))

        collate_batch['ent_ids'] = torch.stack([torch.tensor(x['ent']) for x in batch])
        collate_batch['rel_ids'] = torch.stack([torch.tensor(x['rel']) for x in batch])

        # print([(k, v.size()) for k,v in collate_batch.items()])
        # sys.exit(0)

        return collate_batch

