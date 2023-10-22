import numpy as np
import scipy.sparse as ssp
import random

"""All functions in this file are from  dgl.contrib.data.knowledge_graph"""


def _bfs_relational(adj, adj_list, roots, max_nodes_per_hop=None, n_neigh_per_node=None, triples_dataset=None):
    """
    BFS for graphs.
    Modified from dgl.contrib.data.knowledge_graph to accomodate node sampling
    adj_list: contains (edge_id, node_id) pairs
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = set()
        for node in current_lvl:
            if n_neigh_per_node:
                n_edge_samples = min(n_neigh_per_node, len(adj_list[node]))
            else:
                n_edge_samples = len(adj_list[node])

            # remove all nodes already covered
            node_samples = [x for x in random.sample(adj_list[node].tolist(), n_edge_samples) if x[1] not in visited]
            node_samples = [tuple(x) for x in node_samples]
            
            # node_samples = [x for x in adj_list[node] if x[1] not in visited]

            next_lvl.update(node_samples)

        # support max_nodes_per_hop
        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        # select only node ids from next_lvl
        current_lvl = set.union(set([x[1] for x in next_lvl]))


def _get_neighbors(adj, adj_list, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors.
    Directly copied from dgl.contrib.data.knowledge_graph"""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors


def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)
