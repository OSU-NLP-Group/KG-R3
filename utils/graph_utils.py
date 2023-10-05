import statistics
import numpy as np
import scipy.sparse as ssp
import torch
import networkx as nx
import dgl
import pickle
from collections import defaultdict
from copy import deepcopy


def serialize(data):
    data_tuple = tuple(data.values())
    return pickle.dumps(data_tuple)

def deserialize(data):
    data_tuple = pickle.loads(data)
    keys = ('n1', 'n2', 'r_label', 'nodes', 'edges')
    return dict(zip(keys, data_tuple))

def deserialize_dir(data):
    data_tuple = pickle.loads(data)
    keys = ('n1', 'n2', 'r_label', 'nodes', 'fw_edges', 'bw_edges')
    return dict(zip(keys, data_tuple))

def deserialize_prior(data):
    data_tuple = pickle.loads(data)
    keys = ('n1', 'n2', 'r_label', 'nodes', 'edges', 'ent_prior_scores', 'edge_prior_scores')
    return dict(zip(keys, data_tuple))

def deserialize_seg(data):
    data_tuple = pickle.loads(data)
    keys = ('n1', 'n2', 'r_label', 'nodes', 'edges', 'ent_segment_scores')
    return dict(zip(keys, data_tuple))

def deserialize_seg_dir(data):
    data_tuple = pickle.loads(data)
    keys = ('n1', 'n2', 'r_label', 'nodes', 'ent_segment_scores', 'fw_edges', 'bw_edges')
    return dict(zip(keys, data_tuple))

def get_edge_count(adj_list):
    count = []
    for adj in adj_list:
        count.append(len(adj.tocoo().row.tolist()))
    return np.array(count)


def incidence_matrix(adj_list):
    '''
    adj_list: List of sparse adjacency matrices
    '''

    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
        # print(adjcoo.row.tolist())
        # print(adjcoo.col.tolist())
        # print(adjcoo.data.tolist())

    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def ssp_to_torch(A, device, dense=False):
    '''
    A : Sparse adjacency matrix
    '''
    idx = torch.LongTensor([A.tocoo().row, A.tocoo().col])
    dat = torch.FloatTensor(A.tocoo().data)
    A = torch.sparse.FloatTensor(idx, dat, torch.Size([A.shape[0], A.shape[1]])).to(device=device)
    return A


def ssp_multigraph_to_dgl(graph, n_feats=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """

    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    for rel, adj in enumerate(graph):
        # Convert adjacency matrix to tuples for nx0
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)

    # make dgl graph
    g_dgl = dgl.DGLGraph(multigraph=True)
    g_dgl.from_networkx(g_nx, edge_attrs=['type'])
    # add node features
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl


def collate_dgl(samples):
    # The input `samples` is a list of pairs
    graphs_pos, g_labels_pos, r_labels_pos, graphs_negs, g_labels_negs, r_labels_negs = map(list, zip(*samples))
    batched_graph_pos = dgl.batch(graphs_pos)

    graphs_neg = [item for sublist in graphs_negs for item in sublist]
    g_labels_neg = [item for sublist in g_labels_negs for item in sublist]
    r_labels_neg = [item for sublist in r_labels_negs for item in sublist]

    batched_graph_neg = dgl.batch(graphs_neg)
    return (batched_graph_pos, r_labels_pos), g_labels_pos, (batched_graph_neg, r_labels_neg), g_labels_neg


def move_batch_to_device_dgl(batch, device):
    ((g_dgl_pos, r_labels_pos), targets_pos, (g_dgl_neg, r_labels_neg), targets_neg) = batch

    targets_pos = torch.LongTensor(targets_pos).to(device=device)
    r_labels_pos = torch.LongTensor(r_labels_pos).to(device=device)

    targets_neg = torch.LongTensor(targets_neg).to(device=device)
    r_labels_neg = torch.LongTensor(r_labels_neg).to(device=device)

    g_dgl_pos = send_graph_to_device(g_dgl_pos, device)
    g_dgl_neg = send_graph_to_device(g_dgl_neg, device)

    return ((g_dgl_pos, r_labels_pos), targets_pos, (g_dgl_neg, r_labels_neg), targets_neg)


def send_graph_to_device(g, device):
    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device)

    # edges
    labels = g.edge_attr_schemes()
    for l in labels.keys():
        g.edata[l] = g.edata.pop(l).to(device)
    return g

#  The following three functions are modified from networks source codes to
#  accomodate diameter and radius for dirercted graphs


def eccentricity(G):
    e = {}
    for n in G.nbunch_iter():
        length = nx.single_source_shortest_path_length(G, n)
        e[n] = max(length.values())
    return e


def radius(G):
    e = eccentricity(G)
    e = np.where(np.array(list(e.values())) > 0, list(e.values()), np.inf)
    return min(e)


def diameter(G):
    e = eccentricity(G)
    return max(e.values())

class AdjNode:
    def __init__(self, data, rel='<rel>'):
        self.vertex = data
        self.rel = rel
        self.next = None

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        # self.graph = [None] * self.V
        self.graph = defaultdict(lambda: None)
 
    # Function to add an edge in an undirected graph
    def add_edge(self, src, dest, rel):
        # Adding the node to the source node
        node = AdjNode(dest, rel)
        node.next = self.graph[src]
        self.graph[src] = node
 
        # Adding the source node to the destination as
        # it is the undirected graph
        if src != dest:
            node = AdjNode(src, rel)
            node.next = self.graph[dest]
            self.graph[dest] = node
 
    # Function to print the graph
    def print_graph(self):
        for i in self.graph.keys():
            print("Adjacency list of vertex {}\n head".format(i), end="")
            temp = self.graph[i]
            while temp:
                print(" -> {}".format((temp.vertex, temp.rel)), end="")
                temp = temp.next
            print(" \n")

    @classmethod
    def print_path(cls, path, dataset):
        out_str = ''
        for node in path[:-1]:
            if node[1] != -1:
                out_str = out_str + '{} -> {} -> '.format(dataset.triples_dataset.id2relation[node[1]], dataset.mid2title.get(dataset.triples_dataset.id2entity[node[0]], dataset.triples_dataset.id2entity[node[0]]))
            else:
                out_str = out_str + '{} -> '.format(dataset.mid2title.get(dataset.triples_dataset.id2entity[node[0]], dataset.triples_dataset.id2entity[node[0]]))
        out_str = out_str + '{} -> {}'.format(dataset.triples_dataset.id2relation[path[-1][1]], dataset.mid2title.get(dataset.triples_dataset.id2entity[path[-1][0]], dataset.triples_dataset.id2entity[path[-1][0]]))
        return out_str

    # @profile
    def get_all_paths(self, start_node, end_node, n_edge_samples, n_hops):
        global all_simple_paths
        global current_path
        global visited

        current_path = []
        all_simple_paths = []

        visited = defaultdict(lambda: False)

        self.dfs(start_node, -1, end_node, n_edge_samples, n_hops, start_node, end_node)
        return all_simple_paths

    # @profile
    def dfs(self, from_node, from_rel, to_node, n_edge_samples, n_hops, start_node, end_node):
        global current_path
        global all_simple_paths
        global visited

        if len(all_simple_paths)>=n_edge_samples: # always mark from_node as unvisited and pop from current_path before exiting
            return
        # print('from_node = {}'.format(from_node))
        # print('current_path = {}'.format(len(current_path)))
        # print('current_path = {}'.format(current_path))
        # print('all_simple_paths = {}'.format(len(all_simple_paths)))
        # print('all_simple_paths = {}'.format(all_simple_paths))
        # print('visited = {}'.format(visited))

        if visited[from_node]:
            return
        
        if start_node != end_node:
            visited[from_node] = True

        if len(current_path)>0:
            assert from_rel != -1 # not the start node
            # print(current_path)
            tmp = self.graph[current_path[-1][0]]
            
            while tmp.vertex != from_node or tmp.rel != from_rel: # navigating to from_node in adjacency list of last node in current_path
                tmp = tmp.next
            rel = tmp.rel
        else:
            rel = -1

        if len(current_path)>n_hops+1:
            visited[from_node] = False # always mark from_node as unvisited and pop from current_path before exiting (didn't add yet)
            return

        current_path.append((from_node, rel))

        if from_node==to_node:
            if start_node != end_node:
                # print('flag 1')
                all_simple_paths.append(deepcopy(current_path))
                visited[from_node] = False
                current_path.pop(-1)
                return
            elif len(current_path)>1:
                all_simple_paths.append(deepcopy(current_path))
                visited[from_node] = False
                current_path.pop(-1)
                return


        if len(all_simple_paths)>=n_edge_samples: # always mark from_node as unvisited and pop from current_path before exiting
            # current_path = [] # why do we do this?
            return

        temp = self.graph[from_node]
        while temp:
            # print('temp = {}'.format(temp.vertex))
            self.dfs(temp.vertex, temp.rel, to_node, n_edge_samples, n_hops, start_node, end_node)
            temp = temp.next

        if len(current_path)>0:
            current_path.pop(-1)

        visited[from_node] = False
