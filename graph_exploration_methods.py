import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
from networkx import find_cliques
import thirdparty.bear.utils as utils

def get_num_unique_nodes_in_3clique(G_input):
    G=G_input.to_undirected()
    kcliques = [item for item in find_cliques(G)]
    node_in_3cliques = set()

    for kclique in kcliques:
        if len(kclique) !=3:
            continue
        for node_idx in kclique:

            node_in_3cliques.add(node_idx)
    return len(node_in_3cliques)

def get_num_unique_node_pairs_in_3clique(G_input):
    G=G_input.to_undirected()
    kcliques = [item for item in find_cliques(G)]
    edge_set_in_3cliques = set()

    for kclique in kcliques:
        if len(kclique) !=3:
            continue
        for src_idx in kclique:
            for dest_idx in kclique:
                # if src_idx == dest_idx:
                #     continue
                if G_input.get_edge_data(src_idx, dest_idx) is None:
                    continue
                if (src_idx, dest_idx) in edge_set_in_3cliques:
                    continue
                edge_set_in_3cliques.add((src_idx, dest_idx))
    return len(edge_set_in_3cliques)

def get_stat_in_kclique(G_input):
    G= G_input.to_undirected()
    kcliques = [item for item in find_cliques(G)]
    num_edge_in_kcliques = 0
    edge_set_frequency_in_kcliques=dict()

    for kclique in kcliques:
        if len(kclique) == 2: # skip 2-clique for now as to my current understanding it seems trivial 2-clique, i.e., endpoints of edges.
            continue
        for src_idx in kclique:
            for dest_idx in kclique:
                if G_input.get_edge_data(src_idx, dest_idx) is None:
                    continue
                # if src_idx==dest_idx:
                #     continue
                if (src_idx, dest_idx) in edge_set_frequency_in_kcliques:
                    edge_set_frequency_in_kcliques[(src_idx, dest_idx)] += 1
                    continue
                edge_set_frequency_in_kcliques[(src_idx, dest_idx)]=1
                num_edge_in_kcliques += len(G_input.get_edge_data(src_idx, dest_idx))
    return num_edge_in_kcliques, edge_set_frequency_in_kcliques

def get_kclique_length_hist(G_input,bins=None):
    G=G_input.to_undirected()
    kcliques = [item for item in find_cliques(G)]
    kclique_lengths = [len(item) for item in kcliques]
    if bins is None:
        bins = max(kclique_lengths)
    return plt.hist(kclique_lengths,bins=bins)

def _generate_and_save_correlation_matrix(G, filename, correlation_height_or_width, node_mapping):
    G_new = nx.relabel_nodes(G, node_mapping)

    correlation_matrix = np.zeros([correlation_height_or_width, correlation_height_or_width])
    for edge in G_new.edges():
        idx_i = int(math.floor(edge[0] / len(G.nodes()) * correlation_height_or_width))
        idx_j = int(math.floor(edge[1] / len(G.nodes()) * correlation_height_or_width))
        correlation_matrix[idx_i, idx_j] += 1
    with open(filename + '.npy', 'wb') as fd:
        np.save(fd, correlation_matrix)  # save the correlation matrix
    plt.imsave(filename, correlation_matrix, cmap='hot')

def generate_and_save_correlation_matrix(G, filename, correlation_height_or_width = 100):
    # G is networkx graph
    node_mapping = dict(zip(sorted(G.nodes(), key=lambda k: -G.degree[k]), range(len(G.nodes()))))
    _generate_and_save_correlation_matrix(G, filename, correlation_height_or_width, node_mapping)

def analyze_graph_after_slashburn(G):
    adjacency_matrix = nx.to_scipy_sparse_matrix(G,format='coo')
    perm_H, wing = utils.slashburn(adjacency_matrix)

    #node_mapping = dict(zip(range(len(G.nodes())),perm_H))
    node_mapping = dict(zip(perm_H,range(len(G.nodes()))))
    with open("slashburn.npy",'wb') as fd:
        np.save(fd,perm_H)
    G_new = nx.relabel_nodes(G, node_mapping)
    return G_new