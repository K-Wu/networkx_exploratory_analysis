import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

import graph_exploration_methods

if __name__ == '__main__':
    G = nx.read_gpickle("ogbn-wikikg2.multidigraph.gpickle")
    #print("num_unique_nodes_in_3clique", graph_exploration_methods.get_num_unique_nodes_in_3clique(G))
    num_edge_in_kcliques, edge_set_frequency_in_kcliques = graph_exploration_methods.get_stat_in_kclique(G)
    print("num_edge_in_kcliques: ", num_edge_in_kcliques)
    max_k_k_clique = max(list(map(edge_set_frequency_in_kcliques.__getitem__, edge_set_frequency_in_kcliques.keys())))
    print(plt.hist(list(map(edge_set_frequency_in_kcliques.__getitem__, edge_set_frequency_in_kcliques.keys())),bins=max_k_k_clique))
    #print(graph_exploration_methods.get_kclique_length_hist(G,bins=None))
    #G_new = graph_exploration_methods.analyze_graph_after_slashburn(G)
    #graph_exploration_methods.generate_and_save_correlation_matrix(G_new, "correlation_wikikg2_remapped.png")
