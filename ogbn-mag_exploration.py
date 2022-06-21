import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import graph_exploration_methods
import greedy_clustering_subgraphs
from networkx import find_cliques

if __name__=="__main__":
    G = nx.read_gpickle("ogbn-mag.multidigraph.gpickle")
    #print("num_unique_nodes_in_3clique",graph_exploration_methods.get_num_unique_nodes_in_3clique(G))


    # num_edge_in_kcliques, edge_set_frequency_in_kcliques = graph_exploration_methods.get_stat_in_kclique(G)
    # print("num_edge_in_kcliques: ",num_edge_in_kcliques )
    # max_k_k_clique = max(list(map(edge_set_frequency_in_kcliques.__getitem__, edge_set_frequency_in_kcliques.keys())))
    # print(plt.hist(list(map(edge_set_frequency_in_kcliques.__getitem__, edge_set_frequency_in_kcliques.keys())),
    #                bins=max_k_k_clique))


    #print(plt.hist(list(map(edge_set_frequency_in_kcliques.__getitem__,edge_set_frequency_in_kcliques.keys()))))
    #print(graph_exploration_methods.get_kclique_length_hist(G,bins=None))
    #graph_exploration_methods.generate_and_save_correlation_matrix(G, "correlation_mag.png")




    G_undirected = G.to_undirected()
    kcliques = [item for item in find_cliques(G_undirected)]
    result_cluster = greedy_clustering_subgraphs.greedy_clustering(G,kcliques)
    with open("test_greedy_cluster.mag.debug_only.log",'w') as fd:
        for cluster in result_cluster:
            fd.write("{} , {}".format(len(cluster),greedy_clustering_subgraphs.count_edges_in_a_cluster_in_naive_adjacency_matrix(cluster,G)))
