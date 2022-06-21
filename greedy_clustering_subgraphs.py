import networkx as nx
import queue
from networkx import find_cliques

def count_edges_in_a_cluster_in_naive_adjacency_matrix(node_in_cluster, G):
    result = 0
    for src_node in node_in_cluster:
        for dst_node in node_in_cluster:
            if not G.get_edge_data(src_node, dst_node) is None:
                result+=1
    return result

def greedy_clustering(G,k_cliques,size=16):
    work_queue=queue.PriorityQueue()
    #enqueuing jobs
    result_cluster = list()
    nodes_already_in_result = set()

    # enqueuing all k_cliques
    for k_clique in k_cliques:
        work_queue.put((-len(k_clique),set(k_clique)))
    curr_cluster = set()
    while not work_queue.empty():
        if len(curr_cluster)<size:
            _, node_candidates_to_merge = work_queue.get()
            node_to_merge = set()
            for node in node_candidates_to_merge:
                if node in nodes_already_in_result or node in curr_cluster:
                    continue
                node_to_merge.add(node)
            if len(node_to_merge)<=2 and count_edges_in_a_cluster_in_naive_adjacency_matrix(node_to_merge.union(curr_cluster),G)-count_edges_in_a_cluster_in_naive_adjacency_matrix(curr_cluster,G)<=1:
                continue # discard worthless clusters
            curr_cluster=curr_cluster.union(node_to_merge)
        else:
            result_cluster.append(curr_cluster)
            for node in curr_cluster:
                nodes_already_in_result.add(node)

            curr_cluster=set()

    result_cluster.append(curr_cluster)
    for node in curr_cluster:
        nodes_already_in_result.add(node)
    return result_cluster

if __name__=="__main__":
    G = nx.read_gpickle("ogbn-wikikg2.multidigraph.gpickle")
    G_undirected = G.to_undirected()
    kcliques = [item for item in find_cliques(G_undirected)]
    result_cluster = greedy_clustering(G,kcliques)
    with open("test_greedy_cluster.debug_only.log",'w') as fd:
        for cluster in result_cluster:
            fd.write("{} , {}".format(len(cluster),count_edges_in_a_cluster_in_naive_adjacency_matrix(cluster,G)))

