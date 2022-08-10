from OutputNPYFileFB15k237 import *


# TODO: use a build complete metadata to make sure the dictionary are not muted

# TODO: reduce repetition by removing found clusters from the original graph
# TODO: we may need to have incoming edge dictionary as well as index to reduce time complexity


def count_edges_in_an_outgoing_edge_dict(outgoing_edge_dict, parallel_edges_enabled_flag):
    # TODO: let's count artificial back edge in this function
    count = 0
    for src_node in outgoing_edge_dict:
        for dst_node in outgoing_edge_dict[src_node]:
            for etype in iterate_etypes(outgoing_edge_dict[src_node][dst_node], parallel_edges_enabled_flag):
                count += 1
    return count


def count_edges_from_src_to_node(edge_or_edges, parallel_edges_enabled_flag):
    # this function inputted element of the outgoing_edge_dict, i.e., outgoing_edge_dict[src_idx][dst_idx},
    # and returens the number of edges from src_idx to dst_idx
    # TODO: let's count artificial back edge in this function
    if not parallel_edges_enabled_flag:
        # if parallel_edges_enabled_flag is True, edge_or_edges is the etype integer
        assert (isinstance(edge_or_edges, int) or isinstance(edge_or_edges, np.integer) or (edge_or_edges is None))
        # In bug-free logic, edge_or_edges is None only when the return value is default value of dictionary,
        # which means the key is not in dictionary
        if edge_or_edges is None:
            return 0
        else:
            return 1
    else:
        # if parallel_edges_enabled_flag is False, edge_or_edges is the etype set
        assert (isinstance(edge_or_edges, set))
        return len(edge_or_edges)


def new_edge_placeholder_from_src_to_dst(parallel_edges_enabled_flag):
    # this is the default value of the outgoing_edge_dict element, i.e., outgoing_edge_dict[src_idx][dst_idx]
    # if no parallel edge, it is a nonexistent etype, i.e., None. If parallel edge is enabled, it is an empty set.
    if not parallel_edges_enabled_flag:
        return None
    else:
        return set()


def add_parallel_edge_from_src_to_dst(old_edge_or_edges, to_add_edge, parallel_edges_enabled_flag):
    # NB: no need to add artificial back edge if the whole graph inputted to the top function has already introduced
    # artificial back edge
    if not parallel_edges_enabled_flag:
        # simply returning the to_add_edge_or_edges, which should be etype integer
        # assert(old_edge_or_edges is None)
        return to_add_edge
    else:
        # return the union of old_edge_or_edges and to_add_edge_or_edges. Both are sets of integer representing etypes.
        old_edge_or_edges.add([etype for etype in to_add_edge])
        # Notice this is an in place operation and do not create new set. This is just for alignment with the
        # no-parallel-edge case, where we need to assign etype as the value of the specific key in outgoing_edge_dict
        return old_edge_or_edges


def iterate_etypes(edge_or_edges, parallel_edges_enabled_flag):
    if not parallel_edges_enabled_flag:
        # edge_or_edges is simply an integer representing etype. packing it to be an iterable
        # assert(isinstance(edge_or_edges, int) or isinstance(edge_or_edges,np.integer))
        return [edge_or_edges]
    else:
        # edge_or_edges should be a set of integers representing etype
        # assert(isinstance(edge_or_edges, set) and len(edge_or_edges)>0)
        return edge_or_edges


def build_outgoing_edge_dictionary(edge_srcs, edge_dsts, edge_types, parallel_edges_enabled_flag):
    edge_dict = dict()
    for edge_idx in range(len(edge_srcs)):
        src = edge_srcs[edge_idx]
        dst = edge_dsts[edge_idx]
        etype = edge_types[edge_idx]
        if src not in edge_dict:
            edge_dict[src] = dict()
        if not parallel_edges_enabled_flag:
            assert (dst not in edge_dict[src])
            edge_dict[src][dst] = etype  # NB: no-parallel-edge simplify: etype is an integer, not a set of integers
        else:
            if dst not in edge_dict[src]:
                edge_dict[src][dst] = set()
            edge_dict[src][dst].add(etype)
    return edge_dict


# # add a copy of (b->a, negative etype) if (a->b, etype) exists in original graph
# # in the end of the day, we use a referential copy, i.e., set of tuple (a,b,etype), to remove fake edges
# def get_referential_tuple_edge_set(outgoing_edge_dict, parallel_edges_enabled_flag):
#     result = set()
#     for src_node in outgoing_edge_dict:
#         for dst_node in outgoing_edge_dict[src_node]:
#             for etype in iterate_etypes(outgoing_edge_dict[src_node][dst_node],parallel_edges_enabled_flag):
#                 result.add((src_node, dst_node, etype))
#     return result

def build_undirected_outgoing_edge_dictionary(outgoing_edge_dict, parallel_edges_enabled_flag):
    undirected_outgoing_edge_dict = dict()
    for src_node in outgoing_edge_dict:
        if src_node not in undirected_outgoing_edge_dict:
            undirected_outgoing_edge_dict[src_node] = dict()
        for dst_node in outgoing_edge_dict[src_node]:
            if dst_node not in undirected_outgoing_edge_dict[src_node]:
                undirected_outgoing_edge_dict[src_node][dst_node] = new_edge_placeholder_from_src_to_dst(
                    parallel_edges_enabled_flag)
            for etype in iterate_etypes(outgoing_edge_dict[src_node][dst_node], parallel_edges_enabled_flag):
                undirected_outgoing_edge_dict[src_node][dst_node] = add_parallel_edge_from_src_to_dst(
                    undirected_outgoing_edge_dict[src_node][dst_node], etype, parallel_edges_enabled_flag)
            # add reverse edge
            if dst_node not in undirected_outgoing_edge_dict:
                undirected_outgoing_edge_dict[dst_node] = dict()
            if src_node not in undirected_outgoing_edge_dict[dst_node]:
                undirected_outgoing_edge_dict[dst_node][src_node] = new_edge_placeholder_from_src_to_dst(
                    parallel_edges_enabled_flag)
            for etype in iterate_etypes(outgoing_edge_dict[src_node][dst_node], parallel_edges_enabled_flag):
                # we use -etype-1 to mark artificial reverse edge. the -1 is to ensure distinct when etype is 0
                undirected_outgoing_edge_dict[src_node][dst_node] = add_parallel_edge_from_src_to_dst(
                    undirected_outgoing_edge_dict[src_node][dst_node], -etype - 1, parallel_edges_enabled_flag)

    return undirected_outgoing_edge_dict


def _nomemo_calc_outgoing_and_incoming_degrees_for_all_nodes(outgoing_edge_dict, parallel_edges_enabled_flag):
    outgoing_degrees = dict()
    incoming_degrees = dict()
    for node in outgoing_edge_dict:
        for dest_node in outgoing_edge_dict[node]:
            incoming_degrees[node] = incoming_degrees.get(node, 0) + count_edges_from_src_to_node(
                outgoing_edge_dict[node].get(dest_node,
                                             new_edge_placeholder_from_src_to_dst(parallel_edges_enabled_flag)),
                parallel_edges_enabled_flag)  # NB: no-parallel-edge simplify: increment by one, not counting set length
            outgoing_degrees[dest_node] = outgoing_degrees.get(dest_node, 0) + count_edges_from_src_to_node(
                outgoing_edge_dict[node].get(dest_node,
                                             new_edge_placeholder_from_src_to_dst(parallel_edges_enabled_flag)),
                parallel_edges_enabled_flag)  # NB: no-parallel-edge simplify: increment by one, not counting set length
    return incoming_degrees, outgoing_degrees


# memo using graph outgoing edge dict as key, and dictionary (node_idx -> degree) as value
MEMO_OUTGOING_DEGREES_DICT = dict()
MEMO_INCOMING_DEGREES_DICT = dict()


def memo_calc_outgoing_and_incoming_degrees_for_all_nodes(outgoing_edge_dict, parallel_edges_enabled_flag):
    assert (id(outgoing_edge_dict) not in MEMO_OUTGOING_DEGREES_DICT)
    assert (id(outgoing_edge_dict) not in MEMO_INCOMING_DEGREES_DICT)
    incoming_degrees, outgoing_degrees = \
        _nomemo_calc_outgoing_and_incoming_degrees_for_all_nodes(outgoing_edge_dict,
                                                                 parallel_edges_enabled_flag)
    MEMO_OUTGOING_DEGREES_DICT[id(outgoing_edge_dict)] = outgoing_degrees
    MEMO_INCOMING_DEGREES_DICT[id(outgoing_edge_dict)] = incoming_degrees
    return incoming_degrees, outgoing_degrees


def remove_subgraph_from_memo(graph_outgoing_edge_dict):
    MEMO_OUTGOING_DEGREES_DICT.pop(id(graph_outgoing_edge_dict), None)
    MEMO_INCOMING_DEGREES_DICT.pop(id(graph_outgoing_edge_dict), None)


def calc_outgoing_degrees_for_all_nodes(outgoing_edge_dict, parallel_edges_enabled_flag, likely_calculation=True):
    if id(outgoing_edge_dict) in MEMO_OUTGOING_DEGREES_DICT:
        if likely_calculation:
            print("WARNING: outgoing_degrees_for_all_nodes returns from memo, meaning potential repetitive calculation")
        return MEMO_OUTGOING_DEGREES_DICT[id(outgoing_edge_dict)]
    else:
        MEMO_OUTGOING_DEGREES_DICT[id(outgoing_edge_dict)] = \
            memo_calc_outgoing_and_incoming_degrees_for_all_nodes(outgoing_edge_dict, parallel_edges_enabled_flag)[1]
        return MEMO_OUTGOING_DEGREES_DICT[id(outgoing_edge_dict)]


def calc_incoming_degrees_for_all_nodes(outgoing_edge_dict, parallel_edges_enabled_flag, likely_calculation=True):
    if id(outgoing_edge_dict) in MEMO_INCOMING_DEGREES_DICT:
        if likely_calculation:
            print("WARNING: incoming_degrees_for_all_nodes returns from memo, meaning potential repetitive calculation")
        return MEMO_INCOMING_DEGREES_DICT[id(outgoing_edge_dict)]
    else:
        MEMO_INCOMING_DEGREES_DICT[id(outgoing_edge_dict)] = \
            memo_calc_outgoing_and_incoming_degrees_for_all_nodes(outgoing_edge_dict, parallel_edges_enabled_flag)[0]
        return MEMO_INCOMING_DEGREES_DICT[id(outgoing_edge_dict)]


def calc_outgoing_degrees(outgoing_edge_dict, src_nodes, parallel_edges_enabled_flag):
    outgoing_degrees = calc_outgoing_degrees_for_all_nodes(outgoing_edge_dict, parallel_edges_enabled_flag, False)
    return sum([outgoing_degrees.get(src, 0) for src in src_nodes])
    # src_nodes = set(src_nodes)
    # result = 0
    # for src_node in src_nodes:
    #     outgoing_edge_dict_for_curr_src_node = outgoing_edge_dict.get(src_node, dict())
    #     for dst in outgoing_edge_dict_for_curr_src_node.keys():
    #         result += len(outgoing_edge_dict_for_curr_src_node[dst])
    # return result


def calc_incoming_degrees(outgoing_edge_dict, dst_nodes, parallel_edges_enabled_flag):
    incoming_degrees = calc_incoming_degrees_for_all_nodes(outgoing_edge_dict, parallel_edges_enabled_flag, False)
    return sum([incoming_degrees.get(dst, 0) for dst in dst_nodes])
    # dst_nodes = set(dst_nodes)
    # result = 0
    # for dst_node in dst_nodes:
    #     for src in outgoing_edge_dict.keys():
    #         result += len(outgoing_edge_dict[src].get(dst_node, set()))
    # return result


def calc_outgoing_degree(outgoing_edge_dict, src_node, parallel_edges_enabled_flag):
    return calc_outgoing_degrees(outgoing_edge_dict, {src_node}, parallel_edges_enabled_flag)


def calc_incoming_degree(outgoing_edge_dict, dst_node, parallel_edges_enabled_flag):
    return calc_incoming_degrees(outgoing_edge_dict, {dst_node}, parallel_edges_enabled_flag)


def trim_nodes_from_outgoing_edge_dict(outgoing_edge_dict, nodes_to_trim):
    # one good side effect of creating a new dict is we can use its address as key to memorize calc degree results
    result = dict()
    for src in outgoing_edge_dict.keys():
        if src not in nodes_to_trim:
            result[src] = dict()
            for dst in outgoing_edge_dict[src].keys():
                if dst not in nodes_to_trim:
                    result[src][dst] = outgoing_edge_dict[src][
                        dst]  # NB: no-parallel-edge simplify: logic same as before
    return result


def simple_trim(outgoing_edge_dict, stopping_density, parallel_edges_enabled_flag):
    # the outgoing_edge_dict could be of the whole graph, or only a neighbourhood clique in it
    nodes_in_input = list(outgoing_edge_dict.keys())

    nodes_in_input.sort(
        key=lambda x: calc_outgoing_degree(outgoing_edge_dict, x, parallel_edges_enabled_flag) + calc_incoming_degree(
            outgoing_edge_dict, x, parallel_edges_enabled_flag))

    nodes_to_trim = set()
    # NB: impl counting total degrees of a graph and store as a metadata
    total_degrees = count_all_nodes_degree_in_outgoing_edge_dict(outgoing_edge_dict, parallel_edges_enabled_flag)

    if (total_degrees + 0.0) / ((len(nodes_in_input) - len(nodes_to_trim)) ** 2) > stopping_density:
        return trim_nodes_from_outgoing_edge_dict(outgoing_edge_dict, nodes_to_trim)

    for idx_ele_in_nodes_in_input in reversed(range(len(nodes_in_input))):
        nodes_to_trim.add(nodes_in_input[idx_ele_in_nodes_in_input])
        # TODO: impl counting total degrees of a graph and store as a metadata
        degrees_to_trim = calc_incoming_degrees(outgoing_edge_dict, nodes_to_trim, parallel_edges_enabled_flag) + \
                          calc_outgoing_degrees(outgoing_edge_dict, nodes_to_trim, parallel_edges_enabled_flag)
        if (total_degrees - degrees_to_trim + 0.0) / (
                (len(nodes_in_input) - len(nodes_to_trim)) ** 2) > stopping_density:
            return trim_nodes_from_outgoing_edge_dict(outgoing_edge_dict, nodes_to_trim)

    return dict()  # empty dict, meaning there is nothing left since this region is too sparse


def iterative_trim(outgoing_edge_dict, stopping_density, maximal_iter_num, parallel_edges_enabled_flag):
    # the outgoing_edge_dict could be of the whole graph, or only a neighbourhood clique in it
    for idx_iter in range(maximal_iter_num):
        # TODO: return the total graph during generating edge dict after trim
        outgoing_edge_dict = simple_trim(outgoing_edge_dict, stopping_density, parallel_edges_enabled_flag)
        # NB: impl counting total degrees of a graph and store as a metadata
        if ((count_all_nodes_degree_in_outgoing_edge_dict(outgoing_edge_dict, parallel_edges_enabled_flag)) + 0.0) / (
                len(get_all_nodes_involved_in_outgoing_edge_dict(outgoing_edge_dict)) ** 2) > stopping_density:
            return outgoing_edge_dict
        remove_subgraph_from_memo(outgoing_edge_dict)

    return dict()  # empty dict, meaning there is nothing left since this region is too sparse


def induce_subgraph_by_nodes(graph_outgoing_edge_dict, nodes, parallel_edges_enabled_flag):
    edges_in_induced_graph = dict()
    for src_node in nodes:
        graph_outgoing_edge_dict_for_curr_src_node = graph_outgoing_edge_dict.get(src_node, dict())
        for dst_node in nodes:
            if dst_node in graph_outgoing_edge_dict_for_curr_src_node:
                # edges_in_induced_graph.add((src_node, dst_node,tuple(sorted([etype for etype in graph_outgoing_edge_dict_for_curr_src_node[dst_node]]))))
                if src_node not in edges_in_induced_graph:
                    edges_in_induced_graph[src_node] = dict()
                if dst_node not in edges_in_induced_graph[src_node]:
                    edges_in_induced_graph[src_node][dst_node] = new_edge_placeholder_from_src_to_dst(
                        parallel_edges_enabled_flag)  # NB: no-parallel-edge simplify: placeholder is None,not empty set
                for etype in iterate_etypes(graph_outgoing_edge_dict_for_curr_src_node[dst_node],
                                            parallel_edges_enabled_flag):
                    edges_in_induced_graph[src_node][dst_node] = add_parallel_edge_from_src_to_dst(
                        edges_in_induced_graph[src_node][dst_node], etype,
                        parallel_edges_enabled_flag)  # NB: no-parallel-edge simplify: assign etype to the element
    return edges_in_induced_graph


def get_all_nodes_involved_in_outgoing_edge_dict(outgoing_edge_dict):
    # both source and destination of an edge are put into the result
    result = set()
    for src in outgoing_edge_dict.keys():
        result.add(src)
        for dst in outgoing_edge_dict[src].keys():
            result.add(dst)
    return result


def calc_incoming_degrees_all_nodes(outgoing_edge_dict, parallel_edges_enabled_flag):
    all_nodes_set = get_all_nodes_involved_in_outgoing_edge_dict(outgoing_edge_dict)
    return calc_incoming_degrees(outgoing_edge_dict, all_nodes_set, parallel_edges_enabled_flag)


def calc_outgoing_degrees_all_nodes(outgoing_edge_dict, parallel_edges_enabled_flag):
    all_nodes_set = get_all_nodes_involved_in_outgoing_edge_dict(outgoing_edge_dict)
    return calc_outgoing_degrees(outgoing_edge_dict, all_nodes_set, parallel_edges_enabled_flag)


def count_all_nodes_degree_in_outgoing_edge_dict(outgoing_edge_dict, parallel_edges_enabled_flag,
                                                 likely_calculation=True):
    outgoing_degrees = calc_outgoing_degrees_for_all_nodes(outgoing_edge_dict, parallel_edges_enabled_flag,
                                                           likely_calculation)
    incoming_degrees = calc_incoming_degrees_for_all_nodes(outgoing_edge_dict, parallel_edges_enabled_flag,
                                                           likely_calculation)
    return sum(outgoing_degrees.values()) + sum(incoming_degrees.values())


def expand_outgoing_neighbour(graph_outgoing_edge_dict, src_node_set):
    result = set()
    for src_node in src_node_set:
        graph_outgoing_edge_dict_for_curr_src_node = graph_outgoing_edge_dict.get(src_node, dict())
        result.add(src_node)
        for dst_node in graph_outgoing_edge_dict_for_curr_src_node.keys():
            result.add(dst_node)
    return result


# TODO|NB: change outgoing edge dictionary to involve (neighbour<-node) as well as if it is non-directed graph
# NB: we made this change because nodes heavily connected could be destination instead of purely source
def spawn_neighbour_cluster(graph_outgoing_edge_dict, src_nodes_set, num_iter, parallel_edges_enabled_flag):
    assert (num_iter >= 0)
    if num_iter == 0:
        return induce_subgraph_by_nodes(graph_outgoing_edge_dict, src_nodes_set, parallel_edges_enabled_flag)

    for idx_iter in range(num_iter):
        # this step adds nodes and its reachable neighbours, i.e., (node -> reachable neighbour), into src_node_set
        src_nodes_set = expand_outgoing_neighbour(graph_outgoing_edge_dict, src_nodes_set)
        # this step induces the subgraph of nodes in src_nodes_set,
        # i.e., nodes in the last iteration and their reachable neighbours
        result_edges_in_induced_subgraph = induce_subgraph_by_nodes(graph_outgoing_edge_dict, src_nodes_set,
                                                                    parallel_edges_enabled_flag)
    return result_edges_in_induced_subgraph


def neighbour_greedily_find_clusters(edge_srcs, edge_dsts, edge_types, num_node_trial_threshold_portion,
                                     neighbour_cluster_num_spawn_iter, neighbour_cluster_stopping_density,
                                     parallel_edges_enabled_flag):
    # first build dictionary of nodes and their neighbours
    graph_outgoing_edge_dict = build_outgoing_edge_dictionary(edge_srcs, edge_dsts, edge_types,
                                                              parallel_edges_enabled_flag)
    graph_outgoing_edge_dict = build_undirected_outgoing_edge_dictionary(graph_outgoing_edge_dict,
                                                                         parallel_edges_enabled_flag)

    # pack and unpack to make sure both list and numpy (n) or (1,n) or (n,1) vectors input could be accepted
    all_nodes_in_graph = {*edge_srcs}.union([*edge_dsts])

    total_nodes_trialed_with_empty_result = 0
    curr_remaining_node_trailed_with_empty_result = 0
    remaining_nodes_working_list = list(all_nodes_in_graph)

    clusters_found = []

    # stoping threshold #TODO: add another calibre i.e. edge portion in cluster vs in graph
    # 1. if there is no new cluster found after threshold*num_nodes or all nodes are trialed, then stop
    while 1:
        if total_nodes_trialed_with_empty_result >= len(all_nodes_in_graph) * num_node_trial_threshold_portion:
            break
        if len(remaining_nodes_working_list) == 0 or curr_remaining_node_trailed_with_empty_result >= len(
                remaining_nodes_working_list):
            break
        if curr_remaining_node_trailed_with_empty_result == 0:
            # NB: calculate all nodes' degrees in the memo and use it in this sorting
            remaining_nodes_working_list = sorted(remaining_nodes_working_list,
                                                  key=lambda x: calc_incoming_degree(graph_outgoing_edge_dict, x,
                                                                                     parallel_edges_enabled_flag) +
                                                                calc_outgoing_degree(graph_outgoing_edge_dict, x,
                                                                                     parallel_edges_enabled_flag),
                                                  reverse=True)

        curr_iter_cluster_edges_in_induced_subgraph = spawn_neighbour_cluster(graph_outgoing_edge_dict, {
            remaining_nodes_working_list[curr_remaining_node_trailed_with_empty_result]},
                                                                              neighbour_cluster_num_spawn_iter,
                                                                              parallel_edges_enabled_flag)
        curr_iter_cluster_outgoing_edge_dict = simple_trim(curr_iter_cluster_edges_in_induced_subgraph,
                                                           neighbour_cluster_stopping_density,
                                                           parallel_edges_enabled_flag)
        if len(curr_iter_cluster_outgoing_edge_dict) != 0:
            curr_iter_cluster_nodes_in_induced_subgraph = get_all_nodes_involved_in_outgoing_edge_dict(
                curr_iter_cluster_outgoing_edge_dict)
            clusters_found.append(curr_iter_cluster_nodes_in_induced_subgraph)
            remaining_nodes_working_list = list(
                all_nodes_in_graph.difference(curr_iter_cluster_nodes_in_induced_subgraph))
            curr_remaining_node_trailed_with_empty_result = 0
        else:
            curr_remaining_node_trailed_with_empty_result += 1
            total_nodes_trialed_with_empty_result += 1


# potential neighbour cluster encoding: CSR+node_idx
# metadata: num_clusters
# cluster 0: num_nnzs, num_rows, node_idxes [node_idx_0, node_idx_1, ...], row_ptr, col_idxes, data
# cluster 1: num_nnzs, num_rows, node_idxes [node_idx_0, node_idx_1, ...], row_ptr, col_idxes, data

if __name__ == "__main__":
    edges_srcs, edges_dsts, edges_types = load_ogbn_mag()
    neighbour_greedily_find_clusters(edges_srcs, edges_dsts, edges_types, num_node_trial_threshold_portion=0.1,
                                     neighbour_cluster_num_spawn_iter=2, neighbour_cluster_stopping_density=0.1,
                                     parallel_edges_enabled_flag=False)
    pass
