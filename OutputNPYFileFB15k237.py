# import torch as th
import numpy as np
from matplotlib import pyplot


# def output_npy_file_ogbn_wikikg2():
#     from ogb.linkproppred import LinkPropPredDataset
#     dataset = LinkPropPredDataset(name='ogbl-wikikg2')
#     # num nodes:  2500604, num edges: 16109182
#     num_edges = len(dataset.graph['edge_index'][0])
#     edges_srcs = np.empty((num_edges,), dtype=np.int64)
#     edges_dsts = np.empty((num_edges,), dtype=np.int64)
#     edges_types = np.empty((num_edges,), dtype=np.int64)
#     for edge_idx in range(len(dataset.graph['edge_index'][0])):
#         edges_srcs[edge_idx] = dataset.graph['edge_index'][0][edge_idx]
#         edges_dsts[edge_idx] = dataset.graph['edge_index'][1][edge_idx]
#         edges_types[edge_idx] = dataset.graph['edge_reltype'][edge_idx][0]
#         assert (len(dataset.graph['edge_reltype'][edge_idx]) == 1)
#     np.save("ogbn-wikikg2.coo.srcs.npy", edges_srcs)
#     np.save("ogbn-wikikg2.coo.dsts.npy", edges_dsts)
#     np.save("ogbn-wikikg2.coo.etypes.npy", edges_types)f
#     return edges_srcs, edges_dsts, edges_types

def output_npy_file_ogbn_wikikg2(sorted=False, sorted_by_src=False, transposed=False):

    from ogb.linkproppred import LinkPropPredDataset
    dataset = LinkPropPredDataset(name='ogbl-wikikg2')
    graph = dataset[0]
    # graph is a built-in dictionary
    # no node types provided. Assuming only one node type
    edges_srcs = graph['edge_index'][0]
    edges_dsts = graph['edge_index'][1]
    edges_etypes = graph['edge_reltype'].flatten()
    edge_referential_eids = np.arange(len(edges_srcs), dtype=np.int64)
    transposed_prefix = 'transposed.' if transposed else ''
    if transposed:
        edges_srcs, edges_dsts = edges_dsts, edges_srcs
    if sorted:
        if sorted_by_src:
            edges_srcs, edges_dsts, edges_etypes, edge_referential_eids = sort_coo_by_src_outgoing_edges(edges_srcs,
                                                                                                         edges_dsts,
                                                                                                         edges_etypes,
                                                                                                         edge_referential_eids,
                                                                                                         graph['num_nodes'])
            np.save(transposed_prefix + "ogbn-wikikg2.coo.sorted.by_srcs_outgoing_freq.srcs.npy", edges_srcs)
            np.save(transposed_prefix + "ogbn-wikikg2.coo.sorted.by_srcs_outgoing_freq.dsts.npy", edges_dsts)
            np.save(transposed_prefix + "ogbn-wikikg2.coo.sorted.by_srcs_outgoing_freq.etypes.npy", edges_etypes)
            np.save(transposed_prefix + "ogbn-wikikg2.coo.sorted.by_srcs_outgoing_freq.referential_eids.npy",
                    edge_referential_eids)
        else:
            edges_srcs, edges_dsts, edges_etypes, edge_referential_eids = sort_coo_by_etype(edges_srcs, edges_dsts,
                                                                                            edges_etypes,
                                                                                            edge_referential_eids)
            # TODO: store in int32 to save space
            np.save(transposed_prefix + "wikikg2.coo.sorted.srcs.npy", edges_srcs)  # ,dtype= np.int32)
            np.save(transposed_prefix + "wikikg2.coo.sorted.dsts.npy", edges_dsts)  # ,dtype= np.int32)
            np.save(transposed_prefix + "wikikg2.coo.sorted.etypes.npy", edges_etypes)  # ,dtype= np.int32)
            np.save(transposed_prefix + "wikikg2.coo.sorted.referential_eids.npy",
                    edge_referential_eids)  # ,dtype= np.int32)
    else:
        # TODO: store in int32 to save space
        np.save(transposed_prefix + "wikikg2.coo.srcs.npy", edges_srcs)
        np.save(transposed_prefix + "wikikg2.coo.dsts.npy", edges_dsts)
        np.save(transposed_prefix + "wikikg2.coo.etypes.npy", edges_etypes)
        np.save(transposed_prefix + "wikikg2.coo.referential_eids.npy", edge_referential_eids)
    return edges_srcs, edges_dsts, edges_etypes

def get_src_dest_pair_for_each_etype(edges_srcs, edges_dsts, edges_etypes):
    src_dest_pairs_for_each_etype = {}
    for edge_idx in range(len(edges_srcs)):
        src = edges_srcs[edge_idx]
        dst = edges_dsts[edge_idx]
        etype = edges_etypes[edge_idx]
        if etype not in src_dest_pairs_for_each_etype:
            src_dest_pairs_for_each_etype[etype] = set()
        src_dest_pairs_for_each_etype[etype].add((src, dst))
    return src_dest_pairs_for_each_etype
def output_npy_file_fb15k237(sorted=False, sorted_by_src=False, transposed=False):

    from dgl.data import FB15k237Dataset
    dataset = FB15k237Dataset()
    graph = dataset[0]
    # only one node type
    # num_nodes: 14541, num_edges:620232
    edges_srcs, edges_dsts = graph.edges()
    edges_srcs= edges_srcs.numpy()
    edges_dsts= edges_dsts.numpy()
    edges_etypes = graph.edata['etype'].detach().numpy()
    edge_referential_eids = np.arange(len(edges_srcs), dtype=np.int64)
    transposed_prefix = 'transposed.' if transposed else ''
    if transposed:
        edges_srcs, edges_dsts = edges_dsts, edges_srcs
    # TODO: store in int32 to save space
    if sorted:
        if sorted_by_src:
            edges_srcs, edges_dsts, edges_etypes, edge_referential_eids = sort_coo_by_src_outgoing_edges(edges_srcs,
                                                                                                         edges_dsts,
                                                                                                         edges_etypes,
                                                                                                         edge_referential_eids,
                                                                                                         graph.num_nodes())
            np.save(transposed_prefix + "fb15k237.coo.sorted.by_srcs_outgoing_freq.srcs.npy", edges_srcs)
            np.save(transposed_prefix + "fb15k237.coo.sorted.by_srcs_outgoing_freq.dsts.npy", edges_dsts)
            np.save(transposed_prefix + "fb15k237.coo.sorted.by_srcs_outgoing_freq.etypes.npy", edges_etypes)
            np.save(transposed_prefix + "fb15k237.coo.sorted.by_srcs_outgoing_freq.referential_eids.npy",
                    edge_referential_eids)
        else:
            edges_srcs, edges_dsts, edges_etypes, edge_referential_eids = sort_coo_by_etype(edges_srcs, edges_dsts,
                                                                                            edges_etypes,
                                                                                            edge_referential_eids)
            np.save(transposed_prefix + "fb15k237.coo.sorted.by_etype_freq.srcs.npy", edges_srcs)
            np.save(transposed_prefix + "fb15k237.coo.sorted.by_etype_freq.dsts.npy", edges_dsts)
            np.save(transposed_prefix + "fb15k237.coo.sorted.by_etype_freq.etypes.npy", edges_etypes)
            np.save(transposed_prefix + "fb15k237.coo.sorted.by_etype_freq.referential_eids.npy", edge_referential_eids)
    else:
        np.save(transposed_prefix + 'fb15k237.coo.srcs.npy', edges_srcs)
        np.save(transposed_prefix + 'fb15k237.coo.dsts.npy', edges_dsts)
        np.save(transposed_prefix + 'fb15k237.coo.etypes.npy', edges_etypes)
        np.save(transposed_prefix + 'fb15k237.coo.referential_eids.npy', edge_referential_eids)
    return edges_srcs, edges_dsts, edges_etypes

    # edges_train_edge_mask = graph.edata['train_edge_mask'].detach().numpy()
    # edges_test_edge_mask = graph.edata['test_edge_mask'].detach().numpy()
    # edges_val_edge_mask = graph.edata['val_edge_mask'].detach().numpy()
    # edges_train_mask = graph.edata['train_mask'].detach().numpy()
    # edges_test_mask = graph.edata['test_mask'].detach().numpy()
    # edges_val_mask = graph.edata['val_mask'].detach().numpy()


def load_ogbn_mag():
    # NB: we need to reindex nodes as nodes of any type in this data set originally starts with index 0
    # the ordering of the abosolute node indices from 0 to N-1 is author, paper, institution, field_of_study
    from ogb.nodeproppred import NodePropPredDataset
    dataset = NodePropPredDataset(name='ogbn-mag')
    graph = dataset[0]
    # edges_srcs = graph.edges()[0].detach().numpy()
    # edges_dsts = graph.edges()[0].detach().numpy()
    # edges_etypes = graph.edata['etype'].detach().numpy()
    edge_srcs = graph[0]['edge_index_dict'][('author', 'affiliated_with', 'institution')][0]
    edge_dsts = graph[0]['edge_index_dict'][('author', 'affiliated_with', 'institution')][1] + \
                graph[0]['num_nodes_dict']['author'] + graph[0]['num_nodes_dict']['paper']
    edge_types = [0] * len(edge_srcs)

    edge_srcs2 = graph[0]['edge_index_dict'][('author', 'writes', 'paper')][0]
    edge_dsts2 = graph[0]['edge_index_dict'][('author', 'writes', 'paper')][1] + graph[0]['num_nodes_dict']['author']
    edge_types2 = [1] * len(edge_srcs2)

    edge_srcs3 = graph[0]['edge_index_dict'][('paper', 'cites', 'paper')][0] + graph[0]['num_nodes_dict']['author']
    edge_dsts3 = graph[0]['edge_index_dict'][('paper', 'cites', 'paper')][1] + graph[0]['num_nodes_dict']['author']
    edge_types3 = [2] * len(edge_srcs3)

    edge_srcs4 = graph[0]['edge_index_dict'][('paper', 'has_topic', 'field_of_study')][0] + graph[0]['num_nodes_dict'][
        'author']
    edge_dsts4 = graph[0]['edge_index_dict'][('paper', 'has_topic', 'field_of_study')][1] + graph[0]['num_nodes_dict'][
        'author'] + graph[0]['num_nodes_dict']['paper'] + graph[0]['num_nodes_dict']['institution']
    edge_types4 = [3] * len(edge_srcs4)

    # NB: the same as the ogbn-mag outputting script in dgl-nvtx
    np.save("affliated_with_1.npy", np.concatenate([[edge_srcs], [edge_dsts]], dtype=np.int32))
    np.save("writing_coo_1.npy", np.concatenate([[edge_srcs2], [edge_dsts2]], dtype=np.int32))
    np.save("citing_coo_1.npy", np.concatenate([[edge_srcs3], [edge_dsts3]], dtype=np.int32))
    np.save("is-about_coo_1.npy", np.concatenate([[edge_srcs4], [edge_dsts4]], dtype=np.int32))
    # there should be no bubble in the node indexing
    num_nodes = graph[0]['num_nodes_dict']['author'] + graph[0]['num_nodes_dict']['paper'] + graph[0]['num_nodes_dict'][
        'institution'] + graph[0]['num_nodes_dict']['field_of_study']
    assert (max(max(edge_srcs), max(edge_srcs2), max(edge_srcs3), max(edge_srcs4), max(edge_dsts), max(edge_dsts2),
                max(edge_dsts3), max(edge_dsts4)) == num_nodes - 1)

    return np.concatenate([edge_srcs, edge_srcs2, edge_srcs3, edge_srcs4]), \
           np.concatenate([edge_dsts, edge_dsts2, edge_dsts3, edge_dsts4]), \
           np.concatenate([edge_types, edge_types2, edge_types3, edge_types4])


def remap_etype_according_to_number_of_edges(etypes):
    etype_frequency = np.bincount(etypes.flatten())
    # TODO: check if there is data loss in this np.argsort invocation, i.e., implicit conversion from int64 to int32
    etype_sorted_by_frequency_from_largest_to_smallest = np.argsort(etype_frequency)[::-1]
    original_etype_to_new_etype_map = dict(zip(etype_sorted_by_frequency_from_largest_to_smallest.tolist(),
                                               range(len(etype_sorted_by_frequency_from_largest_to_smallest))))
    remapped_etype = np.array([original_etype_to_new_etype_map[etype] for etype in etypes], dtype=etypes.dtype)
    return remapped_etype


# def sort_coo_by_etype(srcs, dsts, etypes):
#     etypes = remap_etype_according_to_number_of_edges(etypes)
#     # now sort the (src, dst) pair according to their etype
#     sorted_src_dst_etype = sorted(zip(srcs, dsts, etypes), key=lambda x: x[2])
#     sorted_srcs = np.array([x[0] for x in sorted_src_dst_etype])
#     sorted_dsts = np.array([x[1] for x in sorted_src_dst_etype])
#     sorted_etypes = np.array([x[2] for x in sorted_src_dst_etype])
#     return sorted_srcs, sorted_dsts, sorted_etypes

def sort_coo_by_etype(srcs, dsts, etypes, eids):
    etypes = remap_etype_according_to_number_of_edges(etypes)
    # now sort the (src, dst) pair according to their etype
    sorted_src_dst_etype = sorted(zip(srcs, dsts, etypes, eids), key=lambda x: x[2])
    sorted_srcs = np.array([x[0] for x in sorted_src_dst_etype])
    sorted_dsts = np.array([x[1] for x in sorted_src_dst_etype])
    sorted_etypes = np.array([x[2] for x in sorted_src_dst_etype])
    sorted_eids = np.array([x[3] for x in sorted_src_dst_etype])
    return sorted_srcs, sorted_dsts, sorted_etypes, sorted_eids

def get_node_index_remap_dict_according_to_number_of_edges(srcs, number_of_nodes):
    #TODO: involve non-source node here also.
    #TODO: try to preserve the original place of nodes with near-to-none out degree in the remapping
    srcs_frequency = np.bincount(srcs.flatten())
    if srcs_frequency.shape[0] < number_of_nodes:
        srcs_frequency = np.concatenate([srcs_frequency, np.zeros(number_of_nodes - srcs_frequency.shape[0])])
    # TODO: check if there is data loss in this np.argsort invocation, i.e., implicit conversion from int64 to int32
    srcs_sorted_by_frequency_from_largest_to_smallest = np.argsort(srcs_frequency)[::-1]
    original_src_to_new_src_map = dict(zip(srcs_sorted_by_frequency_from_largest_to_smallest.tolist(),
                                           range(len(srcs_sorted_by_frequency_from_largest_to_smallest))))
    return original_src_to_new_src_map
def remap_node_index_according_to_number_of_edges(srcs, dests, number_of_nodes):
    original_src_to_new_src_map = get_node_index_remap_dict_according_to_number_of_edges(srcs, number_of_nodes)
    remapped_src = np.array([original_src_to_new_src_map[src] for src in srcs], dtype=srcs.dtype)
    remapped_dest = np.array([original_src_to_new_src_map[dest] for dest in dests], dtype=dests.dtype)
    return remapped_src, remapped_dest


# def sort_coo_by_src_outgoing_edges(srcs, dsts, etypes):
#     srcs = remap_src_according_to_number_of_edges(srcs)
#     # now sort the (src, dst) pair according to their src idx
#     sorted_src_dst_srcs = sorted(zip(srcs, dsts, etypes), key=lambda x: x[0])
#     sorted_srcs = np.array([x[0] for x in sorted_src_dst_srcs])
#     sorted_dsts = np.array([x[1] for x in sorted_src_dst_srcs])
#     sorted_etypes = np.array([x[2] for x in sorted_src_dst_srcs])
#     return sorted_srcs, sorted_dsts, sorted_etypes

def sort_coo_by_src_outgoing_edges(srcs, dsts, etypes, eids, number_of_nodes):
    srcs, dsts = remap_node_index_according_to_number_of_edges(srcs, dsts, number_of_nodes)
    # now sort the (src, dst) pair according to their src idx
    sorted_src_dst_srcs = sorted(zip(srcs, dsts, etypes, eids), key=lambda x: x[0])
    sorted_srcs = np.array([x[0] for x in sorted_src_dst_srcs])
    sorted_dsts = np.array([x[1] for x in sorted_src_dst_srcs])
    sorted_etypes = np.array([x[2] for x in sorted_src_dst_srcs])
    sorted_eids = np.array([x[3] for x in sorted_src_dst_srcs])
    return sorted_srcs, sorted_dsts, sorted_etypes, sorted_eids


def plot_edge_type_vs_src_idx_hist(srcs, dsts, etypes):
    edges_per_src_idx_dict = {}
    for edge_idx in range(len(srcs)):
        src_idx = srcs[edge_idx]
        dst_idx = dsts[edge_idx]
        etype = etypes[edge_idx]
        if src_idx not in edges_per_src_idx_dict:
            edges_per_src_idx_dict[src_idx] = set()
        edges_per_src_idx_dict[src_idx].add((dst_idx, etype))

    # count the maximal #edge among all edge types per source node and gather the corresponding edge type.
    # count occurrences each edge type gets maximal number of edges among all source nodes
    # TODO: count these through ell-disabling edge number threshold and ell width threshold
    maximal_edge_type_per_src_idx_dict = {}
    for src_idx in edges_per_src_idx_dict:
        edge_type_count_dict = {}
        for edge in edges_per_src_idx_dict[src_idx]:
            dst_idx, edge_type = edge
            edge_type_count_dict[edge_type] = edge_type_count_dict.get(edge_type, 0) + 1
        # get (occurrence, edge type index) of the maximal edge type for current source node
        maximal_edge_type_per_src_idx_dict[src_idx] = max(
            [(edge_type_count_dict[edge_type], edge_type) for edge_type in edge_type_count_dict], key=lambda x: x[0])

    pyplot.plot(list(range(max(maximal_edge_type_per_src_idx_dict.keys()) + 1)),
                [maximal_edge_type_per_src_idx_dict.get(src_idx, (0, 0))[0] for src_idx in
                 range(max(maximal_edge_type_per_src_idx_dict.keys()) + 1)])
    pyplot.show()
    print("max edge portion", (0.0 + sum([maximal_edge_type_per_src_idx_dict.get(src_idx, (0, 0))[0] for src_idx in
                                          range(max(maximal_edge_type_per_src_idx_dict.keys()) + 1)])) / len(srcs))
    maximal_edge_type_and_occurrences = {}
    for src_idx in maximal_edge_type_per_src_idx_dict:
        maximal_edge_type_and_occurrences[
            maximal_edge_type_per_src_idx_dict[src_idx][1]] = maximal_edge_type_and_occurrences.get(
            maximal_edge_type_per_src_idx_dict[src_idx][1], 0) + 1
    pyplot.plot(list(range(max(maximal_edge_type_and_occurrences.keys()) + 1)),
                [maximal_edge_type_and_occurrences.get(edge_type, 0) for edge_type in
                 range(max(maximal_edge_type_and_occurrences.keys()) + 1)])
    pyplot.show()
    print("done")
    return


def output_segment_csr_format(edges_srcs, edges_dsts, edges_types, cutoff_node_amount, dataset_name):
    ####### Print sclar values of the graph #######
    # mere csr cutoff index CT(src node with index>=this threshold will only use mere csr),
    # edge_type_num
    ######## Part 0 ##########
    # node 0 maximal edge type, edge node num X0
    # node 1 maximal edge type, edge node num X1
    # ...
    # node CT-1 maximal edge type, edge node num XCT-1
    ######## Part 1a ##########
    # dense edge type 0 number of non-cut-off source nodes, dense edge type 1 number of non-cut-off source nodes, ...
    ######## Part 1 ##########
    # edge type 0: dense portion edge source node index, dense portion edge source node index2, ...
    # edge type 1: dense portion edge source node index, dense portion edge source node index2, ...
    # ...
    # edge type TE-1: dense portion edge source node index, dense portion edge source node index2, ...
    ######## Part 2 ########## (dense portion edges)
    # Node 0: Maximal edge type, edge dest node index, edge dest node index2, ..., edge dest node index8X0
    # Node 1: Maximal edge type, edge dest node index, edge dest node index2, ..., edge dest node index8X1
    # ...
    # Node CT-1: Maximal edge type, edge dest node index, edge dest node index2, ..., edge dest node index8X(CT-1)
    ######## Part 3 ########## (sparse portion edges)
    # residue separate csr
    edges_per_src_idx_dict = {}
    for edge_idx in range(len(edges_srcs)):
        src_idx = edges_srcs[edge_idx]
        dst_idx = edges_dsts[edge_idx]
        etype = edges_types[edge_idx]
        if src_idx not in edges_per_src_idx_dict:
            edges_per_src_idx_dict[src_idx] = set()
        edges_per_src_idx_dict[src_idx].add((dst_idx, etype))


    maximal_edge_type_per_src_idx_dict = {}
    for src_idx in edges_per_src_idx_dict:
        edge_type_count_dict = {}
        for edge in edges_per_src_idx_dict[src_idx]:
            dst_idx, edge_type = edge
            edge_type_count_dict[edge_type] = edge_type_count_dict.get(edge_type, 0) + 1
        # get edge type index and count of maximal edge type
        maximal_edge_type_per_src_idx_dict[src_idx] = max(
            [(edge_type_count_dict[edge_type], edge_type, src_idx) for edge_type in edge_type_count_dict],
            key=lambda x: x[0])

    reversed_sorted_maximal_edge_type_per_src_idx = reversed(
        sorted([maximal_edge_type_per_src_idx_dict[key] for key in maximal_edge_type_per_src_idx_dict.keys()],
               key=lambda x: x[0]))

    # there are also nodes without being present as source node, we need to add them to the end of the reversed sort list
    num_nodes = max([max(edges_srcs), max(edges_dsts)]) + 1
    zero_source_nodes_maximal_edge_type = [(0, 0, src_idx) for src_idx in
                                           set(range(num_nodes)).difference(maximal_edge_type_per_src_idx_dict.keys())]
    reversed_sorted_maximal_edge_type_per_src_idx = list(
        reversed_sorted_maximal_edge_type_per_src_idx) + zero_source_nodes_maximal_edge_type

    assert (len(reversed_sorted_maximal_edge_type_per_src_idx) == num_nodes)

    # first, permute the indices to let the minimal to the very end
    node_indices_mapping = dict(
        [(element[2], new_idx) for new_idx, element in enumerate(reversed_sorted_maximal_edge_type_per_src_idx)])
    new_maximal_edge_type_per_src_idx_dict = {}
    for src_idx in maximal_edge_type_per_src_idx_dict:
        new_maximal_edge_type_per_src_idx_dict[node_indices_mapping[src_idx]] = maximal_edge_type_per_src_idx_dict[
            src_idx]
    new_edges_per_src_idx_dict = {}
    for edge_idx in range(len(edges_srcs)):
        src_idx = node_indices_mapping[edges_srcs[edge_idx]]
        dst_idx = node_indices_mapping[edges_dsts[edge_idx]]
        etype = edges_types[edge_idx]
        if src_idx not in new_edges_per_src_idx_dict:
            new_edges_per_src_idx_dict[src_idx] = set()
        new_edges_per_src_idx_dict[src_idx].add((dst_idx, etype))

    # prepare part 0
    new_maximal_edge_type_per_src_idx_list = [new_maximal_edge_type_per_src_idx_dict[src_idx] for src_idx in
                                              range(len(new_maximal_edge_type_per_src_idx_dict))]
    part_0_edge_nums = [element[0] for element in new_maximal_edge_type_per_src_idx_list[:-cutoff_node_amount]]
    part_0_edge_types = [element[1] for element in new_maximal_edge_type_per_src_idx_list[:-cutoff_node_amount]]
    np.save(dataset_name + '.segment_csr.part_0.edge_nums.npy', np.array(part_0_edge_nums,
                                                                         dtype=np.int64))  # This array will be exclusive scanned and then padded eventually in C++
    np.save(dataset_name + '.segment_csr.part_0.edge_types.npy', np.array(part_0_edge_types, dtype=np.int64))
    print("maximal non-cut-off index ", len(new_maximal_edge_type_per_src_idx_list[:-cutoff_node_amount]))

    # prepare part 1
    part_1_src_node_per_edge_type = dict()
    for src_idx, element in enumerate(new_maximal_edge_type_per_src_idx_list[:-cutoff_node_amount]):
        edge_num, edge_type, _old_src_idx = element
        if edge_type not in part_1_src_node_per_edge_type:
            part_1_src_node_per_edge_type[edge_type] = []
        part_1_src_node_per_edge_type[edge_type].append(src_idx)
    part_1_src_node_per_edge_type_flatten = []
    part_1a_edge_type_num = []
    for edge_type in range(
            len(part_1_src_node_per_edge_type)):  # asserting all edge_type are numbers and are in [0, 1, ..., edge_type_num-1]
        part_1a_edge_type_num += [len(part_1_src_node_per_edge_type[edge_type])]
        part_1_src_node_per_edge_type_flatten.extend(part_1_src_node_per_edge_type[edge_type])
    np.save(dataset_name + '.segment_csr.part_1.edge_type_num.npy', np.array(part_1a_edge_type_num, dtype=np.int64))
    np.save(dataset_name + '.segment_csr.part_1.src_node_per_edge_type.npy',
            np.array(part_1_src_node_per_edge_type_flatten, dtype=np.int64))
    print("edge type num ", len(part_1_src_node_per_edge_type))
    # prepare part 2 and 3
    part_2_edges = []
    part_2_edges_flatten = []
    part_3_residue_edges = []
    for src_idx in range(len(part_0_edge_types)):
        curr_edge_type = part_0_edge_types[src_idx]
        curr_edge_num = part_0_edge_nums[src_idx]
        part_2_edges.append([])
        for dst_idx, edge_type in new_edges_per_src_idx_dict[src_idx]:
            if edge_type == curr_edge_type:
                part_2_edges[src_idx].append(dst_idx)
                part_2_edges_flatten.append(dst_idx)
            else:
                # residue edges: non-maximal-edge-type edges for non-cut-off nodes
                part_3_residue_edges.append((src_idx, dst_idx, edge_type))
        assert (len(part_2_edges[src_idx]) == curr_edge_num)
    np.save(dataset_name + '.segment_csr.part_2.edges.npy', np.array(part_2_edges_flatten,
                                                                     dtype=np.int64))  # this will be padded eventually in C++. another padded array in C++ is eids associated with this array
    # residue edges: all edges for those cut-off nodes
    for src_idx in range(len(part_0_edge_types), len(new_edges_per_src_idx_dict.keys())):
        for dst_idx, edge_type in new_edges_per_src_idx_dict[src_idx]:
            part_3_residue_edges.append((src_idx, dst_idx, edge_type))

    part_3_srcs, part_3_dsts, part_3_types = zip(*part_3_residue_edges)
    np.save(dataset_name + '.segment_csr.part_3.srcs.npy', np.array(part_3_srcs, dtype=np.int64))
    np.save(dataset_name + '.segment_csr.part_3.dsts.npy', np.array(part_3_dsts, dtype=np.int64))
    np.save(dataset_name + '.segment_csr.part_3.types.npy', np.array(part_3_types, dtype=np.int64))


def load_segment_csr_dataset(dataset_name):
    part_0_edge_nums = np.load(dataset_name + '.segment_csr.part_0.edge_nums.npy')
    part_0_edge_types = np.load(dataset_name + '.segment_csr.part_0.edge_types.npy')
    part_1_edge_type_num = np.load(dataset_name + '.segment_csr.part_1.edge_type_num.npy')
    part_1_src_node_per_edge_type = np.load(dataset_name + '.segment_csr.part_1.src_node_per_edge_type.npy')
    part_2_edges = np.load(dataset_name + '.segment_csr.part_2.edges.npy')
    part_3_srcs = np.load(dataset_name + '.segment_csr.part_3.srcs.npy')
    part_3_dsts = np.load(dataset_name + '.segment_csr.part_3.dsts.npy')
    part_3_types = np.load(dataset_name + '.segment_csr.part_3.types.npy')
    return part_0_edge_nums, part_0_edge_types, part_1_edge_type_num, part_1_src_node_per_edge_type, part_2_edges, part_3_srcs, part_3_dsts, part_3_types


def count_overlap(part_0_edge_nums, part_1_edge_type_num, part_1_src_node_per_edge_type, part_2_edges):
    exclusive_scan_part_0_edge_nums = [0] + np.cumsum(part_0_edge_nums).tolist()
    exclusive_scan_part_1_edge_type_num = [0] + np.cumsum(part_1_edge_type_num).tolist()
    average_reuses = []
    for idx_relation in range(len(part_1_edge_type_num)):
        for idx_threading_block in range((part_1_edge_type_num[idx_relation] + 31) // 32):
            idx_src_nodes = list(
                map(lambda x: part_1_src_node_per_edge_type[exclusive_scan_part_1_edge_type_num[idx_relation] + x],
                    range(idx_threading_block * 32,
                          min((idx_threading_block + 1) * 32, part_1_edge_type_num[idx_relation]))))
            dest_nodes_for_all_source_nodes = list(map(lambda x: set(
                part_2_edges[exclusive_scan_part_0_edge_nums[x]:exclusive_scan_part_0_edge_nums[x + 1]]),
                                                       idx_src_nodes))
            average_reuse = (sum([len(dest_nodes_for_one_src_node) for dest_nodes_for_one_src_node in
                                  dest_nodes_for_all_source_nodes]) + 0.0) / (
                                        0.000001 + len(set().union(*dest_nodes_for_all_source_nodes)))
            average_reuses.append(average_reuse)

    print('average_reuses:', sum(average_reuses) / len(average_reuses))


def main2():
    edges_srcs, edges_dsts, edges_types = output_npy_file_fb15k237(sorted=True, sorted_by_src=True)
    edges_srcs, edges_dsts, edges_types = output_npy_file_fb15k237(sorted=True, sorted_by_src=False)
    # edges_srcs, edges_dsts, edges_types = output_npy_file_ogbn_wikikg2()
    # edges_srcs, edges_dsts, edges_types = load_ogbn_mag()  # mere csr for 400000 nodes, others use maximal_segment csr

    # plot_edge_type_vs_src_idx_hist(edges_srcs, edges_dsts, edges_types)


def main3():
    # edge_srcs, edge_dests, edge_etypes = output_npy_file_fb15k237(sorted=False, sorted_by_src=False, transposed=False)
    # get_src_dest_pair_for_each_etype(edge_srcs, edge_dests, edge_etypes)
    output_npy_file_fb15k237(sorted=True, sorted_by_src=True, transposed=True)
    output_npy_file_fb15k237(sorted=True, sorted_by_src=False, transposed=True)
    output_npy_file_fb15k237(sorted=True, sorted_by_src=True, transposed=False)
    output_npy_file_fb15k237(sorted=True, sorted_by_src=False, transposed=False)

    output_npy_file_fb15k237(sorted=False, sorted_by_src=False, transposed=True)
    output_npy_file_fb15k237(sorted=False, sorted_by_src=False, transposed=False)

    output_npy_file_ogbn_wikikg2(sorted=True, sorted_by_src=True, transposed=False)
    output_npy_file_ogbn_wikikg2(sorted=True, sorted_by_src=False, transposed=False)
    output_npy_file_ogbn_wikikg2(sorted=True, sorted_by_src=True, transposed=True)
    output_npy_file_ogbn_wikikg2(sorted=True, sorted_by_src=False, transposed=True)

    output_npy_file_ogbn_wikikg2(sorted=False, sorted_by_src=False, transposed=True)
    output_npy_file_ogbn_wikikg2(sorted=False, sorted_by_src=False, transposed=False)

def main4():
    output_npy_file_fb15k237(sorted=False, sorted_by_src=False, transposed=False)
    output_npy_file_ogbn_wikikg2(sorted=False, sorted_by_src=False, transposed=False)

if __name__ == "__main__":
    # part_0_edge_nums, part_0_edge_types, part_1_edge_type_num, part_1_src_node_per_edge_type, part_2_edges, part_3_srcs, part_3_dsts, part_3_types = load_segment_csr_dataset(
    #    'ogbn_mag')
    # count_overlap(part_0_edge_nums, part_1_edge_type_num, part_1_src_node_per_edge_type, part_2_edges)
    main4()
