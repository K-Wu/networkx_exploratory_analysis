# import torch as th
import numpy as np
from matplotlib import pyplot


def output_npy_file_ogbn_wikikg2():
    from ogb.linkproppred import LinkPropPredDataset
    dataset = LinkPropPredDataset(name='ogbl-wikikg2')
    # num nodes:  2500604, num edges: 16109182
    num_edges = len(dataset.graph['edge_index'][0])
    edges_srcs = np.empty((num_edges,), dtype=np.int64)
    edges_dsts = np.empty((num_edges,), dtype=np.int64)
    edges_types = np.empty((num_edges,), dtype=np.int64)
    for edge_idx in range(len(dataset.graph['edge_index'][0])):
        edges_srcs[edge_idx] = dataset.graph['edge_index'][0][edge_idx]
        edges_dsts[edge_idx] = dataset.graph['edge_index'][1][edge_idx]
        edges_types[edge_idx] = dataset.graph['edge_reltype'][edge_idx][0]
        assert (len(dataset.graph['edge_reltype'][edge_idx]) == 1)
    np.save("ogbn-wikikg2.coo.srcs.npy", edges_srcs)
    np.save("ogbn-wikikg2.coo.dsts.npy", edges_dsts)
    np.save("ogbn-wikikg2.coo.etypes.npy", edges_types)
    return edges_srcs, edges_dsts, edges_types


def output_npy_file_fb15k237():
    from dgl.data import FB15k237Dataset
    dataset = FB15k237Dataset()
    graph = dataset[0]
    # num_nodes: 14541, num_edges:620232
    edges_srcs, edges_dsts = graph.edges()
    edges_srcs = graph.edges()[0].detach().numpy()
    edges_dsts = graph.edges()[0].detach().numpy()
    edges_etypes = graph.edata['etype'].detach().numpy()
    np.save('fb15k237.coo.srcs.npy', edges_srcs)
    np.save('fb15k237.coo.dsts.npy', edges_dsts)
    np.save('fb15k237.coo.etypes.npy', edges_etypes)
    return edges_srcs, edges_dsts, edges_etypes

    # edges_train_edge_mask = graph.edata['train_edge_mask'].detach().numpy()
    # edges_test_edge_mask = graph.edata['test_edge_mask'].detach().numpy()
    # edges_val_edge_mask = graph.edata['val_edge_mask'].detach().numpy()
    # edges_train_mask = graph.edata['train_mask'].detach().numpy()
    # edges_test_mask = graph.edata['test_mask'].detach().numpy()
    # edges_val_mask = graph.edata['val_mask'].detach().numpy()


def load_ogbn_mag():
    from ogb.nodeproppred import NodePropPredDataset
    dataset = NodePropPredDataset(name='ogbn-mag')
    graph = dataset[0]
    # edges_srcs = graph.edges()[0].detach().numpy()
    # edges_dsts = graph.edges()[0].detach().numpy()
    # edges_etypes = graph.edata['etype'].detach().numpy()
    edge_srcs = graph[0]['edge_index_dict'][('author', 'affiliated_with', 'institution')][0]
    edge_dsts = graph[0]['edge_index_dict'][('author', 'affiliated_with', 'institution')][1]
    edge_types = [0] * len(edge_srcs)

    edge_srcs2 = graph[0]['edge_index_dict'][('author', 'writes', 'paper')][0]
    edge_dsts2 = graph[0]['edge_index_dict'][('author', 'writes', 'paper')][1]
    edge_types2 = [1] * len(edge_srcs2)

    edge_srcs3 = graph[0]['edge_index_dict'][('paper', 'cites', 'paper')][0]
    edge_dsts3 = graph[0]['edge_index_dict'][('paper', 'cites', 'paper')][1]
    edge_types3 = [2] * len(edge_srcs3)

    edge_srcs4 = graph[0]['edge_index_dict'][('paper', 'has_topic', 'field_of_study')][0]
    edge_dsts4 = graph[0]['edge_index_dict'][('paper', 'has_topic', 'field_of_study')][1]
    edge_types4 = [3] * len(edge_srcs4)

    return np.concatenate([edge_srcs, edge_srcs2, edge_srcs3, edge_srcs4]), \
           np.concatenate([edge_dsts, edge_dsts2, edge_dsts3, edge_dsts4]), \
           np.concatenate([edge_types, edge_types2, edge_types3, edge_types4])


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
        # get edge type index and count of maximal edge type
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
            maximal_edge_type_per_src_idx_dict[src_idx][0], 0) + 1
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
    np.save(dataset_name + '.segment_csr.part_0.edge_nums.npy', np.array(part_0_edge_nums, dtype=np.int64))
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
    np.save(dataset_name + '.segment_csr.part_2.edges.npy', np.array(part_2_edges_flatten, dtype=np.int64))
    # residue edges: all edges for those cut-off nodes
    for src_idx in range(len(part_0_edge_types), len(new_edges_per_src_idx_dict.keys())):
        for dst_idx, edge_type in new_edges_per_src_idx_dict[src_idx]:
            part_3_residue_edges.append((src_idx, dst_idx, edge_type))

    part_3_srcs, part_3_dsts, part_3_types = zip(*part_3_residue_edges)
    np.save(dataset_name + '.segment_csr.part_3.srcs.npy', np.array(part_3_srcs, dtype=np.int64))
    np.save(dataset_name + '.segment_csr.part_3.dsts.npy', np.array(part_3_dsts, dtype=np.int64))
    np.save(dataset_name + '.segment_csr.part_3.types.npy', np.array(part_3_types, dtype=np.int64))


if __name__ == "__main__":
    # edges_srcs, edges_dsts, edges_types = output_npy_file_fb15k237()
    # edges_srcs, edges_dsts, edges_types = output_npy_file_ogbn_wikikg2()
    edges_srcs, edges_dsts, edges_types = load_ogbn_mag()  # mere csr for 400000 nodes, others use maximal_segment csr

    plot_edge_type_vs_src_idx_hist(edges_srcs, edges_dsts, edges_types)
