import os

from OutputNPYFileFB15k237 import *


def get_new_node_index(edges_srcs, edges_dsts):
    # with open(os.path.join(".", 'rabbit_reorder', 'demo', "rabbit_reorder_temp.txt"), 'w') as fd:
    #     for i in range(len(edges_srcs)):
    #         fd.write(str(edges_srcs[i]) + " " + str(edges_dsts[i]) + "\n")
    # if not os.path.exists(os.path.join(".", 'rabbit_reorder', 'demo', 'reorder')):
    #     print(
    #         "please compile the rabbit_reorder submodule. NB: sudo apt-get install libgoogle-perftools-dev to install the dependency for compiling rabbit_reorder submodule")
    #     exit(-1)
    # os.system(os.path.join(".", 'rabbit_reorder', 'demo',
    #                        'reorder') + " " + os.path.join(".",
    #                                                         'rabbit_reorder',
    #                                                         'demo',
    #                                                         'rabbit_reorder_temp.txt') + " > " + os.path.join(
    #     ".", 'rabbit_reorder', 'demo', "rabbit_reorder_node_reindex.txt"))
    node_remap = dict()
    with open(os.path.join(".", 'rabbit_reorder', 'demo', "test_purpose_rabbit_reorder_node_reindex.txt")) as fd:
        num_head_lines = 0
        for idx, line in enumerate(fd):
            line = line.strip()
            # skipping the first few lines, i.e., if first letter is not digit, skip the line
            if not line[0].isdigit():
                num_head_lines += 1
                continue
            new_index = int(line)
            node_remap[idx - num_head_lines] = new_index
    return node_remap


def rabbit_reorder(edges_srcs, edges_dsts, edges_types):
    node_remap = get_new_node_index(edges_srcs, edges_dsts)
    for i in range(len(edges_srcs)):
        edges_srcs[i] = node_remap[edges_srcs[i]]
        edges_dsts[i] = node_remap[edges_dsts[i]]
    return edges_srcs, edges_dsts, edges_types

def count_num_edges_in_tiles(edges_srcs, edges_dests, edges_types, tile_dim=32, threshold=0.1):
    num_minimum_nodes_in_tiles = tile_dim * tile_dim * threshold

    potential_tiles = dict()

    for edge_idx in range(len(edges_srcs)):
        src = edges_srcs[edge_idx]
        dst = edges_dests[edge_idx]
        etype = edges_types[edge_idx]
        if src // tile_dim not in potential_tiles:
            potential_tiles[src // tile_dim] = dict()

        src_tile_dict = potential_tiles[src // tile_dim]
        if dst // tile_dim not in src_tile_dict:
            src_tile_dict[dst // tile_dim] = set()

        src_dest_tile_set = src_tile_dict[dst // tile_dim]
        src_dest_tile_set.add((src, dst, etype))

    num_edges_in_tiles = 0
    for src_tile in potential_tiles.keys():
        for dst_tile in potential_tiles[src_tile].keys():
            if len(potential_tiles[src_tile][dst_tile]) >= num_minimum_nodes_in_tiles:
                num_edges_in_tiles += len(potential_tiles[src_tile][dst_tile])

    print("num_edges_in_tiles portion: ", (0.0 + num_edges_in_tiles) / len(edges_srcs))
    return

if __name__ == "__main__":
    edges_srcs, edges_dsts, edges_types = load_ogbn_mag()
    edges_srcs, edges_dsts, edges_types = rabbit_reorder(edges_srcs, edges_dsts, edges_types)
    count_num_edges_in_tiles(edges_srcs, edges_dsts, edges_types)
    output_segment_csr_format(edges_srcs, edges_dsts, edges_types, 400000, "ogbn_mag")
