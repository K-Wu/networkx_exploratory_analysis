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


if __name__ == "__main__":
    edges_srcs, edges_dsts, edges_types = load_ogbn_mag()
    edges_srcs, edges_dsts, edges_types = rabbit_reorder(edges_srcs, edges_dsts, edges_types)

    output_segment_csr_format(edges_srcs, edges_dsts, edges_types, 400000, "ogbn_mag")
