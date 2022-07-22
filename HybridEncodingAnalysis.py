from OutputNPYFileFB15k237 import *

if __name__ == "__main__":
    edges_srcs, edges_dsts, edges_types = load_ogbn_mag()
    output_segment_csr_format(edges_srcs, edges_dsts, edges_types, 400000, "ogbn_mag")
