from ogb.linkproppred import LinkPropPredDataset

dataset = LinkPropPredDataset(name='ogbl-wikikg2')
# edge_index ndarray: (2,16109182)
# edge_reltype ndarray: (16109182, 1)
print(dataset)
import networkx as nx

G = nx.MultiGraph()

G.add_nodes_from(range(dataset.graph['num_nodes']))
for edge_idx in range(len(dataset.graph['edge_index'][0])):
    G.add_edge(dataset.graph['edge_index'][0][edge_idx], dataset.graph['edge_index'][1][edge_idx],
               type=dataset.graph['edge_reltype'][edge_idx][0])

nx.write_gpickle(G, "ogbn-wikikg2.gpickle")
G = nx.read_gpickle("ogbn-wikikg2.gpickle")
pass
