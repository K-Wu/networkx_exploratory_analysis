from ogb.nodeproppred import NodePropPredDataset

dataset = NodePropPredDataset(name='ogbn-mag')

import networkx as nx

G = nx.MultiGraph()
G.add_nodes_from(range(dataset.graph['num_nodes_dict']['author']), type="author")
G.add_nodes_from([(item[0], {'venue': item[1]}) for item in zip(range(dataset.graph['num_nodes_dict']['author'],
                                                                      dataset.graph['num_nodes_dict']['author'] +
                                                                      dataset.graph['num_nodes_dict']['paper']),
                                                                dataset.labels['paper'].flatten().tolist())],
                 type="paper")
G.add_nodes_from(range(dataset.graph['num_nodes_dict']['author'] + dataset.graph['num_nodes_dict']['paper'],
                       dataset.graph['num_nodes_dict']['author'] + dataset.graph['num_nodes_dict']['paper'] +
                       dataset.graph['num_nodes_dict']['institution']), type="institution")
G.add_nodes_from(range(dataset.graph['num_nodes_dict']['author'] + dataset.graph['num_nodes_dict']['paper'] +
                       dataset.graph['num_nodes_dict']['institution'],
                       dataset.graph['num_nodes_dict']['author'] + dataset.graph['num_nodes_dict']['paper'] +
                       dataset.graph['num_nodes_dict']['institution'] + dataset.graph['num_nodes_dict'][
                           'field_of_study']), type="field_of_study")
G.add_edges_from(
    [(item[0], item[1] + dataset.graph['num_nodes_dict']['author'] + dataset.graph['num_nodes_dict']['paper']) for item
     in zip(*dataset.graph['edge_index_dict'][('author', 'affiliated_with', 'institution')].tolist())],
    type='author_affiliated_with_institution')
G.add_edges_from([(item[0], item[1] + dataset.graph['num_nodes_dict']['author']) for item in
                  zip(*dataset.graph['edge_index_dict'][('author', 'writes', 'paper')].tolist())],
                 type='author_writes_paper')
G.add_edges_from(
    [(item[0] + dataset.graph['num_nodes_dict']['author'], item[1] + dataset.graph['num_nodes_dict']['author']) for item
     in zip(*dataset.graph['edge_index_dict'][('paper', 'cites', 'paper')].tolist())], type='paper_cites_paper')
G.add_edges_from([(item[0] + dataset.graph['num_nodes_dict']['author'],
                   item[1] + dataset.graph['num_nodes_dict']['author'] + dataset.graph['num_nodes_dict']['paper'] +
                   dataset.graph['num_nodes_dict']['institution']) for item in
                  zip(*dataset.graph['edge_index_dict'][('paper', 'has_topic', 'field_of_study')].tolist())],
                 type='paper_has_topic_field_of_study')
nx.write_gpickle(G, "ogbn-mag.gpickle")
G = nx.read_gpickle("ogbn-mag.gpickle")
pass
