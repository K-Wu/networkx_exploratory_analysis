# requirements.txt tailored for requirement of using https://github.com/dmlc/GNNLens2
import gnnlens

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, FB15k237Dataset, CoauthorPhysicsDataset, CoauthorCSDataset

cora_dataset = CoraGraphDataset()
cora_graph = cora_dataset[0]
citeseer_dataset = CiteseerGraphDataset()
citeseer_graph = citeseer_dataset[0]

coauthor_cs_dataset = CoauthorCSDataset()
coauthor_cs_graph = coauthor_cs_dataset[0]
coauthor_physics_dataset = CoauthorPhysicsDataset()
coauthor_physics_graph = coauthor_physics_dataset[0]
fb15k237_dataset = FB15k237Dataset()
fb15k237_graph = fb15k237_dataset[0]


from gnnlens import Writer

# Specify the path to create a new directory for dumping data files.
writer = Writer('tutorial_graph2')
writer.add_graph(name='Cora', graph=cora_graph)
writer.add_graph(name='Citeseer', graph=citeseer_graph)
writer.add_graph(name='CoauthorCS', graph=coauthor_cs_graph)
writer.add_graph(name='CoauthorPhysics', graph=coauthor_physics_graph)
writer.add_graph(name='FB15K237', graph=fb15k237_graph)

# Finish dumping
writer.close()