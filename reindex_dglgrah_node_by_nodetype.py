import dgl
from ogb.nodeproppred import DglNodePropPredDataset


if __name__ == "__main__":
        dataset = DglNodePropPredDataset(name="ogbn-mag")
        g = dataset[0][0]
        pass
