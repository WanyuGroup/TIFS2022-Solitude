import os
from functools import partial
from unicodedata import name
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import ToSparseTensor, AddTrainValTestMask
from torch_geometric.utils import to_undirected
from transforms import Normalize, FilterTopClass
import torch.nn.functional as F
from torch_geometric.utils import degree
import numpy as np


class KarateClub(InMemoryDataset):
    url = 'https://raw.githubusercontent.com/benedekrozemberczki/karateclub/master/dataset/node_level'
    available_datasets = {
        'twitch',
        'facebook',
        'github',
        'deezer',
        'lastfm',
        'wikipedia'
    }

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in self.available_datasets

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return ['edges.csv', 'features.csv', 'target.csv']

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for part in ['edges', 'features', 'target']:
            download_url(f'{self.url}/{self.name}/{part}.csv', self.raw_dir)

    def process(self):
        target_file = os.path.join(self.raw_dir, self.raw_file_names[2])
        y = pd.read_csv(target_file)['target']
        y = torch.from_numpy(y.to_numpy(dtype=int))
        num_nodes = len(y)

        edge_file = os.path.join(self.raw_dir, self.raw_file_names[0])
        edge_index = pd.read_csv(edge_file)
        edge_index = torch.from_numpy(edge_index.to_numpy()).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes)  # undirected edges

        feature_file = os.path.join(self.raw_dir, self.raw_file_names[1])
        x = pd.read_csv(feature_file).drop_duplicates()
        x = x.pivot(index='node_id', columns='feature_id', values='value').fillna(0)
        x = x.reindex(range(num_nodes), fill_value=0)
        x = torch.from_numpy(x.to_numpy()).float()
        
        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'KarateClub-{self.name}()'

supported_datasets = {
    'cora': partial(Planetoid, name='cora'),
    'pubmed': partial(Planetoid, name='pubmed'),
    'facebook': partial(KarateClub, name='facebook'),
    'lastfm': partial(KarateClub, name='lastfm', transform=FilterTopClass(10)),
    'wikipedia': partial(KarateClub, name='wikipedia', transform=FilterTopClass(10)),
    'citeseer': partial(Planetoid, name='citeseer'),
}


def load_dataset(
        dataset:        dict(help='name of the dataset', option='-d', choices=supported_datasets) = 'cora',
        data_dir:       dict(help='directory to store the dataset') = './datasets',
        data_range:     dict(help='min and max feature value', nargs=2, type=float) = (0, 1),
        val_ratio:      dict(help='fraction of nodes used for validation') = .25,
        test_ratio:     dict(help='fraction of nodes used for test') = .25,
        ):
    data = supported_datasets[dataset](root=os.path.join(data_dir, dataset))

    data = AddTrainValTestMask(split='train_rest', num_val=val_ratio, num_test=test_ratio)(data[0])
    data = ToSparseTensor(remove_edge_index=False)(data)

    row, col = data.edge_index
    deg = degree(row, data.x.size()[0], dtype=data.x.dtype)
    
    data.name = dataset
    data.num_classes = int(data.y.max().item()) + 1

    if data_range is not None:
        low, high = data_range
        data = Normalize(low, high)(data)

    data.y = F.one_hot(data.y)

    return data