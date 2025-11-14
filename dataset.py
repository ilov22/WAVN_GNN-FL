'''
-dataset.py reads labels.csv & loads image pairs
-creates graph data samples using building_graph
'''

import torch
from torch.utils.data import Dataset
import pandas as pd
from building_graph import BuildGraphFromCSV

class NavGraphDataset(Dataset):
    def __init__(self, CSVpath, imageDirectory, device):
        #CSVpath is path to labels.csv
        #imageDirectory is where the images are stored
        #device is the cpu/cuda
        self.df=pd.read_csv(CSVpath)

        #text directions as integers
        directions=sorted(self.df["direction"].unique())
        self.LabelMap={d: i for i, d in enumerate(directions)}
        self.inverseLabelMap={i: d for d, i in self.LabelMap.items()}

        #build graph dataset
        self.graphs= BuildGraphFromCSV(self.df, imageDirectory, self.LabelMap, device)

    def __len__(self):
        #return # of samples
        return len(self.graphs)
    def __getitem__(self, idx):
        #return 1 graph data sample
        return self.graphs[idx]