'''
This file defines the GNN used to learn from the image-pair graphs
Notes: the class GNN is a simple 2-layer GCN that is followed by a global pooling & linear classification head
Global Pooling - technique that aggregates the features from all of the nodes in a graph into a single, fixed-size vector that represents the entire graph
Linear Classification Head - a simple, fully connected layer that takes the final learned node or graph representations and maps them to the desired output classes
'''

import torch 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNN(torch.nn.Module):
    def __init__(self, inChannels, hiddenChannels, numClasses):
        super(GNN, self).__init__()
        #GCN layer one: transform input node features
        self.conv1=GCNConv(inChannels, hiddenChannels)
        #GCN layer two: refines node embeddings
        self.conv2=GCNConv(hiddenChannels, hiddenChannels)
        #classification head 
        self.lin=torch.nn.Linear(hiddenChannels, numClasses)

    def forward(self, x, edge_index, batch):
        #pass through each network, [numNodes, inChannels] is the node feature matric (x)
        #edgeIndex is the graph connectivity and the batch is the graph index per node
        #finding common landmarks
        x=self.conv1(x, edge_index)
        x=F.relu(x)
        x=self.conv2(x, edge_index)
        x=F.relu(x)

        #graph level embedding (gobal mean pooling)
        x=global_mean_pool(x, batch)
        #linear classification
        x=self.lin(x)
        return F.log_softmax(x, dim=1)
