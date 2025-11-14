'''
- This file building_graph.py handles creation of graph structures from dataset. 
- The 'current' and 'destination' image pairs are treated as a connection in a graph. 
- The direction label (left, right, forward) are the target for training.
- Also extracting image features (using pretrained CNN) to represent each node. 
'''

import os
import torch
import torchvision.transforms as T
import torchvision.models as models
from torch_geometric.data import Data
from PIL import Image

def ExtractImageFeatures(imagePath, model, transform, device):
    #For a path to an image, load, pre-process, and run it through CNN to obtain feature vector
    #Acts as the node's (robot's) feature representation
    image = Image.open(imagePath).convert("RGB")
    image= transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image)
    features=features.squeeze()
    features= features.flatten()
    return features

def BuildGraphFromCSV(csv_df, imageDirectory, labelMap, device):
    #Builds PyTorch Geometric Data Object representing entire dataset graph
    #Each unique image becomes a node; each image pair becomes an edge 
    #csv_df is the Pandas DataFrame from labels.csv
    #images is the folder containing the image files
    #labelMap is the dictionary mapping text labels to integers
    #device is the torch device cpu/cuda 
    baseCNN= models.resnet18(pretrained=True)
    featureExtractor = torch.nn.Sequential(*list(baseCNN.children())[:-1])
    featureExtractor.eval().to(device)

    #Pre-processing to resize & normalise 
    transform=T.Compose(
        [
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataList=[]
    cache={} #Cache features help to avoid re-encoding the same image twice
    for _, row in csv_df.iterrows():
        currentImage=row["current_image"]
        destinationImage=row["destination_image"]
        direction = row["direction"]
        currentPath=os.path.join(imageDirectory, currentImage)
        destinationPath=os.path.join(imageDirectory, destinationImage)
        #If image has already been viewed, re-use features
        if currentImage not in cache:
            cache[currentImage]=ExtractImageFeatures(currentPath, featureExtractor, transform, device)
        if destinationImage not in cache:
            cache[destinationImage]=ExtractImageFeatures(destinationPath, featureExtractor, transform, device)
        x=torch.stack([cache[currentImage], cache[destinationImage]]) #ensuring two nodes/graph
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        y=torch.tensor([labelMap[direction]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        dataList.append(data)
    return dataList