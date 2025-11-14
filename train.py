'''
This file trains the Graph Neural Network model on the Gazebo 'images' dataset. 
Notes: The model loads the dataset from labels.csv and then initialises the GNN model. 
After initialising, it trains with x-entropy loss to evaluate accuracy. 
'''

import torch 
from torch_geometric.loader import DataLoader
from dataset import NavGraphDataset
from GNNmodel import GNN

#configuring
CSVpath="labels.csv"
imageDirectory="images"
device =torch.device("cuda" if torch.cuda.is_available()else "cpu")
EPOCHS=20
batchSize=8
LR=1e-3
hiddenDimension=128

#training model
def train():
    Dataset=NavGraphDataset(CSVpath, imageDirectory, device) #loads dataset

    #to split sample dataset
    trainSize=int(0.8*len(Dataset))
    testSize=len(Dataset)-trainSize
    trainDataset, testDataset=torch.utils.data.random_split(Dataset, [trainSize, testSize])
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    test_Loader=DataLoader(testDataset, batch_size=batchSize, shuffle=False)

    #initailising model
    inChannels=Dataset.graphs[0].x.shape[1]
    numClasses=len(Dataset.LabelMap)
    model=GNN(inChannels, hiddenDimension, numClasses).to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=LR)
    criterion=torch.nn.NLLLoss() #bc model outputs via log softmax

    #training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss=0
        for batch in trainLoader:
            batch=batch.to(device)
            optimizer.zero_grad()
            out=model(batch.x, batch.edge_index, batch.batch)
            loss=criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss +=loss.item()
        avg_loss =total_loss/len(trainLoader)
        acc=evaluate (model, test_Loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}| Test Acc: {acc:.2f}%")

    torch.save(model.state_dict(), "trainedGNN.pth")
    print("Training is complete. Model has been saved as trainedGNN.pth")

def evaluate(model, loader):
    #determining accuracy on the test set
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for batch in loader:
            batch=batch.to(device)
            out=model(batch.x, batch.edge_index, batch.batch)
            preds=out.argmax(dim=1)
            correct += (preds==batch.y).sum().item()
            total += batch.y.size(0)
    return 100.0 * correct /total if total >0 else 0

if __name__=="__main__":
    train()

