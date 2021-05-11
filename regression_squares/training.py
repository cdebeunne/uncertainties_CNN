import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
from utils import ImageDataset, SquareNet

def trainNet(net, trainLoader, evalLoader):
    """
    input : a network as nn.modulen, a train data loader and an evaluation dataloader
    train a given network and save the model as net.pt
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model architecture, Loss function & Optimizer
    net.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    lossTrainList = []
    lossValList = []
    epochs = []

    best_acc = 0.0
    num_epochs = 20
    # Training
    for epoch in range(num_epochs):
        epochs.append(epoch)
        for phase in ['train', 'val']:
            epoch_loss = []
            epoch_acc = 0

            if phase == 'train':
                # Model in training mod
                net.train()
                loader = trainLoader
            else:
                # Model in evaluation mod
                net.eval()
                loader = evalLoader

            for i, data in enumerate(loader):
                # Load images and transfo inside a batch
                im0 = data[0].to(device)
                im1 = data[1].to(device)
                input_net = torch.cat((im0,im1),1)

                obs = data[2]
                groundtruth = obs["transformation"].to(device)

                # Set the gradient to zero
                optimizer.zero_grad()

                # compute or not the gradient
                with torch.set_grad_enabled(phase == 'train'):
                    preds = net(input_net)
                    loss = criterion(preds, groundtruth.float())
                    epoch_loss.append(loss.item())

                    if phase == 'train':
                        # compute the gradient & update parameters
                        loss.backward()
                        optimizer.step()

            # Display loss
            mean_loss_epoch = np.mean(epoch_loss)
            std_loss_epoch = np.std(epoch_loss)

            if phase == 'train':
                lossTrainList.append(mean_loss_epoch)
                print(
                    f'Epoch: {epoch + 1}, Loss_train: {mean_loss_epoch:.4f} (± {std_loss_epoch:.4f})')
            else:
                lossValList.append(mean_loss_epoch)
                print(
                    f'Epoch: {epoch + 1}, Loss_val: {mean_loss_epoch:.4f} (± {std_loss_epoch:.4f})')

                if epoch_acc < best_acc:
                    # Keep curent model with copy function
                    best_acc = epoch_acc

    # Save best classification model
    torch.save(net.state_dict(), 'net.pt')

    plt.plot(epochs, lossTrainList, label="Training Loss")
    plt.plot(epochs, lossValList, label="Validation Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load input data and labels
    data = np.load("square_dataset.npz", allow_pickle=True, encoding="latin1")
    X = data['X']
    Y = data['Y']

    # Split the dataset in train, val and test data (with sklearn.model_selection.train_test_split function)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
    trainDataSet = ImageDataset(X_train,Y_train)
    testDataSet = ImageDataset(X_test,Y_test,train=False)
    trainLoader = DataLoader(trainDataSet,batch_size=256,shuffle=True)
    evalLoader = DataLoader(testDataSet,batch_size=64,shuffle=True)

    net = SquareNet()
    trainNet(net, trainLoader, evalLoader)
    
