# pylint: disable=E1101

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


def trainNet(net, trainLoader, evalLoader, ADF=False):
    """
    input : a network as nn.modulen, a train data loader and an evaluation dataloader
    train a given network and save the model as net.pt
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model architecture, Loss function & Optimizer
    net.to(device)
    criterion = nn.CrossEntropyLoss()
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
                # Load image and class inside a batch
                imgs = data[0].to(device)
                labels = data[1].to(device)

                # Set the gradient to zero
                optimizer.zero_grad()

                # compute or not the gradient
                with torch.set_grad_enabled(phase == 'train'):
                    # Apply the model & compute the loss
                    if ADF:
                        preds,_ = net(imgs)
                        loss = criterion(preds, labels)
                    else:
                        preds = net(imgs)
                        loss = criterion(preds, labels)

                    _, i = torch.max(preds, 1)

                    epoch_acc += torch.sum(i == labels)
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
