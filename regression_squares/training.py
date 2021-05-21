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
from data_generator import show_results
from utils import ImageDataset, SquareNet

def square_regression_loss(outputs, groundtruth, n_iterations=1):
    criterion = nn.MSELoss()
    losses_iter = []
    for i in range (n_iterations):
        result = outputs[f'iteration={i+1}']['transfo_output']
        losses_iter.append(criterion(result, groundtruth))
    loss = sum(losses_iter)
    return loss

def compute_corners(x0, y0, angle0, n_square=50):
    p0 = torch.stack((x0, y0))
    p1 = torch.stack(((x0 + n_square*torch.cos(angle0)), (y0 + n_square*torch.sin(angle0))))
    p3 = torch.stack(((x0 + n_square*torch.sin(angle0)), (y0 + n_square*torch.cos(angle0))))
    p2 = torch.stack(((p3[0] + n_square*torch.cos(angle0)), (p3[1] + n_square*torch.sin(angle0))))
    return p0, p1, p2, p3

def new_loss(outputs, groundtruth, obs, n_iterations=1):
    criterion = nn.MSELoss()
    losses_iter = []
    for i in range (n_iterations):
        x0 = obs['x0'] + groundtruth[:,0]
        y0 = obs['y0'] + groundtruth[:,1]
        angle0 = obs['angle0'] + groundtruth[:,2]
        p0_gt, p1_gt, p2_gt, p3_gt = compute_corners(x0, y0, angle0)
        result = outputs[f'iteration={i+1}']['transfo_output']
        x1 = obs['x0'] + result[:,0]
        y1 = obs['y0'] + result[:,1]
        angle1 = obs['angle0'] + result[:,2]
        p0_pred, p1_pred, p2_pred, p3_pred = compute_corners(x1, y1, angle1)
        res = (criterion(p0_gt, p0_pred)
            + criterion(p1_gt, p1_pred) 
            + criterion(p2_gt, p2_pred) 
            + criterion(p3_gt, p3_pred))/4
        losses_iter.append(res)
        
    loss = sum(losses_iter)
    return loss
 
def trainNet(net, trainLoader, evalLoader):
    """
    input : a network as nn.modulen, a train data loader and an evaluation dataloader
    train a given network and save the model as net.pt
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True)

    # Model architecture, Loss function & Optimizer
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=5e-4)

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

                # debug
                # print(im0.shape)
                # x0 = obs['x0'][1].numpy()
                # y0 = obs['y0'][1].numpy()
                # delta_x, delta_y, delta_angle = groundtruth[1,:].numpy()
                # show_results(im1.numpy()[1,0,:,:], im0.numpy()[1,0,:,:], x0, y0, delta_x, delta_y, delta_angle)

                # Set the gradient to zero
                optimizer.zero_grad()

                # compute or not the gradient
                with torch.set_grad_enabled(phase == 'train'):
                    preds = net(input_net, obs, n_iterations=2)
                    loss = square_regression_loss(preds, groundtruth.float(), n_iterations=2)
                    # loss = new_loss(preds, groundtruth.float(), obs, n_iterations=1)
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
    trainLoader = DataLoader(trainDataSet,batch_size=128,shuffle=True)
    evalLoader = DataLoader(testDataSet,batch_size=128,shuffle=True)

    net = SquareNet()
    trainNet(net, trainLoader, evalLoader)
    
