import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import models, transforms
import data_generator as dg

class ImageDataset(Dataset):

    def __init__(self, images_data, transformations, train=True):
        self.images_data = images_data
        self.transformations = transformations
        self.train = train

    def __len__(self):
        return len(self.transformations)

    def __getitem__(self, index):
        # Get images
        img_data = self.images_data[index]
        im0 = img_data['im0']
        im1 = img_data['im1']

        # Get transformation
        transformation = self.transformations[index]

        # Transform images in tensor
        toPil = transforms.ToPILImage()
        transformTensor = transforms.Compose([
            transforms.ToTensor(),
        ])
        im0 = torch.from_numpy(im0).float().unsqueeze(0)
        im1 = torch.from_numpy(im1).float().unsqueeze(0)

        # observations
        x0, y0 = img_data['x0'], img_data['y0']
        obs = dict({
            'x0': x0,
            'y0': y0,
            'transformation': transformation
        })

        return im0, im1, obs

def conv(in_planes, out_planes, kernel_size=3, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

class SquareNet(nn.Module):

    def __init__(self):
        super(SquareNet, self).__init__()
        # kernels
        self.conv1   = conv(2,   64, kernel_size=5, stride=2)
        self.conv2   = conv(64,  128, kernel_size=3, stride=2)
        self.conv3 = conv(128, 128)
        # an affine operation: y = Wx + b
        self.fc = nn.Linear(524288, 3) 

    # Forward function of the classification model
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.reshape(x, (x.shape[0],-1))
        x = self.fc(x)

        return x

if __name__ == '__main__':
    # Load input data and labels
    data = np.load("square_dataset.npz", allow_pickle=True, encoding="latin1")
    X = data['X']
    Y = data['Y']

    i = 25
    dataset = ImageDataset(X,Y)
    im0 = dataset[i][0]
    print(im0.type)
    im1 = dataset[i][1]
    x0 = dataset[i][2]['x0']
    y0 = dataset[i][2]['y0']
    delta_x, delta_y, delta_angle = dataset[i][2]['transformation']
    dg.show_results(im1.numpy()[0,:,:], im0.numpy()[0,:,:], x0, y0, delta_x, delta_y, delta_angle)

    # model = SquareNet()
    # model.load_state_dict(torch.load('net.pt'))
    # model.eval()
    # input_net = torch.cat((im0,im1),0).unsqueeze(0)
    # print(input_net.shape)
    # print("Network evaluation")
    # print(model(input_net))
    # print("Groundtruth")
    # print(delta_x, delta_y, delta_angle)
