import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import models, transforms
from data_generator import show_results, generate_square, rotateImage

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
        im0 = torch.from_numpy(im0).float().unsqueeze(0)
        im1 = torch.from_numpy(im1).float().unsqueeze(0)

        # observations
        x0, y0, angle0 = img_data['x0'], img_data['y0'], img_data['angle0']
        obs = dict({
            'x0': x0,
            'y0': y0,
            'angle0': angle0,
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
        self.conv   = conv(2,   16, kernel_size=5, stride=2)
        # batch normalization
        self.bn = nn.BatchNorm2d(16)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(262144, 256)
        self.fc2 = nn.Linear(256, 3) 
        self.fc = nn.Linear(262144,3)

    def net_forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = torch.reshape(x, (x.shape[0],-1))
        # x = self.fc1(x)
        # x = self.fc2(x)
        x = self.fc(x)

        return x
    
    def update_transformation(self, transfo_input, transfo_output):
        transfo_updated = transfo_output + transfo_input
        return transfo_updated
    
    def render(self, input_images, obs, transfo_output):
        output_images = torch.zeros(input_images.shape)
        N = 256
        square_size = 50
        output_images[:][0][:][:] = input_images[:][0][:][:]
        index = 0
        # We need to define a new input transformation
        # and to render new images
        if isinstance(obs['x0'], np.int64):
            delta_x, delta_y, delta_angle = transfo_output[0].detach().numpy()
            new_x = obs['x0'] + delta_x 
            new_y = obs['y0'] + delta_y 
            new_angle = obs['angle0'] + delta_angle 
            new_im = generate_square(N, square_size, new_x, new_y, new_angle)
            new_im = torch.from_numpy(new_im).float()
            output_images[index][1][:][:] = new_im

        else:
            for transformation in transfo_output:
                delta_x, delta_y, delta_angle = transformation[:].detach().numpy()
                new_x = obs['x0'][index].numpy() + delta_x 
                new_y = obs['y0'][index].numpy() + delta_y
                new_angle = obs['angle0'][index].numpy() + delta_angle 
                new_im = generate_square(N, square_size, new_x, new_y, new_angle)
                new_im = torch.from_numpy(new_im).float()
                output_images[index][1][:][:] = new_im
                index += 1
        return output_images


    def forward(self, input_net, obs, n_iterations=1):
        input_images = input_net.clone()
        batch_size = input_images.shape[0]
        transfo_input = torch.zeros((batch_size,3))
        outputs = dict()

        for i in range(n_iterations):
            transfo_input = transfo_input.detach()
            model_output = self.net_forward(input_images)
            transfo_output = self.update_transformation(transfo_input, model_output)
            outputs[f'iteration={i+1}'] = {
                'transfo_input': transfo_input,
                'transfo_output': transfo_output,
                'model_output': model_output,
            }
            
            input_images = self.render(input_images, obs, transfo_output)
            transfo_input = transfo_output

        return outputs

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
    obs = dataset[i][2]
    x0 = dataset[i][2]['x0']
    y0 = dataset[i][2]['y0']
    delta_x, delta_y, delta_angle = dataset[i][2]['transformation']
    #show_results(im1.numpy()[0,:,:], im0.numpy()[0,:,:], x0, y0, delta_x, delta_y, delta_angle)

    model = SquareNet()
    input_net = torch.cat((im0,im1),0).unsqueeze(0)
    print(input_net.shape)
    print(model(input_net, obs))

