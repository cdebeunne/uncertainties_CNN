import numpy as np
import pandas as pd
from scipy import ndimage
import imutils
import matplotlib.pyplot as plt

def rotateImage(img, angle, pivot):
    """
    a function that rotate an image with the pivot point as the
    center of rotation
    """
    if pivot[0]<0 or pivot[0]>255:
        print(img.shape)
        return img
    if pivot[1]<0 or pivot[1]>255:
        print(img.shape)
        return img
    padY = [img.shape[1] - pivot[0], pivot[0]]
    padX = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = imutils.rotate(imgP, angle)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

def generate_square(N, square_size, x0, y0, angle):
    """
    generate an image with a parametrized square
    """
    # we need integers values
    x0 = int(round(x0))
    y0 = int(round(y0))
    
    im = np.zeros((N,N))
    x1 = x0+square_size
    y1 = y0+square_size
 
    xv = np.arange(x0,x1)
    yv = np.arange(y0,y1)
    for x in xv:
        for y in yv:
            if (x<256) and (y<256):
                im[x,y] = 1
    
    im = rotateImage(im, angle, [x0,y0])
    return im

def show_results(im0, im1, x0, y0, delta_x, delta_y, delta_angle):
    """
    a plotter to illustrates the results 
    """
    h,w = im0.shape
    new_im = np.zeros((h,w))
    
    # we need integers values for deltas
    delta_x = int(round(delta_x))
    delta_y = int(round(delta_y))

    for x in range(w):
        for y in range(h):
            x_new = x+delta_x
            y_new = y+delta_y
            if (x_new < h) and (y_new<w):
                new_im[x_new,y_new] = im0[x,y]
    
    x1 = x0+delta_x
    y1 = y0+delta_y
    new_im = rotateImage(new_im, delta_angle, [x1,y1])
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(im0)
    plt.title("image 0")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(im1)
    plt.title("image 1")
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(new_im)
    plt.title("transformed image 0")
    plt.show()

def generate_dataset(N, square_size):
    X = []
    Y = []
    for i in range(1500):
        pad = round(np.sqrt(2)*square_size)
        x0 = np.random.random_integers(pad, N-pad)
        y0 = np.random.random_integers(pad, N-pad)
        # x1 = np.random.random_integers(square_size, N-square_size)
        # y1 =  np.random.random_integers(square_size, N-square_size)
        delta_pix = 10
        delta_theta = 10
        x1 = np.random.random_integers(-delta_pix,delta_pix) + x0
        y1 = np.random.random_integers(-delta_pix,delta_pix) + y0
        while (x1 > N-pad) or (y1 > N-pad) or (x1<pad) or (y1<pad):
            x1 = np.random.random_integers(-delta_pix,delta_pix) + x0
            y1 = np.random.random_integers(-delta_pix,delta_pix) + y0

        angle0 = np.random.random_integers(90)
        # angle1 = np.random.random_integers(90)
        angle1 = np.random.random_integers(-delta_theta,delta_theta) + angle0

        im0 = generate_square(N, square_size, x0, y0, angle0)
        im1 = generate_square(N, square_size, x1, y1, angle1)
        delta_x = x1-x0
        delta_y = y1-y0
        delta_angle = angle1-angle0

        x = dict({
            'im0':im0,
            'im1':im1,
            'x0':x0,
            'y0':y0,
            'angle0':angle0
        })
        X.append(x)
        Y.append([delta_x, delta_y, delta_angle])
    np.savez('square_dataset.npz', X=X, Y=Y, protocol=2)
    # df = pd.DataFrame({
    #     'X':X,
    #     'Y':Y,
    # })
    # df.to_pickle('square_dataset.pkl')


if __name__ == '__main__':
    N = 256
    square_size = 50
    generate_dataset(N, square_size)
    data = np.load('square_dataset.npz', allow_pickle=True)
    X = data['X']
    Y = data['Y']
    i = 30
    im0 = X[i]['im0']
    im1 = X[i]['im1']
    x0 = X[i]['x0']
    y0 = X[i]['y0']
    delta_x, delta_y, delta_angle = Y[i]
    print(x0, y0, X[i]['angle0'])
    show_results(im0, im1, x0, y0, delta_x, delta_y, delta_angle)
    
    
