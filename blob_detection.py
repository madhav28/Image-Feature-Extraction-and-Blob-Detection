from common import * 
import matplotlib.pyplot as plt
import numpy as np 
from filters import *
import os
import cv2

def gaussian_filter(image, sigma):
    # Given an image, apply a Gaussian filter with the input kernel size
    # and standard deviation 
    # Input-    image: image of size HxW
    #           sigma: scalar standard deviation of Gaussian Kernel
    # Output-   Gaussian filtered image of size HxW
    H, W = image.shape
    # -- good heuristic way of setting kernel size 
    kernel_size = int(2 * np.ceil(2*sigma) + 1)

    # make sure that kernel size isn't too big and is odd 
    kernel_size = min(kernel_size, min(H,W)//2)     
    if kernel_size % 2 == 0: kernel_size = kernel_size + 1  

    #TODO implement gaussian filtering with size kernel_size x kernel_size 
    # feel free to use your implemented convolution function or a convolution function from a library
    kernel_gaussian = np.zeros((kernel_size, kernel_size))
    k = (kernel_size - 1)/2
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel_gaussian[i,j] = 1/(2*np.pi*sigma**2)*np.exp(-((i-k)**2+(j-k)**2)/(2*sigma**2))   

    return convolve(image, kernel_gaussian)

def scale_space(image, min_sigma, k=np.sqrt(2), S=8):
    # Calcualtes a DoG scale space of the image
    # Input-    image: image of size HxW
    #           min_sigma: smallest sigma in scale space
    #           k: scalar multiplier for scale space
    #           S: number of scales considers
    # Output-   Scale Space of size HxWx(S-1)
    H, W = image.shape
    output = np.zeros([H, W, S-1])
    for i in range(S-1):
        sigma_1 = k**i
        sigma_2 = k**(i+1)
        gauss_1 = gaussian_filter(image, sigma_1)
        gauss_2 = gaussian_filter(image, sigma_2)
        output[:,:,i] = gauss_1 - gauss_2
    return output


##### You shouldn't need to edit the following 3 functions 
def find_maxima(scale_space, k_xy=5, k_s=1):
    # Extract the peak x,y locations from scale space
    # Input-    scale_space: Scale space of size HxWxS
    #           k: neighborhood in x and y 
    #           ks: neighborhood in scale
    # Output-   list of (x,y) tuples; x<W and y<H
    if len(scale_space.shape) == 2:
        scale_space = scale_space[:, :, None] 

    H,W,S = scale_space.shape
    maxima = []
    for i in range(H):
        for j in range(W):
            for s in range(S):
                # extracts a local neighborhood of max size (2k_xy+1, 2k_xy+1, 2k_s+1)
                neighbors = scale_space[max(0, i-k_xy):min(i+k_xy,H), 
                                        max(0, j-k_xy):min(j+k_xy,W), 
                                        max(0, s-k_s) :min(s+k_s,S)]
                mid_pixel = scale_space[i,j,s]
                num_neighbors = np.prod(neighbors.shape) - 1
                # if mid_pixel is larger than all the neighbors; append maxima 
                if np.sum(mid_pixel > neighbors) == num_neighbors:
                    maxima.append( (i,j,s) )
    return maxima

def visualize_scale_space(scale_space, min_sigma, k, file_path=None):
    # Visualizes the scale space
    # Input-    scale_space: scale space of size HxWxS
    #           min_sigma: the minimum sigma used 
    #           k: the sigma multiplier 
    if len(scale_space.shape) == 2:
        scale_space = scale_space[:, :, None] 
    H, W, S = scale_space.shape

    # number of subplots
    p_h = int(np.floor(np.sqrt(S))) 
    p_w = int(np.ceil(S/p_h))
    for i in range(S):
        plt.subplot(p_h, p_w, i+1)
        plt.axis('off')
        plt.title('{:.1f}:{:.1f}'.format(min_sigma * k**i, min_sigma * k**(i+1)))
        plt.imshow(scale_space[:, :, i])

    # plot or save to fig 
    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()    
    plt.close()

def visualize_maxima(image, maxima, min_sigma, k, file_path=None):
    # Visualizes the maxima on a given image
    # Input-    image: image of size HxW
    #           maxima: list of (x,y) tuples; x<W, y<H
    #           file_path: path to save image. if None, display to screen
    # Output-   None 
    H, W = image.shape
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    for maximum in maxima:
        y,x,s= maximum 
        assert x < W and y < H and x >= 0 and y >= 0
        radius = np.sqrt(2 * min_sigma * (k ** s))
        circ = plt.Circle((x, y), radius, color='r', fill=False)
        ax.add_patch(circ)

    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()    
    plt.close()


def main():
    image = read_img('./data/polka.png')

    ### -- Detecting Polka Dots -- ## 
    print("Detect small polka dots")
    # -- Detect Small Circles
    sigma_1, sigma_2 = 2.5, 4
    gauss_1 = gaussian_filter(image, sigma_1)
    gauss_2 = gaussian_filter(image, sigma_2)

    # calculate difference of gaussians
    DoG_small = gauss_1 - gauss_2

    # visualize maxima 
    maxima = find_maxima(DoG_small, k_xy=int(sigma_1))
    visualize_scale_space(DoG_small, sigma_1, sigma_2/sigma_1,'./results/polka_small_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2/sigma_1, './results/polka_small.png')
    
    # -- Detect Large Circles
    print("Detect large polka dots")
    sigma_1, sigma_2 = 7, 10.5
    gauss_1 = gaussian_filter(image, sigma_1)
    gauss_2 = gaussian_filter(image, sigma_2)

    # calculate difference of gaussians 
    DoG_large = gauss_1 - gauss_2
    
    # visualize maxima 
    # Value of k_xy is a sugguestion; feel free to change it as you wish.
    maxima = find_maxima(DoG_large, k_xy=int(sigma_1))
    visualize_scale_space(DoG_large, sigma_1, sigma_2/sigma_1, './results/polka_large_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2/sigma_1, './results/polka_large.png')


    # ## -- TODO Implement scale_space() and try to find both polka dots 
    scale_imgs = scale_space(image, 2.5)
    visualize_scale_space(scale_imgs, 2.5, np.sqrt(2), './results/polka_scale_imgs_DoG.png')

    for k_xy in range(2, 7, 2):
        for k_s in range(1, 4):
            maxima = find_maxima(scale_imgs, k_xy = k_xy, k_s=k_s)
            if not os.path.exists('./results/polka_maximas'):
                os.makedirs('./results/polka_maximas')
            visualize_maxima(image, maxima, 2.5, np.sqrt(2), f'./results/polka_maximas/polka_maximas_kxy_{k_xy}_ks_{k_s}.png')


    # ## -- TODO Try to find the cells in any of the cell images in vgg_cells 
    for idx, f in enumerate(os.listdir('./data/cells')):
        if idx == 4:
            break

        image = read_img(os.path.join('./data/cells', f))
        image = cv2.GaussianBlur(image, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        threshold = (np.max(image)+np.min(image))*0.5
        image[image < threshold] = 0

        scale_imgs = scale_space(image, 2.5)
        maxima = find_maxima(scale_imgs, k_xy = 3, k_s = 3)
        print(f)
        print(f'Number of cells = {len(maxima)}')
        print()
        if not os.path.exists('./results/cells_maximas'):
            os.makedirs('./results/cells_maximas')
        visualize_maxima(image, maxima, 2.5, np.sqrt(2), os.path.join('./results/cells_maximas', f))


if __name__ == '__main__':
    main()
