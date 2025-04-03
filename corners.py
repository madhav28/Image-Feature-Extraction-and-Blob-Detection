import os
from common import read_img, save_img 
import matplotlib.pyplot as plt
import numpy as np
from filters import * 
import cv2

def corner_score(image, u=5, v=5, window_size=(5,5)):
    # Given an input image, x_offset, y_offset, and window_size,
    # return the function E(u,v) for window size W
    # corner detector score for that pixel.
    # Input- image: H x W
    #        u: a scalar for x offset
    #        v: a scalar for y offset
    #        window_size: a tuple for window size
    #
    # Output- results: a image of size H x W
    # Use zero-padding to handle window values outside of the image. 

    H, W = image.shape
    h, w = window_size
    img = np.pad(image, ((h//2+abs(u), h//2+abs(u)), (w//2+abs(v), w//2+abs(v))), mode='constant')     
    output = np.zeros(img.shape)

    for i in range(H):
        for j in range(W):
            w_orig = img[i+abs(u):i+abs(u)+2*h//2, j+abs(v):j+abs(v)+2*w//2]
            w_shift = img[i+abs(u)+u:i+abs(u)+2*h//2+u, j+abs(v)+v:j+abs(v)+2*w//2+v]
            output[i,j] = np.sum((w_shift-w_orig)**2)

    return output

def harris_detector(image, window_size=(5,5)):
    # Given an input image, calculate the Harris Detector score for all pixels
    # Input- image: H x W
    # Output- results: a image of size H x W
    # 
    # You can use same-padding for intensity (or zero-padding for derivatives) 
    # to handle window values outside of the image. 

    ## compute the derivatives 
    Ix = convolve(image, np.array([[0.5, 0, -0.5]]))
    Iy = convolve(image, np.array([[0.5], [0], [-0.5]])) 

    Ixx = Ix*Ix
    Iyy = Iy*Iy
    Ixy = Ix*Iy

    H, W = image.shape
    h, w = window_size
    img = np.pad(image, ((h//2, h//2), (w//2, w//2)), mode='constant')

    kernel_gaussian = np.zeros((5,5))
    var = 1/(2*np.log(2))
    for i in range(5):
        for j in range(5):
            kernel_gaussian[i,j] = 1/(2*np.pi*var)*np.exp(-((i-2)**2+(j-2)**2)/(2*var))
    
    M = np.zeros((H, W, 3))
    M[:,:,0] = convolve(Ixx, kernel_gaussian)
    M[:,:,1] = convolve(Ixy, kernel_gaussian)
    M[:,:,2] = convolve(Iyy, kernel_gaussian)
    
    # For each location of the image, construct the structure tensor and calculate the Harris response
    alpha = 0.05
    response = M[:,:,0]*M[:,:,2] - M[:,:,1]**2 - alpha*(M[:,:,0] + M[:,:,2])**2

    return response

def main():
    # The main function
    ########################
    img = read_img('./data/grace_hopper.png')

    ##### Feature Detection #####  
    if not os.path.exists("./results/feature_detection"):
        os.makedirs("./results/feature_detection")

    # define offsets and window size and calulcate corner score
    u, v, W = -5, 0, (5,5)
    score = corner_score(img, u, v, W)
    save_img(score, "./results/feature_detection/corner_score_left.png")
    u, v, W = 5, 0, (5,5)
    score = corner_score(img, u, v, W)
    save_img(score, "./results/feature_detection/corner_score_right.png")
    u, v, W = 0, 5, (5,5)
    score = corner_score(img, u, v, W)
    save_img(score, "./results/feature_detection/corner_score_up.png")
    u, v, W = 0, -5, (5,5)
    score = corner_score(img, u, v, W)
    save_img(score, "./results/feature_detection/corner_score_down.png")

    harris_corners = harris_detector(img)
    save_img(harris_corners, "./results/feature_detection/harris_response.png")
     

if __name__ == "__main__":
    main()
