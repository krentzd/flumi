#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PIPELINE FOR SIMULATING FM

# 1) Choose pixel-size of voxel grid and scale input fluorophores accordingly DONE
# 2) Read in segmentations and randomly distribute (choose) points from contours DONE
# 3) Scale input PSF to fit voxel grid by fourier or bicubic interpolation --> Match spatial sampling to be same as voxel grid DONE
# 4) Assign fluorophore photon emission as Poisson distribution with a mean that represents average photon emission (e.g. 1000) DONE
# 5) Convolve voxel grid with PSF DONE
# 6) Simulate camera resolution by binning output of voxel grid (e.g. from 5x5x5nm to 100x100x250nm) DONE
# 7) Noise: a) sensor noise from detection accuracy:  Draw value from Poisson distribution of each pixel using pixel-value as mean DONE
# 8) Noise: b) convert absolute photon count simulating analog-to-digiatl conversion with gain set by user (e.g. 1) --> convert from float to int DONE
# 9) Noise: c) additive electronic noise: add Gaussian noise to each pixel using DC-offset as mean (e.g. 100) and sigma (e.g. 1 or 2) DONE
# 10) Save as stack or tiff

import random
import tifffile
import os
import napari

import numpy as np

from tensorly.decomposition import parafac

import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.io import imsave

from scipy.signal import gaussian
from scipy.signal import convolve
from scipy.ndimage import zoom
from scipy.optimize import curve_fit

from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm

def make_directory(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)

def save_image_grid(save_path, grid_shape, *imgs):

    grid_height, grid_width = grid_shape

    fig = plt.figure()

    i = 1
    for img in imgs:
        ax = fig.add_subplot(grid_height, grid_width , i)
        ax.imshow(np.squeeze(img.astype(int)), vmin=0, vmax=255)
        ax.axis('off')
        i += 1

    plt.savefig(save_path)


def visualize(*X):

    color_dict = {0: 'red', 1: 'blue', 2: 'orange', 3: 'green', 4: 'cyan', 5: 'purple', 6: 'brown', 7: 'pink', 8: 'gray', 9: 'olive'}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, x in enumerate(X):
        ax.scatter(x[:,0],  x[:,1], x[:,2], color=color_dict[i], label=str(i))
        ax.legend(loc='upper left', fontsize='x-large')
    plt.show()


def find_divisors(number):

    divisors = []

    for x in range(1, number+1):
        if(number % x == 0):
            divisors.append(x)

    return divisors


def get_shape(divisors, threshold):

    # Smaller threshold leads to smaller memory requirements but slower simulation
    divisors_temp = [0 if divisor < threshold else divisor for divisor in divisors]

    filtered_divisors = list(filter(lambda a: a != 0, divisors_temp))

    # Safe choice, since all dimensions rounded up to nearest multiple of 10
    if len(filtered_divisors) == 0:
        shape = 10
    else:
        shape = min(filtered_divisors)

    return shape


def get_box_shape(image_shape, kernel_size):

    divisors_x = find_divisors(image_shape[0])
    divisors_y = find_divisors(image_shape[1])
    divisors_z = find_divisors(image_shape[2])

    shape_x = get_shape(divisors_x, kernel_size[0])
    shape_y = get_shape(divisors_y, kernel_size[1])
    shape_z = get_shape(divisors_z, kernel_size[2])

    return (shape_x, shape_y, shape_z)

def roundup(x):
    return x if x % 10 == 0 else x + 10 - x % 10

def fit_gaussian(x):

    def gaussian(x, a, b, c):
        return a * np.exp(-((x-b)**2)/(2*c**2))

    xdata = np.linspace(0, len(x), len(x))

    params, __ = curve_fit(gaussian, xdata, x, p0=[1, len(x)/2, 1])

    return params

def get_gaussian(params, scaling):
    return params[0]*np.expand_dims(gaussian(int(params[2])*12*scaling, std=params[2]*scaling), 1)

# Detector part
def xyz_binning(image, x_bin_size, y_bin_size, z_bin_size):

    binned_image = np.zeros((int(image.shape[0]/x_bin_size)+1, int(image.shape[1]/y_bin_size)+1, int(image.shape[2]/z_bin_size)+1))
    num_bins = int(image.shape[0]*image.shape[1]*image.shape[2])/(x_bin_size*y_bin_size*z_bin_size)

    x_last, y_last, z_last = (0, 0, 0)

    for i in tqdm(range(0, int(num_bins)), desc='Binning'):

        bin = image[x_last:x_last+x_bin_size, y_last:y_last+y_bin_size, z_last:z_last+z_bin_size]

        if bin.size == 0:
            continue
        # Noise from detector is distributed as Poisson --> how well does sensor pick up value
        detected_value = np.random.poisson(np.max(bin))
        # Set gain and do ADC
        gain = 1
        digital_value = int(detected_value)*gain
        # Add Gaussian noise
        pixel_value = digital_value + np.random.normal(700, 300)

        # print('Bin value: ', pixel_value)
        binned_image[int(x_last/x_bin_size), int(y_last/y_bin_size), int(z_last/z_bin_size)] = pixel_value

        x_last += x_bin_size

        if (x_last >= image.shape[0]) and (y_last <= image.shape[1]):
            # print('In x')
            x_last = 0
            y_last += y_bin_size

        if (y_last >= image.shape[1]):
            # print('In z')
            x_last = 0
            y_last = 0
            z_last += z_bin_size

        # print('Last val: ', x_last, y_last, z_last)

    return binned_image


def separable3Dconvolve(image, kernel, scaling):

    scaling_xy, scaling_z = scaling

    # 3D SVD decomposition
    try:
        x, y, z = parafac(kernel, 1)
    except ValueError:
        __, (x, y, z) = parafac(kernel, 1)

    # x, y, z = (np.expand_dims(gaussian(256, std=10), 1), np.expand_dims(gaussian(256, std=10), 1), np.expand_dims(gaussian(65, std=10), 1))

    gauss_params = np.abs(fit_gaussian(np.array(np.squeeze(x))))
    x = get_gaussian(gauss_params, scaling_xy)

    gauss_params = np.abs(fit_gaussian(np.array(np.squeeze(y))))
    y = get_gaussian(gauss_params, scaling_xy)

    gauss_params = np.abs(fit_gaussian(np.array(np.squeeze(z))))
    z = get_gaussian(gauss_params, scaling_z)

    # Output from convolution is M+N-1 where M is input image and N is kernel size
    conv_0 = np.memmap('conv_0.dat', dtype='float64', mode='w+', shape=(image.shape[0]+len(x)-1, image.shape[1], image.shape[2]))
    # conv_0 = np.zeros((image.shape[0]+len(x)-1, image.shape[1], image.shape[2]))
    conv_1 = np.memmap('conv_1.dat', dtype='float64', mode='w+', shape=(image.shape[0]+len(x)-1, image.shape[1]+len(y)-1, image.shape[2]))
    # conv_1 = np.zeros((image.shape[0]+len(x)-1, image.shape[1]+len(y)-1, image.shape[2]))
    conv_2 = np.memmap('conv_2.dat', dtype='float64', mode='w+', shape=(image.shape[0]+len(x)-1, image.shape[1]+len(y)-1, image.shape[2]+len(z)-1))
    # conv_2 = np.zeros((image.shape[0]+len(x)-1, image.shape[1]+len(y)-1, image.shape[2]+len(z)-1))

    for i in range(0, image.shape[1]):
        for j in range(0, image.shape[2]):
            conv_0[:,i,j] = convolve(image[:,i,j], np.squeeze(x))[:]

    for i in range(0, conv_0.shape[0]):
        for j in range(0, conv_0.shape[2]):
            conv_1[i,:,j] = convolve(conv_0[i,:,j], np.squeeze(y))[:]

    for i in range(0, conv_1.shape[0]):
        for j in range(0, conv_1.shape[1]):
            conv_2[i,j,:] = convolve(conv_1[i,j,:], np.squeeze(z))[:]

    return conv_2

def box3Dconvolve(image, psf, scaling):

    scaling_xy, scaling_z = scaling

    # This step has to be done sequentially on various sub-cubes (boxes)
    # Create emtpy array
    conv_shape = (int(image.shape[0]+20*scaling_xy), int(image.shape[1]+20*scaling_xy-1), int(image.shape[2]+20*scaling_z-1))
    # conv_image = np.zeros(conv_shape)
    conv_image = np.memmap('conv_image.dat', dtype='float64', mode='w+', shape=conv_shape)

    # Loop over empty array and populate with output from convolution step sequentially in for loop
    box_shape = get_box_shape(image.shape, psf.shape)
    num_box = (image.shape[0]*image.shape[1]*image.shape[2])/(box_shape[0]*box_shape[1]*box_shape[2])

    assert int(num_box) == num_box, 'Box does not fit image'

    x_last, y_last, z_last = (0, 0, 0)

    for i in tqdm(range(0, int(num_box)), desc='Convolving'):

        box = image[x_last:x_last+box_shape[0], y_last:y_last+box_shape[1], z_last:z_last+box_shape[2]]

        if np.mean(box) > 0:
            conv_box = separable3Dconvolve(box, psf, scaling)
            conv_image[x_last:x_last+conv_box.shape[0], y_last:y_last+conv_box.shape[1], z_last:z_last+conv_box.shape[2]] += conv_box

            x_last += box_shape[0]

            if (x_last == image.shape[0]) and (y_last != image.shape[1]):
                x_last = 0
                y_last += box_shape[1]

            if (y_last == image.shape[1]):
                x_last = 0
                y_last = 0
                z_last += box_shape[2]

            conv_image.flush

    return conv_image

def additive3Dconvolve(image, psf, scaling):

    scaling_xy, scaling_z = scaling

    x, y, z = np.nonzero(image)

    # This step has to be done sequentially on various sub-cubes (boxes)
    # Create emtpy array
    conv_shape = (int(image.shape[0]+20*scaling_xy), int(image.shape[1]+20*scaling_xy-1), int(image.shape[2]+20*scaling_z-1))
    # conv_image = np.zeros(conv_shape)
    conv_image = np.memmap('conv_image.dat', dtype='float64', mode='w+', shape=conv_shape)

    # Loop over empty array and populate with output from convolution step sequentially in for loop
    box_shape = (1, 1, 1)

    x_last, y_last, z_last = (0, 0, 0)

    for i in tqdm(range(0, len(x)), desc='Convolving'):

        box = image[x[i]:x[i]+1, y[i]:y[i]+1, z[i]:z[i]+1]

        conv_box = separable3Dconvolve(box, psf, scaling)

        conv_image[x_last:x_last+conv_box.shape[0], y_last:y_last+conv_box.shape[1], z_last:z_last+conv_box.shape[2]] += conv_box

        x_last += box_shape[0]

        if (x_last == image.shape[0]) and (y_last != image.shape[1]):
            x_last = 0
            y_last += box_shape[1]

        if (y_last == image.shape[1]):
            x_last = 0
            y_last = 0
            z_last += box_shape[2]

        conv_image.flush

    return conv_image


if __name__ == '__main__':

    psf_raw = tifffile.imread('PSF_BW.tif')
    # PSF is (z, x, y) --> Must switch axes
    psf = np.swapaxes(psf_raw, 0, 2)

    # with napari.gui_qt():
    #     viewer = napari.view_image(psf)

    coordinates = (np.loadtxt('segmentation.txt'))

    image_shape = ((roundup(int(np.max(coordinates[:,0]))),
                    roundup(int(np.max(coordinates[:,1]))),
                    roundup(int(np.max(coordinates[:,2])))))

    image = np.zeros(image_shape)

    x_c = coordinates[:,0].astype(int)
    y_c = coordinates[:,1].astype(int)
    z_c = coordinates[:,2].astype(int)

    c = x_c, y_c, z_c

    print('Seeding fluorophores...')
    poisson_dist = np.random.poisson(1000, len(x_c))
    image[c] = poisson_dist[:]

    scaling = (20, 50)

    conv_shape = (int(image.shape[0]+20*scaling[0]), int(image.shape[1]+20*scaling[0]-1), int(image.shape[2]+20*scaling[1]-1))
    conv_image = tifffile.memmap('raw_conv_image.tif', dtype='float64', mode='w+', shape=conv_shape)

    conv_image[:] = box3Dconvolve(image, psf, scaling)[:]

    binned_conv_image = xyz_binning(conv_image, scaling[0], scaling[0], scaling[1])

    # Swap axes for ImageJ format
    binned_conv_image = np.swapaxes(binned_conv_image, 2, 0)

    tifffile.imwrite('out.tif', binned_conv_image, imagej=True)

    save_dir = 'z_stack_new_bin_method/'
    make_directory(save_dir)
    for i in tqdm(range(0, binned_conv_image.shape[2]), desc='Saving stack'):
        print(np.min(binned_conv_image[:,:,i]), np.max(binned_conv_image[:,:,i]))
        plt.imshow(binned_conv_image[:,:,i], vmin=0, vmax=10000, cmap='magma')
        plt.savefig(save_dir + str(i) + '.png')
