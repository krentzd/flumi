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

    print('Binned image size: ', binned_image.shape)
    print('Image size: ', image.shape)

    num_bins = int(image.shape[0]*image.shape[1]*image.shape[2])/(x_bin_size*y_bin_size*z_bin_size)

    x_last, y_last, z_last = (0, 0, 0)

    print('Binning image...')

    for i in tqdm(range(0, int(num_bins))):

        # print(int(x_last/x_bin_size), int(y_last/y_bin_size), int(z_last/z_bin_size))

        # print('Bin image values: ', int(image.shape[0]/x_bin_size), int(image.shape[0]/y_bin_size), int(image.shape[0]/z_bin_size))

        try:
            # Noise from detector is distributed as Poisson --> how well does sensor pick up value
            detected_value = np.random.poisson(np.max(image[x_last:x_last+x_bin_size, y_last:y_last+y_bin_size, z_last:z_last+z_bin_size]))
            # Set gain and do ADC
            gain = 1
            digital_value = int(detected_value)*gain
            # Add Gaussian noise
            pixel_value = digital_value + np.random.normal(700, 300)

            # print('Bin value: ', pixel_value)
            binned_image[int(x_last/x_bin_size), int(y_last/y_bin_size), int(z_last/z_bin_size)] = pixel_value
        except ValueError:
            pass
            # print('Failed bin: ', x_bin_size-x_last, y_bin_size-y_last, z_bin_size-z_last)

        x_last += x_bin_size

        if (x_last >= image.shape[0]) and (y_last <= image.shape[1]):
            # print('In x')
            x_last = 0
            y_last += y_bin_size

        if (y_last >= image.shape[1]) and (x_last >= image.shape[0]):
            # print('In z')
            x_last = 0
            y_last = 0
            z_last += z_bin_size

        # print('Last val: ', x_last, y_last, z_last)

    return binned_image


def separable3Dconvolve(image, kernel, scaling):

    scaling_xy, scaling_z = scaling

    # 3D SVD decomposition
    x, y, z = parafac(kernel, 1)
    # x, y, z = (np.expand_dims(gaussian(256, std=10), 1), np.expand_dims(gaussian(256, std=10), 1), np.expand_dims(gaussian(65, std=10), 1))
    print(x.shape, y.shape, z.shape)

    plt.plot(x)
    plt.plot(y)
    plt.plot(z)
    # plt.show()

    xy = np.dot(x, y.T)
    xy_rank = np.linalg.matrix_rank(xy)
    print(xy.shape, xy_rank)

    gauss_params = np.abs(fit_gaussian(np.array(np.squeeze(x))))
    print(gauss_params[0], gauss_params[1], gauss_params[2])
    x = get_gaussian(gauss_params, scaling_xy)
    plt.plot(x)

    gauss_params = np.abs(fit_gaussian(np.array(np.squeeze(y))))
    print(gauss_params[0], gauss_params[1], gauss_params[2])
    y = get_gaussian(gauss_params, scaling_xy)
    plt.plot(y)

    gauss_params = np.abs(fit_gaussian(np.array(np.squeeze(z))))
    print(gauss_params[0], gauss_params[1], gauss_params[2])
    z = get_gaussian(gauss_params, scaling_z)
    plt.plot(z)

    # plt.show()

    # xyz = np.zeros((xy.shape[0], xy.shape[1], z.shape[0]))
    # for i in range(0, z.shape[0]):
    #     xyz[:,:,i] = xy[:,:]*z[i]
    # print(xyz.shape)

    # x, y, z = parafac(xyz, 1)

    # print(np.max(x_t), np.max(y_t), np.max(z_t))
    # print(np.max(x_t)*np.max(y_t)*np.max(z_t))
    #
    # plt.plot(x_t)
    # plt.plot(y_t)
    # plt.plot(z_t)
    # plt.show()

    # for i in tqdm(range(0, xyz.shape[2])):
    #     plt.imshow(xyz[:,:,i], vmin=0, vmax=1, cmap='magma')
    #     plt.savefig('psf/' + str(i) + '.png')

    # Output from convolution is M+N-1 where M is input image and N is kernel size
    conv_0 = np.memmap('conv_0.dat', dtype='float64', mode='w+', shape=(image.shape[0]+len(x)-1, image.shape[1], image.shape[2]))
    # conv_0 = np.zeros((image.shape[0]+len(x)-1, image.shape[1], image.shape[2]))
    conv_1 = np.memmap('conv_1.dat', dtype='float64', mode='w+', shape=(image.shape[0]+len(x)-1, image.shape[1]+len(y)-1, image.shape[2]))
    # conv_1 = np.zeros((image.shape[0]+len(x)-1, image.shape[1]+len(y)-1, image.shape[2]))
    conv_2 = np.memmap('conv_2.dat', dtype='float64', mode='w+', shape=(image.shape[0]+len(x)-1, image.shape[1]+len(y)-1, image.shape[2]+len(z)-1))
    # conv_2 = np.zeros((image.shape[0]+len(x)-1, image.shape[1]+len(y)-1, image.shape[2]+len(z)-1))

    print('Convolving in X...')
    for i in range(0, image.shape[1]):
        for j in range(0, image.shape[2]):
            conv_0[:,i,j] = convolve(image[:,i,j], np.squeeze(x))[:]

    print(np.min(conv_0), np.max(conv_0))

    print('Convolving in Y...')
    for i in range(0, conv_0.shape[0]):
        for j in range(0, conv_0.shape[2]):
            conv_1[i,:,j] = convolve(conv_0[i,:,j], np.squeeze(y))[:]

    print(np.min(conv_1), np.max(conv_1))

    print('Convolving in Z...')
    for i in range(0, conv_1.shape[0]):
        for j in range(0, conv_1.shape[1]):
            conv_2[i,j,:] = convolve(conv_1[i,j,:], np.squeeze(z))[:]

    print(np.min(conv_2), np.max(conv_2))

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
    print(box_shape)
    num_box = (image.shape[0]*image.shape[1]*image.shape[2])/(box_shape[0]*box_shape[1]*box_shape[2])

    assert int(num_box) == num_box, 'Box does not fit image'

    x_last, y_last, z_last = (0, 0, 0)

    for i in range(0, int(num_box)):
        print('Iteration: {}/{}'.format(i+1, int(num_box)))
        box = image[x_last:x_last+box_shape[0], y_last:y_last+box_shape[1], z_last:z_last+box_shape[2]]

        print('Box shape: ', box.shape)

        conv_box = separable3Dconvolve(box, psf, scaling)

        print('Conv Box shape: ', conv_box.shape)

        print('Conv Image size: ', conv_image.shape)

        print('Conv Image slice: ', conv_image[x_last:x_last+conv_box.shape[0], y_last:y_last+conv_box.shape[1], z_last:z_last+conv_box.shape[2]].shape)

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
        print('Last val: ', x_last, y_last, z_last)

    return conv_image

if __name__ == '__main__':

    # psf = np.array([[[0.0, 0.0, 0.0],
    #                 [0.0, 1.0, 0.0],
    #                 [0.0, 0.0, 0.0]],
    #                 [[0.0, 1.0, 0.0],
    #                 [1.0, 2.0, 1.0],
    #                 [0.0, 1.0, 0.0]],
    #                 [[1.0, 2.0, 1.0],
    #                 [2.0, 4.0, 2.0],
    #                 [1.0, 2.0, 1.0]],
    #                 [[0.0, 1.0, 0.0],
    #                 [1.0, 2.0, 1.0],
    #                 [0.0, 1.0, 0.0]],
    #                 [[0.0, 0.0, 0.0],
    #                 [0.0, 1.0, 0.0],
    #                 [0.0, 0.0, 0.0]]])

    psf_raw = tifffile.imread('PSF_BW.tif')
    psf = np.swapaxes(psf_raw, 0, 2)

    # plt.plot(x)
    # plt.plot(y)
    # plt.plot(z)
    # plt.show()

    # # pixelsize of psf
    # xy_pxs, z_pxs = (4, 10)
    #
    # psf = zoom(psf_raw, (xy_pxs, xy_pxs, z_pxs))
    # psf = psf.astype(np.float16)
    #
    # print(np.min(psf), np.max(psf), psf.dtype, '{} GB'.format((psf.size * psf.itemsize)/1average pixel binning python000000000))

    # # Save interpolated PSF to memmap
    # psf = np.memmap('psf.dat', dtype='float16', mode='w+', shape=(psf_raw.shape[0]*xy_pxs, psf_raw.shape[1]*xy_pxs, psf_raw.shape[2]*z_pxs))
    # psf[:] = zoom(psf_raw, (xy_pxs, xy_pxs, z_pxs))[:]
    # psf.flush

    coordinates = (np.loadtxt('FIB_merged_large.txt'))

    print(np.min(coordinates), np.max(coordinates[:,0]), np.max(coordinates[:,1]), np.max(coordinates[:,2]))
    print(coordinates)

    image_shape = ((roundup(int(np.max(coordinates[:,0]))),
                    roundup(int(np.max(coordinates[:,1]))),
                    roundup(int(np.max(coordinates[:,2])))))

    print(image_shape)
    image = np.zeros(image_shape)
    #
    # x_c = coordinates[:,0].astype(int)
    # y_c = coordinates[:,1].astype(int)
    # z_c = coordinates[:,2].astype(int)
    #
    # c = x_c, y_c, z_c
    #
    # print('Seeding fluorophores...')
    # poisson_dist = np.random.poisson(1000, len(x_c))
    # image[c] = poisson_dist[:]

    # conv_image = separable3Dconvolve(image, psf)

    scaling = (20, 50)

    conv_shape = (int(image.shape[0]+20*scaling[0]), int(image.shape[1]+20*scaling[0]-1), int(image.shape[2]+20*scaling[1]-1))
    conv_image = np.memmap('conv_image_out.dat', dtype='float64', mode='r', shape=conv_shape)

    print(np.min(conv_image), np.max(conv_image))

    # conv_image[:] = box3Dconvolve(image, psf, scaling)[:]

    binned_conv_image = xyz_binning(conv_image, 20, 20, 50)

    # conv_image = convolve(image, psf)
    #

    #
    # print('Saving images...')

    save_dir = 'z_stack_box_separable_test_real_psf_approx/'
    make_directory(save_dir)
    print(np.min(binned_conv_image), np.max(binned_conv_image))
    for i in tqdm(range(0, binned_conv_image.shape[2])):
        print(np.min(binned_conv_image[:,:,i]), np.max(binned_conv_image[:,:,i]))
        plt.imshow(binned_conv_image[:,:,i], vmin=0, vmax=10000, cmap='magma')
        plt.savefig(save_dir + str(i) + '.png')
