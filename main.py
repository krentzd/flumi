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
import tempfile
import imageio
# import napari

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

class Simulator:

    def __init__(self, numerical_aperture=0.5, wavelength=650, gain=1, dc_offset=200, noise_sigma=50):
        self.numerical_aperture = numerical_aperture
        self.wavelength = wavelength
        self.gain = gain
        self.dc_offset = dc_offset
        self.noise_sigma = noise_sigma

    def load_structure(self, structure, mean_intensity=1000, voxel_pixelsize=(5,5,5)):
        # Seed fluorophores in volume and assign intensity by Poisson distribution
        self.voxel_pxsz = voxel_pixelsize
        coordinates = np.loadtxt(structure)
        fluorophores = coordinates[:,0].astype(int), coordinates[:,1].astype(int), coordinates[:,2].astype(int)
        volume_shape = ((self._roundup(int(np.max(coordinates[:,0]))),
                         self._roundup(int(np.max(coordinates[:,1]))),
                         self._roundup(int(np.max(coordinates[:,2])))))
        self.volume = np.zeros(volume_shape)
        poisson_dist = np.random.poisson(mean_intensity, len(coordinates))
        self.volume[fluorophores] = poisson_dist[:]

    def load_psf(self, psf, psf_pixelsize=(100, 100, 250)):
        self.psf_pxsz = psf_pixelsize
        # Initialise PSF
        psf_raw = tifffile.imread(psf)
        self.psf = np.swapaxes(psf_raw, 0, 2)

    def run(self, conv_method='box', **params):

        if 'structure' and 'mean_intensity' and 'voxel_pixelsize' in params:
            self.load_structure(params['structure'], params['mean_intensity'], params['voxel_pixelsize'])
        elif 'structure' and 'mean_intensity' in params:
            self.load_structure(params['structure'], params['mean_intensity'])
        elif 'structure' in params:
            self.load_structure(params['structure'])

        if 'psf' and 'psf_pixelsize' in params:
            self.load_psf(params['psf'], params['psf_pixelsize'])
        elif 'psf' in params:
            self.load_psf(params['psf'])

        self.scaling = (self.psf_pxsz[0]/self.voxel_pxsz[0],
                        self.psf_pxsz[1]/self.voxel_pxsz[1],
                        self.psf_pxsz[2]/self.voxel_pxsz[2])

        self.psf_x, self.psf_y, self.psf_z = self._get_gaussian_kernel(self.psf)

        # Output shape from convolution is M+N-1 where M is image size and N kernel size
        conv_shape = (self.volume.shape[0] + len(self.psf_x) - 1,
                      self.volume.shape[1] + len(self.psf_y) - 1,
                      self.volume.shape[2] + len(self.psf_z) - 1)

        with tempfile.NamedTemporaryFile() as conv_volume_temp:
            self.conv_volume = np.memmap(conv_volume_temp.name,
                                         dtype='float64',
                                         mode='w+',
                                         shape=conv_shape)

        if conv_method is 'box':
            self.conv_volume[:] = self.box3Dconvolve(self.volume, self.psf)
        elif conv_method is 'additive':
            self.additive3Dconvolve()

        self.binned_conv_volume = self.xyz_binning(self.conv_volume, int(self.scaling[0]), int(self.scaling[1]), int(self.scaling[2]))

    def save(self, path, type='directory'):

        # if os.path.splitext(path)[1] is 'tiff' or 'tif':
        #     type = 'tif'
        #
        # if type is 'directory':
        print('Saving...')
        if not os.path.exists(path):
            os.makedirs(path)
        for i in tqdm(range(0, self.binned_conv_volume.shape[2]), desc='Saving images to {}'.format(path)):
            save_path = os.path.join(path, str(i) + '.png')
            print(save_path)
            imageio.imwrite(save_path, self.binned_conv_volume[:,:,i])

        # if type is 'tif':
        #     if os.path.splitext(save_name)[1] is None:
        #         save_name = save_name + '.tif'
        #     binned_conv_volume_out = np.swapaxes(self.binned_conv_volume, 2, 0).astype('float32')
        #     tifffile.imwrite(save_name, binned_conv_volume_out, imagej=True)

    def box3Dconvolve(self, volume, kernel):

        with tempfile.NamedTemporaryFile() as conv_volume_temp:
            # Output shape from convolution is M+N-1 where M is image size and N kernel size
            conv_shape = (volume.shape[0] + len(self.psf_x) - 1,
                          volume.shape[1] + len(self.psf_y) - 1,
                          volume.shape[2] + len(self.psf_z) - 1)
            conv_volume = np.memmap(conv_volume_temp.name,
                                    dtype='float64',
                                    mode='w+',
                                    shape=conv_shape)

        box_shape = self._get_box_shape(volume.shape, kernel.shape)
        num_box = (volume.shape[0] * volume.shape[1] * volume.shape[2])/(box_shape[0] * box_shape[1] * box_shape[2])

        assert int(num_box) == num_box, 'ConvError: Box does not fit image!'

        x_last, y_last, z_last = (0, 0, 0)

        for i in tqdm(range(0, int(num_box)), desc='Convolving boxes'):
            box = volume[x_last:x_last+box_shape[0], y_last:y_last+box_shape[1], z_last:z_last+box_shape[2]]
            conv_box = self.separable3Dconvolve(box, kernel)
            conv_volume[x_last:x_last+conv_box.shape[0], y_last:y_last+conv_box.shape[1], z_last:z_last+conv_box.shape[2]] += conv_box

            x_last += box_shape[0]

            if (x_last == volume.shape[0]) and (y_last != volume.shape[1]):
                x_last = 0
                y_last += box_shape[1]

            if (y_last == volume.shape[1]):
                x_last = 0
                y_last = 0
                z_last += box_shape[2]

        return conv_volume

    def separable3Dconvolve(self, volume, kernel):

        psf_x, psf_y, psf_z = self._get_gaussian_kernel(kernel)

        with tempfile.NamedTemporaryFile() as conv_0_temp:
            conv_0_shape = (volume.shape[0] + len(psf_x) - 1,
                            volume.shape[1],
                            volume.shape[2])
            conv_0 = np.memmap(conv_0_temp.name,
                                dtype='float64',
                                mode='w+',
                                shape=conv_0_shape)

        with tempfile.NamedTemporaryFile() as conv_1_temp:
            conv_1_shape = (volume.shape[0] + len(psf_x) - 1,
                            volume.shape[1] + len(psf_y) - 1,
                            volume.shape[2])
            conv_1 = np.memmap(conv_1_temp.name,
                               dtype='float64',
                               mode='w+',
                               shape=conv_1_shape)

        with tempfile.NamedTemporaryFile() as conv_2_temp:
            conv_2_shape = (volume.shape[0] + len(psf_x) - 1,
                            volume.shape[1] + len(psf_y) - 1,
                            volume.shape[2] + len(psf_z) - 1)
            conv_2 = np.memmap(conv_2_temp.name,
                               dtype='float64',
                               mode='w+',
                               shape=conv_2_shape)

        for i in range(0, volume.shape[1]):
            for j in range(0, volume.shape[2]):
                conv_0[:,i,j] = convolve(volume[:,i,j], np.squeeze(psf_x))[:]

        for i in range(0, conv_0.shape[0]):
            for j in range(0, conv_0.shape[2]):
                conv_1[i,:,j] = convolve(conv_0[i,:,j], np.squeeze(psf_y))[:]

        for i in range(0, conv_1.shape[0]):
            for j in range(0, conv_1.shape[1]):
                conv_2[i,j,:] = convolve(conv_1[i,j,:], np.squeeze(psf_z))[:]

        return conv_2

    def xyz_binning(self, image, x_bin_size, y_bin_size, z_bin_size):

        binned_image = np.zeros((int(image.shape[0]/x_bin_size)+1, int(image.shape[1]/y_bin_size)+1, int(image.shape[2]/z_bin_size)+1))
        num_bins = int(image.shape[0]*image.shape[1]*image.shape[2])/(x_bin_size*y_bin_size*z_bin_size)

        x_last, y_last, z_last = (0, 0, 0)

        print(x_last+x_bin_size, y_last+y_bin_size, z_last+z_bin_size)
        for i in tqdm(range(0, int(num_bins)), desc='Binning'):

            bin = image[x_last:x_last+x_bin_size, y_last:y_last+y_bin_size, z_last:z_last+z_bin_size]
            print(bin.shape)
            if bin.size == 0:
                continue
            # Noise from detector is distributed as Poisson --> how well does sensor pick up value
            detected_value = np.random.poisson(np.max(bin))
            # Set gain and do ADC
            digital_value = int(detected_value)*self.gain
            # Add Gaussian noise
            pixel_value = digital_value + np.random.normal(self.dc_offset, self.noise_sigma)
            print(pixel_value)
            # print('Bin value: ', pixel_value)
            binned_image[int(x_last/x_bin_size), int(y_last/y_bin_size), int(z_last/z_bin_size)] = pixel_value

            x_last += x_bin_size

            if (x_last >= image.shape[0]) and (y_last <= image.shape[1]):
                x_last = 0
                y_last += y_bin_size

            if (y_last >= image.shape[1]):
                x_last = 0
                y_last = 0
                z_last += z_bin_size

        return binned_image

    def _fit_gaussian(self, x):

        def _gaussian(x, a, b, c):
            return a * np.exp(-((x-b)**2)/(2*c**2))

        xdata = np.linspace(0, len(x), len(x))

        params, __ = curve_fit(_gaussian, xdata, x, p0=[1, len(x)/2, 1])

        return params

    def _get_gaussian(self, params, scaling):
        return params[0]*np.expand_dims(gaussian(int(params[2]*12*scaling), std=params[2]*scaling), 1)

    def _get_gaussian_kernel(self, kernel):

        try:
            x, y, z = parafac(kernel, 1)
        except ValueError:
            __, (x, y, z) = parafac(kernel, 1)

        gauss_params = np.abs(self._fit_gaussian(np.array(np.squeeze(x))))
        x = self._get_gaussian(gauss_params, self.scaling[0])

        gauss_params = np.abs(self._fit_gaussian(np.array(np.squeeze(y))))
        y = self._get_gaussian(gauss_params, self.scaling[1])

        gauss_params = np.abs(self._fit_gaussian(np.array(np.squeeze(z))))
        z = self._get_gaussian(gauss_params, self.scaling[2])

        return x, y, z

    def _find_divisors(self, number):

        divisors = []

        for x in range(1, number+1):
            if(number % x == 0):
                divisors.append(x)

        return divisors

    def _get_shape(self, divisors, threshold):

        # Smaller threshold leads to smaller memory requirements but slower simulation
        divisors_temp = [0 if divisor < threshold else divisor for divisor in divisors]

        filtered_divisors = list(filter(lambda a: a != 0, divisors_temp))

        # Safe choice, since all dimensions rounded up to nearest multiple of 10
        if len(filtered_divisors) == 0:
            shape = 10
        else:
            shape = min(filtered_divisors)

        return shape

    def _get_box_shape(self, image_shape, kernel_size):

        divisors_x = self._find_divisors(image_shape[0])
        divisors_y = self._find_divisors(image_shape[1])
        divisors_z = self._find_divisors(image_shape[2])

        shape_x = self._get_shape(divisors_x, kernel_size[0])
        shape_y = self._get_shape(divisors_y, kernel_size[1])
        shape_z = self._get_shape(divisors_z, kernel_size[2])

        return (shape_x, shape_y, shape_z)

    def _roundup(self, x):
        return x if x % 10 == 0 else x + 10 - x % 10

if __name__ == '__main__':
    sim = Simulator()
    sim.run(structure='segmentation.txt', psf='PSF_BW.tif')
    sim.save('Segmentation_first_run')
