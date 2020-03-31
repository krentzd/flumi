#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SIMULATION PIPELINE

# 1) Choose pixel-size of voxel grid and scale input fluorophores accordingly
# 2) Read in segmentations and randomly distribute (choose) points from contours
# 3) Scale input PSF to fit voxel grid by fourier or bicubic interpolation --> Match spatial sampling to be same as voxel grid
# 4) Assign fluorophore photon emission as Poisson distribution with a mean that represents average photon emission (e.g. 1000)
# 5) Convolve voxel grid with PSF
# 6) Simulate camera resolution by binning output of voxel grid (e.g. from 5x5x5nm to 100x100x250nm)
# 7) Noise: a) sensor noise from detection accuracy:  Draw value from Poisson distribution of each pixel using pixel-value as mean
# 8) Noise: b) convert absolute photon count simulating analog-to-digiatl conversion with gain set by user (e.g. 1) --> convert from float to int
# 9) Noise: c) additive electronic noise: add Gaussian noise to each pixel using DC-offset as mean (e.g. 100) and sigma (e.g. 1 or 2)
# 10) Save as stack or tiff

import random
import tifffile
import os
import tempfile
import imageio
import argparse

import numpy as np

from tqdm import tqdm

from tensorly.decomposition import parafac

from scipy.signal import gaussian
from scipy.signal import convolve
from scipy.ndimage import zoom
from scipy.optimize import curve_fit


class Simulator:

    def __init__(self, numerical_aperture=1,
                       wavelength=250,
                       gain=1,
                       dc_offset=200,
                       noise_sigma=50,
                       voxel_pixelsize=(5,5,5),
                       psf_pixelsize=(100, 100, 250)):

        self.numerical_aperture = numerical_aperture
        self.wavelength = wavelength
        self.gain = gain
        self.dc_offset = dc_offset
        self.noise_sigma = noise_sigma
        self.voxel_pxsz = voxel_pixelsize
        self.psf_pxsz = psf_pixelsize
        self.psf = None

    def load_structure(self, structure, mean_intensity=1000):
        # Seed fluorophores in volume and assign intensity by Poisson distribution

        coordinates = np.loadtxt(structure)
        fluorophores = coordinates[:,0].astype(int), coordinates[:,1].astype(int), coordinates[:,2].astype(int)
        volume_shape = ((self._roundup(int(np.max(coordinates[:,0]))),
                         self._roundup(int(np.max(coordinates[:,1]))),
                         self._roundup(int(np.max(coordinates[:,2])))))
        self.volume = np.zeros(volume_shape)
        poisson_dist = np.random.poisson(mean_intensity, len(coordinates))
        self.volume[fluorophores] = poisson_dist[:]

    def load_psf(self, psf):
        # Initialise PSF
        psf_raw = tifffile.imread(psf)
        self.psf = np.swapaxes(psf_raw, 0, 2)

    def run(self, conv_method='box', **params):

        if 'structure' and 'mean_intensity' in params:
            self.load_structure(params['structure'], params['mean_intensity'])
        elif 'structure' in params:
            self.load_structure(params['structure'])

        if 'psf' in params and params['psf'] is not None:
            self.load_psf(params['psf'])


        self.scaling = (self.psf_pxsz[0]/self.voxel_pxsz[0],
                        self.psf_pxsz[1]/self.voxel_pxsz[1],
                        self.psf_pxsz[2]/self.voxel_pxsz[2])

        if self.psf is None:
            self.psf_x, self.psf_y, self.psf_z = self._get_gaussians_from_params()
            self.psf_shape = (len(self.psf_x), len(self.psf_y), len(self.psf_z))

        else:
            self.psf_x, self.psf_y, self.psf_z = self._get_gaussians_from_kernel()
            self.psf_shape = (len(self.psf_x), len(self.psf_y), len(self.psf_z))

        # Output shape from convolution is M+N-1 where M is image size and N kernel size
        self.conv_shape = (self.volume.shape[0] + len(self.psf_x) - 1,
                           self.volume.shape[1] + len(self.psf_y) - 1,
                           self.volume.shape[2] + len(self.psf_z) - 1)

        with tempfile.NamedTemporaryFile() as conv_volume_temp:
            self.conv_volume = np.memmap(conv_volume_temp.name,
                                         dtype='float64',
                                         mode='w+',
                                         shape=self.conv_shape)

        if conv_method is 'box':
            self.box3Dconvolve()
        elif conv_method is 'additive':
            self.additive3Dconvolve()

        self.binned_conv_volume = self.acquire()

    def save(self, path, type='directory'):

        if os.path.splitext(path)[1] is 'tiff' or 'tif':
            type = 'tif'

        if type is 'directory':
            if not os.path.exists(path):
                os.makedirs(path)
            for i in tqdm(range(0, self.binned_conv_volume.shape[2]), desc='Saving images to {}'.format(path)):
                save_path = os.path.join(path, str(i) + '.png')
                print(save_path)
                imageio.imwrite(save_path, self.binned_conv_volume[:,:,i].astype('int16'))

        elif type is 'tif':
            if os.path.splitext(path)[1] is '':
                path = path + '.tif'
            binned_conv_volume_out = np.swapaxes(self.binned_conv_volume, 2, 0).astype('int16')
            tifffile.imwrite(path, binned_conv_volume_out, imagej=True)

    def box3Dconvolve(self):

        box_shape = self._get_box_shape(self.volume.shape, self.psf_shape)
        num_box = (self.volume.shape[0] * self.volume.shape[1] * self.volume.shape[2])/(box_shape[0] * box_shape[1] * box_shape[2])

        assert int(num_box) == num_box, 'ConvError: Box does not fit image!'

        x_last, y_last, z_last = (0, 0, 0)

        for i in tqdm(range(0, int(num_box)), desc='Convolving boxes'):
            box = self.volume[x_last:x_last+box_shape[0], y_last:y_last+box_shape[1], z_last:z_last+box_shape[2]]
            conv_box = self.separable3Dconvolve(box,)
            self.conv_volume[x_last:x_last+conv_box.shape[0], y_last:y_last+conv_box.shape[1], z_last:z_last+conv_box.shape[2]] += conv_box

            x_last += box_shape[0]

            if (x_last == self.volume.shape[0]) and (y_last != self.volume.shape[1]):
                x_last = 0
                y_last += box_shape[1]

            if (y_last == self.volume.shape[1]):
                x_last = 0
                y_last = 0
                z_last += box_shape[2]

    def additive3Dconvolve(self):

        x, y, z = np.nonzero(self.volume)
        box_shape = (1, 1, 1)
        conv_norm_box = separable3Dconvolve(np.ones(box_shape))

        for i in tqdm(range(0, len(x)), desc='Convolving'):
            box = self.volume[x[i]:x[i]+1, y[i]:y[i]+1, z[i]:z[i]+1]
            self.conv_volume[x[i]:x[i]+conv_norm_box.shape[0], y[i]:y[i]+conv_norm_box.shape[1], z[i]:z[i]+conv_norm_box.shape[2]] += conv_norm_box*np.max(box)

    def separable3Dconvolve(self, volume):

        with tempfile.NamedTemporaryFile() as conv_0_temp:
            conv_0_shape = (volume.shape[0] + len(self.psf_x) - 1,
                            volume.shape[1],
                            volume.shape[2])
            conv_0 = np.memmap(conv_0_temp.name,
                                dtype='float64',
                                mode='w+',
                                shape=conv_0_shape)

        with tempfile.NamedTemporaryFile() as conv_1_temp:
            conv_1_shape = (volume.shape[0] + len(self.psf_x) - 1,
                            volume.shape[1] + len(self.psf_y) - 1,
                            volume.shape[2])
            conv_1 = np.memmap(conv_1_temp.name,
                               dtype='float64',
                               mode='w+',
                               shape=conv_1_shape)

        with tempfile.NamedTemporaryFile() as conv_2_temp:
            conv_2_shape = (volume.shape[0] + len(self.psf_x) - 1,
                            volume.shape[1] + len(self.psf_y) - 1,
                            volume.shape[2] + len(self.psf_z) - 1)
            conv_2 = np.memmap(conv_2_temp.name,
                               dtype='float64',
                               mode='w+',
                               shape=conv_2_shape)

        for i in range(0, volume.shape[1]):
            for j in range(0, volume.shape[2]):
                conv_0[:,i,j] = convolve(volume[:,i,j], np.squeeze(self.psf_x))[:]

        for i in range(0, conv_0.shape[0]):
            for j in range(0, conv_0.shape[2]):
                conv_1[i,:,j] = convolve(conv_0[i,:,j], np.squeeze(self.psf_y))[:]

        for i in range(0, conv_1.shape[0]):
            for j in range(0, conv_1.shape[1]):
                conv_2[i,j,:] = convolve(conv_1[i,j,:], np.squeeze(self.psf_z))[:]

        return conv_2

    def acquire(self):

        x_bin_size, y_bin_size, z_bin_size = int(self.scaling[0]), int(self.scaling[1]), int(self.scaling[2])
        binned_image_shape = (int(self.conv_shape[0]/x_bin_size+1), int(self.conv_shape[1]/y_bin_size+1), int(self.conv_shape[2]/z_bin_size+1))
        binned_image = np.zeros(binned_image_shape)
        num_bins = int(self.conv_shape[0]*self.conv_shape[1]*self.conv_shape[2])/(x_bin_size*y_bin_size*z_bin_size)

        x_last, y_last, z_last = (0, 0, 0)

        for i in tqdm(range(0, int(num_bins)), desc='Acquiring'):
            bin = self.conv_volume[x_last:x_last+x_bin_size, y_last:y_last+y_bin_size, z_last:z_last+z_bin_size]
            if bin.size == 0:
                continue
            detected_value = np.random.poisson(np.max(bin))
            digital_value = int(detected_value)*self.gain
            pixel_value = digital_value + np.random.normal(self.dc_offset, self.noise_sigma)
            binned_image[int(x_last/x_bin_size), int(y_last/y_bin_size), int(z_last/z_bin_size)] = pixel_value

            x_last += x_bin_size

            if (x_last >= self.conv_shape[0]) and (y_last <= self.conv_shape[1]):
                x_last = 0
                y_last += y_bin_size

            if (y_last >= self.conv_shape[1]):
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

        return params[0]*np.expand_dims(gaussian(int(params[1]*scaling), std=params[2]*scaling), 1)

    def _get_gaussians_from_kernel(self):

        try:
            x, y, z = parafac(self.psf, 1)
        except ValueError:
            __, (x, y, z) = parafac(self.psf, 1)

        gauss_params = np.abs(self._fit_gaussian(np.array(np.squeeze(x))))
        gauss_params[1] = gauss_params[2]*12
        x = self._get_gaussian(gauss_params, self.scaling[0])

        gauss_params = np.abs(self._fit_gaussian(np.array(np.squeeze(y))))
        gauss_params[1] = gauss_params[2]*12
        y = self._get_gaussian(gauss_params, self.scaling[1])

        gauss_params = np.abs(self._fit_gaussian(np.array(np.squeeze(z))))
        gauss_params[1] = gauss_params[2]*12
        z = self._get_gaussian(gauss_params, self.scaling[2])

        return x, y, z

    def _get_gaussians_from_params(self):

        gauss_params = [1, 2*(0.61*self.wavelength)/self.numerical_aperture, (0.21*self.wavelength)/self.numerical_aperture]
        print(gauss_params)
        x = self._get_gaussian(gauss_params, self.scaling[0])

        gauss_params = [1, 2*(0.61*self.wavelength)/self.numerical_aperture, (0.21*self.wavelength)/self.numerical_aperture]
        y = self._get_gaussian(gauss_params, self.scaling[1])

        gauss_params = [1, 2*(0.61*self.wavelength)/self.numerical_aperture, (0.21*self.wavelength)/self.numerical_aperture]
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

    parser = argparse.ArgumentParser()

    parser.add_argument('--struc', required=True)
    parser.add_argument('--save_as', required=True)
    parser.add_argument('--method', default='box')
    parser.add_argument('--psf')
    parser.add_argument('--NA', default=1)
    parser.add_argument('--wl', default=250)
    parser.add_argument('--gain', default=1)
    parser.add_argument('--dc_off', default=200)
    parser.add_argument('--sigma', default=50)
    parser.add_argument('--voxel_size', default=(5, 5, 5))
    parser.add_argument('--psf_size', default=(100, 100, 250))

    args = parser.parse_args()

    sim = Simulator(numerical_aperture=args.NA,
                    wavelength=args.wl,
                    gain=args.gain,
                    dc_offset=args.dc_off,
                    noise_sigma=args.sigma,
                    voxel_pixelsize=args.voxel_size,
                    psf_pixelsize=args.psf_size)

    sim.run(conv_method=args.method, structure=args.struc, psf=args.psf)
    sim.save(args.save_as)
