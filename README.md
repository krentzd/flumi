# Fluorescence Microscopy Simulator (FLUMI)

## What is this?

FLUMI is a straightforward fluorescence microscopy simulator that uses point-clouds as ground truth structures and returns a simulated z-stack either as a TIFF stack or as PNG files in a directory. The simulation is done in five steps: 
1. Fluorophores from the point-cloud are seeded onto a voxel grid
2. Experimental PSFs are approximated as a Gaussian to allow separable (and memory-efficient) convolution
3. Fluorophore photon emission is assigned by drawing from a Poisson distribution
4. The voxel grid is convolved with the approximated PSF by compartmentalising the grid into smaller boxes where the convolution is sequentially computed thus returning a simulated sample.
5. The simulated sample is ‘acquired’ by binning the voxel grid to match the detector resolution and noise is added:
    - The sensor detection accuracy is modelled as a Poisson distribution 
    - In the analog-to-digial-conversion step a user-defined gain is applied
    - Additive electronic noise is modelled as a Gaussian where the DC-offset and sigma can be set by the user

![Simulated stack:](https://github.com/krentzd/project-superball/blob/master/colour_stack.gif)

## How do I use it?

FLUMI can be executed from the command line allowing the user to specify the following arguments:
- input structure (string): `--struc` (required)
- target location (string): `--save_as` (required)
- convolution method (either 'box' or 'additive'): `--method`
- experimental PSF in TIFF format (string): `--psf`
- numerical aperture (float): `--NA`
- wavelength (int): `--wl`
- gain (int): `--gain`
- DC-offset (int): `--dc_off`
- sigma noise (float): `--sigma`
- voxel grid size which is the pixelsize of the input structure (int tuple of shape (x, y, z)): `--voxel_size`
- PSF size which is also the pixelsize of the simulated sensor (int tuple of shape (x, y, z)): `--psf_size`

A simulation using a point-cloud of fluorophores and an experimental PSF that is saved to a directory would look like this:
```
python main.py --struc 'segmentation.txt' --psf 'PSF_BW.tif' --save_as 'Simulation'
```

## Installation 
FLUMI runs on Python 3.7.4 and backward compatibility cannot be guaranteed.
### Install Dependencies  
```
pip install tifffile
```
```
pip install imageio
```
```
pip install argparse
```
```
pip install tqdm
```
```
pip install numpy
```
```
pip install tensorly
```
```
pip install scipy
```

Otherwise create a new conda environment:
```
conda env create -f flumi.yml
```

### Clone Repository
```
git clone https://github.com/krentzd/flumi.git
```

### Run FLUMI
Navigate to the folder called `flumi` and run:
```
python main.py --struc 'segmentation.txt' --psf 'PSF_BW.tif' --save_as 'Simulation'
```

