# Fluorescence Microscopy Simulator(FLUMI

## What is this?

FLUMI is a straightforward fluorescence microscopy simulator that uses point-clouds as ground truth structures and returns a simulated z-stack either as a TIFF stack or as PNG files in a directory. The simulation is done in five steps: 
1. Fluorophores from the point-cloud are seeded onto a voxel grid
2. Experimental PSFs are approximated as a Gaussian to allow separable (and memory-efficient) convolution
3. Fluorophore photon emission is assigned by drawing from a Poisson distribution
4. The voxel grid is convolved with the approximated PSF by compartmentalising the grid into smaller boxes where the convolution is sequentially computed thus returning a simulated sample.
5. The simulated sample is ‘acquired’ by binning the voxel grid to match the detector resolution and noise is added:
  - Sensor noise that results from the detection accuracy which is modelled by drawing the 	detected value for the given  pixel from a Poisson distribution
  - In the analog-to-digial-conversion step a user-defined gain is applied
  - Additive electronic noise is modelled as a Gaussian where the DC-offset and sigma can be  	set by the user
