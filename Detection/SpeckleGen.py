# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:13:29 2024

@author: swordsman
"""

import numpy as np
from PIL import Image, ImageSequence
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from tqdm import tqdm
 

# Function to create a detector blur
def detector_blur(blurx, blury, si):
    import numpy as np
    from scipy.fft import fft2, ifft2, fftshift
    psf = np.zeros((si, si))  # Initialize the point spread function (PSF) array
    for ii in range(0, blurx - 1):
        for jj in range(0, blury - 1):
            ycord = int(np.round(jj + si / 2))
            xcord = int(np.round(ii + si / 2))
            psf[ycord][xcord] = 1  # Set the PSF values
    otf = np.abs(fft2(psf))  # Compute the optical transfer function (OTF) by taking the FFT of the PSF
    return otf

# Function to create a pupil mask
def make_pupil(r1, r2, si):
    import matplotlib.pyplot as plt
    import numpy as np
    if 2 * np.floor(si / 2) == si:  # Check if si is even
        mi = int(np.floor(si / 2))
        pupilx = np.zeros([si, si])
        for i in range(0, si - 1):
            pupilx[i] = range(-mi, mi)
    if 2 * np.floor(si / 2) != si:  # Check if si is odd
        mi = int(np.floor(si / 2))
        pupilx = np.zeros([si, si])
        for i in range(0, si - 1):
            pupilx[i] = range(-mi, mi + 1)
    pupily = np.transpose(pupilx)  # Transpose to get y-coordinates
    dist2 = np.multiply(pupilx, pupilx) + np.multiply(pupily, pupily)  # Calculate squared distance from center
    dist = np.sqrt(dist2)  # Calculate distance from center
    pupil2 = (dist < r1)  # Mask for points within inner radius
    pupil3 = (dist > r2)  # Mask for points outside outer radius
    pupil = np.multiply(pupil2.astype(int), pupil3.astype(int))  # Combine masks to create pupil
    return pupil

# Function to create a complex pupil with focus
def make_cpupil_focus(r1, r2, si, z, lam, dxx, focal):
    import numpy as np
    if 2 * np.floor(si / 2) == si:  # Check if si is even
        mi = int(np.floor(si / 2))
        pupilx = np.zeros([si, si])
        for i in range(0, si - 1):
            pupilx[i] = range(-mi, mi)
    if 2 * np.floor(si / 2) != si:  # Check if si is odd
        mi = int(np.floor(si / 2))
        pupilx = np.zeros([si, si])
        for i in range(0, si - 1):
            pupilx[i] = range(-mi, mi + 1)
    pupily = np.transpose(pupilx)  # Transpose to get y-coordinates
    dist2 = np.multiply(pupilx, pupilx) + np.multiply(pupily, pupily)  # Calculate squared distance from center
    dist = np.sqrt(dist2)  # Calculate distance from center
    pupil2 = (dist < r1)  # Mask for points within inner radius
    pupil3 = (dist > r2)  # Mask for points outside outer radius
    pupil = np.multiply(pupil2.astype(int), pupil3.astype(int))  # Combine masks to create pupil
    lens_phase = dxx * dxx * np.pi * dist2 / (lam * focal)  # Calculate lens phase
    phase1 = 2 * np.pi * np.sqrt(dxx * dxx * dist2 + z * z) / lam  # Calculate phase term 1
    phase2 = 2 * np.pi * np.sqrt(dxx * dxx * dist2 + (focal + .0001) * (focal + .0001)) / lam  # Calculate phase term 2
    cpupil = np.multiply(np.exp(1j * (phase1 + phase2 - lens_phase)), pupil)  # Create complex pupil
    return cpupil

# Function to create an optical transfer function (OTF)
def make_otf2(scale, cpupil):
    from scipy.fft import fft2
    import numpy as np
    psf = fft2(cpupil)  # Compute the point spread function (PSF) by taking the FFT of the complex pupil
    psf = abs(psf)
    psf = np.multiply(psf, psf)  # Square the PSF
    spsf = np.sum(psf)  # Sum of PSF values
    norm_psf = scale * psf / spsf  # Normalize the PSF
    otf = fft2(norm_psf)  # Compute the optical transfer function (OTF) by taking the FFT of the normalized PSF
    return otf

# Function to create a Gaussian beam
def gaussian_beam(amp, xc, yc, wx, wy, si, dx):
    # amp is the beam amplitude
    # xc is the horizontal center of the beam in units of meters
    # yc is the vertical center of the beam in units of meters 
    # wx is the horizontal beam waist
    # wy is the vertical beam waist
    # si is half the width of the beam array
    # dx is the sample spacing in the beam array
    from scipy.fft import fft2
    import numpy as np
    beam = np.zeros((2 * si, 2 * si))  # Initialize the beam array
    for x in range(0, 2 * si):
        xx = dx * (x - si)
        for y in range(0, 2 * si):
            yy = dx * (y - si)
            beam[x, y] = amp * np.exp(-(xx - xc) * (xx - xc) / wx - (yy - yc) * (yy - yc) / wy) / (np.pi * wx * wy)  # Calculate Gaussian beam values
    return beam

si = 2000  # Size of the image
fl = .1  # Focal length of the lens
D = .01  # Diameter of the lens
lam_min = 400e-9  # Minimum wavelength
lam_max = 700e-9  # Maximum wavelength
infield = gaussian_beam(1, 0, 0, .002, .002, 1000, 1e-5)


interations = 20  # Number of iterations to run the simulation
speckles_per_iteration = 50  # Number of speckles to generate per iteration, starting from 400nm
speckle_size = 100  # Size of the speckle images in pixels
speckle_imgs = np.zeros((interations, speckles_per_iteration, speckle_size, speckle_size))  # Initialize the array to store speckle images. 2D array with dimensions of interations and speckles per iteration
increment_scale = 1  # Inverse Scale factor to increment the wavelength by, 1 is 1nm, 4 is 0.25nm
mean_photons_per_pixel = 100

for i in range(0, interations):
    # Generate a new random phase screen for each iteration
    glass = 0.4 * np.random.uniform(0, .001, (si, si))  # Generate a random phase screen from a uniform distribution. 0.4 represent the index of refraction of the glass and the air
    print(f"Iteration {i + 1}/{interations}")
    for lam in tqdm(range(0, speckles_per_iteration), desc=f"Generating speckles for iteration {i + 1}"):
        lamda = lam/increment_scale * 1e-9 + lam_min  # Calculate the wavelength in meters
        r1 = 500 * lam_min / lamda
        pupil = make_pupil(r1, 0, si)

        phase = 2 * np.pi * glass / lamda
        cpupil = np.multiply(pupil, np.exp((0 + 1j) * phase))
        speckle_field = fftshift(fft2(fftshift(cpupil)))
        speckle_intensity = np.multiply(np.abs(speckle_field), np.abs(speckle_field))

        # Rescale to expected photon rate
        photon_rate = speckle_intensity / speckle_intensity.max() * mean_photons_per_pixel
        # Apply Poisson shot noise
        noisy_photon_image = np.random.poisson(photon_rate)
        # Extract the center speckle_size x speckle_size region
        center_x = speckle_intensity.shape[0] // 2
        center_y = speckle_intensity.shape[1] // 2
        half_size = speckle_size // 2
        speckle_imgs[i, lam] = noisy_photon_image[
            center_x - half_size:center_x + half_size,
            center_y - half_size:center_y + half_size
        ]


# Save the speckle images array as a .npy file after generation
np.save('Detection/speckle_imgs.npy', speckle_imgs)

#print the storage size of the speckle images array in a nicely formatted way
print(f"Speckle Images Array Size: {speckle_imgs.nbytes / 1e6} MB")

print("Done")

