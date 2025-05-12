import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from skimage.io import imread
from skimage.color import rgb2gray

# Load data
image_path = './PhaseRetrieval/test_0/image_001.jpg'
intensity_measured = imread(image_path)

# Convert to grayscale if the image is RGB
if intensity_measured.ndim == 3:
    intensity_measured = rgb2gray(intensity_measured)

# Normalize intensity
intensity_measured = intensity_measured / np.sum(intensity_measured)
amplitude_measured = np.sqrt(intensity_measured)
# print("Normalized intensity sum:", np.sum(intensity_measured))

# Parameters
wavelength = 632.8e-9        # HeNe laser wavelength (632.8 nm)
z = 0.09                     # propagation distance (9 cm)
N = 2000                     # pixels
dx_detector = 1.55e-6        # detector pixel size (1.55 um)

# Compute aperture sampling rate based on propagation condition
dx_aperture = (wavelength * z) / (N * dx_detector)
# print(f"Aperture pixel size (dx_aperture): {dx_aperture:.3e} m")


# Make aperture mask
x = np.arange(-N//2, N//2) * dx_aperture
X, Y = np.meshgrid(x, x)
R = np.sqrt(X**2 + Y**2)
aperture_radius = 3e-3       # 6mm diameter aperture (3mm radius)
aperture_mask = (R < aperture_radius).astype(float)

# plt.imshow(aperture_mask, cmap='gray')
# plt.title("Aperture Mask")
# plt.colorbar(label="Aperture")
# plt.xlabel("x (μm)")
# plt.ylabel("y (μm)")
# plt.show()

# Gerchberg–Saxton Algorithm
retrieved_phase = np.random.rand(N, N) * 2 * np.pi
a = ifft2(amplitude_measured * np.exp(1j * retrieved_phase))  # Initial guess

n_iter = 100
for i in range(n_iter):
    # Enforce source amplitude constraint
    b = aperture_mask * np.exp(1j * np.angle(a))
    
    # Forward Fourier Transform
    c = fft2(b)
    
    # Enforce target amplitude constraint
    d = amplitude_measured * np.exp(1j * np.angle(c))
    
    # Inverse Fourier Transform
    a = ifft2(d)

# Extract the retrieved phase
retrieved_phase = np.angle(a)

# Phase histogram
phase_values = retrieved_phase[aperture_mask.astype(bool)]
plt.hist(phase_values.flatten(), bins=200, density=True)
plt.title("Phase Histogram Inside Aperture")
plt.xlabel("Phase (rad)")
plt.ylabel("Probability Density")
plt.grid(True)
plt.show()
