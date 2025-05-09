import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize

# ------------------------
# Parameters
# ------------------------
wavelength = 632.8e-9        # HeNe laser wavelength (632.8 nm)
z = 0.09                     # propagation distance (9 cm)
N = 2000                     # pixels
dx = 1.55e-6                 # pixel size (1.55 um)
k = 2 * np.pi / wavelength   # wavenumber

# ------------------------
# Load or simulate detector intensity
# ------------------------
# Replace with your real measurement
# Load the image
image_path = './PhaseRetrival/test_0/image_001.jpg'
intensity_measured = imread(image_path)

# Convert to grayscale if the image is RGB
if intensity_measured.ndim == 3:
    intensity_measured = rgb2gray(intensity_measured)

# Normalize intensity
intensity_measured = intensity_measured / np.sum(intensity_measured)
intensity_measured /= np.sum(intensity_measured)
amplitude_measured = np.sqrt(intensity_measured)

# ------------------------
# Make aperture mask (circular)
# ------------------------
x = np.arange(-N//2, N//2) * dx
X, Y = np.meshgrid(x, x)
R = np.sqrt(X**2 + Y**2)
aperture_radius = 3e-3       # 1 cm diameter => 5 mm radius
aperture_mask = (R < aperture_radius).astype(float)

# ------------------------
# Angular spectrum propagation
# ------------------------
def angular_spectrum(Uin, z, wavelength, dx):
    N = Uin.shape[0]
    fx = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fx)
    H = np.exp(1j * 2 * np.pi * z * np.sqrt(1/wavelength**2 - FX**2 - FY**2))
    H[np.isnan(H)] = 0  # remove evanescent waves (complex sqrt)
    
    Uout = ifft2(fft2(Uin) * fftshift(H))
    return Uout

def angular_spectrum_back(Uin, z, wavelength, dx):
    return angular_spectrum(Uin, -z, wavelength, dx)

# ------------------------
# Gerchberg–Saxton Algorithm
# ------------------------
phase_guess = np.exp(1j * 2 * np.pi * np.random.rand(N, N))
field_aperture = aperture_mask * phase_guess

n_iter = 50
for i in range(n_iter):
    # Forward propagate
    field_detector = angular_spectrum(field_aperture, z, wavelength, dx)
    
    # Enforce detector amplitude
    field_detector = amplitude_measured * np.exp(1j * np.angle(field_detector))
    
    # Back propagate
    field_aperture = angular_spectrum_back(field_detector, z, wavelength, dx)
    
    # Enforce aperture mask
    field_aperture = aperture_mask * np.exp(1j * np.angle(field_aperture))

# ------------------------
# Plot recovered phase
# ------------------------
recovered_phase = np.angle(field_aperture)

plt.imshow(recovered_phase, cmap='twilight', extent=[-N//2*dx*1e6, N//2*dx*1e6, -N//2*dx*1e6, N//2*dx*1e6])
plt.title("Recovered Phase at Aperture Plane")
plt.colorbar(label="Phase (rad)")
plt.xlabel("x (μm)")
plt.ylabel("y (μm)")
plt.show()

# ------------------------
# Phase histogram
# ------------------------
phase_values = recovered_phase[aperture_mask.astype(bool)]
plt.hist(phase_values.flatten(), bins=100, density=True)
plt.title("Phase Histogram Inside Aperture")
plt.xlabel("Phase (rad)")
plt.ylabel("Probability Density")
plt.grid(True)
plt.show()
