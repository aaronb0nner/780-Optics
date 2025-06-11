#%%
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.transform import warp_polar, rescale, rotate

# Load TIFF stacks
starry_moon_data = imread('InterpolRegis/Moon_Starry_Night_240AM.tif') #(757, 757, 4)
moon_data1 = imread('InterpolRegis/clipx8.tif')  #(66, 2472, 3296)
moon_data2 = imread('InterpolRegis/clipx8_1.tif') #(39, 2472, 3296)
all_frames = np.concatenate((moon_data1, moon_data2), axis=0)

# Use the first frame of the first stack as reference
reference = all_frames[0]

# Register all frames
registered = []
for i, frame in enumerate(all_frames):
    shift, _, _ = phase_cross_correlation(reference, frame)
    aligned = np.fft.ifftn(fourier_shift(np.fft.fftn(frame), shift)).real
    registered.append(aligned)
registered = np.array(registered)  # shape (n_frames, H, W)

# Average the registered frames
averaged_frame = registered.mean(axis=0)

# Estimate background as median pixel value
background_level = np.median(averaged_frame)

# Subtract background
averaged_frame = averaged_frame - background_level

# Clip any negative values to zero
averaged_frame = np.clip(averaged_frame, 0, None)

# %%

starry_moon_frame = starry_moon_data[:,:,0]

# Target shape from your Moon image
H, W = averaged_frame.shape

# Scaling factor
scale_factor = min(H / 757, W / 757)
new_h = int(757 * scale_factor)
new_w = int(757 * scale_factor)

# Interpolate in Fourier domain
FT = fftshift(fft2(starry_moon_frame))
F_padded = np.zeros((new_h, new_w), dtype=complex)

h_start = new_h // 2 - 757 // 2
w_start = new_w // 2 - 757 // 2
F_padded[h_start:h_start+757, w_start:w_start+757] = FT

s_interp_iso = ifft2(ifftshift(F_padded)).real

# Pad to match Moon image shape
pad_vert = (H - new_h) // 2
pad_horz = (W - new_w) // 2

starry_interp = np.pad(
    s_interp_iso,
    ((pad_vert, H - new_h - pad_vert), (pad_horz, W - new_w - pad_horz)),
    mode='constant'
)
# %%
# Compute 2D Fourier transforms
F_moon = np.fft.fft2(averaged_frame)
F_starry = np.fft.fft2(starry_interp)

# Compute magnitudes
M_moon = np.abs(F_moon)
M_starry = np.abs(F_starry)

# Center of images
center = (averaged_frame.shape[0] // 2, averaged_frame.shape[1] // 2)

# Log-polar transform
M_moon_polar = warp_polar(M_moon, center=center, scaling='log')
M_starry_polar = warp_polar(M_starry, center=center, scaling='log')

# Find shift between the log-polar images
shift_polar, error, _ = phase_cross_correlation(M_moon_polar, M_starry_polar, upsample_factor=10)

# Extract shifts
scale_shift = shift_polar[0]  # shift along log-radius axis (vertical)
rotation_shift = shift_polar[1]  # shift along angle axis (horizontal)

# Calculate rotation angle (degrees)
num_angles = M_moon_polar.shape[1]  # total width = number of angles sampled
rotation_angle = (rotation_shift / num_angles) * 360  # degrees

# Calculate scaling factor
log_base = np.exp(np.log(M_moon.shape[0] / 2) / M_moon_polar.shape[0])
scaling_factor = log_base ** scale_shift

# Rescale by inverse of scaling factor
starry_scaled = rescale(starry_interp, 1/scaling_factor, anti_aliasing=True, preserve_range=True)

# Rotate by negative of the rotation angle
starry_aligned = rotate(starry_scaled, -rotation_angle, resize=False, preserve_range=True)

# Compute translation between aligned Moon frame and Starry Night frame
translation_shift, error, _ = phase_cross_correlation(averaged_frame, starry_aligned, upsample_factor=10)

# Extract the x and y translations
shift_y, shift_x = translation_shift

# Shift using Fourier shift
starry_final = np.fft.ifftn(
    fourier_shift(np.fft.fftn(starry_aligned), shift=(shift_y, shift_x))
).real

# Extract final transformation information
scale_y = 1 / scaling_factor  # vertical scaling applied to Moon image
scale_x = 1 / scaling_factor  # same as vertical in this case (isotropic scaling)

print(f"Estimated Registration Parameters:")
print(f"  - Horizontal Scale (X): {scale_x:.4f}")
print(f"  - Vertical Scale (Y):   {scale_y:.4f}")
print(f"  - Rotation Angle:       {rotation_angle:.2f} degrees")
print(f"  - Translation X:        {shift_x:.2f} pixels")
print(f"  - Translation Y:        {shift_y:.2f} pixels")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(averaged_frame, cmap='gray')
plt.title("Moon Frame Reference")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(starry_final, cmap='gray')
plt.title("Starry Night Reference")
plt.axis('off')

plt.tight_layout()
plt.show()
# %%
