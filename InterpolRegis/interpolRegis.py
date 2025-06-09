import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
from concurrent.futures import ThreadPoolExecutor
from skimage.transform import rescale, rotate, warp_polar
from numpy.fft import fftshift, fft2, ifft2, ifftshift

def load_tiff_stack(filename):
    """Load a multi-frame TIFF into a numpy array (frames, height, width)."""
    with Image.open(filename) as img:
        frames = []
        for i in range(img.n_frames):
            img.seek(i)
            frames.append(np.array(img))
    return np.array(frames)

# Load calibration and moon data
clipx8_1 = load_tiff_stack("InterpolRegis/clipx8_1.tif")  #Captured Data at same time
clipx8 = load_tiff_stack("InterpolRegis/clipx8.tif")  #Captured Data at same time but more frames
moon = np.concatenate((clipx8, clipx8_1), axis=0) # Concatenate the two stacks along the first axis
del clipx8
del clipx8_1
starry_night = load_tiff_stack("InterpolRegis/Moon_Starry_Night_240AM.tif")  # Starry Night at 2:40 AM

print(f"moon shape: {moon.shape}")
print(f"starry night shape: {starry_night.shape}")

# Optional: Show first images for verification
show_images = False  # Set to True to show images
if show_images:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(moon[0], cmap='gray')
    axs[0].set_title('moon (First frame)')
    axs[1].imshow(starry_night[0], cmap='gray')
    axs[1].set_title('Moon_Starry_Night_240AM (First frame)')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def align_single_frame(args):
    ref, frame = args
    im = frame.astype(np.float32)
    shift, error, diffphase = phase_cross_correlation(ref, im)
    im_fft = np.fft.fftn(im)
    shifted_fft = fourier_shift(im_fft, shift)
    aligned_frame = np.fft.ifftn(shifted_fft).real
    return aligned_frame

def align_frames(frames):
    """Align all frames in a stack to the first frame using phase cross-correlation (Fourier shift) with multithreading."""
    aligned = np.zeros_like(frames)
    ref = frames[0].astype(np.float32)
    aligned[0] = frames[0]

    args = [(ref, frames[i]) for i in range(1, frames.shape[0])]

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(align_single_frame, args), total=len(args), desc="Aligning moon frames to fit first frame"))

    for i, aligned_frame in enumerate(results, start=1):
        aligned[i] = aligned_frame

    return aligned

moon_aligned = align_frames(moon)

# Condense moon data to a single frame by averaging now that its aligned
moon_averaged = np.mean(moon_aligned, axis=0)
moon_averaged_median = np.median(moon_averaged)
moon_averaged = moon_averaged - moon_averaged_median
moon_averaged = np.clip(moon_averaged, 0, None)  # Zero out negative values
print("Moon data aligned and condensed to a single frame with backround eliminated.")
# plt.figure(figsize=(8, 8))
# plt.imshow(moon_averaged, cmap='gray')
# plt.title('Averaged Moon Frame')
# plt.axis('off')
# plt.show()

# Extract the first frame and first channel from the starry night stack
starry_frame = starry_night[0, :, :, 0]

# Get target dimensions from the Moon image
target_height, target_width = moon_averaged.shape

# Compute isotropic scaling factor to fit 757x757 into Moon frame
base_size = 757
scale = min(target_height / base_size, target_width / base_size)
interp_height = int(base_size * scale)
interp_width = int(base_size * scale)
print(f"Interpolated dimensions: {interp_height} x {interp_width}")

# Fourier domain interpolation: zero-pad/crop in frequency domain
starry_fft = fftshift(fft2(starry_frame))
fft_padded = np.zeros((interp_height, interp_width), dtype=complex)
h0 = interp_height // 2 - base_size // 2
w0 = interp_width // 2 - base_size // 2
fft_padded[h0:h0+base_size, w0:w0+base_size] = starry_fft

starry_interp_freq = ifft2(ifftshift(fft_padded)).real

plt.figure(figsize=(6, 6))
plt.imshow(starry_interp_freq, cmap='gray')
plt.title('Fourier-Interpolated Starry Night')
plt.show()

# Pad interpolated image to match Moon frame size
pad_y = (target_height - interp_height) // 2
pad_x = (target_width - interp_width) // 2
print(f"Padding applied: vertical={pad_y}, horizontal={pad_x}")

starry_padded = np.pad(
    starry_interp_freq,
    ((pad_y, target_height - interp_height - pad_y), (pad_x, target_width - interp_width - pad_x)),
    mode='constant'
)

# Compute 2D FFTs for both images
moon_fft = np.fft.fft2(moon_averaged)
starry_fft = np.fft.fft2(starry_padded)

# Magnitude spectra
moon_mag = np.abs(moon_fft)
starry_mag = np.abs(starry_fft)

# Log-polar transform for scale/rotation registration
center_coords = (target_height // 2, target_width // 2)
moon_polar = warp_polar(moon_mag, center=center_coords, scaling='log')
starry_polar = warp_polar(starry_mag, center=center_coords, scaling='log')

# Register log-polar images to estimate scale and rotation
polar_shift, polar_error, _ = phase_cross_correlation(moon_polar, starry_polar, upsample_factor=10)
scale_log_shift, rot_shift = polar_shift

# Calculate rotation in degrees
angle_bins = moon_polar.shape[1]
rotation_deg = (rot_shift / angle_bins) * 360

# Calculate scaling factor
log_base = np.exp(np.log(target_height / 2) / moon_polar.shape[0])
scale_factor = log_base ** scale_log_shift

# Rescale and rotate the Starry Night image
starry_scaled = rescale(starry_padded, 1/scale_factor, anti_aliasing=True, preserve_range=True)
starry_rotated = rotate(starry_scaled, -rotation_deg, resize=False, preserve_range=True)

# Register translation between Moon and aligned Starry Night
trans_shift, trans_error, _ = phase_cross_correlation(moon_averaged, starry_rotated, upsample_factor=10)
dy, dx = trans_shift

# Apply translation using Fourier shift
starry_registered = np.fft.ifftn(
    fourier_shift(np.fft.fftn(starry_rotated), shift=(dy, dx))
).real

# Output registration parameters
print("Registration Results:")
print(f"  - X Scale: {1/scale_factor:.4f}")
print(f"  - Y Scale: {1/scale_factor:.4f}")
print(f"  - Rotation: {rotation_deg:.2f} deg")
print(f"  - X Translation: {dx:.2f} px")
print(f"  - Y Translation: {dy:.2f} px")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(moon_averaged, cmap='gray')
plt.title("Moon Reference")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(starry_registered, cmap='gray')
plt.title("Registered Starry Night")
plt.axis('off')
plt.tight_layout()
plt.show()
