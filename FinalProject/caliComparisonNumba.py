import numpy as np
from PIL import Image
import os
from numba import njit
from tqdm import tqdm
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def load_image_stack(folder, count):
    """Load a stack of images named image_001.jpg, ..., image_010.jpg from a folder."""
    frames = []
    for i in range(1, count + 1):
        filename = os.path.join(folder, f"image_{i:03d}.jpg")
        with Image.open(filename) as img:
            frames.append(np.array(img))
    return np.array(frames)

# Load calibration and moon data from new locations
cal1000 = load_image_stack("FinalProject/Calibration/rotation2_cali45_1ms", 10)  # 1 ms sky
cal2000 = load_image_stack("FinalProject/Calibration/rotation2_cali45_2ms", 10)  # 2 ms sky
offset1ms = load_image_stack("FinalProject/Polarization/rotation2_offset1ms", 10)  # 1 ms, looking away from moon
print(f"cal1000 shape: {cal1000.shape}")
print(f"cal2000 shape: {cal2000.shape}")
print(f"offset1ms shape: {offset1ms.shape}")


# Show the first image from each dataset
show_images = False  # Set to False to skip showing images
if show_images:
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(cal1000[0], cmap='gray')
    axs[0].set_title('cal1ms (First frame)')
    axs[1].imshow(cal2000[0], cmap='gray')
    axs[1].set_title('cal2ms (First frame)')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Use the entire offset image for offset calculation
offset = np.mean(offset1ms)  # Average over all frames and all pixels
print(f"Calculated offset from 1ms offset data: {offset:.3f} digital counts")

def sanuc(data1, data2):
    mu2 = np.mean(data2-offset, axis=0)
    mu1 = np.mean(data1-offset, axis=0)
    sigma2 = np.var(data2, axis=0)
    sigma1 = np.var(data1, axis=0)
    #G = (sigma2 - sigma1) / (mu2 - mu1)
    diff = mu2 - mu1
    G = np.where(diff != 0, (sigma2 - sigma1) / diff, 0)
    # now G is the gain for each pixel so now we have to choose what to reduce it with
    G_median = np.median(G) #units of digital counts / photons
    # print(f"G = {G_median:.3f}")
    K = (mu2 - mu1) / G_median
    K_sum = np.sum(K)
    # print(f"K = {K_sum:.3f}")
    B = mu1 - (G_median * K)
    B_median = np.median(B) # could multiple by size of array since this is a pixel bias
    # print(f"B = {B_median:.3f}")
    sigma_noise = sigma1 - (G_median * G_median) * K
    sigma_noise_median = np.median(sigma_noise)
    sigma_noise_pixel = np.sqrt(sigma_noise_median) / G_median
    # print(f"Sigma Noise = {sigma_noise_pixel:.3f}")

    return G_median, K_sum, B_median

def search_g_vals_numba(ratios, g_vals, model_ratios):
    g_hat_out = np.zeros_like(ratios)
    for i in range(ratios.shape[0]):
        r = ratios[i]
        if r <= 0:
            g_hat_out[i] = 0.0
            continue
        min_err = 1e20
        best = 0
        for j in range(g_vals.shape[0]):
            err = (model_ratios[j] - r) ** 2
            if err < min_err:
                min_err = err
                best = j
        g_hat_out[i] = g_vals[best]
    return g_hat_out

@njit
def _search_g_chunk(r_chunk, g_vals, model_ratios):
    g_chunk = np.zeros_like(r_chunk)
    for i in range(r_chunk.shape[0]):
        r = r_chunk[i]
        if r <= 0:
            g_chunk[i] = 0.0
            continue
        min_err = 1e20
        best = 0
        for j in range(g_vals.shape[0]):
            err = (model_ratios[j] - r) ** 2
            if err < min_err:
                min_err = err
                best = j
        g_chunk[i] = g_vals[best]
    return g_chunk

def nls_nuc(data1, data2, N1, N2, offset):
    data1_corr = data1 - offset
    data2_corr = data2 - offset

    mu1 = np.mean(data1_corr, axis=0)
    mu2 = np.mean(data2_corr, axis=0)

    eps = 1e-12
    mu1_safe = np.where(mu1 <= 0, eps, mu1)
    ratio = mu2 / mu1_safe

    N = N2 / N1
    num_g = 100000
    g_vals = np.linspace(1e-6, 1.0, num_g)
    model_ratios = (1 - np.exp(-N * g_vals)) / (1 - np.exp(-g_vals))


    H, W = ratio.shape
    ratio_flat = ratio.ravel()

    chunk_size = 10000
    g_hat_flat = np.zeros_like(ratio_flat)
    for i in tqdm(range(0, ratio_flat.size, chunk_size), desc="Estimating g_hat"):
        chunk = ratio_flat[i:i+chunk_size]
        g_hat_flat[i:i+chunk_size] = _search_g_chunk(chunk, g_vals, model_ratios)

    g_hat = g_hat_flat.reshape(H, W)

    denom_safe = np.where(np.abs(1.0 - np.exp(-g_hat)) < eps, eps, 1.0 - np.exp(-g_hat))

    C = mu1 / denom_safe

    C_safe = np.where(C <= 0, eps, C)
    data1_norm = data1_corr / C_safe
    data2_norm = data2_corr / C_safe

    data1_clipped = np.clip(1.0 - data1_norm, eps, 1.0)
    data2_clipped = np.clip(1.0 - data2_norm, eps, 1.0)

    H1 = -np.log(data1_clipped)
    H2 = -np.log(data2_clipped)

    h1_mean = np.mean(H1, axis=0)
    h2_mean = np.mean(H2, axis=0)
    h1_var = np.var(H1, axis=0)
    h2_var = np.var(H2, axis=0)

    denominator_safe = np.where(np.abs(h2_mean - h1_mean) < eps, eps, h2_mean - h1_mean)
    alpha = (h2_var - h1_var) / denominator_safe
    alpha_safe = np.where(np.abs(alpha) < eps, eps, alpha)

    K_hat = h1_mean / alpha_safe

    alpha_flat = alpha.ravel()
    valid_alpha = alpha_flat[alpha_flat > 0]

    # print(f"alpha min: {np.min(alpha):.3e}")
    # print(f"alpha max: {np.max(alpha):.3e}")
    # print(f"alpha mean: {np.mean(alpha):.3e}")
    # print(f"alpha > 0 count: {np.sum(alpha > 0)} / {alpha.size}")

    K_sum = np.sum(K_hat)

    return K_sum


print("\nUsing the SANUC algorithm")
G_sanuc, K_sanuc, B_sanuc = sanuc(cal1000, cal2000)
print(f"The gain of the camera is {G_sanuc:.3f} digital counts / photons")
print(f"The K value is {K_sanuc:.3f} photons")
print(f"The bias per pixel is {B_sanuc:.3f} digital counts\n")
print("\nUsing the NLS NUC Algorithm")
#print(f"Calculated offset from 1ms moon data: {offset:.3f} digital counts")
K_nls = nls_nuc(cal1000, cal2000, 1, 2, offset)
#print(f"The K value is {K_nls:.3f} photons\n")

# Physical constants and system parameters
pixel_size_m = 1.55e-6  # 1.55 µm
sensor_pixels = 2000    # 2000 x 2000 grid
sensor_area_m2 = (pixel_size_m * sensor_pixels) ** 2  # (width * height) in m²
wavelength_m = 632.8e-9  # Approximate for sun emission (500 nm)
photon_energy_joules = (6.626e-34 * 3e8) / wavelength_m  # E = hc/λ

# Exposure times in milliseconds
exposure_sanuc_ms = 1.0  # ms for SANUC
exposure_nls_ms = 1.0    # ms for NLS-NUC

# Convert total photon counts to radiant power
# K_sanuc and K_nls are photons per exposure, so convert to photons/second
K_sanuc_per_sec = K_sanuc * (1000.0 / exposure_sanuc_ms)
K_nls_per_sec = K_nls * (1000.0 / exposure_nls_ms)

power_sanuc_watts = K_sanuc_per_sec * photon_energy_joules
power_nls_watts = K_nls_per_sec * photon_energy_joules

# Total radiant power in milliwatts
power_sanuc_mw = power_sanuc_watts * 1e6
power_nls_mw = power_nls_watts * 1e6

print(f"Estimated radiant power (SANUC): {power_sanuc_mw:.6f} mW")
print(f"Estimated radiant power (NLS-NUC): {power_nls_mw:.6f} mW")


