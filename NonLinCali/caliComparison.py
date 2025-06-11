import numpy as np
from PIL import Image
import os

def load_tiff_stack(filename):
    """Load a multi-frame TIFF into a numpy array (frames, height, width)."""
    with Image.open(filename) as img:
        frames = []
        for i in range(img.n_frames):
            img.seek(i)
            frames.append(np.array(img))
    return np.array(frames)

# Load calibration and moon data
cal1000 = load_tiff_stack("NonLinCali/cal1000.tif")  # 1000 ms sky
cal2000 = load_tiff_stack("NonLinCali/cal2000.tif")  # 2000 ms sky
moon1ms = load_tiff_stack("NonLinCali/moon1ms.tif")  # 1 ms, looking away from moon

import matplotlib.pyplot as plt

# Show the first image from each dataset
show_images = False  # Set to False to skip showing images
if show_images:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cal1000[0], cmap='gray')
    axs[0].set_title('cal1000 (first frame)')
    axs[1].imshow(cal2000[0], cmap='gray')
    axs[1].set_title('cal2000 (first frame)')
    axs[2].imshow(moon1ms[0], cmap='gray')
    axs[2].set_title('moon1ms (first frame)')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Use only the region far from the moon for offset calculation
# x: 300:800, y: 1300:1800 (assuming shape is [frames, height, width])
moon_region = moon1ms[:, 1300:1800, 300:800]
offset = np.mean(moon_region, axis=(0, 1, 2))  # Average over all frames and region
print(f"Calculated offset from 1ms moon data: {offset:.3f} digital counts")

def sanuc(data1, data2):
    mu2 = np.mean(data2, axis=0)
    mu1 = np.mean(data1, axis=0)
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

def nls_nuc(data1, data2):
    import numpy as np

def nls_nuc(data1, data2, N1, N2, offset):
    # Subtract offset from both datasets
    data1_corr = data1 - offset
    data2_corr = data2 - offset

    # Compute mean and variance for each pixel
    mu1 = np.mean(data1_corr, axis=0)
    mu2 = np.mean(data2_corr, axis=0)
    
    # Step 3: Estimate g_hat(x,y) using positive exponent model from the paper

    # Compute ratio of means
    eps = 1e-12  # To prevent division by zero
    mu1_safe = np.where(mu1 <= 0, eps, mu1)
    ratio = mu2 / mu1_safe  # shape (H, W)

    # Integration time ratio
    N = N2 / N1  # e.g., 2 if 2000ms / 1000ms

    # Search space for g values
    num_g = 100000  # or 1_000_000 if you want more resolution
    g_vals = np.linspace(1e-6, 1.0, num_g)  # shape (num_g,)

    # Precompute model ratios using positive exponent (as per paper)
    num = 1 - np.exp(N * g_vals)
    den = 1 - np.exp(g_vals)
    model_ratios = num / den  # shape (num_g,)

    # Flatten ratio for pixel-wise search
    H, W = ratio.shape
    ratio_flat = ratio.ravel()
    g_hat_flat = np.zeros_like(ratio_flat)

    # Search per pixel
    for idx, r_val in enumerate(ratio_flat):
        if r_val <= 0:
            g_hat_flat[idx] = 0.0
        else:
            err = (model_ratios - r_val) ** 2
            best_k = np.argmin(err)
            g_hat_flat[idx] = g_vals[best_k]

    # Reshape back to (H, W)
    g_hat = g_hat_flat.reshape(H, W)

    # Step 4: Compute C(x,y)

    # Use positive exponent (paper says 1 - exp(+g), not negative!)
    denom = 1.0 - np.exp(g_hat)

    # Avoid divide-by-zero
    eps = 1e-12
    denom_safe = np.where(np.abs(denom) < eps, eps, denom)

    # Compute C(x,y)
    C = mu1 / denom_safe  # shape (H, W)

    # Step 5: Transform each frame to H-space

    # Ensure C has no zeros (clamp small values)
    eps = 1e-12
    C_safe = np.where(C <= 0, eps, C)

    # Normalize each frame by C(x,y)
    data1_norm = data1_corr / C_safe  # shape (M, H, W)
    data2_norm = data2_corr / C_safe

    # Clamp values to [0, 1 - eps] to avoid log(0) or log(negative)
    data1_clipped = np.clip(1.0 - data1_norm, eps, 1.0)
    data2_clipped = np.clip(1.0 - data2_norm, eps, 1.0)

    # Take negative log
    H1 = -np.log(data1_clipped)  # shape (M, H, W)
    H2 = -np.log(data2_clipped)

    # Step 6: Compute per-pixel mean and variance of H1 and H2

    # Mean over frames (axis=0)
    h1_mean = np.mean(H1, axis=0)  # shape = (H, W)
    h2_mean = np.mean(H2, axis=0)

    # Variance over frames
    h1_var = np.var(H1, axis=0)    # shape = (H, W)
    h2_var = np.var(H2, axis=0)

    # Step 7: Compute alpha(x,y) and K_hat(x,y)

    # Avoid divide-by-zero in denominator
    eps = 1e-12
    denominator = h2_mean - h1_mean
    denominator_safe = np.where(np.abs(denominator) < eps, eps, denominator)

    # Compute alpha
    alpha = (h2_var - h1_var) / denominator_safe  # shape = (H, W)

    # Avoid divide-by-zero again when computing K_hat
    alpha_safe = np.where(np.abs(alpha) < eps, eps, alpha)

    # Compute estimated photon count at N1
    K_hat = h1_mean / alpha_safe  # shape = (H, W)

    # Step 8: Reduce to summary values

    # Filter out invalid/zero/negative alpha values
    alpha_flat = alpha.ravel()
    valid_alpha = alpha_flat[alpha_flat > 0]

    if valid_alpha.size == 0:
        alpha_median = 0.0
    else:
        alpha_median = np.median(valid_alpha)

    # Photon count summaries
    K_sum = np.sum(K_hat)
    K_mean = np.mean(K_hat)

    # print(f"Final median α: {alpha_median:.6f}")
    # print(f"Total photon count (sum): {K_sum:.2f}")
    # print(f"Mean photon count per pixel: {K_mean:.2f}")

    



    return alpha_median, K_sum


print("\nUsing the SANUC algorithm")
G_sanuc, K_sanuc, B_sanuc = sanuc(cal1000, cal2000)
print(f"The gain of the camera is {G_sanuc:.3f} digital counts / photons")
print(f"The K value is {K_sanuc:.3f} photons")
print(f"The bias per pixel is {B_sanuc:.3f} digital counts\n")
print("\nUsing the NLS NUC Algorithm")
alpha_nls, k_nls = nls_nuc(cal1000, cal2000, 1, 2, offset)
print(f"The alpha value is {alpha_nls:.3f} digital counts / photons")
print(f"The K value is {k_nls:.3f} photons\n")

