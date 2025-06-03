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
    """
    Non‐Linear Statistical Non‐Uniformity Correction (NLSNUC), implemented per‐pixel.

    Inputs:
      - data1: numpy array of shape (M, H, W) taken at integration time N1 (e.g. 1000 ms).
      - data2: numpy array of shape (M, H, W) taken at integration time N2 (e.g. 2000 ms).
      - N1, N2: the two integration‐time factors (scalars).
      - offset: scalar O (digital‐count offset), precomputed as the mean over a dark region.

    Returns:
      - alpha_median: median of alpha(x,y) over all pixels.
      - K_sum: sum of K̂(x,y) over all pixels (the total estimated electrons at N1).
    """
    # 1) Compute per‐pixel means over frames:
    #    Each of shape (M, H, W) → mean over axis=0 → shape (H, W)
    mu1 = np.mean(data1, axis=0)  # E[D₁(x,y)]
    mu2 = np.mean(data2, axis=0)  # E[D₂(x,y)]

    # 2) Subtract offset before forming ratios.
    #    We assume (mu1 - offset) > 0 and (mu2 - offset) > 0 for illuminated pixels.
    s1 = mu1 - offset
    s2 = mu2 - offset

    # Avoid division by zero / negative values in dark pixels:
    #   Wherever s1 <= 0, force a tiny epsilon so we don’t blow up.
    eps = 1e-12
    s1 = np.where(s1 <= 0, eps, s1)
    s2 = np.where(s2 <  0, 0.0, s2)

    # 3) Pixel‐wise ratio r(x,y) = (mu2 – O) / (mu1 – O)
    ratio = s2 / s1  # shape (H, W)

    H, W = mu1.shape
    N = float(N2) / float(N1)

    # 4) Precompute a large 1D array of candidate g‐values from 1e-6 to 1.0 (step = 1e-6).
    #    (You can adjust the range/resolution if this is too slow/large.)
    num_g = int(1e6)  # one million steps
    g_vals = np.linspace(1e-6, 1.0, num_g)  # shape (1e6,)

    # In “ideal NLSNUC” we solve: (1 - exp(-N·g)) / (1 - exp(-g)) = r.  So build that once:
    #    model_ratios[i] = (1 - e^{ - (N * g_vals[i]) }) / (1 - e^{ - g_vals[i] })
    exp_neg_g = np.exp(-g_vals)               # shape (num_g,)
    exp_neg_Ng = np.exp(-N * g_vals)          # shape (num_g,)
    model_ratios = (1.0 - exp_neg_Ng) / (1.0 - exp_neg_g)  # shape (num_g,)

    # 5) For each pixel (i,j), find index k that minimises [ratio[i,j] - model_ratios[k]]^2.
    #    g_hat(i,j) = g_vals[argmin_k].
    g_hat = np.zeros((H, W), dtype=np.float64)

    # Flatten for iteration, then reshape back.
    ratio_flat = ratio.ravel()   # shape (H*W,)
    g_hat_flat = np.zeros_like(ratio_flat)

    for idx, r_val in enumerate(ratio_flat):
        # If r_val is zero (or negative), we can clamp g_hat=0 (no signal).  Otherwise, search:
        if r_val <= 0:
            g_hat_flat[idx] = 0.0
        else:
            # compute squared error against all model_ratios
            err = (model_ratios - r_val) ** 2
            best_k = np.argmin(err)
            g_hat_flat[idx] = g_vals[best_k]
        # end if
    # end for

    g_hat = g_hat_flat.reshape(H, W)  # Now shape (H, W)

    # 6) Compute per‐pixel saturation constant:
    #     C(x,y) = (mu1(x,y) - offset) / [1 - exp(-g_hat(x,y))]
    denom = 1.0 - np.exp(-g_hat)
    # Avoid zero‐division (if g_hat was 0, set denom→eps)
    denom = np.where(denom <= 0, eps, denom)
    C = s1 / denom   # shape (H, W)

    # 7) Transform each frame of data1 & data2 into “H‐space”:
    #    H1(m, x, y) = -ln[ 1 - (D1(m, x, y) - O) / C(x,y) ]  (for m=0..M-1)
    #    H2(m, x, y) = -ln[ 1 - (D2(m, x, y) - O) / C(x,y) ]

    # To broadcast C across the first (frame) dimension, we do: (data1 - O) / C  → shape (M, H, W)
    #   but ensure no out‐of‐range (clamp inside [0,1-eps]).
    D1_off = data1 - offset
    D2_off = data2 - offset

    # Clip (D - O)/C to [0, 1 - eps] so that log(1 - x) never sees x ≥ 1.
    ratio1 = D1_off[..., None] / C  # temporarily shape (M, H, W) if broadcasting is done carefully
    ratio2 = D2_off[..., None] / C

    # Actually, easiest is to subtract offset and then divide by C per‐pixel:
    #    (data1 - offset) is (M,H,W), C is (H,W) → the “/ C” line broadcasts automatically
    H1 = -np.log( np.clip(1.0 - (data1 - offset) / C, 0.0, 1.0 - eps) )
    H2 = -np.log( np.clip(1.0 - (data2 - offset) / C, 0.0, 1.0 - eps) )
    # Both H1, H2 are shape (M, H, W).

    # 8) Compute per‐pixel mean and variance across frames m=0..M-1:
    h1_mean = np.mean(H1, axis=0)   # shape (H, W)
    h2_mean = np.mean(H2, axis=0)

    h1_var  = np.var(H1, axis=0)    # shape (H, W)
    h2_var  = np.var(H2, axis=0)

    # 9) Solve for alpha(x,y):
    #    α(x,y) = [h2_var(x,y) - h1_var(x,y)] / [h2_mean(x,y) - h1_mean(x,y)]
    num = (h2_var - h1_var)
    den = (h2_mean - h1_mean)
    # Avoid division by zero:
    alpha = np.where(np.abs(den) <= eps, 0.0, num / den)

    # 10) Pixel‐wise photon‐count at N1: K_hat(x,y) = h1_mean(x,y) / α(x,y)
    K_hat = np.where(np.abs(alpha) <= eps, 0.0, h1_mean / alpha)

    # 11) Summarize: return median α and total sum of K̂
    alpha_flat = alpha.ravel()
    # Only take the positive α values (ignore any zero/negative outliers in “dark” pixels)
    alpha_pos = alpha_flat[ alpha_flat > 0 ]
    if alpha_pos.size == 0:
        alpha_median = 0.0
    else:
        alpha_median = np.median(alpha_pos)

    K_sum = np.sum(K_hat)

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

