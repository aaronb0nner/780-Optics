import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# === CONFIG ===
file_path = "Detection/speckle_imgs.npy"
patch_size = (2, 2)
max_photon_count = 20  # Group photon counts >20 into a single bin for tractability
top_k_display = 20     # How many top patterns to print

# === LOAD DATA ===
data = np.load(file_path)  # shape: (iterations, wavelengths, H, W)

# === COMPUTE JOINT PMF ===
patch_counts = Counter()
for iteration in range(data.shape[0]):
    for wavelength in range(data.shape[1]):
        img = data[iteration, wavelength]
        H, W = img.shape
        for i in range(0, H - patch_size[0] + 1, patch_size[0]):
            for j in range(0, W - patch_size[1] + 1, patch_size[1]):
                patch = img[i:i + patch_size[0], j:j + patch_size[1]]
                binned = np.clip(patch.flatten(), 0, max_photon_count)
                patch_counts[tuple(binned)] += 1

# Normalize to get empirical PMF
total_patches = sum(patch_counts.values())
joint_pmf = {k: v / total_patches for k, v in patch_counts.items()}

# === DISPLAY RESULTS ===
print(f"\nTop {top_k_display} most common 2x2 photon count patch configurations:")
sorted_pmf = sorted(joint_pmf.items(), key=lambda x: x[1], reverse=True)
for idx, (pattern, prob) in enumerate(sorted_pmf[:top_k_display]):
    print(f"{idx+1:2d}: Patch {pattern} → P = {prob:.5f}")

# === OPTIONAL: Plot Histogram of Top N Patterns ===
patterns = ["\n".join(map(str, tup)) for tup, _ in sorted_pmf[:top_k_display]]
probs = [p for _, p in sorted_pmf[:top_k_display]]

plt.figure(figsize=(12, 6))
plt.bar(range(len(probs)), probs)
plt.xticks(range(len(probs)), patterns, rotation=45, ha='right')
plt.ylabel("Probability")
plt.title(f"Top {top_k_display} Most Common 2×2 Photon Count Patch Patterns")
plt.tight_layout()
plt.show()
