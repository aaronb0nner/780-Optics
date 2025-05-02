import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

# List of .tiff file paths
tiff_files = [
    "./TemporalCoherence/class_data/Seperate_spots_0.5ms_Gain0_offset_54.tif",
    "./TemporalCoherence/class_data/Seperate_spots_1ms_Gain0_offset_54.tif",
    "./TemporalCoherence/class_data/combined_better.tif"
]

# Function to decompose a .tif file into individual frames
def decompose_tiff(file_path):
    frames = []
    with Image.open(file_path) as img:
        for i in range(img.n_frames):  # Iterate through all frames
            img.seek(i)  # Move to the i-th frame
            frames.append(np.array(img))  # Convert frame to numpy array
    return np.array(frames)  # Return as a 3D numpy array

# Decompose each .tif file into its own array
seperate_short = decompose_tiff(tiff_files[0])
seperate_long = decompose_tiff(tiff_files[1])
combined = decompose_tiff(tiff_files[2])

# Print the shape of each array
#print(f"seperate_short shape: {seperate_short.shape}")
#print(f"seperate_long shape: {seperate_long.shape}")
#print(f"combined shape: {combined.shape}")

def sanuc(data1, data2):
    sigma2 = np.var(data2, axis=0)
    sigma1 = np.var(data1, axis=0)
    #these are 1200x1200 arrays
    mu2 = np.mean(data2, axis=0)
    mu1 = np.mean(data1, axis=0)
    #also 1200x1200 arrays
    #G = (sigma2 - sigma1) / (mu2 - mu1)
    G = np.divide((sigma2-sigma1),(mu2-mu1))# fix divided by zero issues with a filter
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

    return G_median, K_sum, B_median, sigma_noise_pixel

print("\nUsing SANUC on the 0.5ms and 1ms exposures to establish camera parameters")
G_full, K_full, B_full, sigma_noise_full = sanuc(seperate_short[:, 850:1050, 400:600], seperate_long[:, 850:1050, 400:600])
print(f"The gain of the camera is {G_full:.3f} digital counts / photons")
print(f"The bias per pixel is {B_full:.3f} digital counts\n")


# Create the "left" and "right" arrays based on the long exposure
left_pattern = seperate_long[:, 100:800, 0:500]  # y: 100-800, x: 0-500
right_pattern = seperate_long[:, 200:800, 700:1200]  # y: 200-800, x: 700-1200
background_pattern = seperate_long[:, 850:1050, 400:600]  # y: 850-1050, x: 400-600

# Calculate the average digital counts in the left and right patterns
left_avg = np.mean(left_pattern)
right_avg = np.mean(right_pattern)
background_pattern_avg = np.mean(background_pattern)

print(f"Average digital counts in left pattern (cropped from full image): {left_avg:.3f}")
print(f"Average digital counts in right pattern (cropped from full image): {right_avg:.3f}")
print(f"Average digital counts in background pattern (cropped from full image): {background_pattern_avg:.3f}")

# Plot the first frame of the seperate_long array with pattern boxes
plt.figure(figsize=(8, 8))
plt.imshow(seperate_long[0], cmap='gray')
plt.colorbar(label='Digital Counts')
plt.title('First frame of seperated data (1ms exposure)')
plt.xlabel('X Pixels')
plt.ylabel('Y Pixels')

# Add rectangles for the left, right, and background patterns
ax = plt.gca()
# Left pattern box
left_rect = Rectangle((0, 100), 500, 700, linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(left_rect)
# Right pattern box
right_rect = Rectangle((700, 200), 500, 600, linewidth=2, edgecolor='blue', facecolor='none')
ax.add_patch(right_rect)
# Background pattern box
background_rect = Rectangle((400, 850), 200, 200, linewidth=2, edgecolor='green', facecolor='none')
ax.add_patch(background_rect)
#plt.show()

combined_pattern = combined[:, 225:830, 575:1150]  # y: 225-830, x: 575-1150

combined_pattern_avg = np.mean(combined_pattern)
print(f"Average digital counts in combined pattern (cropped from full image): {combined_pattern_avg:.3f}")

# Plot the first frame of the combined array with the cutoff zone
plt.figure(figsize=(8, 8))
plt.imshow(combined[0], cmap='gray')
plt.colorbar(label='Digital Counts')
plt.title('First frame of combined data (1ms exposure)')
plt.xlabel('X Pixels')
plt.ylabel('Y Pixels')

# Add rectangle for the combined pattern box
ax = plt.gca()
combined_rect = Rectangle((575, 225), 575, 605, linewidth=2, edgecolor='purple', facecolor='none')
ax.add_patch(combined_rect)
#plt.show()

left_adjusted = left_avg - background_pattern_avg
right_adjusted = right_avg - background_pattern_avg
combined_adjusted = combined_pattern_avg - background_pattern_avg
print(f"\nAverage digital counts in left pattern (adjusted): {left_adjusted:.3f}")
print(f"Average digital counts in right pattern (adjusted): {right_adjusted:.3f}")
print(f"Average digital counts in combined pattern (adjusted): {combined_adjusted:.3f}")

left_total = left_adjusted * left_pattern.shape[1] * left_pattern.shape[2]
right_total = right_adjusted * right_pattern.shape[1] * right_pattern.shape[2]
combined_total = combined_adjusted * combined_pattern.shape[1] * combined_pattern.shape[2]
print(f"\nTotal digital counts in left pattern: {left_total:.3f}")
print(f"Total digital counts in right pattern: {right_total:.3f}")
print(f"Total digital counts in combined pattern: {combined_total:.3f}")

left_photons = left_total / G_full
right_photons = right_total / G_full
combined_photons = combined_total / G_full
print(f"\nTotal photons in left pattern: {left_photons:.4e}")
print(f"Total photons in right pattern: {right_photons:.4e}")
print(f"Total photons in combined pattern: {combined_photons:.4e}")


#NOTE  Just need to get into units of photons at the end
#NOTE do sanuc at the beginning to get the gain and bias with green box area