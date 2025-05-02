import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

file_path = "./SpatialCoherence/632_8nm_youngs.jpg"

temp_image = Image.open(file_path).convert('L')  # Convert to grayscale
image = np.array(temp_image)

# plt.imshow(image, cmap='gray')
# plt.title("Original Image")
# plt.axis('off')
# plt.savefig("./SpatialCoherence/original_image.png", dpi=600)

interference = image[15:216, 340:550]
# plt.imshow(interference, cmap='gray')
# plt.title("Interference Pattern Region")
# plt.axis('off')
# plt.show()
# plt.savefig("./SpatialCoherence/interference_zone.png", dpi=600)

# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# # Original image with rectangle
# ax[0].imshow(image, cmap='gray')
# ax[0].add_patch(Rectangle((340, 15), 210, 201, edgecolor='white', facecolor='none', linewidth=1))
# ax[0].set_title("Original Image with Interference Zone")
# ax[0].axis('off')

# # Interference pattern region
# ax[1].imshow(interference, cmap='gray')
# ax[1].set_title("Interference Pattern Region")
# ax[1].axis('off')

# plt.tight_layout()
# plt.savefig("./SpatialCoherence/interference_pattern_region.png", dpi=600)

strip = interference[145, :]
plt.plot(strip, color='black')
plt.title("Intensity Profile at y=145")
plt.xlabel("x")
plt.ylabel("Intensity")
# for x in [45, 49, 53, 57]:
#     plt.axvline(x=x, color='red', linestyle='--', linewidth=0.8)
# plt.show()
plt.savefig("./SpatialCoherence/slice_plot.png", dpi=600)

data_points = strip
normalized_data = (data_points - np.min(data_points)) / (np.max(data_points) - np.min(data_points))
print(data_points.shape)


# # Perform FFT
# fft_result = np.fft.fft(normalized_data)
# fft_magnitude = np.abs(fft_result)

# # Plot the FFT result
# plt.plot(fft_magnitude, color='blue')
# plt.title("FFT of Normalized Intensity Profile (40-65)")f
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()

