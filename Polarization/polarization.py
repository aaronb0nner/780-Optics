import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Define the folder names
folders = ["rotation2_0", "rotation2_45", "rotation2_90", "rotation2_noLens", "rotation2_noLensHalfX",]

# Initialize variables for each folder
rotation_0_data = None
rotation_45_data = None
rotation_90_data = None
rotation_noLens_data = None #for SANUC @ 2ms
rotation_noLensHalfX_data = None # for SANUC @ 1ms

# Loop through each folder
for folder in folders:
    folder_path = os.path.join(os.getcwd(), folder)
    if os.path.exists(folder_path):
        images = []
        # Dynamically read all image files in the folder
        for file_name in sorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Open the image and convert it to a numpy array
                with Image.open(file_path) as img:
                    images.append(np.array(img))
        # Assign the numpy arrays to the corresponding variable
        if folder == "rotation2_0":
            rotation_0_data = np.array(images)
        elif folder == "rotation2_45":
            rotation_45_data = np.array(images)
        elif folder == "rotation2_90":
            rotation_90_data = np.array(images)
        elif folder == "rotation2_noLens":
            rotation_noLens_data = np.array(images)
        elif folder == "rotation2_noLensHalfX":
            rotation_noLensHalfX_data = np.array(images)
    else:
        print(f"Folder {folder} does not exist.")

# print(rotation_0_data.shape)
# print(rotation_45_data.shape)
# print(rotation_90_data.shape)


def sanuc(data1, data2):
    sigma2 = np.var(data2, axis=0)
    sigma1 = np.var(data1, axis=0)
    mu2 = np.mean(data2, axis=0)
    mu1 = np.mean(data1, axis=0)
    #G = (sigma2 - sigma1) / (mu2 - mu1)
    # Avoid divide by zero by only dividing where (mu2 - mu1) != 0
    denominator = mu2 - mu1
    G = np.where(denominator != 0, (sigma2 - sigma1) / denominator, 0)
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
G_full, K_full, B_full, sigma_noise_full = sanuc(rotation_noLensHalfX_data, rotation_noLens_data)
print(f"The gain of the camera is {G_full:.3f} digital counts / photons")
print(f"The bias per pixel is {B_full:.3f} digital counts\n")



# Plot the first image in each dataset side by side with max value displayed
if rotation_0_data is not None and rotation_45_data is not None and rotation_90_data is not None:
    plt.figure(figsize=(12, 4))
    
    # Plot the first image from rotation_0_data
    plt.subplot(1, 3, 1)
    plt.imshow(rotation_0_data[0], cmap='gray')
    max_val_0 = np.max(rotation_0_data[0])
    plt.title("Rotation 0°")
    plt.axis('off')
    plt.text(20, 200, f"Max: {max_val_0}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot the first image from rotation_45_data
    plt.subplot(1, 3, 2)
    plt.imshow(rotation_45_data[0], cmap='gray')
    max_val_45 = np.max(rotation_45_data[0])
    plt.title("Rotation 45°")
    plt.axis('off')
    plt.text(20, 200, f"Max: {max_val_45}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot the first image from rotation_90_data
    plt.subplot(1, 3, 3)
    plt.imshow(rotation_90_data[0], cmap='gray')
    max_val_90 = np.max(rotation_90_data[0])
    plt.title("Rotation 90°")
    plt.axis('off')
    plt.text(20, 200, f"Max: {max_val_90}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
else:
    print("One or more datasets are empty. Ensure all folders contain valid images.")



# Ensure data is loaded
if rotation_0_data is not None and rotation_45_data is not None and rotation_90_data is not None:
    # Sum the intensity of each image in the datasets
    rotation_0_sums = [np.sum(image) for image in rotation_0_data]
    rotation_45_sums = [np.sum(image) for image in rotation_45_data]
    rotation_90_sums = [np.sum(image) for image in rotation_90_data]

    print("Mean intensity for Rotation 0°: {:.2e}".format(np.mean(rotation_0_sums)))
    print("Mean intensity for Rotation 45°: {:.2e}".format(np.mean(rotation_45_sums)))
    print("Mean intensity for Rotation 90°: {:.2e}".format(np.mean(rotation_90_sums)))


    # Create a box-and-whisker plot
    plt.figure(figsize=(10, 6))
    plt.boxplot([rotation_0_sums, rotation_45_sums, rotation_90_sums], labels=["Rotation 0°", "Rotation 45°", "Rotation 90°"])
    plt.title("Box-and-Whisker Plot of Total Intensity for Each Dataset")
    plt.ylabel("Total Intensity")
    plt.xlabel("Rotation Angle From Full Power")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
else:
    print("One or more datasets are empty. Ensure all folders contain valid images.")

# Sum the intensity of each image in the datasets (digital counts)
rotation_0_sums = [np.sum(image) for image in rotation_0_data]
rotation_45_sums = [np.sum(image) for image in rotation_45_data]
rotation_90_sums = [np.sum(image) for image in rotation_90_data]

# Convert digital counts to photons using the gain from SANUC
rotation_0_photons = [s / G_full for s in rotation_0_sums]
rotation_45_photons = [s / G_full for s in rotation_45_sums]
rotation_90_photons = [s / G_full for s in rotation_90_sums]

print("Mean photons for Rotation 0°: {:.2e}".format(np.mean(rotation_0_photons)))
print("Mean photons for Rotation 45°: {:.2e}".format(np.mean(rotation_45_photons)))
print("Mean photons for Rotation 90°: {:.2e}".format(np.mean(rotation_90_photons)))

# Create a box-and-whisker plot for photons
plt.figure(figsize=(10, 6))
plt.boxplot([rotation_0_photons, rotation_45_photons, rotation_90_photons], labels=["Rotation 0°", "Rotation 45°", "Rotation 90°"])
plt.title("Box-and-Whisker Plot of Total Photons for Each Dataset")
plt.ylabel("Total Photons")
plt.xlabel("Rotation Angle From Full Power")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Constants
h = 6.62607015e-34  # Planck's constant (J·s)
c = 2.99792458e8    # Speed of light (m/s)
wavelength = 632.8e-9  # HeNe laser wavelength (m)
exposure_time = 0.002  # seconds (2 ms)

# Convert photons to energy per image (Joules)
rotation_0_energy = [(n * h * c) / wavelength for n in rotation_0_photons]
rotation_45_energy = [(n * h * c) / wavelength for n in rotation_45_photons]
rotation_90_energy = [(n * h * c) / wavelength for n in rotation_90_photons]

# Convert energy per image to average power (Watts)
rotation_0_power = [E / exposure_time for E in rotation_0_energy]
rotation_45_power = [E / exposure_time for E in rotation_45_energy]
rotation_90_power = [E / exposure_time for E in rotation_90_energy]

print("Mean power for Rotation 0°: {:.2e} W".format(np.mean(rotation_0_power)))
print("Mean power for Rotation 45°: {:.2e} W".format(np.mean(rotation_45_power)))
print("Mean power for Rotation 90°: {:.2e} W".format(np.mean(rotation_90_power)))

# Create a box-and-whisker plot for power
plt.figure(figsize=(10, 6))
plt.boxplot([rotation_0_power, rotation_45_power, rotation_90_power], labels=["Rotation 0°", "Rotation 45°", "Rotation 90°"])
plt.title("Box-and-Whisker Plot of Total Power for Each Dataset")
plt.ylabel("Total Power (W)")
plt.xlabel("Rotation Angle From Full Power")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()