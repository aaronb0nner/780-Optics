import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Define the folder names
folders = ["rotation_0", "rotation_45", "rotation_90"]

# Initialize variables for each folder
rotation_0_data = None
rotation_45_data = None
rotation_90_data = None

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
        if folder == "rotation_0":
            rotation_0_data = np.array(images)
        elif folder == "rotation_45":
            rotation_45_data = np.array(images)
        elif folder == "rotation_90":
            rotation_90_data = np.array(images)
    else:
        print(f"Folder {folder} does not exist.")

# print(rotation_0_data.shape)
# print(rotation_45_data.shape)
# print(rotation_90_data.shape)

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