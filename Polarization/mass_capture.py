import subprocess
import cv2
import numpy as np
import tempfile
import os

# Set parameters
exposure_time = 2000  # in microseconds (2 ms)
capture_width = 2000  # Desired width
capture_height = 2000  # Desired height
rotation = 45  # Rotation angle in degrees for polarizer
num_images = 100  # Number of images to capture per set

# Base directory for storing images
base_dir = "/home/raspberry/Documents/programs/polarization"

# Create a folder for the current rotation
rotation_folder = os.path.join(base_dir, f"rotation_{rotation}")
os.makedirs(rotation_folder, exist_ok=True)

# Capture multiple images
for i in range(num_images):
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
        temp_file_name = temp_file.name
        command = [
            "libcamera-still",
            "--shutter", str(exposure_time),
            "--gain", "1.0",
            "--awbgains", "1.0,1.0",
            "--immediate",
            "--nopreview",
            "--width", str(capture_width),
            "--height", str(capture_height),
            "-o", temp_file_name
        ]

        # Suppress logs by redirecting stdout and stderr to /dev/null
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Image {i + 1} captured and temporarily saved as {temp_file_name}")

        # Load the image using OpenCV
        frame = cv2.imread(temp_file_name)

    # Check if image loaded successfully
    if frame is None:
        print(f"Error: Failed to load image {i + 1}.")
        continue

    # Normalize the image
    normalized_frame = cv2.normalize(frame, None, alpha=0, beta=255,
                                     norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Convert to black and white
    normalized_frame = cv2.cvtColor(normalized_frame, cv2.COLOR_RGB2GRAY)

    # Save the corrected frame in the rotation folder
    output_file = os.path.join(rotation_folder, f"image_{i + 1:03d}.jpg")
    cv2.imwrite(output_file, normalized_frame)
    #print(f"Corrected image saved as {output_file}")

print(f"All {num_images} images captured and saved in {rotation_folder}.")
