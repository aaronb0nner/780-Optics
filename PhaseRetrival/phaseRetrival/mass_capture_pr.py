import subprocess
import cv2
import numpy as np
import tempfile
import os

# Set parameters
exposure_time = 20000  # in microseconds (2 ms)
capture_width = 2000  # Desired width
capture_height = 2000  # Desired height
test_num = 0  # Rotation angle in degrees for polarizer
num_images = 5  # Number of images to capture per set

# Base directory for storing images
base_dir = "/home/raspberry/Documents/programs/phaseRetrival"

# Create a folder for the current test_num
test_folder = os.path.join(base_dir, f"test_{test_num}")
os.makedirs(test_folder, exist_ok=True)

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

    # Convert to black and white (grayscale)
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Save the raw grayscale frame in the rotation folder
    output_file = os.path.join(test_folder, f"image_{i + 1:03d}.jpg")
    cv2.imwrite(output_file, grayscale_frame)
    #print(f"Raw grayscale image saved as {output_file}")

print(f"All {num_images} images captured and saved in {test_folder}.")
