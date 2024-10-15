import os
import numpy as np
import cv2

def convert_npy_to_binary_image(folder_path):
    # List all files in the given folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy"):
            # Load the mask from the .npy file
            mask = np.load(os.path.join(folder_path, filename))
            # Convert the mask to a binary image
            binary_image = np.where(mask > 0.5, 255, 0).astype(np.uint8)
            # Form the output image filename
            output_filename = os.path.splitext(filename)[0] + '.png'
            # Save the binary image as a PNG file
            cv2.imwrite(os.path.join(folder_path, output_filename), binary_image)
            print(f"Converted {filename} to {output_filename}")

# Specify the directory containing the .npy masks
folder_path = './chair126/'
convert_npy_to_binary_image(folder_path)