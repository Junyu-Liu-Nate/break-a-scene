import os
import numpy as np
import cv2

# def convert_npy_to_binary_image(folder_path):
#     # List all files in the given folder
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".npy"):
#             # Load the mask from the .npy file
#             mask = np.load(os.path.join(folder_path, filename))
#             # Convert the mask to a binary image
#             binary_image = np.where(mask > 0.5, 255, 0).astype(np.uint8)
#             # Form the output image filename
#             output_filename = os.path.splitext(filename)[0] + '.png'
#             # Save the binary image as a PNG file
#             cv2.imwrite(os.path.join(folder_path, output_filename), binary_image)
#             print(f"Converted {filename} to {output_filename}")

### For raw image dir from Catlin's dataset
def convert_npy_to_binary_image(folder_path):
    # List all subfolders in the given folder
    part_folders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    for part_folder in part_folders:
        # List all files in the given folder
        for filename in os.listdir(part_folder):
            if filename.endswith(".npy"):
                # Load the mask from the .npy file
                mask_info = np.load(os.path.join(part_folder, filename), allow_pickle=True).item()
                mask = mask_info['segmentation']
                # Convert the mask to a binary image
                binary_image = np.where(mask > 0.5, 255, 0).astype(np.uint8)
                # Form the output image filename
                output_filename = os.path.splitext(filename)[0] + '.png'
                # Save the binary image as a PNG file
                cv2.imwrite(os.path.join(part_folder, output_filename), binary_image)
                print(f"Converted {filename} to {output_filename}")

# Specify the directory containing the .npy masks
folder_path = './chair/image9/'
convert_npy_to_binary_image(folder_path)