import cv2
import numpy as np
import os

def merge_masks(mask_paths):
    if not mask_paths:
        raise ValueError("The list of mask paths is empty")

    # Read the first mask to get the dimensions
    merged_mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
    if merged_mask is None:
        raise ValueError(f"Failed to read the mask image from {mask_paths[0]}")

    # Initialize the merged mask with zeros (black)
    merged_mask = np.zeros_like(merged_mask)

    for path in mask_paths:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read the mask image from {path}")
        merged_mask = cv2.bitwise_or(merged_mask, mask)

    return merged_mask

def merge_part_masks(img_path):
    part_dirs = [f.path for f in os.scandir(img_path) if f.is_dir()]
    for part_dir in part_dirs:
        sub_part_dirs = [f.path for f in os.scandir(part_dir) if f.is_dir()]
        sub_part_paths = []
        for sub_part_dir in sub_part_dirs:
            all_files = os.listdir(sub_part_dir)
            for file in all_files:
                if file.endswith(".png"):
                    sub_part_paths.append(os.path.join(sub_part_dir, file))
        
        merged_mask = merge_masks(sub_part_paths)
        part_dir_name = os.path.basename(part_dir)
        cv2.imwrite(os.path.join(part_dir, part_dir_name+'.png'), merged_mask)

if __name__ == "__main__":
    img_path = './chair/image9/'
    merge_part_masks(img_path)

