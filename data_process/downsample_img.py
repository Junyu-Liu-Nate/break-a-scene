import cv2
import os

def resize_images_in_directory(directory, size=(512, 512)):
    # List all files in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            # Load the image
            image = cv2.imread(file_path)
            # Resize the image
            resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            # Save the resized image, overwriting the original
            cv2.imwrite(file_path, resized_image)
            print(f"Resized and saved: {filename}")

# Specify the directory containing the images to resize
directory = './chair126/'
resize_images_in_directory(directory)