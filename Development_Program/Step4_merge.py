"""
Step4_merge.py
--------------
This script processes and merges images from a 360-degree surround view system.
It performs:
1. Loading and validating images from different camera angles (front, left, rear, right).
2. Merging images using weighted masks and alpha blending.
3. Adjusting luminance to balance lighting across images.
4. Overlaying a car image using perspective transformation.
5. Displaying the final comparison of merged images.

Dependencies:
- OpenCV (cv2)
- NumPy
- PIL (Pillow)

Usage:
Run the script to process images and display results.

Author: [Your Name]
Date: [Date]
"""

import cv2
import numpy as np
from PIL import Image
from param_settings import xl, xr, yt, yb ,Car_dst_points
from image_processing import LuminanceBalancer, ImageStitcher, ImageAdjuster

Dataset_path = '../Dataset/'

def load_image(path):
    """
    Load an image from a given file path.
    
    Args:
        path (str): The path to the image file.
    
    Returns:
        np.ndarray: The loaded image.
    
    Raises:
        FileNotFoundError: If the image file is not found or cannot be loaded.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Error: Could not load image from {path}")
    return image

def main():
    # Load images from different camera angles
    try:
        front = load_image('out_merged_Images/front_warped_image.png')
        left = load_image('out_merged_Images/left_warped_image.png')
        rear = load_image('out_merged_Images/Rear_warped_image.png')
        right = load_image('out_merged_Images/right_warped_image.png')
        car = cv2.imread(f'{Dataset_path}golf_car.png', cv2.IMREAD_UNCHANGED)  # Load car image with alpha channel

        images = [front, left, rear, right]

    except FileNotFoundError as e:
        print(e)
        return  # Exit if any image fails to load

    # Ensure all images are loaded successfully before processing
    if any(img is None for img in images):
        print("One or more images failed to load. Exiting.")
        return

    # Merge images using weighted masks
    final_merged_image = ImageStitcher.get_weights_and_masks(images)
    cv2.imwrite('out_merged_Images/final_merged_image.png', final_merged_image)

    # Blend images using selected mode
    mode = 'alpha_blend'  # Options: 'alpha_blend' or 'hard_overlay'
    merged = ImageAdjuster.merge_images(images, mode=mode, alpha=0.25)

    # Apply white balance correction to merged image
    white_balanced = LuminanceBalancer.make_white_balance(merged)
    cv2.imwrite(f'out_merged_Images/white_balanced_result_{mode}.png', white_balanced)

    # Overlay the car image onto the final merged images
    final_merged_image_car = ImageAdjuster.overlay_image_perspective(final_merged_image, car, Car_dst_points)
    white_balanced_car = ImageAdjuster.overlay_image_perspective(white_balanced, car, Car_dst_points)

    # Combine both processed images side by side for comparison
    comparison_image = np.hstack((white_balanced_car, final_merged_image_car))

    # Resize the comparison image to fit within 1440x990 resolution
    comparison_image_resized = cv2.resize(comparison_image, (1440, 990))

    # Display the comparison image
    cv2.imshow('Comparison: White Balanced Result (Left) vs Final Merged Image (Right)', comparison_image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
