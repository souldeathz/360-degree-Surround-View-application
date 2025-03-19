import cv2
import os
import numpy as np
import time
from param_settings import Golf_img_Path

# Define the path to the dataset
# dataset_path = "Dataset/liverun_Round1"
dataset_path = "../Dataset/liverun_outdoor/football"
car = cv2.imread(Golf_img_Path, cv2.IMREAD_UNCHANGED)  # Load car image with alpha channel
folders = ["front", "left", "rear", "right"]

display_width = 800
display_height = 600

Map_width = 1040 
Map_height = 1191

# Define destination points for perspective transformation
Car_dst_points = np.float32([
    [465, 465],  # Point 1
    [575, 465],  # Point 2
    [465, 685],  # Point 3
    [575, 685]   # Point 4
])

def merge_images(images, mode='hard_overlay', alpha=0.25):
    """
    Merge a list of images using the specified mode.
    Parameters:
    images (list of numpy.ndarray): List of images to be merged. All images must have the same dimensions.
    mode (str): Mode of merging images. Options are 'hard_overlay' and 'alpha_blend'. Default is 'hard_overlay'.
        - 'hard_overlay': Overlays non-zero pixels from each image onto the merged image.
        - 'alpha_blend': Blends images using alpha blending with the specified alpha value.
    alpha (float): Alpha value for blending in 'alpha_blend' mode. Default is 0.25.
    Returns:
    numpy.ndarray: The merged image.
    Note:
    - In 'hard_overlay' mode, non-zero pixels from each image will replace the corresponding pixels in the merged image.
    - In 'alpha_blend' mode, images are blended together with the specified alpha value.
    """
    height, width, _ = images[0].shape
    merged_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    if mode == 'hard_overlay':
        for img in images:
            mask = (img != 0).any(axis=2)
            merged_image[mask] = img[mask]
    
    elif mode == 'alpha_blend':
        total_weight = alpha * len(images)
        for img in images:
            merged_image = cv2.addWeighted(merged_image, 1, img, alpha, 0)
        merged_image = cv2.convertScaleAbs(merged_image * (1/total_weight))
    
    return merged_image

def overlay_image_perspective(background, overlay, dst_points):
    """
    Overlays an image onto a background image using a perspective transformation.
    Args:
        background (numpy.ndarray): The background image onto which the overlay will be applied.
        overlay (numpy.ndarray): The overlay image with an alpha channel.
        dst_points (numpy.ndarray): A 4x2 array of destination points for the perspective transformation.
    Returns:
        numpy.ndarray: The background image with the overlay applied.
    Note:
        The overlay image must have an alpha channel (4th channel) for transparency.
        The dst_points should be in the order: top-left, top-right, bottom-left, bottom-right.
    """
    src_points = np.float32([
        [0, 0],
        [overlay.shape[1] - 1, 0],
        [0, overlay.shape[0] - 1],
        [overlay.shape[1] - 1, overlay.shape[0] - 1]
    ])
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_overlay = cv2.warpPerspective(overlay, matrix, (background.shape[1], background.shape[0]))
    
    alpha_channel = warped_overlay[:, :, 3] / 255.0
    for c in range(0, 3):
        background[:, :, c] = alpha_channel * warped_overlay[:, :, c] + (1 - alpha_channel) * background[:, :, c]
    
    return background

def process_image(image, cameraID):
    """
    Process the image based on the camera ID.
    Args:
        image (numpy.ndarray): The input image to be processed.
        cameraID (str): The ID of the camera (e.g., 'front', 'left', 'rear', 'right').
    Returns:
        tuple: A tuple containing the processed image and the warped RGBA image.
    Note:
        The function reads calibration data from a YAML file, undistorts the image, and applies a perspective warp.
    """
    yaml_filename = os.path.join('yaml', f'calibration_data_{cameraID}.yaml')
    fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    resolution = fs.getNode("resolution").mat()
    H = fs.getNode("homography").mat()
    
    # Process the image based on the camera ID
    if cameraID == "front":
        processed_image = image  # Add actual processing for front camera

    elif cameraID == "left":
        processed_image = image  # Add actual processing for left camera

    elif cameraID == "rear":
        img_src = cv2.rotate(image, cv2.ROTATE_180)
        processed_image = img_src  # Add actual processing for rear camera

    elif cameraID == "right":
        img_src = cv2.rotate(image, cv2.ROTATE_180)
        processed_image = img_src  # Add actual processing for right camera

    else:
        # Default processing if cameraID is not recognized
        processed_image = image
    
    # Undistort the image using the camera matrix and distortion coefficients
    img_src_undistorted = cv2.undistort(processed_image, camera_matrix, dist_coeffs)
    
    # Apply perspective warp using the homography matrix
    warped = cv2.warpPerspective(img_src_undistorted, H, (Map_width, Map_height))
    
    # Convert the warped image to RGBA format
    warped_rgba = cv2.cvtColor(warped, cv2.COLOR_BGR2BGRA)
    warped_rgba[np.all(warped_rgba[:, :, :3] == [0, 0, 0], axis=-1)] = [0, 0, 0, 0]
    
    # Convert the warped RGBA image back to RGB format
    warped_rgb = cv2.cvtColor(warped_rgba, cv2.COLOR_BGRA2BGR)
    
    return processed_image, warped_rgb

# Loop through each folder and display the images
index = 0
while True:
    filename = os.listdir(os.path.join(dataset_path, folders[0]))[index]
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        images = []
        warped_rgba_ = []
        start_time = time.time()  # Start time for processing

        for folder in folders:
            img_path = os.path.join(dataset_path, folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                processed_img,warped_rgba = process_image(img,folder)
                images.append(processed_img)
                warped_rgba_.append(warped_rgba)
        
        if len(images) == 4:

            # Merge images using the provided merge_images function
            mode = 'hard_overlay'
            merged_car_image_ = merge_images(warped_rgba_, mode=mode, alpha=0.25)

            # Overlay car image at specified coordinates
            merged_car_image = overlay_image_perspective(merged_car_image_, car, Car_dst_points)

            end_time = time.time()  # End time for processing
            process_time = end_time - start_time
            print(f"Processed time in {process_time:.2f} seconds")

            # Display Part
            # Merge the images into one
            # Resize images to fit the specified display size

            resized_width = display_width // 2
            resized_height = display_height // 2
            resized_images = [cv2.resize(img, (resized_width, resized_height)) for img in images]
            top_row = np.hstack((resized_images[0], resized_images[1]))
            bottom_row = np.hstack((resized_images[2], resized_images[3]))
            merged_Display_image = np.vstack((top_row, bottom_row))
            
            # Display the merged image
            cv2.imshow("Merged Image", merged_Display_image)
            # Display the final merged image with the car overlay
            cv2.imshow("Merged Image with Car Overlay", merged_car_image)
            cv2.waitKey(10)  # Wait for 0.5 seconds
    
    index += 1
    if index >= len(os.listdir(os.path.join(dataset_path, folders[0]))):
        index = 0
