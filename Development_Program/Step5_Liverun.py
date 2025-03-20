import cv2
import os
import numpy as np
import time
from param_settings import img_car , Car_dst_points ,total_w , total_h
from image_processing import LuminanceBalancer, ImageStitcher, ImageAdjuster
# Video file paths
video_paths = {
    "front": "../Dataset/liverun_outdoor/front.mp4",
    "left": "../Dataset/liverun_outdoor/left.mp4",
    "rear": "../Dataset/liverun_outdoor/rear.mp4",
    "right": "../Dataset/liverun_outdoor/right.mp4",
}

# Open video captures
caps = {key: cv2.VideoCapture(path) for key, path in video_paths.items()}
# Display settings
display_width, display_height = 800, 600
Map_width, Map_height = total_w, total_h

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
    H = fs.getNode("homography").mat()

    if cameraID in ["rear", "right"]:
        image = cv2.rotate(image, cv2.ROTATE_180)
    # Undistort the image using the camera matrix and distortion coefficients
    img_src_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
    # Apply perspective warp using the homography matrix    
    warped = cv2.warpPerspective(img_src_undistorted, H, (Map_width, Map_height))
    # Convert the warped image to RGBA format
    warped_rgba = cv2.cvtColor(warped, cv2.COLOR_BGR2BGRA)
    warped_rgba[np.all(warped_rgba[:, :, :3] == [0, 0, 0], axis=-1)] = [0, 0, 0, 0]
     # Convert the warped RGBA image back to RGB format   
    warped_rgb = cv2.cvtColor(warped_rgba, cv2.COLOR_BGRA2BGR)
    
    return image, warped_rgb

# Video processing loop
while True:
    images = []
    warped_rgba_ = []
    start_time = time.time()

    for cam_id, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            print(f"End of video {cam_id}. Restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video
            continue

        processed_img, warped_rgba = process_image(frame, cam_id)
        images.append(processed_img)
        warped_rgba_.append(warped_rgba)
    
    if len(images) == 4:
        # Merge images using the provided merge_images function
        mode = 'hard_overlay'
        # Overlay car image at specified coordinates        
        merged_car_image_ = ImageAdjuster.merge_images(warped_rgba_, mode=mode, alpha=0.25)
        merged_car_image = ImageAdjuster.overlay_image_perspective(merged_car_image_, img_car, Car_dst_points)

        end_time = time.time()  # End time for processing
        print(f"Processed time: {end_time - start_time:.2f} seconds")

        # Display Part
        # Merge the images into one
        # Resize images to fit the specified display size
        resized_images = [cv2.resize(img, (display_width // 2, display_height // 2)) for img in images]
        top_row = np.hstack((resized_images[0], resized_images[1]))
        bottom_row = np.hstack((resized_images[2], resized_images[3]))
        merged_Display_image = np.vstack((top_row, bottom_row))
        # Display the merged image
        cv2.imshow("Merged Image", merged_Display_image)
        # Display the final merged image with the car overlay        
        cv2.imshow("Merged Image with Car Overlay", merged_car_image)
        cv2.waitKey(1)

# Release resources
for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()
