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

# === สร้าง Blend Mask ต่อกล้อง ===
def create_blend_mask(name, width, height, car_width, car_height):
    mask = np.zeros((height, width), dtype=np.uint8)

    if name == 'front':
        points = np.array([
            [0, 0],
            [width, 0],
            [width, height // 5],
            [(width + car_width) // 2, (height - car_height) // 2],
            [(width - car_width) // 2, (height - car_height) // 2],
            [0, height // 5]
        ], dtype=np.int32)
    elif name == 'rear':
        points = np.array([
            [0, height],
            [width, height],
            [width, height - height // 5],
            [(width + car_width) // 2, (height + car_height) // 2],
            [(width - car_width) // 2, (height + car_height) // 2],
            [0, height - height // 5]
        ], dtype=np.int32)
    elif name == 'left':
        points = np.array([
            [0, 0],
            [0, height],
            [width // 5, height],
            [(width - car_width) // 2, (height + car_height) // 2],
            [(width - car_width) // 2, (height - car_height) // 2],
            [width // 5, 0]
        ], dtype=np.int32)
    elif name == 'right':
        points = np.array([
            [width, 0],
            [width, height],
            [width - width // 5, height],
            [(width + car_width) // 2, (height + car_height) // 2],
            [(width + car_width) // 2, (height - car_height) // 2],
            [width - width // 5, 0]
        ], dtype=np.int32)
    else:
        raise ValueError("Invalid camera name")

    cv2.fillPoly(mask, [points], 255)
    return np.repeat(mask[:, :, np.newaxis], 3, axis=2) / 255.0  # shape: (H, W, 3) float32

# === Blend ภาพ 4 กล้อง ===
def blend_warped_images(warped_images, width, height, car_width, car_height):
    masks = {
        "front": create_blend_mask("front", width, height, car_width, car_height),
        "rear":  create_blend_mask("rear", width, height, car_width, car_height),
        "left":  create_blend_mask("left", width, height, car_width, car_height),
        "right": create_blend_mask("right", width, height, car_width, car_height),
    }

    # จับคู่กล้องกับลำดับ warped_rgba_
    camera_order = ["front", "left", "rear", "right"]
    blended = np.zeros_like(warped_images[0], dtype=np.float32)

    for i, cam_id in enumerate(camera_order):
        blended += warped_images[i].astype(np.float32) * masks[cam_id]

    return np.clip(blended, 0, 255).astype(np.uint8)

# === วางภาพรถไว้ตรงกลาง ===


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
        # รวมภาพ 4 กล้องด้วย blend mask
        merged_blended = blend_warped_images(warped_rgba_, Map_width, Map_height, car_width=250, car_height=400)

        # แปะรถลงภาพที่รวมแล้ว
        end_time = time.time()  # End time for processing
        print(f"Processed time: {end_time - start_time:.2f} seconds")

        # Display Part
        # Merge the images into one
        # Resize images to fit the specified display size
        resized_images = [cv2.resize(img, (display_width // 2, display_height // 2)) for img in images]
        top_row = np.hstack((resized_images[0], resized_images[1]))
        bottom_row = np.hstack((resized_images[2], resized_images[3]))
        merged_Display_image = np.vstack((top_row, bottom_row))

        # Display the final merged image with the car overlay
        resized_merged_car_image = cv2.resize(merged_blended, (500, 900))

        # Display the merged image
        cv2.imshow("Merged Image", merged_Display_image)
        # Display the final merged image with the car overlay        
        cv2.imshow("Merged Image with Car Overlay", resized_merged_car_image)
        cv2.waitKey(1)

# Release resources
for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()


