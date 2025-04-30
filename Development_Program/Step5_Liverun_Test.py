
import cv2
import os
import numpy as np
import time
from param_settings import img_car , Car_dst_points , total_w , total_h
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

def process_image(image, cameraID, remap_data):
    map1 = remap_data["map1"]
    map2 = remap_data["map2"]
    H = remap_data["H"]

    # Step 1: remap
    # remap_start = time.time()
    undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

    # Step 2: rotate
    if cameraID in ["rear", "right"]:
        undistorted = cv2.rotate(undistorted, cv2.ROTATE_180)

    # Step 3: warp
    warped = cv2.warpPerspective(undistorted, H, (Map_width, Map_height))

    # Step 4: transparent BG
    warped_rgba = cv2.cvtColor(warped, cv2.COLOR_BGR2BGRA)
    warped_rgba[np.all(warped_rgba[:, :, :3] == [0, 0, 0], axis=-1)] = [0, 0, 0, 0]
    warped_rgb = cv2.cvtColor(warped_rgba, cv2.COLOR_BGRA2BGR)

    # remap_end = time.time()
    # print(f"[{cameraID}] ⏱️ Remap+warp time: {remap_end - remap_start:.4f} seconds")

    return image, warped_rgb

def prepare_remap_maps(camera_ids):
    remap_map = {}
    for cam_id in camera_ids:
        yaml_filename = os.path.join('yaml', f'calibration_data_{cam_id}.yaml')
        fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
        camera_matrix = fs.getNode("camera_matrix").mat()
        dist_coeffs = fs.getNode("dist_coeffs").mat()
        H = fs.getNode("homography").mat()

        cap = cv2.VideoCapture(video_paths[cam_id])
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError(f"❌ Cannot read frame from {cam_id}")

        h, w = frame.shape[:2]

        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            R=None,
            newCameraMatrix=camera_matrix,
            size=(w, h),
            m1type=cv2.CV_32FC1
        )

        remap_map[cam_id] = {
            "map1": map1,
            "map2": map2,
            "H": H,
            "camera_matrix": camera_matrix
        }

    return remap_map  # ✅ สำคัญมาก


camera_ids = ["front", "left", "rear", "right"]
remap_maps = prepare_remap_maps(camera_ids)
# Video processing loop
while True:
    images = []
    warped_rgba_ = []
    start_time = time.time()

    for cam_id, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            print(f"End of video {cam_id}. Restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        processed_img, warped_rgba = process_image(frame, cam_id, remap_maps[cam_id])
        images.append(processed_img)
        warped_rgba_.append(warped_rgba)
    
    if len(images) == 4:
        mode = 'hard_overlay'
        merged_car_image_ = ImageAdjuster.merge_images(warped_rgba_, mode=mode, alpha=0.25)
        merged_car_image = ImageAdjuster.overlay_image_perspective(merged_car_image_, img_car, Car_dst_points)

        end_time = time.time()
        print(f"Processed time: {end_time - start_time:.2f} seconds")

        resized_images = [cv2.resize(img, (display_width // 2, display_height // 2)) for img in images]
        top_row = np.hstack((resized_images[0], resized_images[1]))
        bottom_row = np.hstack((resized_images[2], resized_images[3]))
        merged_Display_image = np.vstack((top_row, bottom_row))

        resized_merged_car_image = cv2.resize(merged_car_image, (500, 900))

        cv2.imshow("Merged Image", merged_Display_image)
        cv2.imshow("Merged Image with Car Overlay", resized_merged_car_image)
        cv2.waitKey(1)

for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()
