import cv2
import os
import numpy as np
import time
import threading
from param_settings import img_car , Car_dst_points ,total_w , total_h 
from image_processing import LuminanceBalancer, ImageStitcher, ImageAdjuster
import concurrent.futures

class VideoProcessor:
    def __init__(self, video_paths, img_car, display_width=800, display_height=600, map_width=1040, map_height=1191):
        self.video_paths = video_paths
        self.car = img_car
        self.caps = {key: cv2.VideoCapture(path) for key, path in video_paths.items()}
        self.display_width = display_width
        self.display_height = display_height
        self.map_width = map_width
        self.map_height = map_height
        self.car_dst_points = Car_dst_points

        # Load calibration data once**
        self.calibration_data = {}
        for cam_id in ["front", "left", "rear", "right"]:
            yaml_filename = os.path.join('yaml', f'calibration_data_{cam_id}.yaml')
            fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
            self.calibration_data[cam_id] = {
                "camera_matrix": fs.getNode("camera_matrix").mat(),
                "dist_coeffs": fs.getNode("dist_coeffs").mat(),
                "homography": fs.getNode("homography").mat()
            }
            fs.release()
    
    def process_image(self, image, cameraID):
        start_time = time.time()
        
        # Retrieve preloaded calibration data
        calib = self.calibration_data[cameraID]
        camera_matrix = calib["camera_matrix"]
        dist_coeffs = calib["dist_coeffs"]
        H = calib["homography"]

        # **Optimization 2: Reduce Unnecessary Copies**
        if cameraID in ["rear", "right"]:
            image = cv2.rotate(image, cv2.ROTATE_180)

        img_src_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)

        warped = cv2.warpPerspective(img_src_undistorted, H, (self.map_width, self.map_height))
        
        end_time = time.time()
        # print(f"Processing time for {cameraID}: {end_time - start_time:.4f} seconds")
        return img_src_undistorted, warped  # Returning undistorted & warped image directly
    
    def grab_image(self, cam_id, cap, images, warped_rgba_, index):
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        processed_img, warped_rgba = self.process_image(frame, cam_id)
        images[index] = processed_img
        warped_rgba_[index] = warped_rgba
        end_time = time.time()
        # print(f"Frame grab time for {cam_id}: {end_time - start_time:.4f} seconds")

    def run(self):
        while True:
            start_time_total = time.time()
            images = [None] * len(self.caps)
            warped_rgba_ = [None] * len(self.caps)
            threads = []
            start_time_threading = time.time()
            
            for i, (cam_id, cap) in enumerate(self.caps.items()):
                thread = threading.Thread(target=self.grab_image, args=(cam_id, cap, images, warped_rgba_, i))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            end_time_threading = time.time()
            # print(f"Total threading execution time: {end_time_threading - start_time_threading:.4f} seconds "
            #     f"(Includes: Grab frames from video sources → Undistort & Warp frames → Store processed images)")           
            if all(img is not None for img in images):
                start_time_merge = time.time()
                for i in range(len(warped_rgba_)):
                    if warped_rgba_[i].shape[:2] != (self.map_height, self.map_width):
                        warped_rgba_[i] = cv2.resize(warped_rgba_[i], (self.map_width, self.map_height))
                

                # (Normal Fast) Option 1: Use numpy to stack images and find the last non-zero pixel
                # merged_car_image = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
                # for img in warped_rgba_:
                #     mask = (img != 0).any(axis=2)
                #     merged_car_image[mask] = img[mask]

                # (Very Fast) Option 2: Use numpy to stack images and find the last non-zero pixel
                # merged_car_image = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
                # # Stack images along a new axis
                # warped_stack = np.stack(warped_rgba_, axis=0)  # Shape: (4, H, W, 3)
                # # Create a mask for non-zero pixels in each image
                # nonzero_mask = np.any(warped_stack != 0, axis=-1)  # Shape: (4, H, W)
                # # Find the last nonzero pixel index for each position
                # last_nonzero_idx = np.argmax(nonzero_mask[::-1], axis=0)  # Reverse order to get last occurrence
                # # Reverse index to match original order
                # last_nonzero_idx = nonzero_mask.shape[0] - 1 - last_nonzero_idx
                # # Use advanced indexing to select pixels from the last nonzero image
                # merged_car_image = warped_stack[last_nonzero_idx, np.arange(self.map_height)[:, None], np.arange(self.map_width)]

                # (Very very very Fast) Option 3: Use numpy to stack images and find the last non-zero pixel
                final_merged_image = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
                # Stack images along a new axis (shape: (4, H, W, 3))
                warped_stack = np.stack(warped_rgba_, axis=0)
                # Create a mask for non-zero pixels
                mask = np.any(warped_stack != 0, axis=0)
                # Apply np.max() for selecting the highest intensity pixel
                final_merged_image[mask] = np.max(warped_stack, axis=0)[mask]

                end_time_merge = time.time()
                # print(f"Image merging time: {end_time_merge - start_time_merge:.4f} seconds")
                
                end_time_total = time.time()
                print(f"Total processing time: {end_time_total - start_time_total:.4f} seconds")


                merged_car_image = ImageAdjuster.overlay_image_perspective(final_merged_image, self.car, self.car_dst_points)
                resized_width, resized_height = self.display_width // 2, self.display_height // 2
                resized_images = [cv2.resize(img, (resized_width, resized_height)) for img in images]
                top_row = np.hstack((resized_images[0], resized_images[1]))
                bottom_row = np.hstack((resized_images[2], resized_images[3]))
                merged_Display_image = np.vstack((top_row, bottom_row))
                new_width = 800
                new_height = 900
                resized_final_image = cv2.resize(final_merged_image, (new_width, new_height))
                cv2.imshow("Merged 4 POV Images", merged_Display_image)
                cv2.imshow("Merged Car Image", merged_car_image)
                cv2.waitKey(1)
        
        for cap in self.caps.values():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    video_paths = {
        "front": "../Dataset/liverun_outdoor/front.mp4",
        "left": "../Dataset/liverun_outdoor/left.mp4",
        "rear": "../Dataset/liverun_outdoor/rear.mp4",
        "right": "../Dataset/liverun_outdoor/right.mp4",
    }
    processor = VideoProcessor(video_paths, img_car, 800, 600, total_w, total_h)
    processor.run()
