import cv2
import os
import numpy as np
import time
import threading
from collections import deque
from image_processing import LuminanceBalancer, ImageStitcher, ImageAdjuster
from param_settings import img_car, Car_dst_points, total_w, total_h, xl, xr, yt, yb

class ImageProcessor:
    def __init__(self, video_paths, img_car, display_width=800, display_height=600, map_width=total_w, map_height=total_h):
        """
        Initialize the ImageProcessor with video sources and calibration data.
        """
        self.video_paths = video_paths
        self.car = img_car
        self.display_width = display_width
        self.display_height = display_height
        self.map_width = map_width
        self.map_height = map_height
        self.car_dst_points = Car_dst_points
        
        # Initialize VideoCapture for 4 cameras
        self.caps = {key: cv2.VideoCapture(path) for key, path in video_paths.items()}
        
        # Load calibration data (Intrinsic/Extrinsic/Homography) for each camera
        self.calibration_data = {
            cam_id: self.load_calibration_data(cam_id) for cam_id in ["front", "left", "rear", "right"]
        }
        
        # Buffer to store the last 10 frames of processing time for Moving Average calculation
        self.time_history = deque(maxlen=10)

        # --- PRE-COMPUTATION SECTION ---
        # We pre-compute the blending weight matrices (G0-G3) once to save CPU time during the main loop.
        print("Pre-computing Weight Matrices from actual first frames...")
        
        temp_warped = {}
        for cam_id, cap in self.caps.items():
            ret, frame = cap.read()
            if ret:
                data = self.calibration_data[cam_id]
                # Undistort and Warp the first frame of each video
                undistorted = cv2.remap(frame, data["map1"], data["map2"], cv2.INTER_LINEAR)
                warped = cv2.warpPerspective(undistorted, data["homography"], (self.map_width, self.map_height))
                temp_warped[cam_id] = warped
            
            # Reset video capture back to the first frame after pre-computation
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Extract warped frames for stitching logic
        f, l, b, r = temp_warped["front"], temp_warped["left"], temp_warped["rear"], temp_warped["right"]

        # Generate blending masks for the 4 overlapping corners (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
        # Using the same logic as Step 4 (Stitching Logic)
        self.G0, _ = ImageStitcher.get_weight_mask_matrix_liverun(f[:yt, :xl], l[:yt, :xl])
        self.G1, _ = ImageStitcher.get_weight_mask_matrix_liverun(f[:yt, xr:], r[:yt, xr:])
        self.G2, _ = ImageStitcher.get_weight_mask_matrix_liverun(b[yb:, :xl], l[yb:, :xl])
        self.G3, _ = ImageStitcher.get_weight_mask_matrix_liverun(b[yb:, xr:], r[yb:, xr:])

        print("G-Matrices Pre-computed successfully using Step 4 logic!")

    def load_calibration_data(self, cameraID):
        """
        Load YAML calibration files and initialize Undistort maps for faster remapping.
        """
        yaml_filename = os.path.join('yaml', f'calibration_data_{cameraID}.yaml')
        fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
        
        mtx = fs.getNode("camera_matrix").mat()
        dist = fs.getNode("dist_coeffs").mat()
        homography = fs.getNode("homography").mat()
        fs.release()

        # Initialize Rectify Map: Converting Undistort logic into a Map for cv2.remap (3x faster)
        h, w = 720, 1280 # Adjust according to your camera resolution
        map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (w, h), cv2.CV_32FC1)

        return {
            "map1": map1,
            "map2": map2,
            "homography": homography
        }

    def process_image(self, cam_id, frame):
        """
        Apply Undistort and Perspective Warp to a single camera frame.
        """
        start_t = time.time()
        data = self.calibration_data[cam_id] 
        
        # Fast Undistort using remap
        undistorted = cv2.remap(frame, data["map1"], data["map2"], cv2.INTER_LINEAR)
        # Bird-eye view transformation
        warped = cv2.warpPerspective(undistorted, data["homography"], (self.map_width, self.map_height))
        
        proc_time = time.time() - start_t
        return undistorted, warped, proc_time

    def grab_frame(self, cam_id, cap, images, warped_rgba_, processing_times, index):
        """
        Worker function for threading: Read and process a frame from a specific camera.
        """
        ret, frame = cap.read()
        if not ret:
            # Auto-restart video if it reaches the end
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        # Process the captured frame
        undistorted, warped, proc_time = self.process_image(cam_id, frame)
        
        # Store results in the shared lists
        images[index] = undistorted
        warped_rgba_[index] = warped
        processing_times[index] = proc_time

    def run(self):
        """
        Main execution loop: Multi-threaded frame grabbing, stitching, and display.
        """
        cam_names = ["front", "left", "rear", "right"]

        # Create GUI windows with resizing enabled for better visibility
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 600, 450)
        
        cv2.namedWindow("BEV Merged Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("BEV Merged Image", 500, 650)

        while True:
            start_time_total = time.time()
            images = [None] * 4
            warped_rgba_ = [None] * 4
            processing_times = [0.0] * 4
            threads = []

            # Launch 4 threads to process 4 camera inputs simultaneously
            for i, (cam_id, cap) in enumerate(self.caps.items()):
                thread = threading.Thread(target=self.grab_frame, args=(cam_id, cap, images, warped_rgba_, processing_times, i))
                threads.append(thread)
                thread.start()

            # Wait for all threads to finish
            for thread in threads:
                thread.join()

            # Proceed only if all 4 camera frames were successfully captured
            if all(img is not None for img in images):
                
                # Fast stitching using pre-computed G-matrices
                final_merged_image = ImageStitcher.get_weights_and_masks_fast(
                    warped_rgba_, self.G0, self.G1, self.G2, self.G3
                )
                
                # Overlay car image (Optional: uncomment if needed)
                # final_merged_image = ImageAdjuster.overlay_image_perspective(final_merged_image, self.car, self.car_dst_points)

                # --- STATS CALCULATION ---
                current_proc_time = (time.time() - start_time_total) * 1000
                self.time_history.append(current_proc_time)
                avg_proc_time = sum(self.time_history) / len(self.time_history)
                
                # Format string for performance display
                display_stats = f"Avg (10f): {avg_proc_time:.1f} ms | Cur: {current_proc_time:.1f} ms"
                print(display_stats)

                # --- VISUALIZATION: Quad-View ---
                resized_width = self.display_width // 2
                resized_height = self.display_height // 2
                resized_images = [cv2.resize(img, (resized_width, resized_height)) for img in images]

                for i, img in enumerate(resized_images):
                    cam_text = f"{cam_names[i]}: {processing_times[i]*1000:.1f} ms"
                    cv2.putText(img, cam_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                top_row = np.hstack((resized_images[0], resized_images[1]))
                bottom_row = np.hstack((resized_images[2], resized_images[3]))
                quad_view = np.vstack((top_row, bottom_row))

                # --- VISUALIZATION: BEV View ---
                # Draw stats on the final merged image
                cv2.putText(final_merged_image, display_stats, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

                # Update Windows
                cv2.imshow("Image", quad_view)
                cv2.imshow("BEV Merged Image", final_merged_image)

                if cv2.waitKey(1) == ord('q'):
                    break

        # Cleanup
        for cap in self.caps.values():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    video_folder = "../Dataset/liverun_outdoor_VDO_Day"
    # video_folder = "../Dataset/liverun_outdoor_VDO_Night_1"
    # video_folder = "../Dataset/liverun_outdoor_VDO_Night_2"

    video_paths = {
        "front": os.path.join(video_folder, "front.mp4"),
        "left": os.path.join(video_folder, "left.mp4"),
        "rear": os.path.join(video_folder, "rear.mp4"),
        "right": os.path.join(video_folder, "right.mp4"),
    }
    
    processor = ImageProcessor(video_paths, img_car)
    processor.run()



