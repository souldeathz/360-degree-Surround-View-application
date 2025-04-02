import cv2
import os
import numpy as np
import time
import threading
import concurrent.futures
from collections import deque
from param_settings import img_car , Car_dst_points ,total_w , total_h 
from image_processing import ImageAdjuster


class VideoProcessor:
    def __init__(self, video_paths, img_car, display_width=800, display_height=600, map_width=total_w, map_height=total_h):
        self.video_paths = video_paths
        self.car = img_car
        self.caps = {key: cv2.VideoCapture(path) for key, path in video_paths.items()}
        self.display_width = display_width
        self.display_height = display_height
        self.map_width = map_width
        self.map_height = map_height
        self.car_dst_points = Car_dst_points
        self.running = True

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
        # Preload images to avoid delay in processing
        self.processed_frames = deque(maxlen=5) 
    
    def process_image(self, image, cameraID):
        start_time = time.time()
        
        calib = self.calibration_data[cameraID]
        camera_matrix = calib["camera_matrix"]
        dist_coeffs = calib["dist_coeffs"]
        H = calib["homography"]

        img_src_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
        if cameraID in ["rear", "right"]:
            img_src_undistorted = cv2.rotate(img_src_undistorted, cv2.ROTATE_180)

        warped = cv2.warpPerspective(img_src_undistorted, H, (self.map_width, self.map_height))
        process_time = time.time() - start_time
        return img_src_undistorted, warped, process_time
    
    def grab_image(self, cam_id, cap, images, warped_rgba_, processing_times, index):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        processed_img, warped_rgba, proc_time = self.process_image(frame, cam_id)
        images[index] = processed_img
        warped_rgba_[index] = warped_rgba
        processing_times[index] = proc_time

    def processing_loop(self):
        while self.running:
            loop_start_time = time.time()

            images = [None] * len(self.caps)
            warped_rgba_ = [None] * len(self.caps)
            processing_times = [0.0] * len(self.caps)
            threads = []

            for i, (cam_id, cap) in enumerate(self.caps.items()):
                thread = threading.Thread(target=self.grab_image, args=(cam_id, cap, images, warped_rgba_, processing_times, i))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            if all(img is not None for img in images):
                # ----- Merge warped images into BEV -----
                warped_stack = np.stack(warped_rgba_, axis=0)
                mask = np.any(warped_stack != 0, axis=0)
                final_merged_image = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
                final_merged_image[mask] = np.max(warped_stack, axis=0)[mask]

                # ----- Overlay car -----
                merged_car_image = ImageAdjuster.overlay_image_perspective(final_merged_image.copy(), self.car, self.car_dst_points)

                # ----- Resize 4 camera images for display -----
                resized_width = self.display_width // 2
                resized_height = self.display_height // 2
                resized_images = [cv2.resize(img, (resized_width, resized_height)) for img in images]

                cam_names = ["Front", "Left", "Rear", "Right"]
                for i, img in enumerate(resized_images):
                    text = f"{cam_names[i]}: {processing_times[i]*1000:.1f} ms"
                    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                top_row = np.hstack((resized_images[0], resized_images[1]))
                bottom_row = np.hstack((resized_images[2], resized_images[3]))
                merged_Display_image = np.vstack((top_row, bottom_row))

                # ----- Resize final BEV image -----
                resized_final_image = cv2.resize(merged_car_image, (800, 900))

                # ----- Show processing time on both images -----
                loop_proc_time = (time.time() - loop_start_time) * 1000  # in milliseconds
                time_text = f"Total Processing: {loop_proc_time:.1f} ms"

                # Show on BEV image
                cv2.putText(resized_final_image, time_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                print(time_text)
                self.processed_frames.append((merged_Display_image, resized_final_image))


    def display_loop(self):
        while self.running:
            if self.processed_frames:
                merged_Display_image, resized_final_image = self.processed_frames[-1]
                cv2.imshow("Merged 4 POV Images", merged_Display_image)
                cv2.imshow("Merged Car Image", resized_final_image)
                key = cv2.waitKey(1)
                if key == ord('q'):  # กด q เพื่อออก
                    self.running = False
                    break
            time.sleep(0.01)

    def run(self):
        display_thread = threading.Thread(target=self.display_loop, daemon=True)
        display_thread.start()
        self.processing_loop()  # จะหยุดเองถ้า self.running = False

        for cap in self.caps.values():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    video_paths = {
        "front": "../Dataset/liverun_outdoor_Day/front.mp4",
        "left": "../Dataset/liverun_outdoor_Day/left.mp4",
        "rear": "../Dataset/liverun_outdoor_Day/rear.mp4",
        "right": "../Dataset/liverun_outdoor_Day/right.mp4",          
        # "front": "../Dataset/liverun_outdoor/front.mp4",
        # "left": "../Dataset/liverun_outdoor/left.mp4",
        # "rear": "../Dataset/liverun_outdoor/rear.mp4",
        # "right": "../Dataset/liverun_outdoor/right.mp4",
    }
    processor = VideoProcessor(video_paths, img_car)
    processor.run()
