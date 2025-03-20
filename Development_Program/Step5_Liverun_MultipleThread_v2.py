import cv2
import os
import numpy as np
import time
import threading
from image_processing import LuminanceBalancer, ImageStitcher, ImageAdjuster
from param_settings import Golf_img_Path

class ImageProcessor:
    def __init__(self, video_paths, car_image_path, display_width=800, display_height=600, map_width=1040, map_height=1191):
        self.video_paths = video_paths
        self.car = cv2.imread(car_image_path, cv2.IMREAD_UNCHANGED)
        self.caps = {key: cv2.VideoCapture(path) for key, path in video_paths.items()}
        self.display_width = display_width
        self.display_height = display_height
        self.map_width = map_width
        self.map_height = map_height
        self.car_dst_points = np.float32([
            [465, 465],
            [575, 465],
            [465, 685],
            [575, 685]
        ])
        
        # Load calibration data once
        self.calibration_data = {
            cam_id: self.load_calibration_data(cam_id) for cam_id in ["front", "left", "rear", "right"]
        }
    
    def load_calibration_data(self, cameraID):
        yaml_filename = os.path.join('yaml', f'calibration_data_{cameraID}.yaml')
        fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
        data = {
            "camera_matrix": fs.getNode("camera_matrix").mat(),
            "dist_coeffs": fs.getNode("dist_coeffs").mat(),
            "homography": fs.getNode("homography").mat()
        }
        fs.release()
        return data
    
    def process_image(self, image, cameraID):
        calib = self.calibration_data[cameraID]
        camera_matrix, dist_coeffs, H = calib["camera_matrix"], calib["dist_coeffs"], calib["homography"]
        
        if cameraID in ["rear", "right"]:
            image = cv2.rotate(image, cv2.ROTATE_180)
        
        img_src_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
        warped = cv2.warpPerspective(img_src_undistorted, H, (self.map_width, self.map_height))
        
        return img_src_undistorted, warped
    
    def grab_frame(self, cam_id, cap, images, warped_rgba_, index):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        images[index], warped_rgba_[index] = self.process_image(frame, cam_id)

    def run(self):
        while True:
            start_time_total = time.time()
            images = [None] * len(self.caps)
            warped_rgba_ = [None] * len(self.caps)
            threads = []
            start_time_threading = time.time()
            
            for i, (cam_id, cap) in enumerate(self.caps.items()):
                thread = threading.Thread(target=self.grab_frame, args=(cam_id, cap, images, warped_rgba_, i))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            end_time_threading = time.time()
            
            if all(img is not None for img in images):
                start_time_merge = time.time()
                final_merged_image = ImageStitcher.get_weights_and_masks_liverun(warped_rgba_)
                end_time_merge = time.time()
                end_time_total = time.time()
                
                # print(f"Image merging time: {end_time_merge - start_time_merge:.4f} seconds")
                # print(f"Total threading execution time: {end_time_threading - start_time_threading:.4f} seconds "
                #       f"(Includes: Grab frames → Undistort & Warp frames → Store processed images)")
                print(f"Total processing time: {end_time_total - start_time_total:.4f} seconds")
                
                merged_car_image = ImageAdjuster.overlay_image_perspective(final_merged_image, self.car, self.car_dst_points)

                resized_width = self.display_width // 2
                resized_height = self.display_height // 2
                resized_images = [cv2.resize(img, (resized_width, resized_height)) for img in images]
                top_row = np.hstack((resized_images[0], resized_images[1]))
                bottom_row = np.hstack((resized_images[2], resized_images[3]))
                merged_Display_image = np.vstack((top_row, bottom_row))
                new_width = 800
                new_height = 900
                resized_final_image = cv2.resize(merged_car_image, (new_width, new_height))

                cv2.imshow("Merged Image", merged_Display_image)
                cv2.imshow("Final Merged Image", resized_final_image)
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
    car_image_path = Golf_img_Path
    processor = ImageProcessor(video_paths, car_image_path)
    processor.run()
