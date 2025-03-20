import cv2
import numpy as np
import os
import time
from image_processing import LuminanceBalancer, ImageStitcher, ImageAdjuster
from param_settings import img_car , Car_dst_points ,total_w , total_h 

class ImageProcessor:
    def __init__(self, video_paths, img_car, display_width=800, display_height=600, map_width=1040, map_height=1191):
        self.video_paths = video_paths
        self.car = img_car
        self.caps = {key: cv2.VideoCapture(path) for key, path in video_paths.items()}
        self.display_width = display_width
        self.display_height = display_height
        self.map_width = map_width
        self.map_height = map_height
        self.car_dst_points = Car_dst_points
    
    def process_image(self, image, cameraID):
        yaml_filename = os.path.join('yaml', f'calibration_data_{cameraID}.yaml')
        fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
        camera_matrix = fs.getNode("camera_matrix").mat()
        dist_coeffs = fs.getNode("dist_coeffs").mat()
        H = fs.getNode("homography").mat()
        
        if cameraID in ["rear", "right"]:
            image = cv2.rotate(image, cv2.ROTATE_180)
        
        img_src_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
        warped = cv2.warpPerspective(img_src_undistorted, H, (self.map_width, self.map_height))
        warped_rgba = cv2.cvtColor(warped, cv2.COLOR_BGR2BGRA)
        warped_rgba[np.all(warped_rgba[:, :, :3] == [0, 0, 0], axis=-1)] = [0, 0, 0, 0]
        warped_rgb = cv2.cvtColor(warped_rgba, cv2.COLOR_BGRA2BGR)
        
        return image, warped_rgb
    
    def run(self):
        while True:
            images = []
            warped_rgba_ = []
            start_time = time.time()

            for cam_id, cap in self.caps.items():
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video {cam_id}. Restarting...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video
                    continue

                processed_img, warped_rgba = self.process_image(frame, cam_id)
                images.append(processed_img)
                warped_rgba_.append(warped_rgba)
            
            if len(images) == 4:
                final_merged_image = ImageStitcher.get_weights_and_masks(warped_rgba_)
                merged_car_image = ImageAdjuster.overlay_image_perspective(final_merged_image, self.car, self.car_dst_points)
                end_time = time.time()
                process_time = end_time - start_time
                print(f"Processed time in {process_time:.2f} seconds")

                resized_width = self.display_width // 2
                resized_height = self.display_height // 2
                resized_images = [cv2.resize(img, (resized_width, resized_height)) for img in images]
                top_row = np.hstack((resized_images[0], resized_images[1]))
                bottom_row = np.hstack((resized_images[2], resized_images[3]))
                merged_Display_image = np.vstack((top_row, bottom_row))
                
                cv2.imshow("Merged Image", merged_Display_image)
                cv2.imshow("Final Merged Image", merged_car_image)
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
    processor = ImageProcessor(video_paths, img_car,800,600,total_w , total_h)
    processor.run()
