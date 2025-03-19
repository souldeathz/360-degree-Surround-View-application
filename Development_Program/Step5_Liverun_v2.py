import cv2
import numpy as np
import os
import time
from image_processing import LuminanceBalancer, ImageStitcher, ImageAdjuster
from param_settings import Golf_img_Path

class ImageProcessor:
    def __init__(self, dataset_path, car_image_path, display_width=800, display_height=600, map_width=1040, map_height=1191):
        self.dataset_path = dataset_path
        self.car = cv2.imread(car_image_path, cv2.IMREAD_UNCHANGED)
        self.folders = ["front", "left", "rear", "right"]
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

    def merge_images(self, images, mode='hard_overlay', alpha=0.25):
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

    def overlay_image_perspective(self, background, overlay, dst_points):
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

    def process_image(self, image, cameraID):
        yaml_filename = os.path.join('yaml', f'calibration_data_{cameraID}.yaml')
        fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
        camera_matrix = fs.getNode("camera_matrix").mat()
        dist_coeffs = fs.getNode("dist_coeffs").mat()
        resolution = fs.getNode("resolution").mat()
        H = fs.getNode("homography").mat()
        
        if cameraID == "front":
            processed_image = image
        elif cameraID == "left":
            processed_image = image
        elif cameraID == "rear":
            img_src = cv2.rotate(image, cv2.ROTATE_180)
            processed_image = img_src
        elif cameraID == "right":
            img_src = cv2.rotate(image, cv2.ROTATE_180)
            processed_image = img_src 
        else:
            processed_image = image
        
        img_src_undistorted = cv2.undistort(processed_image, camera_matrix, dist_coeffs)
        warped = cv2.warpPerspective(img_src_undistorted, H, (self.map_width, self.map_height))
        warped_rgba = cv2.cvtColor(warped, cv2.COLOR_BGR2BGRA)
        warped_rgba[np.all(warped_rgba[:, :, :3] == [0, 0, 0], axis=-1)] = [0, 0, 0, 0]
        warped_rgb = cv2.cvtColor(warped_rgba, cv2.COLOR_BGRA2BGR)
        
        return processed_image, warped_rgb
    
    def run(self):
            index = 0
            while True:
                filename = os.listdir(os.path.join(self.dataset_path, self.folders[0]))[index]
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    images = []
                    warped_rgba_ = []
                    start_time = time.time()

                    for folder in self.folders:
                        img_path = os.path.join(self.dataset_path, folder, filename)
                        img = cv2.imread(img_path)
                        if img is not None:
                            processed_img, warped_rgba = self.process_image(img, folder)
                            images.append(processed_img)
                            warped_rgba_.append(warped_rgba)
                    
                    if len(images) == 4:

                        # ตัวอย่างการใช้ get_weights_and_masks เพื่อรวมภาพ
                        final_merged_image = ImageStitcher.get_weights_and_masks(warped_rgba_)

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
                        # cv2.imshow("Merged Image with Car Overlay", merged_car_image)
                        cv2.imshow("Final Merged Image", final_merged_image)
                        cv2.waitKey(1)
                
                index += 1
                if index >= len(os.listdir(os.path.join(self.dataset_path, self.folders[0]))):
                    index = 0

if __name__ == '__main__':
    dataset_path = "../Dataset/liverun_outdoor/football"
    car_image_path = Golf_img_Path
    processor = ImageProcessor(dataset_path, car_image_path)
    processor.run()
