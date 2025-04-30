import cv2
import numpy as np
import os
import time
from image_processing import LuminanceBalancer, ImageStitcher, ImageAdjuster
from param_settings import img_car , Car_dst_points ,total_w , total_h 

# ===============================
# CONFIGURATION
# ===============================

# Select blending strategy for stitching 4-camera images:
#   - "basic":     Uses simple alpha blending with overlay (faster, lower quality)
#   - "weighted":  Uses weight map and mask for smooth transitions (slower, higher quality)
BLENDING_STRATEGY = "basic"
# BLENDING_STRATEGY = "weighted"

# ===============================
# CLASS: ImageProcessor
# ===============================

class ImageProcessor:
    def __init__(self, video_paths, img_car, display_width=800, display_height=600, map_width=1040, map_height=1191):
        # Initialize video paths and camera overlays
        self.video_paths = video_paths
        self.car = img_car
        self.caps = {key: cv2.VideoCapture(path) for key, path in video_paths.items()}
        self.display_width = display_width
        self.display_height = display_height
        self.map_width = map_width
        self.map_height = map_height
        self.car_dst_points = Car_dst_points

    def process_image(self, image, cameraID):
        """
        Load calibration (camera_matrix, distortion, homography),
        then undistort and warp the input image to bird’s eye view.
        """
        yaml_filename = os.path.join('yaml', f'calibration_data_{cameraID}.yaml')
        fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
        camera_matrix = fs.getNode("camera_matrix").mat()
        dist_coeffs = fs.getNode("dist_coeffs").mat()
        H = fs.getNode("homography").mat()
        fs.release()

        # Undistort then apply perspective warp
        img_src_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
        warped = cv2.warpPerspective(img_src_undistorted, H, (self.map_width, self.map_height))

        # Convert to transparent RGBA for overlaying
        warped_rgba = cv2.cvtColor(warped, cv2.COLOR_BGR2BGRA)
        warped_rgba[np.all(warped_rgba[:, :, :3] == [0, 0, 0], axis=-1)] = [0, 0, 0, 0]
        warped_rgb = cv2.cvtColor(warped_rgba, cv2.COLOR_BGRA2BGR)

        return image, warped_rgb

    def run(self, blending_strategy="basic"):
        """
        Main processing loop: read video, apply calibration, stitch, and overlay car.
        Press 'q' to quit.
        """
        while True:
            images = []
            warped_rgba_list = []
            start_time = time.time()

            # Read and process frame from each camera
            for cam_id, cap in self.caps.items():
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video {cam_id}. Restarting...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                processed_img, warped_rgba = self.process_image(frame, cam_id)
                images.append(processed_img)
                warped_rgba_list.append(warped_rgba)

            if len(images) == 4:
                # Merge using selected blending strategy
                if blending_strategy == "weighted":
                    final_merged_image = ImageStitcher.get_weights_and_masks(warped_rgba_list)
                else:
                    final_merged_image = ImageAdjuster.merge_images(warped_rgba_list, mode='hard_overlay', alpha=0.25)

                # Overlay car image at center of stitched view
                merged_car_image = ImageAdjuster.overlay_image_perspective(final_merged_image, self.car, self.car_dst_points)

                # Log processing time per frame set
                end_time = time.time()
                print(f"Processed time: {end_time - start_time:.2f} seconds")

                # Display all 4 raw images (2x2 grid)
                resized_images = [cv2.resize(img, (self.display_width // 2, self.display_height // 2)) for img in images]
                top_row = np.hstack((resized_images[0], resized_images[1]))
                bottom_row = np.hstack((resized_images[2], resized_images[3]))
                merged_display_image = np.vstack((top_row, bottom_row))
                cv2.imshow("Merged Image", merged_display_image)

                # Display final stitched image with car overlay
                resized_merged_car_image = cv2.resize(merged_car_image, (500, 900))
                cv2.imshow("Final Merged Image", resized_merged_car_image)

                # Quit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Cleanup
        for cap in self.caps.values():
            cap.release()
        cv2.destroyAllWindows()

# ===============================
# MAIN
# ===============================

if __name__ == '__main__':
    # Define 4 video sources
    video_folder = "../Dataset/liverun_outdoor_VDO_Day"
    # video_folder = "../Dataset/liverun_outdoor_VDO_Night_1"
    # video_folder = "../Dataset/liverun_outdoor_VDO_Night_2"   
    video_paths = {
        "front": os.path.join(video_folder, "front.mp4"),
        "left": os.path.join(video_folder, "left.mp4"),
        "rear": os.path.join(video_folder, "rear.mp4"),
        "right": os.path.join(video_folder, "right.mp4"),
    }

    # Start the processing loop with selected blending strategy
    processor = ImageProcessor(video_paths, img_car, 800, 600, total_w, total_h)
    processor.run(blending_strategy=BLENDING_STRATEGY)
