import cv2
import os
import numpy as np
import time
import threading
from image_processing import LuminanceBalancer, ImageStitcher, ImageAdjuster
from param_settings import img_car , Car_dst_points ,total_w , total_h 

class ImageProcessor:
    def __init__(self, video_paths, img_car, display_width=800, display_height=600,  map_width=total_w, map_height=total_h):
        self.video_paths = video_paths
        self.car = img_car
        self.caps = {key: cv2.VideoCapture(path) for key, path in video_paths.items()}
        self.display_width = display_width
        self.display_height = display_height
        self.map_width = map_width
        self.map_height = map_height
        self.car_dst_points = Car_dst_points
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
        start_time = time.time()
        calib = self.calibration_data[cameraID]
        camera_matrix, dist_coeffs, H = calib["camera_matrix"], calib["dist_coeffs"], calib["homography"]

        img_src_undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)

        warped = cv2.warpPerspective(img_src_undistorted, H, (self.map_width, self.map_height))
        proc_time = time.time() - start_time
        return img_src_undistorted, warped, proc_time

    def grab_frame(self, cam_id, cap, images, warped_rgba_, processing_times, index):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        undistorted, warped, proc_time = self.process_image(frame, cam_id)
        images[index] = undistorted
        warped_rgba_[index] = warped
        processing_times[index] = proc_time

    def run(self):
        cam_names = ["front", "left", "rear", "right"]
        while True:
            start_time_total = time.time()
            images = [None] * len(self.caps)
            warped_rgba_ = [None] * len(self.caps)
            processing_times = [0.0] * len(self.caps)
            threads = []

            for i, (cam_id, cap) in enumerate(self.caps.items()):
                thread = threading.Thread(target=self.grab_frame, args=(cam_id, cap, images, warped_rgba_, processing_times, i))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            if all(img is not None for img in images):
                final_merged_image = ImageStitcher.get_weights_and_masks_liverun(warped_rgba_)
                merged_car_image = ImageAdjuster.overlay_image_perspective(final_merged_image.copy(), self.car, self.car_dst_points)

                resized_width = self.display_width // 2
                resized_height = self.display_height // 2
                resized_images = [cv2.resize(img, (resized_width, resized_height)) for img in images]

                for i, img in enumerate(resized_images):
                    text = f"{cam_names[i]}: {processing_times[i]*1000:.1f} ms"
                    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                top_row = np.hstack((resized_images[0], resized_images[1]))
                bottom_row = np.hstack((resized_images[2], resized_images[3]))
                merged_Display_image = np.vstack((top_row, bottom_row))

                total_proc_time = (time.time() - start_time_total) * 1000
                total_text = f"Total Processing: {total_proc_time:.1f} ms"

                resized_final_image = cv2.resize(merged_car_image, (800, 900))
                cv2.putText(resized_final_image, total_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                print(total_text)

                cv2.imshow("Merged Image", merged_Display_image)
                cv2.imshow("Final Merged Image", resized_final_image)
                if cv2.waitKey(1) == ord('q'):
                    break

        for cap in self.caps.values():
            cap.release()
        cv2.destroyAllWindows()

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
    processor = ImageProcessor(video_paths, img_car)
    processor.run()
