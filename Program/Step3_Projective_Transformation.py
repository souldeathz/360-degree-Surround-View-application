"""
This script performs homography estimation between an undistorted test image and a reference chessboard image.

How it works:
1. Loads camera calibration parameters (intrinsic matrix, distortion coefficients) from YAML files.
2. Reads a test image and the corresponding reference chessboard image.
3. Undistorts the test image using the calibration parameters.
4. Detects chessboard corners in both images and refines their positions for higher accuracy.
5. Checks if the detected chessboard in the test image is upside down and corrects it if necessary.
6. Computes the homography matrix using the detected corners with RANSAC to ensure robustness.
7. Saves the computed homography matrix into the corresponding YAML file.
8. Applies the homography transformation to warp the test image to align with the reference chessboard perspective.
9. Saves the warped image in PNG format, ensuring that black pixels are set to transparent.

Output:
- The script generates:
    yaml/
      ├── calibration_data_front.yaml
      ├── calibration_data_left.yaml
      ├── calibration_data_right.yaml
      ├── calibration_data_rear.yaml
    out_merged_Images/
      ├── front_warped_image.png
      ├── left_warped_image.png
      ├── right_warped_image.png
      ├── rear_warped_image.png
"""

import cv2
import numpy as np
import os

output_folder = 'out_merged_Images'  # Directory to save the warped images

def load_calibration_parameters(yaml_filename):
    """Loads camera calibration parameters (intrinsic matrix, distortion coefficients, resolution) from a YAML file."""
    fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    resolution = fs.getNode("resolution").mat()
    fs.release()
    return camera_matrix, dist_coeffs, resolution

def exCalib(img_src, img_dst, type):
    """Estimates homography between an undistorted test image and a chessboard reference image."""
    
    # Define the chessboard size (number of inner corners)
    # Note: If the chessboard has a different number of squares, adjust pattern_size accordingly.
    # (columns, rows) for inner corners
    pattern_size = (6, 4)  # For a 7x5 chessboard (inner corners 6x4)

    # Convert images to grayscale
    gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

    scale_factor = 1
    resized_gray_dst = cv2.resize(gray_dst, None, fx=scale_factor, fy=scale_factor)

    # Detect chessboard corners in the source image
    ret_src, corners_src = cv2.findChessboardCorners(gray_src, pattern_size, None)
    if not ret_src:
        print("Error: Chessboard corners not found in the source image.")
        return None

    # Detect chessboard corners in the destination (reference) image
    ret_dst, corners_dst = cv2.findChessboardCorners(resized_gray_dst, pattern_size, None)
    if not ret_dst:
        print("Error: Chessboard corners not found in the destination image.")
        return None

    corners_dst = corners_dst / scale_factor  # Adjusting for scale factor if needed

    # Refine the detected corner positions for better accuracy
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
    corners_src = cv2.cornerSubPix(gray_src, corners_src, (11, 11), (-1, -1), criteria)

    # Check if the detected chessboard in the destination image is upside down
    # This is determined by computing the cross product of the vectors formed by the top-left to top-right 
    # and top-left to bottom-left corners
    vector1 = corners_src[1] - corners_dst[0]
    vector2 = corners_src[2] - corners_dst[0]
    cross_product = np.cross(vector1, vector2)
    print("Cross Product:", cross_product)

    if cross_product > 0 and "Right" in type:
        print("Detected corners in the destination image are upside down. Rotating 180 degrees.")
        # Rotate the detected corners by 180 degrees
        corners_src = np.rot90(corners_src.reshape(pattern_size[1], pattern_size[0], 2), 2).reshape(-1, 2)

    corners_dst = cv2.cornerSubPix(gray_dst, corners_dst, (11, 11), (-1, -1), criteria)

    # Draw the detected corners on the source image
    img_src_display = img_src.copy()
    cv2.drawChessboardCorners(img_src_display, pattern_size, corners_src, ret_src)
    
    # Draw the detected corners on the destination image
    img_dst_display = img_dst.copy()
    cv2.drawChessboardCorners(img_dst_display, pattern_size, corners_dst, ret_dst)

    # Display the detected corners on both images
    cv2.imshow('Detected Corners - Source', img_src_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Detected Corners - Destination', img_dst_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Compute homography using RANSAC to ensure robustness
    homography, mask = cv2.findHomography(corners_src, corners_dst, cv2.RANSAC)

    return homography


if __name__ == "__main__":
    views = ["front", "left", "rear", "right"]
    os.makedirs(output_folder, exist_ok=True)

    for test_folder in views:
        chessboard_folder = f'chessboard/warped_chessboard_{test_folder}.png'

        yaml_filename = os.path.join('yaml', f'calibration_data_{test_folder}.yaml')
        camera_matrix, dist_coeffs, resolution = load_calibration_parameters(yaml_filename)
        print(f"Loaded Camera Matrix for {test_folder}:\n", camera_matrix)
        print(f"Loaded Distortion Coefficients for {test_folder}:\n", dist_coeffs)

        # Load the test image and apply undistortion
        img_src = cv2.imread(f'Dataset/Img_distortion_Testing/{test_folder}.jpg')
        if test_folder == 'rear':
            img_src = cv2.rotate(img_src, cv2.ROTATE_180)

        img_src_undistorted = cv2.undistort(img_src, camera_matrix, dist_coeffs)

        # Load the reference chessboard image
        img_dst = cv2.imread(chessboard_folder)

        # Compute the homography matrix
        H = exCalib(img_src_undistorted, img_dst, test_folder)
        
        # Save the homography matrix in the YAML file
        fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", camera_matrix)
        fs.write("dist_coeffs", dist_coeffs)
        fs.write("resolution", resolution)
        fs.write("homography", H)
        fs.release()
        print(f"Homography matrix saved to {yaml_filename}")
        print(f"Computed Homography for {test_folder}:\n", H)

        # Apply the homography to warp the source image to the destination view
        height, width = img_dst.shape[:2]
        warped = cv2.warpPerspective(img_src_undistorted, H, (width, height))

        # Convert black pixels (0,0,0) to transparent (0,0,0,0) for PNG saving
        warped_rgba = cv2.cvtColor(warped, cv2.COLOR_BGR2BGRA)
        warped_rgba[np.all(warped_rgba[:, :, :3] == [0, 0, 0], axis=-1)] = [0, 0, 0, 0]

        # Save the warped image
        output_filename = f'{output_folder}/{test_folder}_warped_image.png'
        cv2.imwrite(output_filename, warped_rgba)
        print(f"Warped image saved to {output_filename}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
