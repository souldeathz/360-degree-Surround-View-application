"""
This script performs camera calibration using chessboard images and applies distortion correction to test images.

How it works:
1. Loads chessboard images from the specified folder to extract calibration points.
2. Defines chessboard size and prepares 3D object points.
3. If `Cal_M = True`, it performs camera calibration using detected chessboard corners.
4. If `Cal_M = False`, it loads predefined intrinsic camera parameters.
5. Reads test images from four different views: 'front', 'left', 'Right', 'rear'.
6. Applies undistortion to the test images using the calibrated camera matrix and distortion coefficients.
7. Saves the original and undistorted images to the 'out_undistorted_Images' folder.
8. Stores the calibration parameters in OpenCV YAML format in the 'yaml' directory.

Output:
- The script generates:
    yaml/
      ├── calibration_data_front.yaml
      ├── calibration_data_left.yaml
      ├── calibration_data_right.yaml
      ├── calibration_data_rear.yaml
    out_undistorted_Images/
      ├── front_original.jpg
      ├── front_undistorted.jpg
      ├── left_original.jpg
      ├── left_undistorted.jpg
      ├── right_original.jpg
      ├── right_undistorted.jpg
      ├── rear_original.jpg
      ├── rear_undistorted.jpg
"""
import os
import numpy as np
import cv2
import glob


def get_intrinsic_matrix(fx, fy, cx, cy):
    """Returns the intrinsic camera matrix."""
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]])


# Defind dataset path 
Dataset_path = '../Dataset/Img_distortion_Testing/'

# Define chessboard size and square size
chessboard_size = (6, 4)  # Number of inner corners (width, height)
square_size = 0.2        # Square size in meters

# Prepare object points (3D points pattern)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

objpoints = []  # List to store 3D object points
imgpoints = []  # List to store 2D image points

# Set calibration folder
calibration_folder = 'Calibration_Img'

# Get all calibration images from the folder
images = glob.glob(f'chessboard/{calibration_folder}/chessboard*.jpg')
print("Found images:", images)

# Set calibration mode
Cal_M = False  # True = Perform new calibration, False = Use pre-calibrated values

camera_matrix = None
dist_coeffs = None
resolution = None

if Cal_M:
    # Perform camera calibration using chessboard images
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            # Refine corner detection for better accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
    
    if len(objpoints) > 0:
        # Perform camera calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        resolution = np.array(gray.shape[::-1], dtype=np.int32)  # (width, height)
        print("Calibrated Camera Matrix:")
        print(camera_matrix)
        print("Distortion Coefficients:")
        print(dist_coeffs)
    else:
        raise ValueError("No suitable calibration images found.")
else:
    # Use pre-calibrated intrinsic parameters
    fx, fy = 536.874764466575, 538.6813722963437
    cx, cy = 637.9072910606346, 328.6645901880612
    camera_matrix = get_intrinsic_matrix(fx, fy, cx, cy)
    dist_coeffs = np.array([-0.29384105, 0.08857583, 0.0017715, -0.00090177, -0.01231684])
    resolution = np.array([1280, 720], dtype=np.int32)  # Expected image resolution

# Create output directories if they do not exist
os.makedirs('yaml', exist_ok=True)
os.makedirs('out_undistorted_Images', exist_ok=True)

# Define multiple test folders to process
test_folders = ['front', 'left', 'right', 'rear']

for test_folder in test_folders:
    print(f"\nProcessing test folder: {test_folder}")

    # Load test image for undistortion
    img_test = cv2.imread(f'{Dataset_path}{test_folder}.jpg')

    if img_test is not None:
        # Undistort the test image
        undistorted = cv2.undistort(img_test, camera_matrix, dist_coeffs)

        # Save original and undistorted test images
        cv2.imwrite(f'out_undistorted_Images/{test_folder}_original.jpg', img_test)
        cv2.imwrite(f'out_undistorted_Images/{test_folder}_undistorted.jpg', undistorted)

        # Display original and undistorted images
        cv2.imshow(f'Original Image - {test_folder}', img_test)
        cv2.imshow(f'Undistorted Image - {test_folder}', undistorted)
        cv2.waitKey(1000)  # Display each image for 1 second before moving to the next
        cv2.destroyAllWindows()
    else:
        print(f"Test image not found for {test_folder}")

cv2.destroyAllWindows()

# Save calibration data in OpenCV YAML format
for view in ["front", "left", "rear", "right"]:
    yaml_filename = os.path.join('yaml', f'calibration_data_{view}.yaml')
    fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coeffs", dist_coeffs)
    fs.write("resolution", resolution)
    fs.release()
    print(f"Saved calibration data to {yaml_filename}")

# Save rotation and translation vectors if available (when Cal_M = True)
if Cal_M and 'rvecs' in locals() and 'tvecs' in locals():
    fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_WRITE)
    fs.write("rvecs", np.array(rvecs))
    fs.write("tvecs", np.array(tvecs))
    fs.release()

print("\nAll test images processed successfully.")