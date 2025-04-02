"""
This script generates a perspective-transformed chessboard pattern for camera calibration.

How it works:
1. Creates a white background image ( image_size_W x image_size_H pixels).
2. Defines four chessboard configurations for different views: 'front', 'left', 'Right', 'rear'.
3. Generates a 5x7 or 7x5 black-and-white chessboard based on each configuration.
4. Computes the perspective transformation matrix and applies a warp to position the chessboard correctly.
5. Adds a white border around the transformed chessboard.
6. Overlays the transformed chessboard onto the background.
7. Displays each generated chessboard for 0.5 seconds.
8. Saves the final chessboard images to the 'chessboard' directory.

Output:
- The script generates four images:
    chessboard/
      ├── warped_chessboard_front.png
      ├── warped_chessboard_left.png
      ├── warped_chessboard_Right.png
      ├── warped_chessboard_rear.png
"""
import cv2
import numpy as np
import os
from param_settings import xl, xr, yt, yb ,Car_dst_points,chessboard_config,total_w,total_h

# Set background image size (white background)
image_size_W = total_w
image_size_H = total_h
background = np.ones((image_size_H, image_size_W, 3), dtype=np.uint8) * 255


# Define the four chessboard configurations
chessboard_config_ = chessboard_config

# Create output directory if not exists
output_dir = "chessboard"
os.makedirs(output_dir, exist_ok=True)

for image_selected, config in chessboard_config_.items():
    print(f"Processing chessboard for: {image_selected}")

    # Extract configuration
    inner_dst_pts       = config["inner_dst_pts"]
    rows, cols          = config["rows"], config["cols"]
    chessboard_width    = config["chessboard_width"]
    chessboard_height   = config["chessboard_height"]

    # Calculate square size
    square_width = chessboard_width // cols
    square_height = chessboard_height // rows

    # Create the chessboard pattern
    chessboard = np.ones((chessboard_height, chessboard_width, 3), dtype=np.uint8) * 255
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:  # Alternate black and white squares
                x_start = j * square_width
                y_start = i * square_height
                cv2.rectangle(chessboard, (x_start, y_start),
                              (x_start + square_width, y_start + square_height),
                              (0, 0, 0), -1)

    # Define source points (chessboard corners)
    inner_src_pts = np.array([
        [0, 0],
        [chessboard_width, 0],
        [0, chessboard_height],
        [chessboard_width, chessboard_height]
    ], dtype=np.float32)

    # Compute perspective transformation matrix
    M = cv2.getPerspectiveTransform(inner_src_pts, inner_dst_pts)

    # Warp the chessboard onto the destination perspective
    warped_chessboard = cv2.warpPerspective(chessboard, M, (image_size_W, image_size_H),
                                            flags=cv2.INTER_NEAREST,
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=(255, 255, 255))

    # Create a white border around the chessboard
    border_thickness = 20
    cv2.rectangle(warped_chessboard,
                  (int(inner_dst_pts[0][0] - border_thickness), int(inner_dst_pts[0][1] - border_thickness)),
                  (int(inner_dst_pts[3][0] + border_thickness), int(inner_dst_pts[3][1] + border_thickness)),
                  (255, 255, 255), border_thickness)

    # Overlay the transformed chessboard onto the background
    result = background.copy()
    result = cv2.bitwise_and(result, warped_chessboard)

    # Display the result
    cv2.imshow(f"Result - {image_selected}", result)
    cv2.waitKey(500)  # Display each image for 0.5 seconds

    # Save the generated chessboard image
    output_path = os.path.join(output_dir, f'warped_chessboard_{image_selected}.png')
    cv2.imwrite(output_path, result)
    print(f"Saved: {output_path}")

cv2.destroyAllWindows()
print("All chessboard images processed successfully.")
