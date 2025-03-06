import cv2
import numpy as np

def merge_images(images, mode='hard_overlay', alpha=0.25):
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

def overlay_image_perspective(background, overlay, dst_points):
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

# Load images
front = cv2.imread('out2/front6_warped_image.png')
left = cv2.imread('out2/Left6_warped_image.png')
rear = cv2.imread('out2/Rear6_warped_image.png')
right = cv2.imread('out2/Right6_warped_image.png')
car = cv2.imread('images/golf_car.png', cv2.IMREAD_UNCHANGED)  # Load car image with alpha channel

# Image overlay order (last image in the list will be on top)
images = [front, left, rear, right]

# Choose image blending mode ['hard_overlay', 'alpha_blend']
# mode = 'alpha_blend'
mode = 'hard_overlay'
result = merge_images(images, mode=mode, alpha=0.25)

# Define destination points for perspective transformation
dst_points = np.float32([
    [465, 465],  # Point 1
    [575, 465],  # Point 2
    [465, 685],  # Point 3
    [575, 685]   # Point 4
])

# Overlay car image at specified coordinates
result = overlay_image_perspective(result, car, dst_points)

# Save and display result
cv2.imwrite(f'out2/merged_result_{mode}.png', result)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()