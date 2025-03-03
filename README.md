
```markdown
def undistort_image(img, camera_matrix, dist_coeffs):
    h, w = img.shape[:2]
    new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_cam_mtx)
    return undistorted

# ตัวอย่างโหลดภาพและ undistort
img_front = cv2.imread('front.jpg')
img_rear = cv2.imread('rear.jpg')
img_left = cv2.imread('left.jpg')
img_right = cv2.imread('right.jpg')

# สมมุติค่าที่ได้จาก calibration
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]])

dist_coeffs = np.array([k1, k2, p1, p2, k3])

img_front = undistort_image(img_front, camera_matrix, dist_coeffs)
img_rear = undistort_image(img_rear, camera_matrix, dist_coeffs)
img_left = undistort_image(img_left, camera_matrix, dist_coeffs)
img_right = undistort_image(img_right, camera_matrix, dist_coeffs)
```
