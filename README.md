Calibration Image
```markdown
# โหลดภาพตารางหมากรุก (เช่นไฟล์ชื่อ chessboard*.jpg)
images = glob.glob('chessboard/chessboard*.jpg')
print(images)

preview_images = []

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # หามุมของตารางหมากรุก
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        # ปรับมุมให้ละเอียด
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        
        # เก็บค่า points
        objpoints.append(objp)
        imgpoints.append(corners2)
        
        # วาดและแสดงผลมุม
        img_draw = cv2.drawChessboardCorners(img.copy(), chessboard_size, corners2, ret)
        # cv2.imshow('Detected Corners', img_draw)
        # cv2.waitKey(0)

        # รวมภาพต้นฉบับกับภาพที่มีมุมที่ detect
        combined_img = cv2.hconcat([img, img_draw])
        preview_images.append(combined_img)

for preview in preview_images:
    cv2.namedWindow('Calibration Preview', cv2.WINDOW_NORMAL)  # อนุญาตให้ปรับขนาดได้
    cv2.resizeWindow('Calibration Preview', 800, 600)  # กำหนดขนาดเป็น 600x300 px
    cv2.imshow('Calibration Preview', preview)
    cv2.waitKey(0)

if len(objpoints) > 0:
    # Calibrate
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

```

Undistort Image

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
