import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

def undistort_image(img, camera_matrix, dist_coeffs):
    h, w = img.shape[:2]
    new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_cam_mtx)
    return undistorted

# โหลดภาพอ้างอิง (Front.jpg) และภาพที่จะนำมารวม (Right.jpg)
# โหลดภาพ

chessboard_size = (6, 4)  # จำนวนมุมภายใน (width, height)
square_size = 0.2  # หน่วยเป็นเมตร (20 cm)

# เตรียม Object Points (รูปแบบ 3D)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# เก็บ Object Points และ Image Points จากทุกภาพ
objpoints = []  # 3D points
imgpoints = []  # 2D points

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

cv2.destroyAllWindows()

img_test = cv2.imread('input/Front.jpg')
# img_test = cv2.imread('input/Rear.jpg')
# img_test = cv2.imread('input/Left.jpg')
# img_test = cv2.imread('input/Right.jpg')

# ทำ Camera Calibration (ต้องมีภาพอย่างน้อย 1 ภาพที่ตรวจจับมุมได้)
if len(objpoints) > 0:
    # Calibrate
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    # Undistort ภาพตัวอย่าง

    undistorted = undistort_image(img_test, camera_matrix, dist_coeffs)
    
    # แสดงผล
    cv2.imwrite("out/Original.jpg", img_test)
    cv2.imwrite("out/Undistorted.jpg", undistorted)
    cv2.imshow('Original', img_test)
    cv2.imshow('Undistorted', undistorted)
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("ไม่พบมุมตารางหมากรุกในภาพใดๆ ไม่สามารถทำ Calibration ได้")