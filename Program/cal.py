import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import yaml

# ฟังก์ชัน callback เมื่อมีการคลิกเมาส์
clicked_points = []

def click_event(event, x, y, flags, param):
    global clicked_points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        # บันทึกจุดที่คลิก
        clicked_points.append((x, y))
        # วาดวงกลมเพื่อแสดงตำแหน่งที่เลือก
        cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4 Points", img_display)

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


Input_folder = 'front'
# Input_folder = 'left'
# Input_folder = 'Right'
# Input_folder = 'rear'
# โหลดภาพตารางหมากรุก (เช่นไฟล์ชื่อ chessboard*.jpg)

images = glob.glob('chessboard/'+Input_folder+'/chessboard*.jpg')
print(images)

preview_images = []

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # หามุมของตารางหมากรุก
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        # ปรับมุมให้ละเอียด
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
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

# for preview in preview_images:
#     cv2.namedWindow('Calibration Preview', cv2.WINDOW_NORMAL)  # อนุญาตให้ปรับขนาดได้
#     cv2.resizeWindow('Calibration Preview', 800, 600)  # กำหนดขนาดเป็น 600x300 px
#     cv2.imshow('Calibration Preview', preview)
#     cv2.waitKey(0)  

cv2.destroyAllWindows()

# Input_folder = 'front2'
# Input_folder = 'left2'
# Input_folder = 'Right2'
Input_folder = 'Rear2'

img_test = cv2.imread('input_test_distortion/'+Input_folder+'.jpg')

# ทำ Camera Calibration (ต้องมีภาพอย่างน้อย 1 ภาพที่ตรวจจับมุมได้)
if len(objpoints) > 0:
    # Calibrate
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    # Undistort ภาพตัวอย่าง

    undistorted = undistort_image(img_test, camera_matrix, dist_coeffs)


    # แสดงผล
    cv2.imwrite("out/"+Input_folder+".jpg", img_test)
    cv2.imwrite("out/"+Input_folder+"_Undistorted.jpg", undistorted)
    cv2.imshow('Original', img_test)
    cv2.imshow('Undistorted', undistorted)
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_display = undistorted.copy()
    cv2.imshow("Select 4 Points", img_display)
    cv2.setMouseCallback("Select 4 Points", click_event)

    # รอให้ผู้ใช้คลิกเลือกครบ 4 จุด
    while len(clicked_points) < 4:
        cv2.waitKey(1)

    cv2.destroyWindow("Select 4 Points")

    # แปลงจุดที่เลือกเป็น np.float32
    src_points = np.float32(clicked_points)
    print("Selected source points:", src_points)

    # size_W = 2560
    # size_H = 1440
    dst_points = np.float32([
        [420, 300],  # จุดที่ 1
        [780, 300],  # จุดที่ 2
        [420, 460],  # จุดที่  3       
        [780, 460]  # จุดที่ 4    
    #     [size_W/4, size_H/4],          # จุดที่ 1
    #     [(size_W/4)*3, size_H/4],       # จุดที่ 2
    #     [size_W/4, (size_H/4)*3],       # จุดที่ 3
    #     [(size_W/4)*3, (size_H/4)*3]    # จุดที่ 4
    ])

    # scale = 0.5  # ลดขนาดลงครึ่งหนึ่ง
    # dst_points = np.float32([
    #     [420 * scale, 300 * scale],  # จุดที่ 1
    #     [780 * scale, 300 * scale],  # จุดที่ 2
    #     [420 * scale, 460 * scale],  # จุดที่ 3
    #     [780 * scale, 460 * scale]   # จุดที่ 4
    # ])

    print("Selected source points:" , dst_points)

    
    # คำนวณ projective transformation matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # ทำ projective transformation
    warped_image = cv2.warpPerspective(undistorted, M, (img_test.shape[1], img_test.shape[0]))
    print(f"Projected image size: {warped_image.shape}")
    # แสดงผล
    cv2.imwrite("out/"+Input_folder+"_Warped.jpg", warped_image)
    cv2.imshow('Warped', warped_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("ไม่พบมุมตารางหมากรุกในภาพใดๆ ไม่สามารถทำ Calibration ได้")

# สร้างโฟลเดอร์ yaml ถ้ายังไม่มี
os.makedirs('yaml', exist_ok=True)

# กำหนดชื่อไฟล์ yaml ตาม Input_folder
yaml_filename = os.path.join('yaml', Input_folder + '.yaml')

# สร้าง dictionary สำหรับเก็บข้อมูล calibration
calibration_data = {
    'camera_matrix': camera_matrix.tolist(),
    'dist_coeffs': dist_coeffs.tolist(),
    'rvecs': [rvec.tolist() for rvec in rvecs],
    'tvecs': [tvec.tolist() for tvec in tvecs]
}

# บันทึกข้อมูล calibration ลงในไฟล์ yaml
with open(yaml_filename, 'w') as yaml_file:
    yaml.dump(calibration_data, yaml_file)

print(f"Calibration data saved to {yaml_filename}")


# สร้างโฟลเดอร์สำหรับเก็บไฟล์ YAML
os.makedirs('yaml', exist_ok=True)

# กำหนดชื่อไฟล์ YAML ตาม `Input_folder`
Input_folder = "calibration_data"  # ต้องกำหนดค่าตัวแปรนี้
yaml_filename = os.path.join('yaml', Input_folder + '.yaml')

# สร้างตัวอย่างค่าของ calibration parameters
# ดึงขนาดของภาพ (ความสูง, ความกว้าง)
height, width = img_test.shape[:2]
# เก็บค่าขนาดภาพใน numpy array
resolution = np.array([width, height], dtype=np.int32)

# ใช้ OpenCV FileStorage เพื่อบันทึกในรูปแบบ OpenCV YAML
fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_WRITE)
fs.write("camera_matrix", camera_matrix)
fs.write("dist_coeffs", dist_coeffs)
fs.write("resolution", resolution)
fs.release()
