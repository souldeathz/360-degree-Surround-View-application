import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# ฟังก์ชัน callback เมื่อมีการคลิกเมาส์ (ไม่ถูกใช้งานในโค้ดนี้)
clicked_points = []
img_display = None

def click_event(event, x, y, flags, param):
    global clicked_points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4 Points", img_display)

def get_intrinsic_matrix(fx, fy, cx, cy):
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]])

# ตั้งค่าขนาดกระดานหมากรุกและขนาดช่อง
chessboard_size = (6, 4)  # จำนวนมุมภายใน (width, height)
square_size = 0.2        # หน่วยเป็นเมตร

# เตรียม Object Points (รูปแบบ 3D)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

objpoints = []  # รายการเก็บ 3D points
imgpoints = []  # รายการเก็บ 2D points

# ตั้งค่าโฟลเดอร์และพารามิเตอร์
calibration_folder = 'Cal_V2'  # โฟลเดอร์ภาพคาลิเบรต

# test_folder = 'Left5'          # โฟลเดอร์ภาพทดสอบ
test_folder = 'front6'
# test_folder = 'left6'
# test_folder = 'Right6'
# test_folder = 'Rear6'
# อ่านภาพตัวอย่างเพื่อทดสอบ undistort
img_test = cv2.imread(f'input_test_distortion/{test_folder}.jpg')

images = glob.glob(f'chessboard/{calibration_folder}/chessboard*.jpg')
print("Found images:", images)


# ตั้งค่าสถานะการคาลิเบรต
Benjamas_param_bool = True 
Cal_M = False    # True = ใช้ภาพคาลิเบรต, False = ใช้ค่าคงที่จากคาลิเบรตครั้งก่อน

camera_matrix = None
dist_coeffs = None
resolution = None

if Benjamas_param_bool:
    if Cal_M:
        # คาลิเบรตจากภาพถ่าย
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners2)
        
        if len(objpoints) > 0:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )
            resolution = np.array(gray.shape[::-1], dtype=np.int32)  # (width, height)
            print("Calibrated Camera Matrix:")
            print(camera_matrix)
            print("Distortion Coefficients:")
            print(dist_coeffs)
        else:
            raise ValueError("ไม่พบภาพคาลิเบรตที่เหมาะสม")
    else:
        # ใช้ค่าคงที่จากคาลิเบรตครั้งก่อน
        fx, fy = 536.874764466575, 538.6813722963437
        cx, cy = 637.9072910606346, 328.6645901880612
        camera_matrix = get_intrinsic_matrix(fx, fy, cx, cy)
        dist_coeffs = np.array([-0.29384105, 0.08857583, 0.0017715, -0.00090177, -0.01231684])
        resolution = np.array([1280, 720], dtype=np.int32)  # ตั้งค่าขนาดภาพตามที่คาดไว้
else:
    fx, fy = 528.94, 530.80
    cx, cy = 633.92, 343.28
    camera_matrix = get_intrinsic_matrix(fx, fy, cx, cy)
    dist_coeffs = np.array([0.0169, -0.2141, 0.3973, -0.2529])
    resolution = np.array([1280, 720], dtype=np.int32)

# Undistort ภาพทดสอบ
if img_test is not None:
    undistorted = cv2.undistort(img_test, camera_matrix, dist_coeffs)
else:
    raise FileNotFoundError("ไม่พบไฟล์ภาพทดสอบ")

# สร้างโฟลเดอร์เอาต์พุต
os.makedirs('yaml', exist_ok=True)
os.makedirs('out', exist_ok=True)

# บันทึกข้อมูลคาลิเบรตในรูปแบบ OpenCV YAML
yaml_filename = os.path.join('yaml', f'calibration_data.yaml')
fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_WRITE)
fs.write("camera_matrix", camera_matrix)
fs.write("dist_coeffs", dist_coeffs)
fs.write("resolution", resolution)

# บันทึก rvecs และ tvecs หากมีค่า (เมื่อ Cal_M = True)
if Cal_M and 'rvecs' in locals() and 'tvecs' in locals():
    fs.write("rvecs", np.array(rvecs))
    fs.write("tvecs", np.array(tvecs))

fs.release()

print(f"บันทึกข้อมูลคาลิเบรตลงใน {yaml_filename}")

# บันทึกภาพผลลัพธ์
cv2.imwrite(f'out/{test_folder}_original.jpg', img_test)
cv2.imwrite(f'out/{test_folder}_undistorted.jpg', undistorted)

# แสดงผลภาพ
cv2.imshow('Original Image', img_test)
cv2.imshow('Undistorted Image', undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()