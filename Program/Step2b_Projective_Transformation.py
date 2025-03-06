import cv2
import numpy as np

Title = "Please select 4 points in the following order: Top-Left, Top-Right, Bottom-Left, Bottom-Right"  

def click_event(event, x, y, flags, param):
    global clicked_points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        # บันทึกจุดที่คลิก
        clicked_points.append((x, y))
        # วาดวงกลมเพื่อแสดงตำแหน่งที่เลือก
        cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow(Title, img_display)

def load_calibration_parameters(yaml_filename):
    fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    resolution = fs.getNode("resolution").mat()
    fs.release()
    return camera_matrix, dist_coeffs, resolution

def exCalib(img_src, img_dst):
    # กำหนดขนาดของ chessboard (จำนวน inner corners)
    # หมายเหตุ: หาก chessboard ของคุณมีจำนวนตารางต่างจากนี้ ให้ปรับ pattern_size ตาม
    # (columns, rows) ของ inner corners
    pattern_size = (6, 4)  # สำหรับ 7x5 ตาราง (inner corners 6x4)

    # แปลงภาพเป็น grayscale
    gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

    scale_factor = 3
    resized_gray_dst = cv2.resize(gray_src, None, fx=scale_factor, fy=scale_factor)

    # ค้นหา corner ของ chessboard ในภาพต้นทาง
    ret_src, corners_src = cv2.findChessboardCorners(gray_src, pattern_size, None)
    if not ret_src:
        print("Error: Chessboard corners not found in src")
        return None

    # ค้นหา corner ของ chessboard ในภาพปลายทาง
    ret_dst, corners_dst = cv2.findChessboardCorners(resized_gray_dst, pattern_size, None)
    if not  ret_dst:
        print("Error: Chessboard corners not found in dst.")
        return None

    corners_dst = corners_dst / scale_factor

    # ปรับแต่งตำแหน่ง corner ให้แม่นยำยิ่งขึ้น
    # ปรับความละเอียดของตำแหน่ง corners ให้แม่นยำขึ้น
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
    corners_src = cv2.cornerSubPix(gray_src, corners_src, (11, 11), (-1, -1), criteria)
    corners_dst = cv2.cornerSubPix(gray_dst, corners_dst, (11, 11), (-1, -1), criteria)
    
        # วาดมุมที่ตรวจพบบนภาพต้นทาง
    img_src_display = img_src.copy()
    cv2.drawChessboardCorners(img_src_display, pattern_size, corners_src, ret_src)
    
    # วาดมุมที่ตรวจพบบนภาพปลายทาง
    img_dst_display = img_dst.copy()
    cv2.drawChessboardCorners(img_dst_display, pattern_size, corners_dst, ret_dst)

    # แสดงผลภาพ
    cv2.imshow('Detected Corners - Source', img_src_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Detected Corners - Destination', img_dst_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # คำนวณ homography โดยใช้ RANSAC เพื่อความทนทานต่อ outlier
    homography, mask = cv2.findHomography(corners_src, corners_dst, cv2.RANSAC)

    return homography


if __name__ == "__main__":

    # ตั้งค่าโฟลเดอร์และพารามิเตอร์

    #src
    test_folder = 'front6'
    # test_folder = 'Left6'    
    # test_folder = 'Rear6'
    
    #dst
    chessboard_folder = 'chessboard/warped_chessboard_layout2.png'

    # โหลด intrinsic parameters จากไฟล์ YAML
    yaml_filename = 'yaml/calibration_data.yaml'
    camera_matrix, dist_coeffs, resolution = load_calibration_parameters(yaml_filename)
    print("Loaded Camera Matrix:\n", camera_matrix)
    print("Loaded Distortion Coefficients:\n", dist_coeffs)

    # หากต้องการ undistort ภาพใหม่ สามารถ uncomment โค้ดด้านล่างได้
    img_src = cv2.imread(f'input_test_distortion/{test_folder}.jpg')
    img_src_undistorted = cv2.undistort(img_src, camera_matrix, dist_coeffs)
    cv2.imshow("Imageg Selected", img_src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img_dst = cv2.imread(chessboard_folder)
    cv2.imshow("Imageg Selected", img_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    H = exCalib(img_src_undistorted, img_dst)
    print("Computed Homography:\n", H)

    # (Optional) apply the homography to warp the source image to the destination view
    height, width = img_dst.shape[:2]
    warped = cv2.warpPerspective(img_src_undistorted, H, (width, height))
    cv2.imshow("Warped Image", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
