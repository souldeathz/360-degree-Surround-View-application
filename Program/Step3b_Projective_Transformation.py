import cv2
import numpy as np
import os


def load_calibration_parameters(yaml_filename):
    fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    resolution = fs.getNode("resolution").mat()
    fs.release()
    return camera_matrix, dist_coeffs, resolution

def exCalib(img_src, img_dst,type):
    # กำหนดขนาดของ chessboard (จำนวน inner corners)
    # หมายเหตุ: หาก chessboard ของคุณมีจำนวนตารางต่างจากนี้ ให้ปรับ pattern_size ตาม
    # (columns, rows) ของ inner corners
    pattern_size = (6, 4)  # สำหรับ 7x5 ตาราง (inner corners 6x4)

    # แปลงภาพเป็น grayscale
    gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

    scale_factor = 1
    resized_gray_dst = cv2.resize(gray_dst, None, fx=scale_factor, fy=scale_factor)

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

    # ตรวจสอบว่ามุมที่ตรวจพบในภาพปลายทางกลับหัวหรือไม่
    # โดยการคำนวณ cross product ของเวกเตอร์ที่เกิดจากมุมบนซ้ายไปมุมบนขวา และมุมบนซ้ายไปมุมล่างซ้าย
    vector1 = corners_src[1] - corners_dst[0]
    vector2 = corners_src[2] - corners_dst[0]
    cross_product = np.cross(vector1, vector2)
    print("Cross Product:", cross_product)

    if cross_product > 0 and "Right" in type:
        print("Detected corners in the destination image are upside down. Rotating 180 degrees.")
        # หมุนมุมที่ตรวจพบในภาพปลายทาง 180 องศา
        corners_src = np.rot90(corners_src.reshape(pattern_size[1], pattern_size[0], 2), 2).reshape(-1, 2)

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
    test_folder = 'front7'
    chessboard_folder = 'chessboard/warped_chessboard_front.png'
    # test_folder = 'Left7'   
    # chessboard_folder = 'chessboard/warped_chessboard_left.png' 
    # test_folder = 'Rear7'
    # chessboard_folder = 'chessboard/warped_chessboard_rear.png' 
    # test_folder = 'Right7'
    # chessboard_folder = 'chessboard/warped_chessboard_right.png' 

    # โหลด intrinsic parameters จากไฟล์ YAML
    view = None
    if 'front' in test_folder.lower():
        view = 'front'
    elif 'left' in test_folder.lower():
        view = 'left'
    elif 'rear' in test_folder.lower():
        view = 'rear'
    elif 'right' in test_folder.lower():
        view = 'right'

    yaml_filename = []
    if view in ["front", "left", "rear", "right"]:
        yaml_filename = os.path.join('yaml', f'calibration_data_{view}.yaml')
        camera_matrix, dist_coeffs, resolution = load_calibration_parameters(yaml_filename)
        print("Loaded Camera Matrix:\n", camera_matrix)
        print("Loaded Distortion Coefficients:\n", dist_coeffs)
    else:
        print(f"Error: Unsupported view '{view}'")

    # หากต้องการ undistort ภาพใหม่ สามารถ uncomment โค้ดด้านล่างได้
    img_src = cv2.imread(f'input_test_distortion/{test_folder}.jpg')
    if 'Rear' in test_folder:
        img_src = cv2.rotate(img_src, cv2.ROTATE_180)

    img_src_undistorted = cv2.undistort(img_src, camera_matrix, dist_coeffs)

    cv2.imshow("Image Selected", img_src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img_dst = cv2.imread(chessboard_folder)


    cv2.imshow("Imageg Selected", img_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    H = exCalib(img_src_undistorted, img_dst, test_folder)
    
    # Save the homography matrix to the same YAML file
    fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    resolution = fs.getNode("resolution").mat()
    fs.release()

    fs = cv2.FileStorage(yaml_filename, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coeffs", dist_coeffs)
    fs.write("resolution", resolution)
    fs.write("homography", H)
    fs.release()
    print(f"Homography matrix saved to {yaml_filename}")
    print("Computed Homography:\n", H)

    output_folder = 'out2'
    # (Optional) apply the homography to warp the source image to the destination view
    height, width = img_dst.shape[:2]
    print("Destination Image Size:", width, height)
    warped = cv2.warpPerspective(img_src_undistorted, H, (width, height))
    

    # Save the warped image to the folder 'out2'
    # output_folder = 'out2'
    # output_filename = f'{output_folder}/{test_folder}_warped_image.jpg'
    # cv2.imwrite(output_filename, warped)
    # print(f"Warped image saved to {output_filename}")
    # cv2.imshow("Warped Image", warped)


        # Convert black pixels (0, 0, 0) to transparent (0, 0, 0, 0) for PNG saving
    warped_rgba = cv2.cvtColor(warped, cv2.COLOR_BGR2BGRA)
    warped_rgba[np.all(warped_rgba[:, :, :3] == [0, 0, 0], axis=-1)] = [0, 0, 0, 0]
    output_filename = f'{output_folder}/{test_folder}_warped_image.png'
    cv2.imwrite(output_filename, warped_rgba)
    print(f"Warped image saved to {output_filename}")
    cv2.imshow("Warped Image", warped_rgba)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
