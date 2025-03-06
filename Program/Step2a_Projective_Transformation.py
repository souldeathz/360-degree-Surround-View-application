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

if __name__ == "__main__":

    # ตั้งค่าโฟลเดอร์และพารามิเตอร์

    # test_folder = 'front6'
    test_folder = 'Left6'    
    # test_folder = 'Rear6'

    # โหลด intrinsic parameters จากไฟล์ YAML
    yaml_filename = 'yaml/calibration_data.yaml'
    camera_matrix, dist_coeffs, resolution = load_calibration_parameters(yaml_filename)
    print("Loaded Camera Matrix:\n", camera_matrix)
    print("Loaded Distortion Coefficients:\n", dist_coeffs)

    # หากต้องการ undistort ภาพใหม่ สามารถ uncomment โค้ดด้านล่างได้
    img_test = cv2.imread(f'input_test_distortion/{test_folder}.jpg')
    cv2.imshow("Imageg Selected", img_test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    undistorted = cv2.undistort(img_test, camera_matrix, dist_coeffs)
    # if 'Left' in test_folder:
    #     undistorted = cv2.rotate(undistorted, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # ให้ผู้ใช้คลิกเลือก 4 จุดในภาพ
    clicked_points = []
    img_display = undistorted.copy()
    cv2.imshow(Title, img_display)
    cv2.setMouseCallback(Title, click_event)
    while len(clicked_points) < 4:
        cv2.waitKey(1)

    cv2.destroyWindow(Title)
    
    src_points = np.float32(clicked_points)
    print("Selected source points:", src_points)

    # กำหนด destination points สำหรับการแปลง perspective

    # if 'front' in test_folder:
    #     dst_points = np.float32([
    #         # [450, 310],
    #         # [570, 310],
    #         # [450, 410],
    #         # [570, 410]
    #         [300, 300],
    #         [520, 300],
    #         [300, 420],
    #         [520, 420]            
    #     ])
    # elif  'Left' in test_folder:
    #         dst_points = np.float32([
    #             # [310, 524],
    #             # [410, 524],
    #             [300, 300],
    #             [410, 420],                
    #             [310, 624],
    #             [410, 624]
    #     ])
    # else:
        # Add other test_folder cases here if needed
    dst_points = np.float32([
        [420, 300],  # จุดที่ 1
        [780, 300],  # จุดที่ 2
        [420, 460],  # จุดที่  3       
        [780, 460]  # จุดที่ 4  
    ])

    # คำนวณ perspective transform matrix และทำการ warp image
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(undistorted, M, (undistorted.shape[1], undistorted.shape[0]))
    print("Warped image shape:", warped_image.shape)

    # บันทึกและแสดงผลภาพที่ผ่านการแปลง
    cv2.imwrite(f'out/{test_folder}_Warped.jpg', warped_image)
    cv2.imshow('Warped Image', warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


