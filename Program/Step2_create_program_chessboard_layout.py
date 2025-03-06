import cv2
import numpy as np

# สร้างภาพพื้นหลังขนาด 1040x1191 สีขาว
image_size_W = 1040
image_size_H = 1191
image = np.ones((image_size_H, image_size_W, 3), dtype=np.uint8) * 255

# กำหนดพิกัดที่ต้องการให้มุมกระดานหมากรุกอยู่ (inner_dst_pts)
# image_selected = 'front'
# inner_dst_pts = np.array([
#     [450, 310],  # มุมซ้ายบน
#     [590, 310],  # มุมขวาบน
#     [450, 410],  # มุมซ้ายล่าง
#     [590, 410]   # มุมขวาล่าง
# ], dtype=np.float32)
# rows, cols = 5, 7
# chessboard_width = 140
# chessboard_height = 100

# image_selected = 'left'
# inner_dst_pts = np.array([
#     [310, 524],  # มุมซ้ายบน
#     [410, 524],  # มุมขวาบน
#     [310, 664],  # มุมซ้ายล่าง
#     [410, 664]   # มุมขวาล่าง
# ], dtype=np.float32)
# rows, cols = 7, 5
# chessboard_width = 100
# chessboard_height = 140

image_selected = 'Right'
inner_dst_pts = np.array([
    [630, 524],  # มุมซ้ายบน
    [730, 524],  # มุมขวาบน
    [630, 664],  # มุมซ้ายล่าง
    [730, 664]   # มุมขวาล่าง
], dtype=np.float32)
rows, cols = 7, 5
chessboard_width = 100
chessboard_height = 140

# image_selected = 'rear'
# inner_dst_pts = np.array([
#     [450, 781],  # มุมซ้ายบน
#     [590, 781],  # มุมขวาบน
#     [450, 891],  # มุมซ้ายล่าง
#     [590, 891]   # มุมขวาล่าง
# ], dtype=np.float32)
# rows, cols = 5, 7
# chessboard_width = 140
# chessboard_height = 100

# สร้างกระดานหมากรุกขนาด 140x100 (7x5 ช่อง)

square_width = chessboard_width // cols
square_height = chessboard_height // rows

chessboard = np.ones((chessboard_height, chessboard_width, 3), dtype=np.uint8) * 255
for i in range(rows):
    for j in range(cols):
        if (i + j) % 2 == 0:
            x_start = j * square_width
            y_start = i * square_height
            cv2.rectangle(chessboard, (x_start, y_start),
                          (x_start + square_width, y_start + square_height),
                          (0, 0, 0), -1)

# กำหนดจุดมุมต้นทาง (inner_src_pts) ของกระดานหมากรุก
inner_src_pts = np.array([
    [0, 0],
    [chessboard_width, 0],
    [0, chessboard_height],
    [chessboard_width, chessboard_height]
], dtype=np.float32)

# คำนวณเมทริกซ์การแปลง Perspective
M = cv2.getPerspectiveTransform(inner_src_pts, inner_dst_pts)

# ทำการ Warp กระดานหมากรุกลงในภาพ
warped_chessboard = cv2.warpPerspective(chessboard, M, (image_size_W, image_size_H),
                                        flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(255, 255, 255))

# สร้างขอบสีขาวรอบกระดานในภาพปลายทาง
border_thickness = 20
cv2.rectangle(warped_chessboard,
              (int(inner_dst_pts[0][0] - border_thickness), int(inner_dst_pts[0][1] - border_thickness)),
              (int(inner_dst_pts[3][0] + border_thickness), int(inner_dst_pts[3][1] + border_thickness)),
              (255, 255, 255), border_thickness)
              

# นำกระดานที่แปลงแล้ววางลงบนภาพพื้นหลัง
result = image.copy()
result = cv2.bitwise_and(result, warped_chessboard)

# แสดงผลและบันทึกภาพ
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f'chessboard/warped_chessboard_{image_selected}.png', result)