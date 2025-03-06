import cv2
import numpy as np

# สร้างภาพพื้นหลังขนาด 700x700 สีขาว
image_size = 1000
image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

# กำหนดพิกัดที่ต้องการให้กระดานหมากรุกอยู่
dst_pts = np.array([
    # [450-200, 310],  # มุมซ้ายบน
    # [570-200, 310],  # มุมขวาบน
    # [450-200, 410],  # มุมซ้ายล่าง
    # [570-200, 410]   # มุมขวาล่าง
    [250-200, 110],  # มุมซ้ายบน
    [970-200, 110],  # มุมขวาบน
    [250-200, 610],  # มุมซ้ายล่าง
    [970-200, 610]   # มุมขวาล่าง
], dtype=np.float32)

# dst_pts = np.array([
#     [450+300, 310],  # มุมซ้ายบน
#     [570+300, 310],  # มุมขวาบน
#     [450+300, 410],  # มุมซ้ายล่าง
#     [570+300, 410]   # มุมขวาล่าง
# ], dtype=np.float32)

# กำหนดขนาดของกระดานหมากรุก (ก่อนแปลงมุมมอง)
chessboard_width = 200  # ความกว้าง 200 pixels
chessboard_height = 100  # ความสูง 100 pixels
src_pts = np.array([
    [0, 0], 
    [chessboard_width, 0], 
    [0, chessboard_height], 
    [chessboard_width, chessboard_height]
], dtype=np.float32)

# คำนวณ Perspective Transform Matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# สร้างกระดานหมากรุก 7x5 ช่อง
rows, cols = 5, 7
square_width = chessboard_width // cols
square_height = chessboard_height // rows
chessboard = np.ones((chessboard_height, chessboard_width, 3), dtype=np.uint8) * 255

for i in range(rows):
    for j in range(cols):
        if (i + j) % 2 == 0:  # ช่องสีดำ
            x_start, y_start = j * square_width, i * square_height
            x_end, y_end = x_start + square_width, y_start + square_height
            cv2.rectangle(chessboard, (x_start, y_start), (x_end, y_end), (0, 0, 0), -1)

# แปลงมุมมองของกระดานหมากรุกให้เข้ากับพิกัดที่กำหนด
warped_chessboard = cv2.warpPerspective(chessboard, M, (image_size, image_size))

# วางภาพกระดานหมากรุกลงบนภาพพื้นหลัง
mask = cv2.cvtColor(warped_chessboard, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY_INV)
for c in range(3):  # วนลูปช่องสี RGB
    image[:, :, c] = np.where(mask == 255, warped_chessboard[:, :, c], image[:, :, c])

# แสดงผลลัพธ์
cv2.imshow("Warped Chessboard", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# บันทึกเป็นไฟล์
cv2.imwrite("chessboard/warped_chessboard_layout2.png", image)
