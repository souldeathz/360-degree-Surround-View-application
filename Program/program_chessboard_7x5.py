import cv2
import numpy as np

# ขนาดของกระดานหมากรุก
rows, cols = 5, 7  # 5 แถว 7 คอลัมน์
square_size = 200  # 20 cm = 200 pixels (ปรับได้ตาม DPI)

# คำนวณขนาดของภาพ
width = cols * square_size
height = rows * square_size

# สร้างภาพสีขาว
chessboard = np.ones((height, width, 3), dtype=np.uint8) * 255

# วาดตารางหมากรุก
for i in range(rows):
    for j in range(cols):
        if (i + j) % 2 == 0:  # สลับช่องดำ-ขาว
            x_start, y_start = j * square_size, i * square_size
            x_end, y_end = x_start + square_size, y_start + square_size
            cv2.rectangle(chessboard, (x_start, y_start), (x_end, y_end), (0, 0, 0), -1)

# แสดงภาพ
cv2.imshow("Chessboard", chessboard)
cv2.waitKey(0)
cv2.destroyAllWindows()

# บันทึกเป็นไฟล์
cv2.imwrite("chessboard/chessboard_7x5.png", chessboard)