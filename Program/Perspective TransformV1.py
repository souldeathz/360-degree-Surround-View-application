import cv2
import numpy as np

# โหลดภาพ
img1 = cv2.imread('Front.jpg')  # ภาพอ้างอิง
img2 = cv2.imread('Right.jpg')  # ภาพที่ต้องการ warp มารวม

if img1 is None or img2 is None:
    print("ไม่สามารถโหลดภาพได้ ตรวจสอบ path ของภาพ")
    exit()

# 1. Padding ขวาของ img1 ด้วย pad = 1000
pad = 1000
img1_padded = cv2.copyMakeBorder(img1, top=0, bottom=pad, left=0, right=pad, 
                                 borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
print("ขนาดของ img1 หลังจาก padding:", img1_padded.shape)


# ตัวอย่างชุดจุดจับคู่ (ในงานจริงควรใช้ฟีเจอร์จับคู่ที่แม่นยำ)
# จุดใน img1 (reference) และ img2
pts_img1 = np.array([
    [ 985,331],   # Top-Left
    [1079,330],   # Top-Right
    [1165,404],   # Bottom-Right
    [1075,422]    # Bottom-Left
], dtype=np.float32)

pts_img2 = np.array([
    [ 32,409],    # Top-Left
    [103,334],    # Top-Right
    [157,325],    # Bottom-Right
    [ 73,412]     # Bottom-Left
], dtype=np.float32)


# คำนวณ Homography จาก img2 ไปยัง img1_padded (reference)
H, mask = cv2.findHomography(pts_img2, pts_img1, cv2.RANSAC)
if H is None:
    print("ไม่สามารถคำนวณ Homography ได้")
    exit()
print("Homography Matrix:\n", H)

# กำหนดขนาด canvas เป็นขนาดของ img1_padded
h_padded, w_padded = img1_padded.shape[:2]

# 2. Warp img2 ด้วย Homography โดยใช้ canvas ขนาด img1_padded
warped_img2 = cv2.warpPerspective(img2, H, (w_padded, h_padded))
if warped_img2 is None or warped_img2.size == 0:
    print("warpPerspective ล้มเหลว")
    exit()

# รวมภาพ: วาง warped_img2 ลงบน img1_padded ด้วย masking (เฉพาะที่มีข้อมูล)
combined = img1_padded.copy()
mask_warped = (warped_img2.sum(axis=2) > 0)
combined[mask_warped] = warped_img2[mask_warped]

cv2.imshow('Combined Image', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()