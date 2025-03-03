import cv2
import numpy as np

# โหลดภาพอ้างอิง (Front.jpg) และภาพที่จะนำมารวม (Right.jpg)
# โหลดภาพ
img1 = cv2.imread('Front.jpg')
img2 = cv2.imread('Right.jpg')

if img1 is None or img2 is None:
    print("ไม่สามารถโหลดภาพได้ ตรวจสอบ path ของภาพ")
    exit()

# ตัวอย่างชุดจุดจับคู่
# (ในงานจริงให้ใช้ฟีเจอร์จับคู่หรือกำหนดจุดที่แม่นยำขึ้น)
pts_img1 = np.array([
    [ 985,331],   # Top-Left
    [ 1079,330],  # Top-Right
    [ 1165,404],  # Bottom-Right
    [ 1075,422]   # Bottom-Left
], dtype=np.float32)

pts_img2 = np.array([
    [ 32,409],    # Top-Left
    [ 103,334],   # Top-Right
    [ 157,325],   # Bottom-Right
    [ 73,412]     # Bottom-Left
], dtype=np.float32)

H, mask = cv2.findHomography(pts_img1, pts_img2, cv2.RANSAC)
if H is None:
    print("ไม่สามารถคำนวณ Homography ได้")
    exit()
print("Homography Matrix:\n", H)

# คำนวณ bounding box สำหรับ warp ของ img1
h1, w1 = img1.shape[:2]
corners_img1 = np.array([
    [0, 0, 1],
    [w1, 0, 1],
    [w1, h1, 1],
    [0, h1, 1]
]).T

warped_corners = H.dot(corners_img1)
warped_corners /= warped_corners[2, :]

h2, w2 = img2.shape[:2]
corners_img2 = np.array([
    [0, 0],
    [w2, 0],
    [w2, h2],
    [0, h2]
])

all_corners = np.vstack((warped_corners[:2, :].T, corners_img2))
[x_min, y_min] = np.int32(all_corners.min(axis=0) - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0) + 0.5)

translation_dist = [-x_min, -y_min]
canvas_width = x_max - x_min
canvas_height = y_max - y_min

print("Canvas size:", canvas_width, "x", canvas_height)

T = np.array([
    [1, 0, translation_dist[0]],
    [0, 1, translation_dist[1]],
    [0, 0, 1]
])

warped_img1 = cv2.warpPerspective(img1, T.dot(H), (canvas_width, canvas_height))
if warped_img1 is None or warped_img1.size == 0:
    print("warpPerspective ล้มเหลว")
    exit()

# สร้าง canvas และวาง img2 ในตำแหน่งที่ถูกต้อง
canvas = warped_img1.copy()
canvas[translation_dist[1]:translation_dist[1]+h2, translation_dist[0]:translation_dist[0]+w2] = img2

if canvas is None or canvas.size == 0:
    print("Canvas ว่างเปล่า")
    exit()

cv2.imshow('Combined Image', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()