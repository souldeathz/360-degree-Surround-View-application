import cv2
import numpy as np
import matplotlib.pyplot as plt

# โหลดภาพและแปลงเป็น BGRA (เพิ่มช่อง alpha)
img1 = cv2.imread('image_F.jpg')
img2 = cv2.imread('image_L.jpg')

img3 = cv2.imread('image_R.jpg')
img4 = cv2.imread('image_REAR.jpg')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2BGRA)
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2BGRA)
# ฟังก์ชัน callback เมื่อมีการคลิกเมาส์
clicked_points = []

# ฟังก์ชันเลือก 4 จุดจากภาพ
def select_points(img):
    global clicked_points , img_display
    clicked_points = []
    img_display = img.copy()
    cv2.imshow("Select 4 Points", img_display)
    cv2.setMouseCallback("Select 4 Points", click_event)
    
    while len(clicked_points) < 4:
        cv2.waitKey(1)
    
    cv2.destroyWindow("Select 4 Points")
    return np.float32(clicked_points)

def click_event(event, x, y, flags, param):
    global clicked_points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        # บันทึกจุดที่คลิก
        clicked_points.append((x, y))
        # วาดวงกลมเพื่อแสดงตำแหน่งที่เลือก
        cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4 Points", img_display)

# ฟังก์ชันเพื่อเปลี่ยน pixel สีดำให้เป็นโปร่งใส (alpha = 0)
def remove_black_alpha(img):
    # ใช้ grayscale เพื่อตรวจจับ pixel ที่เป็นสีดำ
    gray = cv2.cvtColor(img[:,:,:3], cv2.COLOR_BGR2GRAY)
    # สร้าง alpha channel: pixel ที่มีค่า > 0 ให้ alpha=255, ถ้าไม่ให้ alpha=0
    alpha = np.where(gray > 0, 255, 0).astype(np.uint8)
    img[:,:,3] = alpha
    return img

img1 = remove_black_alpha(img1)
img2 = remove_black_alpha(img2)
img3 = remove_black_alpha(img3)
img4 = remove_black_alpha(img4)

# กำหนด corresponding points (ปรับค่าตามภาพจริง)

# เลือกจุดจากทั้งสองภาพ
pts_img1_L = select_points(img1)
print("Selected source points:", pts_img1_L)
pts_img2_L = select_points(img2)
print("Selected destination points:", pts_img2_L)



# คำนวณ Homography matrix
H1, status = cv2.findHomography(pts_img2_L, pts_img1_L, cv2.RANSAC)
# H2, status = cv2.findHomography(pts_img2_R, pts_img1_R, cv2.RANSAC)
# กำหนดขนาดผลลัพธ์
height1, width1 = img1.shape[:2]
height2, width2 = img2.shape[:2]
result_width = width1 + width2
result_height = max(height1, height2)

# Warp ภาพ img2 ที่มี alpha ด้วย Homography
warped_img1 = cv2.warpPerspective(img2, H1, (result_width, result_height),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0, 0))  # กำหนด border เป็นโปร่งใส

# สร้างผลลัพธ์เริ่มต้นเป็นภาพที่โปร่งใสทั้งหมด
result = np.zeros((result_height, result_width, 4), dtype=np.uint8)
# วาง img1 ลงในผลลัพธ์
result[0:height1, 0:width1] = img1

# Composite warped_img2 กับ result โดยใช้ alpha blending
alpha2 = warped_img1[:,:,3:4] / 255.0
alpha1 = result[:,:,3:4] / 255.0

# Composite ช่องสี B,G,R
result[:,:,:3] = warped_img1[:,:,:3] * alpha2 + result[:,:,:3] * (1 - alpha2)
# Composite ช่อง alpha
result[:,:,3] = np.clip((alpha2 + alpha1 * (1 - alpha2)) * 255, 0, 255).astype(np.uint8).squeeze()


pts_img1_R = select_points(result)
print("Selected destination points:", pts_img1_R)
pts_img2_R = select_points(img3)
print("Selected destination points:", pts_img2_R)

# คำนวณ Homography matrix สำหรับภาพที่สาม
H2, status = cv2.findHomography(pts_img2_R, pts_img1_R, cv2.RANSAC)

# Warp ภาพ img3 ที่มี alpha ด้วย Homography
warped_img2 = cv2.warpPerspective(img3, H2, (result_width, result_height),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0, 0))  # กำหนด border เป็นโปร่งใส

# Composite warped_img2 กับ result โดยใช้ alpha blending
alpha2 = warped_img2[:,:,3:4] / 255.0
alpha1 = result[:,:,3:4] / 255.0

# Composite ช่องสี B,G,R
result[:,:,:3] = warped_img2[:,:,:3] * alpha2 + result[:,:,:3] * (1 - alpha2)
# Composite ช่อง alpha
result[:,:,3] = np.clip((alpha2 + alpha1 * (1 - alpha2)) * 255, 0, 255).astype(np.uint8).squeeze()


pts_img1_T = select_points(result)
print("Selected destination points:", pts_img1_T)
pts_img2_Rear = select_points(img4)
print("Selected destination points:", pts_img2_Rear)

# คำนวณ Homography matrix สำหรับภาพที่สี่
H3, status = cv2.findHomography(pts_img2_Rear, pts_img1_T, cv2.RANSAC)

# Warp ภาพ img4 ที่มี alpha ด้วย Homography
warped_img3 = cv2.warpPerspective(img4, H3, (result_width, result_height),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0, 0))  # กำหนด border เป็นโปร่งใส

# Composite warped_img3 กับ result โดยใช้ alpha blending
alpha2 = warped_img3[:,:,3:4] / 255.0
alpha1 = result[:,:,3:4] / 255.0

# Composite ช่องสี B,G,R
result[:,:,:3] = warped_img3[:,:,:3] * alpha2 + result[:,:,:3] * (1 - alpha2)
# Composite ช่อง alpha
result[:,:,3] = np.clip((alpha2 + alpha1 * (1 - alpha2)) * 255, 0, 255).astype(np.uint8).squeeze()

# แสดงผลลัพธ์ (Matplotlib รองรับ alpha channel)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA))
plt.axis('off')
plt.show()




# plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA))
# plt.axis('off')
# plt.show()

# pts_img1_R = select_points(img3)
# print("Selected destination points:", pts_img1_R)
# pts_img2_R = select_points(img4)
# print("Selected destination points:", pts_img2_R)

# # คำนวณ Homography matrix
# H2, status = cv2.findHomography(pts_img2_R, pts_img1_R, cv2.RANSAC)

# # กำหนดขนาดผลลัพธ์
# height1, width1 = img3.shape[:2]
# height2, width2 = img4.shape[:2]
# result_width = width1 + width2
# result_height = max(height1, height2)

# # Warp ภาพ img2 ที่มี alpha ด้วย Homography
# warped_img2 = cv2.warpPerspective(img4, H2, (result_width, result_height),
#                                    flags=cv2.INTER_LINEAR,
#                                    borderMode=cv2.BORDER_CONSTANT,
#                                    borderValue=(0, 0, 0, 0))  # กำหนด border เป็นโปร่งใส

# # สร้างผลลัพธ์เริ่มต้นเป็นภาพที่โปร่งใสทั้งหมด
# result = np.zeros((result_height, result_width, 4), dtype=np.uint8)
# # วาง img1 ลงในผลลัพธ์
# result[0:height1, 0:width1] = img3

# # Composite warped_img2 กับ result โดยใช้ alpha blending
# alpha2 = warped_img2[:,:,3:4] / 255.0
# alpha1 = result[:,:,3:4] / 255.0

# # Composite ช่องสี B,G,R
# result[:,:,:3] = warped_img2[:,:,:3] * alpha2 + result[:,:,:3] * (1 - alpha2)
# # Composite ช่อง alpha
# result[:,:,3] = np.clip((alpha2 + alpha1 * (1 - alpha2)) * 255, 0, 255).astype(np.uint8).squeeze()

# # แสดงผลลัพธ์ (Matplotlib รองรับ alpha channel)
# plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA))
# plt.axis('off')
# plt.show()

