import numpy as np

# สร้างภาพตัวอย่าง 100x200 (ค่าสุ่ม)
front_image = np.random.randint(0, 255, (100, 200), dtype=np.uint8)

# ตัดเฉพาะคอลัมน์ตั้งแต่ต้นจนถึง xl
xl = 50
cropped_image = front_image[:, :xl]

print(cropped_image.shape)  # จะได้ (100, 50)