import cv2
import numpy as np

# Chessboard size
rows, cols = 5, 7  # 5 rows, 7 columns
square_size = 200  # 20 cm = 200 pixels (adjust based on DPI)

# Calculate image dimensions
width = cols * square_size
height = rows * square_size

# Create a white image
chessboard = np.ones((height, width, 3), dtype=np.uint8) * 255

# Draw the chessboard pattern
for i in range(rows):
    for j in range(cols):
        if (i + j) % 2 == 0:  # Alternate black and white squares
            x_start, y_start = j * square_size, i * square_size
            x_end, y_end = x_start + square_size, y_start + square_size
            cv2.rectangle(chessboard, (x_start, y_start), (x_end, y_end), (0, 0, 0), -1)

# Display the image
cv2.imshow("Chessboard", chessboard)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image to a file
cv2.imwrite("chessboard_7x5.png", chessboard)