import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

Golf_img_Path = '../Dataset/golf_car.png'
img_car = cv2.imread(Golf_img_Path, cv2.IMREAD_UNCHANGED)
camera_names = ["front", "back", "left", "right"]

# --------------------------------------------------------------------
# (shift_width, shift_height): how far away the birdview looks outside
# of the calibration pattern in horizontal and vertical directions
shift_w = 400
shift_h = 400

Cal_size_w = 120
Cal_size_h = 120

# Size of the car in pixels
Car_size_w = 110
Car_size_h = 250

# Size of the gap between the calibration pattern and the car
inn_shift_w = 45
inn_shift_h = 45

# Define destination points for perspective transformation
# จุดซ้ายบนของรถ (Top-left)
top_left_x = shift_w + Cal_size_w + inn_shift_w                 
top_left_y = shift_h + Cal_size_h + inn_shift_h                  

Car_dst_points = np.float32([
    [top_left_x, top_left_y],                                     
    [top_left_x + Car_size_w, top_left_y],                        
    [top_left_x, top_left_y + Car_size_h],                        
    [top_left_x + Car_size_w, top_left_y + Car_size_h]            
])
# --------------------------------------------------------------------

# Total width/height of the stitched image

total_w = (2 * (shift_w + Cal_size_w + inn_shift_w)) + Car_size_w
total_h = (2 * (shift_h + Cal_size_h + inn_shift_h)) + Car_size_h
print(f"Total Width: {total_w}, Total Height: {total_h}")
# Four corners of the rectangular region occupied by the car
xl = shift_w + Cal_size_w + inn_shift_w
xr = xl + Car_size_w
yt = shift_h + Cal_size_h + inn_shift_h
yb = yt + Car_size_h

# --------------------------------------------------------------------

project_shapes = {
    "front": (total_w, yt),
    "back":  (total_w, yt),
    "left":  (total_h, xl),
    "right": (total_h, xl)
}

# --------------------------------------------------------------------
# Correct outer boundary (shrink inward from total image size)
outer_xl = shift_w
outer_xr = total_w - shift_w
outer_yt = shift_h
outer_yb = total_h - shift_h

# Correct inner boundary (gap between calibration pattern and car)
inner_xl = xl - inn_shift_w
inner_xr = xr + inn_shift_w
inner_yt = yt - inn_shift_h
inner_yb = yb + inn_shift_h



# Chessboard configurations
chessboard_config = {
    "front": {
        "inner_dst_pts": np.array([
            [shift_w + 150, shift_h + 10],   # 450, 310
            [shift_w + 290, shift_h + 10],   # 590, 310
            [shift_w + 150, shift_h + 110],  # 450, 410
            [shift_w + 290, shift_h + 110],  # 590, 410
        ], dtype=np.float32),
        "rows": 5, "cols": 7,
        "chessboard_width": 140,
        "chessboard_height": 100
    },
    "left": {
        "inner_dst_pts": np.array([
            [shift_w + 10,  shift_h + 224],  # 310, 524
            [shift_w + 110, shift_h + 224],  # 410, 524
            [shift_w + 10,  shift_h + 364],  # 310, 664
            [shift_w + 110, shift_h + 364],  # 410, 664
        ], dtype=np.float32),
        "rows": 7, "cols": 5,
        "chessboard_width": 100,
        "chessboard_height": 140
    },
    "right": {
        "inner_dst_pts": np.array([
            [shift_w + 330, shift_h + 224],  # 630, 524
            [shift_w + 430, shift_h + 224],  # 730, 524
            [shift_w + 330, shift_h + 364],  # 630, 664
            [shift_w + 430, shift_h + 364],  # 730, 664
        ], dtype=np.float32),
        "rows": 7, "cols": 5,
        "chessboard_width": 100,
        "chessboard_height": 140
    },
    "rear": {
        "inner_dst_pts": np.array([
            [shift_w + 150, shift_h + 470],  # 450, 781
            [shift_w + 290, shift_h + 470],  # 590, 781
            [shift_w + 150, shift_h + 570],  # 450, 891
            [shift_w + 290, shift_h + 570],  # 590, 891
        ], dtype=np.float32),
        "rows": 5, "cols": 7,
        "chessboard_width": 140,
        "chessboard_height": 100
    }
}

def main():
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(outer_xl, outer_xr)
    ax.set_ylim(outer_yb, outer_yt)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # --- Draw Outer and Inner Boundaries ---
    outer_rect = patches.Rectangle((outer_xl, outer_yt), outer_xr - outer_xl, outer_yb - outer_yt,
                                   linewidth=2, edgecolor='black', linestyle="dotted", facecolor='none')
    ax.add_patch(outer_rect)

    inner_rect = patches.Rectangle((inner_xl, inner_yt), inner_xr - inner_xl, inner_yb - inner_yt,
                                   linewidth=2, edgecolor='green', linestyle="dashed", facecolor='none')
    ax.add_patch(inner_rect)

    # --- Draw 8 Sections ---
    sections = {
        "LT": (outer_xl, outer_yt, inner_xl - outer_xl, inner_yt - outer_yt),
        "RT": (inner_xr, outer_yt, outer_xr - inner_xr, inner_yt - outer_yt),
        "LB": (outer_xl, inner_yb, inner_xl - outer_xl, outer_yb - inner_yb),
        "RB": (inner_xr, inner_yb, outer_xr - inner_xr, outer_yb - inner_yb),
        "FM": (inner_xl, outer_yt, inner_xr - inner_xl, inner_yt - outer_yt),
        "BM": (inner_xl, inner_yb, inner_xr - inner_xl, outer_yb - inner_yb),
        "LM": (outer_xl, inner_yt, inner_xl - outer_xl, inner_yb - inner_yt),
        "RM": (inner_xr, inner_yt, outer_xr - inner_xr, inner_yb - inner_yt),
    }
    colors = {
        "LT": "orange", "RT": "violet", "LB": "lightgreen", "RB": "plum",
        "FM": "red", "BM": "green", "LM": "yellow", "RM": "purple"
    }
    for label, (x, y, w, h) in sections.items():
        rect = patches.Rectangle((x, y), w, h,
                                 linewidth=1, edgecolor='black', facecolor=colors[label], alpha=0.4)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, fontsize=10, ha='center', va='center', color='black')

    # --- Draw Vehicle ---
    car_rect = patches.Rectangle((xl, yt), xr - xl, yb - yt,
                                 linewidth=2, edgecolor='blue', facecolor='cyan', alpha=0.5)
    ax.add_patch(car_rect)

    # --- Draw Chessboards ---
    for key, params in chessboard_config.items():
        dst_pts = params["inner_dst_pts"]
        x_min, y_min = dst_pts[0]
        x_max, y_max = dst_pts[3]
        chessboard_rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='black', facecolor='black', alpha=0.6
        )
        ax.add_patch(chessboard_rect)

        # Plot chessboard corners
        for (x, y) in dst_pts:
            ax.scatter(x, y, color="black", marker="o", s=10)
            ax.text(x + 5, y - 5, f"({int(x)}, {int(y)})", fontsize=8, color="black")

    # --- Plot Outer, Inner, Car Corners ---
    all_corners = [
        (outer_xl, outer_yt), (outer_xr, outer_yt), (outer_xl, outer_yb), (outer_xr, outer_yb),
        (inner_xl, inner_yt), (inner_xr, inner_yt), (inner_xl, inner_yb), (inner_xr, inner_yb),
        (xl, yt), (xr, yt), (xl, yb), (xr, yb)
    ]

    for (x, y) in all_corners:
        ax.scatter(x, y, color="black", marker="o", s=10)
        ax.text(x - 8, y - 8, f"({int(x)}, {int(y)})", fontsize=8, color="black")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
