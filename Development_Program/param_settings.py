import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

Golf_img_Path = '../Dataset/golf_car.png'
img_car = cv2.imread(Golf_img_Path, cv2.IMREAD_UNCHANGED)
camera_names = ["front", "back", "left", "right"]

# Define destination points for perspective transformation
Car_dst_points = np.float32([
    [465, 465],  # Point 1
    [575, 465],  # Point 2
    [465, 685],  # Point 3
    [575, 685]   # Point 4
])

# --------------------------------------------------------------------
# (shift_width, shift_height): how far away the birdview looks outside
# of the calibration pattern in horizontal and vertical directions
shift_w = 300
shift_h = 300

# Size of the gap between the calibration pattern and the car
inn_shift_w = 45
inn_shift_h = 45

# Total width/height of the stitched image
total_w = 1040
total_h = 1191

# Four corners of the rectangular region occupied by the car
xl = 465
xr = 575
yt = 465
yb = 685
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

def main():

    # Chessboard configurations
    chessboard_config = {
        "front": {
            "inner_dst_pts": np.array([[450, 310], [590, 310], [450, 410], [590, 410]], dtype=np.float32),
            "rows": 5, "cols": 7, "chessboard_width": 140, "chessboard_height": 100
        },
        "left": {
            "inner_dst_pts": np.array([[310, 524], [410, 524], [310, 664], [410, 664]], dtype=np.float32),
            "rows": 7, "cols": 5, "chessboard_width": 100, "chessboard_height": 140
        },
        "right": {
            "inner_dst_pts": np.array([[630, 524], [730, 524], [630, 664], [730, 664]], dtype=np.float32),
            "rows": 7, "cols": 5, "chessboard_width": 100, "chessboard_height": 140
        },
        "rear": {
            "inner_dst_pts": np.array([[450, 781], [590, 781], [450, 891], [590, 891]], dtype=np.float32),
            "rows": 5, "cols": 7, "chessboard_width": 140, "chessboard_height": 100
        }
    }

    # --------------------------------------------------------------------
    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 10))

    # Background (bird's-eye view area)
    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)  # Inverted to match top-down view
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("2D Bird’s-Eye View with Chessboard Overlay")

    # Draw outer boundary (dotted black line)
    outer_rect = patches.Rectangle((outer_xl, outer_yt), outer_xr - outer_xl, outer_yb - outer_yt, 
                                   linewidth=2, edgecolor='black', linestyle="dotted", facecolor='none', label="Outer Shift Area")
    ax.add_patch(outer_rect)

    # Draw inner boundary (dashed green line)
    inner_rect = patches.Rectangle((inner_xl, inner_yt), inner_xr - inner_xl, inner_yb - inner_yt, 
                                   linewidth=2, edgecolor='green', linestyle="dashed", facecolor='none', label="Inner Shift Area")
    ax.add_patch(inner_rect)

    # Draw vehicle boundary (cyan)
    car_rect = patches.Rectangle((xl, yt), xr - xl, yb - yt, linewidth=2, edgecolor='blue', facecolor='cyan', alpha=0.5, label="Vehicle")
    ax.add_patch(car_rect)

    # Draw camera coverage areas
    front_area = patches.Rectangle((outer_xl, outer_yt), total_w - 2 * shift_w, yt - outer_yt, linewidth=1, edgecolor='red', facecolor='red', alpha=0.3, label="Front Camera View")
    back_area = patches.Rectangle((outer_xl, yb), total_w - 2 * shift_w, outer_yb - yb, linewidth=1, edgecolor='green', facecolor='green', alpha=0.3, label="Back Camera View")
    left_area = patches.Rectangle((outer_xl, outer_yt), xl - outer_xl, total_h - 2 * shift_h, linewidth=1, edgecolor='yellow', facecolor='yellow', alpha=0.3, label="Left Camera View")
    right_area = patches.Rectangle((xr, outer_yt), outer_xr - xr, total_h - 2 * shift_h, linewidth=1, edgecolor='purple', facecolor='purple', alpha=0.3, label="Right Camera View")

    ax.add_patch(front_area)
    ax.add_patch(back_area)
    ax.add_patch(left_area)
    ax.add_patch(right_area)

    # Draw Chessboards
    for key, params in chessboard_config.items():
        dst_pts = params["inner_dst_pts"]
        x_min, y_min = dst_pts[0]  # Top-left corner
        x_max, y_max = dst_pts[3]  # Bottom-right corner
        
        chessboard_rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor='black',
            facecolor='black',  # Chessboard in black
            alpha=0.6,
            label=f"{key.capitalize()} Chessboard"
        )
        ax.add_patch(chessboard_rect)

    # Plot coordinate points
    for label, (x, y) in {
        "Car TL": (xl, yt), "Car TR": (xr, yt),
        "Car BL": (xl, yb), "Car BR": (xr, yb),
        "Outer TL": (outer_xl, outer_yt), "Outer TR": (outer_xr, outer_yt),
        "Outer BL": (outer_xl, outer_yb), "Outer BR": (outer_xr, outer_yb),
        "Inner TL": (inner_xl, inner_yt), "Inner TR": (inner_xr, inner_yt),
        "Inner BL": (inner_xl, inner_yb), "Inner BR": (inner_xr, inner_yb)
    }.items():
        ax.scatter(x, y, color="black", marker="o", s=40)  # Plot points
        ax.text(x + 10, y - 10, f"{label}\n({x}, {y})", fontsize=8, color="black")  # Add labels near points

    # Add legend
    ax.legend(loc="upper right")

    # Add an arrow from (0, shift_h) to (shift_w, shift_h)
    ax.annotate("", xy=(shift_w, shift_h), xytext=(0, shift_h),
                arrowprops=dict(arrowstyle="->", color="blue", linewidth=2))

    # Label the arrow
    ax.text(shift_w / 2, shift_h - 10, "Shift Boundary", fontsize=10, color="blue", ha="center")

    # Show plot
    plt.show()

if __name__ == "__main__":
    main()
