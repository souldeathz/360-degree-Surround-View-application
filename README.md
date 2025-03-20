This project is a simple, runnable, and reproducible demo to show how to develop a surround-view system in Python.
The project is not very complex, but it does involve some careful computations. Now we explain the whole process step by step.

# Hardware and software

The hardware used in the Golf car project includes:

<img style="margin:0px auto;display:block" width=400 src="./Hardware_Setup/layout_0.jpg"/>

1. Four USB fisheye cameras, resolution: 1280x720.
2. Jetson AGX Xavier developer kit: [Purchase This](https://developer.nvidia.com/buy-jetson)
3. PCI-Ex USB 3.0 Framegrabber: [IOI U3X4-PCIE4XE304](https://www.ioi.com.cn/products/product_detail.php?pid=P0001&tid=&no=20190429002), a Quad Channel 4-port (1-port x 4) USB 3.0 to PCI Express x4 Gen 2 Host Card.

For more information, refer to the following documents:
- [Jetson AGX Xavier Developer Kit User Guide](https://developer.download.nvidia.com/assets/embedded/secure/jetson/xavier/docs/jetson_agx_xavier_developer_kit_user_guide.pdf?__token__=exp=1742224205~hmac=55e0d3f75e785205cd7f8c9355744d07bfb86ba56890d5da73ee6e2462de2b1a&t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczovL3d3dy5nb29nbGUuY29tLyJ9)
- [Jetson AGX Xavier Document](https://docs.nvidia.com/jetson/archives/r35.1/DeveloperGuide/text/SO/JetsonAgxXavierSeries.html)

📌 **Note:** During the **Development** phase, a **regular notebook** will be used instead of Jetson AGX Xavier and PCI-Ex USB 3.0 Framegrabber.

The software: 

1. Ubuntu
2. Python
3. OpenCV
4. PyQt5.
`PyQt5` is used mainly for multi-threading.

# Prepare work Step 1: Hardware Setup


Camera Installation on Golf Car : The installation of cameras requires careful positioning to ensure that each side has as many common points as possible. This ensures that the merge operation will have a significant overlap, resulting in better alignment and accuracy.


| |  |   |   |
|:-:|:-:|:-:|:-:|
|front|back|left|right|
|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/FOV_Front.jpg"/>|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/FOV_rear.jpg"/>|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/FOV_left.jpg"/>|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/FOV_right.jpg"/>|

# Prepare work Step 2: camera calibration


There is a script [Step1_cal.py](Development_Program/Step1_cal.py) in this project to help
 you calibrate the camera. I'm not going to discuss how to calibrate a camera here, as there are lots of resources on the web.
 
Below are the images taken by the four cameras, in the order `front.png`、`back.png`、`left.png`、`right.png`, they are in the `Dataset/Img_distortion_Testing/` directory.

| |  |   |   |
|:-:|:-:|:-:|:-:|
|front|back|left|right|
|<img style="margin:0px auto;display:block" width=200 src="./Dataset/Img_distortion_Testing/Front.jpg"/>|<img style="margin:0px auto;display:block" width=200 src="./Dataset/Img_distortion_Testing/Rear.jpg"/>|<img style="margin:0px auto;display:block" width=200 src="./Dataset/Img_distortion_Testing/Left.jpg"/>|<img style="margin:0px auto;display:block" width=200 src="./Dataset/Img_distortion_Testing/Right.jpg"/>|

The parameters of these cameras are stored in the yaml files `calibration_data_front.yaml`、`calibration_data_rear.yaml`、`calibration_data_left.yaml`、`calibration_data_right.yaml`, these files can be found in the [yaml](Development_Program/yaml) directory.

You can see there is a black-white calibration pattern on the ground, the size of the pattern is `140mx100cm`, the size of each black/white square is `20cmx20cm`


# Setting projection parameters


Now we compute the projection matrix for each camera. This matrix will transform the undistorted image into a bird's view of the ground. All four projection matrices must fit together to make sure the four projected images can be stitched together.

This is done by putting calibration patterns on the ground, taking the camera images, manually choosing the feature points, and then computing the matrix.

See the illustration below:

<img style="margin:0px auto;display:block" width=800 height=800 src="./Hardware_Setup/paramsettings.png"/>

Firstly you put four calibration boards at the four corners around the car (the blue squares). There are no particular restrictions on how large the board must be, only make sure you can see it clearly in the image.

OF course, each board must be seen by the two adjacent cameras.

Now we need to set a few parameters: (in `cm` units)

+ `Inner TL,TR,BL,BR`：distance between the inner edges of the left/right calibration boards and the car， the distance between the inner edges of the front/back calibration boards and the car。(gap between calibration pattern and car)
+ `Shif Boundary`：How far you will want to look at out of the boards. The bigger these values, the larger the area the birdview image will cover.
+ `totalWidth`, `totalHeight`：Size of the area that the birdview image covers. In this project, the calibration pattern is of width `1040cm` and height `1191cm`, hence the bird view image will cover an area of size . For simplicity,
we let each pixel correspond to 1cm, so the final bird-view image also has a resolution

+ The four corners of the rectangular area where the vehicle is located (marked with red dots in the image) are denoted by the coordinates (xl, yt), (xr, yt), (xl, yb), and (xr, yb), where "l" stands for left, "r" stands for right, "t" stands for top, and "b" stands for bottom. The camera cannot see this rectangular area, and we will use an icon of the vehicle to cover it.

Note that the extension lines of the four sides of the vehicle area divide the entire bird's-eye view into eight parts: front-left (FL), front-center (F), front-right (FR), left (L), right (R), back-left (BL), back-center (B), and back-right (BR). Among them, FL (area I), FR (area II), BL (area III), and BR (area IV) are the overlapping areas of adjacent camera views, and they are the parts that we need to focus on for fusion processing. The areas F, R, L, and R belong to the individual views of each camera and do not require fusion processing.

The above parameters are saved in [param_settings.py](./Development_Program/param_settings.py) 

# select feature points for the projection matrix

The process of transforming raw images into a bird’s-eye view relies on defining a projective transformation, which requires carefully selecting feature points. The key to achieving this transformation is the chessboard layout, which provides a structured reference for computing the projection matrix.

The process of transforming raw images into a bird’s-eye view relies on defining a projective transformation, which requires carefully selecting feature points. The key to achieving this transformation is the chessboard layout, which provides a structured reference for computing the projection matrix.

1. Chessboard as the Key Reference ( [Step2_create_program_chessboard_layout.py](./Development_Program/Step2_create_program_chessboard_layout.py) )
To accurately align each camera’s perspective, we generate a synthetic chessboard layout that acts as the foundation for perspective transformation. Each chessboard is warped and placed in **a predefined mapping area that represents the bird’s-eye view**. The key aspects of this step include:

Creating a 5×7 or 7×5 chessboard grid for each camera (front, left, rear, right). `totalWidth = 1040 cm` and `totalHeight = 1191 cm`, representing the calibrated bird’s-eye view coverage.
Warping the chessboard to match the **expected perspective of the bird’s-eye view**.
Defining control points (feature points) that will later be used to compute homography matrices.
This ensures that the mapping structure is consistent across all cameras and that real-world objects align properly when images are merged.

| |  |   |   |
|:-:|:-:|:-:|:-:|
|front|back|left|right|
|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Undistorted_Images/warped_chessboard_front.png"/>|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Undistorted_Images/warped_chessboard_rear.png"/>|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Undistorted_Images/warped_chessboard_left.png"/>|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Undistorted_Images/warped_chessboard_Right.png"/>|

2. Computing the Homography Transformation (`Step3_Projective_Transformation.py`)
Once the chessboard reference is established, each camera image must be transformed into the **projected space** to match the predefined map. This is done through:  

- **Undistorting the camera images** using calibration parameters (*intrinsic matrix and distortion coefficients*).  
- **Detecting and refining chessboard corners** in both the test image and the reference chessboard.  
- **Computing the homography matrix** using *RANSAC*, ensuring robustness against errors.  
- **Warping the undistorted image** using the computed homography, aligning the raw camera perspective to the bird’s-eye view.  

The **homography matrix** acts as a bridge between the raw camera perspective and the top-down map, enabling each camera to "see" the environment from the correct viewpoint.


| |  |   |   |
|:-:|:-:|:-:|:-:|
|front|back|left|right|
|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Undistorted_Images/warped_chessboard_front_Matching.png"/>|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Undistorted_Images/warped_chessboard_rear_Matching.png"/>|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Undistorted_Images/warped_chessboard_left_Matching.png"/>|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Undistorted_Images/warped_chessboard_Right_Matching.png"/>|
|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Undistorted_Images/front_image_matching.png"/>|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Undistorted_Images/rear_image_matching.png"/>|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Undistorted_Images/left_image_matching.png"/>|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Undistorted_Images/right_image_matching.png"/>|
|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Undistorted_Images/front_warped_image.png"/>|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Undistorted_Images/rear_warped_image.png"/>|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Undistorted_Images/left_warped_image.png"/>|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Undistorted_Images/right_warped_image.png"/>|

# Stitching and smoothing of the birdseye view image

The **homography matrix** acts as a bridge between the raw camera perspective and the top-down map, enabling each camera to "see" the environment from the correct viewpoint.

After computing the necessary transformations, the images from all cameras are converted into **projective images** that fit into the designated bird’s-eye mapping space. These projected images are then used in subsequent steps to create a **seamless 360-degree surround view** by:  

- Ensuring that all images align with the reference map.  
- Providing accurate spatial relationships between objects.  
- Allowing smooth blending and stitching of images.  

By utilizing a structured chessboard layout and precise homography calculations, the system ensures that each camera’s output is properly mapped, creating a **realistic and distortion-free** bird’s-eye perspective.

| |  |   |   |
|:-:|:-:|:-:|:-:|
|front|back|left|right|
|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Undistorted_Images/front_warped_image.png"/>|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Undistorted_Images/rear_warped_image.png"/>|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Undistorted_Images/left_warped_image.png"/>|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Undistorted_Images/right_warped_image.png"/>|

If everything goes well in the previous section, and after executing the script [Step4_merge.py](Development_Program/Step4_merge.py), you will notice the stitched bird's-eye view image:

<img style="margin:0px auto;display:block" width=500 src="./Development_Program/out_merged_Images/final_merged_image.png"/>

### **Detailed Explanation: `ImageStitcher.get_weights_and_masks(images)`** in  [image_processing.py](Development_Program/image_processing.py)

The function **`ImageStitcher.get_weights_and_masks(images)`** plays a crucial role in merging the four camera images (*front, left, rear, right*) into a **seamless surround view**. The merging process is carefully designed to ensure **smooth transitions between images** while maintaining **visual consistency**.

---

**1. Image Segmentation into 8 Sections**  
Instead of merging entire images directly, the system first **divides each image into 8 key sections**:  

| **Section Name** | **Source Images** |
|-----------------|----------------|
| **LT (Left-Top)**  | Front & Left  |
| **RT (Right-Top)** | Front & Right |
| **LB (Left-Bottom)** | Rear & Left  |
| **RB (Right-Bottom)** | Rear & Right |
| **FM (Front-Middle)** | Front Only |
| **BM (Back-Middle)** | Rear Only |
| **LM (Left-Middle)** | Left Only |
| **RM (Right-Middle)** | Right Only |



Each section is responsible for merging overlapping regions where adjacent images intersect.

---

**2. Identifying Overlapping Areas**  
For **each overlapping region**, the function determines how much of each image should be **blended** to create a smooth transition. The system:  
1. **Extracts the intersecting areas** from each camera’s image.  
2. **Computes a weight mask** for the overlapping region to avoid visible edges.  
3. **Blends the overlapping sections smoothly** instead of performing a hard cut.  

For example:  
- The **LT (Left-Top)** section merges the **top-left area of the front image** with the **top-right area of the left image**.  
- The **RT (Right-Top)** section merges the **top-right area of the front image** with the **top-left area of the right image**.  
- The same concept applies to **LB and RB**, but using the **rear image instead of the front**.

| |  |   |
|:-:|:-:|:-:|
|1|2|Merge|
|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Merge_image/FI_front.png"/>| <img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Merge_image/LI_left.png"/> | <img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Merge_image/merged_FI_LI_is_LT.png"/> |
|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Merge_image/FII_front.png"/>| <img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Merge_image/RII_right.png"/> | <img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Merge_image/merged_FI_RII_is_RT.png"/> |
|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Merge_image/LIII_left.png"/>| <img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Merge_image/BIII_back.png"/> | <img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Merge_image/merged_BIII_LIII_is_LB.png"/> |
|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Merge_image/BIV_back.png"/>| <img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Merge_image/RIV_right.png"/> | <img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Merge_image/merged_BIV_RIV_is_RB.png"/> |

---

**3. Applying Weight Masks for Smooth Transitions**
To ensure that no hard edges appear in the merged image, the function applies **weighted blending** using a mask.  

#### **How Weight Masks Work:**  
- Each overlapping region is assigned a **gradual transition mask**.  
- The transition ensures that **pixels near the center of an overlap** take equal contributions from both images.  
- The blending factor is determined dynamically, considering **the overlap amount and pixel intensities**.

For example:  
- If **Image A** contributes **80% of a pixel**, then **Image B** contributes the remaining **20%**.  
- In the center of an overlap, both images contribute **50% each**.  
- At the edge of an overlap, one image dominates to prevent ghosting effects.

This avoids abrupt **brightness changes** and ensures a **seamless merge**.

---

**4. Merging All Sections into a Final Image**  
After merging all overlapping sections (*LT, RT, LB, RB*), the remaining non-overlapping sections (*FM, BM, LM, RM*) are **directly added**.  
Finally, all **8 sections** are combined into the final **full bird’s-eye view image**.

| |  |   |
|:-:|:-:|:-:|
|1|2|3|
|<img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Merge_image/merged_FI_LI_is_LT.png"/>| <img style="margin:0px auto;display:block" width=200 height=200 src="./Hardware_Setup/Result/Merge_image/FM.png"/> | <img style="margin:0px auto;display:block" width=200 src="./Hardware_Setup/Result/Merge_image/merged_FI_RII_is_RT.png"/> |
|<img style="margin:0px auto;display:block" width=200 height=200 src="./Hardware_Setup/Result/Merge_image/LM.png"/>| **Car** | <img style="margin:0px auto;display:block" width=200 height=200 src="./Hardware_Setup/Result/Merge_image/RM.png"/> |
|<img style="margin:0px auto;display:block" width=200 height=200 src="./Hardware_Setup/Result/Merge_image/merged_BIII_LIII_is_LB.png"/>| <img style="margin:0px auto;display:block" width=200 height=200 src="./Hardware_Setup/Result/Merge_image/BM.png"/> | <img style="margin:0px auto;display:block" width=200 height=200 src="./Hardware_Setup/Result/Merge_image/merged_BIV_RIV_is_RB.png"/> |

---

### **Conclusion**  
The `ImageStitcher.get_weights_and_masks(images)` function is **crucial** because it:  
✅ **Divides images into meaningful sections** for precise merging.  
✅ **Identifies and processes overlapping areas** for smooth blending.  
✅ **Uses weight masks** to avoid visible edges and brightness shifts.  
✅ **Constructs a final seamless bird’s-eye view** by integrating all 8 sections.  

This structured approach ensures that the **merged 360-degree view is realistic, distortion-free, and visually consistent**. 