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


# Stitching and smoothing of the birdseye view image

If everything goes well in the previous section, and after executing the script [Step4_merge.py](Development_Program/Step4_merge.py), you will notice the stitched bird's-eye view image:

<img style="margin:0px auto;display:block" width=500 src="./Development_Program/out_merged_Images/final_merged_image.png"/>