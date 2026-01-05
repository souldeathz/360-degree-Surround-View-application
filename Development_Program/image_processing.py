import cv2
import numpy as np
from PIL import Image
from param_settings import xl, xr, yt, yb
import threading
import time
import concurrent.futures

class LuminanceBalancer:
    """Adjusts luminance balance between images"""

    @staticmethod
    def tune(x):
        if x >= 1:
            return x * np.exp((1 - x) * 0.5)
        else:
            return x * np.exp((1 - x) * 0.8)

    @staticmethod
    def adjust_luminance(gray, factor):
        """Adjusts the brightness of a grayscale image"""
        return np.minimum(gray * factor, 255).astype(np.uint8)

    @staticmethod
    def get_mean_statistic(gray, mask):
        """Calculates the accumulated values in the region specified by the mask"""
        return np.sum(gray * mask)

    @staticmethod
    def mean_luminance_ratio(grayA, grayB, mask):
        return (LuminanceBalancer.get_mean_statistic(grayA, mask) /
                LuminanceBalancer.get_mean_statistic(grayB, mask))

    @staticmethod
    def make_luminance_balance(frames, masks):
        """
        Balances the luminance for each channel in four images (front, left, back, right)
        using the predefined masks.
        """
        # Assuming frames are in the order [front, left, back, right]
        front, left, back, right = frames
        m1, m2, m3, m4 = masks

        # Split channels for each image
        Fb, Fg, Fr = cv2.split(front)
        Bb, Bg, Br = cv2.split(back)
        Lb, Lg, Lr = cv2.split(left)
        Rb, Rg, Rr = cv2.split(right)

        # Compute mean luminance ratios for each channel
        a1 = LuminanceBalancer.mean_luminance_ratio(ImageStitcher.RII(Rb), ImageStitcher.FII(Fb), m2)
        a2 = LuminanceBalancer.mean_luminance_ratio(ImageStitcher.RII(Rg), ImageStitcher.FII(Fg), m2)
        a3 = LuminanceBalancer.mean_luminance_ratio(ImageStitcher.RII(Rr), ImageStitcher.FII(Fr), m2)

        b1 = LuminanceBalancer.mean_luminance_ratio(ImageStitcher.BIV(Bb), ImageStitcher.RIV(Rb), m4)
        b2 = LuminanceBalancer.mean_luminance_ratio(ImageStitcher.BIV(Bg), ImageStitcher.RIV(Rg), m4)
        b3 = LuminanceBalancer.mean_luminance_ratio(ImageStitcher.BIV(Br), ImageStitcher.RIV(Rr), m4)

        c1 = LuminanceBalancer.mean_luminance_ratio(ImageStitcher.LIII(Lb), ImageStitcher.BIII(Bb), m3)
        c2 = LuminanceBalancer.mean_luminance_ratio(ImageStitcher.LIII(Lg), ImageStitcher.BIII(Bg), m3)
        c3 = LuminanceBalancer.mean_luminance_ratio(ImageStitcher.LIII(Lr), ImageStitcher.BIII(Br), m3)

        d1 = LuminanceBalancer.mean_luminance_ratio(ImageStitcher.FI(Fb), ImageStitcher.LI(Lb), m1)
        d2 = LuminanceBalancer.mean_luminance_ratio(ImageStitcher.FI(Fg), ImageStitcher.LI(Lg), m1)
        d3 = LuminanceBalancer.mean_luminance_ratio(ImageStitcher.FI(Fr), ImageStitcher.LI(Lr), m1)

        # Compute geometric mean for each channel
        t1 = (a1 * b1 * c1 * d1) ** 0.25
        t2 = (a2 * b2 * c2 * d2) ** 0.25
        t3 = (a3 * b3 * c3 * d3) ** 0.25

        # Compute adjustment factors for front image channels
        x1 = LuminanceBalancer.tune(t1 / ((d1 / a1) ** 0.5))
        x2 = LuminanceBalancer.tune(t2 / ((d2 / a2) ** 0.5))
        x3 = LuminanceBalancer.tune(t3 / ((d3 / a3) ** 0.5))

        Fb = LuminanceBalancer.adjust_luminance(Fb, x1)
        Fg = LuminanceBalancer.adjust_luminance(Fg, x2)
        Fr = LuminanceBalancer.adjust_luminance(Fr, x3)

        # Compute adjustment factors for back image channels
        y1 = LuminanceBalancer.tune(t1 / ((b1 / c1) ** 0.5))
        y2 = LuminanceBalancer.tune(t2 / ((b2 / c2) ** 0.5))
        y3 = LuminanceBalancer.tune(t3 / ((b3 / c3) ** 0.5))

        Bb = LuminanceBalancer.adjust_luminance(Bb, y1)
        Bg = LuminanceBalancer.adjust_luminance(Bg, y2)
        Br = LuminanceBalancer.adjust_luminance(Br, y3)

        # Merge adjusted channels back into complete images
        z1 = LuminanceBalancer.tune(t1 / ((c1 / d1) ** 0.5))
        z2 = LuminanceBalancer.tune(t2 / ((c2 / d2) ** 0.5))
        z3 = LuminanceBalancer.tune(t3 / ((c3 / d3) ** 0.5))

        Lb = LuminanceBalancer.adjust_luminance(Lb, z1)
        Lg = LuminanceBalancer.adjust_luminance(Lg, z2)
        Lr = LuminanceBalancer.adjust_luminance(Lr, z3)

        # Merge adjusted channels right into complete images
        w1 = LuminanceBalancer.tune(t1 / ((a1 / b1) ** 0.5))
        w2 = LuminanceBalancer.tune(t2 / ((a2 / b2) ** 0.5))
        w3 = LuminanceBalancer.tune(t3 / ((a3 / b3) ** 0.5))

        Rb = LuminanceBalancer.adjust_luminance(Rb, w1)
        Rg = LuminanceBalancer.adjust_luminance(Rg, w2)
        Rr = LuminanceBalancer.adjust_luminance(Rr, w3)

        # Merge adjusted channels back into complete images
        adjusted_frames = [
            cv2.merge((Fb, Fg, Fr)),
            cv2.merge((Bb, Bg, Br)),
            cv2.merge((Lb, Lg, Lr)),
            cv2.merge((Rb, Rg, Rr))
        ]
        return adjusted_frames

    @staticmethod
    def make_white_balance(image):
        """Performs white balance adjustment using the average intensity of each channel"""
        B, G, R = cv2.split(image)
        m1, m2, m3 = np.mean(B), np.mean(G), np.mean(R)
        K = (m1 + m2 + m3) / 3
        B = LuminanceBalancer.adjust_luminance(B, K / m1)
        G = LuminanceBalancer.adjust_luminance(G, K / m2)
        R = LuminanceBalancer.adjust_luminance(R, K / m3)
        return cv2.merge((B, G, R))

class ImageStitcher:
    """Performs white balance adjustment using the average intensity of each channel"""
    @staticmethod
    def get_mask(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        return mask

    @staticmethod
    def get_overlap_region_mask(imA, imB):
        overlap = cv2.bitwise_and(imA, imB)
        mask = ImageStitcher.get_mask(overlap)
        mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)
        return mask

    @staticmethod
    def get_outmost_polygon_boundary(img):
        mask = ImageStitcher.get_mask(img)
        mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)
        cnts, hierarchy = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )[-2:]
        C = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[0]
        polygon = cv2.approxPolyDP(C, 0.009 * cv2.arcLength(C, True), True)
        return polygon

    @staticmethod
    def get_weight_mask_matrix(imA, imB, dist_threshold=5):
        """
        Generate a spatial weight matrix for blending two overlapping image regions
        based on proximity to each image's outer polygon boundary.
        """
        # Compute the overlap region between two images where both have valid content
        overlapMask = ImageStitcher.get_overlap_region_mask(imA, imB)
        # Invert the overlap mask to isolate non-overlapping regions from each image
        overlapMaskInv = cv2.bitwise_not(overlapMask)
        indices = np.where(overlapMask == 255) # Pixels where blending is needed

        # Extract non-overlapping regions (used for boundary estimation)
        imA_diff = cv2.bitwise_and(imA, imA, mask=overlapMaskInv)
        imB_diff = cv2.bitwise_and(imB, imB, mask=overlapMaskInv)

        # Initialize the weight matrix G using the mask of image A
        G = (ImageStitcher.get_mask(imA).astype(np.float32) / 255.0)
        G_visual = (G * 255).astype(np.uint8)
        # Extract outer polygon boundaries for both image regions
        polyA = ImageStitcher.get_outmost_polygon_boundary(imA_diff)
        polyB = ImageStitcher.get_outmost_polygon_boundary(imB_diff)
        # Draw polygon A on its image for visualization
        imA_with_poly = imA_diff.copy()
        cv2.polylines(imA_with_poly, [polyA], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw polygon B on its image for visualization
        imB_with_poly = imB_diff.copy()
        cv2.polylines(imB_with_poly, [polyB], isClosed=True, color=(0, 0, 255), thickness=2)

         # Compute weight matrix G by comparing distances to polygon A and B
        for y, x in zip(*indices):
            xy_tuple = (int(x), int(y))
            distToB = cv2.pointPolygonTest(polyB, xy_tuple, True)
            if distToB < dist_threshold:
                distToA = cv2.pointPolygonTest(polyA, xy_tuple, True)
                # Use squared distances to emphasize distance effect
                distToB **= 2
                distToA **= 2
                # Blend ratio based on distance proportion between A and B
                if distToA + distToB != 0:
                    G[y, x] = distToB / (distToA + distToB)
                else:
                    G[y, x] = 0
        return G, overlapMask

    @staticmethod
    def get_weights_and_masks(images):
        """
        Merge image sections and save intermediate results.
        Returns the final merged image.
        """
        front, left, back, right = images

        def save_image(image, filename):
            Image.fromarray(image).save(filename)

        # Merge top-left region (front-left)
        threading.Thread(target=save_image, args=(ImageStitcher.FI(front), "out_Section_Images/FI_front.png")).start()
        threading.Thread(target=save_image, args=(ImageStitcher.LI(left), "out_Section_Images/LI_left.png")).start()

        # Merge top-right region (front-right)
        G0, M0 = ImageStitcher.get_weight_mask_matrix(ImageStitcher.FI(front), ImageStitcher.LI(left))
        # Save weight matrix
        G0_visual = (G0 * 255).astype(np.uint8)
        Image.fromarray(G0_visual).save("out_Section_Images/G0_weight_LT_liverun.png")
        merged_image_LT = ImageStitcher.merge(ImageStitcher.FI(front), ImageStitcher.LI(left), G0)
        threading.Thread(target=save_image, args=(merged_image_LT, "out_Section_Images/merged_FI_LI_is_LT.png")).start()

        threading.Thread(target=save_image, args=(ImageStitcher.FII(front), "out_Section_Images/FII_front.png")).start()
        threading.Thread(target=save_image, args=(ImageStitcher.RII(right), "out_Section_Images/RII_right.png")).start()
        G1, M1 = ImageStitcher.get_weight_mask_matrix(ImageStitcher.FII(front), ImageStitcher.RII(right))
        merged_image_RT = ImageStitcher.merge(ImageStitcher.FII(front), ImageStitcher.RII(right), G1)
        threading.Thread(target=save_image, args=(merged_image_RT, "out_Section_Images/merged_FI_RII_is_RT.png")).start()

        # Merge bottom-left region (back-left)
        threading.Thread(target=save_image, args=(ImageStitcher.BIII(back), "out_Section_Images/BIII_back.png")).start()
        threading.Thread(target=save_image, args=(ImageStitcher.LIII(left), "out_Section_Images/LIII_left.png")).start()
        G2, M2 = ImageStitcher.get_weight_mask_matrix(ImageStitcher.BIII(back), ImageStitcher.LIII(left))
        merged_image_LB = ImageStitcher.merge(ImageStitcher.BIII(back), ImageStitcher.LIII(left), G2)
        threading.Thread(target=save_image, args=(merged_image_LB, "out_Section_Images/merged_BIII_LIII_is_LB.png")).start()

        # Merge bottom-right region (back-right)
        threading.Thread(target=save_image, args=(ImageStitcher.BIV(back), "out_Section_Images/BIV_back.png")).start()
        threading.Thread(target=save_image, args=(ImageStitcher.RIV(right), "out_Section_Images/RIV_right.png")).start()
        G3, M3 = ImageStitcher.get_weight_mask_matrix(ImageStitcher.BIV(back), ImageStitcher.RIV(right))
        merged_image_RB = ImageStitcher.merge(ImageStitcher.BIV(back), ImageStitcher.RIV(right), G3)
        threading.Thread(target=save_image, args=(merged_image_RB, "out_Section_Images/merged_BIV_RIV_is_RB.png")).start()

        # Assemble all regions into the final output image
        final_merged_image = np.zeros_like(front)
        np.copyto(final_merged_image[:yt, :xl], merged_image_LT)
        np.copyto(final_merged_image[:yt, xr:], merged_image_RT)
        np.copyto(final_merged_image[yb:, :xl], merged_image_LB)
        np.copyto(final_merged_image[yb:, xr:], merged_image_RB)
        np.copyto(final_merged_image[:yt, xl:xr], ImageStitcher.FM(front))
        np.copyto(final_merged_image[yb:, xl:xr], ImageStitcher.BM(back))
        np.copyto(final_merged_image[yt:yb, :xl], ImageStitcher.LM(left))
        np.copyto(final_merged_image[yt:yb, xr:], ImageStitcher.RM(right))
        threading.Thread(target=save_image, args=(ImageStitcher.FM(front), "out_Section_Images/FM.png")).start()
        threading.Thread(target=save_image, args=(ImageStitcher.BM(back), "out_Section_Images/BM.png")).start()
        threading.Thread(target=save_image, args=(ImageStitcher.LM(left), "out_Section_Images/LM.png")).start()
        threading.Thread(target=save_image, args=(ImageStitcher.RM(right), "out_Section_Images/RM.png")).start()


        threading.Thread(target=Image.fromarray(final_merged_image).save, args=("out_Section_Images/final_merged_image.png",)).start()
        return final_merged_image

    @staticmethod
    def get_weight_mask_matrix_liverun(imA, imB):
        """
        Generates a Weight Matrix (G) using Distance Transform for smooth image blending.
        Optimized to prevent Overflow and RuntimeWarnings for maximum execution speed.
        """
        # 1. Identify the overlapping region between Image A and Image B
        overlapMask = ImageStitcher.get_overlap_region_mask(imA, imB)
        overlapMaskInv = cv2.bitwise_not(overlapMask)
        
        # 2. Isolate the unique (non-overlapping) regions of each image
        # These unique areas act as the boundaries for the distance calculation
        imA_diff = cv2.bitwise_and(imA, imA, mask=overlapMaskInv)
        imB_diff = cv2.bitwise_and(imB, imB, mask=overlapMaskInv)

        # Convert unique regions into binary masks
        polyA_mask = ImageStitcher.get_mask(imA_diff)
        polyB_mask = ImageStitcher.get_mask(imB_diff)

        # 3. Compute Distance Transform
        # Measures the Euclidean distance (DIST_L2) from each pixel to the nearest boundary
        distToA = cv2.distanceTransform(255 - polyA_mask, cv2.DIST_L2, 5)
        distToB = cv2.distanceTransform(255 - polyB_mask, cv2.DIST_L2, 5)

        # 4. Normalize distance values to prevent Overflow before squaring
        # This keeps the values within a manageable range for float32 calculations
        max_d = np.max([np.max(distToA), np.max(distToB)])
        if max_d > 0:
            distToA /= max_d
            distToB /= max_d

        # 5. Square the distances to create a more natural-looking gradient transition
        distToA = np.square(distToA)
        distToB = np.square(distToB)

        # 6. Calculate the Weight Matrix G while protecting against Division by Zero
        # G represents the influence of Image B at each pixel
        denominator = distToA + distToB
        eps = 1e-6 # Small epsilon value to handle zero-denominator cases
        G = distToB / (denominator + eps)
        
        # Ensure the mask is only applied within the actual content area of Image A
        mask_A = (ImageStitcher.get_mask(imA) > 0)
        final_G = np.zeros_like(G)
        final_G[mask_A] = G[mask_A]
        
        # Clean up any potential NaN or Inf values by replacing them with a neutral 0.5
        final_G = np.nan_to_num(final_G, nan=0.5)

        return final_G.astype(np.float32), overlapMask

    @staticmethod
    def merge(imA, imB, G):
        """
        Blends two images using weight matrix G with high-speed vectorized operations.
        Fixes potential black image issues by expanding G's dimensions and preventing NaN values.
        """
        # Check G's dimensions: if it is (H, W), expand it to (H, W, 1) 
        # to allow broadcasting across the 3 BGR color channels.
        if G.ndim == 2:
            G = G[:, :, np.newaxis]
        
        # Convert images to float32 for calculation to ensure precision and prevent overflow.
        # Extract only the first 3 channels (BGR) to avoid issues with the Alpha channel.
        imA_f = imA[:, :, :3].astype(np.float32)
        imB_f = imB[:, :, :3].astype(np.float32)
        
        # Blending formula: Result = (ImageA * Weight) + (ImageB * (1 - Weight))
        # This creates a smooth gradient transition between the two images.
        res = imA_f * G + imB_f * (1.0 - G)
        
        # Clip values to the valid 0-255 range to prevent color wrapping 
        # then convert back to uint8 for standard image display.
        return np.clip(res, 0, 255).astype(np.uint8)

    @staticmethod
    def get_weights_and_masks_fast(images, G0, G1, G2, G3):
        """
        Performs high-speed 360-degree image stitching using pre-computed weight matrices (G).
        This optimized version uses NumPy slicing to avoid redundant calculations in the main loop.
        """
        # Unpack individual camera frames from the list
        front, left, back, right = images

        # 1. BLENDING CORNERS: Merge the 4 overlapping corner regions
        # Each corner is blended using its specific pre-computed weight matrix (G0-G3)
        # Slicing is used to extract only the overlapping areas (defined by xl, xr, yt, yb)
        
        # Merge Front-Left (Top-Left corner)
        merged_LT = ImageStitcher.merge(front[:yt, :xl], left[:yt, :xl], G0)
        # Merge Front-Right (Top-Right corner)
        merged_RT = ImageStitcher.merge(front[:yt, xr:], right[:yt, xr:], G1)
        # Merge Back-Left (Bottom-Left corner)
        merged_LB = ImageStitcher.merge(back[yb:, :xl], left[yb:, :xl], G2)
        # Merge Back-Right (Bottom-Right corner)
        merged_RB = ImageStitcher.merge(back[yb:, xr:], right[yb:, xr:], G3)

        # 2. INITIALIZE CANVAS: Create an empty BGR image for the final result
        # Note: We use 3 channels (BGR) to prevent transparency issues in the center region.
        h, w = front.shape[:2]
        final_merged_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 3. ASSEMBLE CORNERS: Place the blended corner pieces onto the final canvas
        final_merged_image[:yt, :xl] = merged_LT # Place Top-Left
        final_merged_image[:yt, xr:] = merged_RT # Place Top-Right
        final_merged_image[yb:, :xl] = merged_LB # Place Bottom-Left
        final_merged_image[yb:, xr:] = merged_RB # Place Bottom-Right

        # 4. ASSEMBLE SIDES: Copy the center sections from each camera directly
        # Direct NumPy slicing is the fastest way to populate non-overlapping regions.
        
        # Front-center (Top side)
        final_merged_image[:yt, xl:xr] = front[:yt, xl:xr, :3]
        # Back-center (Bottom side)
        final_merged_image[yb:, xl:xr] = back[yb:, xl:xr, :3]
        # Left-center (Left side)
        final_merged_image[yt:yb, :xl] = left[yt:yb, :xl, :3]
        # Right-center (Right side)
        final_merged_image[yt:yb, xr:] = right[yt:yb, xr:, :3]
        
        return final_merged_image

    @staticmethod
    def get_weight_mask_matrix_liverun(imA, imB, dist_threshold=5):
        # Identify overlapping region between imA and imB
        # The mask will have 255 in pixels where both images contain valid content (non-zero)
        overlapMask = ImageStitcher.get_overlap_region_mask(imA, imB)
        # Invert the overlap mask to obtain regions unique to each image
        overlapMaskInv = cv2.bitwise_not(overlapMask)
        
        # Extract non-overlapping parts from each image using the inverted mask
        # These are the exclusive areas used to estimate each image's actual boundary
        imA_diff = cv2.bitwise_and(imA, imA, mask=overlapMaskInv)
        imB_diff = cv2.bitwise_and(imB, imB, mask=overlapMaskInv)

        # Binary masks for distance transform
        polyA_mask = ImageStitcher.get_mask(imA_diff)
        polyB_mask = ImageStitcher.get_mask(imB_diff)

        # Compute Distance Transform
        distToA = cv2.distanceTransform(255 - polyA_mask, cv2.DIST_L2, 5)  # DIST_L2 = Euclidean distance
        distToB = cv2.distanceTransform(255 - polyB_mask, cv2.DIST_L2, 5)

        # Square the distances
        distToA **= 2
        distToB **= 2

        # Initialize weight matrix G
        G = ImageStitcher.get_mask(imA).astype(np.float32) / 255.0

        # Avoid division by zero
        denominator = distToA + distToB
        mask = denominator > 0  # จุดที่มีค่า denominator ไม่เป็น 0
        G[mask] = distToB[mask] / denominator[mask]

        return G, overlapMask

    @staticmethod
    def get_weights_and_masks_liverun(images):
        """
        Merge image sections and save intermediate results.
        Returns the final merged image.
        """
        front, left, back, right = images


        # Merge top-left region (front-left)
        start_merge = time.time()
        G0, M0 = ImageStitcher.get_weight_mask_matrix_liverun(ImageStitcher.FI(front), ImageStitcher.LI(left))
        merged_image_LT = ImageStitcher.merge(ImageStitcher.FI(front), ImageStitcher.LI(left), G0)

        # Merge top-right region (front-right)
        start_merge = time.time()
        G1, M1 = ImageStitcher.get_weight_mask_matrix_liverun(ImageStitcher.FII(front), ImageStitcher.RII(right))
        merged_image_RT = ImageStitcher.merge(ImageStitcher.FII(front), ImageStitcher.RII(right), G1)

        # Merge bottom-left region (back-left)
        G2, M2 = ImageStitcher.get_weight_mask_matrix_liverun(ImageStitcher.BIII(back), ImageStitcher.LIII(left))
        merged_image_LB = ImageStitcher.merge(ImageStitcher.BIII(back), ImageStitcher.LIII(left), G2)

        # Merge bottom-right region (back-right)
        G3, M3 = ImageStitcher.get_weight_mask_matrix_liverun(ImageStitcher.BIV(back), ImageStitcher.RIV(right))
        merged_image_RB = ImageStitcher.merge(ImageStitcher.BIV(back), ImageStitcher.RIV(right), G3)

        # Assemble all regions into the final output image
        final_merged_image = np.zeros_like(front)
        np.copyto(final_merged_image[:yt, :xl], merged_image_LT)
        np.copyto(final_merged_image[:yt, xr:], merged_image_RT)
        np.copyto(final_merged_image[yb:, :xl], merged_image_LB)
        np.copyto(final_merged_image[yb:, xr:], merged_image_RB)
        np.copyto(final_merged_image[:yt, xl:xr], ImageStitcher.FM(front))
        np.copyto(final_merged_image[yb:, xl:xr], ImageStitcher.BM(back))
        np.copyto(final_merged_image[yt:yb, :xl], ImageStitcher.LM(left))
        np.copyto(final_merged_image[yt:yb, xr:], ImageStitcher.RM(right))
        return final_merged_image

    @staticmethod
    def merge(imA, imB, G):
        G_expanded = np.expand_dims(G, axis=-1)
        return (imA * G_expanded + imB * (1 - G_expanded)).astype(np.uint8)

    # Image cropping functions for extracting specific regions
    @staticmethod
    def FI(front_image):
        return front_image[:yt, :xl]

    @staticmethod
    def FII(front_image):
        return front_image[:yt, xr:]

    @staticmethod
    def FM(front_image):
        return front_image[:yt, xl:xr]

    @staticmethod
    def BIII(back_image):
        return back_image[yb:, :xl]

    @staticmethod
    def BIV(back_image):
        return back_image[yb:, xr:]

    @staticmethod
    def BM(back_image):
        return back_image[yb:, xl:xr]

    @staticmethod
    def LI(left_image):
        return left_image[:yt, :xl]

    @staticmethod
    def LIII(left_image):
        return left_image[yb:, :xl]

    @staticmethod
    def LM(left_image):
        return left_image[yt:yb, :xl]

    @staticmethod
    def RII(right_image):
        return right_image[:yt, xr:]

    @staticmethod
    def RIV(right_image):
        return right_image[yb:, xr:]

    @staticmethod
    def RM(right_image):
        return right_image[yt:yb, xr:]


class ImageAdjuster:
    """Handles image blending and overlay operations."""
    """
    A utility class for merging images and applying perspective transformations.
    """
    @staticmethod
    def merge_images(images, mode='hard_overlay', alpha=0.25):
        """
        Merge multiple images using different blending modes.

        Parameters:
        images (list of np.array): List of images to be merged.
        mode (str): Blending mode ('hard_overlay' or 'alpha_blend').
        alpha (float): Alpha blending factor (only used in 'alpha_blend' mode).

        Returns:
        np.array: Merged image.
        """
        height, width, _ = images[0].shape
        merged_image = np.zeros((height, width, 3), dtype=np.uint8)

        if mode == 'hard_overlay':
            # Apply hard overlay by replacing non-zero pixels from each image
            for img in images:
                mask = (img != 0).any(axis=2)  # Check where the image is non-zero
                merged_image[mask] = img[mask]
        elif mode == 'alpha_blend':
            # Apply alpha blending to merge images smoothly
            total_weight = alpha * len(images)
            for img in images:
                merged_image = cv2.addWeighted(merged_image, 1, img, alpha, 0)
            merged_image = cv2.convertScaleAbs(merged_image * (1 / total_weight))
        
        return merged_image

    @staticmethod
    def overlay_image_perspective(background, overlay, dst_points):
        """
        Overlay an image onto a background with perspective transformation.

        Parameters:
        background (np.array): Background image.
        overlay (np.array): Image to be overlaid.
        dst_points (np.array): Destination points for the perspective transformation.

        Returns:
        np.array: Background image with overlay applied.
        """
        # Define source points from the overlay image
        src_points = np.float32([
            [0, 0],
            [overlay.shape[1] - 1, 0],
            [0, overlay.shape[0] - 1],
            [overlay.shape[1] - 1, overlay.shape[0] - 1]
        ])
        
        # Compute the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Warp the overlay image using the computed transformation
        warped_overlay = cv2.warpPerspective(overlay, matrix, (background.shape[1], background.shape[0]))

        # Extract the alpha channel and normalize it
        alpha_channel = warped_overlay[:, :, 3] / 255.0  # Convert to range [0,1]

        # Apply alpha blending to combine the overlay and background
        for c in range(0, 3):  # Iterate over color channels (B, G, R)
            background[:, :, c] = (alpha_channel * warped_overlay[:, :, c] +
                                   (1 - alpha_channel) * background[:, :, c])
        
        return background