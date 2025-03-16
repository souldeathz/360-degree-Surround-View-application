import cv2
import numpy as np
from PIL import Image
from param_settings import xl, xr, yt, yb
import threading


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
        overlapMask = ImageStitcher.get_overlap_region_mask(imA, imB)
        overlapMaskInv = cv2.bitwise_not(overlapMask)
        indices = np.where(overlapMask == 255)

        imA_diff = cv2.bitwise_and(imA, imA, mask=overlapMaskInv)
        imB_diff = cv2.bitwise_and(imB, imB, mask=overlapMaskInv)

        G = (ImageStitcher.get_mask(imA).astype(np.float32) / 255.0)
        polyA = ImageStitcher.get_outmost_polygon_boundary(imA_diff)
        polyB = ImageStitcher.get_outmost_polygon_boundary(imB_diff)

        for y, x in zip(*indices):
            xy_tuple = (int(x), int(y))
            distToB = cv2.pointPolygonTest(polyB, xy_tuple, True)
            if distToB < dist_threshold:
                distToA = cv2.pointPolygonTest(polyA, xy_tuple, True)
                # ปรับค่าด้วยกำลังสอง
                distToB **= 2
                distToA **= 2
                G[y, x] = distToB / (distToA + distToB)
        return G, overlapMask

    @staticmethod
    def merge(imA, imB, G):
        G_expanded = np.expand_dims(G, axis=-1)
        return (imA * G_expanded + imB * (1 - G_expanded)).astype(np.uint8)

    # ฟังก์ชันตัดส่วนภาพ (crop functions)
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

    @staticmethod
    def get_weights_and_masks(images):
        """
        รวมภาพในแต่ละส่วนและบันทึกไฟล์ผลลัพธ์
        คืนค่า final merged image
        """
        front, left, back, right = images

        def save_image(image, filename):
            Image.fromarray(image).save(filename)

        # บันทึกภาพ crop เพื่อตรวจสอบ (สามารถ comment ได้)
        threading.Thread(target=save_image, args=(ImageStitcher.FI(front), "out_Section_Images/FI_front.png")).start()
        threading.Thread(target=save_image, args=(ImageStitcher.LI(left), "out_Section_Images/LI_left.png")).start()

        # รวมภาพซ้ายบน
        G0, M0 = ImageStitcher.get_weight_mask_matrix(ImageStitcher.FI(front), ImageStitcher.LI(left))
        merged_image_LT = ImageStitcher.merge(ImageStitcher.FI(front), ImageStitcher.LI(left), G0)
        threading.Thread(target=save_image, args=(merged_image_LT, "out_Section_Images/merged_FI_LI_is_LT.png")).start()

        # รวมภาพขวาบน
        threading.Thread(target=save_image, args=(ImageStitcher.FII(front), "out_Section_Images/FII_front.png")).start()
        threading.Thread(target=save_image, args=(ImageStitcher.RII(right), "out_Section_Images/RII_right.png")).start()
        G1, M1 = ImageStitcher.get_weight_mask_matrix(ImageStitcher.FII(front), ImageStitcher.RII(right))
        merged_image_RT = ImageStitcher.merge(ImageStitcher.FII(front), ImageStitcher.RII(right), G1)
        threading.Thread(target=save_image, args=(merged_image_RT, "out_Section_Images/merged_FI_RII_is_RT.png")).start()

        # รวมภาพซ้ายล่าง
        threading.Thread(target=save_image, args=(ImageStitcher.BIII(back), "out_Section_Images/BIII_back.png")).start()
        threading.Thread(target=save_image, args=(ImageStitcher.LIII(left), "out_Section_Images/LIII_left.png")).start()
        G2, M2 = ImageStitcher.get_weight_mask_matrix(ImageStitcher.BIII(back), ImageStitcher.LIII(left))
        merged_image_LB = ImageStitcher.merge(ImageStitcher.BIII(back), ImageStitcher.LIII(left), G2)
        threading.Thread(target=save_image, args=(merged_image_LB, "out_Section_Images/merged_BIII_LIII_is_LB.png")).start()

        # รวมภาพขวาล่าง
        threading.Thread(target=save_image, args=(ImageStitcher.BIV(back), "out_Section_Images/BIV_back.png")).start()
        threading.Thread(target=save_image, args=(ImageStitcher.RIV(right), "out_Section_Images/RIV_right.png")).start()
        G3, M3 = ImageStitcher.get_weight_mask_matrix(ImageStitcher.BIV(back), ImageStitcher.RIV(right))
        merged_image_RB = ImageStitcher.merge(ImageStitcher.BIV(back), ImageStitcher.RIV(right), G3)
        threading.Thread(target=save_image, args=(merged_image_RB, "out_Section_Images/merged_BIV_RIV_is_RB.png")).start()

        # บรรจุภาพที่ไม่ได้ merge (FM, BM, LM, RM)
        final_merged_image = np.zeros_like(front)
        np.copyto(final_merged_image[:yt, :xl], merged_image_LT)
        np.copyto(final_merged_image[:yt, xr:], merged_image_RT)
        np.copyto(final_merged_image[yb:, :xl], merged_image_LB)
        np.copyto(final_merged_image[yb:, xr:], merged_image_RB)
        np.copyto(final_merged_image[:yt, xl:xr], ImageStitcher.FM(front))
        np.copyto(final_merged_image[yb:, xl:xr], ImageStitcher.BM(back))
        np.copyto(final_merged_image[yt:yb, :xl], ImageStitcher.LM(left))
        np.copyto(final_merged_image[yt:yb, xr:], ImageStitcher.RM(right))
        threading.Thread(target=Image.fromarray(final_merged_image).save, args=("out_Section_Images/final_merged_image.png",)).start()
        return final_merged_image


class ImageAdjuster:
    """Handles image blending and overlay operations."""

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