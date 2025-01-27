import os
import torch
import cv2
import numpy as np

from physics_atv_visual_mapping.image_processing.processing_blocks.base import (
    ImageProcessingBlock,
)


class ThermalBlock(ImageProcessingBlock):
    """
    Image processing block that convert 16bit thermal to 8bit color
    """

    def __init__(self, process, enhance, rectify, distortion, device):
        self.process = process
        self.enhance = enhance
        self.rectify = rectify
        self.distortion = distortion
        print('thermal initialized')
    
    def rectify_image(self, image, intrinsics):
        distortion_coeffs = np.array(self.distortion)
        rectified_image = cv2.undistort(image, intrinsics, distortion_coeffs)
        
        return rectified_image

    def enhance_image(self, image):
        # Expects 8-bit. Best: after hist_99, do clahe + bilateral filtering
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe = clahe.apply(image)
        bilateral = cv2.bilateralFilter(clahe, 5, 20, 15)
        return bilateral

    def process_image(self, image_in, type):
        if type == "minmax":
            image_out = (image_in - np.min(image_in)) / (np.max(image_in) - np.min(image_in)) * 255
        elif type == "hist_99":
            if np.max(image_in) < 35000:
                image_out = (image_in - np.min(image_in)) / (np.max(image_in) - np.min(image_in)) * 255
            else:
                im_srt = np.sort(image_in.reshape(-1))
                upper_bound = im_srt[round(len(im_srt) * 0.99) - 1]
                lower_bound = im_srt[round(len(im_srt) * 0.01)]

                img = image_in
                img[img < lower_bound] = lower_bound
                img[img > upper_bound] = upper_bound
                image_out = ((img - lower_bound) / (upper_bound - lower_bound)) * 255.0
                image_out = image_out.astype(np.uint8)
        else:
            image_out = image_in / 255

        return image_out.astype(np.uint8)

    def run(self, image, intrinsics):
        img_out = self.process_image(image, self.process)
        if self.enhance:
            img_out = self.enhance_image(img_out)
        if self.rectify:
            img_out = self.rectify_image(img_out, intrinsics.cpu().numpy())

        return img_out, intrinsics
