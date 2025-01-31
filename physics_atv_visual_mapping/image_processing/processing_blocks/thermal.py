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

    def __init__(self, process, enhance, rectify, distortion, colormap=None, device='cpu'):
        self.process = process
        self.enhance = enhance
        self.rectify = rectify
        self.distortion = distortion

        self.colormap_dict = {
            "jet": cv2.COLORMAP_JET,  # Similar to FLIR's "Ironbow"
            "inferno": cv2.COLORMAP_INFERNO,  # FLIR-like AGC coloring
            "plasma": cv2.COLORMAP_PLASMA,
            "turbo": cv2.COLORMAP_TURBO,
            "hot": cv2.COLORMAP_HOT,  # Intensity-based, strong reds
        }
        self.colormap = self.colormap_dict.get(colormap) if colormap else None

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
        # converts 16bit to 8bit
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

    def apply_colormap(self, image):
        # applies a thermal colormap for visualization
        # For now manually define the mask for non-vehicle regions
        non_vehicle_region = image[:470, :]  # Exclude bottom rows

        lower_bound = np.percentile(non_vehicle_region, 2)
        upper_bound = np.percentile(non_vehicle_region, 98)

        # Clip and normalize the entire image using non-vehicle values
        image_clipped = np.clip(image, lower_bound, upper_bound)
        image_8bit = ((image_clipped - lower_bound) / (upper_bound - lower_bound)) * 255.0
        image_8bit = image_8bit.astype(np.uint8)

        return cv2.applyColorMap(image_8bit, self.colormap)

    def run(self, image, intrinsics):
        img_out = self.process_image(image, self.process)
        if self.rectify:
            img_out = self.rectify_image(img_out, intrinsics.cpu().numpy())
        if self.enhance:
            img_out = self.enhance_image(img_out)
        if self.colormap:
            img_out = self.apply_colormap(img_out) #Output [H,W,3] now
            img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

        return img_out, intrinsics
