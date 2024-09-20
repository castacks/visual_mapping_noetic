import os
import torch

from physics_atv_visual_mapping.image_processing.processing_blocks.base import ImageProcessingBlock

class FeatUpBlock(ImageProcessingBlock):
    """
    Image processing block that runs featup on the image
    """
    def __init__(self, fp, device):
        self.featup = torch.load(fp, map_location=device)
        self.featup.eval()

    def run(self, image, intrinsics, image_orig):
        with torch.no_grad():
            img_out, _ = self.featup(image, image_orig)

        ix = image.shape[3]
        dx = img_out.shape[3]
        iy = image.shape[2]
        dy = img_out.shape[2]

        intrinsics[:, 0, 0] *= (dx/ix)
        intrinsics[:, 0, 2] *= (dx/ix)

        intrinsics[:, 1, 1] *= (dy/iy)
        intrinsics[:, 1, 2] *= (dy/iy)

        return img_out, intrinsics
