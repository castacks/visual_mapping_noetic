import cv2
import torch
import numpy as np

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

from physics_atv_visual_mapping.image_processing.processing_blocks.base import ImageProcessingBlock

class GANavBlock(ImageProcessingBlock):
    """
    Image processing block that from GANav for semantic logits
    """
    def __init__(self, seg_config, seg_checkpoint, device):
        self.seg_model = init_segmentor(seg_config, seg_checkpoint, device)
        self.outsize = (688, 550) #this is the size GANav runs on
        self.device = device

    def run(self, image, intrinsics, image_orig):
        seg_res = []

        #AAA GARBAGE
        for img in image:
            img = (img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, self.outsize, interpolation = cv2.INTER_AREA)

            res = inference_segmentor(self.seg_model, img, logit_in=True)
            seg = torch.tensor(res[0], device=self.device).float()
            seg_res.append(seg)

        seg_res = torch.stack(seg_res, dim=0)

        #reconpute intrinsics
        rx = seg_res.shape[3] / image.shape[3]
        ry = seg_res.shape[2] / image.shape[2]

        intrinsics[:, 0, 0] *= rx
        intrinsics[:, 0, 2] *= rx

        intrinsics[:, 1, 1] *= ry
        intrinsics[:, 1, 2] *= ry

        return seg_res.log(), intrinsics
