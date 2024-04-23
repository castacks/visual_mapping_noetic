import cv2
import torch
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

class SAMExtractor:
    def __init__(self, sam_type="vit_l", sam_path="/home/physics_atv/workspace/segment_anything_models/sam_vit_l_0b3195.pth"):
        self.sam = sam_model_registry["vit_l"](checkpoint="/home/physics_atv/workspace/segment_anything_models/sam_vit_l_0b3195.pth").to('cuda')
        self.predictor = SamPredictor(self.sam)

        self.insize = torch.tensor([840, 476])

        self.yimax = int(64*(476/840))
        self.outsize = torch.tensor([64, self.yimax])

        self.n_feats = 256

    def get_features(self, img):
        """
        Args:
            img: [W x H x C] np image array
        """
        assert img.dtype == np.uint8, 'conversion expects uint8 RGB image'
        _img = cv2.resize(img, dsize=(840, 476), interpolation=cv2.INTER_AREA)
        self.predictor.set_image(_img)
        feats = self.predictor.get_image_embedding()[0, :, :self.yimax, :]
        return feats
