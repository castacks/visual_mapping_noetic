import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from physics_atv_visual_mapping.image_processing.processing_blocks.base import ImageProcessingBlock

# from vfm.models.base import LanguageImageModel
# from vfm.utils import load_image

class RADIOLangBlock:
    def __init__(self, models_dir, adaptor, device):
        self.adaptor = adaptor
        torch.hub.set_dir(os.path.join(models_dir, "torch_hub"))
        radio_fp = os.path.join(models_dir, "torch_hub", "NVlabs_RADIO_main")
        self.model = self._load_model(radio_fp)
        self.adaptor = self.model.adaptors[self.adaptor]  # type: ignore
        self.device = device

    def _load_model(self, radio_fp) -> torch.nn.Module:
        model = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version="c-radio_v3-b",
            progress=True,
            skip_validation=True,
            adaptor_names=[self.adaptor],  # TODO: remove when fixed
        )
        return model.cuda().eval()  # type: ignore

    def _preprocess_image(self, image):
        res = self.model.get_nearest_supported_resolution(*image.shape[-2:])  # type: ignore
        return F.interpolate(image, res, mode="bilinear", align_corners=False)

    def run(self, image, intrinsics, image_orig):
        img_out = self.embed_image(image)

        ix = image.shape[3]
        dx = img_out.shape[3]
        iy = image.shape[2]
        dy = img_out.shape[2]

        intrinsics[:, 0, 0] *= dx / ix
        intrinsics[:, 0, 2] *= dx / ix

        intrinsics[:, 1, 1] *= dy / iy
        intrinsics[:, 1, 2] *= dy / iy

        return img_out, intrinsics

    def _preprocess_text(self, txt: str) -> Tensor:
        with torch.no_grad():
            tokens = self.adaptor.tokenizer([txt]).cuda()  # type: ignore
            # tokens["input_ids"] = tokens["input_ids"].cuda()
        return tokens

    def embed_image(self, image) -> Tensor:
        image_tensor = self._preprocess_image(image)
        B, C, H, W = image_tensor.shape
        with torch.no_grad():
            PATCHED_H, PATCHED_W = (
                H // self.model.patch_size,
                W // self.model.patch_size,
            )  # type: ignore
            feats = (
                self.model(image_tensor)["backbone"]
                .features.reshape(B, PATCHED_H, PATCHED_W, -1)
            )
            feats = self.adaptor.head_mlp(feats)  # type: ignore

        feats = F.normalize(feats, dim=-1)

        #[BxWxHxC] -> [BxCxWxH]
        return feats.permute(0,3,1,2)

    def embed_text(self, txt: str) -> Tensor:
        with torch.no_grad():
            tokens = self._preprocess_text(txt)
            embeddings = self.adaptor.encode_text(tokens, normalize=True).squeeze()  # type: ignore
        return embeddings
