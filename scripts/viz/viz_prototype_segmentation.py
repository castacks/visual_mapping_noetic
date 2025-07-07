import os
import yaml
import time
import argparse

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from physics_atv_visual_mapping.image_processing.processing_blocks.radio_lang import RADIOLangBlock
from physics_atv_visual_mapping.image_processing.processing_blocks.traversability_prototypes import TraversabilityPrototypesBlock
from physics_atv_visual_mapping.image_processing.image_pipeline import setup_image_pipeline
from physics_atv_visual_mapping.utils import *
"""
Create a traversability prototypes object
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="path to dataset")
    parser.add_argument("--config", type=str, required=True, help="path to config")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))

    image_pipeline = setup_image_pipeline(config)

    assert len(image_pipeline.blocks) == 2, "image pipeline is not RADIOLang + TravPtypes!"

    radio_block = image_pipeline.blocks[0]
    ptype_block = image_pipeline.blocks[1]

    ptype_keys = ptype_block.ptype_keys

    assert isinstance(radio_block, RADIOLangBlock), "first block is not RADIOLangBlock!"
    assert isinstance(ptype_block, TraversabilityPrototypesBlock), "second block is not TraversabilityPrototypesBlock!"

    for img_fp in os.listdir(args.data_dir):
        img = cv2.imread(os.path.join(args.data_dir, img_fp))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.

        #[BxCxWxH]
        img = torch.tensor(img, device=image_pipeline.device).unsqueeze(0).permute(0,3,1,2)

        feat_img, _ = image_pipeline.run(img, torch.rand(1, 3, 3))

        img = img[0]
        feat_img = feat_img[0]
        seg_prob_img = feat_img.softmax(dim=0) #not really correct to do this as classes may not be exhaustive
        seg_img = seg_prob_img.argmax(dim=0)

        is_det = feat_img.max(dim=0)[0] > 0.25
        seg_img[~is_det] = -1

        fig, axs = plt.subplots(1, 2+len(ptype_keys))
        axs[0].imshow(img.permute(1,2,0).cpu().numpy())
        axs[1].imshow(seg_img.cpu().numpy(), cmap='tab10')

        for i, pk in enumerate(ptype_keys):
            axs[2+i].set_title(pk)
            axs[2+i].imshow(feat_img[i].cpu().numpy(), vmin=0.0, vmax=0.25)

        plt.show()