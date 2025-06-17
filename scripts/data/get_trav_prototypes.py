import os
import yaml
import time
import argparse

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from physics_atv_visual_mapping.image_processing.image_pipeline import setup_image_pipeline
from physics_atv_visual_mapping.utils import *
"""
Create a traversability prototypes object
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="path to dataset")
    parser.add_argument("--config", type=str, required=True, help="path to config")
    parser.add_argument("--full_pipeline", action='store_true', help='set this flag to use the full image proc pipeline (else use first layer)')
    parser.add_argument("--save_to", type=str, required=True)
    parser.add_argument("--viz", action='store_true', help='set this flag for debug viz')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))

    #if config already has a pca, remove it.
    if args.full_pipeline:
        print('using full image proc pipeline')
    else:
        print('using first image proc pipeline block')
        config['image_processing'] = [config['image_processing'][0]]

    image_pipeline = setup_image_pipeline(config)

    #set up io
    obstacle_mask_dir = os.path.join(args.data_dir, 'obstacle_masks')
    nonobstacle_mask_dir = os.path.join(args.data_dir, 'nonobstacle_masks')
    raw_img_dir = os.path.join(args.data_dir, 'raw')

    prototypes = {
        'obstacle': [],
        'nonobstacle': []
    }

    for mfp in os.listdir(obstacle_mask_dir):
        basename = mfp.split('.')[0][:-5]
        img_raw_fp = os.path.join(raw_img_dir, "{}.png".format(basename))
        mask_fp = os.path.join(obstacle_mask_dir, mfp)

        img_raw = cv2.imread(img_raw_fp)
        mask_raw = cv2.imread(mask_fp)

        img = torch.tensor(img_raw / 255., device=image_pipeline.device).permute(2,0,1).unsqueeze(0)
        I = torch.zeros(1, 3, 3) #intrinsics dont matter here
        feat_img, _ = image_pipeline.run(img, I)
        feat_img = feat_img[0].permute(1,2,0)

        #deliberately allow interpolation to only select vfm pixels fully contained in the mask
        mask_resize = cv2.resize(mask_raw, (feat_img.shape[1], feat_img.shape[0]))

        # can change to vfm pixels touching the mask by setting threshold lower
        mask = torch.tensor(mask_resize[..., 0] > 254, device=image_pipeline.device)

        masked_feats = feat_img[mask]

        ptype = masked_feats.mean(dim=0)

        prototypes['obstacle'].append({
            'label': basename,
            'ptype': ptype.cpu()
        })

        if args.viz:
            fig, axs = plt.subplots(1, 4)
            extent = (0, img_raw.shape[1], img_raw.shape[0], 0)

            axs[0].set_title('raw image')
            axs[1].set_title('image + mask')
            axs[2].set_title('VFM image')
            axs[3].set_title('Selected VFM pixels')

            axs[0].imshow(img_raw)
            axs[1].imshow(img_raw)
            axs[1].imshow(mask_raw, alpha=0.5, extent=extent)

            feat_viz_img = normalize_dino(feat_img) 

            axs[2].imshow(feat_viz_img.cpu().numpy())

            feat_viz_img[~mask] = 0

            axs[3].imshow(feat_viz_img.cpu().numpy())

            plt.show()

    for mfp in os.listdir(nonobstacle_mask_dir):
        basename = mfp.split('.')[0][:-5]
        img_raw_fp = os.path.join(raw_img_dir, "{}.png".format(basename))
        mask_fp = os.path.join(nonobstacle_mask_dir, mfp)

        img_raw = cv2.imread(img_raw_fp)
        mask_raw = cv2.imread(mask_fp)

        img = torch.tensor(img_raw / 255., device=image_pipeline.device).permute(2,0,1).unsqueeze(0)
        I = torch.zeros(1, 3, 3) #intrinsics dont matter here
        feat_img, _ = image_pipeline.run(img, I)
        feat_img = feat_img[0].permute(1,2,0)

        #deliberately allow interpolation to only select vfm pixels fully contained in the mask
        mask_resize = cv2.resize(mask_raw, (feat_img.shape[1], feat_img.shape[0]))

        # can change to vfm pixels touching the mask by setting threshold lower
        mask = torch.tensor(mask_resize[..., 0] > 254, device=image_pipeline.device)

        masked_feats = feat_img[mask]

        ptype = masked_feats.mean(dim=0)

        prototypes['nonobstacle'].append({
            'label': basename,
            'ptype': ptype.cpu()
        })

        if args.viz:
            fig, axs = plt.subplots(1, 4)
            extent = (0, img_raw.shape[1], img_raw.shape[0], 0)

            axs[0].set_title('raw image')
            axs[1].set_title('image + mask')
            axs[2].set_title('VFM image')
            axs[3].set_title('Selected VFM pixels')

            axs[0].imshow(img_raw)
            axs[1].imshow(img_raw)
            axs[1].imshow(mask_raw, alpha=0.5, extent=extent)

            feat_viz_img = normalize_dino(feat_img) 

            axs[2].imshow(feat_viz_img.cpu().numpy())

            feat_viz_img[~mask] = 0

            axs[3].imshow(feat_viz_img.cpu().numpy())

            plt.show()

    torch.save(prototypes, args.save_to)