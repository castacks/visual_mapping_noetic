import os
import cv2
import tqdm
import yaml
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

from physics_atv_visual_mapping.image_processing.anyloc_utils import DinoV2ExtractFeatures
from physics_atv_visual_mapping.geometry_utils import TrajectoryInterpolator
from physics_atv_visual_mapping.pointcloud_colorization.torch_color_pcl_utils import *
from physics_atv_visual_mapping.utils import pose_to_htm

def normalize_img(img):
    vmin = img.view(-1, 3).min(dim=0)[0].view(1,1,3)
    vmax = img.view(-1, 3).max(dim=0)[0].view(1,1,3)
    return (img-vmin)/(vmax-vmin)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to config')
    parser.add_argument('--data_dir', type=str, required=True, help='path to KITTI-formatted dataset to process')
    parser.add_argument('--pca_fp', type=str, required=True, help='path to pca')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'r'))
    print(config)
    ## load Dino extractor
    dino = DinoV2ExtractFeatures(
        dino_model=config['dino_type'],
        layer=config['dino_layer'],
        input_size=config['image_insize'],
        device=config['device']
    )

    img_dir = os.path.join(args.data_dir, config['image']['folder'])
    img_ts = np.loadtxt(os.path.join(img_dir, 'timestamps.txt'))

    pca = torch.load(args.pca_fp, map_location=config['device'])

    dino_buf = torch.zeros(dino.output_size[1], dino.output_size[0], pca['V'].shape[-1]).to(config['device'])
    dino_cnt = 0

    for i in tqdm.tqdm(range(len(img_ts))):
        img_fp = os.path.join(img_dir, '{:06d}.png'.format(i))
        img = cv2.imread(img_fp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.

        dino_feats = dino(img)[0]
        dino_feats_centered = (dino_feats - pca['mean'].view(1,1,-1)).view(-1, dino_feats.shape[-1])
        dino_feats_pca = dino_feats_centered @ pca['V']
        dino_feats_pca = dino_feats_pca.view(dino_feats.shape[0], dino_feats.shape[1], -1)

        dino_cnt += 1
        c1 = ((dino_cnt-1) / dino_cnt)
        c2 = (1 / dino_cnt)
        dino_buf = c1 * dino_buf + c2 * dino_feats_pca

    ## lol second loop to compute the residual pca
    residual_buf = []
    for i in tqdm.tqdm(range(len(img_ts))):
        img_fp = os.path.join(img_dir, '{:06d}.png'.format(i))
        img = cv2.imread(img_fp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.

        dino_feats = dino(img)[0]
        dino_feats_centered = (dino_feats - pca['mean'].view(1,1,-1)).view(-1, dino_feats.shape[-1])
        dino_feats_pca = dino_feats_centered @ pca['V']
        dino_feats_pca = dino_feats_pca.view(dino_feats.shape[0], dino_feats.shape[1], -1)

        dino_feats_residual = dino_feats_pca - dino_buf
        residual_buf.append(dino_feats_residual.view(-1, dino_feats_residual.shape[-1]))

    residual_buf = torch.cat(residual_buf, dim=0)
    Rmean = residual_buf.mean(dim=0, keepdim=True)
    RU, RS, RV = torch.pca_lowrank(residual_buf - Rmean, q=pca['V'].shape[1])

    ##viz loop
    for i in tqdm.tqdm(range(len(img_ts))):
        img_fp = os.path.join(img_dir, '{:06d}.png'.format(i))
        img = cv2.imread(img_fp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.

        dino_feats = dino(img)[0]
        dino_feats_centered = (dino_feats - pca['mean'].view(1,1,-1)).view(-1, dino_feats.shape[-1])
        dino_feats_pca = dino_feats_centered @ pca['V']
        dino_feats_pca = dino_feats_pca.view(dino_feats.shape[0], dino_feats.shape[1], -1)

        dino_feats_res = dino_feats_pca - dino_buf
        dino_feats_res_centered = dino_feats_res - Rmean
        dino_feats_res_pca = dino_feats_res_centered @ RV

        fig, axs = plt.subplots(2, 4, figsize=(24, 12))

        extent = (
            0,
            img.shape[1],
            0,
            img.shape[0]
        )

        img1 = normalize_img(dino_feats_pca[..., :3]).cpu().numpy()
        img2 = normalize_img(dino_buf[..., :3]).cpu().numpy()
        img3 = normalize_img(dino_feats_res_pca[..., :3]).cpu().numpy()

        axs[0, 0].imshow(img)
        axs[0, 0].set_title('orig image')

        axs[0, 1].imshow(img1)
        axs[0, 1].set_title('dino image')

        axs[0, 2].imshow(img2)
        axs[0, 2].set_title('avg dino embedding')

        axs[0, 3].imshow(img3)
        axs[0, 3].set_title('dino image - avg')

        axs[1, 0].imshow(img)
        axs[1, 0].set_title('orig image')

        axs[1, 1].imshow(img, extent=extent)
        axs[1, 1].imshow(img1, alpha=0.8, extent=extent)
        axs[1, 1].set_title('dino image')

        axs[1, 2].imshow(img, extent=extent)
        axs[1, 2].imshow(img2, alpha=0.8, extent=extent)
        axs[1, 2].set_title('avg dino embedding')

        axs[1, 3].imshow(img, extent=extent)
        axs[1, 3].imshow(img3, alpha=0.8, extent=extent)
        axs[1, 3].set_title('dino image - avg')

        plt.savefig('fool/{:04d}.png'.format(i))
        plt.close()
