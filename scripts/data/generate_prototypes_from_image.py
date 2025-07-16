import os
import yaml
import time
import argparse

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from physics_atv_visual_mapping.image_processing.image_pipeline import setup_image_pipeline
from physics_atv_visual_mapping.image_processing.processing_blocks.traversability_prototypes import get_feat_img_prototype_cosine_sim
from physics_atv_visual_mapping.utils import *

"""
Create a set of image prototypes fron a folder of example images. Expecting a file structure like this:

root
    <class type>
        raw
            xxxx.png
            ...
        mask
            xxxx.png
    ...
    metadata.yaml

Where metadata.yaml is:
    <class type>:
        obstacle: <T/F>
"""

def proc_img(img):
    return torch.tensor(img_raw / 255.).permute(2,0,1).unsqueeze(0)

def proc_mask(img):
    #cast to float bc torch doesnt support bool interp
    return torch.tensor(img > 254).all(dim=-1, keepdim=True).permute(2,0,1).unsqueeze(0).float()

def compute_f1_scores(preds, labels, threshs):
    """
    Compute f1 scores for a bunch of thresholds

    Args:
        preds: [wxh] array of det scores
        labels: [wxh] array of labels (binary)
        threshs: [k] array of thresholds to try

    F1 = 2*TP / (2*TP + FP + FN)
    """
    #flatten all the image stuff to 1d
    _preds = preds.flatten().reshape(1, -1)
    _labels = labels.flatten().reshape(1, -1)
    _threshs = threshs.reshape(-1, 1)

    _classifs = _preds > _threshs #[KxWH]

    tp = (_classifs & _labels).sum(axis=-1) + 1e-10
    fp = (_classifs & ~_labels).sum(axis=-1) + 1e-10
    fn = (~_classifs & _labels).sum(axis=-1) + 1e-10
    tn = (~_classifs & ~_labels).sum(axis=-1) + 1e-10

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="path to dataset")
    parser.add_argument("--config", type=str, required=True, help="path to config")
    parser.add_argument("--viz", action='store_true', help='set this flag for debug viz')
    parser.add_argument("--add_lang", action='store_true', help='set this flag to add an additional lang proto (using the folder name)')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))

    #if config already has a pca, remove it.
    print('using first image proc pipeline block')
    config['image_processing'] = [config['image_processing'][0]]

    image_pipeline = setup_image_pipeline(config)

    dataset_config_fp = os.path.join(args.data_dir, 'metadata.yaml')
    dataset_config = yaml.safe_load(open(dataset_config_fp, 'r'))

    for obj_class, obj_metadata in dataset_config.items():
        ptypes = {k:[] for k in [
            'names',
            'embeddings',
            'is_obstacle',
            'modality'
        ]}

        obj_dir = os.path.join(args.data_dir, obj_class)
        raw_dir = os.path.join(obj_dir, 'raw')
        mask_dir = os.path.join(obj_dir, 'mask')
        results_dir = os.path.join(obj_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)

        img_fps = os.listdir(raw_dir)

        all_feat_images = []
        all_masks = []
        all_imgs = []

        if args.add_lang:
            text_embed = image_pipeline.blocks[-1].embed_text(obj_class)

            ptypes['names'].append(f"{obj_class}_text")
            ptypes['is_obstacle'].append(obj_metadata['obstacle'])
            ptypes['modality'].append('text')
            ptypes['embeddings'].append(text_embed)

        for ifp in img_fps:
            ptypes['names'].append(f"{obj_class}_{ifp}")
            ptypes['is_obstacle'].append(obj_metadata['obstacle'])
            ptypes['modality'].append('image')

            raw_fp = os.path.join(raw_dir, ifp)
            mask_fp = os.path.join(mask_dir, ifp)

            img_raw = cv2.imread(raw_fp)
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            mask_raw = cv2.imread(mask_fp)

            img = proc_img(img_raw).to(image_pipeline.device)
            I = torch.zeros(1, 3, 3) #intrinsics dont matter here
            feat_img, _ = image_pipeline.run(img, I)
            feat_img = feat_img[0].permute(1,2,0)

            #deliberately allow interpolation to only select vfm pixels fully contained in the mask
            mask = proc_mask(mask_raw)
            mask = torch.nn.functional.interpolate(
                mask,
                feat_img.shape[:2],
                mode='bilinear'
            )

            #changing this thresh affects the frac of mask pixels needed
            mask = (mask > 0.5).squeeze()

            masked_feats = feat_img[mask]

            ptype = masked_feats.mean(dim=0)
            ptypes['embeddings'].append(ptype)

            all_feat_images.append(feat_img)
            all_masks.append(mask)
            all_imgs.append(img_raw)

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

        save_fp = os.path.join(obj_dir, 'ptypes.pt')
        ptypes['embeddings'] = torch.stack(ptypes['embeddings'], dim=0)
        torch.save(ptypes, save_fp)

        ## do stats
        """
        For every {image/mask}-proto pair:
            1. compute cosine similarity
            2. compute optimal decision boundary
        """
        #[BxCxHxW]
        all_feat_imgs = torch.stack(all_feat_images, dim=0).permute(0,3,1,2)
        all_masks = torch.stack(all_masks, dim=0)
        protos = ptypes['embeddings']
        all_csim = get_feat_img_prototype_cosine_sim(all_feat_imgs, protos)

        #nprotos x n_images
        proto_threshs = np.zeros([protos.shape[0], all_feat_imgs.shape[0]])

        for i in range(all_feat_imgs.shape[0]):
            raw_img = all_imgs[i]
            feat_img = all_feat_imgs[i].cpu().numpy()
            mask = all_masks[i].cpu().numpy()
            csim = all_csim[i].cpu().numpy()

            fig, axs = plt.subplots(4, protos.shape[0], figsize=(5*protos.shape[0], 16))
            fig.suptitle(img_fps[i])
            axs[0, 0].set_ylabel('image + gt mask')
            axs[1, 0].set_ylabel('image + csim')
            axs[2, 0].set_ylabel('image + pred')
            axs[3, 0].set_ylabel('classif scores')

            for ii in range(protos.shape[0]):
                # compute f1 score for a bunch of thresholds
                img_proto_csim = csim[ii]
                det_threshs = np.linspace(0., 1., 101)

                precision, recall, f1 = compute_f1_scores(img_proto_csim, mask, det_threshs)
                best_thresh = det_threshs[f1.argmax()]
                best_det = img_proto_csim >= best_thresh
                proto_threshs[ii, i] = best_thresh

                plabel = ptypes['names'][ii]
                extent = (0, raw_img.shape[1], 0, raw_img.shape[0])

                axs[0, ii].imshow(raw_img, extent=extent)
                axs[0, ii].imshow(mask, cmap='coolwarm', extent=extent, alpha=0.3)
                axs[0, ii].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

                axs[1, ii].imshow(raw_img, extent=extent)
                axs[1, ii].imshow(img_proto_csim, cmap='coolwarm', extent=extent, alpha=0.3)
                axs[1, ii].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

                axs[2, ii].imshow(raw_img, extent=extent)
                axs[2, ii].imshow(best_det, cmap='coolwarm', extent=extent, alpha=0.3)
                axs[2, ii].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

                axs[3, ii].plot(det_threshs, precision, c='b', label='precision')
                axs[3, ii].plot(det_threshs, recall, c='g', label='recall')
                axs[3, ii].plot(det_threshs, f1, c='r', label='f1')
                axs[3, ii].axvline(best_thresh, color='r', label=f'best f1')

                if ii == 0:
                    axs[3, ii].legend()

                axs[0, ii].set_title(f"{plabel}\n(thr={best_thresh:.2f}, f1={f1.max():.2f})")

            fig.tight_layout()
            plt.savefig(os.path.join(results_dir, img_fps[i]), bbox_inches='tight')
            plt.close()

        proto_mean = proto_threshs.mean(axis=-1)
        proto_std = proto_threshs.std(axis=-1)
        fig, axs = plt.subplots(protos.shape[0], 1, figsize=(12, protos.shape[0] * 4))
        fig.suptitle('Proto thresh mean/var (want low var)')

        for i in range(protos.shape[0]):
            pname = ptypes['names'][i]
            is_img = ptypes['modality'][i] == 'image'

            if is_img:
                img_fp = os.path.join(raw_dir, pname.split('_')[-1])
                img = cv2.imread(img_fp)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                axs[i].imshow(img)
            else:
                axs[i].imshow(np.zeros_like(raw_img))

            axs[i].set_ylabel(f"{pname}\nthresh = {proto_mean[i]:.2f}+-{proto_std[i]:.2f}", rotation=0, labelpad=80)
            axs[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        fig.tight_layout()
        plt.savefig(os.path.join(results_dir, 'det_threshs.png'))
        plt.close()