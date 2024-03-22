import os
import yaml
import tqdm
import time
import argparse

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from physics_atv_visual_mapping.image_processing.image_pipeline import setup_image_pipeline
from physics_atv_visual_mapping.pointcloud_colorization.torch_color_pcl_utils import *
from physics_atv_visual_mapping.localmapping.localmapping import *
from physics_atv_visual_mapping.utils import *
from physics_atv_visual_mapping.geometry_utils import TrajectoryInterpolator

"""
Run the dino mapping offline on the kitti-formatted dataset
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help='path to dataset')
    parser.add_argument('--config', type=str, required=True, help='path to config')
    parser.add_argument('--odom', type=str, required=False, default='odom', help='name of odom folder')
    parser.add_argument('--pcl', type=str, required=False, default='pcl', help='name of pcl folder')
    parser.add_argument('--image', type=str, required=False, default='image_left_color', help='name of image folder')
    parser.add_argument('--output_folder', type=str, required=False, default='local_dino_map', help='name of result folder')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'r'))

    #setup io
    os.makedirs(os.path.join(args.run_dir, args.output_folder), exist_ok=True)

    intrinsics = torch.tensor(config['intrinsics']['K']).reshape(3, 3).to(config['device'])
    extrinsics = pose_to_htm(np.concatenate([
            np.array(config['extrinsics']['p']),
            np.array(config['extrinsics']['q'])
        ], axis=-1)).to(config['device'])

    image_pipeline = setup_image_pipeline(config)

    ## first create the trajectory interpolator
    odom_dir = os.path.join(args.run_dir, args.odom)
    poses = np.load(os.path.join(odom_dir, 'odometry.npy'))
    pose_ts = np.loadtxt(os.path.join(odom_dir, 'timestamps.txt'))
    mask = np.abs(pose_ts[1:]-pose_ts[:-1]) > 1e-4
    poses = poses[1:][mask]
    pose_ts = pose_ts[1:][mask]

    traj_interp = TrajectoryInterpolator(pose_ts, poses, tol=0.5)

    ## next compute gridmap sample times
    pcl_dir = os.path.join(args.run_dir, args.pcl)
    image_dir = os.path.join(args.run_dir, args.image)

    pcl_ts = np.loadtxt(os.path.join(pcl_dir, 'timestamps.txt'))
    image_ts = np.loadtxt(os.path.join(image_dir, 'timestamps.txt'))

    # timestamp check
#    x = np.arange(len(pose_ts))
#    plt.scatter(x, pose_ts, label='pose ({} samples)'.format(pose_ts.shape[0]))
#    plt.scatter(x, pcl_ts, label='pcl ({} samples)'.format(pcl_ts.shape[0]))
#    plt.scatter(x, image_ts, label='image ({} samples)'.format(image_ts.shape[0]))
#    plt.legend()
#    plt.title('timestamp check (all colors should overlap)')
#    plt.show()

    # sync pcl to image
    pcl_to_image_tdiffs = np.abs(pcl_ts.reshape(1, -1) - image_ts.reshape(-1, 1))
    image_idxs = np.argmin(pcl_to_image_tdiffs, axis=0)
    image_errs = np.min(pcl_to_image_tdiffs, axis=0)

    print('image timesync errs: mean: {:.4f}s, max: {:.4f}s'.format(image_errs.mean(), image_errs.max()))

    ## finally, actually run the mapping
    localmap_agg = None

    for pi in tqdm.tqdm(range(len(pcl_ts))):
        pcl_fp = os.path.join(args.run_dir, args.pcl, '{:06d}.npy'.format(pi))
        image_fp = os.path.join(args.run_dir, args.image, '{:06d}.png'.format(image_idxs[pi]))

        metadata = config['localmapping']['metadata']
        pcl = torch.from_numpy(np.load(pcl_fp)).float().to(config['device'])
        img = cv2.imread(image_fp)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
        img = torch.tensor(img).unsqueeze(0).permute(0,3,1,2)

        #run the image through pipeline
        t1 = time.time()

        dino_img, dino_intrinsics = image_pipeline.run(img, intrinsics.unsqueeze(0))

        #move back to channels-last
        dino_img = dino_img[0].permute(1,2,0)
        dino_intrinsics = dino_intrinsics[0]

        t2 = time.time()

        I = get_intrinsics(dino_intrinsics).to(config['device'])
        E = get_extrinsics(extrinsics).to(config['device'])

        P = obtain_projection_matrix(I, E)

        pixel_coordinates = get_pixel_from_3D_source(pcl[:, :3], P)
        lidar_points_in_frame, pixels_in_frame, ind_in_frame = get_points_and_pixels_in_frame(
            pcl[:, :3],
            pixel_coordinates,
            dino_img.shape[0],
            dino_img.shape[1]
        )

        dino_features = dino_img[pixels_in_frame[:, 1], pixels_in_frame[:, 0]]
        dino_pcl = torch.cat([pcl[ind_in_frame][:, :3], dino_features], dim=-1)

        t3 = time.time()

        pose = traj_interp(pcl_ts[pi])
        pose = pose_to_htm(pose).to(config['device'])
        dino_pcl = transform_points(dino_pcl, pose)

        _metadata = {
            'origin': torch.tensor(metadata['origin']).float().to(config['device']),
            'length_x': torch.tensor(metadata['length_x']).float().to(config['device']),
            'length_y': torch.tensor(metadata['length_y']).float().to(config['device']),
            'resolution': torch.tensor(metadata['resolution']).float().to(config['device']),
        }
        _metadata['origin'] += pose[:2, -1]

        localmap, known_mask, metadata_out = localmap_from_pointcloud(dino_pcl[:, :3], dino_pcl[:, 3:], _metadata)
        localmap = {
            'data': localmap,
            'known': known_mask,
            'metadata': metadata_out
        }

        t4 = time.time()

        if localmap_agg is None:
            localmap_agg = localmap
        else:
            localmap_agg = aggregate_localmaps(localmap, localmap_agg, ema=config['localmapping']['ema'])

        t5 = time.time()

        ## save outputs
#        out_data_fp = os.path.join(args.run_dir, args.output_folder, '{:06d}_data.npy'.format(pi))
#        out_metadata_fp = os.path.join(args.run_dir, args.output_folder, '{:06d}_metadata.yaml'.format(pi))

        #store gridmaps as channels-first
#        data_out = localmap_agg['data'].permute(2,0,1).cpu().numpy()
        metadata_out = {
            'origin': localmap_agg['metadata']['origin'].cpu().numpy().tolist(),
            'width': localmap_agg['metadata']['length_x'].item(),
            'height': localmap_agg['metadata']['length_y'].item(),
            'resolution': localmap_agg['metadata']['resolution'].item(),
#            'feature_keys': ['dino_{}'.format(i) for i in range(data_out.shape[0])],
        }

#        np.save(out_data_fp, data_out)
#        with open(out_metadata_fp, 'w') as fh:
#            yaml.dump(metadata_out, fh)

        t6 = time.time()

        if pi % 10 == 0:
            img_viz = img[0].permute(1,2,0).cpu()
            dino_img_viz = dino_img.cpu()

            extent = (
                metadata_out['origin'][0],
                metadata_out['origin'][0] + metadata_out['width'],
                metadata_out['origin'][1],
                metadata_out['origin'][1] + metadata_out['height']
            )

            img_extent = (
                0.,
                dino_img_viz.shape[1],
                0.,
                dino_img_viz.shape[0],
            )


            fig, axs = plt.subplots(2, 3, figsize=(36, 12))
            axs = axs.T.flatten()

            axs[0].imshow(img_viz, extent=img_extent)

            axs[1].imshow(img_viz, extent=img_extent)
            axs[1].scatter(pixel_coordinates[:, 0][ind_in_frame].cpu(), dino_img_viz.shape[0]-pixel_coordinates[:, 1][ind_in_frame].cpu(), s=0.1, c='r', alpha=0.3)

#            axs[2].imshow(img_viz, extent=img_extent)
            axs[2].imshow(normalize_dino(dino_img_viz).cpu(), extent=img_extent, alpha=1.0)

            axs[3].imshow(normalize_dino(localmap_agg['data']).permute(1,0,2).cpu(), origin='lower', extent=extent)
            axs[3].scatter(pose[0, -1].cpu(), pose[1, -1].cpu(), c='r', marker='x')

            axs[4].imshow(img_viz, extent=img_extent)
            axs[4].imshow(normalize_dino(dino_img_viz).cpu(), extent=img_extent, alpha=0.7)
            axs[4].scatter(pixel_coordinates[:, 0][ind_in_frame].cpu(), dino_img_viz.shape[0]-pixel_coordinates[:, 1][ind_in_frame].cpu(), s=0.1, c='r', alpha=0.8)

            axs[5].scatter(dino_pcl[:, 0].cpu(), dino_pcl[:, 1].cpu(), c='r', s=0.1, alpha=1.0)
            axs[5].set_xlim(extent[0], extent[1])
            axs[5].set_ylim(extent[2], extent[3])
            axs[5].set_aspect(1.)

            plt.savefig('bbb/{:04d}.png'.format(pi))
            plt.close()

        t7 = time.time()

        print('TIMING:')
        print('\t image proc:    {:.4f}s'.format(t2-t1))
        print('\t pcl project:   {:.4f}s'.format(t3-t2))
        print('\t localmap gen:  {:.4f}s'.format(t4-t3))
        print('\t localmap agg:  {:.4f}s'.format(t5-t4))
        print('\t data save:     {:.4f}s'.format(t6-t5))
        print('\t fig gen:       {:.4f}s'.format(t7-t6))
