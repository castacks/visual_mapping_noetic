import os
import yaml
import tqdm
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
    parser.add_argument('--gridmap', type=str, required=False, default='local_gridmap', help='name of gridmap folder')
    parser.add_argument('--output_folder', type=str, required=False, default='local_dino_map', help='name of result folder')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'r'))

    #setup io
    # os.makedirs(os.path.join(args.run_dir, args.output_folder), exist_ok=True)
    exps = os.listdir(args.run_dir)
    for exp in exps:
        run_dir = os.path.join(args.run_dir,exp)
        os.makedirs(os.path.join(run_dir, args.output_folder), exist_ok=True)

        intrinsics = torch.tensor(config['intrinsics']['K']).reshape(3, 3).to(config['device'])
        # extrinsics = pose_to_htm(np.concatenate([
        #         np.array(config['extrinsics']['p']),
        #         np.array(config['extrinsics']['q'])
        #     ], axis=-1)).to(config['device'])

        extrinsics = torch.Tensor([[ 0.0519, -0.9984, -0.0213,  0.0000],
            [-0.2505,  0.0076, -0.9681,  0.4000],
            [ 0.9667,  0.0555, -0.2497, -0.3000],
            [ 0.0000,  0.0000,  0.0000,  1.0000]])

        image_pipeline = setup_image_pipeline(config)

        ## first create the trajectory interpolator
        odom_dir = os.path.join(run_dir, args.odom)
        poses = np.load(os.path.join(odom_dir, 'odometry.npy'))
        pose_ts = np.loadtxt(os.path.join(odom_dir, 'timestamps.txt'))
        mask = np.abs(pose_ts[1:]-pose_ts[:-1]) > 1e-4
        poses = poses[1:][mask]
        pose_ts = pose_ts[1:][mask]

        traj_interp = TrajectoryInterpolator(pose_ts, poses, tol=0.5)

        ## next compute gridmap sample times
        gridmap_dir = os.path.join(run_dir, args.gridmap)
        pcl_dir = os.path.join(run_dir, args.pcl)
        image_dir = os.path.join(run_dir, args.image)

        gridmap_ts = np.loadtxt(os.path.join(gridmap_dir, 'timestamps.txt'))
        pcl_ts = np.loadtxt(os.path.join(pcl_dir, 'timestamps.txt'))
        image_ts = np.loadtxt(os.path.join(image_dir, 'timestamps.txt'))

        # timestamp check
    #    x = np.arange(len(pose_ts))
    #    plt.scatter(x, pose_ts, label='pose ({} samples)'.format(pose_ts.shape[0]))
    #    plt.scatter(x, gridmap_ts, label='gridmap ({} samples)'.format(gridmap_ts.shape[0]))
    #    plt.scatter(x, pcl_ts, label='pcl ({} samples)'.format(pcl_ts.shape[0]))
    #    plt.scatter(x, image_ts, label='image ({} samples)'.format(image_ts.shape[0]))
    #    plt.legend()
    #    plt.title('timestamp check (all colors should overlap)')
    #    plt.show()

        #sync the gridmap times with image and pcl (note that I'm choosing to break causality for accuracy)
        gridmap_to_pcl_tdiffs = np.abs(gridmap_ts.reshape(1, -1) - pcl_ts.reshape(-1, 1))
        pcl_idxs = np.argmin(gridmap_to_pcl_tdiffs, axis=0)
        pcl_errs = np.min(gridmap_to_pcl_tdiffs, axis=0)

        gridmap_to_image_tdiffs = np.abs(gridmap_ts.reshape(1, -1) - image_ts.reshape(-1, 1))
        image_idxs = np.argmin(gridmap_to_image_tdiffs, axis=0)
        image_errs = np.min(gridmap_to_image_tdiffs, axis=0)

        print('pcl timesync errs: mean: {:.4f}s, max: {:.4f}s'.format(pcl_errs.mean(), pcl_errs.max()))
        print('image timesync errs: mean: {:.4f}s, max: {:.4f}s'.format(image_errs.mean(), image_errs.max()))

        ## finally, actually run the mapping
        localmap_agg = None

        for gi in tqdm.tqdm(range(len(gridmap_ts))):
            metadata_fp = os.path.join(run_dir, args.gridmap, '{:06d}_metadata.yaml'.format(gi))
            pcl_fp = os.path.join(run_dir, args.pcl, '{:06d}.npy'.format(pcl_idxs[gi]))
            image_fp = os.path.join(run_dir, args.image, '{:06d}.png'.format(image_idxs[gi]))

            metadata = yaml.safe_load(open(metadata_fp, 'r'))
            pcl = torch.from_numpy(np.load(pcl_fp)).float().to(config['device'])
            img = cv2.imread(image_fp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
            img = torch.tensor(img).unsqueeze(0).permute(0,3,1,2)

            dino_img, dino_intrinsics = image_pipeline.run(img, intrinsics.unsqueeze(0))

            #move back to channels-last
            dino_img = dino_img[0].permute(1,2,0)
            dino_intrinsics = dino_intrinsics[0]

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

            pose = traj_interp(gridmap_ts[gi])
            pose = pose_to_htm(pose).to(config['device'])
            dino_pcl = transform_points(dino_pcl, pose)

            _metadata = {
                'origin': torch.tensor(metadata['origin']).float().to(config['device']),
                'length_x': torch.tensor(metadata['width']).float().to(config['device']),
                'length_y': torch.tensor(metadata['height']).float().to(config['device']),
                'resolution': torch.tensor(metadata['resolution']).float().to(config['device']),
            }

            localmap, known_mask, metadata_out = localmap_from_pointcloud(dino_pcl[:, :3], dino_pcl[:, 3:], _metadata)
            localmap = {
                'data': localmap,
                'known': known_mask,
                'metadata': metadata_out
            }

            if localmap_agg is None:
                localmap_agg = localmap
            else:
                localmap_agg = aggregate_localmaps(localmap, localmap_agg, ema=config['localmapping']['ema'])

            ## save outputs
            out_data_fp = os.path.join(run_dir, args.output_folder, '{:06d}_data.npy'.format(gi))
            out_metadata_fp = os.path.join(run_dir, args.output_folder, '{:06d}_metadata.yaml'.format(gi))

            #store gridmaps as channels-first
            data_out = localmap_agg['data'].permute(2,0,1).cpu().numpy()
            metadata_out = {
                'origin': localmap_agg['metadata']['origin'].cpu().numpy().tolist(),
                'width': localmap_agg['metadata']['length_x'].item(),
                'height': localmap_agg['metadata']['length_y'].item(),
                'resolution': localmap_agg['metadata']['resolution'].item(),
                'feature_keys': ['dino_{}'.format(i) for i in range(data_out.shape[0])],
            }

            np.save(out_data_fp, data_out)
            with open(out_metadata_fp, 'w') as fh:
                yaml.dump(metadata_out, fh)

    #        if gi % 100 == 0:
    #            plt.imshow(normalize_dino(localmap_agg['data']).cpu());plt.show()
