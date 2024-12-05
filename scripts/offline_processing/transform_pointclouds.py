import os
import tqdm
import argparse

import torch
import numpy as np
import open3d as o3d

from tartandriver_utils.geometry_utils import TrajectoryInterpolator

from physics_atv_visual_mapping.utils import *

"""
Run the dino mapping offline on the kitti-formatted dataset
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="path to dataset")
    parser.add_argument(
        "--odom", type=str, required=False, default="odom", help="name of odom folder"
    )
    parser.add_argument(
        "--pcl", type=str, required=False, default="pcl", help="name of pcl folder"
    )
    parser.add_argument(
        "--output_dir", type=str, required=False, default="pcl_odom", help="name of pcl folder"
    )
    parser.add_argument(
        "--debug", action='store_true', help="whether to show debug viz"
    )
    args = parser.parse_args()

    # setup io
    save_dir = os.path.join(args.run_dir, args.output_dir)
    os.makedirs(save_dir, exist_ok=True)

    ## first create the trajectory interpolator
    odom_dir = os.path.join(args.run_dir, args.odom)
    poses = np.load(os.path.join(odom_dir, "odometry.npy"))
    pose_ts = np.loadtxt(os.path.join(odom_dir, "timestamps.txt"))
    mask = np.ones_like(pose_ts).astype(bool)
    mask[1:] = np.abs(pose_ts[1:] - pose_ts[:-1]) > 1e-4
    poses = poses[mask]
    pose_ts = pose_ts[mask]

    traj_interp = TrajectoryInterpolator(pose_ts, poses, tol=0.5)

    ## next compute gridmap sample times
    pcl_dir = os.path.join(args.run_dir, args.pcl)

    pcl_ts = np.loadtxt(os.path.join(pcl_dir, "timestamps.txt"))
    
    #copy timestamps
    np.savetxt(os.path.join(save_dir, "timestamps.txt"), pcl_ts)

    for ii in tqdm.tqdm(range(len(pcl_ts))):
        pcl_t = pcl_ts[ii]
        pcl_fp = os.path.join(args.run_dir, args.pcl, "{:06d}.npy".format(ii))
        save_fp = os.path.join(save_dir, "{:06d}.npy".format(ii))

        pcl = torch.from_numpy(np.load(pcl_fp)).float()

        pose = traj_interp(pcl_t)
        pose = pose_to_htm(pose)

        pcl_odom = transform_points(pcl, pose)

        np.save(save_fp, pcl_odom)

    #debug viz:
    if args.debug:
        viz_agg = []
        for ii in tqdm.tqdm(range(len(pcl_ts))):
            pcl_fp = os.path.join(save_dir, "{:06d}.npy".format(ii))
            pts = np.load(pcl_fp)

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(pts[:, :3])

            if pts.shape[1] == 6:
                pc.colors = o3d.utility.Vector3dVector(pts[:, 3:])

            viz_agg.append(pc)

            if ii % 50 == 0:
                o3d.visualization.draw_geometries(viz_agg)