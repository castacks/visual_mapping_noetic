import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import os
import cv2
from scipy.spatial.transform import Rotation as R
from PIL import Image
import matplotlib.pyplot as plt
import torch
# from physics_atv_visual_mapping.pointcloud_colorization.torch_color_pcl_utils import *
# from traj_lib import generate_trajectory_library
from torchvision import transforms
from torchvision.transforms.functional import hflip
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
import rasterio
import torch

DATA_DIR_BASE = '/home/tartandriver/workspace/datasets/lrdp'


class FrontierDataset(Dataset):
    def __init__(self, dir, num_bins=48, transform=None, apply_footprint = True, augment = False, start_stop = None):
        """
        Args:
            grid_dir (str): Path to the directory containing occupancy grids (100x100 .npy files).
            traj_dir (str): Path to the directory containing trajectories (N x 1000 x 2 .npy files).
            transform (callable, optional): Optional transform to apply to each grid.
        """
        self.dir = dir
        self.image_dir = os.path.join(dir, 'image')
        self.dino_dir = os.path.join(dir, 'dino_img')
        self.cost_dir = os.path.join(dir, 'cost_img')
        self.odom_dir = os.path.join(dir, 'odometry')
        self.gps_dir = os.path.join(dir, 'gps_odometry')
        self.map_dir = os.path.join(dir, 'bev_map_reduce')
        self.synth_dir = os.path.join(dir, 'synth_demo_v3')
        self.los_dir = os.path.join(dir, 'los_v3')
        self.track_dir = os.path.join(dir, 'masks')
        

        self.num_bins = num_bins

        gascola_tif_path = "/home/tartandriver/tartandriver_ws/src/core/mission_manager/gps_maps/gascola.tif"
        # gascola_cmap_path = 'global_costmap_newest.npy'
        gascola_cmap_path = 'global_costmap_gp_v5_slope.npy'
        
        trabuco_tif_path = "/home/tartandriver/tartandriver_ws/trabuco/trabuco_c.tif"
        trabuco_cmap_path = 'global_costmap_trabuco.npy'

        rr_tif_path = "/home/tartandriver/tartandriver_ws/src/core/mission_manager/gps_maps/renegade_c2021.tif"
        rr_cmap_path = 'global_costmap_renegade_v6_slope.npy'

        if 'rr' in dir:
            tif_path = rr_tif_path
            cmap_path = rr_cmap_path
            self.synth_dir = os.path.join(dir, 'synth_demo_v4')
            self.los_dir = os.path.join(dir, 'los_v4')
        elif 'trabuco' in dir:
            tif_path = trabuco_tif_path
            cmap_path = trabuco_cmap_path
        else:
            tif_path = gascola_tif_path
            cmap_path = gascola_cmap_path

        os.makedirs(self.los_dir, exist_ok=True)
        os.makedirs(self.synth_dir, exist_ok=True)
        os.makedirs(self.synth_dir+'_plot', exist_ok=True)

        self.tif_path = tif_path
        self.cmap_path = cmap_path
        costmap = np.load(cmap_path)

        self.tif = rasterio.open(tif_path)
        self.TT = torch.from_numpy(np.array(self.tif.transform).reshape(3,3))
        self.TT_inv = torch.inverse(self.TT)
        self.tif_res, _ = self.tif.res
        
        gps_odom = np.loadtxt(os.path.join(self.gps_dir, 'data.txt'))
        self.gps = gps_odom

        demo_x, demo_y = -self.gps[:,1], self.gps[:,0]

        fix = demo_x < 170000
        idx = np.arange(len(self.gps))

        good_idx = idx[~fix]
        bad_idx = idx[fix]

        # For each bad index, find nearest good index in time #TODO this is a temporary fix!
        nearest_good = good_idx[
            np.abs(good_idx[:, None] - bad_idx).argmin(axis=0)
        ]

        start_stop_map = {
            DATA_DIR_BASE + '/04_garage_to_turnpike_afternoon': [170,900],
            DATA_DIR_BASE + '/2023-11-14-15-02-21_figure_8': [135,4955],
            # DATA_DIR_BASE + '/2023-11-14-15-02-21_figure_8': [1160,4955],
            # DATA_DIR_BASE + '/2023-11-14-15-02-21_figure_8': [475,650],
            DATA_DIR_BASE + '/turnpike_2023-09-12-12-53-32': [130,6520], #moved 100 from both sides
            # DATA_DIR_BASE + '/turnpike_2023-09-12-12-53-32': [2108,2200], #moved 100 from both sides

            DATA_DIR_BASE + '/20251009_3_red_course': [353,2200],
            DATA_DIR_BASE + '/20251009_4_fig8_to_horseshoe': [464,3840],
            # DATA_DIR_BASE + '/20251009_4_fig8_to_horseshoe': [664,3840],

            DATA_DIR_BASE + '/20251009_5_turnpike': [920,3395],
            DATA_DIR_BASE + '/2025-10-16-18-00-49_snow_bag': [160,2820],
            # DATA_DIR_BASE + '/2025-10-16-18-00-49_snow_bag': [860,2820],
            DATA_DIR_BASE + '/rr_demo_lap_6': [280,2750],
            DATA_DIR_BASE + '/rr_from_gulley8': [10,1538],
            DATA_DIR_BASE + '/rr_from_gulley9': [20,1199],
            DATA_DIR_BASE + '/rr_power_tower_sidehill6': [80,1151],
            # DATA_DIR_BASE + '/rr_power_tower_sidehill6': [180,1151],
            DATA_DIR_BASE + '/rr_sparse_shrubs_7': [220,2830],
        }
        idx_start, idx_stop = start_stop_map[dir]

        # Replace bad GPS values
        self.gps[bad_idx] = self.gps[nearest_good]
        demo_x, demo_y = -self.gps[:,1], self.gps[:,0]
        # costmap -= .18
        # costmap = np.clip(costmap, 0, 1.)
        # costmap /= costmap.max()

        self.apply_footprint = apply_footprint
        #TODO get rid of this check when we have costmap
        if self.apply_footprint:
            # print(demo_x)
            all_gps = np.empty([0,2])
            for dataset_dir in start_stop_map.keys():
                dataset_odom = np.loadtxt(os.path.join(dataset_dir, 'gps_odometry', 'data.txt'))
                gps_x, gps_y = -dataset_odom[:,1], dataset_odom[:,0]
                gps = np.hstack([gps_x.reshape(-1,1), gps_y.reshape(-1,1)])
                all_gps = np.vstack([all_gps, gps])

            idx, idy = self.tif.index(all_gps[:,0], all_gps[:,1])
            idx = np.array(idx)
            idy = np.array(idy)
            keep = (idx > 0) & (idx <= costmap.shape[0]) & (idy > 0) & (idy <= costmap.shape[1])
            idx = idx[keep]
            idy = idy[keep]


            from scipy.ndimage import binary_dilation

            radius = 3
            # Initialize mask for all points
            mask = np.zeros_like(costmap, dtype=bool)
            mask[idx, idy] = True

            # Create circular structuring element
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            footprint = x**2 + y**2 <= radius**2

            # Dilate mask
            inflated_mask = binary_dilation(mask, structure=footprint)

            # Apply inflation
            # costmap[inflated_mask] = 0.1
            costmap[inflated_mask] = np.minimum(.15,costmap[inflated_mask])
            # costmap[inflated_mask] *= .1

            # plt.imshow(costmap, cmap='magma')
            # plt.show()

        self.costmap = costmap

        self.horizon = 250

        self.augment = augment
        self.goal_cond = True

        odom = np.loadtxt(os.path.join(self.odom_dir, 'data.txt'))
        all_min = odom.min(axis=0)
        all_max = odom.max(axis=0)
        min_x, min_y = all_min[:2]
        max_x, max_y = all_max[:2]
        self.odom = odom

        self.indices = sorted([
            int(fname.split('.')[0])
            for fname in os.listdir(self.image_dir)
            if fname.endswith('.png') and fname.split('.')[0].isdigit()
        ])

        self.indices = self.indices[idx_start:idx_stop]

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(  # Normalize with ImageNet mean/std
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
            ])

        if not os.path.isdir(self.dino_dir):
            print("GENERATING FEAT IMAGES FOR " + dir)
            os.makedirs(self.dino_dir)
            self.gen_dino()
        
        img_mask = cv2.imread('/home/tartandriver/tartandriver_ws/src/perception/physics_atv_visual_mapping/data/masks/yamaha/image_left_color_mask.png')
        img_mask = img_mask[:,:,0]
        img_mask *= 0 #trabuco different robot

        idx_str = f"{self.indices[0]:08d}"
        dino_path = os.path.join(self.dino_dir, f"{idx_str}_data.npy")
        if not os.path.exists(dino_path):
            dino_path = dino_path.replace("_data", "")
        feat_img = np.load(dino_path)
        img_mask = cv2.resize(img_mask, (feat_img.shape[-1], feat_img.shape[-2]),interpolation=cv2.INTER_NEAREST)
        self.img_mask = img_mask == 255

    def to_local_frame(self, points_xy, origin_x, origin_y, origin_yaw):
        """
        Convert global (x, y) points to a local frame.

        Parameters:
            points_xy : np.ndarray of shape (N, 2)
                Global coordinates of the points.
            origin_x : float
                X coordinate of the local frame origin.
            origin_y : float
                Y coordinate of the local frame origin.
            origin_yaw : float (radians)
                Yaw of the local frame relative to the global frame.

        Returns:
            np.ndarray of shape (N, 2) with local frame coordinates.
        """
        # Translate points so origin is at (0, 0)
        translated = points_xy - np.array([origin_x, origin_y])

        # Rotation matrix for -yaw (to align global to local)
        c, s = np.cos(-origin_yaw), np.sin(-origin_yaw)
        R = np.array([[c, -s],
                      [s,  c]])

        # Rotate translated points
        local_points = translated @ R.T
        return local_points

    def to_global_frame(self, points_local, origin_x, origin_y, origin_yaw):
        """
        Convert local (x, y) points to global frame.

        Parameters:
            points_local : np.ndarray of shape (N, 2)
                Local coordinates of the points.
            origin_x : float
                X coordinate of the local frame origin in global frame.
            origin_y : float
                Y coordinate of the local frame origin in global frame.
            origin_yaw : float (radians)
                Yaw of the local frame relative to the global frame.

        Returns:
            np.ndarray of shape (N, 2) with global coordinates.
        """
        # Rotation matrix for +yaw
        c, s = np.cos(origin_yaw), np.sin(origin_yaw)
        R = np.array([[c, -s],
                      [s,  c]])

        # Rotate then translate
        rotated = points_local @ R.T
        global_points = rotated + np.array([origin_x, origin_y])
        return global_points

    def to_global_frame_torch(self, points_local, origin):
        """
        Convert local (x,y) points to global frame (batched, differentiable).

        Args:
            points_local : (B, N, 2) tensor
            origin_x : (B,) tensor
            origin_y : (B,) tensor
            origin_yaw : (B,) tensor (radians)

        Returns:
            global_points : (B, N, 2) tensor
        """
        B, N, _ = points_local.shape

        c = torch.cos(origin[:,2])  # (B,)
        s = torch.sin(origin[:,2])  # (B,)

        # Build rotation matrices (B, 2, 2)
        R = torch.stack([torch.stack([c, -s], dim=-1),
                         torch.stack([s,  c], dim=-1)], dim=-2)

        # Rotate: (B, N, 2) @ (B, 2, 2)^T → (B, N, 2)
        rotated = torch.bmm(points_local, R.transpose(1, 2))

        # Translate: broadcast (B, N, 2) + (B, 1, 2)
        global_points = rotated + origin[:,:2].unsqueeze(1)

        return global_points

    def __len__(self):
        return len(self.indices)

    def transform_img(self, img):
        return self.transform(img)

    def load_feat_image(self, idx_str):
        dino_path = os.path.join(self.dino_dir, f"{idx_str}_data.npy")
        if not os.path.exists(dino_path):
            dino_path = dino_path.replace("_data", "")
        feat_img = np.load(dino_path)

        feat_img = torch.from_numpy(feat_img)
        if len(feat_img.shape) == 2:
            feat_img = feat_img.unsqueeze(0)

        # feat_img[:,self.img_mask] = 0

        return feat_img

    def gen_dino(self):
        from physics_atv_visual_mapping.image_processing.image_pipeline import setup_image_pipeline
        
        # config = {"models_dir": '/home/tartandriver/tartandriver_ws/models',
        #     "image_processing": [{
        #         "type": 'radio_lang',
        #         "args": {'image_insize': [848, 448],
        #             'radio_type': 'c-radio_v3-b',
        #             'adaptor_type': 'siglip2'}}],
        #     "device": 'cuda'
        # }

        config = {"models_dir": '/home/tartandriver/tartandriver_ws/models',
            "image_processing": [{
                "type": 'radio',
                "args": {'image_insize': [848, 448],
                    'radio_type': 'c-radio_v3-b',
                    }}],
            "device": 'cuda'
        }

        image_pipeline = setup_image_pipeline(config)

        for idx in tqdm(self.indices):
            idx_str = f"{idx:08d}"

            img_path = os.path.join(self.image_dir, f"{idx_str}.png")
            og_img = Image.open(img_path).convert("RGB")
            og_img = np.array(og_img).astype(np.float32)/255.
            # raw_img = og_img.copy()
            og_img = np.transpose(og_img, [2,0,1])
            og_img = torch.from_numpy(og_img).unsqueeze(0).cuda()

            # print(og_img.min(), og_img.max())
            image_intrinsics = torch.eye(4).unsqueeze(0)
            with torch.no_grad():
                # feat_img = dino(og_img).cpu().numpy()[0]
                feat_img, feature_intrinsics = image_pipeline.run(
                    og_img, image_intrinsics
                )
                feat_img = feat_img[0].cpu().numpy()

            feat_path = os.path.join(self.dino_dir, f"{idx_str}.npy")

            np.save(feat_path, feat_img)

    def gen_examples(self):
        RADIUS = 200  # desired radius
        NUM_GOALS = 120  # number of goals around the circle

        costmap = self.costmap.copy()
        lethal_thresh = .55
        # path_thresh = 3000
        # path_thresh = 1000
        path_thresh = 800
        if 'rr' in self.dir:
            # lethal_thresh = .55
            lethal_thresh = .5
            path_thresh = 800
            RADIUS = 300
            self.horizon = 750
        lethal_add = 100.
        costmap[costmap >= lethal_thresh] += lethal_add
        costmap += 1.

        planning_map = self.costmap.copy()
        # planning_map[planning_map < lethal_thresh] *= .1
        
        demo_x, demo_y = -self.gps[:,1], self.gps[:,0]

        rgb_map = self.tif.read([1,2,3])
        rgb_map = np.transpose(rgb_map, [1,2,0])

        scount = 0
        for idx in tqdm(self.indices):
            scount += 1

            cur_odom = self.gps[idx]
            yaw = R.from_quat(cur_odom[3:7]).as_euler('xyz')[2] + np.pi/2
            # cur_x, cur_y = cur_odom[:2]
            cur_x, cur_y = -cur_odom[1], cur_odom[0]
            cur_row, cur_col = self.tif.index(cur_x, cur_y)
            start = (cur_row, cur_col)

            angles = np.linspace(0, 2*np.pi, NUM_GOALS, endpoint=False)
            goals = np.array([start + RADIUS * np.array([np.sin(a), np.cos(a)]) for a in angles]).astype(int)

            costs = costmap[goals[:,0], goals[:,1]]
            keep = costs < lethal_add
            # goals = goals[keep]

            all_costs = []
            all_goals = []

            tplanning_map = planning_map.copy()
            # tplanning_map += 1
            tyaw = yaw + np.pi/2
            tyaw = -(tyaw - np.pi/2)
            # tyaw = yaw
            # add_wall_behind(tcostmap, start, tyaw)
            add_limit_turn(tplanning_map, start, tyaw, cost = 10)
            add_wall_behind(tplanning_map, start, tyaw, cost = 10)
            # plt.imshow(tplanning_map, vmin=0, vmax=1)
            # plt.show()

            V, all_paths = get_plan_together(tplanning_map, start, goals, RADIUS)
            all_costs = []
            demo_idx = []
            for i,path in enumerate(all_paths):
                path = path[::-1]
                # path = path[:200]
                sub_idxs = np.linspace(0, path.shape[0]-1, 200).astype(int)
                path = path[sub_idxs]
                tpath = path.astype(int)
                cost = costmap[tpath[:,0], tpath[:,1]].sum()
                all_costs.append(cost)
                all_paths[i] = path
                demo_idx.append(tpath)

            if len(all_paths) == 0:
                print("NO PATHS")
                V, all_paths = get_plan_together(planning_map.copy(), start, goals, RADIUS)
                all_costs = []
                demo_idx = []
                for i,path in enumerate(all_paths):
                    path = path[::-1]
                    # path = path[:200]
                    sub_idxs = np.linspace(0, path.shape[0]-1, 200).astype(int)
                    path = path[sub_idxs]
                    tpath = path.astype(int)
                    cost = costmap[tpath[:,0], tpath[:,1]].sum()
                    all_costs.append(cost)
                    all_paths[i] = path
                    demo_idx.append(tpath)

            all_paths = np.array(all_paths)
            all_costs = np.array(all_costs)
            demo_idx = np.array(demo_idx)
            # print(all_costs)

            global_lib = all_paths
            
            #append global to global_lib, unrot to trajectory_library
            demo_odom_pre = self.gps[idx+1:idx+self.horizon + 1,:2]

            og_demo_local_unrot = self.to_local_frame(demo_odom_pre, cur_odom[0], cur_odom[1], yaw)
            og_demo_local = og_demo_local_unrot.copy()
            og_demo_local[:,0] = -1*og_demo_local_unrot[:,1]
            og_demo_local[:,1] = og_demo_local_unrot[:,0]
            og_demo_local /= self.tif_res
            demo_odom = demo_odom_pre.copy()
            demo_odom[:,0] = -1*demo_odom_pre[:,1]
            demo_odom[:,1] = demo_odom_pre[:,0]

            traj_lib = self.to_local_frame(global_lib, start[0], start[1], yaw)
            
            templib = traj_lib.copy()
            traj_lib[:,:,0] = templib[:,:,1]
            traj_lib[:,:,1] = -templib[:,:,0]
            trajectory_library = traj_lib

            # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
            # for j in range(len(trajectory_library)):
            #     # ax[1].scatter(goal[1], goal[0], c='green')
            #     cost = all_costs[j]
            #     if cost > path_thresh:
            #         c = '-r'
            #     else:
            #         c = '-g'
            #     ax[0].plot(trajectory_library[j,:,0], trajectory_library[j,:,1], c)
            # ax[0].plot(og_demo_local[:,0], og_demo_local[:,1], '--r')

            # # ax[1].imshow(self.costmap)
            # ax[1].imshow(tplanning_map, vmin=0, vmax=1)
            # ax[1].scatter(start[1], start[0], c='red', label='Start')
            # for j, path in enumerate(all_paths):
            #     ax[1].scatter(goals[j,1], goals[j,0], c='green')
            #     cost = all_costs[j]
            #     if cost > path_thresh:
            #         c = '-r'
            #     else:
            #         c = '-g'
            #     ax[1].plot(path[:,1], path[:,0], c)
            # plt.show()

            sub_idxs = np.linspace(0, og_demo_local.shape[0]-1, global_lib.shape[1]).astype(int)
            og_demo_local = og_demo_local[sub_idxs]
            demo_odom = demo_odom[sub_idxs]

            row, col = self.tif.index(demo_odom[:,0], demo_odom[:,1])
            og_dem_ir = np.array(row)
            og_dem_ic = np.array(col)
            og_demo_idx = np.stack([og_dem_ir, og_dem_ic], axis=-1)
            og_cost = costmap[og_dem_ir,og_dem_ic]

            pathcosts = costmap[demo_idx[:,:,0], demo_idx[:,:,1]]
            costs_sum = pathcosts.sum(axis=-1)
            costs = pathcosts.mean(axis=-1) - 1

            valid = costs_sum < path_thresh

            if valid.sum() == 0:
                valid = costs_sum < path_thresh*1.2

            if valid.sum() == 0:
                pathcosts += 1000
            else:
                global_lib = global_lib[valid]
                demo_idx = demo_idx[valid]
                trajectory_library = trajectory_library[valid]
                costs = costs[valid]
                pathcosts = pathcosts[valid]
                costs_sum = costs_sum[valid]

            add_demo = False
            if np.linalg.norm(og_demo_local[-1]) > 50.:
                # import pdb;pdb.set_trace()
                trajectory_library = np.concatenate([trajectory_library, og_demo_local[np.newaxis]])
                pathcosts = np.concatenate([pathcosts, og_cost.reshape(1,-1)])
                demo_idx = np.concatenate([demo_idx, og_demo_idx[np.newaxis]])

                assert len(trajectory_library) == len(pathcosts)
                add_demo = True

            try:
                fvi = compute_furthest_visible_vectorized(
                    demo_idx, costmap, thresh=lethal_add
                )
                fvi = apply_fov_constraint_vectorized(fvi, trajectory_library, 0, fov_deg=93)
            except:
                # pathcosts += 1000
                # fvi = [80]*trajectory_library.shape[0]
                print(idx)
                # continue
                
            if scount % 10 == 0:
                fig, ax = plt.subplots(1,2)
                # plt.imshow(trainset.costmap, cmap='gray_r', origin='lower', extent=[0, grid_size[1]*resolution, 0, grid_size[0]*resolution])
                # ax[1].imshow(costmap, cmap='bone', interpolation='None')
                ax[1].imshow(rgb_map)
                ax[1].imshow(np.clip(costmap,0,lethal_add), alpha=.3)
                # Plot valid trajectories in green
                # print(fvi)
                if len(demo_idx) > 0:
                    for i,traj in enumerate(demo_idx):
                        # color = 'g' if valid[i] else 'r'
                        if i == len(demo_idx) - 1 and add_demo:
                            ax[1].plot(traj[:, 1], traj[:, 0], '--w', linewidth=.5,zorder=2)
                            ax[1].scatter(traj[fvi[i], 1], traj[fvi[i], 0], c='w', s=8, zorder=3)
                        else:
                            ax[1].plot(traj[:, 1], traj[:, 0], '-',color=CMAP(costs[i]), linewidth=.5, zorder=1)
                            ax[1].scatter(traj[fvi[i], 1], traj[fvi[i], 0], s=5,zorder=3)

                    idx_str = f"{idx:08d}"
                    
                    img_path = os.path.join(self.image_dir, f"{idx_str}.png")
                    og_img = Image.open(img_path).convert("RGB")
                    ax[0].imshow(og_img)
                    
                    viz_range = 300
                    ax[1].set_xlim(demo_idx[0,0,1] - viz_range, demo_idx[0,0,1] + viz_range)
                    ax[1].set_ylim(demo_idx[0,0,0] + viz_range, demo_idx[0,0,0] - viz_range)
                    
                    # plt.show()
                    # s=r
                    # demo_viz_path = os.path.join(self.synth_dir + '_plot', f"{idx_str}.npy")
                    plt.savefig(self.synth_dir + '_plot/' + str(idx) + ".jpg", dpi=300, bbox_inches='tight')
                    plt.clf()
                    plt.close()

            trajectory_library = np.concatenate([trajectory_library, -1.+pathcosts[..., np.newaxis]], axis=-1)

            idx_str = f"{idx:08d}"
            demo_path = os.path.join(self.synth_dir, f"{idx_str}.npy")
            np.save(demo_path, trajectory_library)

            fvi_path = os.path.join(self.los_dir, f"{idx_str}.npy")
            np.save(fvi_path, fvi)

    def gen_los(self):
        scount = 0
        for idx in tqdm(self.indices):
            scount += 1

            idx_str = f"{idx:08d}"
            demo_path = os.path.join(self.synth_dir, f"{idx_str}.npy")

            fvi_path = os.path.join(self.los_dir, f"{idx_str}.npy")
            # if os.path.exists(fvi_path): continue

            trajectory_library = np.load(demo_path)[:,:,:2]

            cur_odom = self.gps[idx]
            yaw = R.from_quat(cur_odom[3:7]).as_euler('xyz')[2] + np.pi/2
            # cur_x, cur_y = cur_odom[:2]
            cur_x, cur_y = -cur_odom[1], cur_odom[0]

            global_lib = self.to_global_frame(trajectory_library, cur_x, cur_y, yaw)

            og_shape = global_lib.shape
            flatlib = global_lib.reshape(-1,2)
            row, col = self.tif.index(flatlib[:,0], flatlib[:,1])
            dem_ir = np.array(row).reshape(og_shape[:2])
            dem_ic = np.array(col).reshape(og_shape[:2])
            
            demo_idx = np.stack([dem_ir, dem_ic], axis=-1)

            fvi = compute_furthest_visible_vectorized(
                demo_idx, self.costmap, thresh=0.47
            )
            fvi_fov = apply_fov_constraint_vectorized(fvi, trajectory_library, 0, fov_deg=93)
            
            # if scount % 20 == 0:
            #     fig, ax = plt.subplots(1,2)
            #     # plt.imshow(trainset.costmap, cmap='gray_r', origin='lower', extent=[0, grid_size[1]*resolution, 0, grid_size[0]*resolution])
            #     ax[1].imshow(self.costmap, cmap='bone', interpolation='None')
            #     # ax[1].imshow(rgb_map)
            #     # Plot valid trajectories in green
            #     # print(fvi)
            #     for i,traj in enumerate(demo_idx):
            #         # color = 'g' if valid[i] else 'r'
            #         ax[1].plot(traj[:, 1], traj[:, 0], '-',color='cyan', linewidth=.5)
            #         ax[1].scatter(traj[fvi_fov[i], 1], traj[fvi_fov[i], 0], s=5)

            #     idx_str = f"{idx:08d}"
                
            #     img_path = os.path.join(self.image_dir, f"{idx_str}.png")
            #     og_img = Image.open(img_path).convert("RGB")
            #     ax[0].imshow(og_img)
                
            #     viz_range = 200
            #     ax[1].set_xlim(demo_idx[0,0,1] - viz_range, demo_idx[0,0,1] + viz_range)
            #     ax[1].set_ylim(demo_idx[0,0,0] + viz_range, demo_idx[0,0,0] - viz_range)
                
            #     plt.show()
            #     # demo_viz_path = os.path.join(self.synth_dir + '_plot', f"{idx_str}.npy")
            #     # plt.savefig(self.synth_dir + '_plot/' + str(idx) + ".jpg", dpi=300, bbox_inches='tight')
            #     # plt.clf()
            #     # plt.close()

            np.save(fvi_path, fvi_fov)

    def __getitem__(self, idx):
        index = self.indices[idx]
        idx_str = f"{index:08d}"

        img_path = os.path.join(self.image_dir, f"{idx_str}.png")
        og_img = Image.open(img_path).convert("RGB")
        img = self.transform_img(og_img)
        og_img = np.array(og_img)

        feat_img = self.load_feat_image(idx_str)

        synth_path = os.path.join(self.synth_dir, f"{idx_str}.npy")
        synth_demo_all = np.load(synth_path)

        fvi_path = os.path.join(self.los_dir, f"{idx_str}.npy")
        fvi = np.load(fvi_path)

        cost_mask = synth_demo_all[:,:,-1] <= 1.
        row_sum = np.sum((synth_demo_all[:,:,-1] - .1) * cost_mask, axis=1)

        clean_rows = ~np.any(synth_demo_all[:,:50,-1] > 1., axis=1)
        clean_rows[-1] = True

        row_count = np.sum(cost_mask, axis=1)
        all_costs = row_sum / np.maximum(row_count, 1)

        synth_demo_xy = synth_demo_all[:,:,:2]
        # fvi = np.minimum(fvi, 80)

        cur_odom = self.gps[index]
        rpy = R.from_quat(cur_odom[3:7]).as_euler('xyz')
        yaw = rpy[2] + np.pi/2
        roll, pitch = rpy[:2]
        cur_x, cur_y = -cur_odom[1], cur_odom[0]
        
        global_lib = self.to_global_frame(synth_demo_xy, cur_x, cur_y, yaw)

        og_shape = global_lib.shape
        flatlib = global_lib.reshape(-1,2)
        row, col = self.tif.index(flatlib[:,0], flatlib[:,1])
        dem_ir = np.array(row).reshape(og_shape[:2])
        dem_ic = np.array(col).reshape(og_shape[:2])
        
        demo_idx = np.stack([dem_ir, dem_ic], axis=-1)

        demo_odom_pre = self.gps[index+1:index+self.horizon + 1,:2]
        demo_odom = demo_odom_pre.copy()
        demo_odom[:,0] = -1*demo_odom_pre[:,1]
        demo_odom[:,1] = demo_odom_pre[:,0]

        sub_idxs = np.linspace(0, demo_odom_pre.shape[0]-1, global_lib.shape[1]).astype(int)
        demo_odom = demo_odom[sub_idxs]

        row, col = self.tif.index(demo_odom[:,0], demo_odom[:,1])
        og_dem_ir = np.array(row)
        og_dem_ic = np.array(col)
        og_demo_idx = np.stack([og_dem_ir, og_dem_ic], axis=-1)

        num_bins = self.num_bins
        bins = torch.linspace(-torch.pi/2, torch.pi/2, num_bins+1)

        binned_headings = 0.5 * (bins[:-1] + bins[1:])

        try:
            traj_mask = np.load(os.path.join(self.track_dir, f"{idx_str}.npy"))
        except:
            traj_mask = np.zeros_like(og_img)

        do_augment = np.random.choice([True, False])
        do_augment = do_augment and self.augment
        if do_augment:
            feat_img = hflip(feat_img)
            binned_headings *= -1
            og_img = np.fliplr(og_img).copy()
            synth_demo_xy[...,1] *= -1
            traj_mask = np.fliplr(traj_mask).copy()
            
        traj_vec = synth_demo_xy[np.arange(len(synth_demo_xy)),fvi,:2] - synth_demo_xy[:,0,:2]
        traj_headings = np.arctan2(traj_vec[:, 1], traj_vec[:, 0])

        dists = np.linalg.norm(traj_vec, axis=1)
        keep = dists > 5.1
        keep = keep & clean_rows

        traj_headings = traj_headings[keep]
        all_costs = all_costs[keep]

        traj_headings = torch.from_numpy(traj_headings)
        all_costs = torch.from_numpy(all_costs).float()

        bin_indices = torch.bucketize(traj_headings, bins) - 1
        bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

        #"costs" is actually score bc i'm dumb
        binned_costs = torch.zeros(num_bins).float()
        # binned_costs.scatter_reduce_(0, bin_indices, 1. - 3*all_costs, reduce='mean', include_self=False)
        binned_costs.scatter_reduce_(0, bin_indices, 1. - (3.*all_costs)**2, reduce='mean', include_self=False)
        # binned_costs.scatter_reduce_(0, bin_indices, torch.ones_like(all_costs), reduce='mean', include_self=False)
        # binned_costs.scatter_reduce_(0, bin_indices, 1. - (3.*all_costs)**2, reduce='amax', include_self=False)

        binned_costs = torch.clip(binned_costs,0,1)

        binned_counts = torch.zeros(num_bins).float()
        binned_counts.scatter_reduce_(0, bin_indices, torch.ones_like(all_costs), reduce='sum', include_self=False)
        
        # binned_costs[binned_counts < 3] = 0

        if torch.isnan(binned_costs).any():
            print("SOMETHING IS WRONG")
            s=r

        return {
            'img': feat_img,
            'rgb_img': img,
            'img_raw': og_img,
            'origin': torch.Tensor([cur_x, cur_y, yaw]).float(),
            # 'origin_idx': torch.Tensor(start).float(),
            'og_demo_idx': torch.Tensor(og_demo_idx).float(),
            # 'costs_all': torch.Tensor(costs).float(),
            # 'headings': torch.Tensor(headings).float(),
            'costs': torch.Tensor(binned_costs).float(),
            'headings_binned': torch.Tensor(binned_headings).float(),
            'flipped': do_augment,
            'traj_mask': torch.Tensor(traj_mask).float()
        }

