import os
import copy
import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import torch.nn.functional as F
import time
from physics_atv_visual_mapping.frontier_estimation.toy_traj_los_dirichlet_astar import DinoHeadingCostHeadV2, overlay_heatmap_on_image
from ros_torch_converter.datatypes.marker_array import MarkerArrayTorch


def circular_difference(a, b):
    diff = (a - b + torch.pi) % (2 * torch.pi) - torch.pi
    return diff

def wrap_angle(angle):
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi

def circular_shift_interp(pred, centers, shift_angle):
    """
    pred: (B,) prediction over heading bins
    centers: (B,) bin centers in radians
    shift_angle: scalar
    """
    # Target angles in original frame
    target_angles = wrap_angle(centers - shift_angle)

    # Convert angles to fractional bin indices
    bin_width = centers[1] - centers[0]
    idx_float = (target_angles - centers[0]) / bin_width

    idx0 = torch.floor(idx_float).long() % len(pred)
    idx1 = (idx0 + 1) % len(pred)

    w1 = idx_float - torch.floor(idx_float)
    w0 = 1.0 - w1

    return w0 * pred[idx0] + w1 * pred[idx1]


# >>> R.from_quat([0.805, -0.212, 0.150, 0.534]).as_euler('xyz', degrees=True)
# array([115.82588826, -27.88015828, -11.81636727])
# >>> R.from_quat([0.597, -0.593, 0.384, 0.380]).as_euler('xyz', degrees=True)
# array([-179.76542289,  -65.40248954,  -89.76542289])
# >>> R.from_quat([-.217, .812, -.523,-.144]).as_euler('xyz', degrees=True)
# array([-117.6462801 ,  -27.42044516, -166.86978011])



class LRNAHInferenceNode():
    """
    TorchCoordinator wrapper for running LRNAH
    """
    def __init__(self, config, device='cuda'):
        """
        Args:
            keep_uncolorized: set this flag to false to keep points w/o colorization
        """
        
        # self.pose_key = remap['pose']
        # self.odom_key = remap['state']
        # self.wpts_in_key = remap['wpts_in']
        # self.wpts_out_key = remap['wpts_out']
        # self.feat_img_key = remap['feature_image']
        # self.debug_viz_key = remap['wpts_viz']
        # self.rgb_img_key = remap['image']
        # self.heatmap_key = remap['heatmap']

        frontier_conf = config['frontier_estimation']

        model_path = frontier_conf['model_path']
        # weights = torch.load('/home/tartandriver/tartandriver_ws/misc/model.pt')
        weights = torch.load(model_path)




        #not actually trained with this number of bins... i think it was 60
        num_bins = 72*2
        self.num_bins = num_bins

        model = DinoHeadingCostHeadV2(in_channels=768, num_bins=num_bins)
        model.eval()
        model.load_state_dict(weights['head'], strict=False)
        model.cuda()
        
        
        #TODO grab this programatically
        #also assumes same intrinsics for all 3 cams
        K = np.array([[600.,   0., 480.],
                        [  0., 600., 300.],
                        [  0.,   0.,   1.]]).reshape(3,3)
        
        model.register_headings(K, 600, 960)

        self.model = model

        self.bins = torch.linspace(-torch.pi, torch.pi, num_bins+1)
        self.binned_headings = 0.5 * (self.bins[:-1] + self.bins[1:])

        self.ema_costs = None
        self.alpha = .6
        self.h_thresh = 7.5
        self.sigma_g = .9
        self.sigma_p = .9

        self.mu_p = None

        self.prev_yaw = None

    def run(self, data):
        
        # current_height = data[self.pose_key].transform[2, -1]

        # print(data['feature_images'].shape)

        xyz = data["pos"]
        # print(xyz)
        #TODO scipy version
        # rot = R.from_matrix(data["rot"].cpu().numpy())
        rot = R.from_dcm(data["rot"].cpu().numpy())
        rot_inv = rot.inv()
        yaw = rot.as_euler('zyx')[0]

        cur_wpts = torch.Tensor(data["goal"]).cpu()
        world_pos = data["veh2world"].cpu()
        xyz_world = world_pos[:3,-1]
        rot_world = R.from_dcm(world_pos[:3,:3].cpu().numpy())
        rot_inv_world = rot_world.inv()
        yaw_world = rot_world.as_euler('zyx')[0]

        goal_vec_world = (cur_wpts[:2] - xyz_world[:2]).cpu().numpy()
        goal_vec_world = np.array([*goal_vec_world, 0.])
        goal_vec_local = rot_inv_world.apply(goal_vec_world)
        goal_heading_local = np.arctan2(goal_vec_local[1], goal_vec_local[0])

        feat_imgs = data['feature_images'].permute(0,3,1,2)
        
        with torch.no_grad():
            heatmaps, pred_costs = self.model(feat_imgs)
        pred_costs = pred_costs.cpu()

        overlays = []
        now = time.perf_counter()
        for i in range(len(heatmaps)):
            rgb = data['images'][i].cpu().permute(1,2,0).numpy()
            heatmap = heatmaps[i].cpu().numpy()
            # heatmap -= 2
            # print(heatmap.shape, rgb.shape)
            heatmap[heatmap< -6.] = 0
            overlay = overlay_heatmap_on_image(rgb, 
                                                heatmap, 
                                                max_val = 8., 
                                                threshold=0.01, 
                                                alpha=0.6)
            overlay = overlay/255.
            # overlay = overlay[:,:,::-1]
            overlays.append(overlay)
            # cv2.imshow('test', overlay)
            # cv2.waitKey(1)

        # print(time.perf_counter() - now, "OVERLAY TIME")

        pred_costs_merged = torch.zeros_like(self.binned_headings)
        shifts = np.deg2rad([89.77-11.82, 0, 89.77-166.87])
        for i,shift in enumerate(shifts):
            aligned = circular_shift_interp(pred_costs[i], self.binned_headings, shift)
            pred_costs_merged = torch.maximum(pred_costs_merged, aligned)

        # aligned = circular_shift_interp(pred_costs[0], self.binned_headings, np.deg2rad(115))
        # aligned[aligned != 0] +=  10*(0+1)
        # pred_costs_merged = torch.maximum(pred_costs_merged, aligned)

        pred_costs = pred_costs_merged
        # pred_cost_viz = pred_costs.clone()

        #schmittle. actually they do something else in the code
        pred_costs[pred_costs < self.h_thresh] = 0.
        pred_costs = pred_costs/(pred_costs.sum() + 1e-6)
        pred_costs = pred_costs.cpu()

        if self.ema_costs is None:
            self.ema_costs = pred_costs
            self.prev_yaw = yaw
        else:
            # pred_costs = self.alpha*pred_costs + (1. - self.alpha)*self.ema_costs
            # self.ema_costs = pred_costs
            valid_low = self.binned_headings[0]
            valid_high = self.binned_headings[-1]

            shift_angle = circular_difference(yaw, self.prev_yaw)
            shifted_centers = self.binned_headings + shift_angle
            # shifted_centers = (shifted_centers + torch.pi) % (2*torch.pi) - torch.pi
            bin_width = self.binned_headings[1] - self.binned_headings[0]
            bin_shift = shift_angle / bin_width
            bin_shift = int(torch.round(bin_shift).item())
            indices = (torch.arange(self.num_bins) - bin_shift)# % self.num_bins

            #clamping should be fine since ignoring
            indices_clamped = indices.clamp(0, self.num_bins - 1)

            valid_mask = (shifted_centers >= valid_low) & (shifted_centers <= valid_high)

            shifted_ema= self.ema_costs[indices_clamped]
            pred_costs[valid_mask] = self.alpha * pred_costs[valid_mask] + (1.-self.alpha)*shifted_ema[valid_mask]
            self.ema_costs = pred_costs
            self.prev_yaw = yaw

        mu_g = goal_heading_local
        diff_g = circular_difference(self.binned_headings, mu_g)
        g = torch.exp(-0.5 * (diff_g / self.sigma_g) ** 2)

        if self.mu_p is None:
            p = torch.ones_like(g)
        else:
            diff_p = circular_difference(self.binned_headings, self.mu_p)
            p = torch.exp(-0.5 * (diff_p / self.sigma_p) ** 2)

        g = g/g.sum()
        p = p/p.sum()

        v = pred_costs * g * p
        best_id = torch.argmax(v)
        best_heading = self.binned_headings[best_id]
        self.mu_p = best_heading

        radius = 50
        pred_cost_viz = pred_costs.clone()
        # pred_cost_viz = v.clone()
        pred_cost_viz /= pred_cost_viz.max()
        markers = self.pred_cost_to_markers(
            pred_cost_viz, xyz.cpu(), rot, place_dist=radius
        )
        # markers.frame_id = wpts_msg.frame_id
        # markers.stamp = wpts_msg.stamp

        pred_heading = best_heading
        local_heading = np.array([np.cos(pred_heading), np.sin(pred_heading), 0.])
        # Rotate local heading into world frame
        world_heading = rot_world.apply(local_heading)

        # Compute target point
        pred_wp = xyz_world.cpu() + torch.from_numpy(radius * world_heading).cpu()

        res = {
            'wpts_viz': markers,
            'overlays': overlays,
            'pred_wp': pred_wp
        }

        return res

        cur_pose = data[self.pose_key].transform
        cur_odom = data[self.odom_key].state
        cur_vel = torch.linalg.norm(cur_odom[7:9])

        wpts_msg = data[self.wpts_in_key]
        cur_wpts = wpts_msg.goals

        xyz = cur_odom[:3]
        rot = R.from_quat(cur_odom[3:7].cpu().numpy())
        rot_inv = rot.inv()
        yaw = rot.as_euler('zyx')[0]

        if self.rgb_img_key in data:
            #TODO I THINK WE"VE BEEN FEEDING BGR IMAGES NOT RGB TO IMAGE PIPELINE
            rgb = data[self.rgb_img_key].image.cpu().numpy()[:,:,::-1]
            heatmap = heatmap[0].cpu().numpy()
            heatmap -= 2
            overlay = overlay_heatmap_on_image(rgb, 
                                               heatmap, 
                                               max_val = 5., 
                                               threshold=0.01, 
                                               alpha=0.6, cmap='jet')
            overlay = overlay/255.
            data[self.heatmap_key] = ImageTorch.from_numpy(overlay, 'cpu')

        # import pdb;pdb.set_trace()
        
        goal_vec_world = (cur_wpts[0,:2] - xyz[:2]).cpu().numpy()
        goal_vec_world = np.array([*goal_vec_world, 0.])
        goal_vec_local = rot_inv.apply(goal_vec_world)
        goal_heading_local = np.arctan2(goal_vec_local[1], goal_vec_local[0])

        #TODO this is kinda wrong
        mu_g = np.clip(goal_heading_local, -np.pi/2, np.pi)

        #this will be a lil messed up bc binned_headings is cropped
        diff_g = circular_difference(self.binned_headings, mu_g)
        g = torch.exp(-0.5 * (diff_g / self.sigma_g) ** 2)

        if self.mu_p is None:
            p = torch.ones_like(g)
        else:
            diff_p = circular_difference(self.binned_headings, self.mu_p)
            p = torch.exp(-0.5 * (diff_p / self.sigma_p) ** 2)

        g = g/g.sum()
        p = p/p.sum()

        v = pred_costs * g * p
        best_id = torch.argmax(v)
        best_heading = self.binned_headings[best_id]
        self.mu_p = best_heading

        # pred_cost_viz = pred_costs.clone()/pred_costs.max()
        pred_cost_viz = pred_costs * p
        pred_cost_viz /= pred_cost_viz.max()

        radius = 50
        pred_heading = best_heading
        local_heading = np.array([np.cos(pred_heading), np.sin(pred_heading), 0.])
        

        # Rotate local heading into world frame
        world_heading = rot.apply(local_heading)

        # Compute target point
        pred_wp = xyz + torch.from_numpy(radius * world_heading).to(xyz.device)

        pred_wp[-1] += 1

        pred_wpts = torch.vstack((pred_wp.reshape(1,-1), cur_wpts))

        pred_wpts_msg = GoalArrayTorch.from_torch(pred_wpts)
        pred_wpts_msg.frame_id = wpts_msg.frame_id
        pred_wpts_msg.stamp = wpts_msg.stamp
        data[self.wpts_out_key] = pred_wpts_msg

        
        markers = self.pred_cost_to_markers(
            pred_cost_viz.cpu(), xyz.cpu(), rot, place_dist=radius
        )
        markers.frame_id = wpts_msg.frame_id
        markers.stamp = wpts_msg.stamp
        data[self.debug_viz_key] = markers

    def pred_cost_to_markers(self, pred_cost, xyz, rot, device='cpu',
                         place_dist=50.0):
        """
        Convert pred_cost into a set of **point markers** placed
        on the circle at distance place_dist.
        """

        # 1) Non-zero bins
        mask = pred_cost > -10
        idxs = torch.where(mask)[0]
        vals = pred_cost[idxs]

        if len(idxs) == 0:
            return MarkerArrayTorch.from_torch(torch.zeros(0, 3, device=device))

        # 2) Map costs to RGBA
        cmap = cm.magma
        rgba_np = cmap(vals.cpu().numpy())
        rgba = torch.from_numpy(rgba_np).float().to(device)
        rgba[:, 3] = 1.0

        N = len(idxs)

        # 3) Headings → unit vectors in local frame
        h1 = self.binned_headings[idxs]
        local1 = torch.stack([
            torch.cos(h1), torch.sin(h1), torch.zeros_like(h1)
        ], dim=1)

        # 4) Rotate into world frame
        roll, pitch, yaw = rot.as_euler('xyz', degrees=False)
        r_yaw_only = R.from_euler('z', yaw, degrees=False)
        w1 = torch.from_numpy(r_yaw_only.apply(local1.cpu().numpy())).float().to(device)

        # 5) Compute point positions
        xyz_3 = xyz.reshape(1, 3)
        p1 = xyz_3 + place_dist * w1
        p1[:, 2] += 5.0  # raise above ground

        # ------------ KEY CHANGES BELOW -------------

        # Each p1[i] is now a marker position
        points = p1                                     # (N,3)

        # Marker type is SPHERE 2 (or CUBE)
        types = torch.full((N,), 2, device=device)

        # Size of spheres
        scales = torch.full((N, 3), 1.5, device=device)

        # Namespace
        ns = ["pred_cost"] * N

        # No line markers
        line_points = [None] * N

        # ------------ CONSTRUCT MARKER ARRAY ---------

        mat = MarkerArrayTorch.from_torch(
            points=points,
            colors=rgba,
            scales=scales,
            types=types,
            ns=ns,
            line_points=line_points
        )

        return mat

       
    def to(self, device):
        self.device = device
        return self
