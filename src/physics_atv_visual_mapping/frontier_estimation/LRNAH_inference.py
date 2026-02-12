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
from physics_atv_visual_mapping.frontier_estimation.network.frontier_estimator import VFMFrontierEstimator
from physics_atv_visual_mapping.frontier_estimation.viz_utils import overlay_heatmap_on_image
import yaml

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

        frontier_conf_path = config['frontier_estimation']['model_path']

        with open(os.path.join(frontier_conf_path, 'config.yaml'), 'r') as file:
            # Use safe_load for security when loading from untrusted sources
            config = yaml.safe_load(file)
        model_dir = os.path.join(frontier_conf_path, 'model.pt')

        model_conf = config['model']

        weights = torch.load(model_dir)
        

        model = VFMFrontierEstimator(model_conf)
        model.eval()
        model.load_state_dict(weights['head'], strict=False)
        model.cuda()
        
        #TODO grab this programatically
        #also assumes same intrinsics for all 3 cams
        K = np.array([[600.,   0., 480.],
                        [  0., 600., 300.],
                        [  0.,   0.,   1.]]).reshape(3,3)
        
        num_bins = model.binner.num_bins*2
        self.num_bins = num_bins
        model.binner.num_bins = num_bins
        model.policy.num_bins = num_bins
        model.register_headings(K=K,
                                H_orig=600,
                                W_orig=960,
                                device='cuda')

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
        #TODO scipy version?
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
        # print(pred_costs)

        overlays = []
        now = time.perf_counter()
        for i in range(len(heatmaps)):
            rgb = data['images'][i].cpu().permute(1,2,0).numpy()
            heatmap = heatmaps[i].cpu().numpy()
            # heatmap -= 2
            # print(heatmap.shape, rgb.shape)
            heatmap[heatmap< -6.] = 0
            # overlay = rgb
            overlay = overlay_heatmap_on_image(rgb, 
                                                heatmap, 
                                                max_val = 5., 
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

        pred_cost_viz = pred_costs.clone()
        # pred_cost_viz = v.clone()
        # pred_cost_viz /= pred_cost_viz.max()
        pred_cost_viz /= 12.

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


    def pred_cost_to_markers(self, pred_cost, xyz, rot, device='cpu',
                         place_dist=50.0):
        """
        Convert pred_cost into a set of **point markers** placed
        on the circle at distance place_dist.
        """

        # 1) Non-zero bins
        mask = pred_cost >= 0
        idxs = torch.where(mask)[0]
        vals = pred_cost[idxs]

        if len(idxs) == 0:
            return MarkerArrayTorch.from_torch(torch.zeros(0, 3, device=device))

        # 2) Map costs to RGBA
        cmap = cm.magma
        rgba_np = cmap(vals.cpu().numpy())
        rgba = torch.from_numpy(rgba_np).float().to(device)
        rgba[:, 3] = 1.0
        rgba[pred_cost==0,3] = 1.0

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

        # ------------------------------------------------
        # # Build a colored circle using LINE_LIST segments
        # # ------------------------------------------------

        # Sort by heading so the circle is ordered
        order = torch.argsort(h1)
        p1 = p1[order]
        rgba = rgba[order]

        # Optionally close the circle
        p1_next = torch.roll(p1, shifts=-1, dims=0)
        rgba_seg = rgba  # color per segment

        N = p1.shape[0]

        # Each marker is ONE line segment (2 points)
        line_points = [
            torch.stack([p1[i], p1_next[i]], dim=0)
            for i in range(N)
        ]

        # Dummy single points (required by your API, unused)
        points = torch.zeros(N, 3, device=device)

        # LINE_LIST markers
        types = torch.full((N,), 5, device=device)

        # Line width (scale.x is used, y/z ignored)
        scales = torch.zeros(N, 3, device=device)
        scales[:, 0] = 1.4   # line thickness

        ns = ["pred_cost_circle"] * N

        quats = [None] * N


        # # ------------------------------------------------
        # # Arrows pointing outward, colored + scaled by score
        # # ------------------------------------------------

        # Normalize scores
        # vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-6)
        # vals_norm = np.clip(vals, 0,1)
        vals_norm = vals
        idxs = torch.argsort(vals_norm)[-10:]
        vals_norm = vals_norm[idxs]
        vals_norm = (vals_norm - vals_norm.min()) / (vals_norm.max() - vals_norm.min() + 1e-6)
        p1 = p1[idxs]
        N = p1.shape[0]

        # Arrow lengths (meters)
        min_len = .2
        max_len = 20.0
        lengths = min_len + vals_norm * (max_len - min_len)

        # Arrow thickness
        shaft_diam = 1.0
        head_diam  = 2.2
        head_len   = 1.5

        # Marker positions (base of arrow)
        # points = p1  # arrows start on the circle

        # Marker type
        types_arrow = torch.full((N,), 0, device=device)

        # Scale per arrow
        scales_arrow = torch.zeros(N, 3, device=device)
        scales_arrow[:, 0] = lengths      # arrow length
        scales_arrow[:, 1] = shaft_diam   # shaft diameter
        scales_arrow[:, 2] = head_diam    # head diameter

        ns_arrow = ["pred_cost_arrows"] * N

        # ------------------------------------------------
        # Build arrow orientations (heading-aligned)
        # ------------------------------------------------

        # Heading unit vectors (already computed as w1)
        # Convert direction → yaw
        yaws = torch.atan2(w1[idxs, 1], w1[idxs, 0])

        # Convert yaw → quaternion (z-axis rotation)
        quats += torch.stack([
            torch.zeros_like(yaws),            # x
            torch.zeros_like(yaws),            # y
            torch.sin(yaws / 2),               # z
            torch.cos(yaws / 2)                # w
        ], dim=1)


        points = torch.vstack([points, p1])
        rgba = torch.vstack([rgba, rgba[idxs]])
        scales = torch.vstack([scales, scales_arrow])
        types = torch.hstack([types, types_arrow])
        ns += ns_arrow
        line_points += [None] * N

        mat = MarkerArrayTorch.from_torch(
            points=points,
            colors=rgba,
            scales=scales,
            types=types,
            ns=ns,
            line_points=line_points
        )

        mat.orientations = quats

        return mat



       
    def to(self, device):
        self.device = device
        return self
