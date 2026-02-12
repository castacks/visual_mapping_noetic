import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F

def apply_fov_constraint_vectorized(fvi, trajectory_library, camera_yaw, fov_deg=90):
    """
    Fully vectorized FOV application to furthest-visible indices.
    """
    N, T, _ = trajectory_library.shape

    vecs = trajectory_library                      
    angles = np.arctan2(vecs[...,1], vecs[...,0])  

    rel = angles - camera_yaw
    rel = (rel + np.pi) % (2*np.pi) - np.pi      

    half_fov = np.deg2rad(fov_deg) / 2
    inside_fov = np.abs(rel) <= half_fov          
    idx_grid = np.arange(T)[None, :]               
    # Mask: indices that are <= fvi for each trajectory
    allowed_by_fvi = idx_grid <= fvi[:, None]     

    valid = inside_fov & allowed_by_fvi            

    masked_vals = valid * idx_grid                

    new_fvi = masked_vals.max(axis=1)     

    return new_fvi

def compute_furthest_visible_vectorized(demo_idx, costmap, thresh=0.5, samples_per_cell=1):
   
    demo_idx = np.asarray(demo_idx)
    assert demo_idx.ndim == 3 and demo_idx.shape[2] == 2
    N, T, _ = demo_idx.shape

  
    start = demo_idx[:, :1, :].astype(np.float32)  
    end   = demo_idx.astype(np.float32)            

    dists = np.linalg.norm(end - start, axis=-1)   # (N,T)

    S = max(2, int(np.ceil(dists.max() * samples_per_cell)))
    t = np.linspace(0.0, 1.0, S, endpoint=True, dtype=np.float32)  # (S,)
    t_b = t.reshape((1, 1, S, 1))

   
    start_b = start[:, :, None, :]  
    end_b = end[:, :, None, :]  
    line_samples = start_b * (1.0 - t_b) + end_b * t_b 

    rr = np.rint(line_samples[..., 0]).astype(np.int32)  # rows
    cc = np.rint(line_samples[..., 1]).astype(np.int32)  # cols

    H, W = costmap.shape
    rr = np.clip(rr, 0, H - 1)
    cc = np.clip(cc, 0, W - 1)

    sampled_costs = costmap[rr, cc] 

    occluded = sampled_costs >= thresh 

    blocked = occluded.any(axis=-1) 

    any_blocked = blocked.any(axis=1)  # (N,)

    # furthest_visible = first_blocked - 1
    # # clip to valid index range [0, T-1]
    # furthest_visible = np.clip(furthest_visible, 0, T - 1).astype(np.int32)
    furthest_visible = np.max(np.where(~blocked, np.arange(T), -1), axis=1)
    furthest_visible = furthest_visible.astype(np.int32)

    return furthest_visible

def gradient_optimize_torch(
        base_lib,                # (N,T,2) torch.float32 (local frame)
        cur_odom,                # numpy 7-dim pose [x,y,z,qx,qy,qz,qw]
        sampler,                 # has sample_raster(...) and optionally sample_gradients(...)
        to_global_frame_torch,   # (B,T,2)->(B,T,2) function
        iterations=20,
        lr=0.005,
        curvature_weight=100.1,
        step_dev_weight=0.01,
        allow_step_scale=True,
        max_step_scale_dev=0.2,  # bound on step_scale deviation from 1
        device="cuda"
):
    """
    Arc-length optimizer INITIALIZED from base_lib.

    Returns: optimized trajectories in GLOBAL frame (N,T,2) numpy.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    base_lib = base_lib.to(device)
    N, T, _ = base_lib.shape
    assert T >= 2, "T must be at least 2"

    # -----------------------------
    # Extract ego pose (same as you used before)
    # -----------------------------
    yaw = R.from_quat(cur_odom[3:7]).as_euler('xyz')[2] + np.pi/2
    cur_x, cur_y = -cur_odom[1], cur_odom[0]
    origin = torch.tensor([cur_x, cur_y, yaw], dtype=torch.float32, device=device).unsqueeze(0)

    # -----------------------------
    # Base increments, lengths, and headings (local frame)
    # -----------------------------
    base_dxy = base_lib[:, 1:, :] - base_lib[:, :-1, :]     # (N, T-1, 2)
    base_step_len = torch.norm(base_dxy, dim=-1)           # (N, T-1)
    # headings for each step: atan2(dy, dx) -> length (N, T-1)
    base_headings = torch.atan2(base_dxy[..., 1], base_dxy[..., 0])  # (N, T-1)

    # Build theta sequence of length T so that theta[0] = base_headings[:,0]
    # theta_seq = [theta0, heading0, heading1, ..., heading_{T-2}] -> len T
    theta0 = base_headings[:, 0:1]                         # (N,1)
    theta_seq = torch.cat([theta0, base_headings], dim=1)  # (N, T)

    # Initial dtheta = differences of theta_seq -> (N, T-1)
    dtheta_init = theta_seq[:, 1:] - theta_seq[:, :-1]
    # wrap to [-pi, pi]
    dtheta_init = (dtheta_init + torch.pi) % (2 * torch.pi) - torch.pi

    # -----------------------------
    # Variables to optimize
    # -----------------------------
    # dtheta starts at library values (so initial trajectory == library)
    dtheta = dtheta_init.clone().detach().requires_grad_(True)
    dtheta = torch.nn.Parameter(dtheta)                    # (N, T-1)

    # step_scale multiplicative factor per step (initial 1.0)
    if allow_step_scale:
        step_scale = torch.ones_like(base_step_len, device=device, dtype=torch.float32)
        step_scale = torch.nn.Parameter(step_scale)       # (N, T-1)
    else:
        step_scale = None

    params = [dtheta] + ([step_scale] if step_scale is not None else [])
    opt = torch.optim.Adam(params, lr=lr)

    # -----------------------------
    # Integration helper (local frame)
    # -----------------------------
    def integrate(dtheta_param, step_scale_param):
        """
        Input:
          dtheta_param: (N, T-1)
          step_scale_param: (N, T-1) or None
        Output:
          traj_local: (N, T, 2)
        """
        # Build theta sequence: theta[0] = theta0 (from base), then cumsum dtheta
        theta0_local = theta0                          # (N,1) from base
        theta = torch.cumsum(torch.cat([theta0_local, dtheta_param], dim=1), dim=1)  # (N,T)

        # per-step lengths = base_step_len * step_scale (or just base_step_len)
        if step_scale_param is None:
            sl = base_step_len                         # (N, T-1)
        else:
            sl = base_step_len * step_scale_param     # (N, T-1)

        dx = torch.cos(theta[:, :-1]) * sl            # (N, T-1)
        dy = torch.sin(theta[:, :-1]) * sl            # (N, T-1)

        increments = torch.stack([dx, dy], dim=-1)    # (N, T-1, 2)
        pos0 = base_lib[:, :1]                        # (N,1,2)
        traj = torch.cat([pos0, increments], dim=1).cumsum(dim=1)  # (N, T, 2)

        return traj, theta

    # -----------------------------
    # Optimization loop
    # -----------------------------
    for it in range(iterations):
        opt.zero_grad()

        traj_local, theta = integrate(dtheta, step_scale)

        # convert to global frame (batch)
        traj_global = to_global_frame_torch(traj_local, origin.expand(N, -1))  # (N,T,2)

        # vlib = traj_global.clone().detach().cpu()
        # for i,traj in enumerate(vlib):
        #     # color = 'g' if valid[i] else 'r'
        #     plt.plot(traj[:, 0], traj[:, 1])
        # plt.show()

        # raster cost
        costs = sampler.sample_raster(traj_global)       # expected (N, T, 1) or (N,T)
        # normalize cost shape
        if costs.ndim == 3 and costs.shape[-1] == 1:
            costs = costs[..., 0]
        costmap_loss = costs.mean()
        costmap_loss *= .1

        # curvature penalty: penalize squared turning rates (dtheta)
        all_curv_loss = (dtheta ** 2).mean()
        all_curv_loss *= 10
        max_dtheta = np.pi/18
        curv_loss = (dtheta.abs() - max_dtheta).clamp(min=0).sum()
        # print(dtheta.min(), dtheta.max())
        curv_loss = curv_loss**2
        curv_loss *= curvature_weight

        # step deviation penalty: keep step_scale near 1
        if step_scale is not None:
            step_dev = ((step_scale - 1.0) ** 2).mean()
        else:
            step_dev = torch.tensor(0.0, device=device)

        step_dev *= step_dev_weight

        loss = costmap_loss + curv_loss + step_dev + all_curv_loss
        # print(costmap_loss, curv_loss, step_dev, all_curv_loss)

        loss.backward()
        opt.step()

        # optional: clamp step_scale so it doesn't deviate too much
        # if step_scale is not None:
        #     with torch.no_grad():
        #         step_scale.clamp_(1.0 - max_step_scale_dev, 1.0 + max_step_scale_dev)
        #     # also wrap dtheta to [-pi, pi] to keep angles numerically nice
        #     with torch.no_grad():
        #         dtheta[:] = (dtheta + torch.pi) % (2 * torch.pi) - torch.pi

    # -----------------------------
    # Final trajectory (global)
    # -----------------------------
    with torch.no_grad():
        final_local, _ = integrate(dtheta, step_scale)
        # final_global = to_global_frame_torch(final_local, origin.expand(N, -1))

    return final_local.cpu().numpy()

class CostmapEval():
    def __init__(self, device='cuda'):
        tif_path = "/home/tartandriver/tartandriver_ws/src/core/mission_manager/gps_maps/gascola.tif"
        self.device = device
        self.tif = rasterio.open(tif_path)
        self.TT = torch.from_numpy(np.array(self.tif.transform).reshape(3,3)).float().to(device)
        self.TT_inv = torch.inverse(self.TT).to(device)
        costmap = np.load('global_costmap_newest.npy')

        self.set_map(costmap)

    def set_map(self, costmap):
        costmap_blurred = np.clip(costmap,0,1)
        costmap_blurred = cv2.GaussianBlur(costmap, (3, 3), 0)

        self.costmap = torch.from_numpy(costmap_blurred).to(self.device)

        # grad_y, grad_x = np.gradient(costmap_blurred)  # rows -> y, cols -> x
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(grad_y)
        # ax[1].imshow(grad_x)
        # plt.show()
        # self.grad_x = torch.from_numpy(grad_x).float().to(self.device)
        # self.grad_y = torch.from_numpy(grad_y).float().to(self.device)

    def sample_raster(self,traj):
        B, T, _ = traj.shape

        row = (traj[..., 0] - self.TT[0,-1])/self.TT[0,0]
        col = (traj[..., 1] - self.TT[1,-1])/self.TT[1,1]

        H, W = self.costmap.shape

        # Normalize pixel coords to [-1, 1]
        gx = 2.0 * (row / (W - 1)) - 1.0
        gy = 2.0 * (col / (H - 1)) - 1.0

        # grid_sample expects (B, H_out, W_out, 2)
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(1)  # (B, 1, T, 2)

        raster_batch = self.costmap.unsqueeze(0).expand(B, -1, -1, -1)  # (B,C,H,W), shared storage
        sampled = F.grid_sample(raster_batch, grid, mode="bilinear", align_corners=True)
        # sampled: (B, C, 1, T)

        # cviz = self.costmap.clone()
        # cviz[row.flatten().long(), col.flatten().long()] = 5
        # plt.imshow(cviz.cpu().numpy())
        # plt.scatter(row.detach().cpu().flatten(), col.detach().cpu().flatten(), c=sampled.detach().cpu().numpy(), cmap='magma')
        # # plt.show()
        # plt.savefig("grid_sample.png", dpi=300, bbox_inches='tight')
        # plt.clf()
        # plt.close()

        return sampled.squeeze(2).permute(0, 2, 1)  # (B, T, C)
    
    def sample_gradients(self, traj):
        B, T, _ = traj.shape
        row = (traj[...,0] - self.TT[0,-1])/self.TT[0,0]
        col = (traj[...,1] - self.TT[1,-1])/self.TT[1,1]

        H, W = self.costmap.shape

        # Normalize to [-1,1] for grid_sample
        gx = 2.0 * (col / (W-1)) - 1.0
        gy = 2.0 * (row / (H-1)) - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(1)

        grad_x_sampled = F.grid_sample(self.grad_x.unsqueeze(0).unsqueeze(0).expand(B,1,H,W), grid, align_corners=True)
        grad_y_sampled = F.grid_sample(self.grad_y.unsqueeze(0).unsqueeze(0).expand(B,1,H,W), grid, align_corners=True)

        # Convert to world coordinates
        dx = grad_x_sampled.squeeze(2).permute(0,2,1) / self.TT[0,0]
        dy = grad_y_sampled.squeeze(2).permute(0,2,1) / self.TT[1,1]
        
        return torch.stack([dx, dy], dim=-1)