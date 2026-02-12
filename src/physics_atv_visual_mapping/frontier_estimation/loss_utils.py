import torch
import numpy as np
from physics_atv_visual_mapping.frontier_estimation.registry import Registry
from torch.nn import MSELoss
import torch.nn.functional as F

LOSS_REGISTRY = Registry()

@LOSS_REGISTRY.register("emd")
class EMDLoss:
    def __init__(self):
        pass

    def __call__(self, res):
        pred_probs = res['pred_probs']
        gt_probs = res['gt_probs']
        cdf_pred = torch.cumsum(pred_probs, dim=1)
        cdf_target = torch.cumsum(gt_probs, dim=1)
        return torch.mean((cdf_pred - cdf_target)**2)

@LOSS_REGISTRY.register("rankloss")
class RankLoss:
    def __init__(self):
        pass

    def __call__(self, res):
        costs = res['gt_costs']
        evidence = res['pred_costs'] - 1.

        assert costs.shape == evidence.shape

        mask_bad  = (costs == 0)
        mask_good = (costs > 0)

        e_bad  = evidence[mask_bad]
        e_good = evidence[mask_good]

        if e_bad.numel() > 0 and e_good.numel() > 0:
            loss_rank = F.softplus(
                e_bad[:, None] - (e_good[None, :] + 0.)
            ).mean()
        else:
            loss_rank = torch.tensor(0.0, device=evidence.device)

        return loss_rank

@LOSS_REGISTRY.register("reg")
class RegLoss:
    def __init__(self):
        pass

    def __call__(self, res):
        pred_costs = res['pred_costs']
        return (pred_costs-1.).abs().sum(dim=1).mean()
    
@LOSS_REGISTRY.register("vertical_var")
class VerticalVarianceLoss:
    def __init__(self):
        pass

    def __call__(self, res):
        heatmap_ = res['pred_heatmap']
        H, W = heatmap_.shape[-2:]
        heatmap = heatmap_.view(heatmap_.shape[0],-1) - heatmap_.min()
        gt_costs = res['gt_costs']
        bin_indices = res['bin_indices'].clone()
        bin_indices -= bin_indices.min() #hack? maybe not
        B, _ = heatmap.shape
        num_bins = gt_costs.shape[1]
        # num_bins = bin_indices.max() 
        

        w_sum = torch.zeros(B, num_bins, device=heatmap.device)  # [B, num_bins]
        w_sum.scatter_add_(1, bin_indices.unsqueeze(0).expand(B, -1), heatmap)  # [B, num_bins] (sum of exp_z per bin)

        wy_sum = torch.zeros(B, num_bins, device=heatmap.device)  # [B, num_bins]

        y_coords = torch.arange(H, device=heatmap.device).float()/H  # [H] row indices
        y_coords = y_coords.unsqueeze(1).expand(H, W)  # [H, W], repeat across the width dimension
        y_coords = y_coords.flatten()  # [H * W]
        y_coords = y_coords.unsqueeze(0).expand(B, -1)  # [B, H] repeat for each batch

        wy_sum.scatter_add_(1, bin_indices.unsqueeze(0).expand(B, -1), heatmap * y_coords)  # [B, num_bins] (weighted vertical sum)

        wy2_sum = torch.zeros(B, num_bins, device=heatmap.device)  # [B, num_bins]
        wy2_sum.scatter_add_(1, bin_indices.unsqueeze(0).expand(B, -1), heatmap * (y_coords ** 2))  # [B, num_bins] (weighted vertical squared sum)

        mean_y = wy_sum / (w_sum + 1e-6)   # [B, num_bins] (E[y] per bin)
        mean_y2 = wy2_sum / (w_sum + 1e-6) # [B, num_bins] (E[y^2] per bin)

        var_y = mean_y2 - mean_y**2      # [B, num_bins] (Var[y] per bin)

        valid = gt_costs > 0

        var_y = var_y[valid]

        return var_y.mean()
    
@LOSS_REGISTRY.register("LRN")
class LRNLoss:
    def __init__(self):
        self.loss_fn = MSELoss(reduction='mean')

    def __call__(self, res):
        traj_mask = res['gt_heatmap']
        heatmap = res['pred_heatmap']

        positive_mask = traj_mask > 0
        x_indices = positive_mask.nonzero(as_tuple=True)
        column_mask = torch.zeros_like(traj_mask, dtype=torch.bool)
        column_mask[x_indices[0], :, x_indices[2]] = True
        traj_mask[column_mask] = 0
        traj_mask[positive_mask] = 1
        mask = traj_mask >= 0
        
        heatmap_up = F.interpolate(heatmap.unsqueeze(1), size=traj_mask.shape[-2:], mode='bilinear', align_corners=False)
        heatmap_up = heatmap_up[:,0]

        loss = self.loss_fn(heatmap_up[mask], traj_mask[mask])
        return loss

class LossComposer:
    def __init__(self, cfg):
        self.losses = []

        for name, spec in cfg.items():
            weight = spec.get("weight", 1.0)
            kwargs = spec.get("kwargs", {})

            loss_fn = LOSS_REGISTRY.build(name, **kwargs)

            self.losses.append({
                "name": name,
                "weight": weight,
                "fn": loss_fn,
            })

    def __call__(self, res):
        total = torch.tensor(0.0).cuda()
        logs = {}

        for entry in self.losses:
            val = entry["fn"](res)
            if entry["weight"] != -1:
                total = total + entry["weight"] * val

            logs[f"loss/{entry['name']}"] = val.detach()

        logs["loss/total"] = total.detach()
        return total, logs


