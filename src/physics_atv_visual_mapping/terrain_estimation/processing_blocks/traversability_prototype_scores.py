import yaml
import torch
import torch_scatter

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.base import TerrainEstimationBlock

class TraversabilityPrototypeScore(TerrainEstimationBlock):
    """
    Compute a per-cell min and max height
    """
    def __init__(self, voxel_metadata, voxel_n_features, vfm_feature_key, mask_key, prototype_fp, det_thresh_fp, device='cpu'):
        super().__init__(voxel_metadata, voxel_n_features, device)
        self.vfm_feature_key = vfm_feature_key
        self.mask_key = mask_key
        self.det_thresh_vals = yaml.safe_load(open(det_thresh_fp, 'r'))

        prototypes = torch.load(prototype_fp)
        
        self.ptype_keys = prototypes['names']
        self.ptype_obstacle = prototypes['is_obstacle'].to(self.device)
        self.ptype_modalities = prototypes['modality']
        self.compute_det_threshs()
    
    def compute_det_threshs(self):
        threshs = torch.tensor(
            [self.det_thresh_vals[mod] for mod in self.ptype_modalities],
            dtype=torch.float, device=self.device
        )
        print('det threshs = {}'.format(threshs))
        self.ptype_det_threshs = threshs

    def to(self, device):
        self.device = device
        return self

    @property
    def output_keys(self):
        return ["obstacle_max_csim", "nonobstacle_max_csim", "cost"]

    def run(self, voxel_grid, bev_grid):
        vfm_fks = [x for x in bev_grid.feature_keys if self.vfm_feature_key in x]
        vfm_fidxs = [bev_grid.feature_keys.index(x) for x in vfm_fks]

        mask_idx = bev_grid.feature_keys.index(self.mask_key)

        res_idxs = [bev_grid.feature_keys.index(x) for x in self.output_keys]
        
        ptype_scores = bev_grid.data[..., vfm_fidxs]
        mask = bev_grid.data[..., mask_idx] > 1e-4

        ptype_scores = (ptype_scores - self.ptype_det_threshs.view(1, 1, -1)).clip(0., 1.)

        obstacle_csim = ptype_scores[..., self.ptype_obstacle]
        nonobstacle_csim = ptype_scores[..., ~self.ptype_obstacle]

        obs_csim_max = obstacle_csim.max(dim=-1)[0]
        nonobs_csim_max = nonobstacle_csim.max(dim=-1)[0]

        cost = (obs_csim_max > 0.).float()
        cost[~mask] = 0.
        
        res_data = torch.stack([
            obs_csim_max,
            nonobs_csim_max,
            cost
        ], dim=-1)

        bev_grid.data[..., res_idxs] = res_data

        return bev_grid