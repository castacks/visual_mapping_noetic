import torch
import torch_scatter

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.base import TerrainEstimationBlock
from physics_atv_visual_mapping.terrain_estimation.processing_blocks.utils import get_adjacencies

class MRFTerrainEstimation(TerrainEstimationBlock):
    """
    Compute a per-cell min and max height
    """
    def __init__(self, voxel_metadata, voxel_n_features, input_layer, mask_layer, itrs, alpha, beta, lr, device):
        super().__init__(voxel_metadata, voxel_n_features, device)
        self.input_layer = input_layer
        self.mask_layer = mask_layer
        self.itrs = itrs
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        
    def to(self, device):
        self.device = device
        return self

    @property
    def output_keys(self):
        return ["terrain"]

    def run(self, voxel_grid, bev_grid):
        input_idx = bev_grid.feature_keys.index(self.input_layer)
        input_data = bev_grid.data[..., input_idx].clone()

        mask_idx = bev_grid.feature_keys.index(self.mask_layer)
        mask = bev_grid.data[..., mask_idx] > 1e-4

        terrain_estimate = input_data.clone()

        for i in range(self.itrs):
            dz = torch.zeros_like(terrain_estimate)

            measurement_update = input_data - terrain_estimate
            measurement_update[~mask] = 0.

            terrain_adj = get_adjacencies(data=terrain_estimate)
            valid_adj = get_adjacencies(data=mask)

            neighbor_update = terrain_adj - terrain_estimate.unsqueeze(0)
            neighbor_update[~valid_adj] = 0.
            neighbor_update = neighbor_update.sum(dim=0) / valid_adj.sum(dim=0)
            neighbor_update[~mask] = 0.

            dz = self.lr * (self.alpha*measurement_update + self.beta*neighbor_update)

            terrain_estimate += dz
        
        output_data_idx = bev_grid.feature_keys.index(self.output_keys[0])
        bev_grid.data[..., output_data_idx] = terrain_estimate

        return bev_grid