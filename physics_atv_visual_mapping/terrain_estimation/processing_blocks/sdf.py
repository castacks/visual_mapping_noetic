import torch
import skfmm
import numpy as np

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.base import TerrainEstimationBlock

class SDF(TerrainEstimationBlock):
    """
    Compute a per-cell min and max height
    """
    def __init__(self, voxel_metadata, voxel_n_features, input_layer, signed, device):
        super().__init__(voxel_metadata, voxel_n_features, device)
        self.input_layer = input_layer
        self.signed = signed
        
    def to(self, device):
        self.device = device
        return self

    @property
    def output_keys(self):
        return ["{}_sdf".format(self.input_layer)]

    def run(self, voxel_grid, bev_grid):
        dx = self.voxel_metadata.resolution[0].item()

        input_idx = bev_grid.feature_keys.index(self.input_layer)
        input_data = bev_grid.data[..., input_idx]
        output_idx = bev_grid.feature_keys.index(self.output_keys[0])

        _mask = (input_data > 1e-4).cpu().numpy()
        seed = np.ones(_mask.shape)
        seed[_mask] = -1
        distance_transform = skfmm.distance(seed, dx=dx)
        
        if not self.signed:
            distance_transform = np.clip(distance_transform, 0, float("inf"))

        sdf = torch.tensor(distance_transform, dtype=torch.float32, device=self.device)
        bev_grid.data[..., output_idx] = sdf
        return bev_grid