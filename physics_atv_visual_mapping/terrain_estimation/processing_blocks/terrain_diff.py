import torch
import torch_scatter

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.base import TerrainEstimationBlock
from physics_atv_visual_mapping.terrain_estimation.processing_blocks.utils import sobel_x_kernel, sobel_y_kernel, apply_kernel

class TerrainDiff(TerrainEstimationBlock):
    """
    Compute a per-cell min and max height
    """
    def __init__(self, voxel_metadata, voxel_n_features, terrain_layer, max_elevation_layer, max_elevation_mask_layer, overhang, device):
        super().__init__(voxel_metadata, voxel_n_features, device)
        self.terrain_layer = terrain_layer
        self.max_elevation_layer = max_elevation_layer
        self.max_elevation_mask_layer = max_elevation_mask_layer
        self.overhang = overhang

    def to(self, device):
        self.device = device
        return self

    @property
    def output_keys(self):
        return ["diff"]

    def run(self, voxel_grid, bev_grid):
        terrain_idx = bev_grid.feature_keys.index(self.terrain_layer)
        terrain_data = bev_grid.data[..., terrain_idx].clone()
        
        max_elevation_idx = bev_grid.feature_keys.index(self.max_elevation_layer)
        max_elevation_data = bev_grid.data[..., max_elevation_idx].clone()

        max_elevation_mask_idx = bev_grid.feature_keys.index(self.max_elevation_mask_layer)
        max_elevation_mask = bev_grid.data[..., max_elevation_mask_idx] > 1e-4

        #divide by resolution to get slope as m/m instead of m/cell
        diff = (max_elevation_data - terrain_data).clip(0., self.overhang)
        diff[~max_elevation_mask] = 0.

        diff_idx = bev_grid.feature_keys.index(self.output_keys[0])
        bev_grid.data[..., diff_idx] = diff

        return bev_grid