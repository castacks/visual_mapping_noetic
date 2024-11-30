import torch
import torch_scatter

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.base import TerrainEstimationBlock
from physics_atv_visual_mapping.terrain_estimation.processing_blocks.utils import sobel_x_kernel, sobel_y_kernel, apply_kernel

class Slope(TerrainEstimationBlock):
    """
    Compute a per-cell min and max height
    """
    def __init__(self, voxel_metadata, voxel_n_features, input_layer, mask_layer, device):
        super().__init__(voxel_metadata, voxel_n_features, device)
        self.input_layer = input_layer
        self.mask_layer = mask_layer

        self.sobel_x = sobel_x_kernel().to(device)
        self.sobel_y = sobel_y_kernel().to(device)
        self.box = torch.ones_like(self.sobel_x)/9.

    def to(self, device):
        self.sobel_x = sobel_x.to(device)
        self.sobel_y = sobel_y.to(device)
        self.device = device
        return self

    @property
    def output_keys(self):
        return ["slope_x", "slope_y", "slope"]

    def run(self, voxel_grid, bev_grid):
        terrain_idx = bev_grid.feature_keys.index(self.input_layer)
        terrain_data = bev_grid.data[..., terrain_idx].clone()

        mask_idx = bev_grid.feature_keys.index(self.mask_layer)
        mask = bev_grid.data[..., mask_idx] > 1e-4

        #only take slopes if all convolved elements valid
        valid_mask = apply_kernel(kernel=self.box, data=mask.float()) > 0.9999

        #divide by resolution to get slope as m/m instead of m/cell
        slope_x = apply_kernel(kernel=self.sobel_x, data=terrain_data) / bev_grid.metadata.resolution[0]
        slope_x[~valid_mask] = 0.
        slope_y = apply_kernel(kernel=self.sobel_y, data=terrain_data) / bev_grid.metadata.resolution[1]
        slope_y[~valid_mask] = 0.
        slope = torch.hypot(slope_x, slope_y)

        slope_x_idx = bev_grid.feature_keys.index(self.output_keys[0])
        bev_grid.data[..., slope_x_idx] = slope_x

        slope_y_idx = bev_grid.feature_keys.index(self.output_keys[1])
        bev_grid.data[..., slope_y_idx] = slope_y

        slope_idx = bev_grid.feature_keys.index(self.output_keys[2])
        bev_grid.data[..., slope_idx] = slope

        return bev_grid