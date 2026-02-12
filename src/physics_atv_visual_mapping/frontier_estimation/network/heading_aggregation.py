from physics_atv_visual_mapping.frontier_estimation.registry import Registry

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


BINNER_REGISTRY = Registry()

def compute_pixel_headings(K, H, W, H_orig=None, W_orig=None, device='cpu'):
    """
    Compute per-pixel horizontal headings for a camera image of arbitrary size.
    Works for both odd and even image dimensions.

    Args:
        K: camera intrinsic matrix [3,3] (numpy or torch) for original resolution
        H, W: output image size
        H_orig, W_orig: original image size K was calibrated for (if None, assume H_orig=H, W_orig=W)
        device: 'cpu' or 'cuda'

    Returns:
        pixel_headings: [H, W] tensor with heading angles in radians (-pi/2 to pi/2)
    """
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).float().to(device)
    else:
        K = K.float().to(device)

    if H_orig is None:
        H_orig = H
    if W_orig is None:
        W_orig = W

    # Rescale intrinsics
    scale_x = W / W_orig
    scale_y = H / H_orig
    fx = K[0, 0] * scale_x
    fy = K[1, 1] * scale_y
    cx = K[0, 2] * scale_x
    cy = K[1, 2] * scale_y

    # Create centered floating-point pixel grid
    # Coordinates of pixel centers range from 0 to W-1 and 0 to H-1
    xs = torch.linspace(0, W-1, W, device=device)
    ys = torch.linspace(0, H-1, H, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')  # [H, W]

    # Compute horizontal heading relative to optical center
    pixel_headings = torch.atan2(grid_x - cx, fx)

    return pixel_headings  # [H, W]

class BaseHeadingBinner(nn.Module):
    num_bins: int

    def initialize(self, decoder, K, device, **kwargs):
        raise NotImplementedError


@BINNER_REGISTRY.register("pinhole")
class PixelHeadingBinner(BaseHeadingBinner):
    def __init__(
        self,
        num_bins,
        flip_sign=True,
        in_hw=(28, 53),
        pixel_norm="median",
    ):
        super().__init__()
        self.num_bins = num_bins
        self.flip_sign = flip_sign
        self.in_hw = in_hw
        self.pixel_norm = pixel_norm

    @torch.no_grad()
    def initialize(self, decoder, K, H_orig, W_orig, device):
        dummy = torch.zeros(
            1,
            decoder.in_channels,
            *self.in_hw,
            device=device,
        )

        heatmap = decoder(dummy)
        H, W = heatmap.shape[-2:]

        headings = compute_pixel_headings(K, H, W, H_orig, W_orig).to(device)
        if self.flip_sign:
            headings *= -1

        # edges = torch.linspace(
        #     -np.pi/2, np.pi/2, self.num_bins + 1, device=device
        # )

        edges = torch.linspace(
            -np.pi, np.pi, self.num_bins + 1, device=device
        )

        bin_indices = torch.bucketize(headings, edges) - 1
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        bin_indices = bin_indices.flatten()

        # --- Pixel statistics ---
        pixel_counts = torch.bincount(
            bin_indices, minlength=self.num_bins
        ).float()

        if self.pixel_norm == "median":
            denom = pixel_counts[pixel_counts != 0].median()
        elif self.pixel_norm == "sum":
            denom = pixel_counts.sum()
        else:
            denom = 1.0

        pixel_props = pixel_counts / (denom + 1e-6)

        valid_idxs = torch.where(pixel_counts != 0)[0]

        # --- Register buffers ---
        self.register_buffer("bin_indices", bin_indices)
        self.register_buffer("pixel_counts", pixel_counts)
        self.register_buffer("pixel_props", pixel_props)
        self.valid_idxs = valid_idxs
