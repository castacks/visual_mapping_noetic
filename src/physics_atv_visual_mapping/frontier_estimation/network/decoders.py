import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from physics_atv_visual_mapping.frontier_estimation.registry import Registry
DECODER_REGISTRY = Registry()

class BaseDecoder(nn.Module):
    out_channels: int  # optional metadata

    def forward(self, x):
        raise NotImplementedError


@DECODER_REGISTRY.register("upsample_v2")
class UpsampleHeatmapDecoder(BaseDecoder):
    def __init__(
        self,
        in_channels,
        channels=(256, 128, 32),
        kernel_sizes=[3,3,3],
        upsample_mode="bilinear",
        padding_mode="reflect",
    ):
        super().__init__()

        self.in_channels = in_channels

        layers = [
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(in_channels, channels[0], kernel_size=1),
            nn.GroupNorm(32, channels[0]),
            nn.ReLU(inplace=True),
        ]

        prev_ch = channels[0]
        for ch, kernel_size in zip(channels, kernel_sizes):
            layers += [
                nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False),
                nn.Conv2d(
                    prev_ch, ch,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    padding_mode=padding_mode,
                    bias=False,
                ),
                nn.GroupNorm(32, ch),
                nn.ReLU(inplace=True),
            ]
            prev_ch = ch

        layers.append(nn.Conv2d(prev_ch, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        x = F.elu(x)
        return x.squeeze(1)






class DinoHeadingCostHead_Legacy(nn.Module):
    def __init__(self, in_channels=1152, hidden_dim=128, num_bins=128, upsample_scale=8):
        super().__init__()
        self.num_bins = num_bins
        self.upsample_scale = upsample_scale
        self.in_channels = in_channels

        # Decide number of doubling layers to reach upsample_scale
        num_upsample_layers = int(np.ceil(np.log2(upsample_scale)))
        current_scale = 1

        print("NUM UPSAMPLE ", num_upsample_layers)


        layers = []
        layers += [
            nn.Conv2d(in_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        ]

        for k, out_ch in zip([7, 5, 3], [256, 128, 64]):
            layers += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(512 if out_ch == 256 else prev_ch,
                        out_ch,
                        kernel_size=k,
                        padding=k // 2,
                        padding_mode='replicate',
                        bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            prev_ch = out_ch

        layers.append(nn.Conv2d(prev_ch, 1, kernel_size=1))
        self.decoder = nn.Sequential(*layers)


    def register_headings(self, K, H_orig, W_orig):
        dummy_input = torch.zeros(1, self.in_channels,28, 53).cuda()
        with torch.no_grad():
            heatmap = self.decoder(dummy_input)
        H, W = heatmap.shape[2], heatmap.shape[3]

        ys = torch.linspace(-(H-1)/2, (H-1)/2, H)
        xs = torch.linspace(-(W-1)/2, (W-1)/2, W)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H_up, W_up]

        # pixel_headings = compute_pixel_headings(K, grid_y, grid_x, H_orig=H_orig, W_orig=W_orig)
        pixel_headings = compute_pixel_headings(K, H, W, H_orig=H_orig, W_orig=W_orig)
        #TODO is this ok to do
        pixel_headings *= -1
        
        # import pdb;pdb.set_trace()

        bin_edges = torch.linspace(-np.pi/2, np.pi/2, self.num_bins + 1)
        bin_indices = torch.bucketize(pixel_headings, bin_edges) - 1
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)

        bin_indices_flat = bin_indices.flatten().cuda()
        self.register_buffer('bin_indices', bin_indices_flat)
        pixel_counts = torch.bincount(bin_indices_flat, minlength=self.num_bins).float()
        self.register_buffer('pixel_counts', pixel_counts)

        self.valid_idxs = torch.where(pixel_counts != 0)[0]

        # import pdb;pdb.set_trace()

    def forward(self, x, tau=2.):
        """
        Args:
            x: [B, C, H, W] input features
            tau: temperature for softmax-like aggregation
        Returns:
            heatmap: [B, H, W] per-pixel scores
            alpha: [B, num_bins] Dirichlet parameters
            probs: [B, num_bins] expected probability per heading
        """
        B, _, H_small, W_small = x.shape
        device = x.device

        # --- Per-pixel heatmap ---
        heatmap = self.decoder(x).squeeze(1)  # [B, H, W]
        heatmap = F.elu(heatmap)  # preserves some negative values; could use ReLU if preferred

        # --- Flatten pixels ---
        heatmap_flat = heatmap.flatten(1)  # [B, H*W]
        B, N = heatmap_flat.shape

        # --- Compute softmax-style per-heading evidence ---
        z_scaled = tau * heatmap_flat  # [B, N]
        # z_scaled = torch.clamp(tau * heatmap_flat, max=20.0)
        exp_z = torch.exp(z_scaled)

        # binned sum of exponentiated logits
        evidence = torch.zeros(B, self.num_bins, device=device)
        evidence.scatter_add_(1, self.bin_indices.unsqueeze(0).expand(B, -1), exp_z)

        # --- logsumexp approximation per heading ---
        e_h = torch.log(evidence + 1e-8)  # [B, num_bins]

        # --- Dirichlet parameters ---
        alpha = 1.0 + F.softplus(e_h)  # [B, num_bins], alpha > 1 ensures minimal prior

        # --- Expected probabilities per heading ---
        probs = alpha / alpha.sum(dim=1, keepdim=True)  # E[pi_h]

        return heatmap, alpha