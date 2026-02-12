from physics_atv_visual_mapping.frontier_estimation.registry import Registry

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


POLICY_REGISTRY = Registry()

class BaseHeadingPolicy(nn.Module):
    def forward(self, heatmap, bin_indices, pixel_props):
        raise NotImplementedError
    
@POLICY_REGISTRY.register("dirichlet")
class DirichletPolicy(BaseHeadingPolicy):
    def __init__(self, num_bins=36, tau=2.0):
        super().__init__()
        self.num_bins = num_bins
        self.tau = tau

    def forward(self, heatmap, bin_indices, pixel_props):
        B = heatmap.shape[0]
        z = self.tau * heatmap.flatten(1)
        exp_z = torch.exp(z)

        evidence = torch.zeros(B, self.num_bins, device=heatmap.device)
        evidence.scatter_add_(
            1,
            bin_indices.unsqueeze(0).expand(B, -1),
            exp_z,
        )


        evidence /= pixel_props.unsqueeze(0) + 1e-6


        alpha = 1. + F.softplus(torch.log(evidence + 1e-8))

        return alpha
