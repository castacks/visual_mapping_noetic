import torch.nn as nn
from physics_atv_visual_mapping.frontier_estimation.network.decoders import DECODER_REGISTRY
from physics_atv_visual_mapping.frontier_estimation.network.heading_aggregation import BINNER_REGISTRY
from physics_atv_visual_mapping.frontier_estimation.network.heading_policy import POLICY_REGISTRY

class VFMFrontierEstimator(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()

        decoder_name = config['decoder']['name']
        decoder_kwargs = config['decoder']['kwargs']

        binner_name = config['binner']['name']
        binner_kwargs = config['binner']['kwargs']

        policy_name = config['policy']['name']
        policy_kwargs = config['policy']['kwargs']

        self.decoder = DECODER_REGISTRY.build(decoder_name, **decoder_kwargs)
        self.binner = BINNER_REGISTRY.build(binner_name, **binner_kwargs)
        self.policy = POLICY_REGISTRY.build(policy_name, **policy_kwargs)

        self.policy.num_bins = self.binner.num_bins

    def register_headings(self, **kwargs):
        self.binner.initialize(self.decoder, **kwargs)

    def forward(self, x):
        
        heatmap = self.decoder(x)
        logits = self.policy(
            heatmap,
            self.binner.bin_indices,
            self.binner.pixel_props,
        )
        return heatmap, logits
