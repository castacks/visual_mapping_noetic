import os
import yaml
import time
import argparse

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from physics_atv_visual_mapping.image_processing.image_pipeline import setup_image_pipeline
from physics_atv_visual_mapping.utils import *
"""
Create a traversability prototypes object
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to config")
    parser.add_argument("--ptype_config", type=str, required=True, help='path to list of prototypes (likely in data/ptypes.yaml)')
    parser.add_argument("--save_to", type=str, required=True, help='path to save prototypes')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))

    image_pipeline = setup_image_pipeline(config)
    
    radio_model = image_pipeline.blocks[0]

    ptype_conf = yaml.safe_load(open(args.ptype_config, "r"))
    ptypes = {
        "names": [],
        "embeddings": [],
        "is_obstacle": [],
        "modality": [],
    }

    for ptype in ptype_conf:
        ptypes['names'].append(ptype['name'])
        ptypes['is_obstacle'].append(ptype['obstacle'])

        text_embed = radio_model.embed_text(ptype['desc'])
        ptypes['embeddings'].append(text_embed)
        ptypes['modality'].append('text')

    ptypes['embeddings'] = torch.stack(ptypes['embeddings'], dim=0)  
    ptypes['is_obstacle'] = torch.tensor(ptypes['is_obstacle'], dtype=torch.bool)

    torch.save(ptypes, args.save_to)