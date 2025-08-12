import yaml
import rospy
import torch
import argparse

from physics_atv_visual_mapping.image_processing.image_pipeline import setup_image_pipeline
from physics_atv_visual_mapping.utils import *

from vfm_voxel_interfaces.srv import UpdatePrototype

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to config")
    parser.add_argument('--service_id', type=str, required=False, default='/crl_rzr/vfm_voxel_mapping/update_prototypes')
    parser.add_argument('--id', type=str, required=False, default='debug')
    parser.add_argument('--nonobstacle', action='store_true', help='set flag if not obstacle')
    parser.add_argument('--desc', type=str, required=True, help='the description of the prototype to embed')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, "r"))

    image_pipeline = setup_image_pipeline(config)
    
    radio_model = image_pipeline.blocks[0]
    
    text_embed = radio_model.embed_text(args.desc)
    data = text_embed.cpu().numpy()

    print('adding prototype id {} (desc = {})'.format(args.id, args.desc))

    update_ptype_srv = rospy.ServiceProxy(args.service_id, UpdatePrototype)
    resp = update_ptype_srv(
        id=args.id,
        is_obstacle=False if args.nonobstacle else True,
        modality='text',
        data=data
    )

    print('received res = {}, fp = {}'.format(resp.success, resp.save_path))