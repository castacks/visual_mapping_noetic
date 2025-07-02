import rospy
import torch
import argparse

from vfm_voxel_interfaces.srv import UpdatePrototype

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--service_id', type=str, required=False, default='/crl_rzr/vfm_voxel_mapping/update_prototypes')
    parser.add_argument('--id', type=str, required=False, default='debug')
    parser.add_argument('--nonobstacle', action='store_true', help='set flag if not obstacle')
    parser.add_argument('--ndim', required=False, default=768, help='dim of the visual encoder')
    args = parser.parse_args()

    data = torch.rand(args.ndim).numpy()

    update_ptype_srv = rospy.ServiceProxy(args.service_id, UpdatePrototype)
    resp = update_ptype_srv(
        id=args.id,
        is_obstacle=False if args.nonobstacle else True,
        data=data
    )

    print('received res = {}, fp = {}'.format(resp.success, resp.save_path))