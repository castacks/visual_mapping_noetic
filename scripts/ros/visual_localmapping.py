import yaml
import rospy
import ros_numpy
import tf2_ros
import torch
import numpy as np

np.float = np.float64 #hack for numpify

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry

from physics_atv_visual_mapping.localmapping.localmapping import *
from physics_atv_visual_mapping.utils import *
    
class VisualMappingNode:
    """
    Hacky implementation of visual mapping node for debug
    """
    def __init__(self, config):
        self.localmap = None
        self.pcl_msg = None
        self.odom_msg = None
        self.odom_frame = None
        self.device = config['device']
        self.base_metadata = config['localmapping']['metadata']
        self.localmap_ema = config['localmapping']['ema']

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pcl_sub = rospy.Subscriber(config['pointcloud']['topic'], PointCloud2, self.handle_pointcloud, queue_size=1)
        self.odom_sub = rospy.Subscriber(config['odometry']['topic'], Odometry, self.handle_odom, queue_size=10)

        self.rate = rospy.Rate(10)

    def handle_pointcloud(self, msg):
        #temp hack
        self.pcl_msg = msg
        self.pcl_msg.header.frame_id = 'vehicle'

    def handle_odom(self, msg):
        if self.odom_frame is None:
            self.odom_frame = msg.header.frame_id

        self.odom_msg = msg

    def preprocess_inputs(self):
        """
        Return the update pcl and new metadata
        """
        if self.pcl_msg is None:
            rospy.logwarn_throttle(1.0, 'no pcl msg received')
            return None

        if self.odom_msg is None:
            rospy.logwarn_throttle(1.0, 'no odom msg received')
            return None

        if self.odom_msg.child_frame_id != self.pcl_msg.header.frame_id:
            rospy.logwarn_throttle(1.0, 'for now, need pcls in the child frame of odom (got {}, expected {})'.format(self.pcl_msg.header.frame_id, self.odom_msg.child_frame_id))
            return None

        if not self.tf_buffer.can_transform(self.odom_frame, self.pcl_msg.header.frame_id, self.pcl_msg.header.stamp):
            rospy.logwarn_throttle(1.0, 'cant tf from {} to {} at {}'.format(self.odom_frame, self.pcl_msg.header.frame_id, self.pcl_msg.header.stamp))
            return None

        tf_msg = self.tf_buffer.lookup_transform(self.odom_frame, self.pcl_msg.header.frame_id, self.pcl_msg.header.stamp)

        pcl_htm = tf_msg_to_htm(tf_msg).to(self.device)
        pcl = pcl_msg_to_xyzrgb(self.pcl_msg).to(self.device)
        pcl_odom = transform_points(pcl, pcl_htm)

        _metadata = {
            'origin': torch.tensor([
                self.odom_msg.pose.pose.position.x-0.5*self.base_metadata['length_x'],
                self.odom_msg.pose.pose.position.y-0.5*self.base_metadata['length_y']
            ]).float().to(self.device),
            'length_x': torch.tensor(self.base_metadata['length_x']).float().to(self.device),
            'length_y': torch.tensor(self.base_metadata['length_y']).float().to(self.device),
            'resolution': torch.tensor(self.base_metadata['resolution']).float().to(self.device),
        }

        mask = torch.linalg.norm(pcl_odom[:, 3:], dim=-1) > 1e-4

        return {
            'pcl': pcl_odom[mask],
            'metadata': _metadata
        }

    def update_localmap(self, pcl, metadata):
        pcl_pos = pcl[:, :3]
        pcl_data = pcl[:, 3:]
        localmap, known_mask, metadata_out = localmap_from_pointcloud(pcl_pos, pcl_data, metadata)
        localmap_update = {
            'data': localmap,
            'known': known_mask,
            'metadata': metadata_out
        }

        if self.localmap is None:
            return localmap_update

        else:
            return aggregate_localmaps(localmap_update, self.localmap, ema=self.localmap_ema)

    def spin(self):
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['figure.raise_window'] = False
        import time

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        plt.show(block=False)
        while not rospy.is_shutdown():
            rospy.loginfo('spinning...')

            t1 = time.time()
            res = self.preprocess_inputs()
            t2 = time.time()

            if res:
                rospy.loginfo('updating localmap...')

                t3 = time.time()
                self.localmap = self.update_localmap(**res)
                t4 = time.time()

                rospy.loginfo('Timing:\n\tpreproc: {:.6f}s\n\tlocalmap: {:.6f}s'.format(t2-t1, t4-t3))

                #debug viz
                extent = (
                    self.localmap['metadata']['origin'][0].item(),
                    self.localmap['metadata']['origin'][0].item() + self.localmap['metadata']['length_x'].item(),
                    self.localmap['metadata']['origin'][1].item(),
                    self.localmap['metadata']['origin'][1].item() + self.localmap['metadata']['length_y'].item()
                )
                ax.cla()
                ax.imshow(self.localmap['data'].permute(1,0,2).cpu(), extent=extent, origin='lower')
                ax.scatter(
                    self.odom_msg.pose.pose.position.x,
                    self.odom_msg.pose.pose.position.y,
                    marker='x',
                    c='r'
                )
                plt.pause(1e-2)

            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('visual_mapping')

    config_fp = rospy.get_param("~config_fp")
    config = yaml.safe_load(open(config_fp, 'r'))

    visual_mapping_node = VisualMappingNode(config)

    visual_mapping_node.spin()
