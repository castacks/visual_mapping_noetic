import yaml
import rospy
import ros_numpy
import tf2_ros
import torch
import numpy as np
import cv_bridge

np.float = np.float64 #hack for numpify

from sensor_msgs.msg import PointCloud2, CameraInfo, Image, CompressedImage
from nav_msgs.msg import Odometry

from physics_atv_visual_mapping.image_processing.anyloc_utils import DinoV2ExtractFeatures
from physics_atv_visual_mapping.pointcloud_colorization.torch_color_pcl_utils import *
from physics_atv_visual_mapping.localmapping.localmapping import *
from physics_atv_visual_mapping.utils import *
    
class DinoMappingNode:
    """
    Hacky implementation of visual mapping node for debug
    """
    def __init__(self, config):
        self.localmap = None
        self.pcl_msg = None
        self.odom_msg = None
        self.img_msg = None
        self.odom_frame = None
        self.device = config['device']
        self.base_metadata = config['localmapping']['metadata']
        self.localmap_ema = config['localmapping']['ema']
        self.last_update_time = 0.

        self.dino = DinoV2ExtractFeatures(
            dino_model=config['dino']['dino_type'],
            layer=config['dino']['dino_layer'],
            input_size=config['dino']['image_insize'],
            device=config['device']
        )
        self.dino_pca = torch.load(config['pca']['fp'], map_location=config['device'])

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.bridge = cv_bridge.CvBridge()

        self.compressed_img = config['image_processing']['image_compressed']
        if self.compressed_img:
            self.image_sub = rospy.Subscriber(config['image_processing']['image_topic'], CompressedImage, self.handle_img, queue_size=1)
        else:
            self.image_sub = rospy.Subscriber(config['image_processing']['image_topic'], Image, self.handle_img, queue_size=1)

        self.intrinsics = torch.tensor(rospy.wait_for_message(config['image_processing']['camera_info_topic'], CameraInfo).K, device=config['device']).reshape(3,3).float()
        self.dino_intrinsics = None

        self.extrinsics = pose_to_htm(np.concatenate([
            np.array(config['extrinsics']['p']),
            np.array(config['extrinsics']['q'])
        ], axis=-1))

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

    def handle_img(self, msg):
        self.img_msg = msg

    def preprocess_inputs(self):
        """
        Return the update pcl and new metadata
        """
        if self.pcl_msg is None:
            rospy.logwarn_throttle(1.0, 'no pcl msg received')
            return None

        pcl_time = self.pcl_msg.header.stamp.to_sec()
        if abs(pcl_time - self.last_update_time) < 1e-3:
            return None

        if self.odom_msg is None:
            rospy.logwarn_throttle(1.0, 'no odom msg received')
            return None

        if self.img_msg is None:
            rospy.logwarn_throttle(1.0, 'no img msg received')
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
#        pcl = pcl_msg_to_xyz(self.pcl_msg).to(self.device)
        pcl_odom = transform_points(pcl.clone(), pcl_htm)

        _metadata = {
            'origin': torch.tensor([
                self.odom_msg.pose.pose.position.x-0.5*self.base_metadata['length_x'],
                self.odom_msg.pose.pose.position.y-0.5*self.base_metadata['length_y']
            ]).float().to(self.device),
            'length_x': torch.tensor(self.base_metadata['length_x']).float().to(self.device),
            'length_y': torch.tensor(self.base_metadata['length_y']).float().to(self.device),
            'resolution': torch.tensor(self.base_metadata['resolution']).float().to(self.device),
        }

        self.last_update_time = self.pcl_msg.header.stamp.to_sec()

        #camera stuff
        if self.compressed_img:
            img = self.bridge.compressed_imgmsg_to_cv2(self.img_msg, desired_encoding='rgb8')/255.
        else:
            img = self.bridge.imgmsg_to_cv2(self.img_msg, desired_encoding='rgb8')/255.

        #run the image through dino
        dino_img = self.dino(img)[0]
        dino_img_norm = dino_img.view(-1, dino_img.shape[-1]) - self.dino_pca['mean'].view(1,-1)
        dino_pca = dino_img_norm.unsqueeze(1) @ self.dino_pca['V'].unsqueeze(0)
        dino_img = dino_pca.view(dino_img.shape[0], dino_img.shape[1], -1)

        #need to scale intrinsics to dino resolution
        dino_rx = img.shape[0] / dino_img.shape[0]
        dino_ry = img.shape[1] / dino_img.shape[1]
        dino_intrinsics = self.intrinsics.clone()
        dino_intrinsics[0, 0]/=dino_rx
        dino_intrinsics[0, 2]/=dino_rx
        dino_intrinsics[1, 1]/=dino_ry
        dino_intrinsics[1, 2]/=dino_ry

        #need to compute the transform from the odom frame to the image frame
        #do it this way to account for pcl time sync
        if not self.tf_buffer.can_transform(self.pcl_msg.header.frame_id, self.img_msg.header.frame_id, self.img_msg.header.stamp):
            rospy.logwarn_throttle(1.0, 'cant tf from {} to {} at {}'.format(self.pcl_msg.header.frame_id, self.img_msg.header.frame_id, self.img_msg.header.stamp))
            return None

        I = get_intrinsics(dino_intrinsics).to(self.device)
        E = get_extrinsics(self.extrinsics).to(self.device)

        P = obtain_projection_matrix(I, E)

        pixel_coordinates = get_pixel_from_3D_source(pcl[:, :3], P)
        lidar_points_in_frame, pixels_in_frame, ind_in_frame = get_points_and_pixels_in_frame(
            pcl[:, :3],
            pixel_coordinates,
            dino_img.shape[0],
            dino_img.shape[1]
        )

        dino_features = dino_img[pixels_in_frame[:, 1], pixels_in_frame[:, 0]]
        dino_pcl = torch.cat([pcl_odom[ind_in_frame][:, :3], dino_features], dim=-1)

        return {
            'pcl': pcl_odom,
            'metadata': _metadata,
            'image': img,
            'dino_image': dino_img,
            'dino_pcl': dino_pcl,
            'pixel_projection': pixel_coordinates[ind_in_frame]
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

        fig, axs = plt.subplots(2, 2, figsize=(4, 3))
        axs = axs.flatten()
        plt.show(block=False)
        while not rospy.is_shutdown():
            rospy.loginfo('spinning...')

            t1 = time.time()
            res = self.preprocess_inputs()
            t2 = time.time()

            if res:
                rospy.loginfo('updating localmap...')

                t3 = time.time()
                self.localmap = self.update_localmap(
                    pcl=res['dino_pcl'],
                    metadata=res['metadata']
                )
                t4 = time.time()

                rospy.loginfo('Timing:\n\tpreproc: {:.6f}s\n\tlocalmap: {:.6f}s'.format(t2-t1, t4-t3))

                #debug viz
                extent = (
                    self.localmap['metadata']['origin'][0].item(),
                    self.localmap['metadata']['origin'][0].item() + self.localmap['metadata']['length_x'].item(),
                    self.localmap['metadata']['origin'][1].item(),
                    self.localmap['metadata']['origin'][1].item() + self.localmap['metadata']['length_y'].item()
                )

                img_extent = (
                    0,
                    res['dino_image'].shape[1],
                    0,
                    res['dino_image'].shape[0],
                )

                for ax in axs:
                    ax.cla()

                vmin = self.localmap['data'][..., :3].view(-1, 3).min(dim=0)[0].view(1,1,3)
                vmax = self.localmap['data'][..., :3].view(-1, 3).max(dim=0)[0].view(1,1,3)
                localmap_viz = (self.localmap['data'][..., :3] - vmin) / (vmax-vmin)

                dino_img_viz = ((res['dino_image'][..., :3]-vmin)/(vmax-vmin)).clip(0., 1.)

                axs[0].imshow(localmap_viz.permute(1,0,2).cpu(), extent=extent, origin='lower')
                axs[0].scatter(
                    self.odom_msg.pose.pose.position.x,
                    self.odom_msg.pose.pose.position.y,
                    marker='x',
                    c='r'
                )

                axs[1].imshow(res['image'], extent=img_extent)
                axs[1].imshow(dino_img_viz.cpu(), extent=img_extent)

                axs[2].imshow(res['image'], extent=img_extent)
                axs[2].scatter(res['pixel_projection'][:, 0].cpu(), dino_img_viz.shape[0]-res['pixel_projection'][:, 1].cpu(), c='r', s=1., alpha=0.5)

                dino_pt_cs = ((res['dino_pcl'][::10, 3:6]-vmin[0])/(vmax[0]-vmin[0])).clip(0., 1.)
                axs[3].scatter(res['dino_pcl'][::10, 0].cpu(), res['dino_pcl'][::10, 1].cpu(), c=dino_pt_cs.cpu(), s=1.)
                axs[3].set_xlim(extent[0], extent[1])
                axs[3].set_xlim(extent[2], extent[3])
                axs[3].set_aspect(1.)

                plt.pause(1e-2)

            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('visual_mapping')

    config_fp = rospy.get_param("~config_fp")
    config = yaml.safe_load(open(config_fp, 'r'))

    visual_mapping_node = DinoMappingNode(config)

    visual_mapping_node.spin()
