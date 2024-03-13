import yaml
import rospy
import numpy as np
np.float = np.float64 #hack for numpify

import ros_numpy
import tf2_ros
import torch
import cv_bridge
import rospkg
import os

import distinctipy as COLORS

from sensor_msgs.msg import PointCloud2, CameraInfo, Image, CompressedImage
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Float32

from physics_atv_visual_mapping.image_processing.anyloc_utils import DinoV2ExtractFeatures, VLAD
from physics_atv_visual_mapping.pointcloud_colorization.torch_color_pcl_utils import *
from physics_atv_visual_mapping.localmapping.localmapping import *
from physics_atv_visual_mapping.utils import *

class DinoColorizationNode:
    """
    Hacky implementation of visual mapping node for debug
    """
    def __init__(self, config):
        self.localmap = None
        self.pcl_msg = None
        self.img_msg = None
        self.odom_frame = None
        self.device = config['device']
        self.last_update_time = 0.

        if 'pca' in config:
            self.mode = 'pca'
        else:
            self.mode = 'vlad'

        if self.mode == 'pca':
            desc_facet: Literal["query", "key", "value", "token"] = "token"
            self.dino_pca = torch.load(config['pca']['fp'], map_location=config['device'])

        elif self.mode == 'vlad':
            desc_facet: Literal["query", "key", "value", "token"] = "value"
            vlad_config = config['vlad']
            vlad = VLAD(vlad_config['n_clusters'], desc_dim=None,cache_dir=vlad_config['cache_dir'])
            vlad.fit(None)
            self.vlad = vlad
            self.Csub = np.array(COLORS.get_colors(32, rng=1))
            self.Csub = np.concatenate([self.Csub,np.zeros([1,3])],axis=0)
        else:
            print("NO MODE SET")

        rp = rospkg.RosPack()
        dino_dir = os.path.join(rp.get_path("physics_atv_visual_mapping"), "models/hub")

        self.dino = DinoV2ExtractFeatures(dino_dir,
            dino_model=config['dino']['dino_type'],
            layer=config['dino']['dino_layer'],
            input_size=config['dino']['image_insize'],
            facet=desc_facet,
            device=config['device']
        )

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

        self.pcl_pub = rospy.Publisher('/dino_pcl', PointCloud2, queue_size=1)
        self.image_pub = rospy.Publisher('/dino_image', Image, queue_size=1)

        self.rate = rospy.Rate(10)
        self.viz = config['viz']

    def handle_pointcloud(self, msg):
        #temp hack
        self.pcl_msg = msg
        self.pcl_msg.header.frame_id = 'vehicle'

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

        if self.img_msg is None:
            rospy.logwarn_throttle(1.0, 'no img msg received')
            return None

        self.last_update_time = self.pcl_msg.header.stamp.to_sec()

        #camera stuff
        if self.compressed_img:
            img = self.bridge.compressed_imgmsg_to_cv2(self.img_msg, desired_encoding='rgb8')/255.
        else:
            img = self.bridge.imgmsg_to_cv2(self.img_msg, desired_encoding='rgb8').astype(np.float32)/255.

        #run the image through dino
        dino_img = self.dino(img)[0]

        ###
        if self.mode == 'pca':
            dino_img_norm = dino_img.view(-1, dino_img.shape[-1]) - self.dino_pca['mean'].view(1,-1)
            dino_pca = dino_img_norm.unsqueeze(1) @ self.dino_pca['V'].unsqueeze(0)
            dino_img = dino_pca.view(dino_img.shape[0], dino_img.shape[1], -1)
        elif self.mode == 'vlad':
            res = self.vlad.generate_res_vec(dino_img.view(-1, dino_img.shape[-1]))
            dino_img = res.abs().sum(dim=2).view(self.dino.output_size[1],self.dino.output_size[0],-1)
            # print(res_sum.shape)
            # da = res.abs().sum(dim=2).argmin(dim=1).view(self.dino.output_size[1],self.dino.output_size[0])
            # dino_img = self.Csub[da.long()].float()

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

        pcl = pcl_msg_to_xyz(self.pcl_msg).to(self.device)

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
        dino_pcl = torch.cat([pcl[ind_in_frame][:, :3], dino_features], dim=-1)

        return {
            'pcl': pcl,
            'image': img,
            'dino_image': dino_img,
            'dino_pcl': dino_pcl,
            'pixel_projection': pixel_coordinates[ind_in_frame]
        }

    def make_pcl_msg(self, pcl):
        """
        Convert dino pcl into message
        """

        pcl_pos = pcl[:, :3].cpu().numpy()

        pcl_cs = pcl[:, [5,4,3]]
        vmin = pcl_cs.min(dim=0)[0].view(1,3)
        vmax = pcl_cs.max(dim=0)[0].view(1,3)
        pcl_cs =((pcl_cs - vmin) / (vmax - vmin)).cpu().numpy()

        msg = self.xyz_array_to_point_cloud_msg(
            points=pcl.cpu().numpy(),
            frame=self.pcl_msg.header.frame_id,
            timestamp=self.pcl_msg.header.stamp,
            rgb_values=(pcl_cs*255.).astype(np.uint8)
        )

        return msg

    def xyz_array_to_point_cloud_msg(self, points, frame, rgb_values, timestamp=None):
        """
        Modified from: https://github.com/castacks/physics_atv_deep_stereo_vo/blob/main/src/stereo_node_multisense.py
        Please refer to this ros answer about the usage of point cloud message:
            https://answers.ros.org/question/234455/pointcloud2-and-pointfield/
        :param points:
        :param header:
        :return:
        """
        header = Header()
        header.frame_id = frame
        if timestamp is None:
            timestamp = rospy.Time().now()
        header.stamp = timestamp
        msg = PointCloud2()
        msg.header = header
        if len(points.shape)==3:
            msg.width = points.shape[0]
            msg.height = points.shape[1]
        else:
            msg.width = points.shape[0]
            msg.height = 1
        msg.is_bigendian = False
        # organized clouds are non-dense, since we have to use std::numeric_limits<float>::quiet_NaN()
        # to fill all x/y/z value; un-organized clouds are dense, since we can filter out unfavored ones
        msg.is_dense = False

        msg.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                      PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1), 
                      PointField('rgb', 12, PointField.UINT32, 1), ]

        dino_n = points.shape[1] - 3

        for i in range(dino_n):
            msg.fields.append(
                PointField('dino_{}'.format(i), 4*i+16, PointField.FLOAT32, 1)
            )

        msg.point_step = 4*(points.shape[1]+1)

        msg.row_step = msg.point_step * msg.width

        arrnames = tuple(['x', 'y', 'z', 'rgba'] + ['dino_{}'.format(i) for i in range(dino_n)])
        arrformats = tuple(['f4', 'f4', 'f4', 'u4'] + ['f4' for i in range(dino_n)])

        xyzcolor = np.zeros( (points.shape[0], 1), \
        dtype={
            "names": arrnames,
            "formats": arrformats} )

        xyzcolor["x"] = points[:, 0].reshape((-1, 1))
        xyzcolor["y"] = points[:, 1].reshape((-1, 1))
        xyzcolor["z"] = points[:, 2].reshape((-1, 1))
        color_rgba = np.zeros((points.shape[0], 4), dtype=np.uint8) + 255
        color_rgba[:,:3] = rgb_values[:,:3]
        xyzcolor["rgba"] = color_rgba.view('uint32')

        for i in range(dino_n):
            xyzcolor['dino_{}'.format(i)] = points[:, 3+i].reshape((-1, 1))

        msg.data = xyzcolor.tostring()

        return msg

    def publish_messages(self, res):
        """
        Publish the dino pcl and dino map
        """
        pcl_msg = self.make_pcl_msg(res['dino_pcl'])
        self.pcl_pub.publish(pcl_msg)

        if self.mode == 'pca':
            vmin = res['dino_image'][..., :3].view(-1, 3).min(dim=0)[0].view(1,1,3)
            vmax = res['dino_image'][..., :3].view(-1, 3).max(dim=0)[0].view(1,1,3)
            viz_img = ((res['dino_image'][..., :3]-vmin)/(vmax-vmin)).clip(0., 1.).cpu().numpy() * 255
        elif self.mode == 'vlad':
            da = np.argmin(res['dino_image'].cpu().numpy(),axis=-1)
            viz_img = (self.Csub[da]*255).astype(np.int32)

        # img_msg = self.bridge.cv2_to_imgmsg(vimg, "bgr8")
        img_msg = self.bridge.cv2_to_imgmsg(viz_img.astype(np.uint8), "rgb8")
        img_msg.header.stamp = pcl_msg.header.stamp
        self.image_pub.publish(img_msg)

    def spin(self):
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['figure.raise_window'] = False
        import time

        if self.viz:
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs = axs.flatten()
            plt.show(block=False)

        while not rospy.is_shutdown():
            rospy.loginfo('spinning...')

            t1 = time.time()
            res = self.preprocess_inputs()
            t2 = time.time()

            if res:
                self.publish_messages(res)

                t3 = time.time()

                rospy.loginfo('Timing:\n\tpreproc: {:.6f}s\n\tserialize: {:.6f}s'.format(t2-t1, t3-t2))

                if self.viz:
                    img_extent = (
                        0,
                        res['dino_image'].shape[1],
                        0,
                        res['dino_image'].shape[0],
                    )

                    for ax in axs:
                        ax.cla()

                    vmin = res['dino_image'][..., :3].view(-1, 3).min(dim=0)[0].view(1,1,3)
                    vmax = res['dino_image'][..., :3].view(-1, 3).max(dim=0)[0].view(1,1,3)

                    dino_img_viz = ((res['dino_image'][..., :3]-vmin)/(vmax-vmin)).clip(0., 1.)

                    axs[0].imshow(res['image'], extent=img_extent)
                    axs[0].imshow(dino_img_viz.cpu(), extent=img_extent, alpha=0.5)
                    axs[0].set_title('Dino + FPV')

                    axs[1].imshow(res['image'], extent=img_extent)
                    axs[1].scatter(res['pixel_projection'][:, 0].cpu(), dino_img_viz.shape[0]-res['pixel_projection'][:, 1].cpu(), c='r', s=1., alpha=0.02)
                    axs[1].set_title('pcl projection')

                    dino_pt_cs = ((res['dino_pcl'][::10, 3:6]-vmin[0])/(vmax[0]-vmin[0])).clip(0., 1.)
                    axs[2].scatter(res['dino_pcl'][::10, 0].cpu(), res['dino_pcl'][::10, 1].cpu(), c=dino_pt_cs.cpu(), s=1.)
                    axs[2].set_xlim(-50., 50.)
                    axs[2].set_ylim(-50., 50.)
                    axs[2].set_aspect(1.)
                    axs[2].set_title('dino PCL')

                    plt.pause(1e-2)

            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('visual_mapping')

    config_fp = rospy.get_param("~config_fp")
    config = yaml.safe_load(open(config_fp, 'r'))

    visual_mapping_node = DinoColorizationNode(config)

    visual_mapping_node.spin()
