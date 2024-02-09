#!/usr/bin/python3
import numpy as np

import rospy
# import matplotlib.pyplot as plt
np.float = np.float64

import ros_numpy
import cv2
from cv_bridge import CvBridge
from functools import reduce
import time
import std_msgs.msg
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo, CompressedImage
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
import message_filters
import tf2_ros
from tf.transformations import quaternion_matrix, translation_matrix

# from pointcloud_colorization.color_pcl_utils import *
from pointcloud_colorization.torch_color_pcl_utils import *
import torch
import time

from physics_atv_visual_mapping.image_processing.anyloc_utils import DinoV2ExtractFeatures, VLAD

import distinctipy as COLORS

class ColorizePclNode:
    '''Class to create colorized pointclouds from a 3D pointcloud and an image with known intrinsics and extrinsics.

    Definitions:
            1. Pixel coordinates are x,y coordinates of a point in an image, where the origin of the pixel coordinate system is at the top left of the image, with (+)x going to the right, and (+)y going down.
            2. Image coordinates are x, y coordinates of a point in an image, where the origin of the image coordinate system is at the center of the frame, with (+)x pointing to the left, and (+)y pointing up.
            3. Camera coordinates are 3D x, y, z coordinates of a point in 3D space with respect to the camera, with the origin of the camera coordinate system at the focal point of the camera, with (+)x pointing left, (+)y pointing up, and (+)z pointing forward.
            4. Source (e.g. world, or lidar) coordinates are 3D x, y, z coordinates of a point in 3D space with respect to a source frame (which could be world coordinates or body-centric coordinates). Here, (+)x points forward, (+)y points to the left, and (+)z points up
    '''
    def __init__(self, tf_in_optical=True):
        self.tf_in_optical = tf_in_optical

        self.got_pcl = False
        self.got_rgb = False
        self.got_odom = False

        self.lidar_points = None
        self.image = None
        self.odom = None
        self.pose = None

        self.device = 'cuda'

        #replace with params file
        config = {'DINO':{'desc_layer': 10, 'model': "dinov2_vitb14", 'downsample_factor' : 1.5,
                            'VLAD': {'n_clusters': 8, 'cache_dir' : '/home/physics_atv/physics_atv_ws/src/perception/physics_atv_visual_mapping/data/dino_clusters/8_clusters_532'}}} #hard coding path for now sorry

        if 'DINO' in config:
            self.dino = True

        if self.dino:
            desc_layer: int = config['DINO']['desc_layer']
            desc_facet: Literal["query", "key", "value", "token"] = "value"
            encoder = DinoV2ExtractFeatures(config['DINO']['model'], desc_layer,
                desc_facet, device="cuda")
            self.dino_encoder = encoder
            self.down_sample_factor = config['DINO']['downsample_factor']
            self.dino_input_size = (int(14*(546//(14*self.down_sample_factor))),int(14*(1036//(14*self.down_sample_factor))))
            # self.dino_input_size = (350,644)

            self.dino_out_size = (int(self.dino_input_size[0]/14),int(self.dino_input_size[1]/14))
            # self.dino_transform = transforms.Compose([transforms.Resize(self.dino_input_size)])

            if 'VLAD' in config['DINO']:
                vlad_config = config['DINO']['VLAD']
                vlad = VLAD(vlad_config['n_clusters'], desc_dim=None,cache_dir=vlad_config['cache_dir'])
                vlad.fit(None)
                self.vlad = vlad
                Csub = np.array(COLORS.get_colors(32))
                self.Csub = torch.from_numpy(Csub).to(self.device)
            elif 'PCA' in config['DINO']:
                print("PCA NOT INTEGRATED YET :O")

            else:
                ...

        ## Set up subscribers
        input_lidar_topic = rospy.get_param('~input_lidar_topic', 'wanda/lidar_points')
        input_img_topic = rospy.get_param('~input_img_topic', '/wanda/stereo_right/image_rect_color/compressed')
        input_pose_topic = rospy.get_param('~input_pose_topic', 'wanda/odom')

        input_camera_info_topic = rospy.get_param('~input_camera_info_topic', 'wanda/stereo_right/camera_info')

        camera_optical_frame = rospy.get_param('~camera_frame', '/wanda/stereo_right_optical_frame')
        lidar_frame = rospy.get_param('~lidar_frame', '/wanda/ouster_link')
        self.image_compressed = rospy.get_param('~image_compressed', True)

        # Get camera intrinsics and extrinsics directly from camera info and tf2
        self.intrinsics = torch.tensor(rospy.wait_for_message(input_camera_info_topic, CameraInfo).K).reshape(3,3).float()

        if self.dino:
            #TODO double check this
            # self.dino_intrinsics = self.rgb_intrinsics // 14
            self.dino_intrinsics = self.intrinsics / (14.*self.down_sample_factor)
            self.dino_intrinsics[-1,-1] = 1

        self.tfBuffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(self.tfBuffer)
        self.lidar_to_camera_transform = None

        rate_trans = rospy.Rate(5)
        while self.lidar_to_camera_transform is None:
            try:
                self.lidar_to_camera_transform = self.tfBuffer.lookup_transform(camera_optical_frame, lidar_frame, rospy.Time(0))
#            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            except:
                print("Can't find transform from camera {} to lidar {}".format(camera_optical_frame, lidar_frame))
                rate_trans.sleep()

        print("Found transform!")

        rotation = self.lidar_to_camera_transform.transform.rotation
        translation = self.lidar_to_camera_transform.transform.translation
        rot_quat = np.array([rotation.x, rotation.y, rotation.z, rotation.w])
        translation_vec = np.array([translation.x, translation.y, translation.z])

        rot_mat = quaternion_matrix(rot_quat)
        rot_mat[0:3, -1] = translation_vec

        self.extrinsics = torch.from_numpy(rot_mat).float()

        rospy.loginfo('PCL topic: {}'.format(input_lidar_topic))
        rospy.loginfo('Img topic: {} (compressed={})'.format(input_img_topic, self.image_compressed))
        rospy.loginfo('Pose topic: {}'.format(input_pose_topic))

        lidar_sub = message_filters.Subscriber(input_lidar_topic, PointCloud2, queue_size=100)
        pose_sub = message_filters.Subscriber(input_pose_topic, Odometry, queue_size=100)

        if self.image_compressed:
            image_sub = message_filters.Subscriber(input_img_topic, CompressedImage, queue_size=100)
        else:
            image_sub = message_filters.Subscriber(input_img_topic, Image, queue_size=100)

        time_synchronizer = message_filters.ApproximateTimeSynchronizer([lidar_sub, image_sub, pose_sub], 100, 0.5, allow_headerless=False)
        time_synchronizer.registerCallback(self.lidar_img_odom_sync_cb)

        ## Set up publishers
#        self.orig_pcl_pub = rospy.Publisher('~orig_pcl', PointCloud2, queue_size=1)
        self.color_pcl_pub = rospy.Publisher('~colorized_pcl', PointCloud2, queue_size=1)

        ## Also store lidar frame id
        self.lidar_frame_id = None

        ## CV bridge
        self.bridge = CvBridge()

        # ## Load lidar points and image. If debugging, uncomment lines below to load files. Once done debugging, subscribe to the relevant topics
        # self.image = np.load("/home/mateo/SARA/src/sara_ws/src/longrange_perception/data/lidar_img_sara.npy")
        # self.lidar_points = np.load("/home/mateo/SARA/src/sara_ws/src/longrange_perception/data/lidar_points_sara.npy")

    def lidar_img_odom_sync_cb(self, lidar_msg, img_msg, odom_msg):
        '''Callback function for lidar, image, and odom subscribers called directly by TimeSynchronizer'''
        # print('here')
        point_cloud = ros_numpy.numpify(lidar_msg)
        pc_x = point_cloud['x'].flatten()
        pc_y = point_cloud['y'].flatten()
        pc_z = point_cloud['z'].flatten()

        points=np.ones((pc_x.shape[0],3))
        points[:,0]=pc_x
        points[:,1]=pc_y
        points[:,2]=pc_z
        self.lidar_points = points
        self.lidar_frame_id = lidar_msg.header.frame_id
        self.got_pcl = True

        if self.image_compressed:
            compressed_image = np.frombuffer(img_msg.data, dtype=np.uint8).copy()
            self.image = cv2.imdecode(compressed_image, cv2.IMREAD_UNCHANGED)
        else:
            self.image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        # self.image = torch.from_numpy(self.image)

        self.lidar_stamp = img_msg.header.stamp #! HACK FOR NOW to get it passed through local mapping's time sync
        self.got_rgb = True

        self.pose = odom_msg.pose.pose
        self.odom = odom_msg

        # print("Got new callback: ", time.time()-self.last_sync_cb_time)
        # self.last_sync_cb_time = time.time()

    def publish_colorized_pcl(self):
        '''Creates and publishes the colorized pointcloud.'''

        #print("Got PCL?: {}".format(self.got_pcl))
        #print("Got RGB?: {}".format(self.got_rgb))

        if self.got_pcl and self.got_rgb:
            now = time.perf_counter()

            if self.dino:
                image = cv2.resize(self.image,(self.dino_input_size[1],self.dino_input_size[0]))
                image = torch.from_numpy(image).cuda()
                dino_in = image.float()/255.0
                dino_in = dino_in.permute(2,0,1).unsqueeze(0).cuda()
                # dino_image = self.dino_encoder(dino_in)[0].reshape(self.dino_out_size[0],self.dino_out_size[1],-1)
                feat_vec = self.dino_encoder(dino_in)[0]
                # print(feat_vec.device)
                res = self.vlad.generate_res_vec(feat_vec)

                da = res.abs().sum(dim=2).argmin(dim=1).reshape(self.dino_out_size[0], self.dino_out_size[1]).to(self.device)
                # print('u2 ', time.perf_counter() - unc_now)
                # da = F.interpolate(da[None, None, ...].to(float),
                # (img.shape[0],img.shape[1]), mode='nearest')[0, 0].to(da.dtype)
                # print('u3 ', time.perf_counter() - unc_now)

                image = self.Csub[da.long()]
                # print(self.Csub.dtype)

                intrinsics = self.dino_intrinsics
            else:
                intrinsics = self.intrinsics

            ## Get rid of invalid lidar points
            lidar_points = torch.from_numpy(self.lidar_points).float().to(self.device)
            self.valid_points = remove_invalid(lidar_points)

            ## Obtain projection matrix from lidar coordinates to pixel coordinates
            I = get_intrinsics(intrinsics, self.tf_in_optical).to(self.device)
            E = get_extrinsics(self.extrinsics, self.tf_in_optical).to(self.device)
            self.P = obtain_projection_matrix(I, E)

            ## For each 3D point in lidar space, obtain location in pixel space
            self.pixel_coordinates = get_pixel_from_3D_source(self.valid_points, self.P)

            ## Remove pixel points/lidar points outside of image frame
            self.lidar_points_in_frame, self.pixels_in_frame, self.ind_in_frame = get_points_and_pixels_in_frame(self.valid_points, self.pixel_coordinates, image.shape[0], image.shape[1])

            rgb_values_in_frame = get_rgb_from_pixel_coords(image, self.pixels_in_frame).cpu().numpy() * 255
            # self.rgb_values /= 255

            # self.rgb_values = torch.zeros(self.valid_points.shape,dtype = torch.uint8)
            # print(self.rgb_values.dtype, rgb_values_in_frame.dtype)
            # self.rgb_values[self.ind_in_frame] = rgb_values_in_frame

            ## For remaining lidar points, add RGB pixel info to lidar point
            # self.colorized_point_cloud_msg = create_point_cloud(self.valid_points, parent_frame=self.lidar_frame_id, colors=self.rgb_values)
            self.colorized_point_cloud_msg = create_point_cloud(self.lidar_points_in_frame.cpu().numpy(), parent_frame=self.lidar_frame_id, colors=rgb_values_in_frame)

            self.colorized_point_cloud_msg.header.stamp = self.lidar_stamp

            self.color_pcl_pub.publish(self.colorized_point_cloud_msg)
#            self.orig_pcl_pub.publish(self.original_point_cloud_msg)
            print(time.perf_counter() - now)

if __name__=="__main__":


    rospy.init_node("ColorizePclNode", log_level=rospy.INFO)
    cpn = ColorizePclNode()
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        #print("Publishing pointcloud")
        cpn.publish_colorized_pcl()
        rate.sleep()
