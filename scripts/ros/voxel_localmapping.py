import rospy
import yaml
import copy
import numpy as np

np.float = np.float64  # hack for numpify

import ros_numpy
import tf2_ros
import torch
import cv_bridge

from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap

from ros_torch_converter.datatypes.pointcloud import FeaturePointCloudTorch

from physics_atv_visual_mapping.image_processing.image_pipeline import setup_image_pipeline
from physics_atv_visual_mapping.pointcloud_colorization.torch_color_pcl_utils import *
from physics_atv_visual_mapping.terrain_estimation.terrain_estimation_pipeline import setup_terrain_estimation_pipeline
from physics_atv_visual_mapping.localmapping.voxel.voxel_localmapper import VoxelLocalMapper
from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata
from physics_atv_visual_mapping.utils import *
import time

class VoxelMappingNode:
    def __init__(self, config):
        self.device = config["device"]
        self.base_metadata = config["localmapping"]["metadata"]
        self.localmap_ema = config["localmapping"]["ema"]
        self.layer_key = (
            config["localmapping"]["layer_key"]
            if "layer_key" in config["localmapping"].keys()
            else None
        )
        self.layer_keys = (
            self.make_layer_keys(config["localmapping"]["layer_keys"])
            if "layer_keys" in config["localmapping"].keys()
            else None
        )
        self.last_update_time = 0.0

        self.image_pipeline = setup_image_pipeline(config)
        self.setup_localmapper(config)
        self.do_terrain_estimation = "terrain_estimation" in config.keys()
        if self.do_terrain_estimation:
            rospy.loginfo("doing terrain estimation")
            self.terrain_estimator = setup_terrain_estimation_pipeline(config)

        self.vehicle_frame = config['vehicle_frame']
        self.mapping_frame = config['mapping_frame']

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = cv_bridge.CvBridge()

        self.setup_ros_interface(config)

        self.timer = rospy.Timer(rospy.Duration(config['rate']), self.spin)

    def setup_ros_interface(self, config):
        #guarantee images in same order
        self.image_keys = []
        self.image_data = {}
        self.image_pubs = {}
        self.image_subs = {}

        self.pcl_msg = None

        for img_key, img_conf in config['images'].items():
            self.image_data[img_key] = {}
            self.image_keys.append(img_key)

            self.image_data[img_key]['intrinsics'] = get_intrinsics(np.array(img_conf["intrinsics"]["P"]).reshape(3, 3)).float().to(self.device)
            self.image_data[img_key]['extrinsics'] = pose_to_htm(np.concatenate(
                    [
                        np.array(img_conf["extrinsics"]["p"]),
                        np.array(img_conf["extrinsics"]["q"]),
                    ], axis=-1,)).to(self.device)
            self.image_data[img_key]['message'] = None

            self.image_subs[img_key] = rospy.Subscriber(img_conf['image_topic'], Image, self.handle_img, callback_args=img_key)
            self.image_pubs[img_key] = rospy.Publisher("feature_images/{}".format(img_key), Image)

        self.pcl_sub = rospy.Subscriber(config["pointcloud"]["topic"], PointCloud2, self.handle_pointcloud)

        self.pcl_pub = rospy.Publisher("feature_pc", PointCloud2)
        self.voxel_pub = rospy.Publisher("feature_voxels", PointCloud2)

        # self.odom_sub = rospy.Subscriber(config["odometry"]["topic"], Odometry, self.handle_odom, 10)

        # self.timing_pub = self.create_publisher(Float32, "/dino_proc_time", 10)

    def setup_localmapper(self, config):
        """
        check that the localmapper metadata is good
        """
        self.mapper_type = config["localmapping"]["mapper_type"]
        metadata = LocalMapperMetadata(**self.base_metadata)

        assert self.mapper_type == 'voxel', "need mapper type to be either 'voxel'"
        assert metadata.ndims == 3, "need 3d metadata for voxel mapping"
        self.localmapper = VoxelLocalMapper(
            metadata,
            n_features=config["localmapping"]["n_features"],
            ema=config["localmapping"]["ema"],
            device=config["device"],
        )

    def make_layer_keys(self, layer_keys):
        out = []
        for lk in layer_keys:
            for i in range(lk["n"]):
                out.append("{}_{}".format(lk["key"], i))
        return out

    def handle_pointcloud(self, msg):
        # rospy.loginfo('handling pointcloud')
        self.pcl_msg = msg

    # def handle_odom(self, msg):
    #     rospy.loginfo('handling odom')
    #     if self.odom_frame is None:
    #         self.odom_frame = msg.header.frame_id
    #     self.odom_msg = msg

    def handle_img(self, msg, img_key):
        # rospy.loginfo('handling img {}'.format(img_key))
        self.image_data[img_key]['message'] = msg

    def preprocess_inputs(self):
        if self.pcl_msg is None:
            rospy.logwarn("no pcl msg received")
            return None

        pcl_time = self.pcl_msg.header.stamp.to_sec()
        if abs(pcl_time - self.last_update_time) < 1e-3:
            return None

        for img_key, img_data in self.image_data.items():
            if img_data['message'] is None:
                rospy.logwarn("no {} msg received".format(img_key))
                return None

        # need to wait for tf to be available
        try:
            tf_vehicle_to_pcl_msg = self.tf_buffer.lookup_transform(
                self.vehicle_frame,
                self.pcl_msg.header.frame_id,
                self.pcl_msg.header.stamp,
                timeout=rospy.Duration(0.1)
            )
        except (tf2_ros.TransformException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("cant tf from {} to {} at {}".format(
                self.vehicle_frame,
                self.pcl_msg.header.frame_id,
                self.pcl_msg.header.stamp,
            ))
            return None

        try:
            tf_odom_to_pcl_msg = self.tf_buffer.lookup_transform(
                self.mapping_frame,
                self.pcl_msg.header.frame_id,
                self.pcl_msg.header.stamp,
                timeout=rospy.Duration(0.1)
            )
        except (tf2_ros.TransformException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("cant tf from {} to {} at {}".format(
                self.mapping_frame,
                self.pcl_msg.header.frame_id,
                self.pcl_msg.header.stamp,
            ))
            return None

        pcl = pcl_msg_to_xyz(self.pcl_msg).to(self.device)

        vehicle_to_pcl_htm = tf_msg_to_htm(tf_vehicle_to_pcl_msg).to(self.device)

        pcl_in_vehicle = transform_points(pcl.clone(), vehicle_to_pcl_htm)

        odom_to_pcl_htm = tf_msg_to_htm(tf_odom_to_pcl_msg).to(self.device)
        pcl_in_odom = transform_points(pcl.clone(), odom_to_pcl_htm)

        odom_to_vehicle_htm = odom_to_pcl_htm @ torch.linalg.inv(vehicle_to_pcl_htm)

        self.last_update_time = pcl_time

        # Camera processing
        images = []
        image_intrinsics = []
        image_extrinsics = []

        for img_key in self.image_keys:
            img = (
                self.bridge.imgmsg_to_cv2(self.image_data[img_key]['message'], desired_encoding="rgb8").astype(
                    np.float32
                )
                / 255.0
            )
            img = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)

            images.append(img)
            image_intrinsics.append(self.image_data[img_key]['intrinsics'])
            image_extrinsics.append(self.image_data[img_key]['extrinsics'])

        images = torch.cat(images, dim=0)
        image_intrinsics = torch.stack(image_intrinsics, dim=0)
        image_extrinsics = torch.stack(image_extrinsics, dim=0)

        feature_images, feature_intrinsics = self.image_pipeline.run(
            images, image_intrinsics
        )

        feature_images = feature_images.permute(0, 2, 3, 1)
        image_Ps = get_projection_matrix(feature_intrinsics, image_extrinsics)

        coords, valid_mask = get_pixel_projection(pcl_in_vehicle, image_Ps, feature_images)
        pc_features, cnt = colorize(coords, valid_mask, feature_images)
        pc_features = pc_features[cnt > 0]
        feature_pcl = FeaturePointCloudTorch.from_torch(pts=pcl_in_odom, features=pc_features, mask=(cnt > 0))

        pos = odom_to_vehicle_htm[:3, -1]

        rospy.loginfo('colorized {}/{} points'.format(feature_pcl.features.shape[0], feature_pcl.pts.shape[0]))

        return {
            "pos": pos,
            "pcl": pcl_in_odom,
            "images": images,
            "feature_images": feature_images,
            "feature_pc": feature_pcl,
        }

    def make_voxel_msg(self, voxel_grid):
        msg = FeatureVoxelGrid()
        msg.header.stamp = self.pcl_msg.header.stamp
        msg.header.frame_id = self.mapping_frame

        msg.metadata.origin.x = voxel_grid.metadata.origin[0].item()
        msg.metadata.origin.y = voxel_grid.metadata.origin[1].item()
        msg.metadata.origin.z = voxel_grid.metadata.origin[2].item()

        msg.metadata.length.x = voxel_grid.metadata.length[0].item()
        msg.metadata.length.y = voxel_grid.metadata.length[1].item()
        msg.metadata.length.z = voxel_grid.metadata.length[2].item()

        msg.metadata.resolution.x = voxel_grid.metadata.resolution[0].item()
        msg.metadata.resolution.y = voxel_grid.metadata.resolution[1].item()
        msg.metadata.resolution.z = voxel_grid.metadata.resolution[2].item()

        msg.num_voxels = voxel_grid.features.shape[0]
        msg.num_features = voxel_grid.features.shape[1]

        if self.layer_keys is None:
            msg.feature_keys = [
                "{}_{}".format(self.layer_key, i) for i in range(voxel_grid.features.shape[1])
            ]
        else:
            msg.feature_keys = copy.deepcopy(self.layer_keys)

        msg.indices = voxel_grid.indices.tolist()

        feature_msg = Float32MultiArray()
        feature_msg.layout.dim.append(
            MultiArrayDimension(
                label="column_index",
                size=voxel_grid.features.shape[0],
                stride=voxel_grid.features.shape[0],
            )
        )
        feature_msg.layout.dim.append(
            MultiArrayDimension(
                label="row_index",
                size=voxel_grid.features.shape[0],
                stride=voxel_grid.features.shape[0] * voxel_grid.features.shape[1],
            )
        )

        feature_msg.data = voxel_grid.features.flatten().tolist()

        return msg

    def make_gridmap_msg(self, bev_grid):
        """
        convert dino into gridmap msg

        Publish all the feature channels, plus a visualization and elevation layer

        Note that we assume all the requisite stuff is available (pcl, img, odom) as this
        should only be called after a dino map is successfully produced
        """
        gridmap_msg = GridMap()

        gridmap_data = bev_grid.data.cpu().numpy()

        # setup metadata
        gridmap_msg.header.stamp = self.img_msg.header.stamp
        gridmap_msg.header.frame_id = self.odom_frame

        gridmap_msg.layers = bev_grid.feature_keys

        #temp hack
        gridmap_msg.basic_layers = ["min_elevation_filtered_inflated_mask"]
        mask_idx = gridmap_msg.layers.index("min_elevation_filtered_inflated_mask")
        mask = gridmap_data[..., mask_idx] > 0.1
        gridmap_data[..., mask_idx][~mask] = float('nan')

        gridmap_msg.info.resolution = self.localmapper.metadata.resolution[0].item()
        gridmap_msg.info.length_x = self.localmapper.metadata.length[0].item()
        gridmap_msg.info.length_y = self.localmapper.metadata.length[1].item()
        gridmap_msg.info.pose.position.x = (
            self.localmapper.metadata.origin[0].item() + 0.5 * gridmap_msg.info.length_x
        )
        gridmap_msg.info.pose.position.y = (
            self.localmapper.metadata.origin[1].item() + 0.5 * gridmap_msg.info.length_y
        )
        gridmap_msg.info.pose.position.z = self.odom_msg.pose.pose.position.z
        gridmap_msg.info.pose.orientation.w = 1.0
        # transposed_layer_data = np.transpose(gridmap_data, (0, 2,1))
        # flipped_layer_data = np.flip(np.flip(transposed_layer_data, axis=1), axis=2)

        # gridmap_data has the shape (rows, cols, layers)
        # Step 1: Flip the 2D grid layers in both directions (reverse both axes)
        flipped_data = np.flip(gridmap_data, axis=(0, 1))  # Flips along both axes

        # Step 2: Transpose the first two dimensions (x, y) for each layer
        transposed_data = np.transpose(
            flipped_data, axes=(1, 0, 2)
        )  # Transpose rows and cols

        # Step 3: Flatten each 2D layer, maintaining the layers' structure (flattening across x, y)
        flattened_data = transposed_data.reshape(-1, gridmap_data.shape[-1])
        accum_time = 0
        for i in range(gridmap_data.shape[-1]):
            layer_data = gridmap_data[..., i]
            gridmap_layer_msg = Float32MultiArray()
            gridmap_layer_msg.layout.dim.append(
                MultiArrayDimension(
                    label="column_index",
                    size=layer_data.shape[0],
                    stride=layer_data.shape[0],
                )
            )
            gridmap_layer_msg.layout.dim.append(
                MultiArrayDimension(
                    label="row_index",
                    size=layer_data.shape[0],
                    stride=layer_data.shape[0] * layer_data.shape[1],
                )
            )

            # gridmap reverses the rasterization
            start_time = time.time()
            gridmap_layer_msg.data = flattened_data[:, i].tolist()
            end_time = time.time()
            accum_time += end_time - start_time

            # gridmap_layer_msg.data = flipped_layer_data[i].flatten().tolist()
            gridmap_msg.data.append(gridmap_layer_msg)
        rospy.loginfo("time to flatten layer {}: {}".format(i, accum_time))
        # add dummy elevation
        gridmap_msg.layers.append("elevation")
        layer_data = (
            np.zeros_like(gridmap_data[..., 0])
            + self.odom_msg.pose.pose.position.z
            - 1.73
        )
        gridmap_layer_msg = Float32MultiArray()
        gridmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="column_index",
                size=layer_data.shape[0],
                stride=layer_data.shape[0],
            )
        )
        gridmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="row_index",
                size=layer_data.shape[0],
                stride=layer_data.shape[0] * layer_data.shape[1],
            )
        )

        gridmap_layer_msg.data = layer_data.flatten().tolist()
        gridmap_msg.data.append(gridmap_layer_msg)

        return gridmap_msg

    def make_voxel_viz_msg(self, voxel_grid):
        pts = voxel_grid.grid_indices_to_pts(voxel_grid.raster_indices_to_grid_indices(voxel_grid.raster_indices))
        feats = voxel_grid.features
        mask = voxel_grid.feature_mask

        fpc = FeaturePointCloudTorch.from_torch(pts=pts, features=feats, mask=mask)

        return self.make_pcl_msg(fpc)

    def make_pcl_msg(self, pcl, vmin=None, vmax=None):
        """
        Convert dino pcl into message
        """
        start_time = time.time()
        pcl_pos = pcl.pts[pcl.feat_mask].cpu().numpy()

        pcl_cs = pcl.features[:, :3]
        if vmin is None or vmax is None:
            vmin = pcl_cs.min(dim=0)[0].view(1, 3)
            vmax = pcl_cs.max(dim=0)[0].view(1, 3)
        else:
            vmin = vmin.view(1, 3)
            vmax = vmax.view(1, 3)
        pcl_cs = ((pcl_cs - vmin) / (vmax - vmin)).cpu().numpy()

        after_init_time = time.time()

        points = pcl_pos
        rgb_values = (pcl_cs * 255.0).astype(np.uint8)

        msg = self.xyz_array_to_point_cloud_msg(
            points=pcl_pos,
            frame=self.mapping_frame,
            timestamp=self.pcl_msg.header.stamp,
            rgb_values=rgb_values
        )

        return msg

    def xyz_array_to_point_cloud_msg(
        self, points, frame, timestamp=None, rgb_values=None
    ):
        """
        Modified from: https://github.com/castacks/physics_atv_deep_stereo_vo/blob/main/src/stereo_node_multisense.py
        Please refer to this ros answer about the usage of point cloud message:
            https://answers.ros.org/question/234455/pointcloud2-and-pointfield/
        :param points:
        :param header:
        :return:
        """
        # Create the message header
        header = Header()
        header.frame_id = frame
        header.stamp = timestamp

        msg = PointCloud2()
        msg.header = header

        # Determine point cloud dimensions (organized vs unorganized)
        if len(points.shape) == 3:  # Organized point cloud
            msg.width = points.shape[0]
            msg.height = points.shape[1]
        else:  # Unorganized point cloud
            msg.width = points.shape[0]
            msg.height = 1

        msg.is_bigendian = False
        msg.is_dense = False  # Set to False since organized clouds are non-dense

        if rgb_values is None:
            # XYZ only
            msg.fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            msg.point_step = 12  # 3 fields (x, y, z) each 4 bytes (float32)
            msg.row_step = msg.point_step * msg.width
            xyz = points.astype(np.float32)
            msg.data = xyz.tobytes()  # Convert to bytes
        else:
            # XYZ and RGB
            msg.fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
            ]
            msg.point_step = (
                16  # 4 fields (x, y, z, rgb) with rgb as a packed 32-bit integer
            )
            msg.row_step = msg.point_step * msg.width

            # Prepare the data array with XYZ and RGB
            xyzcolor = np.zeros(
                (points.shape[0],),
                dtype={
                    "names": ("x", "y", "z", "rgba"),
                    "formats": ("f4", "f4", "f4", "u4"),
                },
            )

            # Assign XYZ values
            xyzcolor["x"] = points[:, 0]
            xyzcolor["y"] = points[:, 1]
            xyzcolor["z"] = points[:, 2]

            # Prepare RGB values (packed into a 32-bit unsigned int)
            rgb_uint32 = np.zeros(points.shape[0], dtype=np.uint32)
            rgb_uint32 = (
                np.left_shift(rgb_values[:, 0].astype(np.uint32), 16)
                | np.left_shift(rgb_values[:, 1].astype(np.uint32), 8)
                | rgb_values[:, 2].astype(np.uint32)
            )
            xyzcolor["rgba"] = rgb_uint32

            msg.data = xyzcolor.tobytes()  # Convert to bytes

        return msg
    
    def make_img_msg(self, dino_img, img_key, vmin=None, vmax=None):
        if vmin is None or vmax is None:
            viz_img = normalize_dino(dino_img[..., :3])
        else:
            vmin = vmin.view(1, 1, 3)
            vmax = vmax.view(1, 1, 3)
            viz_img = (dino_img[..., :3] - vmin) / (vmax-vmin)
            viz_img = viz_img.clip(0., 1.)

        viz_img = viz_img.cpu().numpy() * 255
        img_msg = self.bridge.cv2_to_imgmsg(viz_img.astype(np.uint8), "rgb8")
        img_msg.header.stamp = self.image_data[img_key]['message'].header.stamp
        return img_msg

    def publish_messages(self, res):
        """
        Publish the dino pcl and dino map
        """
        pts = self.localmapper.voxel_grid.grid_indices_to_pts(
            self.localmapper.voxel_grid.raster_indices_to_grid_indices(
                self.localmapper.voxel_grid.indices
            )
        )
        colors = self.localmapper.voxel_grid.features[:, :3]
        vmin = colors.min(dim=0)[0]
        vmax = colors.max(dim=0)[0]

        all_idxs = torch.cat([self.localmapper.voxel_grid.indices, self.localmapper.voxel_grid.all_indices])
        unique, cnts = torch.unique(all_idxs, return_counts=True)
        non_colorized_idxs = unique[cnts==1]

        non_colorized_pts = self.localmapper.voxel_grid.grid_indices_to_pts(
            self.localmapper.voxel_grid.raster_indices_to_grid_indices(non_colorized_idxs)
        )

        color_placeholder = 0.1 * torch.ones(non_colorized_pts.shape[0], 3, device=non_colorized_pts.device)

        pts = torch.cat([pts, non_colorized_pts], dim=0)
        colors = torch.cat([colors, color_placeholder], dim=0)

        voxel_viz_msg = self.make_pcl_msg(
            torch.cat([pts, colors], axis=-1), vmin=vmin, vmax=vmax
        )
        self.voxel_viz_pub.publish(voxel_viz_msg)

        # voxel_msg = self.make_voxel_msg(self.localmapper.voxel_grid)
        # self.voxel_pub.publish(voxel_msg)

        pcl_msg = self.make_pcl_msg(res["dino_pcl"], vmin=vmin, vmax=vmax)
        self.pcl_pub.publish(pcl_msg)

        img_msg = self.make_img_msg(res["dino_image"], vmin=vmin, vmax=vmax)
        self.image_pub.publish(img_msg)

        gridmap_msg = self.make_gridmap_msg(self.bev_grid)
        self.gridmap_pub.publish(gridmap_msg)

        timing_msg = Float32()
        self.timing_pub.publish(timing_msg)

    def spin(self, event):
        rospy.loginfo("spinning...")

        preproc_start_time = time.time()
        res = self.preprocess_inputs()
        preproc_end_time = time.time()

        if res:
            rospy.loginfo("updating localmap...")

            update_start_time = time.time()
            self.localmapper.update_pose(res["pos"])
            self.localmapper.add_feature_pc(
                pos=res['pos'], feat_pc=res["feature_pc"], do_raytrace=True
            )

            if self.do_terrain_estimation:
                self.bev_grid = self.terrain_estimator.run(self.localmapper.voxel_grid)

            update_end_time = time.time()

            pub_start_time = time.time()
            msg = self.make_pcl_msg(res['feature_pc'])
            self.pcl_pub.publish(msg)

            msg = self.make_voxel_viz_msg(self.localmapper.voxel_grid)
            self.voxel_pub.publish(msg)

            for i, img_key in enumerate(self.image_keys):
                feat_img = res["feature_images"][i]
                pub = self.image_pubs[img_key]
                msg = self.make_img_msg(feat_img, img_key)
                pub.publish(msg)

            pub_end_time = time.time()

            rospy.loginfo(
                "preprocess time: {}".format(preproc_end_time - preproc_start_time)
            )
            rospy.loginfo(
                "update time: {}".format(update_end_time - update_start_time)
            )
            rospy.loginfo(
                "publish time: {}".format(pub_end_time - pub_start_time)
            )
            rospy.loginfo("total time: {}".format(time.time() - preproc_start_time))

if __name__ == '__main__':
    rospy.init_node('visual_mapping')

    # config_fp = rospy.get_param("~config_fp")

    config_fp = '/catkin_ws/src/vfm_voxel_mapping/visual_mapping_noetic/config/ros/rzr_voxel_mapping.yaml'
    config = yaml.safe_load(open(config_fp, 'r'))

    visual_mapping_node = VoxelMappingNode(config)
    rospy.spin()