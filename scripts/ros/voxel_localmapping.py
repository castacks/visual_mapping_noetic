import rospy
import yaml
import copy
import time
import numpy as np

np.float = np.float64  # hack for numpify

import ros_numpy
import tf2_ros
import torch
import cv_bridge
import message_filters

from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap

from ros_torch_converter.datatypes.pointcloud import FeaturePointCloudTorch

from physics_atv_visual_mapping.image_processing.image_pipeline import setup_image_pipeline
from physics_atv_visual_mapping.pointcloud_colorization.torch_color_pcl_utils import *
from physics_atv_visual_mapping.terrain_estimation.terrain_estimation_pipeline import setup_terrain_estimation_pipeline
from physics_atv_visual_mapping.localmapping.voxel.voxel_localmapper import VoxelLocalMapper, VoxelGrid
from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata
from physics_atv_visual_mapping.image_processing.processing_blocks.traversability_prototypes import TraversabilityPrototypesBlock
from physics_atv_visual_mapping.utils import *

class VoxelMappingNode:
    def __init__(self, config):
        self.device = config["device"]
        self.base_metadata = config["localmapping"]["metadata"]
        self.localmap_ema = config["localmapping"]["ema"]
        self.last_update_time = 0.0

        self.image_pipeline = setup_image_pipeline(config)
        self.setup_localmapper(config)
        self.do_terrain_estimation = "terrain_estimation" in config.keys()

        if self.do_terrain_estimation:
            rospy.loginfo("doing terrain estimation")
            self.terrain_estimator = setup_terrain_estimation_pipeline(config)

        self.use_masks = any(["mask" in img_conf.keys() for img_conf in config["images"].values()])

        self.do_proj_cleanup = config['do_proj_cleanup']

        self.vehicle_frame = config['vehicle_frame']
        self.mapping_frame = config['mapping_frame']

        self.gridmap_pub_keys = config['gridmap_pub_keys'] if 'gridmap_pub_keys' in config.keys() else None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = cv_bridge.CvBridge()

        self.setup_ros_interface(config)

        self.timer = rospy.Timer(rospy.Duration(config['rate']), self.spin)

    def setup_ros_interface(self, config):
        #guarantee images in same order
        """
        As far as I know, we can't do an ApproximateTimeSynchronizer with
        variable number of topics, so we have to hard-code the three camera keys
        """
        self.image_keys = ['image_left', 'image_front', 'image_right']
        assert all([ik in config['images'].keys() for ik in self.image_keys]), "unfortunately, we require all of {} in the config".format(self.image_keys)

        self.image_data = {}
        self.image_pubs = {}
        self.image_subs = {}
        self.image_masks = []

        self.pcl_msg = None

        for img_key in self.image_keys:
            img_conf = config['images'][img_key]
            self.image_data[img_key] = {}

            intrinsics_msg = rospy.wait_for_message(img_conf['camera_info_topic'], CameraInfo)
            intriniscs_arr = np.array(intrinsics_msg.P).reshape(3, 4)[:, :3]
            self.image_data[img_key]['intrinsics'] = get_intrinsics(intriniscs_arr).float().to(self.device)
            
            self.image_data[img_key]['message'] = None

            self.image_subs[img_key] = message_filters.Subscriber(img_conf['image_topic'], Image)
            self.image_pubs[img_key] = rospy.Publisher("feature_images/{}".format(img_key), Image, queue_size=10)

            if self.use_masks:
                rospy.loginfo("looking for {} mask at {}".format(img_key, img_conf["mask"]))
                self.image_masks.append(self.get_mask(img_conf["mask"]))
        
        self.image_masks = torch.stack(self.image_masks, dim=0).unsqueeze(-1) #[BxHxWx1]

        self.pcl_sub = message_filters.Subscriber(config["pointcloud"]["topic"], PointCloud2)

        self.pcl_pub = rospy.Publisher("feature_pc", PointCloud2, queue_size=10)
        self.voxel_pub = rospy.Publisher("feature_voxels", PointCloud2, queue_size=10)

        self.gridmap_pub = rospy.Publisher("gridmap", GridMap, queue_size=10)

        subs = [self.pcl_sub] + [self.image_subs[k] for k in self.image_keys]

        self.time_sync = message_filters.ApproximateTimeSynchronizer(subs, 10, slop=0.01)
        self.time_sync.registerCallback(self.handle_data)

    def get_mask(self, mask_fp, size=(960, 594)):
        """
        Get mask from file. Note that since we batch-process, we need to resize masks
        """
        img_npy = cv2.imread(mask_fp)
        img_npy_resize = cv2.resize(img_npy, size, interpolation=cv2.INTER_NEAREST) #best not to change values
        img_torch = torch.tensor(img_npy_resize, device=self.device)
        mask = img_torch[..., 0] == 0 #pixels to mask are black in png

        rospy.loginfo("{} mask px".format(mask.sum()))
        return mask

    def handle_data(self, pc_msg, img_left_msg, img_front_msg, img_right_msg):
        # logstr = "sync check:\n\tcurr time: {}".format(rospy.Time.now().to_sec())
        # logstr += "\n\tpointcloud:  {}".format(pc_msg.header.stamp.to_sec())
        # logstr += "\n\timage left:  {}".format(img_left_msg.header.stamp.to_sec())
        # logstr += "\n\timage front: {}".format(img_front_msg.header.stamp.to_sec())
        # logstr += "\n\timage right: {}".format(img_right_msg.header.stamp.to_sec())
        # rospy.loginfo(logstr)

        self.pcl_msg = pc_msg
        self.image_data['image_left']['message'] = img_left_msg
        self.image_data['image_front']['message'] = img_front_msg
        self.image_data['image_right']['message'] = img_right_msg

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

        pc_frame = self.pcl_msg.header.frame_id
        pc_stamp = self.pcl_msg.header.stamp

        # need to wait for tf to be available
        tf_vehicle_to_pcl_msg = self.get_tf(self.vehicle_frame, pc_frame, pc_stamp)
        if tf_vehicle_to_pcl_msg is None:
            return None

        tf_odom_to_pcl_msg = self.get_tf(self.mapping_frame, pc_frame, pc_stamp)
        if tf_odom_to_pcl_msg is None:
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
            img_frame = self.image_data[img_key]['message'].header.frame_id
            img_stamp = self.image_data[img_key]['message'].header.stamp

            tf_odom_to_veh_pc_msg = self.get_tf(self.mapping_frame, pc_frame, pc_stamp)
            if tf_odom_to_veh_pc_msg is None:
                return None

            # rospy.loginfo("pc time: {}, img time: {}, diff: {}".format(pc_stamp.to_sec(), img_stamp.to_sec(), (img_stamp - pc_stamp).to_sec()))

            tf_odom_to_veh_img_msg = self.get_tf(self.mapping_frame, pc_frame, img_stamp)
            if tf_odom_to_veh_img_msg is None:
                return None

            tf_veh_to_img_msg = self.get_tf(img_frame, pc_frame, img_stamp)
            if tf_veh_to_img_msg is None:
                return None

            odom_to_veh_pc_htm = tf_msg_to_htm(tf_odom_to_veh_pc_msg).to(self.device)
            odom_to_veh_img_htm = tf_msg_to_htm(tf_odom_to_veh_img_msg).to(self.device)
            veh_to_img_htm = tf_msg_to_htm(tf_veh_to_img_msg).to(self.device)

            veh_to_veh_htm = odom_to_veh_img_htm @ torch.linalg.inv(odom_to_veh_pc_htm)

            extrinsics_corrected = veh_to_veh_htm @ veh_to_img_htm

            # rospy.loginfo('extrinsics_correction: {}'.format(veh_to_veh_htm))

            images.append(img)
            image_intrinsics.append(self.image_data[img_key]['intrinsics'])
            image_extrinsics.append(extrinsics_corrected)

        images = torch.cat(images, dim=0)
        image_intrinsics = torch.stack(image_intrinsics, dim=0)
        image_extrinsics = torch.stack(image_extrinsics, dim=0)

        feature_images, feature_intrinsics = self.image_pipeline.run(
            images, image_intrinsics
        )

        feature_images = feature_images.permute(0, 2, 3, 1)
        image_Ps = get_projection_matrix(feature_intrinsics, image_extrinsics)

        coords, valid_mask = get_pixel_projection(pcl_in_vehicle, image_Ps, feature_images)

        if self.use_masks:
            rx = self.image_masks.shape[2] / feature_images.shape[2]
            ry = self.image_masks.shape[1] / feature_images.shape[1]

            I_mask = feature_intrinsics.clone()
            I_mask[:, 0] *= rx
            I_mask[:, 1] *= ry
            P_mask = get_projection_matrix(I_mask, image_extrinsics)

            mask_coords, mask_valid_mask = get_pixel_projection(pcl_in_vehicle, P_mask, self.image_masks)
            mask_feats, mask_cnt = colorize(mask_coords, mask_valid_mask, self.image_masks, bilinear_interpolation=False, reduce=False)

            #find all point feats that are valid projections and land on the mask
            is_masked = mask_valid_mask & mask_feats[..., 0]
            valid_mask = valid_mask & ~is_masked

        if self.do_proj_cleanup:
            valid_mask2 = cleanup_projection(pcl_in_vehicle, coords, valid_mask, feature_images)
            valid_mask = valid_mask & valid_mask2

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

    def get_tf(self, src_frame, dst_frame, stamp, timeout=0.1):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                src_frame,
                dst_frame,
                stamp,
                timeout=rospy.Duration(timeout)
            )
            return tf_msg
        except (tf2_ros.TransformException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("cant tf from {} to {} at {}".format(
                src_frame,
                dst_frame,
                stamp
            ))
            return None

    def make_gridmap_msg(self, bev_grid):
        """
        convert dino into gridmap msg

        Publish all the feature channels, plus a visualization and elevation layer

        Note that we assume all the requisite stuff is available (pcl, img, odom) as this
        should only be called after a dino map is successfully produced
        """
        gridmap_msg = GridMap()

        if not all([k in bev_grid.feature_keys for k in self.gridmap_pub_keys]):
            rospy.logwarn('not all pub keys in internal BEV map. Skipping...')
            return gridmap_msg

        gridmap_data = bev_grid.data.cpu().numpy()

        # setup metadata
        gridmap_msg.info.header.stamp = self.pcl_msg.header.stamp
        gridmap_msg.info.header.frame_id = self.mapping_frame

        gridmap_msg.layers = bev_grid.feature_keys if self.gridmap_pub_keys is None else self.gridmap_pub_keys + ["min_elevation_filtered_inflated_mask"]

        #temp hack
        gridmap_msg.basic_layers = ["min_elevation_filtered_inflated_mask"]
        mask_idx = bev_grid.feature_keys.index("min_elevation_filtered_inflated_mask")
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
        gridmap_msg.info.pose.position.z = 0.
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
        # for i in range(gridmap_data.shape[-1]):
        for k in gridmap_msg.layers:
            i = bev_grid.feature_keys.index(k)
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

        msg = self.xyz_array_to_point_cloud_msg(
            points=pts[mask].cpu().numpy(),
            frame=self.mapping_frame,
            timestamp=self.pcl_msg.header.stamp,
            intensity=feats[..., 0].cpu().numpy()
        )

        return msg

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
        self, points, frame, timestamp=None, rgb_values=None, intensity=None,
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

        if rgb_values is None and intensity is None:
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
        elif rgb_values is None and intensity is not None:
            msg.fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            msg.point_step = 16  # 4 fields (x, y, z, i) each 4 bytes (float32)
            msg.row_step = msg.point_step * msg.width
            xyz = points.astype(np.float32)
            intensity = intensity.astype(np.float32)
            xyzi = np.concatenate([xyz, intensity.reshape(-1, 1)], axis=-1)
            msg.data = xyzi.tobytes()  # Convert to bytes

        elif rgb_values is not None and intensity is None:
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
                pos=res['pos'], feat_pc=res["feature_pc"], do_raytrace=False
            )

            if self.do_terrain_estimation:
                self.bev_grid = self.terrain_estimator.run(self.localmapper.voxel_grid)

            torch.cuda.synchronize()
            update_end_time = time.time()

            pub_start_time = time.time()

            if self.do_terrain_estimation:
                msg = self.make_gridmap_msg(self.bev_grid)
                self.gridmap_pub.publish(msg)


            if isinstance(self.image_pipeline.blocks[-1], TraversabilityPrototypesBlock):
                n_pos_ptypes = len(self.image_pipeline.blocks[-1].obstacle_keys)

                pos_scores = self.localmapper.voxel_grid.features[..., :n_pos_ptypes].max(dim=-1)[0]
                neg_scores = self.localmapper.voxel_grid.features[..., n_pos_ptypes:].max(dim=-1)[0]
                score = pos_scores - neg_scores
                score_feats = torch.stack([score] * 3, dim=-1)

                #copy the voxel grid for pub
                score_voxel_grid = VoxelGrid(
                    metadata=self.localmapper.voxel_grid.metadata,
                    n_features=3,
                    device=self.localmapper.voxel_grid.device
                )
                score_voxel_grid.raster_indices = self.localmapper.voxel_grid.raster_indices.clone()
                score_voxel_grid.feature_mask = self.localmapper.voxel_grid.feature_mask.clone()
                score_voxel_grid.features = score_feats

                score_min = score_feats.min(dim=0)[0]
                score_max = score_feats.max(dim=0)[0]

                msg = self.make_voxel_viz_msg(score_voxel_grid)
                self.voxel_pub.publish(msg)

                pc_pos_scores = res["feature_pc"].features[..., :n_pos_ptypes].max(dim=-1)[0]
                pc_neg_scores = res["feature_pc"].features[..., n_pos_ptypes:].max(dim=-1)[0]
                pc_score = pc_pos_scores - pc_neg_scores
                pc_score_msg = self.xyz_array_to_point_cloud_msg(
                    points = res["feature_pc"].pts[res["feature_pc"].feat_mask].cpu().numpy(),
                    frame=self.mapping_frame,
                    timestamp=self.pcl_msg.header.stamp,
                    intensity = pc_score.cpu().numpy()
                )
                self.pcl_pub.publish(pc_score_msg)

                for i, img_key in enumerate(self.image_keys):
                    feat_img = res["feature_images"][i]
                    #hack to remake pseudo det images

                    n_pos_ptypes = len(self.image_pipeline.blocks[-1].obstacle_keys)
                    pos_score = feat_img[..., :n_pos_ptypes].max(dim=-1)[0]
                    neg_score = feat_img[..., n_pos_ptypes:].max(dim=-1)[0]
                    score = pos_score - neg_score
                    score_img = torch.stack([score] * 3, dim=-1)

                    pub = self.image_pubs[img_key]
                    msg = self.make_img_msg(score_img, img_key, vmin=score_min, vmax=score_max)
                    pub.publish(msg)

            else:
                msg = self.make_voxel_viz_msg(self.localmapper.voxel_grid)
                self.voxel_pub.publish(msg)

                for i, img_key in enumerate(self.image_keys):
                    feat_img = res["feature_images"][i]
                    pub = self.image_pubs[img_key]
                    msg = self.make_img_msg(feat_img, img_key)
                    pub.publish(msg)
                    
                msg = self.make_pcl_msg(res['feature_pc'])
                self.pcl_pub.publish(msg)

            for i, img_key in enumerate(self.image_keys):
                feat_img = res["feature_images"][i]
                #hack to remake pseudo det images

                if isinstance(self.image_pipeline.blocks[-1], TraversabilityPrototypesBlock):
                    n_pos_ptypes = len(self.image_pipeline.blocks[-1].obstacle_keys)
                    pos_score = feat_img[..., :n_pos_ptypes].max(dim=-1)[0]
                    neg_score = feat_img[..., n_pos_ptypes:].max(dim=-1)[0]
                    score = pos_score - neg_score
                    score_img = torch.stack([score] * 3, dim=-1)
                else:
                    score_img = feat_img

                pub = self.image_pubs[img_key]
                msg = self.make_img_msg(score_img, img_key)
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

    config_fp = rospy.get_param("~config_fp")

    # config_fp = '/catkin_ws/src/vfm_voxel_mapping/visual_mapping_noetic/config/ros/rzr_voxel_mapping.yaml'
    config = yaml.safe_load(open(config_fp, 'r'))

    visual_mapping_node = VoxelMappingNode(config)
    rospy.spin()