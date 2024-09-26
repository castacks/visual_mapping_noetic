#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
import numpy as np
from threading import Lock
import time
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import matplotlib.pyplot as plt
from rosbag_to_dataset.dtypes.gridmap import GridMapConvert
import matplotlib

CMAP = matplotlib.cm.get_cmap('magma')

class LethalHeightCost(Node):
    def __init__(self, odom_topic, gridmap_topic, costmap_topic):
        super().__init__('lethal_height_cost')

        self._lock = Lock()
        self.odom_msg = None
        self.hz_counter = 0

        self.gridmap_sub = self.create_subscription(GridMap, gridmap_topic, self.handle_map, 1)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.handle_odom, 1)

        self.timer = self.create_timer(0.1, self.run_map)
        self.costmap_pub = self.create_publisher(GridMap, costmap_topic, 2)

        self.new_msg = False
        self.cost = 0.0
        self.channels = []
        self.grid_map_cvt = GridMapConvert(channels=self.channels, size=[1, 1])

        print('DONE WITH INIT')

    def handle_odom(self, msg):
        self.velocity = np.linalg.norm([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
        self.odom_msg = msg

    def handle_map(self, msg):
        print("handling map")
        with self._lock:
            self.dino_map_msg = msg
            self.new_msg = True
            if len(self.channels) == 0:
                for layer in msg.layers:
                    if 'height' in layer:
                        self.channels.append(layer)
                    if 'VLAD' in layer:
                        self.channels.append(layer)
                self.grid_map_cvt.channels = self.channels

    def run_map(self):
        now = time.perf_counter()
        print('----')
        if self.odom_msg is not None:
            if not self.new_msg:
                print('no new map')
                return

            with self._lock:
                print('got map')
                if self.dino_map_msg is not None:
                    info = self.dino_map_msg.info
                    header = self.dino_map_msg.header
                    nx = round(info.length_x / info.resolution)
                    ny = round(info.length_y / info.resolution)
                    self.get_logger().info(f"Map size: {nx} x {ny} info x: {info.length_x} y: {info.length_y} res: {info.resolution}")
                    self.grid_map_cvt.size = [nx, ny]

                    gridmap = self.grid_map_cvt.ros_to_numpy(self.dino_map_msg)

                    self.new_msg = False
                else:
                    gridmap = None

            if gridmap is None:
                print("NO MAP")
                return

            costmap = gridmap['data'][0]
            costmap[:, :] = 0
            self.get_logger().info(f"Costmap shape: {costmap.shape}")

            costmap_msg = self.costmap_to_gridmap(costmap, info, header)
            self.costmap_pub.publish(costmap_msg)

            if self.hz_counter == 50000:
                self.hz_counter = 0
            self.hz_counter += 1

            print(time.perf_counter() - now, 'time')
        else:
            print('no odom')
            return

    def costmap_to_gridmap(self, costmap, info, header, costmap_layer='costmap'):
        costmap_msg = GridMap()
        costmap_msg.header = header
        costmap_msg.info = info
        costmap_msg.layers = [costmap_layer]

        costmap_layer_msg = Float32MultiArray()
        costmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="column_index",
                size=costmap.shape[0],
                stride=costmap.shape[0]
            )
        )
        costmap_layer_msg.layout.dim.append(
            MultiArrayDimension(
                label="row_index",
                size=costmap.shape[0],
                stride=costmap.shape[0] * costmap.shape[1]
            )
        )
        costmap_layer_msg.data = costmap[::-1, ::-1].flatten().tolist()
        costmap_msg.data.append(costmap_layer_msg)

        costmap_msg.layers.append('elevation')
        layer_data = np.zeros_like(costmap) + self.odom_msg.pose.pose.position.z - 1.73
        elevation_layer_msg = Float32MultiArray()
        elevation_layer_msg.layout.dim = costmap_layer_msg.layout.dim
        elevation_layer_msg.data = layer_data.flatten().tolist()
        costmap_msg.data.append(elevation_layer_msg)

        gridmap_cs = (CMAP(costmap / .7) * 255).astype(np.int32)
        gridmap_color = gridmap_cs[..., 0] * (2**16) + gridmap_cs[..., 1] * (2**8) + gridmap_cs[..., 2]
        gridmap_color = gridmap_color.view(dtype=np.float32)

        costmap_msg.layers.append('rgb_viz')
        rgb_viz_msg = Float32MultiArray()
        rgb_viz_msg.layout.dim = costmap_layer_msg.layout.dim
        rgb_viz_msg.data = gridmap_color[::-1, ::-1].flatten().tolist()
        costmap_msg.data.append(rgb_viz_msg)

        return costmap_msg


def main(args=None):
    rclpy.init(args=args)
    node = LethalHeightCost('/zedx/zed_node/odom', '/dino_gridmap', '/cherie_costmap')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
