#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import numpy as np
import time 

class PointCloudDownsampler:
    def __init__(self):
        self.point_cloud_sub = rospy.Subscriber('/zedx/zed_node/point_cloud/cloud_registered', PointCloud2, self.point_cloud_callback)
        self.point_cloud_pub = rospy.Publisher('/pcl_downsampled', PointCloud2, queue_size=10)

    def point_cloud_callback(self, msg):
        start_time = time.time()
        points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)))

        # Downsample the point cloud to every 10th point
        downsampled_points = points[::3]

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = msg.header.frame_id

        downsampled_cloud = pc2.create_cloud(header, msg.fields, downsampled_points)
        self.point_cloud_pub.publish(downsampled_cloud)
        print("Downsampling took", time.time() - start_time, "for #points", points.shape)

if __name__ == '__main__':
    rospy.init_node('point_cloud_downsampler', anonymous=True)
    pc_downsampler = PointCloudDownsampler()
    rospy.spin()
