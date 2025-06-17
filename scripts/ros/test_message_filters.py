import rospy
import copy
import time
import numpy as np

import message_filters

from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap

class MessageFilterNode:
    """
    Test message filters
    """
    def __init__(self):
        img_topics = [
            '/crl_rzr/multisense_left/aux/image_rect_color',
            '/crl_rzr/multisense_right/aux/image_rect_color',
            '/crl_rzr/multisense_front/aux/image_rect_color',
        ]

        self.pc_msg = None
        self.img_msgs = {k:None for k in img_topics}
        
        # no sync
        # self.pcl_sub = rospy.Subscriber('/crl_rzr/velodyne_merged_points', PointCloud2, self.handle_pointcloud)
        # self.img_sub = rospy.Subscriber('/crl_rzr/multisense_right/aux/image_rect_color', Image, self.handle_image)

        self.pcl_sub = message_filters.Subscriber('/crl_rzr/velodyne_merged_points', PointCloud2)
        self.left_img_sub = message_filters.Subscriber()

        self.time_sync = message_filters.ApproximateTimeSynchronizer([self.pcl_sub] + self.img_subs, 10, slop=0.1)
        self.time_sync.registerCallback(self.handle_data)

        self.timer = rospy.Timer(rospy.Duration(0.2), self.spin)

    def handle_data(self, pc_msg, img_msgs):
        rospy.loginfo('timesync callbakc')
        rospy.loginfo(img_msgs)
        self.pc_msg = pc_msg
        self.img_msg = img_msg

    def handle_pointcloud(self, msg):
        self.pc_msg = msg

    def handle_image(self, msg):
        self.img_msg = msg

    def spin(self, event):
        rospy.loginfo('check sync...')
        if self.pc_msg is not None and self.img_msg is not None:
            rospy.loginfo("pc  time = {}".format(self.pc_msg.header.stamp.to_sec()))
            rospy.loginfo("img time = {}".format(self.img_msg.header.stamp.to_sec()))

if __name__ == "__main__":
    rospy.init_node('test_message_filters')

    visual_mapping_node = MessageFilterNode()
    rospy.spin()