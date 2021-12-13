
import sys
import time
import rospy
import os
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import PointCloud2 as msg_PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Imu as msg_Imu
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2
import pyrealsense2 as rs2
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2
import torch
import argparse

from model import get_torchvision_maskrcnn
from transform import get_transforms
from sensor_msgs.msg import CameraInfo


class Retriever():
    def __init__(self, depth_image_topic, depth_info_topic):
        self.filepath = sys.argv[1]
        self.bridge = CvBridge()
        self.rgb_id = 0
        self.depth_id = 0

        if not os.path.exists(self.filepath):
            os.mkdir(self.filepath)

        self.sub_rgb = rospy.Subscriber('/camera/color/image_raw',msg_Image, self.rgb_callback)
        self.sub_depth = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', msg_Image, self.imageDepthCallback)
        # self.sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw',msg_Image,self.callback)
        self.sub_info = rospy.Subscriber(depth_info_topic, CameraInfo, self.imageDepthInfoCallback)


    def rgb_callback(self,ros_pics):
        rospy.loginfo(self.rgb_id)
        try: 
            my_image = self.bridge.imgmsg_to_cv2(ros_pics, ros_pics.encoding)

        except CvBridgeError as e:
            print("CvBridge could not convert images from realsense to opencv")
        height,width, channels = my_image.shape
        my_height = my_image.shape[1]
        
        # img_path = os.path.join(self.filepath, 'rgb_%d_%s-%s.png' % (self.rgb_id, my_image.header.stamp.secs, my_image.header.stamp.nsecs))
        img_path = os.path.join(self.filepath, 'rgb_%d.png' % (self.rgb_id))
        #print (my_image)

        my_image = cv2.cvtColor(my_image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('window',my_image)
        
        cv2.imwrite(img_path, my_image)
        cv2.waitKey(50)
        #print(my_height)
        self.rgb_id += 1

    def imageDepthCallback(self, ros_depth):
        
        rospy.loginfo(self.depth_id)
        
        try:
            my_depth = self.bridge.imgmsg_to_cv2(ros_depth, ros_depth.encoding)

        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            return

        # img_path = os.path.join(self.filepath, 'depth_%d_%s-%s.npy' % (self.depth_id, my_depth.header.stamp.secs, my_depth.header.stamp.nsecs))
        img_path = os.path.join(self.filepath, 'depth_%d.npy' % (self.depth_id))
        # cv2.imshow('depth',my_depth )
        np.save(img_path, my_depth)
        # cv2.waitKey(50)
        
        self.depth_id += 1
    

    def imageDepthInfoCallback(self, cameraInfo):
        try:
            if self.intrinsics:
                return
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.K[2]
            self.intrinsics.ppy = cameraInfo.K[5]
            self.intrinsics.fx = cameraInfo.K[0]
            self.intrinsics.fy = cameraInfo.K[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in cameraInfo.D]
        except CvBridgeError as e:
            print(e)
            return

def main():
    rospy.init_node("rs2_listener",anonymous = True)

    depth_image_topic = '/camera/aligned_depth_to_color/image_raw'
    depth_info_topic = '/camera/aligned_depth_to_color/camera_info'
    c = Retriever(depth_image_topic ,  depth_info_topic)
    
    rospy.spin()


if __name__ == '__main__':

    main()
