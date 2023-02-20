#! /usr/bin/env python3
import rospy
from geometry_msgs.msg import *
from std_msgs.msg import *
from open_base.msg import *
import random
import math
import numpy as np
import csv
import rospkg
import time
import os
import cv2
from cv_bridge import CvBridge
import time
import imutils
import heapq
from krssg_ssl_msgs.msg import *
import pandas as pd
import argparse

vision = SSL_DetectionFrame()

def vision_callback(data):
    global vision
    vision = data

if __name__ == "__main__":
    rospy.init_node("grsim_ros_bridge", anonymous=False)

    a = Movement()
    a.movement = 1
    a.generic.type = 2
    a.generic.target.x = 0.2
    a.generic.target.y = 0.0
    pose_2d = Pose2D()
    a.bezier.targetTranslation.append(pose_2d)
    a.bezier.targetRotation.append(0)
    r = rospy.Rate(10)

    target_x = 2.0
    target_y = 2.0

    pos_x = 0
    pos_y = 0
    orientation = 0

    rospy.Subscriber('/vision', SSL_DetectionFrame, vision_callback, tcp_nodelay=True, queue_size=1)

    pub = rospy.Publisher('/open_base_1/command', Movement, queue_size=10)

    while not rospy.is_shutdown():
        try:
            pos_x = vision.robots_yellow[0].x
            pos_y = vision.robots_yellow[0].y
            orientation = vision.robots_yellow[0].orientation
        except:
            pass

        current_distance = round(math.hypot(target_x - pos_x, target_y - pos_y),2)
        goal_angle = math.atan2(target_y - pos_y, target_x - pos_x)

        heading = goal_angle - orientation

        if (abs(heading) < 0.5):
            a.generic.target.x = 0.3
            a.generic.target.y = 0.0
            a.generic.target.theta = 0.0
        else:
            pass
        print(current_distance, heading)

        # pub.publish(a)

        time.sleep(0.1)
