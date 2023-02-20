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
from sensor_msgs.msg import CompressedImage
from utils.defisheye import Defisheye
from utils.panorama import Stitcher
import time
import imutils
import heapq
from krssg_ssl_msgs.msg import *
import pandas as pd
import argparse

class TestLocal:
    def __init__(self):
        self.bridge = CvBridge()
        self.stitcher = Stitcher()
        self.image_right_down = None
        self.image_right_up = None
        self.image_left_down = None
        self.image_left_up = None
        self.defisheye1 = Defisheye(dtype='linear', format='fullframe', fov=180, pfov=120)
        self.defisheye2 = Defisheye(dtype='linear', format='fullframe', fov=180, pfov=120)
        rospy.Subscriber('/rrbot/camera1/image_raw/compressed', CompressedImage, self.image_left_down_callback, tcp_nodelay=True, queue_size=1)
        rospy.Subscriber('/rrbot/camera2/image_raw/compressed', CompressedImage, self.image_right_up_callback, tcp_nodelay=True, queue_size=1)
        rospy.Subscriber('/rrbot/camera4/image_raw/compressed', CompressedImage, self.image_left_up_callback, tcp_nodelay=True, queue_size=1)
        rospy.Subscriber('/rrbot/camera3/image_raw/compressed', CompressedImage, self.image_right_down_callback, tcp_nodelay=True, queue_size=1)

        self.msg = SSL_DetectionFrame()
        self.msg.robots_yellow.append(SSL_DetectionRobot())
        self.msg.robots_yellow.append(SSL_DetectionRobot())
        self.msg.robots_yellow.append(SSL_DetectionRobot())
        self.msg.robots_yellow.append(SSL_DetectionRobot())
        self.msg.robots_yellow.append(SSL_DetectionRobot())

        self.msg.robots_blue.append(SSL_DetectionRobot())
        self.msg.robots_blue.append(SSL_DetectionRobot())
        self.msg.robots_blue.append(SSL_DetectionRobot())
        self.msg.robots_blue.append(SSL_DetectionRobot())
        self.msg.robots_blue.append(SSL_DetectionRobot())

        self.pos_cyano_x = np.eye(10)
        self.pos_cyano_y = np.eye(10)

        self.theta = np.eye(10)

        self.robot_x = np.eye(10)
        self.robot_y = np.eye(10)

        self.robot_pixel_x = np.eye(10)
        self.robot_pixel_y = np.eye(10)

        self.vision_pub = rospy.Publisher('/vision', SSL_DetectionFrame, tcp_nodelay=True, queue_size=1)

        self.down_left_pix_corner_x = 335
        self.down_left_pix_corner_y = 665
        self.down_right_pix_corner_x = 943
        self.down_right_pix_corner_y = 660
        self.up_left_pix_corner_x = 334
        self.up_left_pix_corner_y = 54
        self.up_right_pix_corner_x = 943
        self.up_right_pix_corner_y = 660

        self.down_left_pix_center_x = 946
        self.down_left_pix_center_y = 46
        self.down_right_pix_center_x = 321
        self.down_right_pix_center_y = 41
        self.up_left_pix_center_x = 957
        self.up_left_pix_center_y = 676
        self.up_right_pix_center_x = 321
        self.up_right_pix_center_y = 41

        self.max_pixel_y = 750
        self.min_pixel_y = 30

        self.size_field = 4.0

        self.id_detected_robot = 0

        # # Loading and compiling presaved trained CNN
        # self.model = load_model('/home/adminutec/catkin_ws/src/ssl_vision/scripts/drawing_classification.h5')
        #
        # self.label = {0: "Circle", 1: "Square", 2: "Triangle"}

    def image_left_down_callback(self, msg):
        self.image_left_down = self.bridge.compressed_imgmsg_to_cv2(msg)#self.defisheye1.convert(self.bridge.compressed_imgmsg_to_cv2(msg))

    def image_right_up_callback(self, msg):
        self.image_right_up = self.bridge.compressed_imgmsg_to_cv2(msg)#self.defisheye1.convert(self.bridge.compressed_imgmsg_to_cv2(msg))

    def image_left_up_callback(self, msg):
        self.image_left_up = self.bridge.compressed_imgmsg_to_cv2(msg)#self.defisheye1.convert(self.bridge.compressed_imgmsg_to_cv2(msg))

    def image_right_down_callback(self, msg):
        self.image_right_down = self.bridge.compressed_imgmsg_to_cv2(msg)#self.defisheye1.convert(self.bridge.compressed_imgmsg_to_cv2(msg))

    def config_image(self, image_in):
        img_gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)
        img_canny = cv2.Canny(img_blur, 50, 50)
        kernel = np.ones((0, 0))
        img_dilate = cv2.dilate(img_canny, kernel, iterations=1)
        img_erode = cv2.erode(img_dilate, kernel, iterations=1)

        ids = []
        ids_contour = []

        contours, hierarchies = cv2.findContours(img_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                ids_contour.append(cv2.contourArea(cnt))
                ids.append(cnt)

        ids_contour = sorted(ids_contour)

        return ids, ids_contour

    def find_id(self, ids_contour, ids, color, frame, camera):

        for cnt in ids:
            M = cv2.moments(cnt)
            x = int(M["m10"]/M["m00"])
            y = int(M["m01"]/M["m00"])

            if camera == "left_down":
                pos_x = round(((x - self.down_left_pix_corner_x) / (self.down_left_pix_center_x - self.down_left_pix_corner_x) * self.size_field/2),2)
                pos_y = round(((y - self.down_left_pix_corner_y) / (self.down_left_pix_center_y - self.down_left_pix_corner_y) * self.size_field/2),2)
            elif (camera == "right_down"):
                pos_x = self.size_field/2 + round(((x - self.down_left_pix_corner_x) / (self.down_left_pix_center_x - self.down_left_pix_corner_x) * self.size_field/2),2)
                pos_y = round(((y - self.down_left_pix_corner_y) / (self.down_left_pix_center_y - self.down_left_pix_corner_y) * self.size_field/2),2)
            elif (camera == "left_up"):
                pos_x = round(((x - self.up_left_pix_corner_x) / (self.up_left_pix_center_x - self.up_left_pix_corner_x) * self.size_field/2),2)
                pos_y = self.size_field - round(((y - self.up_left_pix_corner_y) / (self.up_left_pix_center_y - self.up_left_pix_corner_y) * self.size_field/2),2)
            else:
                pos_x = self.size_field - round(((x - self.up_right_pix_corner_x) / (self.up_right_pix_center_x - self.up_right_pix_corner_x) * self.size_field/2),2)
                pos_y = self.size_field/2 + round(((y - self.up_right_pix_corner_y) / (self.up_right_pix_center_y - self.up_right_pix_corner_y) * self.size_field/2),2)

            # self.predict_one(magenta_image)

            # if cv2.contourArea(cnt) == min(ids_contour):
            if cv2.arcLength(cnt, True) < 50:
                if (color == "magenta"):
                    self.id_detected_robot = 0
                elif(color == "yellow"):
                    self.id_detected_robot = 3
                else:
                    self.id_detected_robot = 6
                    print("Perimeter: "+str(cv2.arcLength(cnt, True)))
            elif cv2.arcLength(cnt, True) > 100:
                if (color == "magenta"):
                    self.id_detected_robot = 1
                elif(color == "yellow"):
                    self.id_detected_robot = 4
                else:
                    self.id_detected_robot = 7
                    print("Perimeter: "+str(cv2.arcLength(cnt, True)))
            else:
                if (color == "magenta"):
                    self.id_detected_robot = 2
                elif(color == "yellow"):
                    self.id_detected_robot = 5
                else:
                    self.id_detected_robot = 8
                    print("Perimeter: "+str(cv2.arcLength(cnt, True)))
            cv2.putText(frame, "Robot: "+str(self.id_detected_robot), (int(x + 10), int(y + 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.putText(frame, "Pos: "+str(pos_x)+", "+str(pos_y), (int(x+10), int(y + 70)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            # print("-------------------------")
            # print(cv2.arcLength(cnt, True))

            self.robot_x[self.id_detected_robot][0] = pos_x
            self.robot_x[self.id_detected_robot] = np.roll(self.robot_x[self.id_detected_robot], 1)
            self.robot_y[self.id_detected_robot][0] = pos_y
            self.robot_y[self.id_detected_robot] = np.roll(self.robot_y[self.id_detected_robot], 1)
            self.robot_pixel_x[self.id_detected_robot][0] = x
            self.robot_pixel_x[self.id_detected_robot] = np.roll(self.robot_pixel_x[self.id_detected_robot], 1)
            self.robot_pixel_y[self.id_detected_robot][0] = y
            self.robot_pixel_y[self.id_detected_robot] = np.roll(self.robot_pixel_y[self.id_detected_robot], 1)
            # print(self.robot_x)

            pos_x_mean = self.robot_x[self.id_detected_robot].mean()
            pos_y_mean = self.robot_y[self.id_detected_robot].mean()

            if (self.id_detected_robot > 4):
                self.msg.robots_yellow[self.id_detected_robot - 5].x = pos_x_mean
                self.msg.robots_yellow[self.id_detected_robot - 5].y = pos_y_mean
                self.msg.robots_yellow[self.id_detected_robot - 5].pixel_x = x
                self.msg.robots_yellow[self.id_detected_robot - 5].pixel_y = y
            else:
                self.msg.robots_blue[self.id_detected_robot].x = pos_x_mean
                self.msg.robots_blue[self.id_detected_robot].y = pos_y_mean
                self.msg.robots_blue[self.id_detected_robot].pixel_x = x
                self.msg.robots_blue[self.id_detected_robot].pixel_y = y

            self.vision_pub.publish(self.msg)

            cv2.circle(frame, (int(pos_x_mean), int(pos_y_mean)), 5, (255, 255, 255), -1)

    def treat_image(self, image, shape, image_name):
        frame = image

        # resize the frame, blur it, and convert it to the HSV color space
        # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # lower_white = np.array([150,0,0])
        # upper_white = np.array([[255,150,120]])

        lower_blue = np.array([150,0,0])
        upper_blue = np.array([[255,150,120]])

        lower_white = np.array([200,200,200])
        upper_white = np.array([[255,255,255]])

        lower_yellow = np.array([0,200,200])
        upper_yellow = np.array([[180,255,255]])

        lower_magenta = np.array([140,0,160])
        upper_magenta = np.array([[255,160,255]])

        lower_orange = np.array([80,100,200])
        upper_orange = np.array([[130,255,255]])

        # lower_cyano = np.array([200,160,120])
        # upper_cyano = np.array([[255,200,255]])
        lower_cyano = np.array([150,0,0])
        upper_cyano = np.array([[255,150,120]])

        lower_mask = cv2.inRange(frame, lower_white, upper_white)
        white_image = cv2.bitwise_and(frame, frame, mask=lower_mask)

        lower_mask = cv2.inRange(frame, lower_yellow, upper_yellow)
        yellow_image = cv2.bitwise_and(frame, frame, mask=lower_mask)

        lower_mask = cv2.inRange(frame, lower_magenta, upper_magenta)
        magenta_image = cv2.bitwise_and(frame, frame, mask=lower_mask)

        lower_mask = cv2.inRange(frame, lower_blue, upper_blue)
        blue_image = cv2.bitwise_and(frame, frame, mask=lower_mask)

        lower_mask = cv2.inRange(frame, lower_orange, upper_orange)
        orange_image = cv2.bitwise_and(frame, frame, mask=lower_mask)

        lower_mask = cv2.inRange(frame, lower_cyano, upper_cyano)
        cyano_image = cv2.bitwise_and(frame, frame, mask=lower_mask)

        # Magenta
        ids, ids_contour = self.config_image(magenta_image)

        self.find_id(ids_contour, ids, "magenta", frame, image_name)

        # Yellow
        ids, ids_contour = self.config_image(yellow_image)

        self.find_id(ids_contour, ids, "yellow", frame, image_name)

        # white
        ids, ids_contour = self.config_image(white_image)

        self.find_id(ids_contour, ids, "white", frame, image_name)

        # Cyano
        ids, ids_contour = self.config_image(cyano_image)
        print(len(ids))

        for cnt in ids:
            M = cv2.moments(cnt)
            x = int(M["m10"]/M["m00"])
            y = int(M["m01"]/M["m00"])

            if image_name == "left_down":
                pos_x = round(((x - self.down_left_pix_corner_x) / (self.down_left_pix_center_x - self.down_left_pix_corner_x) * self.size_field/2),2)
                pos_y = round(((y - self.down_left_pix_corner_y) / (self.down_left_pix_center_y - self.down_left_pix_corner_y) * self.size_field/2),2)
            elif (image_name == "right_down"):
                pos_x = self.size_field/2 + round(((x - self.down_left_pix_corner_x) / (self.down_left_pix_center_x - self.down_left_pix_corner_x) * self.size_field/2),2)
                pos_y = round(((y - self.down_left_pix_corner_y) / (self.down_left_pix_center_y - self.down_left_pix_corner_y) * self.size_field/2),2)
            elif (image_name == "left_up"):
                pos_x = round(((x - self.up_left_pix_corner_x) / (self.up_left_pix_center_x - self.up_left_pix_corner_x) * self.size_field/2),2)
                pos_y = self.size_field - round(((y - self.up_left_pix_corner_y) / (self.up_left_pix_center_y - self.up_left_pix_corner_y) * self.size_field/2),2)
            else:
                pos_x = self.size_field - round(((x - self.up_right_pix_corner_x) / (self.up_right_pix_center_x - self.up_right_pix_corner_x) * self.size_field/2),2)
                pos_y = self.size_field/2 + round(((y - self.up_right_pix_corner_y) / (self.up_right_pix_center_y - self.up_right_pix_corner_y) * self.size_field/2),2)

            lowest_distance = 10
            id_robot = 0

            print("-------------------------")
            for i in range(0, self.pos_cyano_x.shape[0]):
                print(pos_x, pos_y)
                print(np.mean(self.robot_x[i]), np.mean(self.robot_y[i]))
                distance = ((pos_x - np.mean(self.robot_x[i]))**2 + (pos_y - np.mean(self.robot_y[i]))**2)**0.5
                print(distance)
                print("-------------------------")
                if distance < lowest_distance:
                    lowest_distance = distance
                    id_robot = i

            print(id_robot)

            self.pos_cyano_x[id_robot][0] = pos_x
            self.pos_cyano_x[id_robot] = np.roll(self.pos_cyano_x[id_robot], 1)
            self.pos_cyano_y[id_robot][0] = pos_y
            self.pos_cyano_y[id_robot] = np.roll(self.pos_cyano_y[id_robot], 1)

            delta_x = self.pos_cyano_x[id_robot][0] - self.robot_x[id_robot][0]
            delta_y = self.pos_cyano_y[id_robot][0] - self.robot_y[id_robot][0]
            self.theta[id_robot][0] = math.atan2(delta_y, delta_x)
            self.theta[id_robot] = np.roll(self.theta[id_robot], 1)
            # print(self.theta)
            # if (id_robot == 3):
            cv2.putText(frame, "Ori: "+str(self.theta[id_robot].mean()), (int(self.robot_pixel_x[id_robot][1]+10), int(self.robot_pixel_y[id_robot][1] + 90)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), -1)

            if (id_robot > 4):
                self.msg.robots_yellow[id_robot - 5].orientation = self.theta[id_robot].mean()
            else:
                self.msg.robots_blue[id_robot].orientation = self.theta[id_robot].mean()

            self.vision_pub.publish(self.msg)

        # cv2.imshow('Frame', cyano_image)

        return frame

    def step(self):
        if self.image_left_down is None or self.image_right_up is None or self.image_right_down is None or self.image_left_up is None: #or self.image_right_down is None:
            return False
        # frame = self.stitcher.stitch([self.image_left, self.image_right])
        # frame_left_d = self.image_left_down.copy()
        # frame_right_u = self.image_right_up.copy()
        # frame_left_u = self.image_left_up.copy()
        # frame_right_d = self.image_right_down.copy()
        # frame_right_d = self.image_left_down.copy()
        # size = frame_left_d.shape

        # frame_left = frame_left[round(size[0]*0.01):round(size[0]*0.77), round(size[1]*0.25):round(size[1]*1.0)]
        # frame_right = frame_right[round(size[0]*0.1):round(size[0]*0.83), round(size[1]*0.01):round(size[1]*0.72)]

        frame_left_d = self.treat_image(self.image_left_down.copy(), self.image_left_down.shape, "left_down")

        frame_left_u = self.treat_image(self.image_left_up.copy(), self.image_left_up.shape, "left_up")

        frame_right_u = self.treat_image(self.image_right_up.copy(), self.image_right_up.shape, "right_up")

        frame_right_d = self.treat_image(self.image_right_down.copy(), self.image_right_down.shape, "right_down")

        up_image = np.concatenate((frame_left_u, frame_right_u), axis=1)
        down_image = np.concatenate((frame_left_d, frame_right_d), axis=1)
        final_image = np.concatenate((up_image, down_image), axis=0)
        # cv2.imshow('Frame', blue_image)
        cv2.imshow('Frame', final_image)
        return True

rospy.init_node('ssl_master')
test_local = TestLocal()
key = cv2.waitKey(1)
while key != ord('q'):
    start = time.time()
    val = test_local.step()
    if not val:
        continue
    key = cv2.waitKey(1)
    # time.sleep(1/60)
    fps = round(1 / (time.time() - start), 1)
    # print('\rFPS:', fps)

print("[INFO] cleaning up...")
cv2.destroyAllWindows()

# if __name__ == "__main__":
#     rospy.init_node("grsim_ros_bridge", anonymous=False)

#     a = Movement()
#     a.movement = 1
#     a.generic.type = 2
#     a.generic.target.x = 0.1
#     a.generic.target.y = 0.2
#     pose_2d = Pose2D()
#     a.bezier.targetTranslation.append(pose_2d)
#     a.bezier.targetRotation.append(0)
#     r = rospy.Rate(10)

#     pub = rospy.Publisher('/open_base_1/command', Movement, queue_size=10)

#     while not rospy.is_shutdown():

#         pub.publish(a)

#         print(a)

#         time.sleep(0.1)
