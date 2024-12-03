import sys
import time
from utilities import Logger

from rclpy.time import Time

from utilities import *
from rclpy.node import Node
from geometry_msgs.msg import Twist


from rclpy.qos import QoSProfile
from rclpy import qos
from nav_msgs.msg import Odometry as odom

from sensor_msgs.msg import Imu
from kalman_filter import kalman_filter

from rclpy import init, spin, spin_once

import numpy as np
import message_filters



rawSensors=0; kalmanFilter=1

TURTLEBOT = 4
if TURTLEBOT == 4:
    odom_qos=QoSProfile(reliability=2, durability=2, history=1, depth=10)
else:
    odom_qos=QoSProfile(
        reliability=qos.ReliabilityPolicy.RELIABLE, 
        durability=qos.DurabilityPolicy.VOLATILE, 
        history=qos.HistoryPolicy.KEEP_LAST, depth=10)



class localization(Node):
    
    def __init__(self, type):

        super().__init__("localizer")
        
        
        self.ekf_logger=Logger( f"CSVs/{STARTTIME}-{EUCLIDIAN}-robotPose-EKF.csv" , 
                               ["imu_ax", "imu_ay", "kf_ax", "kf_ay","kf_vx","kf_w","kf_x", "kf_y","stamp"])
        self.loc_logger=Logger( f"CSVs/{STARTTIME}-{EUCLIDIAN}-robotPose-odom.csv" , 
                               ["odom_x", "odom_y", "odom_theta","stamp"])
        self.pose=None
        
        if type==rawSensors:
            self.initRawSensors()
        elif type==kalmanFilter:
            self.initKalmanfilter()
            self.kalmanInitialized = False
        else:
            print("We don't have this type for localization", sys.stderr)
            return            
    
        self.timelast=time.time()
    
    def initRawSensors(self):
        self.create_subscription(odom, "/odom", self.odom_callback, qos_profile=odom_qos)

    def initKalmanfilter(self):
        
        self.odom_sub=message_filters.Subscriber(self, odom, "/odom", qos_profile=odom_qos)
        self.imu_sub=message_filters.Subscriber(self, Imu, "/imu", qos_profile=odom_qos)
        
        time_syncher=message_filters.ApproximateTimeSynchronizer([self.odom_sub, self.imu_sub], queue_size=10, slop=0.1)
        
        time_syncher.registerCallback(self.fusion_callback)
        
    
    def fusion_callback(self, odom_msg: odom, imu_msg: Imu):

        if not self.kalmanInitialized:
            x=np.array([odom_msg.pose.pose.position.x,
                        odom_msg.pose.pose.position.y,
                        euler_from_quaternion(odom_msg.pose.pose.orientation),
                        0,
                        0,
                        0])        
            
            # Best Q and R matrices from lab 3 are used
            Q_CONST = 0.1
            R_CONST = 0.9
            
            Q= Q_CONST * np.array([[1,0,0,0,0,0], # Q array is Q constant multiplied by 6x6 I matrix from state vector x=[x,y,th,w,v,vdot]
                            [0,1,0,0,0,0],
                            [0,0,1,0,0,0],
                            [0,0,0,1,0,0],
                            [0,0,0,0,1,0],
                            [0,0,0,0,0,1]])

            R= R_CONST * np.array([[1,0,0,0], # R array is R constant multiplied by 4x4 I matrix from odometry and IMU z=[v,w,ax,ay]
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1],])
            P=Q
                        
            self.kf=kalman_filter(P,Q,R, x)
            
            self.kalmanInitialized = True

        
        dt = time.time() - self.timelast

        self.timelast=time.time()


        z=np.array([odom_msg.twist.twist.linear.x,
                    odom_msg.twist.twist.angular.z,
                    imu_msg.linear_acceleration.x,
                    imu_msg.linear_acceleration.y])
        
        self.loc_logger.log_values([odom_msg.pose.pose.position.x,
                    odom_msg.pose.pose.position.y,
                    euler_from_quaternion(odom_msg.pose.pose.orientation), 
                    odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec*1e-9]) # Time stamp])
        
        self.kf.predict(dt)
        self.kf.update(z)
        
        xhat=self.kf.get_states()
        
        self.pose=np.array([xhat[0],
                            xhat[1],
                            normalize_angle(xhat[2]),
                            odom_msg.header.stamp])
        
        self.ekf_logger.log_values([
                            z[2], # ax
                            z[3], # ay
                            xhat[5], # EKF ax
                            xhat[4]*xhat[3], # EKF ay
                            xhat[4]*np.cos(xhat[2]), # EKF v
                            xhat[3], # EKF w
                            xhat[0], # EKF x
                            xhat[1], # EKF y
                            odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec*1e-9 # Time stamp
                            ])
        
    def odom_callback(self, pose_msg):
        
        self.pose=[ pose_msg.pose.pose.position.x,
                    pose_msg.pose.pose.position.y,
                    euler_from_quaternion(pose_msg.pose.pose.orientation),
                    pose_msg.header.stamp]
        
        self.loc_logger.log_values([pose_msg.pose.pose.position.x,
                    pose_msg.pose.pose.position.y,
                    euler_from_quaternion(pose_msg.pose.pose.orientation), 
                    pose_msg.header.stamp.sec + pose_msg.header.stamp.nanosec*1e-9]) # Time stamp])
        

        
    def getPose(self):
        return self.pose


if __name__=="__main__":
    
    init()
    
    LOCALIZER=localization()
    
    
    spin(LOCALIZER)
