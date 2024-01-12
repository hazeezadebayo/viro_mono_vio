#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Quaternion, Vector3, TwistStamped, TransformStamped
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster
from scipy.linalg import sqrtm # import scipy.linalg
from copy import deepcopy
from numpy.linalg import inv 

import cv2, math, sys, warnings, time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

from copy import deepcopy
from threading import Lock

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.filterwarnings("ignore", category=np.ComplexWarning)







# [image_subscriber]: Camera Info: sensor_msgs.msg.CameraInfo(header=std_msgs.msg.Header(stamp=builtin_interfaces.msg.Time(sec=38, nanosec=736000000), 
# frame_id='kinect_camera_optical'), 
# height=480, width=640, 
# distortion_model='plumb_bob', 
# d=[0.0, 0.0, 0.0, 0.0, 0.0], 
# k=array([528.43375656,   0.        , 320.5       ,   0.        ,
#        528.43375656, 240.5       ,   0.        ,   0.        ,
#          1.        ]), 
# r=array([1., 0., 0., 0., 1., 0., 0., 0., 1.]), 
# p=array([528.43375656,   0.        , 320.5       ,  -0.        ,
#          0.        , 528.43375656, 240.5       ,   0.        ,
#          0.        ,   0.        ,   1.        ,   0.        ]), 
# binning_x=0, binning_y=0, 
# roi=sensor_msgs.msg.RegionOfInterest(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False))

    # could be replaced with alpha = old depth / new depth of any pixel u = (u,v)', where alpha is the scale
    # or
    # sample coordinates x, y of (t) and x, y of (t-1) from the odom when picture is snapped and use them directly below
    # or
    # that xyz - xyz
    # or
    # that T (x y z) from rt left camera relative to right --- although i highly doubt this. it might work sha

    #def getAbsoluteScale(self, frame_id):
    #    """ Obtains the absolute scale utilizing
    #    the ground truth poses. (KITTI dataset)"""
    #    z_prev / z
    #    return np.sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev))




# pip3 uninstall opencv-contrib-python
# pip3 install opencv-contrib-python

# source install/setup.sh
# source /opt/ros/humble/setup.bash
# ros2 run viro_mono_vio mono_vo_node.py




# -----------------------------------------
# -----------------------------------------
        

### -------------------------------------



# Covariance for EKF simulation
Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    1.0  # variance of velocity
]) ** 2  # predict state covariance
R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance
#  Simulation parameter
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2
DT = 0.5 # 0.1  # time tick [s]
show_animation = True

# def calc_input():
#     v = 1.0  # [m/s]
#     yawrate = 0.1  # [rad/s]
#     u = np.array([[v], [yawrate]])
#     return u

def observation(xTrue, xd, u):
    xTrue = motion_model(xTrue, u)
    # add noise to gps x-y
    z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)
    # add noise to input
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)
    xd = motion_model(xd, ud)
    return xTrue, z, xd, ud

def motion_model(x, u):
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])
    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])
    x = F @ x + B @ u
    return x

def observation_model(x):
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    z = H @ x
    return z

def jacob_f(x, u):
    """
    Jacobian of Motion Model:
    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])
    return jF

def jacob_h():
    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    return jH

def ekf_estimation(xEst, PEst, z, u):
    #  Predict
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q
    #  Update
    jH = jacob_h()
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R
    K = PPred @ jH.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
    return xEst, PEst







class KalmanFilter:
    def __init__(self, init_x, init_y, R_value=10): 
        # Variable to store the previous time    
        self.prv_time = 0  
        # Initializing state vector X with initial camera position
        self.X = np.array([[init_x], [init_y], [0], [0]])
        # Measurement noise covariance matrix R 2X2 represents uncertainity in the measurements 
        # Look into initializing these values
        self.R = np.array([ [R_value, 0],
                            [0, R_value] ]) # Measurement noise covariance matrix
        # Initializing covariance matrix P 4X4 with initial values
        self.P = np.array([ [10, 0, 0, 0],
                            [0, 10, 0, 0],
                            [0, 0, 1000, 0],
                            [0, 0, 0, 1000] ])
        # State transition matrix A 4X4
        self.A = np.array([ [1.0, 0, 1.0, 0],
                            [0, 1.0, 0, 1.0],
                            [0, 0, 1.0, 0],
                            [0, 0, 0, 1.0] ])
        # Measurement matrix H 2X4
        self.H = np.array([ [1.0, 0, 0, 0],
                            [0, 1.0, 0, 0] ])
        # Identity matrix I 4X4
        self.I = np.identity(4)
        # Zero measurement vector Z 2X1
        self.Z = np.zeros([2, 1])  

    # The predict() function helps in predicting the future state and reducing the uncertainty in the state 
    # estimation by incorporating the process model and process noise
    # X state vector representing the current state estimation
    # P Covariance matrix representing the uncertainty in the state estimation
    # Q Process covariance matrix representing the uncertainty in the state estimation
    # A State transition matrix relating the current state to the next state
    def predict(self, Q):
        # Predict the next state using the state transition matrix A
        self.X = np.matmul(self.A, self.X)     # multiply thr state transition matrix A with the current state vector X. Result assigned back to X
        # Transpose of A
        At = np.transpose(self.A)
        # Update the covariance matrix P using the predicted state and process noise Q
        # Covariance matrix P is updated by multiplying A with the Product of P and 
        # Transpose of State Transition Matrix A, and then adding the process noise covariance matrix Q
        self.P = np.add(np.matmul(self.A, np.matmul(self.P, At)), Q)  
        # Return the predicted state and updated covariance matrix
        return self.X, self.P

    # The update() function helps in incorporating the measured data into the state estimation process,
    # adjusting the state estimate based on the measurement and reducing the the uncertainity in the state estimate
    # Z Measurement vector representing the observed values
    # X State vector representing the current state estimation
    # P Covariance matrix representing the uncertainty in the state estimation
    # H Measurement matrix relating the state to the measurement
    # R Measurment noise covariance matrix representing the uncertainty in the measurement model
    # I Identity matrix
    def update(self, Z):
        # Calculate the innovation or measurement residual
        Y = np.subtract(Z, np.matmul(self.H, self.X))
        # Transpose of H
        Ht = np.transpose(self.H)
        # Calculate the innovation covariance matrix S
        S = np.add(np.matmul(self.H, np.matmul(self.P, Ht)), self.R)
        # Calculate the Kalman gain K
        K = np.matmul(self.P, Ht)
        Si = inv(S)
        K = np.matmul(K, Si)    
        # Update state and covariance matrices using Kalman gain
        self.X = np.add(self.X, np.matmul(K, Y))
        self.P = np.matmul(np.subtract(self.I ,np.matmul(K, self.H)), self.P)
        # Return updated state and covariance matrices
        return self.X, self.P




    # -----------------------------------------
    # -----------------------------------------
            
class SensorFusionUKF(Node):
    def __init__(self):
        super().__init__('sensor_fusion_ukf')

        # TF Broadcaster for fused transformation
        self.tf_broadcaster = TransformBroadcaster(self)

        imu_topic = '/imu'
        mono_vo_pose_topic = '/mono_vo_pose'
        mono_vl_pose_topic = '/mono_vl_pose'

        # self.last_time = 0
        # State Vector [x y yaw v]'
        self.xEst = np.zeros((4, 1))
        self.xTrue = np.zeros((4, 1))
        self.PEst = np.eye(4)

        self.xDR = np.zeros((4, 1))  # Dead reckoning

        # history
        self.hxEst = self.xEst
        self.hxTrue = self.xTrue
        # self.hxDR = self.xTrue
        self.hz = np.zeros((2, 1))

        self.v = 0.0
        self.theta = 0.0
        self.old_vo_data = np.array([
                0.0, # ret0
                0.0, # ret1
            ]) 
        self.imu_data = np.array([
                0.0, # ret2
                0.0, # ret4
                0.0  # ret5
            ])  
        self.vo_data = np.array([
                0.0, # ret0
                0.0, # ret1
            ]) 

        self.trajectory = [] 
        self.traj = np.zeros((600,600,3), dtype=np.uint8)
        plt.figure()

        # ROS2 Subscribers
        self.vimu_subscription = self.create_subscription(
            Imu,
            imu_topic,
            self.imu_callback,
            10)
        
        self.vo_pose_subscription = self.create_subscription(
            PoseStamped,
            mono_vo_pose_topic,
            self.vo_pose_callback,
            10)

        self.vl_pose_subscription = self.create_subscription(
            PoseStamped,
            mono_vl_pose_topic,
            self.vl_pose_callback,
            10)

        # ROS2 Publishers
        self.fused_pose_pub = self.create_publisher(PoseStamped, 'fused_pose', 10)

        # ROS2 timer
        self.timer = self.create_timer(DT, self.timer_callback)

        # kalman filter 
        self.imu_kf = KalmanFilter(init_x=0.0, init_y=0.0, R_value=10)
        self.vo_kf = KalmanFilter(init_x=0.0, init_y=0.0, R_value=10)


    # -----------------------------------------
    # -----------------------------------------

    def euler_to_quaternion(self, yaw, pitch, roll):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qx, qy, qz, qw]


    def quaternion_to_euler(self, x, y, z, w):
        # roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x**2 + y**2)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        # pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        # yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y**2 + z**2)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    # -----------------------------------------
    # -----------------------------------------

    def _rotationMatrixToEulerAngles(self, R):
        # Calculates rotation matrix to euler angles
        # The result is the same as MATLAB except the order
        # of the euler angles ( x and z are swapped ).
        
        def isRotationMatrix(R):
            Rt = np.transpose(R)
            shouldBeIdentity = np.dot(Rt, R)
            I = np.identity(3, dtype=R.dtype)
            n = np.linalg.norm(I - shouldBeIdentity)
            return n < 1e-6        
        assert (isRotationMatrix(R))

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    # -----------------------------------------
    # -----------------------------------------  

    # -----------------------------------------
    # -----------------------------------------

    def vl_pose_callback(self, msg):
        self.get_logger().info('PoseStamped data:')
        self.get_logger().info('Pose: %s' % msg.pose)
        self.get_logger().info('Header: %s' % msg.header)

    # -----------------------------------------
    # -----------------------------------------

    def imu_callback(self, msg):
        # self.get_logger().info('IMU data: ', msg)
        # imu_data = np.array([ # orientation
        #                     msg.orientation.x,
        #                     msg.orientation.y,
        #                     msg.orientation.z,
        #                     msg.orientation.w,
        #                     # angular_velocity
        #                     msg.angular_velocity.x, 
        #                     msg.angular_velocity.y,
        #                     msg.angular_velocity.z,
        #                     # linear_acceleration
        #                     msg.linear_acceleration.x, 
        #                     msg.linear_acceleration.y,
        #                     msg.linear_acceleration.z,])
        # So, Update the state estimator with IMU data
        # msg.orientation.z might be used for ret[2] (yaw), 
        # msg.angular_velocity.z for ret[4] (yaw rate), and 
        # msg.linear_acceleration.x for ret[5] (longitudinal acceleration).
        # input data validation
        self.norm_q_tolerance = 0.1 # # to block [0,0,0,0] quaternion
        self.norm_acc_threshold = 9.7 # 9.800087415439947 # 0.1 # to block [0,0,0] linear_acceleration
        # validate imu message
        acc = msg.linear_acceleration
        q = msg.orientation
        acc_vec = np.array([acc.x, acc.y, acc.z])
        q_vec = np.array([q.x, q.y, q.z, q.w])
        norm_acc = np.linalg.norm(acc_vec)
        norm_q = np.linalg.norm(q_vec)
        # self.get_logger().info("--> norm_acc: "+str(norm_acc)+", norm_q: "+str(np.abs(norm_q-1.0)))
        if self.norm_acc_threshold <= norm_acc and np.abs(norm_q-1.0) < self.norm_q_tolerance:
            self.valid_imu = True
            # ("imu input is invalid. (linear_acceleration="+str(acc_vec)+", orientation="+str(q_vec)+")")
            roll, pitch, yaw = self.quaternion_to_euler(
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,)
            # For each new measurement z
            # get the current time in seconds since the epoch
            cur_time = time.time()
            filtered_yaw, filtered_ang_vel_z = self.kalman_filter(self.imu_kf, yaw, msg.angular_velocity.z, cur_time)
            self.imu_data = np.array([
                filtered_yaw, # yaw, # filtered_yaw, # ret2
                filtered_ang_vel_z, # msg.angular_velocity.z, # filtered_ang_vel_z, # msg.angular_velocity.z, # ret4
                msg.linear_acceleration.x # ret5
            ])   
        else:
            self.valid_imu = False
            self.imu_data = np.array([
                0.0, # ret2
                0.0, # ret4
                0.0  # ret5
            ]) 
        
    # -----------------------------------------
    # -----------------------------------------
  
    def calc_input(self):
        v = self.v # 1.0  # [m/s]
        yawrate = self.imu_data[1] # 0.1  # [rad/s]
        u = np.array([[v], [yawrate]])
        return u

    # -----------------------------------------
    # -----------------------------------------
     
    def vo_pose_callback(self, msg):
        # self.get_logger().info('PoseStamped data:')
        # self.get_logger().info('Pose: %s' % msg.pose)
        # vo_data = np.array([msg.pose.position.x, 
        #                     msg.pose.position.y,
        #                     msg.pose.position.z,
        #                     msg.orientation.x,
        #                     msg.orientation.y,
        #                     msg.orientation.z,
        #                     msg.orientation.w,]) 
        self.get_logger().info(" -- "+str(self.vo_data[0])+" :: "+str(self.vo_data[1]))
        cur_time = time.time()
        filtered_x, filtered_y = self.kalman_filter(self.vo_kf, msg.pose.position.x, msg.pose.position.y, cur_time)
        self.vo_data = np.array([
                filtered_x, # ret0
                filtered_y, # ret1
            ]) 
        # Calculate the differences
        dx = self.vo_data[0] - self.old_vo_data[0]
        dy = self.vo_data[1] - self.old_vo_data[1]
        # Calculate velocities in x and y directions
        vx = dx / DT
        vy = dy / DT
        # Calculate total velocity and direction
        self.v = math.sqrt(vx**2 + vy**2)
        self.theta = math.atan2(vy, vx)
        # update old vo data 
        self.old_vo_data = self.vo_data


    # -----------------------------------------
    # -----------------------------------------

    # The `kalman_filter()` function applies the Kalman filter algorithm to the given sensor measurements,
    # incorporating system dynamics, noise models, and update steps to provide a robust estimation of the true state.
    def kalman_filter(self, kf, x_m, y_m, cur_time, noise_ax=0.005, noise_ay=0.005):
        # Filtering Loop 
        # --------------
        # The x and y values of the new measurement are extracted into the new_measurement list 
        new_measurement = [x_m, y_m]
        # Time step calculation
        # The time difference dt between the current measurement time stamp and the previous measurement time stamp is calculated
        dt = cur_time - kf.prv_time # cur_time = measurement_time_stamps
        dt_2 = dt * dt
        dt_3 = dt_2 * dt
        dt_4 = dt_3 * dt
        # update the previous time
        kf.prv_time = cur_time 
        # Update of matrix A with dt value
        # The element of matrix A are updated to incorporate the value of dt for modeling the system dynamics
        kf.A[0][2] = dt
        kf.A[1][3] = dt
        # Update of Q matrix
        # The elements of the process noise covariance matrix Q are updated based on dt and noise magnitudes 
        # along the x noise_ax and y nise_ay axes respectively 
        # noise_ax ----> Acceleration noise in x direction
        # noise_ay ----> Acceleration noise in y direction
        # Zero process covariance matrix Q 4X4
        Q = np.zeros([4, 4])
        Q[0][0] = dt_4/4*noise_ax
        Q[0][2] = dt_3/2*noise_ax
        Q[1][1] = dt_4/4*noise_ay
        Q[1][3] = dt_3/2*noise_ay
        Q[2][0] = dt_3/2*noise_ax
        Q[2][2] = dt_2*noise_ax
        Q[3][1] = dt_3/2*noise_ay
        Q[3][3] = dt_2*noise_ay
        # Update sensor readings
        # The measurement vector Z is updated with current x and y coordinate values
        kf.Z[0][0] = new_measurement[0]
        kf.Z[1][0] = new_measurement[1]
        # Perform prediction step
        #The `predict()` function is called with the state estimation (`X`), covariance matrix (`P`), process noise covariance matrix (`Q`), and system matrix (`A`) as inputs.
        #The `predict()` function should returns the updated state estimation and covariance matrix, which are assigned back to `X` and `P`.
        kf.predict(Q)
        # Perform update step
        # The `update()` function is called with the updated measurement vector (`Z`), state estimation (`X`), covariance matrix (`P`), measurement matrix (`H`), measurement noise covariance matrix (`R`), and identity matrix (`I`).
        # The `update()` function returns the updated state estimation and covariance matrix, which are assigned back to `X` and `P`.
        X, P = kf.update(kf.Z)
        # Append filtered x and y values to respective lists
        #The x and y values from the updated state estimation (`X`) are extracted and appended to the `x_filtered` and `y_filtered` lists respectively.
        x_filtered = X[0][0]  # Filtered x-coordinate values
        y_filtered = X[1][0]  # Filtered y-coordinate values
        return x_filtered, y_filtered

    # -----------------------------------------
    # -----------------------------------------

    def timer_callback(self): 
        # filter process       
        # self.get_logger().info("ok-- start!!"+str(self.v)+" "+str(self.imu_data[0]))
        u = self.calc_input()

        self.xTrue, z, self.xDR, ud = observation(self.xTrue, self.xDR, u)

        self.xEst, self.PEst = ekf_estimation(self.xEst, self.PEst, z, ud)

        # store data history
        self.hxEst = np.hstack((self.hxEst, self.xEst))
        # self.hxDR = np.hstack((self.hxDR, self.xDR))
        self.hxTrue = np.hstack((self.hxTrue, self.xTrue))
        self.hz = np.hstack((self.hz, z))

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(self.hz[0, :], self.hz[1, :], ".g")
            plt.plot(self.hxTrue[0, :].flatten(),
                    self.hxTrue[1, :].flatten(), "-b")
            # plt.plot(self.hxDR[0, :].flatten(),
            #         self.hxDR[1, :].flatten(), "-k")
            plt.plot(self.hxEst[0, :].flatten(),
                    self.hxEst[1, :].flatten(), "-r")
            # plot_covariance_ellipse(xEst[0, 0], xEst[1, 0], PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)

        # # Publish the fused pose
        # fused_pose_msg = PoseStamped()
        # fused_pose_msg.pose.position.x = fused_state[0]
        # fused_pose_msg.pose.position.y = fused_state[1]
        # fused_pose_msg.pose.position.z = fused_state[2]
        # # Assuming the state contains roll, pitch, and yaw in the last three elements
        # fused_pose_msg.pose.orientation = Quaternion(
        #     roll=fused_state[3], pitch=fused_state[4], yaw=fused_state[5])
        # # self.fused_pose_pub.publish(fused_pose_msg)

        # # Broadcast the fused transformation as a TF frame
        # tf_msg = TransformStamped()
        # tf_msg.header.stamp = self.get_clock().now().to_msg()
        # tf_msg.header.frame_id = 'world'
        # tf_msg.child_frame_id = 'robot_frame'  # Replace with your robot frame ID
        # tf_msg.transform.translation = Vector3(x=fused_state[0], y=fused_state[1], z=fused_state[2])
        # tf_msg.transform.rotation = Quaternion(roll=fused_state[3], pitch=fused_state[4], yaw=fused_state[5])
        # self.tf_broadcaster.sendTransform(tf_msg)



    



# -----------------------------------------
# -----------------------------------------
  
def main(args=None):
    rclpy.init(args=args)
    try: 
        sensor_fusion_node = SensorFusionUKF()
        rclpy.spin(sensor_fusion_node)
    except:   # except SystemExit: # <- process the exception 
        sensor_fusion_node.destroy_node()  #    rclpy.logging.get_logger("Route_pub").info('Exited')
        rclpy.shutdown()

# -----------------------------------------
# -----------------------------------------

if __name__ == '__main__':
    main()






























































































# ...........................................................................
# ........................... SIMPLE KALMAN FILTER ..........................
# ...........................................................................
    
# self.imu_kf = KalmanFilter(R=0.01, Q=0.001) # # Measurement uncertainty | # Process uncertainty
# self.vo_kf = KalmanFilter(R=0.001, Q=0.01) # # Measurement uncertainty | # Process uncertainty

##  # For each new measurement z
##  self.imu_kf.predict() 
##  self.imu_kf.update(np.array([yaw, msg.angular_velocity.z]))  # Update with 2D measurement | self.kf.update(msg.angular_velocity.z) 
##  # The filtered state vector is now in self.kf.x
##  filtered_yaw, filtered_ang_vel_z = self.imu_kf.x.flatten() # filtered_ang_vel_z = self.kf.x

## self.vo_kf.predict() 
## self.vo_kf.update(np.array([msg.pose.position.x, msg.pose.position.y])) 
## filtered_x, filtered_y = self.vo_kf.x.flatten()

# class KalmanFilter:
#     def __init__(self, R, Q, dim_x = 2, dim_z = 2):
#         self.dim_x = dim_x # Dimension of state vector
#         self.dim_z = dim_z  # Dimension of measurement vector
#         self.A = np.eye(self.dim_x)  # State transition matrix
#         self.H = np.eye(self.dim_z)  # Measurement matrix
#         self.R = R * np.eye(self.dim_z)  # Measurement uncertainty
#         self.Q = Q * np.eye(self.dim_x)  # Process uncertainty
#         self.P = np.eye(self.dim_x)  # Error covariance matrix
#         self.x = np.zeros((self.dim_x, 1))  # Initial state

#     def predict(self):
#         # Update time state
#         self.x = np.dot(self.A, self.x)
#         self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
#         return self.x

#     def update(self, z):
#         # Compute Kalman Gain
#         S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
#         K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

#         # Update the estimate via z
#         self.x = self.x + np.dot(K, (z.reshape(-1, 1) - np.dot(self.H, self.x)))

#         # Update the error covariance
#         self.P = np.dot((np.eye(K.shape[0]) - np.dot(K, self.H)), self.P)

# ...........................................................................
# ...........................................................................




    # def _imu_callback(self, msg):
    #     q = msg.orientation
    #     yaw_y = 2. * (q.x * q.y + q.z * q.w)
    #     yaw_x = q.w**2 - q.z**2 - q.y * 2 + q.x**2
    #     self._pose[2] = np.arctan2(yaw_y, yaw_x)
    #     self._imu_time = msg.header.stamp.to_sec()




    # def complementary(self):
    #     alpha = 0.98
    #     enc = math.atan2(self.dis_right-self.dis_left, wheelbase)
    #     self.theta = alpha*(self.theta + self.gyro_z * self.dt) + (1 - alpha)*enc

    # def kalman(self):
    #     enc = math.atan2(self.dis_right - self.dis_left, wheelbase)
    #     z = self.theta + enc
    #     #predict
    #     theta = self.theta + self.gyro_z * self.dt
    #     p = self.kalman_p + self.kalman_q
    #     #update
    #     self.kalman_k = p * self.kalman_h / (p * self.kalman_h**2 + self.kalman_r)
    #     self.theta = theta + self.kalman_k * (z - self.kalman_h * theta)
    #     self.kalman_p = (1 - self.kalman_p*self.kalman_h) * p
    #     #print 'Kalman K', self.kalman_k, 'Kalman P', self.kalman_p




    # # def imu_callback(self, msg):
    #     # IMU data contains linear acceleration in m/s^2
    #     self._acceleration.x = msg.linear_acceleration.x
    #     self._acceleration.y = msg.linear_acceleration.y
    #     self._acceleration.z = msg.linear_acceleration.z - GRAVITATIONAL_ACCELERATION
    #     self.get_logger().debug("imu data received")

    #     dt : Duration = self._old_time - self.get_clock().now() 
    #     dt = dt.nanoseconds * 1e-9

    #     # Integrate velocity to obtain position
    #     self._position.x += self._velocity.x * dt
    #     self._position.y += self._velocity.y * dt
    #     self._position.z += self._velocity.z * dt

    #     # Integrate acceleration to obtain velocity
    #     self._velocity.x += self._acceleration.x * dt
    #     self._velocity.y += self._acceleration.y * dt
    #     self._velocity.z += self._acceleration.z * dt


    #     self._old_time = self.get_clock().now() 

    #     # Publish the position
    #     self._pub_imu_pose.publish(self._position)




    # state[0] and state[1]: x and y coordinates of the robot.
    # state[2]: orientation or yaw angle of the robot.
    # state[3]: longitudinal velocity of the robot.
    # state[4]: yaw rate (rate of change of yaw angle).
    # state[5]: longitudinal acceleration.

    # ret[0] and ret[1] are updated based on the current x, y, and yaw angle of the robot, using its velocity.
    # ret[2] is updated based on the current yaw angle and the yaw rate.
    # ret[3] is updated based on the current longitudinal velocity and longitudinal acceleration.
    # ret[4] remains unchanged, representing the yaw rate from the IMU.
    # ret[5] remains unchanged, representing the longitudinal acceleration from the IMU.

    # def predict(self, timestep, inputs=[]):
    #     """
    #     performs a prediction step
    #     :param timestep: float, amount of time since last prediction
    #     """

    #     self.lock.acquire()

    #     sigmas_out = np.array([self.iterate(x, timestep, inputs) for x in self.sigmas.T]).T

    #     x_out = np.zeros(self.n_dim)

    #     # for each variable in X
    #     for i in range(self.n_dim):
    #         # the mean of that variable is the sum of
    #         # the weighted values of that variable for each iterated sigma point
    #         x_out[i] = sum((self.mean_weights[j] * sigmas_out[i][j] for j in range(self.n_sig)))

    #     p_out = np.zeros((self.n_dim, self.n_dim))
    #     # for each sigma point
    #     for i in range(self.n_sig):
    #         # take the distance from the mean
    #         # make it a covariance by multiplying by the transpose
    #         # weight it using the calculated weighting factor
    #         # and sum
    #         diff = sigmas_out.T[i] - x_out
    #         diff = np.atleast_2d(diff)
    #         p_out += self.covar_weights[i] * np.dot(diff.T, diff)

    #     # add process noise
    #     p_out += timestep * self.q

    #     self.sigmas = sigmas_out
    #     self.x = x_out
    #     self.p = p_out

    #     self.lock.release()

        # print ("--------------------------------------------------------")
        # updating isn't bad either
        # create measurement noise covariance matrices

        # r_compass = np.zeros([1, 1]) # for 2
        # r_compass[0][0] = 0.02
        # r_encoder = np.zeros([2, 2]) #no longer 3 gives 0,1
        # r_encoder[0][0] = 0.001
        # r_encoder[1][1] = 0.001
        # remember that the updated states should be zero-indexed
        # the states should also be in the order of the noise and data matrices
        # imu_data = np.array([imu_yaw_rate, imu_accel])
        # compass_data = np.array([compass_hdg])
        # encoder_data = np.array([encoder_x, encoder_y])
        # state_estimator.update([4, 5], imu_data, r_imu) #  gyro ad accelero
        # state_estimator.update([2], compass_data, r_compass)  #theta
        # state_estimator.update([3], encoder_vel, r_encoder)
        # state_estimator.update([0, 1], encoder_data, r_encoder) # x, y
    



