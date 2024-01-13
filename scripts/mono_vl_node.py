#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2, math, sys, warnings
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped
import ament_index_python.packages

from datetime import datetime
import time, os, yaml
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

from viro_mono_vio.srv import MonoVio

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
# ros2 run mono_vo mono_vo_node.py











class MonoVLnode(Node):
    def __init__(self):
        super().__init__('mono_vl_node')

        # Declare and read parameters
        self.declare_parameter("marker_size", 50.0)
        self.declare_parameter("image_topic", '/kinect_camera/image_raw')
        self.declare_parameter("depth_image_topic", '/kinect_camera/depth/image_raw')
        self.declare_parameter("camera_info_topic", '/kinect_camera/camera_info')

        self.marker_size = self.get_parameter("marker_size").get_parameter_value().double_value
        image_raw_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        depth_image_topic = self.get_parameter("depth_image_topic").get_parameter_value().string_value
        camera_info = self.get_parameter("camera_info_topic").get_parameter_value().string_value

        # Make sure we have a valid dictionary id:
        self.dictionary_id = cv2.aruco.DICT_6X6_250  # DICT_4X4_50 | DICT_ARUCO_ORIGINAL

        self.bridge = CvBridge()

        self.aruco_x, self.aruco_y, self.aruco_th = 0.0, 0.0, 0.0
        self.start_time = datetime.now().second
        self.list__xk = []
        self.list__yk = []
        
        self.cv_image = None
        self.prev_keypoint = None
        self.prev_image = None
        self.pp = None # principal point of the camera #  (image_width / 2, image_height / 2).
        self.focal_length = None
        self.trajectory = [] 
        self.traj = np.zeros((600,600,3), dtype=np.uint8)
        plt.figure()

        self.marker_size = 13.5 # [-cm]  # should be metre so that the v_cam estimate can be in real world metres.
        self.current_detected_id = 'none'
        self.aruco_stat_time = time.time_ns()
        self.show_video = True

        # self.ids_to_find = {"ids":"x_bw, y_bw, t_bw",
        #                     "1":[1.4, 0.0, 180.0],    # -  tested: directly opposite agv  
        #                     "2":[9.0, -5.0, 135.0],   # \  tested:
        #                     "3":[10.5, -4.6, 180.0],  # /  tested: slanted at 45 degrees in agv view direction
        #                     "4":[1.0, 1.5, -90.0],    # |  tested: 90 degrees to agv view      
        #                     "5":[]}

        # Get package path
        package_path = ament_index_python.packages.get_package_share_directory('viro_mono_vio')
        self.config_path = os.path.join(package_path, "config")
        self.file_path = os.path.join(self.config_path, "mono_vl_aruco.yaml")

        self.ids_to_find = {}
        # Check if the file exists
        if os.path.exists(self.file_path):
            # Open the file
            with open(self.file_path, 'r') as file:
                try:
                    # Load the YAML data
                    data = yaml.safe_load(file)  
                    # Check if the data is not empty
                    if data:
                        # Access ar_tags
                        ar_tags = data.get('ar_tags', [])
                        # Check if ar_tags is not empty
                        if len(ar_tags) != 0:
                            print("ar_tags is not empty.")
                            for tag in ar_tags:
                                self.ids_to_find[str(int(tag[0]))] = [float(tag[1]), float(tag[2]), float(tag[3])]
                                print(" - ids_to_find - ", self.ids_to_find)
                        else:
                            print("ar_tags is empty.")
                    else:
                        print("The file is empty.")
                except yaml.YAMLError as error:
                    print("Error reading YAML file: ", error)
        else:
            print("The file does not exist.")
        # print('ids_to_find: ', self.ids_to_find)

        # Set up subscriptions
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            camera_info,
            self.camera_info_callback,
            10)

        self.image_subscription = self.create_subscription(
            Image,
            image_raw_topic,
            self.image_callback,
            10)

        self.depth_image_subscription = self.create_subscription(
            Image,
            depth_image_topic,
            self.depth_image_callback,
            10)

        # Set up publishers
        self.pose_pub = self.create_publisher(PoseStamped, 'mono_vl_pose', 10)

    # -----------------------------------------
    # -----------------------------------------  

    def image_callback(self, msg):
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8") # grayscale image | (msg, "bgr8") # coloured image
        # cv2.imshow("Depth Image Window", self.cv_image)
        # cv2.waitKey(1)

    # -----------------------------------------
    # -----------------------------------------  

    def plot_trajectory(self):
        # Unpack the trajectory points
        x, y = zip(*self.trajectory)
        # Clear the current figure's content
        plt.clf()
        # Create the plot
        plt.plot(x, y, 'ro-')
        plt.title('Robot Trajectory')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        # Draw the plot and pause for a short period
        plt.draw()
        plt.pause(0.001)


    # -----------------------------------------
    # -----------------------------------------

    def euler_to_quaternion(self, yaw, pitch, roll):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qx, qy, qz, qw]

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

    def quaternion_from_matrix(self, matrix):
        """Return quaternion from rotation matrix.

        >>> R = rotation_matrix(0.123, (1, 2, 3))
        >>> q = quaternion_from_matrix(R)
        >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
        True

        """
        q = np.empty((4, ), dtype=np.float64)
        M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
        t = np.trace(M)
        if t > M[3, 3]:
            q[3] = t
            q[2] = M[1, 0] - M[0, 1]
            q[1] = M[0, 2] - M[2, 0]
            q[0] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
        return q

    # -----------------------------------------
    # -----------------------------------------  

    def estimate_camera_prop(self, img1):
        # Get image dimensions
        height, width = img1.shape
        # Estimate principal point (assume it's at the center of the images)
        ppx = width / 2
        ppy = height / 2
        # Estimate focal length (assume a certain field of view)
        fov = 60  # field of view in degrees, adjust based on your camera specifications
        # fx = img1.shape[1] / (2 * np.tan(np.radians(fov / 2)))
        # fy = fx  # assume square pixels
        # If the image is not square, the focal length in the x and y directions might be different.
        # Here we assume that the field of view is specified for the diagonal of the image.
        diagonal = np.sqrt(width**2 + height**2)
        fx = fy = diagonal / (2 * np.tan(np.radians(fov / 2)))
        # Print estimated camera matrix
        print('Estimated camera matrix:')
        print(f'fx: {fx}, fy: {fy}, ppx: {ppx}, ppy: {ppy}')
        # print(np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]]))
        cam_mat = np.array([[fx, 0, ppx],
                            [0, fy, ppy],
                            [0, 0, 1]])
        # Define the distortion coefficients d
        distCoef = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        return cam_mat, distCoef

    # -----------------------------------------
    # -----------------------------------------

    def depth_image_callback(self, msg):
        cv_depthimage = self.bridge.imgmsg_to_cv2(msg, "32FC1")

        marker_dict = cv2.aruco.getPredefinedDictionary(self.dictionary_id)
        param_markers = cv2.aruco.DetectorParameters()  

        if self.cv_image is None:
            return
        
        if len(self.cv_image.shape) == 2:
            gray_image = self.cv_image # gray_image = cv2.flip(self.cv_image, 1)
            # print("The image is grayscale")
        else:
            flipped = cv2.flip(self.cv_image, 1)
            gray_image = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)     
            # print("The image is not grayscale")
         
        try:     

            cv2.line(gray_image, (280,0),(280,480),(255,255,0),1)
            cv2.line(gray_image, (400,0),(400,480),(255,255,0),1)
                  
            #########################################################
            #-------------------- ARUCO LOCALIZER ------------------#
            #########################################################

            marker_corners, marker_IDs, rejected = cv2.aruco.detectMarkers(
                image=gray_image,
                dictionary=marker_dict,
                parameters=param_markers)

            if marker_corners:
                self.get_logger().info('----------------------')  

                #camera to marker rotation and translation.
                ret = cv2.aruco.estimatePoseSingleMarkers(marker_corners, self.marker_size, self.camera_matrix, self.dist_coeff)
                rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]

                #-- Draw the detected marker and put a reference frame over it
                cv2.aruco.drawDetectedMarkers(gray_image, marker_corners)
                cv2.drawFrameAxes(gray_image, self.camera_matrix, self.dist_coeff, rvec, tvec, 10)

                #-- Obtain the rotation matrix tag->camera
                R_ct = np.matrix(cv2.Rodrigues(rvec)[0])  # cv2.Rodrigues(rvec)
                R_tc = R_ct.T

                #-- Get the attitude in terms of euler 321 (Needs to be flipped first)
                roll_marker, pitch_marker, yaw_marker = self._rotationMatrixToEulerAngles(R_tc) # self._R_flip*R_tc
                # print(' euler: ', 'r:', roll_marker, 'p:',  pitch_marker, 'y:',  yaw_marker)

                total_markers = range(0, marker_IDs.size)
                for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):

                    corners = corners.reshape(4, 2)
                    corners = corners.astype(int)

                    M = cv2.moments(corners)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])                
                    area = round(int(cv2.arcLength(corners,True))/self.marker_size, 1)
                    # print('cx, cy: ', cx, cy, area )

                    # Get the depth value at (cx, cy)
                    depth_in_meters = cv_depthimage[cy, cx]
                    # Calculate in real-world camera coordinates | self.camera_matrix = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]]) | intrinsic matrix
                    x = (cx - self.camera_matrix[0, 2]) * depth_in_meters / self.camera_matrix[0, 0] 
                    y = (cy - self.camera_matrix[1, 2]) * depth_in_meters / self.camera_matrix[1, 1]
                    z = depth_in_meters
                    v_cam = np.array([x, y, z])  # z is forward though, in camera frame
                    # print("Camera Coordinate System: x=%.3lf, y=%.3lf, z=%.3lf\n" % (z, x, y))  

                    # ideally i do not have to use a depth camera for this and i could use the below code to estimate 
                    # the relative position of the camera with the qr marker. this only works if marker size is specified in metres/real world units.
                    # say we dont have a depth camera and we just want to estimate x, y from the image itself without real world x y
                    # -- Now get Position and attitude of the camera respect to the marker
                    # v_cam = -R_tc*np.matrix(tvec).T # rotate and translate the position of the marker (given by tvec) from the camera frame to the marker frame
                    # print("Camera Coordinate System: x=%.3lf, y=%.3lf, z=%.3lf\n" % (v_cam[2], v_cam[0], v_cam[1]))  

                    self.aruco_x, self.aruco_y, self.aruco_th = self.estimate_pose(ids, yaw_marker, v_cam)
                    if self.aruco_x != None and self.aruco_y != None and self.aruco_th != None:
                        # define region of interest 
                        if (210 < cx < 550) and (3.7 < area < 75.8): 
                            # print("id: "+str(ids[0])+": measured cx "+str(cx)+", cy "+str(cy)+" and area "+str(area)+", within good visible range. node will publish robot's estimated location.") 
                            print("id: "+str(ids[0])+": x "+str(self.aruco_x)+", y "+str(self.aruco_y)+" and th "+str(self.aruco_th))
                            msg = PoseStamped()
                            msg.header.frame_id = "marker_"+str(ids[0])
                            msg.header.stamp = self.get_clock().now().to_msg()
                            msg.pose.position.x = self.aruco_x
                            msg.pose.position.y = self.aruco_y
                            msg.pose.position.z = 0.0
                            quat = self.euler_to_quaternion(self.aruco_th, 0, 0)
                            msg.pose.orientation.x = quat[0]
                            msg.pose.orientation.y = quat[1]
                            msg.pose.orientation.z = quat[2]
                            msg.pose.orientation.w = quat[3]
                            self.pose_pub.publish(msg)
                        else:
                            pass
   
                    cv2.putText(gray_image, f"Area: {area}", (cx-120,cy-80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,255), 1)
                    cv2.putText(gray_image,f"Dist: {depth_in_meters}",(cx,cy),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,150,255),2)
                    cv2.putText(gray_image, f"id: {ids[0]}", (30,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 2)
     
            #########################################################
            #-------------------------------------------------------#
            #########################################################
    
            if self.show_video: # Display the frame: Show images 
                # cv2.imshow("Depth Image Window", cv_depthimage)
                cv2.imshow("Image Window", self.cv_image)
                #--- use 'q' to quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()

        except RuntimeError:
            self.get_logger().info('13 i got here')
            return

# -----------------------------------------
# -----------------------------------------

    def estimate_pose(self, ids, yaw_marker, v_cam):
        try: 
            # self.get_logger().info('str(ids[0]): '+str(ids[0]))
            if (str(ids[0]) in self.ids_to_find): # and (str(ids[0]) != self.current_detected_id):      
                coordinates = self.ids_to_find[str(ids[0])]
                x_bw = coordinates[0]
                y_bw = coordinates[1]
                t_bw = math.radians(coordinates[2])
                if (str(ids[0]) != self.current_detected_id): 
                    self.start_time = datetime.now().second
                    self.current_detected_id = str(ids[0])
                    self.list__xk = []
                    self.list__yk = []

                # rotation matrix 
                t_map_qr = [[math.cos(t_bw), -math.sin(t_bw), x_bw],
                            [math.sin(t_bw),  math.cos(t_bw), y_bw],
                            [0, 0, 1]] 
                # self.get_logger().info('1 t_map_qr: '+str(t_map_qr))
            
                rot_qr_cam = [[math.cos(yaw_marker), -math.sin(yaw_marker), 0],
                                [math.sin(yaw_marker), math.cos(yaw_marker), 0],
                                [0, 0, 1]] 
                # self.get_logger().info('1 rot_qr_cam: '+str(rot_qr_cam))
            
                trans_qr_cam = [[1, 0, v_cam[2]],
                                [0, 1, v_cam[0]], 
                                [0, 0, 1]] 
                # self.get_logger().info('1 trans_qr_cam: '+str(trans_qr_cam))
            
                t_map_cam = np.dot(np.dot(t_map_qr, rot_qr_cam), trans_qr_cam) 
                x_map_cam, y_map_cam, yaw_map_cam = t_map_cam[0][2], t_map_cam[1][2], math.atan2(t_map_cam[1][0], t_map_cam[0][0]); 
                # print('x_', x_map_cam, 'y_', y_map_cam, 't_', yaw_map_cam)

                self.list__xk.append(x_map_cam)
                self.list__yk.append(y_map_cam)

                if (0.9 < abs(datetime.now().second - self.start_time)):
                    
                    self.current_detected_id = 'none'

                    # Finding the IQR
                    percentile25x = np.quantile(self.list__xk, 0.25) 
                    percentile75x = np.quantile(self.list__xk, 0.75) 

                    percentile25y = np.quantile(self.list__yk, 0.25) 
                    percentile75y = np.quantile(self.list__yk, 0.75) 

                    # Finding upper and lower limit
                    iqr_x = percentile75x - percentile25x
                    upper_limitx = percentile75x + 1.5 * iqr_x
                    lower_limitx = percentile25x - 1.5 * iqr_x

                    iqr_y = percentile75y - percentile25y
                    upper_limity = percentile75y + 1.5 * iqr_y
                    lower_limity = percentile25y - 1.5 * iqr_y

                    # Trimming
                    inlier_list_x, inlier_list_y = [], []
                    for i in range(len(self.list__xk)):
                        if (self.list__xk[i] > lower_limitx) and (self.list__xk[i] < upper_limitx): 
                            inlier_list_x.append(self.list__xk[i])
                    mean_x = sum(inlier_list_x) / len(inlier_list_x)
                    # self.get_logger().info('1: mean_x '+str(mean_x))

                    for i in range(len(self.list__yk)):
                        if (self.list__yk[i] > lower_limity) and (self.list__yk[i] < upper_limity):
                            inlier_list_y.append(self.list__yk[i])
                    mean_y = sum(inlier_list_y) / len(inlier_list_y)
                    # self.get_logger().info('1: mean_y '+str(mean_y))

                    # Reset the lists to use them again
                    self.list__xk = []
                    self.list__yk = []

                    return float(mean_x), float(mean_y), float(yaw_map_cam)   
                return None, None, None 

            else:
                # self.get_logger().info('- 1 ')
                return None, None, None # print("Nothing detected.")
            
        except ZeroDivisionError:
            # self.get_logger().info('- 2 ')
            return None, None, None # continue
        
        except ValueError:
            # self.get_logger().info('- 3 ')
            return None, None, None # continue         
  
    # -----------------------------------------
    # -----------------------------------------  

    def camera_info_callback(self, msg):
        # self.get_logger().info('Camera Info: %s' % msg)
        # Focal lengths in pixel coordinates
        fx = msg.k[0]  # Focal length in x direction
        fy = msg.k[4]  # Focal length in y direction
        # Principal point
        ppx = msg.k[2]  # x coordinate of principal point
        ppy = msg.k[5]  # y coordinate of principal point
        # Distortion coefficients
        self.dist_coeff = np.array(msg.d)  # Distortion coefficients
        # Image dimensions
        # image_height = msg.height  # Image height
        # image_width = msg.width  # Image width
        # self.get_logger().info('Camera Info: fx=%f, fy=%f, \
        #                        ppx=%f, ppy=%f, distortion_coeff=%s, \
        #                        image_height=%d, image_width=%d' \
        #                        % (fx, fy, ppx, ppy, distortion_coeff, image_height, image_width))
        # if self.focal_length is None and self.pp is None:
        self.camera_matrix = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
        # self.pp = (ppx, ppy) # (ppx + ppy) / 2 # principal point of the camera #  (image_width / 2, image_height / 2).
        # self.focal_length = (fx + fy) / 2 
        self.destroy_subscription(self.camera_info_subscription)

    # -----------------------------------------
    # -----------------------------------------  

def main(args=None):
    rclpy.init(args=args)
    try: 
        mono_vl_subscriber = MonoVLnode()
        rclpy.spin(mono_vl_subscriber)
    except:   # except SystemExit: # <- process the exception 
        mono_vl_subscriber.destroy_node()  #    rclpy.logging.get_logger("Route_pub").info('Exited')
        rclpy.shutdown()


if __name__ == '__main__':
    main()


























# device_serial_no = '843112072968'
        # if device_serial_no != 'none':
        #     ctx = rs.context()
        #     if len(ctx.devices) > 0:
        #         for d in ctx.devices:
        #             print ('Found device: ', \
        #                     d.get_info(rs.camera_info.name), ' ', \
        #                     d.get_info(rs.camera_info.serial_number))

        #             if device_serial_no != d.get_info(rs.camera_info.serial_number):

        #                 show_video = True

        #                 # define pipeline
        #                 pipeline = rs.pipeline()
        #                 config = rs.config()

        #                 config.enable_device(d.get_info(rs.camera_info.serial_number)) 

        #                 config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        #                 config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
                        
        #                 # Start streaming
        #                 pipeline.start(config)

        #                 self.rs_aruco_pose(pipeline, marker_size, show_video)

        #     else:
        #         print("No Intel Device connected")
        # else:
        #     self.generic_aruco_pose(pipeline, marker_size, show_video)


        # except RuntimeError:
        #     print("Azeez: Frame didn't arrive within 5000")
        #     ctx = rs.context()
        #     devices = ctx.query_devices()
        #     for dev in devices:
        #         dev.hardware_reset()
        #     self.rs_aruco_pose(pipeline, marker_size, show_video)
        #     time.sleep(1)
        #     return

# import spatial
# def removeDuplicates(points, threshold=30):
#     """Remove duplicate points that are within a certain Euclidean distance from each other."""
#     tree = spatial.KDTree(points)
#     groups = list(tree.query_ball_tree(tree, threshold))
#     new_points = []
#     for group in groups:
#         new_points.append(np.mean(points[group], axis=0))
#     return np.array(new_points)


# def deRotatePatch(patch):
#     # Compute the Harris matrix
#     harris_matrix = cv2.cornerHarris(patch, 2, 3, 0.04)
#     # Compute the eigenvalues and eigenvectors of the Harris matrix
#     _, _, eigenvectors = np.linalg.svd(harris_matrix)
#     # Find the direction of the most dominant gradient
#     dominant_direction = eigenvectors[0]
#     # Compute the angle to rotate the patch
#     angle = np.arctan2(dominant_direction[1], dominant_direction[0])
#     # Create a rotation matrix
#     rotation_matrix = cv2.getRotationMatrix2D((patch.shape[1]/2, patch.shape[0]/2), angle, 1)
#     # De-rotate the patch
#     de_rotated_patch = cv2.warpAffine(patch, rotation_matrix, (patch.shape[1], patch.shape[0]), flags=cv2.INTER_CUBIC)
#     return de_rotated_patch
    
        # self.T_vectors.append(tuple([[0], [0], [0]]))
        # self.R_matrices.append(tuple(np.zeros((3, 3))))
    
# self.tracker_type = 'Farneback' # 'Farneback' | 'KLT'
# def Farneback_featureTracking(prev_img, cur_img, prev_pts):
#     farneback_params = dict(pyr_scale=0.5, levels=5, winsize=13, iterations=10, poly_n=5, poly_sigma=1.1, flags=0)
#     flow = cv2.calcOpticalFlowFarneback(prev_img, cur_img, None, **farneback_params)
#     indices_yx = prev_pts.astype(int)
#     valid_indices = (
#         (indices_yx[:, 0] >= 0) & (indices_yx[:, 0] < flow.shape[0]) & (indices_yx[:, 1] >= 0) & (indices_yx[:, 1] < flow.shape[1])
#     )
#     flow_pts = flow[indices_yx[valid_indices, 0], indices_yx[valid_indices, 1]]
#     cur_pts = prev_pts[valid_indices] + flow_pts
#     px_diff = np.linalg.norm(prev_pts[valid_indices] - cur_pts, axis=1)
#     diff_mean = np.mean(px_diff)
#     return prev_pts[valid_indices], cur_pts, diff_mean

        # if self.tracker_type == 'KLT':
        #     self.px_ref, self.px_cur, px_diff = KLT_featureTracking(prev_img, cur_img, self.px_ref)
        # elif self.tracker_type == 'Farneback':
        #      self.px_ref, self.px_cur, px_diff = Farneback_featureTracking(prev_img, cur_img, self.px_ref)
        # else:
        #     raise ValueError(f"Unknown tracker type: {self.tracker_type}")
    
# def matching(matcher_type, description1, description2):
#     if matcher_type == 'BFMATCHER':
#         bf = cv2.BFMatcher()
#         matches = bf.knnMatch(description1, description2, k=2)
#     elif matcher_type == 'FLANN':
#         FLANN_INDEX_KDTREE = 1
#         index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#         search_params = dict(checks=50)
#         flann = cv2.FlannBasedMatcher(index_params, search_params)
#         matches = flann.knnMatch(description1, description2, k=2)
#     else:
#         raise ValueError(f"Unknown matcher type: {matcher_type}")
#     return matches
