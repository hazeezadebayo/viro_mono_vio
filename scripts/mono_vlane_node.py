#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from viro_mono_vio.srv import MonoVio



# ros2 run viro_mono_vio mono_lane_detection.py



class LaneDetectionModule(Node):
    def __init__(self):
        super().__init__('mono_lane_detection_node')

        self.declare_parameter('image_raw_topic', '/kinect_camera/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        image_raw_topic = self.get_parameter('image_raw_topic').get_parameter_value().string_value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value 

        self.curve_list = []
        self.avg_val = 10
        self.display = 1

        self.start_lane_detection = True # False

        self.angular_velocity = 0.02
        self.linear_velocity = 0.05

        # self.initial_trackbar_vals = [90, 115, 40, 220]
        self.initial_trackbar_vals = [127, 210, 0, 240]  

        cv2.namedWindow('Trackbars')
        cv2.resizeWindow('Trackbars', 360, 240)
        cv2.createTrackbar('Width Top', 'Trackbars', self.initial_trackbar_vals[0], 480 // 2, self.nothing)
        cv2.createTrackbar('Height Top', 'Trackbars', self.initial_trackbar_vals[1], 240, self.nothing)
        cv2.createTrackbar('Width Bottom', 'Trackbars', self.initial_trackbar_vals[2], 480 // 2, self.nothing)
        cv2.createTrackbar('Height Bottom', 'Trackbars', self.initial_trackbar_vals[3], 240, self.nothing)

        self.image_subscription = self.create_subscription(
            Image,
            image_raw_topic,
            self.image_callback,
            10)
        self.bridge = CvBridge()
        
        # Declare a publisher for cmd_vel: | vx: meters per second | w: radians per second 
        # publish Twist messages for navigation
        self.cmd_vel_msg = Twist()
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        # Declare services: save pre-dock/dock pose to yaml service, dock or undock robot, dock status
        self.start_follower_service = self.create_service(MonoVio, '/mono_vio/lane_detection/switch', self.lane_service_cb)




    def lane_service_cb(self, request, response):
        # int8 tag_id
        # int8 switch_no
        # ---
        # bool status
        # string message
        switch = request.switch_no
        if switch == 1:
            self.start_lane_detection = True
            response.status = True
            response.message = "dock service call successful."
        elif switch == 0:
            self.start_lane_detection = False
            response.status = True
            response.message = "dock service call successful."
        else:
            response.status = False
            response.message = "invalid service call request."
        # self.get_logger().info(str()+" : started!") 
        return response


    def get_lane_curve(self, img, display=2):
        img_copy = img.copy()
        img_result = img.copy()

        # Step 1
        img_thres = self.thresholding(img)
    
        # Step 2
        hT, wT, c = img.shape
        points = self.val_trackbar(w=wT, h=hT)
        img_warp = self.warp_img(img_thres, points, wT, hT)
        img_warp_points = self.draw_points(img_copy, points)

        # Step 3
        middle_point, img_hist = self.get_histogram(img_warp, display=True, min_per=0.5, region=4)
        curve_average_point, img_hist = self.get_histogram(img_warp, display=True, min_per=0.9)
        curve_raw = curve_average_point - middle_point

        # Step 4
        self.curve_list.append(curve_raw)
        if len(self.curve_list) > self.avg_val:
            self.curve_list.pop(0)
        curve = int(sum(self.curve_list) / len(self.curve_list))

        # Step 5
        if display != 0:
            img_inv_warp = self.warp_img(img_warp, points, wT, hT, inv=True)
            img_inv_warp = cv2.cvtColor(img_inv_warp, cv2.COLOR_GRAY2BGR)
            img_inv_warp[0:hT // 3, 0:wT] = 0, 0, 0
            img_lane_color = np.zeros_like(img)
            img_lane_color[:] = 0, 255, 0
            img_lane_color = cv2.bitwise_and(img_inv_warp, img_lane_color)
            img_result = cv2.addWeighted(img_result, 1, img_lane_color, 1, 0)
            midY = 450
            cv2.putText(img_result, str(curve), (wT // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
            cv2.line(img_result, (wT // 2, midY), (wT // 2 + (curve * 3), midY), (255, 0, 255), 5)
            cv2.line(img_result, ((wT // 2 + (curve * 3)), midY - 25), (wT // 2 + (curve * 3), midY + 25), (0, 255, 0), 5)
            for x in range(-30, 30):
                w = wT // 20
                cv2.line(img_result, (w * x + int(curve // 50), midY - 10),
                         (w * x + int(curve // 50), midY + 10), (0, 0, 255), 2)
        if display == 1:
            img_stacked = self.stack_images(0.7, ([img, img_warp_points, img_warp],
                                                  [img_hist, img_lane_color, img_result]))
            cv2.imshow('ImageStack', img_stacked)
        elif display == 2:
            cv2.imshow('Result', img_result)

        # Normalization
        curve = curve / 100
        if curve > 1:
            curve = 1
        if curve < -1:
            curve = -1
        return curve


    @staticmethod
    def thresholding(img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 102])
        upper_white = np.array([179, 255, 255])
        mask_white = cv2.inRange(img_hsv, lower_white, upper_white)
        return mask_white


    @staticmethod
    def warp_img(img, points, w, h, inv=False):
        pts1 = np.float32(points)
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        if inv:
            matrix = cv2.getPerspectiveTransform(pts2, pts1)
        else:
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp = cv2.warpPerspective(img, matrix, (w, h))
        return img_warp


    @staticmethod
    def draw_points(img, points):
        for x in range(4):
            cv2.circle(img, (int(points[x][0]), int(points[x][1])), 15, (0, 0, 255), cv2.FILLED)
        return img


    @staticmethod
    def get_histogram(img, min_per=0.1, display=False, region=1):
        if region == 1:
            hist_values = np.sum(img, axis=0)
        else:
            hist_values = np.sum(img[img.shape[0] // region:, :], axis=0)
        max_value = np.max(hist_values)
        min_value = min_per * max_value
        index_array = np.where(hist_values <= min_value)
        base_point = int(np.average(index_array))
        if display:
            img_hist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
            for x, intensity in enumerate(hist_values):
                cv2.line(img_hist, (x, img.shape[0]), (x, int(img.shape[0] - intensity // 255 // region)),
                         (255, 0, 255), 1)
                cv2.circle(img_hist, (base_point, img.shape[0]), 15, (0, 255, 255), cv2.FILLED)
            return base_point, img_hist
        return base_point


    @staticmethod
    def stack_images(scale, img_array):
        rows = len(img_array)
        cols = len(img_array[0])
        rows_available = isinstance(img_array[0], list)
        width = img_array[0][0].shape[1]
        height = img_array[0][0].shape[0]
        if rows_available:
            for x in range(0, rows):
                for y in range(0, cols):
                    if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                        img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                    else:
                        img_array[x][y] = cv2.resize(img_array[x][y],
                                                    (img_array[0][0].shape[1], img_array[0][0].shape[0]), None,
                                                    scale, scale)
                    if len(img_array[x][y].shape) == 2: img_array[x][y] = cv2.cvtColor(img_array[x][y],
                                                                                      cv2.COLOR_GRAY2BGR)
            image_blank = np.zeros((height, width, 3), np.uint8)
            hor = [image_blank] * rows
            hor_con = [image_blank] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(img_array[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if img_array[x].shape[:2] == img_array[0].shape[:2]:
                    img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
                else:
                    img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None,
                                              scale, scale)
                if len(img_array[x].shape) == 2: img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
            hor = np.hstack(img_array)
            ver = hor
        return ver


    @staticmethod
    def nothing(a):
        pass

    def val_trackbar(self, w=480, h=240):
        width_top = cv2.getTrackbarPos('Width Top', 'Trackbars')
        heighth_top = cv2.getTrackbarPos('Height Top', 'Trackbars')
        width_bottom = cv2.getTrackbarPos('Width Bottom', 'Trackbars')
        height_bottom = cv2.getTrackbarPos('Height Bottom', 'Trackbars')
        points = np.float32([(width_top, heighth_top), (w - width_top, heighth_top),
                             (width_bottom, height_bottom), (w - width_bottom, height_bottom)])
        return points


    def region_of_interest(self, img, vertices):
        # Create a mask to keep only the region of interest
        mask = np.zeros_like(img)
        match_mask_color = 255
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image


    def draw_lines(self, img, lines):
        # Create a blank image to draw lines on
        img = np.copy(img)
        blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        # Draw lines on the blank image
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)
        # Combine the original image with the image containing drawn lines
        img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
        return img


    def process(self, image, display=2):
        # Get image dimensions
        height, width, _ = image.shape
        # Define the region of interest (ROI) vertices
        region_of_interest_vertices = [
            (0, height),
            (width / 2, height / 2),
            (width, height) ]
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Apply Canny edge detection
        canny_image = cv2.Canny(gray_image, 100, 50)
        # Extract the region of interest using the defined vertices
        cropped_image = self.region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))
        # Apply HoughLinesP transform to detect lines
        lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi / 180, threshold=50,
                                lines=np.array([]), minLineLength=40, maxLineGap=100)
        # Draw lines on the original image
        image_with_lines = self.draw_lines(image, lines)
        # Display the processed image if requested
        if display == 2 or display == 1:
            cv2.imshow('Processed Video', image_with_lines)
        return image_with_lines



    # -----------------------------------------
    # -----------------------------------------  

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8") # coloured image
        # cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")  # grayscale image
        if self.start_lane_detection == True:
            try:
                img = cv2.resize(cv_image, (480, 240)) # cv_image
                height, width, _ = img.shape
                self.get_logger().info("\n height: "+str(height)+" | width: "+str(height))
                image_w_lines = self.process(img, self.display)
                curve_cmd = self.get_lane_curve(img, self.display)
                self.get_logger().info("\n curve: "+str(curve_cmd)) 

                self.min_curve_thresh = 0.02

                # [ [:-0.02]-ve |    0    | +ve [0.02:] ]
                if curve_cmd > self.min_curve_thresh + 0.04:
                    self.cmd_vel_msg.linear.x  =  1 * self.linear_velocity
                    self.cmd_vel_msg.angular.z = -4 * self.angular_velocity 

                elif curve_cmd > self.min_curve_thresh:
                    self.cmd_vel_msg.linear.x  = self.linear_velocity/2
                    self.cmd_vel_msg.angular.z = -1 * self.angular_velocity 

                elif curve_cmd < -1 * self.min_curve_thresh:
                    self.cmd_vel_msg.linear.x  = self.linear_velocity/2
                    self.cmd_vel_msg.angular.z = self.angular_velocity 

                elif curve_cmd < -1 * (self.min_curve_thresh + 0.04):
                    self.cmd_vel_msg.linear.x  = 1 * self.linear_velocity
                    self.cmd_vel_msg.angular.z = 4 * self.angular_velocity 

                else:
                    self.cmd_vel_msg.linear.x = self.linear_velocity
                    self.cmd_vel_msg.angular.z = 0.0

                self.cmd_vel_pub.publish(self.cmd_vel_msg) # Publish the velocity message | vx: meters per second | w: radians per second
                
            except ValueError:
                self.cmd_vel_msg.linear.x = 0.0
                self.cmd_vel_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(self.cmd_vel_msg) # Publish the velocity message | vx: meters per second | w: radians per second
                
                print("Line could not be detected!")

            if self.display == 2 or self.display == 1:
                cv2.imshow('Image Window', cv_image)
                cv2.waitKey(1)

    # -----------------------------------------
    # -----------------------------------------  

         

 
    # -----------------------------------------
    # -----------------------------------------  

def main(args=None):
    rclpy.init(args=args)

    mono_lane_detection_node = LaneDetectionModule()
    rclpy.spin(mono_lane_detection_node)
    mono_lane_detection_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()















        
# if __name__ == '__main__':
#     cap = cv2.VideoCapture('/home/hazeezadebayo/viro_ws/src/viro_mono_vio/config/video.mp4')
#     lane_detection = LaneDetectionModule()
#     frame_counter = 0
#     display = 2

#     while True:
#         try:
#             frame_counter += 1
#             if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frame_counter:
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                 frame_counter = 0
#             success, img = cap.read()
#             if img is not None:
#                 img = cv2.resize(img, (480, 240))
#                 image_w_lines = lane_detection.process(img, display)
#                 curve = lane_detection.get_lane_curve(img, display)
#                 print("\n curve: ", curve)
#         except ValueError:
#             print("Line could not be detected!")

#         cv2.imshow('Vid', img)
#         cv2.waitKey(1)

# .......................................
# understanding HSV 
# .......................................

# def empty(a):
#     pass 

# cv2.namedWindow("HSV")
# cv2.resizeWindow('HSV',640,240)
# cv2.createTrackbar("HUE Min", "HSV",0,179,empty)
# cv2.createTrackbar("HUE Max", "HSV",179,179,empty)
# cv2.createTrackbar("SAT Min", "HSV",10,255,empty)
# cv2.createTrackbar("SAT Max", "HSV",25,255,empty)
# cv2.createTrackbar("VALUE Min", "HSV",0,255,empty)
# cv2.createTrackbar("VALUE Max", "HSV",255,255,empty)

# cap = cv2.VideoCapture('video.mp4')

# while True:

#     _,img = cap.read()
#     imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
#     h_min = cv2.getTrackbarPos("HUE Min","HSV")
#     h_max = cv2.getTrackbarPos("HUE Max","HSV")
#     s_min = cv2.getTrackbarPos("SAT Min","HSV")
#     s_max = cv2.getTrackbarPos("SAT Max","HSV")
#     v_min = cv2.getTrackbarPos("VALUE Min","HSV")
#     v_max = cv2.getTrackbarPos("VALUE Max","HSV")
#     lower = np.array([h_min,s_min,v_min])
#     upper = np.array([h_max,s_max,v_max])
#     mask = cv2.inRange(imgHsv,lower,upper)
#     result = cv2.bitwise_and(img,img, mask = mask)
#     mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
#     hStack = np.hstack([img,mask,result])
#     #cv2.imshow('Orginal',img)
#     #cv2.imshow('HSV Colo Space', imgHsv)
#     #cv2.imshow('Mask',mask)
#     cv2.imshow('Horizantal Stacking',hStack)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
        
# .......................................








