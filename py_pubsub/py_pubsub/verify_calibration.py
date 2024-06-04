import time
import rclpy
from enum import Enum
from threading import Thread
from rclpy.node import Node
from rclpy.task import Future
from typing import NamedTuple
from geometry_msgs.msg import Twist

from hiwin_interfaces.srv import RobotCommand

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

import pyrealsense2 as rs
import numpy as np
import cv2

MY_BASE = 12
CUE_TOOL = 12
CAM_TOOL = 11
CALI_TOOL = 9

DEFAULT_VELOCITY = 10
DEFAULT_ACCELERATION = 10

class States(Enum):
    INIT = 0
    FINISH = 1
    MOVE_TO_PHOTO_POSE = 2
    TAKE_PHOTO = 3
    OUTPUT_CLICKPOINT = 4
    MOVE_TO_CLICKPOINT = 5
    MOVE_TO_PLACE_POSE = 6
    CHECK_POSE = 7
    CLOSE_ROBOT = 8
    WAITING = 9

# 24.127
# [0.608,-0.009,-89.183]
# fix_abs_cam = [313.5, 152.0, -452.751, 0.0, 0.0, 0.0]
# fix_abs_cam = [302.471, 164.051, -467.763, 1.086, 0.0, 0.136]
fix_abs_cam = [298.557, 157.735, -456.251, 1.417, -0.513, 0.665]
cam_to_cue = [-86.943, 33.0, 105.5, 0.0, 0.0, 0.0]

click_event_list = [[]*2]
calibrated_intrinsics_f = [1362.38, 1360.45]
calibrated_intrinsics_pp = [938.315, 552.935]

def on_move(event):
    if event.inaxes:
        print(f'data coords {event.xdata} {event.ydata},',
              f'pixel coords {event.x} {event.y}')

def save_on_click(event):
    if event.button is MouseButton.LEFT:
        print(f'data coords {event.xdata} {event.ydata},',
              f'pixel coords {event.x} {event.y}')
        print("click saved")
        #save xdata and ydate to list
        click_event_list.append([event.xdata, event.ydata])

def click():
    img = plt.imread('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/verify_pics.jpg')
    plt.imshow(img)

    # binding_id = plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', save_on_click)
    plt.show()
    #delete first index of list cuz it's 0 somehow
    del click_event_list[0]
    print(click_event_list)
    #convert list to numpy array
    # click_event = np.array(click_event_list)
    # round_clickpoint = np.around(click_event, decimals=0)
    # print('clicked point\n',click_event)
    # print(round_clickpoint)

    return click_event_list

def pixel_mm_convert(pixel):
    actuallengh = 626
    pixellengh = 1920
    mm = actuallengh/pixellengh*pixel
    return mm

def take_pics():
    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    a = 0
    # Start streaming
    pipeline.start(config)

    # Instructions for user
    print('Press m to take pictures')
    print('Press q to quit camera\n')
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        # if not color_frame:
        #     continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Draw pool table outline
        start_outline_top = (0, 74)
        end_outline_top = (1920, 74)
        start_outline_bot = (0, 1006)
        end_outline_bot = (1920, 1006)
        table_outline_top  = cv2.line(color_image,start_outline_top,end_outline_top,color=(0,0,0),thickness=2)
        table_outline_bot = cv2.line(table_outline_top,start_outline_bot,end_outline_bot,color=(0,0,0),thickness=2)  

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', table_outline_bot)
        key=cv2.waitKey(1)
        if key&0xFF==ord('m'):
            a=a+1
            print("picture take")
            cv2.imwrite('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/verify_pics.jpg',table_outline_bot)
        if key&0xFF==ord('q'):
            cv2.destroyAllWindows()
            break

    # Stop streaming
    pipeline.stop()

def realsense_intrinsics(x, y):
    width = 1920
    height = 1080
    fps = 30
    depth = 1
    
    calibrated_intrinsics_f = [1362.38, 1360.45]
    calibrated_intrinsics_pp = [938.315, 552.935]

    dis_coeffs = [0.0693826933, 0.445315521, 0.00291064076, -0.000845071017, -1.99098719]

    _intrinsics = rs.intrinsics()
    _intrinsics.width = width
    _intrinsics.height = height
    _intrinsics.ppx = calibrated_intrinsics_pp[0]
    _intrinsics.ppy = calibrated_intrinsics_pp[1]
    _intrinsics.fx = calibrated_intrinsics_f[0]
    _intrinsics.fy = calibrated_intrinsics_f[1]
    #_intrinsics.model = cameraInfo.distortion_model
    _intrinsics.model  = rs.distortion.none
    _intrinsics.coeffs = dis_coeffs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    # config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    # ストリーミング開始

    profile = pipeline.start(config)
    # depth_intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.depth)).get_intrinsics()
    color_intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.color)).get_intrinsics()

    pixel = [x, y]
    ca_point = rs.rs2_deproject_pixel_to_point(color_intrinsics, pixel, depth)
    # print('calibrated point:',ca_point)

    x_ = int(ca_point[0] * calibrated_intrinsics_f[0] + calibrated_intrinsics_pp[0])
    y_ = int(ca_point[1] * calibrated_intrinsics_f[1] + calibrated_intrinsics_pp[1])
    # print('calibrated intrinsics x:',x_)
    # print('calibrated intrinsics y:',y_)

    pipeline.stop()
    return x_, y_
   
class VefifyCalibration(Node):
    def __init__(self):
        super().__init__('verify_calibration')
        self.hiwin_client = self.create_client(RobotCommand, 'hiwinmodbus_service')
        self.object_pose = None
        self.object_cnt = 0
        self.fix_campoint = Twist()
        self.i = 0
        self.click_point = [[],[]]
        self.intrin_clickpoint = []
        self.pointnum = 0

    def calibration_state(self, state: States) -> States:
        if state == States.INIT:
            self.get_logger().info('Moving to calculated camera point')
            pose = Twist()
            [pose.linear.x, pose.linear.y, pose.linear.z] = fix_abs_cam[0:3]
            [pose.angular.x, pose.angular.y, pose.angular.z] = fix_abs_cam[3:6]
            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.PTP,
                tool = CAM_TOOL,
                pose = pose
                )
            res = self.call_hiwin(req)
            if res.arm_state == RobotCommand.Response.IDLE:
                nest_state = States.WAITING
            else:
                nest_state = None

        elif state == States.WAITING:
            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.WAITING
            )
            res = self.call_hiwin(req)
            if res.arm_state == RobotCommand.Response.IDLE:
                nest_state = States.TAKE_PHOTO
            else:
                nest_state = None

        # if state == States.INIT:
        #     self.get_logger().info('Reading arm current position')
        #     req = self.generate_robot_request(
        #         cmd_mode=RobotCommand.Request.CHECK_POSE)
        #     res = self.call_hiwin(req)
        #     self.fix_campoint = res.current_position
        #     print(res.current_position)
        #     nest_state = States.TAKE_PHOTO
        
        elif state == States.TAKE_PHOTO:
            self.get_logger().info('Taking photo')
            take_pics()
            time.sleep(1.0)
            nest_state = States.OUTPUT_CLICKPOINT

        elif state == States.OUTPUT_CLICKPOINT:
            self.get_logger().info('Input clickpoint for robot arm')
            self.click_point = click()
            print('click point\n',self.click_point)
            self.pointnum = len(self.click_point)
            #################
            for i in range(0,self.pointnum):
                realx, realy = realsense_intrinsics(self.click_point[i][0], self.click_point[i][1])
                temp = [realx, realy]
                self.intrin_clickpoint.append(temp)
            print('real click point\n', self.intrin_clickpoint)
            #################    
            nest_state = States.MOVE_TO_CLICKPOINT
        
        elif state == States.MOVE_TO_CLICKPOINT:
            self.get_logger().info('Moving to click point')
            pose = Twist()
            x = pixel_mm_convert(self.click_point[self.i][0])
            y = pixel_mm_convert(self.click_point[self.i][1]-83)
            [pose.linear.x, pose.linear.y, pose.linear.z] = [x, y, -10.0]
            [pose.angular.x, pose.angular.y] = fix_abs_cam[3:5]
            pose.angular.z = - 90.0
            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.PTP,
                tool = CALI_TOOL,
                base = MY_BASE,
                pose = pose
            )
            res = self.call_hiwin(req)
            if res.arm_state == RobotCommand.Response.IDLE:
                key = input('Enter p to go photopose\nEnter n to go next clickpoint')
                if key == 'p':
                    nest_state = States.MOVE_TO_PHOTO_POSE
                elif key == 'n':
                    self.i += 1
                    if self.i >= self.pointnum:
                        self.get_logger().info('last clickpoint reached, moving to photo pose')
                        nest_state = States.MOVE_TO_PHOTO_POSE
                    else:
                        nest_state = States.MOVE_TO_CLICKPOINT
                else:
                    nest_state = None 
            else:
                nest_state = None

        elif state == States.MOVE_TO_PHOTO_POSE:
            self.click_point = [[],[]]
            self.intrin_clickpoint = [[],[]]
            self.pointnum = 0
            self.get_logger().info('Moving to photo pose')
            self.i = 0
            pose = Twist()
            [pose.linear.x, pose.linear.y, pose.linear.z] = fix_abs_cam[0:3]
            [pose.angular.x, pose.angular.y, pose.angular.z] = fix_abs_cam[3:6]
            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.PTP,
                tool = CAM_TOOL,
                pose = pose
            )
            res = self.call_hiwin(req)
            if res.arm_state == RobotCommand.Response.IDLE:
                key = input('Enter close to close robot\nEnter next to choose next sets of clickpoint')
                if key == 'close':
                    nest_state = States.FINISH
                elif key == 'next':
                    nest_state =States.OUTPUT_CLICKPOINT
                else:
                    nest_state = None 

        else:
            nest_state = None
            self.get_logger().info('input state not supported')

        return nest_state

    def _main_loop(self):
        state = States.INIT
        while state != States.FINISH:
            state = self.calibration_state(state)
            if state == None:
                break
        self.destroy_node()

    def _wait_for_future_done(self, future: Future, timeout=-1):
        time_start = time.time()
        while not future.done():
            time.sleep(0.01)
            if timeout > 0 and time.time() - time_start > timeout:
                self.get_logger().error('Wait for service timeout!')
                return False
        return True
    
    def generate_robot_request(
            self, 
            holding=True,
            cmd_mode=RobotCommand.Request.PTP,
            cmd_type=RobotCommand.Request.POSE_CMD,
            velocity=DEFAULT_VELOCITY,
            acceleration=DEFAULT_ACCELERATION,
            tool=9,
            base=MY_BASE,
            digital_output_pin=0,
            digital_output_cmd=RobotCommand.Request.DIGITAL_OFF,
            pose=Twist(),
            joints=[float('inf')]*6,
            circ_s=[],
            circ_end=[],
            jog_joint=6,
            jog_dir=0
            ):
        request = RobotCommand.Request()
        request.digital_output_pin = digital_output_pin
        request.digital_output_cmd = digital_output_cmd
        request.acceleration = acceleration
        request.jog_joint = jog_joint
        request.velocity = velocity
        request.tool = tool
        request.base = base
        request.cmd_mode = cmd_mode
        request.cmd_type = cmd_type
        request.circ_end = circ_end
        request.jog_dir = jog_dir
        request.holding = holding
        request.joints = joints
        request.circ_s = circ_s
        request.pose = pose
        return request

    def call_hiwin(self, req):
        while not self.hiwin_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('service not available, waiting again...')
        future = self.hiwin_client.call_async(req)
        if self._wait_for_future_done(future):
            res = future.result()
        else:
            res = None
        return res

    

    def start_main_loop_thread(self):
        self.main_loop_thread = Thread(target=self._main_loop)
        self.main_loop_thread.daemon = True
        self.main_loop_thread.start()

def main(args=None):
    rclpy.init(args=args)

    calibration = VefifyCalibration()
    calibration.start_main_loop_thread()

    rclpy.spin(calibration)
    rclpy.shutdown()

if __name__ == "__main__":
    main()