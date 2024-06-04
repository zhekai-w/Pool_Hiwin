import time
import rclpy
from enum import Enum
from threading import Thread
from rclpy.node import Node
from rclpy.task import Future
from geometry_msgs.msg import Twist

from hiwin_interfaces.srv import RobotCommand
from yolo_strategy_interfaces.srv import YoloStrategy

import matplotlib.pyplot as plt
import math
import py_pubsub.darknet as darknet
import pyrealsense2 as rs
import numpy as np
import cv2
import quaternion as qtn
import py_pubsub.transformations as transformations

MY_BASE = 12
CUE_TOOL = 12
# CAM_TOOL = 11
CALI_TOOL = 9

DEFAULT_VELOCITY = 10
DEFAULT_ACCELERATION = 10

# fix_abs_cam = [48.809, 270.743, 383.971, -179.499, -3.45, 89.95]
fix_abs_cam = [48.809, 310.996, 383.971, -180.0, -3.06, 90.0]
tool_to_cam = [-35.991, 122.302, -68.14]
CAM_TO_TABLE = 481

"""
神經網路檔案位置_檢測全部拼圖
"""
ALL_cfg_path ="/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/cfg/yolov4-obj.cfg"      #'./cfg/yolov4-obj.cfg'
ALL_weights_path = '/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/cfg/weights/ALL/yolov4-obj_best.weights'
ALL_data_path = '/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/cfg/hiwin_C_WDA_v4.data'


"""
載入神經網路
"""
ALL_network, ALL_class_names, ALL_class_colors = darknet.load_network(
        ALL_cfg_path,
        ALL_data_path,
        ALL_weights_path,
        batch_size=1
)

class States(Enum):
    INIT = 0
    FINISH = 1
    MOVE_TO_PHOTO_POSE = 2
    TAKE_PHOTO = 3
    YOLO_DETECT = 4
    SECOND_PHOTO = 5
    HITBALL_POSE = 6
    HITBALL = 7
    GET_BALL_POINT = 8
    CHECK_POSE = 9
    CLOSE_ROBOT = 10
    WAITING = 11
    SEC_IO =  12
    MOVE_TO_BALL_POSE = 13
    RECALCULATE = 14

"""
影像檢測
    輸入:(影像位置,神經網路,物件名稱集,信心值閥值(0.0~1.0))
    輸出:(檢測後影像,檢測結果)
    註記:
"""
def image_detection(image, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

"""
座標轉換
    輸入:(YOLO座標,原圖寬度,原圖高度)
    輸出:(框的左上座標,框的右下座標)
    註記:
"""
def bbox2points(bbox,W,H):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """ 
    width = darknet.network_width(ALL_network)      # YOLO壓縮圖片大小(寬)
    height = darknet.network_height(ALL_network)    # YOLO壓縮圖片大小(高)

    x, y, w, h = bbox                           # (座標中心x,座標中心y,寬度比值,高度比值)
    x = x*W/width
    y = y*H/height
    w = w*W/width
    h = h*H/height
    # 輸出框座標_YOLO格式
    # print("     (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(x, y, w, h))
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    
    return xmin, ymin, xmax, ymax

"""
原圖繪製檢測框線
    輸入:(檢測結果,原圖位置,框線顏色集)
    輸出:(影像結果)
    註記:
"""

def draw_boxes(detections, image, colors):
    ball_imformation = [[-999 for i in range(4)] for j in range(20)]
    i = 0

    H,W,_ = image.shape                      # 獲得原圖長寬

    # cv2.line(image,(640,0),(640,720),(0,0,255),5)

    for label, confidence, bbox in detections:
        xmin, ymin, xmax, ymax = bbox2points(bbox,W,H)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
        # 輸出框座標_加工格式座標(左上點座標,右上點座標)
        #print("\t{}\t: {:3.2f}%    (x1: {:4.0f}   y1: {:4.0f}   x2: {:4.0f}   y2: {:4.0f})".format(label, float(confidence), xmin, ymin, xmax, ymax))
        
        mx = float(xmax + xmin)/2
        my = float(ymax + ymin)/2

        # cv2.circle(image, (int(mx),int(my)), 33, (0,0,255), 3)
        if label == 'C':
            ball_imformation[i] = [0.0, float(confidence), mx, my]
        elif label == 'M':
            ball_imformation[i] = [1.0, float(confidence), mx, my]
        i+=1
        
    return image, ball_imformation

def detect_ALL(img,thresh=0.8):
    out,detections = image_detection(img,ALL_network, ALL_class_names, ALL_class_colors,thresh)
    out2, ball_imformation= draw_boxes(detections, img, ALL_class_colors)
    # cv2.imshow('out2', out2)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return out2, ball_imformation

def extract_YOLO_info(first_or_second):
    # Camera intrinsics and distortions coeff
    cameraMatrix = np.array([
                            [1363.8719422654856,  0, 938.9732418889218],
                            [ 0, 1362.7328132779226, 548.3344055737161],
                            [ 0,  0,  1]
                            ])
    distCoeffs = np.array([0.16291880696523953, -0.4619911499670495, 
                           0.00023885421077117, -0.0005976317960594, 
                           0.3376508830773949])
    confidence = []
    objectballx = []
    objectbally = []
    mid_error = []
    if first_or_second == 1:
        img = cv2.imread('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/verification_use.jpg')
        # img = cv2.imread('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/pics/PoolBall_2201.jpg')
        undistorted_image = cv2.undistort(img, cameraMatrix, distCoeffs, None)
        detected_img, ballinfo = detect_ALL(undistorted_image)
        cv2.imwrite('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/verification_detected.jpg', detected_img)
    elif first_or_second == 2:
        img = cv2.imread('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/verify_second_photo.jpg')
        # img = cv2.imread('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/pics/PoolBall_2201.jpg')
        undistorted_image = cv2.undistort(img, cameraMatrix, distCoeffs, None)
        detected_img, ballinfo = detect_ALL(undistorted_image)
        cv2.imwrite('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/second_photo_detected.jpg', detected_img)
    # check for non float value in ballinfo since flat_list does not accept data other than float
    cnt = 0
    for i in range(len(ballinfo)):
        if ballinfo[i][1] != -999:
            cnt += 1
        else:
            break
    #flatten list of 2darray to 1darray
    flat_list = []
    for i in range(0,cnt):
        flat_list.extend(ballinfo[i])

    # convert flat array to usable array 
    # in this case objectballx(y)[], with cuex(y) in objectballx(y)[-1] and confidence[]
    cueindex = None
    for i in range(0,len(flat_list),4):
        if flat_list[i] == 0:
            confidence.append(flat_list[i+1])
            objectballx.append(flat_list[i+2])
            objectbally.append(flat_list[i+3])
            #################3
        else:
            cueindex = i
    if cueindex != None:
        confidence.append(flat_list[cueindex+1])
        objectballx.append(flat_list[cueindex+2])
        objectbally.append(flat_list[cueindex+3])

    if first_or_second == 1:
        return confidence, objectballx, objectbally, detected_img   
        
    elif first_or_second == 2:
        for i in range(len(objectballx)):
            dev_x = objectballx[i] - 1920/2
            dev_y = objectbally[i] - 1080/2
            temp_error = math.sqrt((dev_x)**2+(dev_y)**2)
            mid_error.append(temp_error)
            min_error_index = mid_error.index(min(mid_error))
            mid_ball_x = objectballx[min_error_index]
            mid_ball_y = objectbally[min_error_index]
        return mid_ball_x, mid_ball_y, dev_x, dev_y

def take_pics(first_or_second):
    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    a = 0
    # Start streaming
    pipeline.start(config)  

    # Instructions for user
    print('Press m to take pictures\n')
    print('Press q to quit camera\n')
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        # if not color_frame:
        #     continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())  

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        key=cv2.waitKey(1)
        if key&0xFF==ord('m'):
            a=a+1
            print("picture taken")
            if first_or_second == 1:
                cv2.imwrite('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/verification_use.jpg',color_image)
            elif first_or_second == 2:
                cv2.imwrite('/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/testpics/verify_second_photo.jpg', color_image)
        if key&0xFF==ord('q'):
            cv2.destroyAllWindows()
            break
    # Stop streaming
    pipeline.stop() 

def pixel_mm_convert(cam_to_table_h, pixels):
    fov_x = 69/2
    fov_y = 42/2
    p_x = 1920/2
    p_y = 1080/2
   
    cam_to_table_x = cam_to_table_h/math.tan(math.radians(90-fov_x))
    cam_to_table_y = cam_to_table_h/math.tan(math.radians(90-fov_y))
    
    dev_x = cam_to_table_x/p_x*(pixels[0]-p_x)
    dev_y = cam_to_table_y/p_y*(pixels[1]-p_y)
    cam_to_ball_pose = [dev_x, dev_y, 1.0]
       
    return cam_to_ball_pose

def convert_arm_pose(ball_pose, arm_pose):
    base2tool = np.array(arm_pose)
    base2tool[:3] /= 1000

    # tool to camera quaternion 
    tool2cam_quaternion = [-0.0277325088427568, -0.026177641210702426, 0.7034336548574274, 0.7097370867214506]
    tool2cam_trans = [0.12230235996651653, -0.03599119674476696, 0.0681407254283748]

    tool2cam_rot = qtn.as_rotation_matrix(np.quaternion(tool2cam_quaternion[3],
                                                        tool2cam_quaternion[0], 
                                                        tool2cam_quaternion[1], 
                                                        tool2cam_quaternion[2]))
    tool2cam_trans = np.array([tool2cam_trans])
    
    # print("tool2cam_trans = {}".format(tool2cam_trans))
    
    tool2cam_mat = np.append(tool2cam_rot, tool2cam_trans.T, axis=1)
    tool2cam_mat = np.append(tool2cam_mat, np.array([[0., 0., 0., 1.]]), axis=0)
    
    quat = transformations.quaternion_from_euler(base2tool[3]*3.14/180,
                                                 base2tool[4]*3.14/180,
                                                 base2tool[5]*3.14/180,axes= "sxyz")
    
    base2tool_rot = qtn.as_rotation_matrix(np.quaternion(quat[3], 
                                                         quat[0], 
                                                         quat[1], 
                                                         quat[2]))
    
    base2tool_trans = np.array([base2tool[:3]])
    
    base2tool_mat = np.append(base2tool_rot, base2tool_trans.T, axis=1)
    base2tool_mat = np.append(base2tool_mat, np.array([[0., 0., 0., 1.]]), axis=0)
    
   
    cam2ball_trans = np.array([[ball_pose[0]/1000,
                                ball_pose[1]/1000,
                                ball_pose[2]]])
    unit_matrix = np.identity(3)
    cam2ball_mat = np.append(unit_matrix, cam2ball_trans.T, axis=1)
    cam2ball_mat = np.append(cam2ball_mat, np.array([[0., 0., 0., 1.]]), axis=0)

    base2cam_mat = np.matmul(base2tool_mat, tool2cam_mat)
    base2ball_mat = np.matmul(base2cam_mat, cam2ball_mat)

    ax, ay, az = transformations.euler_from_matrix(base2ball_mat)
    base2ball_translation = transformations.translation_from_matrix(base2ball_mat)*1000.0

    calibrated_ball_pose = [base2ball_translation[0], 
                            base2ball_translation[1], 
                            base2ball_translation[2],
                            ax*180/3.14, 
                            ay*180/3.14, 
                            az*180/3.14]

    return calibrated_ball_pose

class VerifyBallPose(Node):
    def __init__(self):
        super().__init__('verify_ball_pose')
        self.hiwin_client = self.create_client(RobotCommand, 'hiwinmodbus_service')
        self.fix_campoint = Twist()
        self.objectballx = []
        self.objectbally = []
        self.confidence = []
        self.ball_cnt = 0
        self.ball_pose = []
        self.second_photo = []
        self.fix_z = 80.0
        self.table_z = fix_abs_cam[2] + tool_to_cam[2] - CAM_TO_TABLE

    def verification_state(self, state: States) -> States:
        if state == States.INIT:
            self.get_logger().info('Moving to photo pose...')
            pose = Twist()
            [pose.linear.x, pose.linear.y, pose.linear.z] = fix_abs_cam[0:3]
            [pose.angular.x, pose.angular.y, pose.angular.z] = fix_abs_cam[3:6]
           
            req = self.generate_robot_request(
            cmd_mode=RobotCommand.Request.PTP,
            pose = pose
            )
            res = self.call_hiwin(req)
            if res.arm_state == RobotCommand.Response.IDLE:
                nest_state = States.TAKE_PHOTO
            # req = self.generate_robot_request(
            #     cmd_mode=RobotCommand.Request.CHECK_POSE
            #     )
            # res = self.call_hiwin(req)
            # self.fix_campoint = res.current_position
            # print(res.current_position)
            # nest_state = States.TAKE_PHOTO

        elif state == States.TAKE_PHOTO:
            self.get_logger().info('Taking photo for YOLO')
            take_pics(1)
            nest_state = States.YOLO_DETECT

        elif state == States.YOLO_DETECT:
            self.get_logger().info('Extracting YOLO info...')
            self.confidence, self.objectballx, self.objectbally, _ = extract_YOLO_info(1)
            print("Objectballx:", self.objectballx)
            print("Objectbally:", self.objectbally)
            nest_state = States.SECOND_PHOTO
        
        elif state == States.SECOND_PHOTO: 
            if self.ball_cnt < len(self.objectballx):
                self.get_logger().info('Total {} balls'.format(len(self.objectballx)))
                self.get_logger().info('Camera moving to index_{} ball'.format(self.ball_cnt))
                # 球相對於相機點位
                ball_pose_mm = pixel_mm_convert(CAM_TO_TABLE, [self.objectballx[self.ball_cnt], self.objectbally[self.ball_cnt]])
                # 球相對於手臂點位
                self.ball_pose = convert_arm_pose(ball_pose_mm, fix_abs_cam)
                pose = Twist()
                [pose.linear.x, pose.linear.y, pose.linear.z] = [self.ball_pose[0] - tool_to_cam[0],
                                                                 self.ball_pose[1] - tool_to_cam[1],
                                                                 self.fix_z]
                # change 
                [pose.angular.x, pose.angular.y, pose.angular.z] = fix_abs_cam[3:6]
                # [pose.angular.x, pose.angular.y] = self.ball_pose[3:5]
                # pose.angular.z = self.ball_pose[5]+90.

                input("Press Enter to continue...")
                req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.PTP,
                pose = pose,
                )
                res = self.call_hiwin(req)
                if res.arm_state == RobotCommand.Response.IDLE:
                    take_pics(2)
            
                time.sleep(1)
                mid_x, mid_y, dev_x, dev_y = extract_YOLO_info(2)
                ball_relative_cam = pixel_mm_convert(self.fix_z - abs(tool_to_cam[2]) + abs(self.table_z), [mid_x, mid_y])
                print("first mid:", [mid_x, mid_y])
                print("first dev:", [dev_x, dev_y])
                req = self.generate_robot_request(cmd_mode=RobotCommand.Request.CHECK_POSE)
                res = self.call_hiwin(req)
                self.second_photo = res.current_position
                [pose.linear.x, pose.linear.y, pose.linear.z] = [self.second_photo[0] + tool_to_cam[0] + ball_relative_cam[0],
                                                                 self.second_photo[1] + tool_to_cam[1] - ball_relative_cam[1],
                                                                 -75.0]
                [pose.angular.x, pose.angular.y, pose.angular.z] = [-180., 0., 90.]
                self.ball_cnt += 1
                input("Press Enter to continue...")
                self.get_logger().info('Tool moving to index_{} ball'.format(self.ball_cnt))
                req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.PTP,
                tool = CUE_TOOL,
                pose = pose
                )
                res = self.call_hiwin(req)
                if res.arm_state == RobotCommand.Response.IDLE and self.ball_cnt < len(self.objectballx):
                    # take_pics(2)
                    # mid_x, mid_y, dev_x, dev_y = extract_YOLO_info(2)
                    # print("second mid:", [mid_x, mid_y])
                    # print("second dev:", [dev_x, dev_y])

                    
                    nest_state = States.SECOND_PHOTO
                else:
                    input("Press Enter to move to photo pose...")
                    # take_pics(2)
                    # mid_x, mid_y, dev_x, dev_y = extract_YOLO_info(2)
                    # print("second mid:", [mid_x, mid_y])
                    # print("second dev:", [dev_x, dev_y])
                    nest_state = States.MOVE_TO_PHOTO_POSE
        
        elif state == States.MOVE_TO_PHOTO_POSE:
            self.get_logger().info('Moving to photo pose')
            pose = Twist()
            [pose.linear.x, pose.linear.y, pose.linear.z] = fix_abs_cam[0:3]
            [pose.angular.x, pose.angular.y, pose.angular.z] = fix_abs_cam[3:6]
            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.PTP,
                pose = pose
            )
            res = self.call_hiwin(req)
            if res.arm_state == RobotCommand.Response.IDLE:
                nest_state = States.CLOSE_ROBOT
            else:
                nest_state = None
        elif state == States.CLOSE_ROBOT:
            self.get_logger().info('CLOSE_ROBOT')
            req = self.generate_robot_request(cmd_mode=RobotCommand.Request.CLOSE)
            res = self.call_hiwin(req)
            nest_state = States.FINISH

        else:
            nest_state = None
            self.get_logger().error('Input state not supported!')

        return nest_state
    
    def _main_loop(self):
        state = States.INIT
        while state != States.FINISH:
            state = self.verification_state(state)
            if state == None:
                break
        self.destroy_node()

    def generate_robot_request(
            self, 
            holding=True,
            cmd_mode=RobotCommand.Request.PTP,
            cmd_type=RobotCommand.Request.POSE_CMD,
            velocity=DEFAULT_VELOCITY,
            acceleration=DEFAULT_ACCELERATION,
            tool=0,
            base=0,
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
    
    def _wait_for_future_done(self, future: Future, timeout=-1):
        time_start = time.time()
        while not future.done():
            time.sleep(0.01)
            if timeout > 0 and time.time() - time_start > timeout:
                self.get_logger().error('Wait for service timeout!')
                return False
        return True
    
    def start_main_loop_thread(self):
        self.main_loop_thread = Thread(target=self._main_loop)
        self.main_loop_thread.daemon = True
        self.main_loop_thread.start()
    
def main(args=None):
    rclpy.init(args=args)

    strategy = VerifyBallPose()
    strategy.start_main_loop_thread()
    rclpy.spin(strategy)
    rclpy.shutdown()  

if __name__=='__main__':
    main()
