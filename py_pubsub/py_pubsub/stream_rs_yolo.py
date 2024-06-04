import pyrealsense2 as rs
import numpy as np
import cv2
import darknet 
# import py_pubsub.darknet as darknet
import rclpy 
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
# from ros2_yolov4_interfaces.msg import Detection


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

def draw_boxes(detections, image):
    H,W,_ = image.shape                      
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox, W, H)
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
    return image

def detect_ALL(img,thresh=0.8):
    out,detections = image_detection(img,ALL_network, ALL_class_names, ALL_class_colors,thresh)
    out2= draw_boxes(detections, img)

    # cv2.imshow('out2', out2)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return out2, detections

class DetectionPublisher(Node):
    def __init__(self):
        super().__init__('detection_publisher')
        self.publisher = self.create_publisher(Detection, '/yolo_detection/detections', 10)
        self.timer = self.create_timer(0.033, self.publish_detection) # 60 fps

        # Create a context object. This object owns the handles to all connected realsense devices
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        # Start streaming
        pipeline.start(config)
    
    def publish_detection(self):
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        # if not color_frame:
        #     continue

        # Convert frames to numpy arrays
        color_img = np.asanyarray(color_frame.get_data())
        detected_img, ball_info = detect_ALL(color_img)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('YOLOv4 Live Detection', detected_img)

        for label, confidence, bbox in ball_info:
            msg = Detection()
            msg.label = label
            msg.confidence = confidence
            msg.bbox = bbox
            self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DetectionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.')
    node.cap.release()
    cv2.destroyAllWindows()
    pipeline.stop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__": 
    # main()

    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)

    while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            # if not color_frame:
            #     continue

            # Convert frames to numpy arrays
            color_img = np.asanyarray(color_frame.get_data())
            detected_img, ball_info = detect_ALL(color_img)

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('YOLOv4 Live Detection', detected_img)
            key=cv2.waitKey(1)
            if key&0xFF==ord('q'):
                cv2.destroyAllWindows()
                break

    # Stop streaming
    pipeline.stop()

    
