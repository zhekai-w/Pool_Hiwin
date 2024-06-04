import cv2
import ros2_yolov4.darknet as darknet

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from ros2_yolov4_interfaces.msg import Detection


ALL_cfg_path ="/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/cfg/yolov4-obj.cfg"      #'./cfg/yolov4-obj.cfg'
ALL_weights_path = '/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/cfg/weights/ALL/yolov4-obj_best.weights'
ALL_data_path = '/home/zack/work/ROS2_ws/src/py_pubsub/py_pubsub/cfg/hiwin_C_WDA_v4.data'

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
        left, top, right, bottom = bbox2points(bbox,W,H)
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
    return image

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.publisher = self.create_publisher(Detection, '/yolo_detection/detections', 10)

    def image_callback(self, msg):
        self.bridge = CvBridge()
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        detected_image, detections = image_detection(cv_image, ALL_network, ALL_class_names, ALL_class_colors,thresh=0.5)
        cv2.imshow('YOLOv4 Live Detection', detected_image)
        cv2.waitKey(1)
        

        self.publish_detections(detections)
        # print('Detection:', detections)

    def publish_detections(self, detections):
        for label, confidence, bbox in detections:
            msg = Detection()
            msg.label = label
            msg.confidence = confidence
            msg.bbox = bbox
            self.publisher.publish(msg)
            print('Msg:', msg.bbox)
            print("---------")

def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.')
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
